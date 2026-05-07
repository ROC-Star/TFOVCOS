import numpy as np
import open_clip
import torch
from loguru import logger
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch import nn
from torch.nn import functional as F
from methods.ovcoser.prompts import get_prompt_template_by_name


class ViTCLIP(nn.Module):
    def __init__(self, model_name='ViT-L-14', template_set="camoprompts",):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name="ViT-L-14-quickgelu",pretrained="dfn2b"
            )
        self.mean = OPENAI_DATASET_MEAN
        self.std = OPENAI_DATASET_STD
        self.text_tokenizer = open_clip.get_tokenizer('ViT-L-14-quickgelu')

        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.frozen_vlm()
        # self.patch_size = int(model_name.split("/")[-1])
        self.patch_size = int(model_name.split("-")[-1])
        self.img_h, self.img_w = None, None
        self.visual_dim = self.clip_model.visual.transformer.width
        self.out_indices = [5, 11, 17, 23]  # for ViT-L/14

        self.template_set = get_prompt_template_by_name(template_set)
        logger.info(
            f"Create the CLIP ({model_name + '-' + 'datacomp_xl_s13B_b90K'}) with template_set {self.template_set}")

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device
    def frozen_vfm(self):
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(self.device)

    def frozen_vlm(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval().to(self.device)

    @torch.no_grad()
    def get_text_embs_by_template1(self, text_list):
        """
        return: NumberofClasses,D
        """
        self.eval()
        self.clip_model.transformer.batch_first = False
        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer(text).to(self.device)
            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768
            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        self.eval()
        self.clip_model.transformer.batch_first = False
        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(self.device)
            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768
            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768


    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    @staticmethod
    def upsample_pos_emb(emb, new_size):
        first, emb = emb[:1, :], emb[1:, :]
        n, d = emb.size(0), emb.size(1)
        size = int(np.sqrt(n))
        emb = emb.permute(1, 0).view(1, d, size, size)
        emb = F.interpolate(emb, size=new_size, mode='bilinear')
        emb = emb.view(d, -1).contiguous().permute(1, 0)
        emb = torch.cat([first, emb], 0)
        return emb.half()


    def get_visual_feats(self, x):
        self.img_h, self.img_w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.clip_model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self._expand_token(self.clip_model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        x = x + self.upsample_pos_emb(self.clip_model.visual.positional_embedding, (self.img_h, self.img_w))

        x = self.clip_model.visual.patch_dropout(x)
        x = self.clip_model.visual.ln_pre(x)

        latent_visual = self.clip_model.visual.transformer(x)
        
        cls_token =  latent_visual[:, 0, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        cls_token = self.clip_model.visual.ln_post(cls_token)  # [b,1024]
        cls_token = cls_token @ self.clip_model.visual.proj


        final_visual = self.clip_model.visual.ln_post(latent_visual[:, 1:, :])  # [b,576,1024]
        
        final_visual = final_visual @ self.clip_model.visual.proj
        final_visual = final_visual / final_visual.norm(dim=-1, keepdim=True)
        return final_visual, cls_token

if __name__ =='__main__':
    clip = ViTCLIP()

    image = torch.randn([1,3,336,336])
    visual,cls = clip.get_visual_feats(image)
    print(visual.shape,cls.shape)
    fg = [
      "The moth in the image is camouflaged, blending seamlessly with the bark of a tree",
      "Its texture closely resembles the rough, weathered surface of the tree, with patches of green and brown that mimic the natural patterns of the bark",
      "The moths distinctive features are its ability to mimic the trees appearance, making it difficult to distinguish from the background"
    ]
    text_embs = clip.get_text_embs_by_template1(fg)
    print(text_embs.shape)
    # visual = torch.randn([1, 576, 768])
    # text = torch.randn([61, 768])
    # simi = visual @ text.T
    # print(simi.shape)
    # logits = simi.permute(0, 2, 1).reshape(-1, simi.shape[-1], 24, 24)
    # print(logits.shape)



