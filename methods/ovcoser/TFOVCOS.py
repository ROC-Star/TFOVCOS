import os
import cv2
from typing import Tuple
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
import numpy as np
import contextlib, io
import warnings
from PIL import Image

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from utils.LLMTools import get_text_from_img_llava
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path
from transformers.utils import logging as hf_logging 
warnings.filterwarnings("ignore")

from methods.base.base_ops import PixelNormalizer, resize_to, rescale_2x
from methods.ovcoser.VLM_ViT2 import ViTCLIP
from methods.ovcoser.loss import seg_loss, l1_ssim_loss, edge_dice_loss
from segment_anything_hq import sam_model_registry, SamPredictor,build_sam_vit_h
from utils.visual_tools.tools import plot_image_with_bboxes_and_points


class TFOVCoser(nn.Module):
    def __init__(self, iterations=2, template_set="camoprompts"):
        super().__init__()
        assert iterations in [1, 2, 3]
        self.iterations = iterations
        logger.info(f"Current Iteration: {iterations}")

        self.clip = ViTCLIP(template_set=template_set)
        self.normalizer = PixelNormalizer(mean=self.clip.mean, std=self.clip.std)
        self.test_class_embs = None
        self.train_class_embs = None

        self.use_box_prompt = True  # Set to True if you want to use box prompt
        self.box_expand_ratio = 0.1  # Ratio to expand the bounding box, adjust as needed



        self.sam = build_sam_vit_h(checkpoint='./sam_hq_vit_h.pth')
        self.frozen_sam()
        self.sam_predictor = SamPredictor(self.sam)

        

        #MLLM config 
        self.mllm_config = {
            'llm_model_name':'LLaVA1.5',
            'prompt_q':'3attriTheBgSynCamo',
            'use_gene_prompt':False,
            'clip_use_bg_text':True,
        }
        self.llava_config= {
            "w_caption": True ,
            "model_base": None,
            "num_chunks": 1,
            "chunk_idx": 0,
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
        }
           

    def frozen_sam(self):
        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam.to(self.clip.device)



    def get_visual_feats(self, image):
        image = self.normalizer(image)
        final_visual, cls_token = self.clip.get_visual_feats(image)
        return final_visual, cls_token


    def map_classifier(self, cls_token, normed_class_embs):
        normed_image_embs = F.normalize(cls_token, dim=-1)
        class_logits = normed_image_embs @ normed_class_embs.T  
        class_logits = self.clip.clip_model.logit_scale.exp() * class_logits   
        return class_logits

    
    

    def similarity_map_to_points(self, sm, image_shape, t=0.8, down_sample=1, is_fg=True, max_pos=40, max_neg=40, use_quantile=True, fg_q=0.2, bg_q=0.97):

        side = int(sm.shape[0] ** 0.5)
        sm = sm.reshape(1, 1, side, side)
        # down sample to smooth results
        down_side = int(side // down_sample)
        sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
        if sm.dim() == 1:
            side = int(sm.shape[0] ** 0.5)
            assert side * side == sm.shape[0], "相似度长度不是完全平方，无法重排为方格。"
            sm = sm.view(1, 1, side, side)
        elif sm.dim() == 2:
            side_h, side_w = sm.shape
            assert side_h == side_w, "当前函数假设 h==w，可在此处扩展为矩形支持。"
            sm = sm.view(1, 1, side_h, side_w)
            side = side_h
        else:
            raise ValueError("sm 期望为展平或 2D 方阵。")

        if down_sample and down_sample > 1:
            new_side = max(4, int(side / down_sample))
            sm = F.interpolate(sm, (new_side, new_side), mode='bilinear', align_corners=False)
        heat = sm[0, 0]
        h, w = heat.shape

        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

        if use_quantile:
            hi = torch.quantile(heat, torch.tensor(fg_q, device=heat.device))
            lo = torch.quantile(heat, torch.tensor(bg_q, device=heat.device))
        else:
            hi = t
            lo = 1 - t

        if is_fg:
            fg_mask = heat >= hi
            bg_mask = heat <= lo
        else:
            fg_mask = heat <= lo
            bg_mask = heat >= hi

        relax_steps = [0.90, 0.80, 0.70]
        step_idx = 0
        while fg_mask.sum() < 1 and step_idx < len(relax_steps):
            q = relax_steps[step_idx]
            hi = torch.quantile(heat, torch.tensor(q, device=heat.device)) if use_quantile else hi * 0.9
            if is_fg:
                fg_mask = heat >= hi
            else:
                fg_mask = heat <= hi
            step_idx += 1
        if fg_mask.sum() < 1:
            max_idx = torch.argmax(heat)
            fg_mask = torch.zeros_like(heat, dtype=torch.bool)
            fg_mask.view(-1)[max_idx] = True

        min_fg_points = 20        
        target_fg_points = 40     



        if fg_mask.sum() < min_fg_points:
            flat_heat = heat.view(-1)
            k = min(target_fg_points, flat_heat.numel(), max_pos) 
            topk_idx = torch.topk(flat_heat, k=k, largest=is_fg).indices
            new_mask = torch.zeros_like(flat_heat, dtype=torch.bool)
            new_mask[topk_idx] = True
            fg_mask = new_mask.view_as(heat)

        
        if bg_mask.sum() < 1:
            
            min_idx = torch.argmin(heat)
            bg_mask = torch.zeros_like(heat, dtype=torch.bool)
            bg_mask.view(-1)[min_idx] = True

       
        fg_idx = torch.nonzero(fg_mask, as_tuple=False)
        bg_idx = torch.nonzero(bg_mask, as_tuple=False)

        if fg_idx.shape[0] > max_pos:
            fg_scores = heat[fg_mask]
            topk_vals, topk_inds = torch.topk(fg_scores, max_pos)
            new_fg_mask = torch.zeros_like(heat, dtype=torch.bool)
            flat_fg = torch.nonzero(fg_mask.view(-1), as_tuple=False).view(-1)
            select_flat = flat_fg[topk_inds]
            new_fg_mask.view(-1)[select_flat] = True
            fg_mask = new_fg_mask
            fg_idx = torch.nonzero(fg_mask, as_tuple=False)
        if bg_idx.shape[0] > max_neg:
            perm = torch.randperm(bg_idx.shape[0], device=bg_idx.device)[:max_neg]
            chosen = bg_idx[perm]
            new_bg_mask = torch.zeros_like(heat, dtype=torch.bool)
            new_bg_mask[chosen[:, 0], chosen[:, 1]] = True
            bg_mask = new_bg_mask
            bg_idx = chosen

        H, W = image_shape
        scale_y = H / h
        scale_x = W / w

        def to_img_coords(idx_tensor: torch.Tensor):
            if idx_tensor.numel() == 0:
                return np.zeros((0, 2), dtype=np.int32)
            ys = (idx_tensor[:, 0].float() + 0.5) * scale_y
            xs = (idx_tensor[:, 1].float() + 0.5) * scale_x
            pts = torch.stack([xs, ys], dim=1)
            pts = pts.round().clamp(min=0)
            pts[:, 0] = pts[:, 0].clamp(max=W - 1)
            pts[:, 1] = pts[:, 1].clamp(max=H - 1)
            return pts.int().cpu().numpy()

        fg_points = to_img_coords(fg_idx)
        bg_points = to_img_coords(bg_idx)

        labels_fg = np.ones(len(fg_points), dtype='uint8')
        labels_bg = np.zeros(len(bg_points), dtype='uint8')

        return fg_points.tolist(), labels_fg, bg_points.tolist(), labels_bg
    
    def get_mask(self, sim_map_fg, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                assert image.size(0) == 1, f"期望单张图像 B=1, 得到 B={image.size(0)}"
                image = image[0]
           
            if image.dim() == 3 and image.size(0) in (1, 3):
                if image.size(0) == 1:
                    image = image.repeat(3, 1, 1)
                image_np = image.detach().cpu().permute(1, 2, 0).numpy()
            else:
                raise ValueError("图像张量维度不符合 CHW 格式。")
        else:
            image_np = image  
        if image_np.dtype != np.uint8:
            img_min, img_max = float(image_np.min()), float(image_np.max())
            if 0.0 <= img_min and img_max <= 1.0:
                image_np = (image_np * 255.0).round().astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        H, W = image_np.shape[:2]
        self.sam_predictor.set_image(image_np)

        if sim_map_fg.dim() == 2:
            L, C = sim_map_fg.shape
        else:
            raise ValueError("sim_map_fg 应为 [L, C] 形状。")

        mask_list = []
        logit_list = []
        side = int(L ** 0.5)
        assert side * side == L, "L 不是平方数，无法重排到网格用于可视化。"


        
        for cls_idx in range(C):
            sim_vec = sim_map_fg[:, cls_idx]

            fg_pts, fg_labels, bg_pts, bg_labels = self.similarity_map_to_points(
                sim_vec, image_shape=(H, W), use_quantile=True,
                fg_q=0.90, bg_q=0.10, max_pos=7, max_neg=4
            )
            points = fg_pts + bg_pts
            labels = fg_labels.tolist() + bg_labels.tolist()
            if len(points) == 0: 
                points = [[W // 2, H // 2]]
                labels = [1]
            # points_np = np.array(points, dtype=np.int32)  # sam1
            # labels_np = np.array(labels, dtype=np.int32)
            points_np = np.array(points, dtype=np.float32) # sam2
            labels_np = np.array(labels, dtype=np.float32)
            box_input = None
            if self.use_box_prompt and len(fg_pts) > 0:
                fg_arr = np.array(fg_pts, dtype=np.int32)
                xs, ys = fg_arr[:, 0], fg_arr[:, 1]
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
               
                if x1 == x0:
                    x1 = min(W - 1, x1 + 1)
                    x0 = max(0, x0 - 1)
                if y1 == y0:
                    y1 = min(H - 1, y1 + 1)
                    y0 = max(0, y0 - 1)
                bw = x1 - x0 + 1
                bh = y1 - y0 + 1
                expand = int(max(bw, bh) * self.box_expand_ratio)
               
                bw = x1 - x0 + 1
                bh = y1 - y0 + 1
                
                local_expand_ratio = min(self.box_expand_ratio, 0.05)
                expand = int(max(bw, bh) * local_expand_ratio)

                if expand > 0:
                    x0 = max(0, x0 - expand)
                    y0 = max(0, y0 - expand)
                    x1 = min(W - 1, x1 + expand)
                    y1 = min(H - 1, y1 + expand)
                box_input = np.array([x0, y0, x1, y1], dtype=np.int32)

            side = int(sim_vec.shape[0] ** 0.5)
            sm_norm_fg_n = sim_vec.reshape(1, 1, side, side)
           
            map_fg = F.interpolate(sm_norm_fg_n, (H, W), mode='bilinear', align_corners=False).squeeze()


          
            mask_logit_origin, scores, _ = self.sam_predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                box=box_input,
                multimask_output=True,
                hq_token_only = True,  # sam2.1
                return_logits=True
            )
            best_id = int(np.argmax(scores))
            best_logit = mask_logit_origin[best_id]
            best_mask = best_logit > self.sam_predictor.model.mask_threshold # hqsam1
            # best_mask = best_logit > self.sam_predictor.mask_threshold # hqsam2.1

            mask_list.append(best_mask)
            logit_list.append(best_logit)

        return mask_list, logit_list

    def fuse_mask(self,mask_logit_origin_l, sam_thr, fuse='avg'):

        num_mask = len(mask_logit_origin_l)
        if fuse == 'avg':
            mask_logit_origin = sum(mask_logit_origin_l) / num_mask  #
            mask_logit = torch.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
            mask = mask_logit_origin > sam_thr
        

        mask = mask.astype('uint8')
        mask_logit *= 255
        mask_logit = mask_logit.astype('uint8')

        return mask, mask_logit
    def hierarchical_clustering(self, text_embeddings):
        h_threshold = 0
        cluster_dist_percentile = 35
        with torch.no_grad():
            X = F.normalize(text_embeddings, dim=-1).detach().cpu().numpy()
        D = cosine_distances(X)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method='average')  
        if h_threshold and h_threshold > 0:
            thr = h_threshold
        else:
            non_diag = D[~np.eye(D.shape[0], dtype=bool)]
            thr = float(np.percentile(non_diag, cluster_dist_percentile))
        clusters = fcluster(Z, t=thr, criterion='distance') - 1
        return clusters
    
    def _dynamic_patch_k_and_probs(self, sims_in_cluster: torch.Tensor) -> Tuple[int, torch.Tensor]:
        sim_temperature = 0.07
        temp = sim_temperature
        topk_min = 2
        topk_max = 12
        probs = torch.softmax(sims_in_cluster / temp, dim=0)
        ent = -(probs * (probs.clamp_min(1e-8)).log()).sum()
        max_ent = torch.log(torch.tensor(probs.numel(), dtype=probs.dtype, device=probs.device)).clamp_min(1e-8)
        ent_norm = (ent / max_ent).clamp(0, 1)
        k = int(round(topk_min + (topk_max - topk_min) * float(ent_norm)))
        k = max(topk_min, min(k, topk_max))
        return k, probs
    

    def _get_cluster_indices(self, cluster_id: int,class_embeddings) -> np.ndarray:
        cluster_info = self.hierarchical_clustering(class_embeddings)
        return np.where(cluster_info == cluster_id)[0]

    def _get_most_similar_class(self, similarities: np.ndarray, indices: np.ndarray) -> int:

        similarities_in_cluster = similarities[indices]
        idx_max_similarity = np.argmax(similarities_in_cluster)
        return int(indices[idx_max_similarity])

    def _compute_topk_similarities(
        self, 
        image_features: torch.Tensor, 
        text_embedding: torch.Tensor, 
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        similarities = torch.matmul(image_features, text_embedding.t()).squeeze(1)
        k = min(k, image_features.size(0))
        return torch.topk(similarities, k)

    def _compute_adjusted_embedding(
        self, 
        text_embedding: torch.Tensor, 
        mean_vector: torch.Tensor, 
        alpha: float
    ) -> torch.Tensor:

        adjusted = (1 - alpha) * text_embedding + alpha * mean_vector
        return F.normalize(adjusted, dim=0)
    
    def hierarchical_prompt1(
        self, 
        image_features: torch.Tensor, 
        image_class_similarities: torch.Tensor,
        class_embeddings
    ) -> torch.Tensor:

        image_class_sims = image_class_similarities.squeeze(0)
        image_class_similarities_np = image_class_sims.detach().cpu().numpy()
        image_features = image_features.squeeze(0).to(class_embeddings.device)
        text_features = class_embeddings.clone().detach().to(class_embeddings.device)
        adjusted_text_features = text_features.clone()
        cluster_info = self.hierarchical_clustering(class_embeddings)
        mean_vector_weight = 0.0
        neg_vector_weight = 0.0

        image_features_norm = F.normalize(image_features, dim=1)

        num_clusters = int(len(set(cluster_info)))
        for cluster_id in range(num_clusters):

            indices = self._get_cluster_indices(cluster_id, class_embeddings)
            if indices.size == 0:
                continue

           
            sims_cluster = image_class_sims[indices]

           
            k_dyn, probs_cluster = self._dynamic_patch_k_and_probs(sims_cluster)

           
            class_i = self._get_most_similar_class(image_class_similarities_np, indices)
            text_embedding_i = F.normalize(text_features[class_i].unsqueeze(0), dim=1)

           
            pos_sims = torch.matmul(image_features_norm, text_embedding_i.t()).squeeze(1)

           
            k_eff = min(k_dyn, image_features_norm.size(0))
            _, pos_idx = torch.topk(pos_sims, k_eff, largest=True)
            pos_mean = F.normalize(image_features[pos_idx].mean(dim=0), dim=0)

           
            neg_k = max(1, min(k_eff, image_features_norm.size(0)))
            _, neg_idx = torch.topk(pos_sims, neg_k, largest=False)
            neg_mean = F.normalize(image_features[neg_idx].mean(dim=0), dim=0)

            
            probs_cluster = probs_cluster.detach()
            for local_j, cls_j in enumerate(indices):
                cls_j = int(cls_j)
                p = float(probs_cluster[local_j])
                alpha_j = mean_vector_weight * p
                beta_j = neg_vector_weight * p

                t_j = F.normalize(text_features[cls_j], dim=0)
                adjusted = (1.0 - alpha_j) * t_j + alpha_j * pos_mean - beta_j * neg_mean
                adjusted_text_features[cls_j] = F.normalize(adjusted, dim=0)

        return adjusted_text_features

    
    def body(self, image, normed_class_embs, fg_text_embs=None):
        B,C,H,W = image.shape
        final_visual, cls_token = self.get_visual_feats(image)
        sim_map = final_visual @ fg_text_embs.T  # [B,576,61]
        # sim_map = final_visual @ normed_class_embs.T * self.clip.clip_model.logit_scale.exp() + self.clip.clip_model.logit_bias 
        sim_map = 1 - sim_map 
        sim_map = sim_map[0,:,:] # [576,61]
       
        mask_l, mask_logit_ori = self.get_mask(sim_map,image)
        vis_mask_acc, vis_mask_logit_acc = self.fuse_mask(mask_logit_ori, self.sam_predictor.model.mask_threshold) #hqsam1
       

       
        mask_delta_l = [np.sum((mask_i - vis_mask_acc) ** 2) for mask_i in mask_l]  # distance of each mask to fused one
        idxMaskSim = np.argmin(mask_delta_l)
        vis_tensor_np = mask_l[idxMaskSim]  # numpy array (H,W) bool/uint8
        vis_tensor = torch.from_numpy(vis_tensor_np).unsqueeze(0).to(image.device)  # [1,1,H,W]
        logits = {}


        
       
        normed_image_embs = F.normalize(cls_token, dim=-1)
        image_class = normed_image_embs @ normed_class_embs.T 

        adjusted_class_embs = self.hierarchical_prompt1(
            image_features=final_visual,
            image_class_similarities=image_class,
            class_embeddings=normed_class_embs
        )
        cls_logits = self.map_classifier(cls_token, adjusted_class_embs)
        logits["seg"] = vis_tensor
        logits["cls"] = cls_logits

        return logits
    def get_text_from_img(self, pil_img, prompt_q, llm_dict, use_gene_prompt, get_bg_text,
                        reset_prompt_qkeys=False, new_prompt_qkeys_l=None,
                        bg_cat_list=[],
                        post_process_per_cat_fg=False,class_name=None,image_tensor=None,device=None):
        model = llm_dict['model']
        vis_processors = llm_dict['vis_processors']
        tokenizer = llm_dict['tokenizer']
        conv_mode = llm_dict['conv_mode']
        temperature = llm_dict['temperature']
        w_caption = llm_dict['w_caption']
        return get_text_from_img_llava(pil_img, prompt_q,
                    model, vis_processors, tokenizer,
                    get_bg_text=get_bg_text,
                    conv_mode=conv_mode,
                    temperature=temperature,
                    w_caption=w_caption,
                    target_class_name=class_name,
                    image_tensor=image_tensor,
                    device=device
                    )

    # def train_forward(self, data, gt_classes, class_names: list, **kwargs):
    #     image = data["image"]

        
    #     if self.train_class_embs is None:
    #         self.train_class_embs = self.clip.get_text_embs_by_template(class_names)
    #     normed_class_embs = self.train_class_embs  # Nc,D

    #     logits_dict = self.body(image, normed_class_embs)

    #     losses = []
    #     loss_str = []

    #     mask = data["mask"]
    #     depth = data["depth"]
    #     with torch.no_grad():
    #         edge_ks = 5
    #         eroded_mask = -F.max_pool2d(-mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
    #         dilated_mask = F.max_pool2d(mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
    #         edge = dilated_mask - eroded_mask
    #         edge = edge.gt(0).float()
    #     logits_dict["edge"] = edge

    #     for i in range(1, self.iterations + 1):
    #         seg_logits = logits_dict[f"seg{i}"]
    #         segl = seg_loss(logits=seg_logits, mask=mask)
    #         losses.append(segl)
    #         loss_str.append(f"segl: {segl.item():.5f}")

    #         d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-3"], mask=depth)
    #         losses.append(d_l1ss)
    #         loss_str.append(f"d_l1ss3: {d_l1ss.item():.5f}")
    #         d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-2"], mask=depth)
    #         losses.append(d_l1ss)
    #         loss_str.append(f"d_l1ss2: {d_l1ss.item():.5f}")
    #         d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-1"], mask=depth)
    #         losses.append(d_l1ss)
    #         loss_str.append(f"d_l1ss1: {d_l1ss.item():.5f}")

    #         e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-3"], edge=edge)
    #         losses.append(e_dice)
    #         loss_str.append(f"e_dice3: {e_dice.item():.5f}")
    #         e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-2"], edge=edge)
    #         losses.append(e_dice)
    #         loss_str.append(f"e_dice2: {e_dice.item():.5f}")
    #         e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-1"], edge=edge)
    #         losses.append(e_dice)
    #         loss_str.append(f"e_dice1: {e_dice.item():.5f}")

    #     return dict(
    #         vis={k: v if k == "edge" else v.sigmoid() for k, v in logits_dict.items() if not k.startswith("cls")},
    #         loss=sum(losses),
    #         loss_str=" ".join(loss_str),
    #     )  

    def load_llava(self,device=None):
        
        disable_torch_init()
        
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        hf_logging.set_verbosity_error()
        model_path = 'liuhaotian/llava-v1.5-13b'
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = self.llava_config['model_base']

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, model_base, model_name, device_map=device
            )

        model.eval()
        model.to(device)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        llm_dict = {
            'model': model,
            'vis_processors': image_processor,
            'tokenizer': tokenizer,
            'conv_mode': conv_mode,
            'temperature': self.llava_config["temperature"],
            'w_caption': self.llava_config["w_caption"],
        }
        return llm_dict
    
    def _tensor_to_pil(self, img: torch.Tensor) -> Image.Image:
        if img.dim() == 4:
            img = img[0]
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
               arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def test_forward(self, data, gt_classes, class_names: list, **kwargs):

        image = data["image"]
        device = image.device 
        gt_classes = gt_classes[0]
        image_raw = self._tensor_to_pil(image)
        llm_dict = self.load_llava(device=device)
        textfg_phrase_list, textbg_phrase_list,textfg_word_list, textbg_word_list,bbox_ori_list = self.get_text_from_img(image_raw, self.mllm_config['prompt_q'], llm_dict, self.mllm_config['use_gene_prompt'], self.mllm_config['clip_use_bg_text'],class_name=gt_classes,image_tensor=image,device=device)

      
        fg_text_embs = self.clip.get_text_embs_by_template1(textfg_phrase_list) #[3,768]


        if self.test_class_embs is None:
            self.test_class_embs = self.clip.get_text_embs_by_template(class_names)
        normed_class_embs = self.test_class_embs

        logits_dict = self.body(image,normed_class_embs,fg_text_embs)
       

        seg_logits = logits_dict["seg"]
        cls_logits = logits_dict[f"cls"]

        cls_id_per_image = torch.argmax(cls_logits, dim=-1)
        pred_classes = [class_names[i] for i in cls_id_per_image]
        return dict(prob=seg_logits.sigmoid(), classes=pred_classes)

    def forward(self, *arg, **kwargs):
        if self.training:
            return self.train_forward(*arg, **kwargs)
        else:
            return self.test_forward(*arg, **kwargs)

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("clip."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            else:
                param.requires_grad = True
                param_groups["retrained"].append(param)
        logger.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
