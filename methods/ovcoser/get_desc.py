import json
from typing import Any, Dict, List
import warnings
from PIL import Image
from utils.LLMTools import get_text_from_img_llava
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path
import os
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import contextlib
import io


class GetDesc:
    def __init__(self):
        disable_torch_init()
        # MLLM config 
        self.mllm_config = {
            'llm_model_name': 'LLaVA1.5',
            'prompt_q': '3attriTheBgSynCamo',
            'use_gene_prompt': False,
            'clip_use_bg_text': True,
        }
        self.llava_config = {
            "w_caption": True,
            "model_base": None,
            "num_chunks": 1,
            "chunk_idx": 0,
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
        }
        self.llm_dict = None  # 缓存加载的模型

    def load_llava(self, device=None):
        """加载LLaVA模型，如果已经加载则直接返回"""
        if self.llm_dict is not None:
            return self.llm_dict
            
        disable_torch_init()
        # 静默 transformers / hub 的日志与进度条
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        model_path = 'liuhaotian/llava-v1.5-13b'
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = self.llava_config['model_base']

        # 用 stdout/stderr 重定向屏蔽加载信息
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

        self.llm_dict = {
            'model': model,
            'vis_processors': image_processor,
            'tokenizer': tokenizer,
            'conv_mode': conv_mode,
            'temperature': self.llava_config["temperature"],
            'w_caption': self.llava_config["w_caption"],
        }
        return self.llm_dict

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        _, ext = os.path.splitext(filename.lower())
        return ext in exts

    @staticmethod
    def extract_class_from_filename(filename: str) -> str:
        """
        从文件名中提取类别名
        """
        name_no_ext = os.path.splitext(os.path.basename(filename))[0]
        for sep in ['_', '-', ' ']:
            if sep in name_no_ext:
                return name_no_ext.split(sep)[0]
        return name_no_ext

    def process_directory(self, img_dir: str, output_json: str, device: str = None) -> Dict[str, Any]:
        """
        批量处理目录下的图片：从文件名提取 class，调用 LLaVA 获取描述，并将结果写入 JSON
        """
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"图片目录不存在: {img_dir}")

        # 只加载一次模型
        llm_dict = self.load_llava(device=device)

        results: Dict[str, Any] = {}
        files = sorted([f for f in os.listdir(img_dir) if self._is_image_file(f)])
        
        if not files:
            warnings.warn(f"目录中未找到图片文件: {img_dir}")
            return results

        print(f"开始处理 {len(files)} 张图片...")
        
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(img_dir, fname)
            try:
                class_name = self.extract_class_from_filename(fname)
                print(f"处理进度: {i}/{len(files)} - 图片: {fname}, 类别: {class_name}")
                
                pil_img = Image.open(fpath).convert('RGB')
                
                # 调用get_desc方法获取描述
                textfg_phrase_list, textbg_phrase_list, textfg_word_list, textbg_word_list, bbox_ori_list = self.get_desc(
                    image_raw=pil_img,
                    gt_classes=class_name,
                    device=device,
                    llm_dict=llm_dict  # 传入已加载的模型
                )

                results[fname] = {
                    'class': class_name,
                    'textfg_phrase_list': textfg_phrase_list,
                    'textbg_phrase_list': textbg_phrase_list,
                    'textfg_word_list': textfg_word_list,
                    'textbg_word_list': textbg_word_list,
                    'bbox_ori_list': bbox_ori_list,
                }

            except Exception as e:
                warnings.warn(f"处理图片失败: {fname}，错误: {e}")
                # 记录失败的信息
                results[fname] = {
                    'class': class_name if 'class_name' in locals() else 'unknown',
                    'error': str(e),
                    'textfg_phrase_list': [],
                    'textbg_phrase_list': [],
                    'textfg_word_list': [],
                    'textbg_word_list': [],
                    'bbox_ori_list': [],
                }
                continue

        # 保证输出目录存在
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"处理完成！结果已保存到: {output_json}")
        return results

    def get_text_from_img(self, pil_img, prompt_q, llm_dict, use_gene_prompt, get_bg_text,
                         reset_prompt_qkeys=False, new_prompt_qkeys_l=None,
                         bg_cat_list=[], post_process_per_cat_fg=False, 
                         class_name=None, image_tensor=None, device=None):
        """调用LLaVA获取图片描述"""
        model = llm_dict['model']
        vis_processors = llm_dict['vis_processors']
        tokenizer = llm_dict['tokenizer']
        conv_mode = llm_dict['conv_mode']
        temperature = llm_dict['temperature']
        w_caption = llm_dict['w_caption']

        return get_text_from_img_llava(
            pil_img, prompt_q,
            model, vis_processors, tokenizer,
            get_bg_text=get_bg_text,
            conv_mode=conv_mode,
            temperature=temperature,
            w_caption=w_caption,
            target_class_name=class_name,
            image_tensor=image_tensor,
            device=device
        )

    def get_desc(self, image_raw, gt_classes=None, device=None, llm_dict=None):
        """获取单张图片的描述"""
        # 如果没有传入模型，则加载
        if llm_dict is None:
            llm_dict = self.load_llava(device=device)

        textfg_phrase_list, textbg_phrase_list, textfg_word_list, textbg_word_list, bbox_ori_list = self.get_text_from_img(
            image_raw, 
            self.mllm_config['prompt_q'], 
            llm_dict, 
            self.mllm_config['use_gene_prompt'], 
            self.mllm_config['clip_use_bg_text'],
            class_name=gt_classes,
            device=device
        )

        return textfg_phrase_list, textbg_phrase_list, textfg_word_list, textbg_word_list, bbox_ori_list

    def cleanup(self):
        """清理资源"""
        if self.llm_dict and 'model' in self.llm_dict:
            del self.llm_dict['model']
            self.llm_dict = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    getdesc = GetDesc()
    
    # 配置路径
    img_path = '/home/renpeng/OVCamo_Datasets/OVCamo_Test/image/'
    out_dir = '/home/renpeng/Projects/TFOVCOS/env/descriptions1.json'
    device = "cuda:0"  # 或 "cpu"

    try:
        # 处理目录中的所有图片
        data = getdesc.process_directory(img_path, out_dir, device=device)
        print(f"成功处理 {len([v for v in data.values() if 'error' not in v])} 张图片")
        
        # 统计失败的数量
        failed_count = len([v for v in data.values() if 'error' in v])
        if failed_count > 0:
            print(f"有 {failed_count} 张图片处理失败")
            
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    finally:
        # 清理资源
        getdesc.cleanup()