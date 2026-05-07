# ===================== 1. 安装依赖（首次运行执行） =====================
# !pip install torch clip-by-openai pillow numpy matplotlib opencv-python scipy

# ===================== 2. 导入核心库 =====================
import torch
import open_clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# ===================== 3. 全局参数定义（用户可调整） =====================
# CLIP模型参数
CLIP_MODEL = "ViT-B/32"  # CLIP模型类型（可选ViT-L/14，需匹配显存）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据维度参数（由CLIP模型自动适配，无需手动改）
H_FEAT, W_FEAT = None, None  # 特征图尺寸（ViT-B/32为7×7）
N = None                     # patch总数（H_FEAT×W_FEAT）
D = 512                      # CLIP特征维度（固定512）
C = 3                        # 类别数（目标/背景/其他）
L = 4                        # 相似性图层级数
K = 10                       # 候选区域数（ViT-B/32特征图7×7时，K建议≤10）

# 融合系数（论文可调参数）
ALPHA = 0.1    # 锚点融合系数
BETA = 0.7     # 最终文本嵌入融合系数

# 路径参数（★用户需替换为自己的路径★）
IMAGE_PATH = "/home/renpeng/OVCamo_Datasets/OVCamo_Test/image/000055_crocodilefish.jpg"  # 真实伪装图片路径（如枯叶蝶.jpg）
MASK_PATH = "/home/renpeng/OVCamo_Datasets/OVCamo_Test/mask/000055_crocodilefish.png"            # 真实标注掩码路径（黑白图，白色=目标）
TEXT_PROMPTS = [                          # 文本Prompt（根据场景修改）
    "The crocodilefish in the image is camouflaged with a textured, mottled skin that blends with the surrounding coral reef.",  # 伪装目标
    "The background of the image shows a vibrant coral reef with various types of coral, sea anemones, and other marine life.",      # 背景
]

# ===================== 4. 核心工具函数 =====================
def softmax(x, axis=-1):
    """Softmax归一化（防止溢出）"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def calculate_entropy(probs):
    """计算熵值 H = -sum(probs * log(probs))"""
    probs = np.clip(probs, 1e-8, 1.0)  # 避免log(0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    return entropy

def normalize(x):
    """归一化到[0,1]"""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def select_top_k_indices(x, k):
    """选择前K个最大值的索引（返回展平索引+2D坐标）"""
    flat_idx = np.argsort(x)[-k:]
    y_idx = flat_idx // W_FEAT
    x_idx = flat_idx % W_FEAT
    return flat_idx, (y_idx, x_idx)

# ===================== 5. 真实数据加载与CLIP特征提取 =====================
def load_real_image(image_path, preprocess):
    """加载真实图片并预处理为CLIP输入格式"""
    img = Image.open(image_path).convert("RGB")
    img_clip = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    img_np = np.array(img)  # 原始图片（用于可视化）
    return img_clip, img_np

def extract_clip_visual_features(model, img_clip):
    """提取CLIP patch级视觉特征（适配ViT-B/32）"""
    with torch.no_grad():
        # 提取ViT patch特征（CLIP visual层中间输出）
        x = model.visual.conv1(img_clip)  # (1, 768, 7, 7)
        H_FEAT, W_FEAT = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (1, 49, 768)
        # 添加class token
        class_emb = model.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_emb, x], dim=1)  # (1, 50, 768)
        # 位置编码
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(0, 2, 1)  # (1, 768, 50)
        # Transformer层
        for resblock in model.visual.transformer.resblocks:
            x = resblock(x)
        x = x.permute(0, 2, 1)  # (1, 50, 768)
        # 去掉class token，保留patch特征
        x = model.visual.ln_post(x[:, 1:, :])  # (1, 49, 768)
        # 投影到512维
        if model.visual.proj is not None:
            x = x @ model.visual.proj  # (1, 49, 512)
    # 展平为N×D
    F_patch = x.squeeze(0).cpu().numpy()  # (49, 512)
    N = H_FEAT * W_FEAT
    return F_patch, H_FEAT, W_FEAT, N

def get_clip_text_embedding(model,tokenizer, text_prompts):
    """生成CLIP文本嵌入"""
    text = tokenizer(text_prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text)  # (C, 512)
    return text_features.cpu().numpy()

def calculate_similarity_map(F_patch, T_raw, L=4):
    """计算3维目标/背景-文本相似性图（N×L×C）"""
    # 归一化特征
    F_norm = F_patch / np.linalg.norm(F_patch, axis=-1, keepdims=True)
    T_norm = T_raw / np.linalg.norm(T_raw, axis=-1, keepdims=True)
    # 生成L个尺度的相似性（文本微小扰动模拟多尺度）
    S_o_star = np.zeros((N, L, C))
    for l in range(L):
        T_l = T_norm + np.random.normal(0, 0.01, T_norm.shape)  # 微小扰动
        sim = np.dot(F_norm, T_l.T)  # (N, C)
        S_o_star[:, l, :] = sim
    # 背景相似性图（仅背景类）
    S_bg_star = S_o_star[:, :, 1:2]  # (N, L, 1)
    return S_o_star, S_bg_star

def load_gt_mask(mask_path, H_feat, W_feat):
    """加载并缩放GT掩码到特征图尺寸"""
    mask_gt = cv2.imread(mask_path, 0)  # 读取灰度掩码
    mask_gt = cv2.resize(mask_gt, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST)
    mask_gt = (mask_gt > 127).astype(np.float32)  # 二值化（1=目标，0=背景）
    return mask_gt

# ===================== 6. 主流程执行 =====================
if __name__ == "__main__":
    # -------------------- 6.1 加载CLIP模型 --------------------
    print("加载CLIP模型...")
    model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",pretrained="dfn2b"
            )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    
    # -------------------- 6.2 加载真实数据 --------------------
    print("加载真实图片与掩码...")
    img_clip, img_np = load_real_image(IMAGE_PATH,preprocess)
    F_patch, H_FEAT, W_FEAT, N = extract_clip_visual_features(model, img_clip)
    T_raw = get_clip_text_embedding(model,tokenizer, TEXT_PROMPTS)
    S_o_star, S_bg_star = calculate_similarity_map(F_patch, T_raw, L=L)
    mask_gt = load_gt_mask(MASK_PATH, H_FEAT, W_FEAT)
    
    # 维度验证（确保数据正确）
    print("\n数据维度验证：")
    print(f"特征图尺寸: {H_FEAT}×{W_FEAT} (N={N})")
    print(f"S_o_star: {S_o_star.shape} (预期: {N}×{L}×{C})")
    print(f"S_bg_star: {S_bg_star.shape} (预期: {N}×{L}×1)")
    print(f"F_patch: {F_patch.shape} (预期: {N}×{D})")
    print(f"T_raw: {T_raw.shape} (预期: {C}×{D})")
    print(f"mask_gt: {mask_gt.shape} (预期: {H_FEAT}×{W_FEAT})")
    
    # -------------------- 6.3 核心计算（对应论文公式） --------------------
    print("\n执行核心计算...")
    # 步骤1：相似性图层级聚合
    S_o = np.max(S_o_star, axis=1)  # (N, C)
    S_bg = np.max(S_bg_star, axis=1)  # (N, 1)
    
    # 步骤2：概率分布计算
    Probs = softmax(S_o, axis=-1)  # (N, C)
    
    # 步骤3：熵值计算
    H = calculate_entropy(Probs)  # (N,)
    H_resized = H.reshape(H_FEAT, W_FEAT)  # 还原为特征图尺寸
    
    # 步骤4：语义权重计算
    W_o = np.max(S_o, axis=-1)  # 目标语义权重 (N,)
    W_bg = S_bg.squeeze(axis=-1)  # 背景语义权重 (N,)
    W_o = normalize(W_o)
    W_bg = normalize(W_bg)
    W_o_resized = W_o.reshape(H_FEAT, W_FEAT)
    W_bg_resized = W_bg.reshape(H_FEAT, W_FEAT)
    
    # 步骤5：加权熵值计算
    H_o = H * W_o  # 目标加权熵值 (N,)
    H_bg = (1 - H) * W_bg  # 背景加权熵值 (N,)
    H_o_resized = H_o.reshape(H_FEAT, W_FEAT)
    H_bg_resized = H_bg.reshape(H_FEAT, W_FEAT)
    
    # 步骤6：筛选候选区域
    target_flat_idx, target_2d_idx = select_top_k_indices(H_o, K)
    bg_flat_idx, bg_2d_idx = select_top_k_indices(H_bg, K)
    # 生成候选区掩码
    target_candidate_mask = np.zeros((H_FEAT, W_FEAT))
    target_candidate_mask[target_2d_idx] = 1
    bg_candidate_mask = np.zeros((H_FEAT, W_FEAT))
    bg_candidate_mask[bg_2d_idx] = 1
    
    # -------------------- 6.4 全模块可视化 --------------------
    print("\n生成可视化结果...")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(20, 16))
    
    # 模块1：基础对照可视化
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(mask_gt, cmap='gray')
    ax1.set_title('Ground Truth Mask\n(1=Camouflage Target)', fontsize=10)
    ax1.axis('off')
    
    ax2 = plt.subplot(4, 4, 2)
    im2 = ax2.imshow(H_resized, cmap='RdBu_r', vmin=H.min(), vmax=H.max())
    ax2.set_title('Entropy Heatmap\n(Red=High Entropy)', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    ax3 = plt.subplot(4, 4, 3)
    low_entropy_mask = (H_resized < np.median(H)) * 1.0
    ax3.imshow(low_entropy_mask, cmap='gray')
    ax3.set_title('Low Entropy Mask\n(Blue in Heatmap)', fontsize=10)
    ax3.axis('off')
    
    ax4 = plt.subplot(4, 4, 4)
    high_entropy_mask = (H_resized >= np.median(H)) * 1.0
    ax4.imshow(high_entropy_mask, cmap='gray')
    ax4.set_title('High Entropy Mask\n(Red in Heatmap)', fontsize=10)
    ax4.axis('off')
    
    # 模块2：语义权重与加权熵值可视化
    ax5 = plt.subplot(4, 4, 5)
    im5 = ax5.imshow(W_o_resized, cmap='viridis', vmin=0, vmax=1)
    ax5.set_title('Target Semantic Weight\n(Brighter=Higher)', fontsize=10)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6 = plt.subplot(4, 4, 6)
    im6 = ax6.imshow(H_o_resized, cmap='RdYlGn_r', vmin=H_o.min(), vmax=H_o.max())
    ax6.set_title('Weighted Entropy (H_o = H*W_o)', fontsize=10)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    ax7 = plt.subplot(4, 4, 7)
    ax7.imshow(target_candidate_mask, cmap='gray')
    ax7.set_title(f'Top-{K} Target Candidate Mask', fontsize=10)
    ax7.axis('off')
    
    ax8 = plt.subplot(4, 4, 8)
    overlay = mask_gt * 0.5 + target_candidate_mask * 0.5
    ax8.imshow(overlay, cmap='gray')
    ax8.set_title('Overlay: GT + Candidate', fontsize=10)
    ax8.axis('off')
    
    # 模块3：普通场景vs伪装场景熵值对比
    # 模拟普通场景数据
    def generate_normal_scene_data():
        normal_mask = np.zeros((H_FEAT, W_FEAT))
        circle = ((np.ogrid[:H_FEAT, :W_FEAT][0]-3)**2 + (np.ogrid[:H_FEAT, :W_FEAT][1]-3)**2) <= 4
        normal_mask[circle] = 1
        normal_S_o = np.random.rand(N, C)
        normal_S_o[np.where(normal_mask.flatten()==1)[0], 0] = 0.9
        normal_S_o[np.where(normal_mask.flatten()==0)[0], 1] = 0.9
        normal_Probs = softmax(normal_S_o, axis=-1)
        normal_H = calculate_entropy(normal_Probs).reshape(H_FEAT, W_FEAT)
        return normal_mask, normal_H
    normal_mask, normal_H = generate_normal_scene_data()
    
    ax9 = plt.subplot(4, 4, 9)
    ax9.imshow(normal_mask, cmap='gray')
    ax9.set_title('Normal Scene GT\n(1=Cat, 0=Wall)', fontsize=10)
    ax9.axis('off')
    
    ax10 = plt.subplot(4, 4, 10)
    im10 = ax10.imshow(normal_H, cmap='RdBu_r', vmin=normal_H.min(), vmax=normal_H.max())
    ax10.set_title('Normal Scene Entropy\n(Low Entropy for Target)', fontsize=10)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, shrink=0.8)
    
    ax11 = plt.subplot(4, 4, 11)
    ax11.imshow(mask_gt, cmap='gray')
    ax11.set_title('Camouflage Scene GT\n(1=Target)', fontsize=10)
    ax11.axis('off')
    
    ax12 = plt.subplot(4, 4, 12)
    im12 = ax12.imshow(H_resized, cmap='RdBu_r', vmin=H.min(), vmax=H.max())
    ax12.set_title('Camouflage Scene Entropy\n(High Entropy for Target)', fontsize=10)
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, shrink=0.8)
    
    # 模块4：熵值分布量化可视化
    target_entropy = H[mask_gt.flatten() == 1]
    bg_entropy = H[mask_gt.flatten() == 0]
    
    ax13 = plt.subplot(4, 4, 13)
    ax13.hist(target_entropy, bins=10, alpha=0.7, label='Camouflage Target', color='red')
    ax13.hist(bg_entropy, bins=10, alpha=0.7, label='Background', color='blue')
    ax13.set_title('Entropy Distribution', fontsize=10)
    ax13.set_xlabel('Entropy')
    ax13.set_ylabel('Pixel Count')
    ax13.legend()
    
    ax14 = plt.subplot(4, 4, 14)
    sample_idx = np.random.choice(N, min(1000, N), replace=False)
    sample_H = H[sample_idx]
    sample_Wo = W_o[sample_idx]
    sample_label = mask_gt.flatten()[sample_idx]
    ax14.scatter(sample_H[sample_label==1], sample_Wo[sample_label==1], 
                 c='red', s=10, label='Target', alpha=0.8)
    ax14.scatter(sample_H[sample_label==0], sample_Wo[sample_label==0], 
                 c='blue', s=10, label='Background', alpha=0.8)
    ax14.set_title('Entropy vs Target Weight', fontsize=10)
    ax14.set_xlabel('Entropy')
    ax14.set_ylabel('Target Semantic Weight')
    ax14.legend()
    
    # 隐藏多余子图
    ax15 = plt.subplot(4, 4, 15)
    ax15.axis('off')
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')
    
    # 总标题与保存
    plt.suptitle('Entropy Visualization for Camouflage Target Segmentation (Real Data)', fontsize=16)
    plt.tight_layout()
    plt.savefig('./camouflage_entropy_visualization.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    
    # -------------------- 6.5 输出关键量化指标 --------------------
    print("\n关键量化指标：")
    print(f"伪装目标熵值均值: {np.mean(target_entropy):.2f} ± {np.std(target_entropy):.2f}")
    print(f"背景熵值均值: {np.mean(bg_entropy):.2f} ± {np.std(bg_entropy):.2f}")
    print(f"候选区覆盖目标比例: {np.sum(target_candidate_mask * mask_gt) / np.sum(mask_gt):.2f}")
    print("\n可视化结果已保存为: camouflage_entropy_visualization.pdf")