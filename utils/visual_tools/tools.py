import os

import matplotlib
# from IPython.core.display_functions import display

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt




def plot_image_with_bboxes_and_points(
    cur_image,
    bbox=None,
    points=None,
    labels=None,
    word_fg=None,
    word_bg=None,
    mask=None,
    save_path=None,   # 想保存就给路径；不想保存就留 None
    show=True         # 想显示就 True
):
    # 1) 明确创建 figure/ax
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cur_image)

    # 2) 画 bbox
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)

    # 3) 画点
    if points is not None and labels is not None:
        for pt, label in zip(points, labels):
            x, y = pt
            color = 'red' if label == 1 else 'blue'
            ax.scatter(x, y, color=color, s=10, marker='o')

    # 4) 叠加 mask
    if mask is not None:
        ax.imshow(mask, cmap='jet', alpha=0.5)

    # 5) 文本（你那边传的是 [word, phrase] 列表，也能显示；想更精确可改成 str）
    if word_bg is not None and word_fg is not None:
        ax.text(10, cur_image.shape[0] - 30, f"Foreground: {word_fg}",
                color='lime', fontsize=12, weight='bold')
        ax.text(10, cur_image.shape[0] - 60, f"Background: {word_bg}",
                color='cyan', fontsize=12, weight='bold')

    ax.axis('off')

    # 6) 可选保存
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    # 7) 在 notebook 中用 display 显示，然后再关闭
    # if show:
    #     display(fig)
    plt.close(fig)