import numpy as np
import cv2
import torch

def get_multi_bboxes(cam, cam_thr=None, value_thr_in_bbx=None):
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.ascontiguousarray(cam)
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    boxes = []
    if len(contours) != 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            max_in_contour = np.array([cam[index] for index in get_all_index_from_contour(contour, cam)]).max()
            if max_in_contour >= value_thr_in_bbx * np.max(cam) and area != 0:
                x, y, w, h = cv2.boundingRect(contour)
                estimated_bbox = [x, y, x + w, y + h]
                boxes.append(estimated_bbox)
    else:
        estimated_bbox = [0, 0, 1, 1]
        boxes.append(estimated_bbox)

    return boxes

def get_points_labels_from_bbx(bbxes, attn, sample_point=5):
    points = []
    labels = []
    # 如果没有有效框，返回空列表（让调用方去做兜底，而不是 None）
    if bbxes == [[0, 0, 1, 1]] or bbxes is None or len(bbxes) == 0:
        return [], []

    H, W = attn.shape[:2]
    attn = np.asarray(attn)

    for bbx in bbxes:
        x, y, x1, y1 = map(int, bbx)
        x = max(0, min(x, W - 1)); x1 = max(0, min(x1, W - 1))
        y = max(0, min(y, H - 1)); y1 = max(0, min(y1, H - 1))
        if x1 <= x or y1 <= y:
            continue

        bbx_region = attn[y:y1, x:x1]  # 半开区间，避免 w/h=0
        if bbx_region.size == 0:
            continue

        flat = bbx_region.reshape(-1)
        n = flat.size
        k = min(sample_point, n)

        # 前景/背景各取 k 个且尽量不重复
        fore_idx = np.argpartition(flat, -k)[-k:]
        back_idx = np.argpartition(flat, k)[:k]

        fy, fx = np.unravel_index(fore_idx, bbx_region.shape)
        by, bx = np.unravel_index(back_idx, bbx_region.shape)

        fg_pts = np.stack([fx + x, fy + y], axis=1)
        bg_pts = np.stack([bx + x, by + y], axis=1)

        # 去重
        def _unique_pts(arr):
            if arr.size == 0:
                return arr
            arr_u = np.unique(arr, axis=0)
            return arr_u

        fg_pts = _unique_pts(fg_pts)
        bg_pts = _unique_pts(bg_pts)

        # 若不足 sample_point，则在框内随机补点，避免全部重合
        rng = np.random.default_rng()
        def _fill_pts(arr):
            if arr.shape[0] >= sample_point:
                return arr[:sample_point]
            need = sample_point - arr.shape[0]
            rx = rng.integers(x, x1, size=need)
            ry = rng.integers(y, y1, size=need)
            extra = np.stack([rx, ry], axis=1)
            return np.vstack([arr, extra])

        fg_pts = _fill_pts(fg_pts)
        bg_pts = _fill_pts(bg_pts)

        input_points = np.vstack([fg_pts, bg_pts]).tolist()
        label = [1] * sample_point + [0] * sample_point

        points.append(input_points)
        labels.append(label)

    return points, labels

def get_all_index_from_contour(contour, cam):
    mask = np.zeros_like(cam)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    contour_pixels = cv2.findNonZero(mask)
    if contour_pixels is None:
        return []
    return [(x[0][1], x[0][0]) for x in contour_pixels.tolist()]

# 使用示例
# 假设您的cam形状是[1, 576, 61]
cam_tensor = torch.randn([1, 576, 61])  # 示例数据

# 1. 首先对cam进行归一化
cam_normalized = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-8)

# 2. 转换为numpy数组并取第一个通道（因为形状是[1,576,61]）
cam_2d = cam_normalized[0,:,:]  # 形状变为[576, 61]

# 3. 确保数组是连续的
cam_2d = np.ascontiguousarray(cam_2d)

# 4. 获取边界框
boxes = get_multi_bboxes(cam_2d, cam_thr=0.3, value_thr_in_bbx=0.9)
print(f"检测到的边界框: {boxes}")

# 5. 从边界框中获取点和标签
points, labels = get_points_labels_from_bbx(boxes, cam_2d, sample_point=5)

print(f"点数: {len(points)}")
print(f"标签数: {len(labels)}")
    
