r"""
方向 F：U-Net Patch 训练检测（使用 segmentation_models_pytorch）

安装依赖（在服务器上运行一次即可）：
    pip install segmentation-models-pytorch

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-F.py \
    --epochs 50 \
    --batch-size 10 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 1024 \
    --input-w 512 \
    --patch-mode \
    --patch-size 384 \
    --clf-pos-weight 5.0 \
    --tversky-alpha 0.3 \
    --tversky-beta 0.7 \
    --val-heatmap-threshold 0.5 \
    --val-heatmap-dilation 15 \
    --val-iou-threshold 0.1 \
    --patience 5 \
    --neg-hard-ratio 0.8 \
    --min-detection-area 200 \
    --box-nms-thresh 0.3 \
    --monitor-metric fbeta2 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --hide-progress-bar

与方向 E 的核心差异：
  - 方向 E：全图训练（1024×512），正样本像素占比约 0.2%，pos_weight=50 仍无法
            抵抗背景梯度主导（背景:病灶梯度 ≈ 10:1），训练中期出现保守性坍缩
            （TP/FP 同步收缩，几乎不预测任何病灶），前 4 epoch 往往是全程最佳。
  - 方向 F：Patch 训练 + 全图推理。训练时以 GT bbox 为中心裁取 patch_size×patch_size
            patch，正负样本各 50%（负样本来自无病灶图像的随机裁剪），正样本像素比
            提升至 5–20%，梯度不平衡比从 ~10:1 降至 ~2:1，保守性坍缩的梯度根因被
            消除。验证和推理仍使用全图（U-Net 全卷积，支持任意尺寸输入）。

GT 生成：将 xyxy bbox 转为硬掩码（框内像素=1.0，框外=0.0），在 patch 坐标系中
生成，用 BCEWithLogitsLoss + Tversky loss 联合监督。

后处理复用方向 E 管线：sigmoid → 阈值 → 膨胀 → 连通域 → 外接矩形 → IoU 匹配 GT。

─────────────────────────────────────────────────────────────────────────────
改版历史
─────────────────────────────────────────────────────────────────────────────

rec_46（初版）
  基础：方向 E 的 rec_45_upd_3 代码（Tversky loss、差异化 lr、无冻结）
  核心改动：Patch 训练（--patch-mode），替换全图训练 DataLoader
  正样本 patch：以 GT bbox 为中心裁取，随机 jitter，resize 到 patch_size×patch_size
  负样本 patch：从纯阴性图像随机裁取，每 epoch 重采样以与正样本数量匹配
  pos_weight：50 → 5（patch 内正样本比提升，超大权重反而冗余）
  结果：保守性坍缩根因消除（全程 TP≈140-185），但 FP 极高（@0.9 约 400-900），
        模型在全图推理时对正常乳腺实质产生大量假激活，最优 F1 0.3308（epoch 8）。
        最优阈值始终为 0.9，说明低置信度预测弥散于全图。
  问题诊断：负样本仅来自纯阴性图像，模型未学会在有病灶图像的非病灶区域保持低输出；
            训练（256²）→推理（1024×512）域差异：全图背景面积约 8× patch，模型从未
            见过，对乳腺实质产生过度激活。
  监控指标：F1（事后分析更适合用 Recall，因框级 IoU 制约了平凡解）。

rec_46_upd_1（负样本改造 + Recall 监控）
  基础：rec_46 代码
  核心改动 1：PatchDataset 负样本改造——50% 来自纯阴性图像（easy neg），
              50% 来自正样本图像的非 bbox 区域（hard neg）。Hard neg 要求裁取的
              crop 与所有 GT box 的交叠（交叉面积/GT 面积）< 0.3，最多尝试 15 次，
              失败则直接使用（保守 fallback）。这样模型能学到"乳腺实质 ≠ 病灶"，
              降低全图推理时的假激活。
  核心改动 2：新增 --monitor-metric（f1 / recall / fbeta2），--neg-hard-ratio（default 0.5）。
              recall 模式固定阈值 0.5 的 Recall；fbeta2 模式监控多阈值最大 F2 分数
              （β=2，Recall 权重是 Precision 的 4 倍，兼顾 FP 惩罚）；f1 保持原有行为。
  预期：FP 降低，@0.5 或 @0.7 阈值的 F1 提升；Recall 曲线更稳定。

rec_46_upd_2（hard neg 比例提升 + Fbeta2 监控）
  基础：rec_46_upd_1 代码
  核心改动 1：修复 best_recall 计算——从跨阈值 argmax 改为固定 thresh=0.5 的 Recall，
              避免 checkpoint 退化为低阈值高 FP 版本。
  核心改动 2：新增 --monitor-metric fbeta2；F2 = 5*P*R/(4P+R)，β=2 使 Recall 权重
              是 Precision 的 4 倍，argmax 跨阈值安全（低 Precision 自然压制低阈值）。
  核心改动 3：--neg-hard-ratio 建议提升到 0.8，给模型更多正样本图的非病灶区域负样本。
  核心改动 4（rec_46_upd_2 补丁）：新增 --min-detection-area（default 200 px²）传入
              heatmap_to_boxes()，替换硬编码的 min_component_area=50。原始图像分辨率
              912×1520，GT box 面积 1% 分位为 759 px²，50 px² ≈ 7×7 像素（噪声级别）
              导致散点激活各自计为一个 FP 框。200 px² ≈ 14×14 像素 ≈ 2.3mm，可过滤
              亚病灶噪声而不影响真实病灶检测。
  预期：checkpoint 选择更合理；FP 进一步下降而 Recall 维持高位。

rec_46_upd_3（NMS + 更大 patch size）
  基础：rec_46_upd_2 代码
  核心改动 1：新增 --box-nms-thresh（default 0.0 = 禁用）。在 heatmap_to_boxes() 连通
              域提取后对所有候选框做贪心 IoU-based NMS，合并同一区域内的碎片化 FP
              框，减少重复计数。推荐值 0.3。
  核心改动 2：默认 patch-size 建议从 256 改为 384，增大训练时感受野，使模型能看到
              激活区域的周边组织，从根源减少高置信 FP（病灶样实质误激活）。
              batch-size 相应从 16 降至 10（384×384 约是 256×256 的 2.25×，RTX 4090
              24GB 下 batch=10 安全；如出现 OOM 可降至 8）。
  核心改动 3：patience 建议从 20 降至 5。rec_46_upd_2 实验表明 F2 在 epoch 4 达峰，
              随后系统性退化（过拟合），patience=20 浪费计算且不能改善 checkpoint。
  预期：NMS 直接降低碎片化 FP 计数；更大 patch 在 Stage 1 层面减少高置信 FP；
        更短 patience 使训练在退化前及时停止。
"""

from __future__ import annotations

import os
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import atexit
import datetime
import json
import random
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # type: ignore

try:
    from torchvision.models import resnet50 as _resnet50, ResNet50_Weights as _ResNet50Weights
except ImportError:
    try:
        from torchvision.models import resnet50 as _resnet50  # type: ignore
        _ResNet50Weights = None  # type: ignore
    except ImportError:
        _resnet50 = None  # type: ignore
        _ResNet50Weights = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8 with 3 channels, with CLAHE for grayscale."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] > 4:
            img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Greedy NMS. Returns indices of kept boxes (sorted by score descending)."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while len(order) > 0:
        idx = int(order[0])
        keep.append(idx)
        if len(order) == 1:
            break
        rest = order[1:]
        bx = boxes[idx]
        br = boxes[rest]
        ix1 = np.maximum(bx[0], br[:, 0])
        iy1 = np.maximum(bx[1], br[:, 1])
        ix2 = np.minimum(bx[2], br[:, 2])
        iy2 = np.minimum(bx[3], br[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_bx = (bx[2] - bx[0]) * (bx[3] - bx[1])
        area_br = (br[:, 2] - br[:, 0]) * (br[:, 3] - br[:, 1])
        union = area_bx + area_br - inter
        iou = inter / np.maximum(union, 1e-6)
        order = rest[iou < iou_thresh]
    return keep


def heatmap_to_boxes(
    heatmap: np.ndarray,
    threshold: float,
    dilation_size: int = 30,
    min_component_area: int = 50,
    nms_iou_thresh: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if heatmap.size == 0 or float(heatmap.max()) < threshold:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    mask = (heatmap >= threshold).astype(np.uint8)
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        mask = cv2.dilate(mask, kernel)
    n_labels, labels_map = cv2.connectedComponents(mask, connectivity=8)
    boxes_list: List[List[float]] = []
    scores_list: List[float] = []
    for label in range(1, n_labels):
        component_mask = labels_map == label
        if int(component_mask.sum()) < min_component_area:
            continue
        ys, xs = np.where(component_mask)
        boxes_list.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0])
        scores_list.append(float(heatmap[component_mask].max()))
    if not boxes_list:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes_arr = np.array(boxes_list, dtype=np.float32)
    scores_arr = np.array(scores_list, dtype=np.float32)
    if nms_iou_thresh > 0.0 and len(boxes_arr) > 1:
        keep = _nms_boxes(boxes_arr, scores_arr, nms_iou_thresh)
        boxes_arr = boxes_arr[keep]
        scores_arr = scores_arr[keep]
    return boxes_arr, scores_arr


def compute_iou_matches(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    matched_gt = [False] * int(gt_boxes.shape[0])
    matched_pred = [False] * int(pred_boxes.shape[0])
    for pi in range(pred_boxes.shape[0]):
        best_iou = 0.0
        best_gi = -1
        px1, py1, px2, py2 = float(pred_boxes[pi, 0]), float(pred_boxes[pi, 1]), float(pred_boxes[pi, 2]), float(pred_boxes[pi, 3])
        for gi in range(gt_boxes.shape[0]):
            if matched_gt[gi]:
                continue
            gx1, gy1, gx2, gy2 = float(gt_boxes[gi, 0]), float(gt_boxes[gi, 1]), float(gt_boxes[gi, 2]), float(gt_boxes[gi, 3])
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt[best_gi] = True
            matched_pred[pi] = True
    tp = sum(matched_pred)
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - sum(matched_gt)
    return int(tp), int(fp), int(fn)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray          # (N, 4) xyxy in original image coords
    orig_size: Tuple[float, float]  # (H, W)


def load_samples(
    csv_path: Path,
    images_root: Path,
    split_name: str,
) -> List[Sample]:
    """Load all samples for a split from the VinDr CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    samples: List[Sample] = []
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
        boxes: np.ndarray = valid.to_numpy(dtype=np.float32) if not valid.empty else np.zeros((0, 4), dtype=np.float32)

        image_path = images_root / str(patient_id) / f"{image_id}"

        if boxes.size > 0:
            invalid = int(np.sum((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])))
            if invalid > 0:
                print(f"[Warning] Found {invalid} invalid boxes in {image_path}")

        first = group.iloc[0]
        orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
        orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

        samples.append(Sample(
            patient_id=str(patient_id),
            image_id=str(image_id),
            image_path=image_path,
            boxes=boxes,
            orig_size=(orig_h, orig_w),
        ))

    return samples


def patient_level_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split samples into train/val by patient to avoid leakage."""
    patients = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    train_idx = [i for i, s in enumerate(samples) if s.patient_id not in val_patients]
    val_idx = [i for i, s in enumerate(samples) if s.patient_id in val_patients]
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# GT heatmap generation
# ─────────────────────────────────────────────────────────────────────────────

def make_bbox_heatmap(
    h: int,
    w: int,
    boxes: np.ndarray,   # (N, 4) xyxy in (h, w) coordinate system
) -> np.ndarray:
    """Generate a binary GT heatmap by filling each GT bbox with 1."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if boxes.size == 0:
        return heatmap
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] = 1.0
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — full-image (used for validation)
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """Full-image segmentation dataset.  Used for validation only in patch mode.

    Each item is (image_tensor [3, H, W], heatmap_tensor [1, H, W]).
    Image is resized to (input_h, input_w).  GT boxes are scaled accordingly,
    then converted to hard binary bbox masks (pixels inside any GT box = 1, else 0).
    """

    def __init__(
        self,
        samples: List[Sample],
        indices: List[int],
        input_h: int,
        input_w: int,
        augment: bool = False,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 0.2,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_brightness_delta = aug_brightness_delta
        self.rng = random.Random(seed + epoch * 1013)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[self.indices[idx]]

        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            hm_t = torch.zeros(1, self.input_h, self.input_w, dtype=torch.float32)
            return img_t, hm_t

        orig_h, orig_w = img.shape[:2]
        boxes = sample.boxes.copy()

        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        scaled_boxes = boxes.copy()
        if scaled_boxes.size > 0:
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 3] *= scale_y

        if self.augment:
            if self.rng.random() < self.aug_hflip_prob:
                img_resized = img_resized[:, ::-1, :].copy()
                if scaled_boxes.size > 0:
                    old_x1 = scaled_boxes[:, 0].copy()
                    old_x2 = scaled_boxes[:, 2].copy()
                    scaled_boxes[:, 0] = self.input_w - old_x2
                    scaled_boxes[:, 2] = self.input_w - old_x1
            if self.aug_brightness_delta > 0:
                delta = (self.rng.random() * 2 - 1) * self.aug_brightness_delta * 255
                img_resized = np.clip(img_resized.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        heatmap = make_bbox_heatmap(self.input_h, self.input_w, scaled_boxes)

        img_t = image_to_tensor(img_resized)
        hm_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
        return img_t, hm_t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — patch-based (used for training in patch mode)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """Patch-based segmentation dataset for balanced positive/negative training.

    Builds a balanced list of items each epoch:
      - Positive patches: one crop per GT bounding box, centered on the box
        with optional random jitter, ensuring the bbox is contained in the crop.
      - Easy negative patches: random crops from pure-negative training images
        (no GT lesions), resampled each epoch to match the positive count.
      - Hard negative patches: random crops from positive-sample images that do
        NOT significantly overlap any GT bbox (intersection/GT_area < 0.3).
        Fraction controlled by neg_hard_ratio (default 0.5).
        Teaches the model to suppress activations on normal breast tissue
        near (but not on) lesions, reducing FP in full-image inference.

    Crops are taken from the full input_h×input_w resized image.  The GT mask
    is computed for the full resized image, then cropped to the same region.
    Both crop and mask are resized to patch_size×patch_size before returning.

    Validation still uses SegDataset (full-image inference) — unchanged.
    """

    def __init__(
        self,
        samples: List[Sample],
        indices: List[int],
        input_h: int,
        input_w: int,
        patch_size: int = 256,
        margin_factor: float = 2.5,
        neg_patch_ratio: float = 1.0,
        neg_hard_ratio: float = 0.5,
        augment: bool = False,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 0.2,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        self.samples = samples
        self.input_h = input_h
        self.input_w = input_w
        self.patch_size = patch_size
        self.margin_factor = margin_factor
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_brightness_delta = aug_brightness_delta
        self.rng = random.Random(seed + epoch * 7919)

        # Build positive items: one entry per GT bbox
        positive_items: List[Tuple[int, int]] = []  # (sample_idx, box_idx)
        easy_neg_indices: List[int] = []   # pure-negative images
        hard_neg_indices: List[int] = []   # positive images (hard neg crops)
        for idx in indices:
            s = samples[idx]
            if s.boxes.shape[0] > 0:
                for bi in range(s.boxes.shape[0]):
                    positive_items.append((idx, bi))
                hard_neg_indices.append(idx)
            else:
                easy_neg_indices.append(idx)

        n_pos = len(positive_items)
        n_neg = max(1, int(n_pos * neg_patch_ratio))
        n_hard = int(n_neg * max(0.0, min(1.0, neg_hard_ratio)))
        n_easy = n_neg - n_hard

        # Fallback: if no pure-negative images exist, use all indices for easy neg
        if not easy_neg_indices:
            easy_neg_indices = list(indices)
        # Fallback: if no positive images exist for hard neg, use easy neg pool
        if not hard_neg_indices:
            hard_neg_indices = easy_neg_indices

        # Resample negative sources each epoch for diversity
        rng_init = random.Random(seed + epoch * 1013)
        sampled_easy = rng_init.choices(easy_neg_indices, k=n_easy) if n_easy > 0 else []
        sampled_hard = rng_init.choices(hard_neg_indices, k=n_hard) if n_hard > 0 else []

        # Build flat item list and shuffle
        self.items: List[Tuple[str, int, int]] = []
        for si, bi in positive_items:
            self.items.append(("pos", si, bi))
        for si in sampled_easy:
            self.items.append(("neg_easy", si, -1))
        for si in sampled_hard:
            self.items.append(("neg_hard", si, -1))
        rng_init.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_type, sample_idx, box_idx = self.items[idx]
        sample = self.samples[sample_idx]

        # Load full image
        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            blank = torch.zeros(3, self.patch_size, self.patch_size, dtype=torch.float32)
            return blank, torch.zeros(1, self.patch_size, self.patch_size, dtype=torch.float32)

        orig_h, orig_w = img.shape[:2]

        # Resize to model input size
        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Scale box coordinates to input_h×input_w space
        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        boxes = sample.boxes.copy()
        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0] * scale_x, 0, self.input_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2] * scale_x, 0, self.input_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1] * scale_y, 0, self.input_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3] * scale_y, 0, self.input_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # Determine crop region
        if item_type == "pos":
            if boxes.shape[0] == 0:
                # Fallback to negative crop if all boxes were filtered out
                cx1, cy1, cx2, cy2 = self._get_negative_crop()
            else:
                safe_idx = box_idx % boxes.shape[0]
                cx1, cy1, cx2, cy2 = self._get_positive_crop(boxes[safe_idx])
        elif item_type == "neg_hard":
            cx1, cy1, cx2, cy2 = self._get_hard_negative_crop(boxes)
        else:  # neg_easy
            cx1, cy1, cx2, cy2 = self._get_negative_crop()

        # Build GT mask for full image then crop
        heatmap = make_bbox_heatmap(self.input_h, self.input_w, boxes)
        img_patch = img_resized[cy1:cy2, cx1:cx2].copy()
        hm_patch = heatmap[cy1:cy2, cx1:cx2].copy()

        # Resize to patch_size × patch_size
        img_patch = cv2.resize(img_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        hm_patch = cv2.resize(hm_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if self.rng.random() < self.aug_hflip_prob:
                img_patch = img_patch[:, ::-1, :].copy()
                hm_patch = hm_patch[:, ::-1].copy()
            if self.aug_brightness_delta > 0:
                delta = (self.rng.random() * 2 - 1) * self.aug_brightness_delta * 255
                img_patch = np.clip(img_patch.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        img_t = image_to_tensor(img_patch)
        hm_t = torch.from_numpy(hm_patch).unsqueeze(0)  # (1, H, W)
        return img_t, hm_t

    def _get_positive_crop(self, box: np.ndarray) -> Tuple[int, int, int, int]:
        """Return crop region (rx1, ry1, rx2, ry2) centered on the GT box
        with random jitter, ensuring the bbox is contained in the crop."""
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Crop side: large enough to contain bbox with margin, at least patch_size
        crop_side = max(self.patch_size, int(max(bw, bh) * self.margin_factor))
        crop_side = min(crop_side, min(self.input_h, self.input_w))

        # Random jitter: shift center by up to 50% of available slack on each axis
        half = crop_side / 2.0
        slack_x = max(0.0, half - bw / 2.0)
        slack_y = max(0.0, half - bh / 2.0)
        cx += (self.rng.random() * 2 - 1) * slack_x * 0.5
        cy += (self.rng.random() * 2 - 1) * slack_y * 0.5

        # Compute initial crop bounds
        rx1 = int(cx - half)
        ry1 = int(cy - half)
        rx2 = rx1 + crop_side
        ry2 = ry1 + crop_side

        # Shift to stay within image bounds
        if rx1 < 0:
            rx2 -= rx1
            rx1 = 0
        if ry1 < 0:
            ry2 -= ry1
            ry1 = 0
        if rx2 > self.input_w:
            rx1 -= rx2 - self.input_w
            rx2 = self.input_w
        if ry2 > self.input_h:
            ry1 -= ry2 - self.input_h
            ry2 = self.input_h
        rx1, ry1 = max(0, rx1), max(0, ry1)
        return rx1, ry1, rx2, ry2

    def _get_negative_crop(self) -> Tuple[int, int, int, int]:
        """Return a random crop region for a negative patch."""
        max_x = max(0, self.input_w - self.patch_size)
        max_y = max(0, self.input_h - self.patch_size)
        rx1 = self.rng.randint(0, max_x) if max_x > 0 else 0
        ry1 = self.rng.randint(0, max_y) if max_y > 0 else 0
        return rx1, ry1, rx1 + self.patch_size, ry1 + self.patch_size

    def _crop_overlaps_gt(self, rx1: int, ry1: int, rx2: int, ry2: int, boxes: np.ndarray) -> bool:
        """Return True if the crop significantly overlaps any GT box.

        Criterion: intersection area / GT box area > 0.3 for any box.
        A crop that contains >30% of a GT box is considered a positive region.
        """
        if boxes.shape[0] == 0:
            return False
        for box in boxes:
            bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            ix1 = max(rx1, bx1)
            iy1 = max(ry1, by1)
            ix2 = min(rx2, bx2)
            iy2 = min(ry2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            gt_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            if inter / gt_area > 0.3:
                return True
        return False

    def _get_hard_negative_crop(self, boxes: np.ndarray) -> Tuple[int, int, int, int]:
        """Return a random crop region that does NOT significantly overlap GT boxes.

        Used for hard negative patches from positive-sample images.  Tries up to
        15 times to find a non-overlapping crop; falls back to the last attempt.
        """
        rx1 = ry1 = rx2 = ry2 = 0
        for _ in range(15):
            rx1, ry1, rx2, ry2 = self._get_negative_crop()
            if not self._crop_overlaps_gt(rx1, ry1, rx2, ry2, boxes):
                return rx1, ry1, rx2, ry2
        return rx1, ry1, rx2, ry2  # fallback: return last attempt


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(
    medical_backbone_path: Optional[str] = None,
) -> "torch.nn.Module":
    """Build a U-Net with ResNet50 encoder.

    Requires segmentation_models_pytorch.  The RadImageNet ResNet50 weights
    are loaded into the encoder using the same key-mapping logic as E.
    conv1 is adapted for grayscale-as-3channel input (channel-averaged weights).
    """
    if smp is None:
        raise RuntimeError(
            "segmentation_models_pytorch is not installed. "
            "Run: pip install segmentation-models-pytorch"
        )

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,   # we load RadImageNet manually below
        in_channels=3,
        classes=1,
        activation=None,        # raw logits; sigmoid applied externally
    )

    # Load ImageNet weights into encoder first as a baseline
    if _resnet50 is not None:
        try:
            try:
                imagenet_model = _resnet50(weights=_ResNet50Weights.DEFAULT)
            except Exception:
                imagenet_model = _resnet50(pretrained=True)  # type: ignore[call-arg]
            encoder_sd = {
                k: v for k, v in imagenet_model.state_dict().items()
                if not k.startswith("fc")
            }
            _result = model.encoder.load_state_dict(encoder_sd, strict=False)
            if _result is not None:
                missing, unexpected = _result
                print(f"[Info] Loaded ImageNet weights into encoder (missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print("[Info] Loaded ImageNet weights into encoder.")
        except Exception as exc:
            print(f"[Warning] Could not load ImageNet weights into encoder: {exc}")

    # Override with RadImageNet if provided
    if medical_backbone_path is not None:
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
            _idx_to_resnet = {
                "0": "conv1", "1": "bn1",
                "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
            }
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            _result = model.encoder.load_state_dict(stripped, strict=False)
            if _result is not None:
                missing, unexpected = _result
                print(f"[Info] Loaded RadImageNet backbone into encoder (missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print("[Info] Loaded RadImageNet backbone into encoder.")
        except Exception as exc:
            print(f"[Warning] Could not load medical backbone ({exc}). Keeping ImageNet weights.")

    # Adapt conv1 for grayscale-as-3channel (average over input channels)
    try:
        with torch.no_grad():
            mean_w = model.encoder.conv1.weight.mean(dim=1, keepdim=True)
            model.encoder.conv1.weight.copy_(mean_w.expand_as(model.encoder.conv1.weight))
        print("[Info] conv1 weights averaged across channels for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def tversky_loss(
    pred_sigmoid: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss. alpha weights FP, beta weights FN.

    With beta > alpha, FN is penalized more than FP, encouraging higher recall.
    Setting alpha=beta=0.5 is equivalent to Dice loss.
    """
    pred_flat = pred_sigmoid.reshape(-1)
    tgt_flat = target.reshape(-1)
    tp = (pred_flat * tgt_flat).sum()
    fp = (pred_flat * (1.0 - tgt_flat)).sum()
    fn = ((1.0 - pred_flat) * tgt_flat).sum()
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 5.0,
    bce_alpha: float = 0.5,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    """BCE + Tversky combined loss.

    bce_alpha: weight of BCE term (1 - bce_alpha applied to Tversky).
    tversky_alpha / tversky_beta: FP / FN weights in Tversky loss.
    """
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    pred_sigmoid = torch.sigmoid(logits)
    tversky = tversky_loss(pred_sigmoid, target, alpha=tversky_alpha, beta=tversky_beta)
    return bce_alpha * bce + (1.0 - bce_alpha) * tversky


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int,
    pos_weight: float = 5.0,
    bce_alpha: float = 0.5,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    disable_tqdm: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for i, (imgs, heatmaps) in enumerate(pbar):
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)

        logits = model(imgs)  # (B, 1, H, W)
        loss = combined_loss(
            logits, heatmaps,
            pos_weight=pos_weight,
            bce_alpha=bce_alpha,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / float(accumulation_steps)).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    model: torch.nn.Module,
    samples: List[Sample],
    val_indices: List[int],
    device: torch.device,
    input_h: int,
    input_w: int,
    val_batch_size: int,
    iou_threshold: float,
    dilation_size: int,
    min_component_area: int,
    nms_iou_thresh: float,
    epoch: int,
    epochs: int,
    disable_tqdm: bool = False,
) -> Dict[str, float]:
    """Validate with full-image U-Net inference: predicted heatmap → boxes → F1."""
    model.eval()

    multi_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    multi_stats: Dict[float, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in multi_thresholds}
    total_gt_boxes = 0

    pbar = tqdm(val_indices, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for sample_idx in pbar:
            sample = samples[sample_idx]
            orig_img = normalize_image(read_image_unicode(sample.image_path))
            orig_h, orig_w = orig_img.shape[:2]

            gt_boxes = sample.boxes.astype(np.float32).copy()
            if gt_boxes.size > 0:
                keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
                gt_boxes = gt_boxes[keep]

            # Resize and infer
            img_resized = cv2.resize(orig_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            img_t = image_to_tensor(img_resized).unsqueeze(0).to(device)  # (1, 3, H, W)
            logits = model(img_t)  # (1, 1, H, W)
            pred_heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W) in [0,1]

            # Scale predicted heatmap back to original image coordinates for IoU matching
            pred_heatmap_orig = cv2.resize(pred_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            for thresh in multi_thresholds:
                pred_boxes, _ = heatmap_to_boxes(pred_heatmap_orig, thresh, dilation_size,
                                                 min_component_area=min_component_area,
                                                 nms_iou_thresh=nms_iou_thresh)
                tp, fp, fn = compute_iou_matches(pred_boxes, gt_boxes, iou_threshold)
                multi_stats[thresh]["tp"] += tp
                multi_stats[thresh]["fp"] += fp
                multi_stats[thresh]["fn"] += fn

            total_gt_boxes += int(gt_boxes.shape[0]) if gt_boxes.size > 0 else 0

    # Compute F1 and Recall per threshold
    f1_per_thresh: Dict[float, float] = {}
    recall_per_thresh: Dict[float, float] = {}
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_per_thresh[thresh] = f1
        recall_per_thresh[thresh] = recall

    best_f1_thresh = max(f1_per_thresh, key=lambda t: f1_per_thresh[t])
    best_f1 = f1_per_thresh[best_f1_thresh]
    # Fix: use fixed thresh=0.5 instead of argmax across all thresholds.
    # Argmax always picks the lowest threshold (higher recall but unusable FP),
    # causing checkpoints to degrade toward low-confidence, high-FP predictions.
    best_recall_thresh: float = 0.5
    best_recall = recall_per_thresh[best_recall_thresh]
    # Fbeta2 (beta=2): weights recall 4x over precision; argmax is safe here
    # because low precision (from high FP at low thresholds) naturally penalises F2.
    fbeta2_per_thresh: Dict[float, float] = {}
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fbeta2 = (1 + 4) * prec * rec / max(4 * prec + rec, 1e-9)
        fbeta2_per_thresh[thresh] = fbeta2
    best_fbeta2_thresh = max(fbeta2_per_thresh, key=lambda t: fbeta2_per_thresh[t])
    best_fbeta2 = fbeta2_per_thresh[best_fbeta2_thresh]

    parts = []
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        parts.append(
            f"@{thresh}: TP={tp} FP={fp} FN={fn} "
            f"Rec={recall_per_thresh[thresh]:.3f} F1={f1_per_thresh[thresh]:.4f}"
        )
    print(f"  [Val] GT_boxes={total_gt_boxes} | {' | '.join(parts)}")
    print(
        f"  [BestF1] F1={best_f1:.4f} @ thresh={best_f1_thresh} | "
        f"[BestRecall] Recall={best_recall:.4f} @ thresh={best_recall_thresh} | "
        f"[BestFbeta2] F2={best_fbeta2:.4f} @ thresh={best_fbeta2_thresh}"
    )

    result: Dict[str, float] = {
        "best_thresh_f1": float(best_f1),
        "best_f1_thresh": float(best_f1_thresh),
        "best_recall": float(best_recall),
        "best_recall_thresh": float(best_recall_thresh),
        "best_fbeta2": float(best_fbeta2),
        "best_fbeta2_thresh": float(best_fbeta2_thresh),
        "val_gt_boxes": float(total_gt_boxes),
    }
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        result[f"tp@{thresh}"] = float(tp)
        result[f"fp@{thresh}"] = float(fp)
        result[f"fn@{thresh}"] = float(fn)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(save_path: Path, model: torch.nn.Module, meta: Dict[str, Any]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net patch-based segmentation model for VinDr lesion detection (Direction F).")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-h", type=int, default=1024, help="Resize height for model input")
    parser.add_argument("--input-w", type=int, default=512, help="Resize width for model input")
    parser.add_argument("--clf-pos-weight", type=float, default=50.0,
                        help="BCE pos_weight for foreground pixels. In patch mode, positive pixel "
                             "ratio is much higher (~5-20%%), so a value of 5.0 is recommended "
                             "instead of the full-image default of 50.0.")
    parser.add_argument("--bce-alpha", type=float, default=0.5, help="Weight of BCE in combined loss (1-alpha for Tversky)")
    parser.add_argument("--val-heatmap-threshold", type=float, default=0.5)
    parser.add_argument("--val-heatmap-dilation", type=int, default=30)
    parser.add_argument("--val-iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-detection-area", type=int, default=200,
                        help="Minimum connected-component area (pixels^2) to count as a detection. "
                             "Filters sub-lesion noise activations. At 912x1520 original resolution, "
                             "200 px^2 ≈ 14x14 pixels ≈ 2.3mm; 1%% GT box area = 759 px^2. "
                             "Default 200 keeps all real lesions while removing noise.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2)
    parser.add_argument("--medical-backbone-path", type=Path, default=None)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1,
                        help="LR multiplier for pretrained encoder. Decoder uses full --lr. "
                             "Prevents catastrophic forgetting of pretrained features.")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0,
                        help="Number of epochs to keep encoder frozen (requires_grad=False). "
                             "Default 0 = no freeze (differential LR handles encoder protection).")
    parser.add_argument("--tversky-alpha", type=float, default=0.3,
                        help="FP weight in Tversky loss. Lower value = less FP penalty.")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                        help="FN weight in Tversky loss. Higher value = more FN penalty = higher recall.")
    # Patch training
    parser.add_argument("--patch-mode", action="store_true",
                        help="Enable patch-based training DataLoader (PatchDataset). Each epoch builds "
                             "a balanced list of positive patches (centered on GT bboxes) and negative "
                             "patches (random crops from negative images). Validation uses full-image "
                             "inference unchanged. Recommended to fix pixel-level class imbalance.")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Side length of square training patches (pixels, in input_h×input_w space).")
    parser.add_argument("--patch-margin-factor", type=float, default=2.5,
                        help="When bbox is larger than patch_size: "
                             "crop_side = max(patch_size, max(bbox_w, bbox_h) × margin_factor).")
    parser.add_argument("--neg-patch-ratio", type=float, default=1.0,
                        help="Negative-to-positive patch count ratio per epoch. "
                             "1.0 = equal counts (recommended).")
    parser.add_argument("--neg-hard-ratio", type=float, default=0.5,
                        help="Fraction of negative patches that are hard negatives (crops from "
                             "positive-sample images that do NOT overlap GT bboxes). "
                             "0.0 = all easy negatives (rec_46 behavior); "
                             "0.5 = half hard (default, rec_46_upd_1); "
                             "1.0 = all hard negatives.")
    parser.add_argument("--monitor-metric", type=str, default="f1",
                        choices=["f1", "recall", "fbeta2"],
                        help="Metric to monitor for checkpoint saving and early stopping. "
                             "'f1' = best-threshold F1 (default, rec_46 behavior); "
                             "'recall' = Recall at fixed thresh=0.5; "
                             "'fbeta2' = F2 score (beta=2, Recall 4x Precision weight), "
                             "argmax over thresholds -- recommended for screening.")
    parser.add_argument("--box-nms-thresh", type=float, default=0.0,
                        help="IoU threshold for greedy NMS applied to predicted boxes after "
                             "connected-component extraction. 0.0 = disabled (default, preserves "
                             "previous behaviour). Typical values: 0.3–0.5. Merges overlapping "
                             "boxes from the same lesion region, reducing duplicate FP counts.")
    parser.add_argument("--hide-progress-bar", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    repo_root = repo_root_from_file()

    csv_path = args.csv_path or repo_root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = args.images_root or repo_root / "data" / "processed" / "images_png"
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.F.pth"

    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    all_samples = load_samples(csv_path, images_root, split_name="training")
    print(f"Total samples: {len(all_samples)}")

    train_idx, val_idx = patient_level_split(all_samples, val_ratio=0.15, seed=int(args.seed))
    val_pos_idx = [i for i in val_idx if all_samples[i].boxes.shape[0] > 0]

    n_train_pos = sum(1 for i in train_idx if all_samples[i].boxes.shape[0] > 0)
    n_train_neg = len(train_idx) - n_train_pos
    print(f"Train: {len(train_idx)} images (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val: {len(val_idx)} images | Val positive (used for detection eval): {len(val_pos_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input size: {args.input_h}×{args.input_w}")
    encoder_lr = float(args.lr) * float(args.encoder_lr_multiplier)
    decoder_lr = float(args.lr)
    print(f"Batch size: {args.batch_size} | Decoder LR: {decoder_lr} | Encoder LR: {encoder_lr} "
          f"(frozen for first {args.freeze_encoder_epochs} epochs) | Epochs: {args.epochs} | Patience: {args.patience}")
    if args.patch_mode:
        print(f"Patch mode: ON | Patch size: {args.patch_size}×{args.patch_size} | "
              f"Margin factor: {args.patch_margin_factor} | Neg:Pos ratio: {args.neg_patch_ratio} | "
              f"Neg hard ratio: {args.neg_hard_ratio}")
    else:
        print("Patch mode: OFF (full-image training)")
    print(f"Monitor metric: {args.monitor_metric}")

    model = build_unet(
        medical_backbone_path=str(args.medical_backbone_path) if args.medical_backbone_path else None,
    )
    model.to(device)
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params, "lr": decoder_lr},
        ],
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.01
    )

    if int(args.freeze_encoder_epochs) > 0:
        model.encoder.requires_grad_(False)
        print(f"[Freeze] Encoder frozen for first {args.freeze_encoder_epochs} epoch(s).")

    history: List[Dict[str, float]] = []
    best_metric = 0.0
    best_epoch = 0
    no_improve = 0

    monitor_metric_name = str(args.monitor_metric)
    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None) -> None:
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
        end_time = time.time()
        print(f"\n[Exit] Reason: {reason or 'normal'}")
        print(f"Best {monitor_metric_name}={best_metric:.4f} at epoch {best_epoch}")

    atexit.register(_on_exit, "atexit")

    def _sig_handler(signum: int, frame: Any) -> None:
        _on_exit(f"signal {signum}")
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except (OSError, ValueError):
            pass

    for epoch in range(int(args.epochs)):
        print(f"\n{'-' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'-' * 72}")

        if epoch == int(args.freeze_encoder_epochs) and int(args.freeze_encoder_epochs) > 0:
            model.encoder.requires_grad_(True)
            print(f"[Unfreeze] Encoder unfrozen at epoch {epoch + 1} with lr={encoder_lr:.2e}")

        if bool(args.patch_mode):
            train_dataset = PatchDataset(
                samples=all_samples,
                indices=train_idx,
                input_h=int(args.input_h),
                input_w=int(args.input_w),
                patch_size=int(args.patch_size),
                margin_factor=float(args.patch_margin_factor),
                neg_patch_ratio=float(args.neg_patch_ratio),
                neg_hard_ratio=float(args.neg_hard_ratio),
                augment=bool(args.augment),
                aug_hflip_prob=float(args.aug_hflip_prob),
                aug_brightness_delta=float(args.aug_brightness_delta),
                seed=int(args.seed),
                epoch=epoch,
            )
        else:
            train_dataset = SegDataset(
                samples=all_samples,
                indices=train_idx,
                input_h=int(args.input_h),
                input_w=int(args.input_w),
                augment=bool(args.augment),
                aug_hflip_prob=float(args.aug_hflip_prob),
                aug_brightness_delta=float(args.aug_brightness_delta),
                seed=int(args.seed),
                epoch=epoch,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=int(args.epochs),
            accumulation_steps=int(args.accumulation_steps),
            pos_weight=float(args.clf_pos_weight),
            bce_alpha=float(args.bce_alpha),
            tversky_alpha=float(args.tversky_alpha),
            tversky_beta=float(args.tversky_beta),
            disable_tqdm=bool(args.hide_progress_bar),
        )
        lr_scheduler.step()

        val_metrics = validate(
            model=model,
            samples=all_samples,
            val_indices=val_pos_idx,
            device=device,
            input_h=int(args.input_h),
            input_w=int(args.input_w),
            val_batch_size=int(args.val_batch_size),
            iou_threshold=float(args.val_iou_threshold),
            dilation_size=int(args.val_heatmap_dilation),
            min_component_area=int(args.min_detection_area),
            nms_iou_thresh=float(args.box_nms_thresh),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=bool(args.hide_progress_bar),
        )

        if monitor_metric_name == "recall":
            cur_monitor = float(val_metrics.get("best_recall", 0.0))
            monitor_thresh = float(val_metrics.get("best_recall_thresh", 0.5))
        elif monitor_metric_name == "fbeta2":
            cur_monitor = float(val_metrics.get("best_fbeta2", 0.0))
            monitor_thresh = float(val_metrics.get("best_fbeta2_thresh", 0.5))
        else:
            cur_monitor = float(val_metrics.get("best_thresh_f1", 0.0))
            monitor_thresh = float(val_metrics.get("best_f1_thresh", 0.5))

        row: Dict[str, float] = {"epoch": float(epoch + 1), "loss": float(avg_loss)}
        row.update(val_metrics)
        history.append(row)

        report_tp = int(val_metrics.get(f"tp@{monitor_thresh}", 0))
        report_fp = int(val_metrics.get(f"fp@{monitor_thresh}", 0))
        report_fn = int(val_metrics.get(f"fn@{monitor_thresh}", 0))
        report_recall = report_tp / max(report_tp + report_fn, 1)
        report_prec = report_tp / max(report_tp + report_fp, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"{monitor_metric_name}={cur_monitor:.4f} @ thresh={monitor_thresh} "
            f"(TP={report_tp} FP={report_fp} FN={report_fn} Recall={report_recall:.3f} Prec={report_prec:.3f}) | "
            f"lr={lr_scheduler.get_last_lr()[-1]:.6f}"
        )

        improved = cur_monitor > best_metric + float(args.min_delta)
        if improved:
            best_metric = cur_monitor
            no_improve = 0
            best_epoch = epoch + 1
            save_checkpoint(
                save_path=save_path,
                model=model,
                meta={
                    "epoch": epoch + 1,
                    monitor_metric_name: cur_monitor,
                    "monitor_thresh": monitor_thresh,
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                },
            )
            print(f"  [Checkpoint] Epoch {epoch + 1} | Saved ({monitor_metric_name}={best_metric:.4f}) -> {save_path}")
        else:
            no_improve += 1
            if int(args.patience) > 0 and no_improve >= int(args.patience):
                print(f"Early stopping triggered: no improvement for {no_improve} epochs.")
                break

    print(f"\nTraining complete. Best {monitor_metric_name}={best_metric:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {save_path}")
    _exit_state["reported"] = True


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start time:  {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"End time:    {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed:     {h:02d}:{m:02d}:{s:02d}")
