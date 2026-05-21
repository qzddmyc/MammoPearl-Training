r"""
方向 H：RetinaNet 全图直接检测（torchvision）

安装依赖（torchvision 已满足，无额外依赖）。

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-H.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 1024 \
    --input-w 512 \
    --patience 10 \
    --monitor-metric fbeta2_ref \
    --medical-backbone-path models/raw/ResNet50.pt \
    --save-path models/bbox_resnet50.H.pth \
    --augment \
    --aug-contrast-range 0.8 1.2 \
    --aug-scale-min 0.85 \
    --focal-alpha 0.25 \
    --lesion-types Mass \
    --hide-progress-bar

与方向 F（UNet）的核心差异：
  - 方向 F：热图分割 → sigmoid → 连通域 → NMS（间接检测）
    主要缺陷：heatmap 后处理产生大量 FP；召回率上限约 83%；F2≈0.47
  - 方向 H：RetinaNet + ResNet50-FPN 直接回归 box（直接检测）
    · Focal Loss 内置极端正负样本不平衡处理（γ=2.0，α=0.25），
      无需手动调 pos_weight/Tversky
    · FPN 提取 P3-P7 五层特征，同时覆盖 32-512px 尺度病灶
    · 直接输出 (box, score)，消除热图后处理管线产生的 FP 来源
    · 使用相同的 RadImageNet ResNet50 backbone 权重
    · 正样本图像过采样（--pos-oversample-factor=4.0）补偿 7% 正样本比例

─────────────────────────────────────────────────────────────────────────────
改版历史
─────────────────────────────────────────────────────────────────────────────

rec_48（初版）
  - RetinaNet + ResNet50-FPN，全图 1024×512 直接检测
  - anchor sizes=(32,64,128,256,512)，aspect_ratios=(0.5,1.0,2.0)

upd_1（rec_48 修复与日志改进）
  [fix] labels=torch.zeros（前景类索引 0）替换 torch.ones
        → 修复 num_classes=1 时 CUDA index-out-of-bounds 崩溃
  [fix] resnet_fpn_backbone(backbone_name=...) 改为关键字参数
        → 消除 torchvision ≥0.13 的 DeprecationWarning
  [fix] 验证日志 [BestRecall] 标签改为 Recall@0.3 (ref)
        → 原标签误导性（实为固定阈值参考值，并非最大 recall）
  [new] 新增 --monitor-metric fbeta2_ref 选项
        → 使用固定 score=0.3 处的 F2 作为 checkpoint/early-stop 标准
        → 比 fbeta2（best across all thresh，始终落在 score=0.1）噪声更低

upd_2（rec_48_upd_2 数据增强与宽高比修复）
  根因分析（rec_48_upd_1 F2@0.3 在 epoch 11 封顶于 0.2176 的原因）：
    1. 宽高比扭曲：原图 1520×912（AR=1.667）直接 resize 到 1024×512（AR=2.0），
       水平/垂直缩放比不等（0.561 vs 0.674），导致圆形病灶被拉伸 20%，
       模型学到扭曲特征，验证泛化能力受限。历史实验（rec_34–47）均未遇此问题，
       因旧框架用 torchvision 内部等比 resize。本次属新发现盲点。
    2. 数据增强过弱：仅水平翻转 + 亮度抖动，无对比度/缩放多样性，
       过拟合在 epoch 12 后显现（训练 loss 持续下降，验证 F2 不升）。
  [fix] AR-preserving pad：读图后先将 H 方向 pad 到 W×target_AR（加黑边），
        再等比 resize 到 1024×512 → 修复 20% AR 扭曲，作用于训练和验证。
        (VinDr-Mammo 固定 1520×912 → pad H to 1824 → resize 1024×512)
  [new] --aug-contrast-range MIN MAX：对比度抖动（乘性偏移，以图像均值为轴），
        默认 1.0 1.0（禁用，向后兼容）。rec_48_upd_2 推荐 0.8 1.2。
  [new] --aug-scale-min S：随机 zoom-out 增强（等比缩小至 S–1.0 倍后居中 pad），
        默认 1.0（禁用，向后兼容）。rec_48_upd_2 推荐 0.85。

upd_3（epoch-seeded WeightedRandomSampler）
  根因分析（rec_48_upd_1/upd_2 均在 epoch 7-8 出现断崖的原因）：
    WeightedRandomSampler 在训练前创建一次，各 epoch 顺序消耗 PyTorch 全局随机
    状态（seed=42 固定）。epoch 7-8 恰好抽到"困难批次"（小病灶/低对比度正样本
    密集），梯度冲突导致 loss 反升、验证 F2@0.3 断崖。两次训练的断崖完全同步，
    证明是确定性批次组成问题，而非模型本身的收敛问题。
  [fix] 每个 epoch 用独立 generator（manual_seed = base_seed + epoch）重建
        WeightedRandomSampler 和 DataLoader，打破跨 epoch 的批次确定性。
        各 epoch 的批次组成仍然完全可复现（同 seed 多次运行结果一致），
        但不再依赖前序 epoch 的随机状态消耗，消除了 epoch 7 固定谷底。
  [new] --focal-alpha F：Focal Loss 前景权重 α（torchvision 默认 0.25）。
        α=0.25 使背景梯度是正样本的 3 倍，模型学会保守预测（score 集中在
        0.1-0.2），导致 F2@0.3 受限。提高 α 可上移置信度分布。
        默认 0.25（向后兼容）。rec_48_upd_3 推荐 0.4。
  [new] checkpoint meta 新增 best_fbeta2 / best_fbeta2_thresh / focal_alpha：
        训练时遍历阈值得到的最优 F2 及对应阈值一并保存进 .pth。
        部署时直接读取，无需手动调参：
          ckpt = torch.load("models/bbox_resnet50.H.pth")
          deploy_thresh = ckpt["meta"]["best_fbeta2_thresh"]
          # → 在此阈值处 ckpt["meta"]["best_fbeta2"] 最大

upd_4（纯 Mass 检测 baseline）
  根因分析（rec_48_upd_3 Recall@0.1=0.404 封顶的原因）：
    验证集 267 GT boxes 中，仅 Mass（56.6%）具备清晰可检测的视觉特征。
    其余类型存在结构性漏检问题：
      · Suspicious_Calcification（20.2%）：约 30% AR>2，anchor fg_iou<0.5 被
        标为 ignored；约 20% size<32px 完全超出 anchor 覆盖；其余训练信号不稳定。
      · Asymmetry / Architectural_Distortion（12.8%）：需双侧对比才能识别，
        单视图特征图无法与背景区分，即使 anchor 覆盖也无学习信号。
    理论上限 Recall@0.1 ≈ 55-65%（非 100%），当前 40.4% 已逼近该上限。
  [new] --lesion-types TYPES：逗号分隔的病灶类型名称，只将指定类型的 box 作为
        正样本 GT，其余 box 被忽略（图像若所有 box 均被过滤则视为负样本）。
        默认 None（使用全部 box，向后兼容）。upd_4 推荐 Mass。
  [revert] focal_alpha 从 0.4 退回 0.25：upd_3 的 α=0.4 使 Recall@0.1
        从 0.502 降至 0.404（-20%），虽然 TP@0.7 提升，但 upd_4 目标是
        建立 Mass 检测的 recall 上限，优先覆盖率。upd_4 推荐 α=0.25。
"""

from __future__ import annotations

import os
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import atexit
import datetime
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    _TORCHVISION_DET_OK = True
except ImportError:
    _TORCHVISION_DET_OK = False

try:
    from torchvision.models import resnet50 as _resnet50
    try:
        from torchvision.models import ResNet50_Weights as _ResNet50Weights
        _IMAGENET_WEIGHTS: Any = _ResNet50Weights.DEFAULT
    except ImportError:
        _ResNet50Weights = None  # type: ignore[assignment]
        _IMAGENET_WEIGHTS = True  # pretrained=True fallback
except ImportError:
    _resnet50 = None  # type: ignore[assignment]
    _IMAGENET_WEIGHTS = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


# =============================================================================
# Utilities
# =============================================================================

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
    """Convert image to RGB uint8 with CLAHE for grayscale."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
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
        px1, py1, px2, py2 = float(pred_boxes[pi, 0]), float(pred_boxes[pi, 1]), \
                              float(pred_boxes[pi, 2]), float(pred_boxes[pi, 3])
        for gi in range(gt_boxes.shape[0]):
            if matched_gt[gi]:
                continue
            gx1, gy1, gx2, gy2 = float(gt_boxes[gi, 0]), float(gt_boxes[gi, 1]), \
                                  float(gt_boxes[gi, 2]), float(gt_boxes[gi, 3])
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


# =============================================================================
# Data structures
# =============================================================================

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
    lesion_types: Optional[List[str]] = None,
) -> List[Sample]:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    samples: List[Sample] = []
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        if lesion_types:
            type_mask = pd.Series(False, index=group.index)
            for lt in lesion_types:
                if lt in group.columns:
                    type_mask = type_mask | (group[lt] == 1)
            group = group[type_mask]
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
    patients = sorted({s.patient_id for s in samples})  # sorted → deterministic order
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    train_idx = [i for i, s in enumerate(samples) if s.patient_id not in val_patients]
    val_idx = [i for i, s in enumerate(samples) if s.patient_id in val_patients]
    return train_idx, val_idx


# =============================================================================
# Dataset
# =============================================================================

class DetectionDataset(Dataset):
    """Full-image detection dataset for torchvision RetinaNet.

    Each item is (image_tensor [3, H, W], target_dict) where target_dict
    contains 'boxes' [N, 4] (xyxy) and 'labels' [N] (all 1 = lesion).
    Negative images return empty boxes/labels.
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
        aug_contrast_min: float = 1.0,
        aug_contrast_max: float = 1.0,
        aug_scale_min: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_brightness_delta = aug_brightness_delta
        self.aug_contrast_min = aug_contrast_min
        self.aug_contrast_max = aug_contrast_max
        self.aug_scale_min = aug_scale_min
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[self.indices[idx]]

        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            target: Dict[str, torch.Tensor] = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            return img_t, target

        orig_h, orig_w = img.shape[:2]
        boxes = sample.boxes.copy()

        # Clip and filter degenerate boxes
        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # AR-preserving pad: make H/W match target AR before resize to eliminate distortion.
        # VinDr-Mammo images are 1520×912 (AR=1.667); target 1024×512 (AR=2.0) → pad H.
        target_ar = self.input_h / max(self.input_w, 1)
        actual_ar = orig_h / max(orig_w, 1)
        if actual_ar < target_ar - 1e-6:  # image too wide → pad height
            padded_h = int(round(orig_w * target_ar))
            pad_top = (padded_h - orig_h) // 2
            pad_bottom = padded_h - orig_h - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 1] += pad_top
                boxes[:, 3] += pad_top
            orig_h = padded_h
        elif actual_ar > target_ar + 1e-6:  # image too tall → pad width
            padded_w = int(round(orig_h / target_ar))
            pad_left = (padded_w - orig_w) // 2
            pad_right = padded_w - orig_w - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 0] += pad_left
                boxes[:, 2] += pad_left
            orig_w = padded_w

        # Resize image
        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Scale boxes to input_h×input_w space
        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        if boxes.size > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            # Re-filter after scaling
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # Augmentation
        if self.augment:
            if self.rng.random() < self.aug_hflip_prob:
                img_resized = img_resized[:, ::-1, :].copy()
                if boxes.size > 0:
                    old_x1 = boxes[:, 0].copy()
                    old_x2 = boxes[:, 2].copy()
                    boxes[:, 0] = self.input_w - old_x2
                    boxes[:, 2] = self.input_w - old_x1
            if self.aug_brightness_delta > 0:
                delta = (self.rng.random() * 2 - 1) * self.aug_brightness_delta * 255
                img_resized = np.clip(img_resized.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            if self.aug_contrast_max > self.aug_contrast_min + 1e-6:
                factor = self.aug_contrast_min + self.rng.random() * (self.aug_contrast_max - self.aug_contrast_min)
                mean_val = float(img_resized.mean())
                img_resized = np.clip(
                    mean_val + (img_resized.astype(np.float32) - mean_val) * factor, 0, 255
                ).astype(np.uint8)
            if self.aug_scale_min < 1.0 - 1e-6:
                scale = self.aug_scale_min + self.rng.random() * (1.0 - self.aug_scale_min)
                if scale < 1.0 - 1e-6:
                    scaled_h = max(int(self.input_h * scale), 1)
                    scaled_w = max(int(self.input_w * scale), 1)
                    img_small = cv2.resize(img_resized, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    pad_top_zo = (self.input_h - scaled_h) // 2
                    pad_left_zo = (self.input_w - scaled_w) // 2
                    img_zoomed = np.zeros_like(img_resized)
                    img_zoomed[pad_top_zo:pad_top_zo + scaled_h, pad_left_zo:pad_left_zo + scaled_w] = img_small
                    img_resized = img_zoomed
                    if boxes.size > 0:
                        boxes[:, 0] = boxes[:, 0] * scale + pad_left_zo
                        boxes[:, 2] = boxes[:, 2] * scale + pad_left_zo
                        boxes[:, 1] = boxes[:, 1] * scale + pad_top_zo
                        boxes[:, 3] = boxes[:, 3] * scale + pad_top_zo
                        keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                        boxes = boxes[keep]

        img_t = image_to_tensor(img_resized)  # [3, H, W] float32 in [0, 1]

        if boxes.size > 0:
            target = {
                "boxes": torch.from_numpy(boxes.astype(np.float32)),
                "labels": torch.zeros(boxes.shape[0], dtype=torch.int64),  # class 0 = foreground (num_classes=1)
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        return img_t, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def make_oversampling_weights(
    samples: List[Sample],
    indices: List[int],
    pos_oversample_factor: float = 4.0,
) -> torch.Tensor:
    """Assign higher weight to positive-sample images for WeightedRandomSampler."""
    weights = []
    for i in indices:
        weights.append(pos_oversample_factor if samples[i].boxes.shape[0] > 0 else 1.0)
    return torch.DoubleTensor(weights)


# =============================================================================
# Model
# =============================================================================

def build_retinanet(
    medical_backbone_path: Optional[str] = None,
    num_classes: int = 1,
    anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 5,
    min_size: int = 512,
    max_size: int = 1024,
    trainable_backbone_layers: int = 5,
    nms_thresh: float = 0.3,
    score_thresh: float = 0.05,
    detections_per_img: int = 500,
    focal_alpha: float = 0.25,
) -> "RetinaNet":
    """Build a RetinaNet with ResNet50-FPN backbone.

    Loads ImageNet weights first, then overrides with RadImageNet if provided.
    conv1 weights are averaged across channels for grayscale-as-3channel input.
    """
    if not _TORCHVISION_DET_OK:
        raise RuntimeError(
            "torchvision detection module not available. "
            "Requires torchvision >= 0.11 with detection support."
        )

    # Build ResNet50-FPN backbone
    try:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=_IMAGENET_WEIGHTS,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone.")
    except TypeError:
        # Older torchvision API
        backbone = resnet_fpn_backbone(  # type: ignore[call-arg]
            backbone_name="resnet50",
            pretrained=True,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone (legacy API).")

    # Override with RadImageNet if provided
    if medical_backbone_path is not None:
        _idx_to_resnet = {
            "0": "conv1", "1": "bn1",
            "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
        }
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            raw_sd = (
                ckpt.get("state_dict", ckpt.get("model", ckpt))
                if isinstance(ckpt, dict)
                else ckpt
            )
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            result = backbone.body.load_state_dict(stripped, strict=False)
            missing = len(result.missing_keys) if result is not None else "?"
            unexpected = len(result.unexpected_keys) if result is not None else "?"
            print(f"[Info] Loaded RadImageNet backbone (missing={missing}, unexpected={unexpected}).")
        except Exception as exc:
            print(f"[Warning] Could not load RadImageNet backbone ({exc}). Keeping ImageNet weights.")

    # Adapt conv1 for grayscale-as-3channel (average over input channels)
    try:
        with torch.no_grad():
            mean_w = backbone.body.conv1.weight.mean(dim=1, keepdim=True)
            backbone.body.conv1.weight.copy_(mean_w.expand_as(backbone.body.conv1.weight))
        print("[Info] conv1 weights averaged for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1.")

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )

    # Override focal loss alpha (torchvision default: 0.25)
    try:
        model.head.classification_head.focal_loss_alpha = float(focal_alpha)
        print(f"[Info] Focal loss alpha set to {focal_alpha}.")
    except AttributeError:
        print(f"[Warning] Could not set focal_loss_alpha on this torchvision version.")

    return model


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: "RetinaNet",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int = 1,
    disable_tqdm: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())  # type: ignore[arg-type]

        if not torch.isfinite(losses):
            optimizer.zero_grad(set_to_none=True)
            continue

        (losses / float(accumulation_steps)).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(losses.item())
        count += 1
        if not disable_tqdm:
            pbar.set_postfix(loss=f"{losses.item():.4f}")  # type: ignore[union-attr]

    return running_loss / max(count, 1)


# =============================================================================
# Validation
# =============================================================================

def validate(
    model: "RetinaNet",
    samples: List[Sample],
    val_indices: List[int],
    device: torch.device,
    input_h: int,
    input_w: int,
    iou_threshold: float,
    collection_score_thresh: float = 0.01,
    nms_thresh: float = 0.3,
    epoch: int = 0,
    epochs: int = 1,
    disable_tqdm: bool = False,
) -> Dict[str, float]:
    """Validate with full-image RetinaNet inference at multiple score thresholds."""
    model.eval()

    # Temporarily lower score_thresh to collect all candidate boxes
    orig_score_thresh = model.score_thresh
    orig_nms_thresh = model.nms_thresh
    orig_det_per_img = model.detections_per_img
    model.score_thresh = collection_score_thresh
    model.nms_thresh = nms_thresh
    model.detections_per_img = 1000

    score_thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    stats: Dict[float, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in score_thresholds
    }
    total_gt_boxes = 0

    pbar = tqdm(val_indices, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for sample_idx in pbar:
            sample = samples[sample_idx]
            try:
                img = normalize_image(read_image_unicode(sample.image_path))
            except FileNotFoundError:
                continue

            orig_h, orig_w = img.shape[:2]
            gt_boxes = sample.boxes.astype(np.float32).copy()

            # AR-preserving pad (must match training preprocessing)
            target_ar = input_h / max(input_w, 1)
            actual_ar = orig_h / max(orig_w, 1)
            if actual_ar < target_ar - 1e-6:  # pad height
                padded_h = int(round(orig_w * target_ar))
                pad_top = (padded_h - orig_h) // 2
                pad_bottom = padded_h - orig_h - pad_top
                img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
                if gt_boxes.size > 0:
                    gt_boxes[:, 1] += pad_top
                    gt_boxes[:, 3] += pad_top
                orig_h = padded_h
            elif actual_ar > target_ar + 1e-6:  # pad width
                padded_w = int(round(orig_h / target_ar))
                pad_left = (padded_w - orig_w) // 2
                pad_right = padded_w - orig_w - pad_left
                img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
                if gt_boxes.size > 0:
                    gt_boxes[:, 0] += pad_left
                    gt_boxes[:, 2] += pad_left
                orig_w = padded_w

            # Scale GT boxes to input_h×input_w coordinate space
            scale_x = input_w / max(orig_w, 1)
            scale_y = input_h / max(orig_h, 1)
            if gt_boxes.size > 0:
                gt_boxes[:, 0] *= scale_x
                gt_boxes[:, 2] *= scale_x
                gt_boxes[:, 1] *= scale_y
                gt_boxes[:, 3] *= scale_y
                keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
                gt_boxes = gt_boxes[keep]

            img_resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            img_t = image_to_tensor(img_resized).to(device)

            outputs = model([img_t])  # List[Dict] with 'boxes', 'scores', 'labels'
            pred_boxes = outputs[0]["boxes"].cpu().numpy()    # (K, 4) xyxy
            pred_scores = outputs[0]["scores"].cpu().numpy()  # (K,)

            for thresh in score_thresholds:
                mask = pred_scores >= thresh
                filtered_boxes = pred_boxes[mask]
                tp, fp, fn = compute_iou_matches(filtered_boxes, gt_boxes, iou_threshold)
                stats[thresh]["tp"] += tp
                stats[thresh]["fp"] += fp
                stats[thresh]["fn"] += fn

            total_gt_boxes += int(gt_boxes.shape[0]) if gt_boxes.size > 0 else 0

    # Restore model thresholds
    model.score_thresh = orig_score_thresh
    model.nms_thresh = orig_nms_thresh
    model.detections_per_img = orig_det_per_img

    # Compute metrics
    f1_per_thresh: Dict[float, float] = {}
    recall_per_thresh: Dict[float, float] = {}
    fbeta2_per_thresh: Dict[float, float] = {}
    for thresh in score_thresholds:
        tp = stats[thresh]["tp"]
        fp = stats[thresh]["fp"]
        fn = stats[thresh]["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fbeta2 = (1 + 4) * prec * rec / max(4 * prec + rec, 1e-9)
        f1_per_thresh[thresh] = f1
        recall_per_thresh[thresh] = rec
        fbeta2_per_thresh[thresh] = fbeta2

    best_f1_thresh = max(f1_per_thresh, key=lambda t: f1_per_thresh[t])
    best_f1 = f1_per_thresh[best_f1_thresh]
    best_recall_thresh: float = 0.3   # fixed reference threshold
    best_recall = recall_per_thresh[best_recall_thresh]
    best_fbeta2_thresh = max(fbeta2_per_thresh, key=lambda t: fbeta2_per_thresh[t])
    best_fbeta2 = fbeta2_per_thresh[best_fbeta2_thresh]
    ref_fbeta2_thresh: float = 0.3    # fixed reference threshold for cross-epoch comparison
    ref_fbeta2 = fbeta2_per_thresh[ref_fbeta2_thresh]

    parts = []
    for thresh in score_thresholds:
        tp = stats[thresh]["tp"]
        fp = stats[thresh]["fp"]
        fn = stats[thresh]["fn"]
        parts.append(
            f"@{thresh}: TP={tp} FP={fp} FN={fn} "
            f"Rec={recall_per_thresh[thresh]:.3f} F1={f1_per_thresh[thresh]:.4f} F2={fbeta2_per_thresh[thresh]:.4f}"
        )
    print(f"  [Val] GT_boxes={total_gt_boxes} | {' | '.join(parts)}")
    print(
        f"  [BestF1] F1={best_f1:.4f} @ score={best_f1_thresh} | "
        f"Recall@{best_recall_thresh}={best_recall:.4f} (ref) | "
        f"[BestFbeta2] F2={best_fbeta2:.4f} @ score={best_fbeta2_thresh} | "
        f"F2@{ref_fbeta2_thresh}={ref_fbeta2:.4f} (ref)"
    )

    result: Dict[str, float] = {
        "best_f1": float(best_f1),
        "best_f1_thresh": float(best_f1_thresh),
        "best_recall": float(best_recall),
        "best_recall_thresh": float(best_recall_thresh),
        "best_fbeta2": float(best_fbeta2),
        "best_fbeta2_thresh": float(best_fbeta2_thresh),
        "ref_fbeta2": float(ref_fbeta2),
        "ref_fbeta2_thresh": float(ref_fbeta2_thresh),
        "val_gt_boxes": float(total_gt_boxes),
    }
    for thresh in score_thresholds:
        result[f"tp@{thresh}"] = float(stats[thresh]["tp"])
        result[f"fp@{thresh}"] = float(stats[thresh]["fp"])
        result[f"fn@{thresh}"] = float(stats[thresh]["fn"])
    return result


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    save_path: Path,
    model: "RetinaNet",
    meta: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, save_path)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RetinaNet-ResNet50-FPN detection model for VinDr lesion detection (Direction H)."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size. Default 4 for full 1024×512 images on a 24GB GPU.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1,
                        help="LR multiplier for pretrained ResNet50 body. "
                             "FPN + head use full --lr.")
    parser.add_argument("--input-h", type=int, default=1024,
                        help="Resize height for model input.")
    parser.add_argument("--input-w", type=int, default=512,
                        help="Resize width for model input.")
    parser.add_argument("--val-iou-threshold", type=float, default=0.1,
                        help="IoU threshold for matching predicted boxes to GT during validation.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps. Use 2–4 if GPU memory is tight.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training augmentation (hflip, brightness jitter).")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2)
    parser.add_argument("--aug-contrast-range", type=float, nargs=2, default=[1.0, 1.0],
                        metavar=("MIN", "MAX"),
                        help="Contrast jitter range (multiplicative factor around image mean). "
                             "1.0 1.0 = disabled (default). Recommended: 0.8 1.2.")
    parser.add_argument("--aug-scale-min", type=float, default=1.0,
                        help="Minimum zoom-out scale for random scale augmentation. "
                             "1.0 = disabled (default). Recommended: 0.85.")
    parser.add_argument("--medical-backbone-path", type=Path, default=None,
                        help="Path to RadImageNet ResNet50 checkpoint (.pt). "
                             "If None, uses ImageNet weights only.")
    parser.add_argument("--pos-oversample-factor", type=float, default=4.0,
                        help="Weight multiplier for positive (lesion) images in the training sampler. "
                             "4.0 → positive images sampled ~4× more than negatives. "
                             "Compensates for the 7%% natural positive image ratio.")
    parser.add_argument("--anchor-sizes", type=str, default="32,64,128,256,512",
                        help="Comma-separated anchor sizes (one per FPN level). "
                             "Default: 32,64,128,256,512.")
    parser.add_argument("--nms-thresh", type=float, default=0.3,
                        help="NMS IoU threshold applied during inference. Default 0.3.")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Minimum score threshold for reported detections. Default 0.05.")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal Loss foreground weight alpha. Higher values increase "
                             "positive-sample gradient weight, shifting score distribution "
                             "upward. torchvision default 0.25; try 0.4 to reduce "
                             "confidence suppression. Default 0.25 (backward-compatible).")
    parser.add_argument("--monitor-metric", type=str, default="fbeta2",
                        choices=["f1", "recall", "fbeta2", "fbeta2_ref"],
                        help="Metric to monitor for checkpoint saving and early stopping. "
                             "fbeta2_ref uses F2 at the fixed reference threshold (0.3), "
                             "which is more stable than fbeta2 (best across all thresholds).")
    parser.add_argument("--hide-progress-bar", action="store_true")
    parser.add_argument("--lesion-types", type=str, default=None,
                        help="Comma-separated lesion type names to keep as positive GT boxes "
                             "(e.g. 'Mass' or 'Mass,Focal_Asymmetry'). Images whose boxes are "
                             "all filtered out become negative samples. Default: None (use all "
                             "annotated boxes, backward compatible).")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    repo_root = repo_root_from_file()
    csv_path = args.csv_path or repo_root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = args.images_root or repo_root / "data" / "processed" / "images_png"
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.H.pth"

    print(f"Start time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    all_samples = load_samples(csv_path, images_root, split_name="training",
                               lesion_types=[t.strip() for t in args.lesion_types.split(",")]
                               if args.lesion_types else None)
    print(f"Total samples: {len(all_samples)}")
    if args.lesion_types:
        print(f"Lesion type filter: {args.lesion_types}")

    train_idx, val_idx = patient_level_split(all_samples, val_ratio=0.15, seed=int(args.seed))
    val_pos_idx = [i for i in val_idx if all_samples[i].boxes.shape[0] > 0]

    n_train_pos = sum(1 for i in train_idx if all_samples[i].boxes.shape[0] > 0)
    n_train_neg = len(train_idx) - n_train_pos
    val_gt_total = sum(all_samples[i].boxes.shape[0] for i in val_pos_idx)
    print(f"Train: {len(train_idx)} images (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val: {len(val_idx)} images | Val positive images: {len(val_pos_idx)} | Val GT boxes: {val_gt_total}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input size: {args.input_h}×{args.input_w}")

    encoder_lr = float(args.lr) * float(args.encoder_lr_multiplier)
    head_lr = float(args.lr)
    print(
        f"Encoder LR: {encoder_lr:.2e} | Head/FPN LR: {head_lr:.2e} | "
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | Patience: {args.patience}"
    )
    print(f"Pos oversample factor: {args.pos_oversample_factor:.1f}")
    print(f"Monitor metric: {args.monitor_metric}")

    # Parse anchor sizes
    anchor_size_vals = [int(s.strip()) for s in str(args.anchor_sizes).split(",")]
    anchor_sizes = tuple((s,) for s in anchor_size_vals)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size_vals)
    print(f"Anchor sizes: {anchor_sizes} | Aspect ratios: (0.5, 1.0, 2.0) per level")

    # Build model
    model = build_retinanet(
        medical_backbone_path=(
            str(args.medical_backbone_path) if args.medical_backbone_path else None
        ),
        num_classes=1,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        min_size=int(args.input_w),   # min dim = width
        max_size=int(args.input_h),   # max dim = height
        trainable_backbone_layers=5,
        nms_thresh=float(args.nms_thresh),
        score_thresh=float(args.score_thresh),
        detections_per_img=500,
        focal_alpha=float(args.focal_alpha),
    )
    model.to(device)

    # Differential LR: encoder body vs FPN + head
    body_param_ids = {id(p) for p in model.backbone.body.parameters()}
    body_params = list(model.backbone.body.parameters())
    other_params = [p for p in model.parameters() if id(p) not in body_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": encoder_lr},
            {"params": other_params, "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.01
    )

    # Build training dataset (constant across epochs, augmentation is stochastic)
    train_dataset = DetectionDataset(
        samples=all_samples,
        indices=train_idx,
        input_h=int(args.input_h),
        input_w=int(args.input_w),
        augment=bool(args.augment),
        aug_hflip_prob=float(args.aug_hflip_prob),
        aug_brightness_delta=float(args.aug_brightness_delta),
        aug_contrast_min=float(args.aug_contrast_range[0]),
        aug_contrast_max=float(args.aug_contrast_range[1]),
        aug_scale_min=float(args.aug_scale_min),
        seed=int(args.seed),
    )

    # Weighted sampler: oversample positive images
    sampler_weights = make_oversampling_weights(
        all_samples, train_idx, pos_oversample_factor=float(args.pos_oversample_factor)
    )
    # train_sampler and train_loader are rebuilt each epoch inside the loop
    # (epoch-seeded generator) to break cross-epoch batch determinism.

    # Training state
    best_metric = 0.0
    best_epoch = 0
    no_improve = 0
    monitor_metric_name = str(args.monitor_metric)

    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None) -> None:
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
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

    # Training loop
    for epoch in range(int(args.epochs)):
        print(f"\n{'─' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'─' * 72}")

        # Rebuild sampler with an epoch-specific seed so each epoch draws an
        # independent batch sequence (still reproducible: same seed → same run).
        _epoch_gen = torch.Generator()
        _epoch_gen.manual_seed(int(args.seed) + epoch)
        train_sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_idx),
            replacement=True,
            generator=_epoch_gen,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            sampler=train_sampler,
            num_workers=int(args.num_workers),
            collate_fn=detection_collate_fn,
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
            iou_threshold=float(args.val_iou_threshold),
            collection_score_thresh=0.01,
            nms_thresh=float(args.nms_thresh),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=bool(args.hide_progress_bar),
        )

        if monitor_metric_name == "recall":
            cur_monitor = float(val_metrics.get("best_recall", 0.0))
            monitor_thresh = float(val_metrics.get("best_recall_thresh", 0.3))
        elif monitor_metric_name == "fbeta2":
            cur_monitor = float(val_metrics.get("best_fbeta2", 0.0))
            monitor_thresh = float(val_metrics.get("best_fbeta2_thresh", 0.3))
        elif monitor_metric_name == "fbeta2_ref":
            cur_monitor = float(val_metrics.get("ref_fbeta2", 0.0))
            monitor_thresh = float(val_metrics.get("ref_fbeta2_thresh", 0.3))
        else:
            cur_monitor = float(val_metrics.get("best_f1", 0.0))
            monitor_thresh = float(val_metrics.get("best_f1_thresh", 0.3))

        report_tp = int(val_metrics.get(f"tp@{monitor_thresh}", 0))
        report_fp = int(val_metrics.get(f"fp@{monitor_thresh}", 0))
        report_fn = int(val_metrics.get(f"fn@{monitor_thresh}", 0))
        report_recall = report_tp / max(report_tp + report_fn, 1)
        report_prec = report_tp / max(report_tp + report_fp, 1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"{monitor_metric_name}={cur_monitor:.4f} @ score={monitor_thresh:.1f} "
            f"(TP={report_tp} FP={report_fp} FN={report_fn} "
            f"Recall={report_recall:.3f} Prec={report_prec:.3f}) | "
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
                    # Also save the sweep-optimal F2 threshold for deployment use.
                    # When monitor_metric is fbeta2_ref (fixed 0.3), monitor_thresh=0.3
                    # but best_fbeta2_thresh gives the threshold that maximises F2.
                    "best_fbeta2": float(val_metrics.get("best_fbeta2", 0.0)),
                    "best_fbeta2_thresh": float(val_metrics.get("best_fbeta2_thresh", 0.3)),
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                    "anchor_sizes": str(args.anchor_sizes),
                    "nms_thresh": float(args.nms_thresh),
                    "focal_alpha": float(args.focal_alpha),
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
    start = time.time()
    main()
    end = time.time()
    h, rem = divmod(int(end - start), 3600)
    m, s = divmod(rem, 60)
    print(f"End time:    {datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed:     {h:02d}:{m:02d}:{s:02d}")
