r"""
方向 K：RetinaNet + GlobalContextEncoder（全图上下文融合，无 ROI 重打分）

安装依赖（torchvision 已满足，无额外依赖）。

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-K.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 1024 \
    --input-w 512 \
    --patience 10 \
    --monitor-metric fbeta2_ref \
    --medical-backbone-path models/raw/ResNet50.pt \
    --save-path models/bbox_resnet50.K.pth \
    --augment \
    --aug-contrast-range 0.8 1.2 \
    --aug-scale-min 0.85 \
    --focal-alpha 0.25 \
    --min-box-side 24.0 \
    --max-box-ar 3.0 \
    --cliff-patience-ratio 0.6 \
    --global-channels 128 \
    --hide-progress-bar

与方向 H 和方向 J 的核心差异：

  方向 H（基线）：
    纯 RetinaNet + ResNet50-FPN，全图直接检测，best F2@0.3=0.2788（ep9）。

  方向 J（已失败）：
    H + GlobalContextEncoder/GlobalAwareBackbone + ROI 重打分头。
    best F2@0.3=0.1571（ep23）。失败根因分析：
      · ROI 头每批正例极少，BCE 训练信号极不稳定，导致 roi_score 逐 epoch 大幅抖动。
      · 几何均值 √(det×roi) 过激进，roi_score 低时把大量候选压至 0.3 以下。
    无法判断失败原因是"全局融合无效"还是"ROI 头破坏了 det_score"。

  方向 K（本文件）：
    H + GlobalContextEncoder/GlobalAwareBackbone（单变量验证全局融合效果）。
    完全移除 ROI 重打分头，推理分数直接使用 det_score（同 H）。
    实验目的：分离"全局上下文融合"对 F2@0.3 的独立贡献。
    · K > H → 全局融合有效，J 的失败主因是 ROI 头。
    · K ≈ H → 全局融合本身也无效，需要换思路。

GlobalContextEncoder：
  输入全图 4× 下采样（256×128 for 1024×512），3 个 stride-2 BN-ReLU Conv 模块
  将空间压缩 8× → 输出 [B, 128, 32, 16]（与 FPN P5 同分辨率）。
  F.interpolate 上/下采样到各 FPN 层分辨率后 concat 再 1×1 融合。

GlobalAwareBackbone：
  包装 resnet_fpn_backbone，对 P3~P7 五层各注入全局上下文。
  1×1 融合 conv 以 identity init（全局分支初始贡献为零），
  保留预训练 RetinaNet 的起始行为，避免随机初始化破坏 early training。

─────────────────────────────────────────────────────────────────────────────
改版历史
─────────────────────────────────────────────────────────────────────────────

rec_51（初版）
  - 以方向 H（upd_6）为基础，加入 GlobalContextEncoder + GlobalAwareBackbone。
  - 移除方向 J 的 RoiRefinementHead 及所有 ROI loss 辅助函数。
  - 训练和验证中每批/每图调用 set_global_image（4× 下采样）。
  - 差分 LR：model.backbone.base.body（ResNet50，低 LR = lr × encoder_lr_multiplier）
             vs FPN + 检测头 + global_enc + fusion convs（高 LR = lr）。
  - 新增 --global-channels 参数（默认 128）。
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
import torch.nn as nn
import torch.nn.functional as F
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
    min_box_side: float = 0.0,
    max_box_ar: float = float("inf"),
    input_w: int = 512,
) -> List[Sample]:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    samples: List[Sample] = []
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        first = group.iloc[0]  # read orig_h/w before filtering
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

        # Detectability filter (in resized-image-space coordinates)
        if boxes.size > 0 and (min_box_side > 0.0 or max_box_ar < float("inf")):
            orig_w_val = float(first["width"]) if pd.notna(first["width"]) else float(input_w)
            scale = float(input_w) / max(orig_w_val, 1.0)
            bw = (boxes[:, 2] - boxes[:, 0]) * scale
            bh = (boxes[:, 3] - boxes[:, 1]) * scale
            min_sides = np.minimum(bw, bh)
            ars = np.maximum(bw, bh) / np.maximum(min_sides, 1e-3)
            keep = np.ones(len(boxes), dtype=bool)
            if min_box_side > 0.0:
                keep &= (min_sides >= min_box_side)
            if max_box_ar < float("inf"):
                keep &= (ars <= max_box_ar)
            boxes = boxes[keep]

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
# Global Context Modules
# =============================================================================

class GlobalContextEncoder(nn.Module):
    """Lightweight CNN that encodes full-breast context from a downsampled image.

    Input:  [B, 3, H/4, W/4]  (default: [B, 3, 256, 128] for 1024×512 full image)
    Output: [B, out_ch, H/32, W/32]  (default: [B, 128, 32, 16])

    Three stride-2 conv blocks reduce spatial resolution by 8× total.
    The output spatial size matches FPN P5 when the full image is 1024×512.
    F.interpolate is used to resize this context map to match each FPN level.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # 256×128 → 128×64
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 128×64 → 64×32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64×32 → 32×16
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GlobalAwareBackbone(nn.Module):
    """Wraps a BackboneWithFPN, injecting global context into every FPN level.

    Usage:
        # Before each forward pass (training or inference):
        model.backbone.set_global_image(small_img)   # [B, 3, H/4, W/4]
        # Then call model normally — backbone.forward() will inject global features.

    Fusion initialization: the 1×1 fusion convs are initialized so the global
    branch contribution starts near zero (identity pass-through for local features).
    This preserves the pretrained RetinaNet behaviour at the start of training.
    """

    def __init__(
        self,
        base_backbone: Any,
        global_encoder: GlobalContextEncoder,
        num_fpn_levels: int = 5,
        fpn_channels: int = 256,
        global_channels: int = 128,
    ) -> None:
        super().__init__()
        self.base = base_backbone
        self.global_enc = global_encoder
        self.out_channels: int = fpn_channels

        # Fusion convs: concat([local(256), global(128)]) → 256
        self.fusion = nn.ModuleList([
            nn.Conv2d(fpn_channels + global_channels, fpn_channels, kernel_size=1, bias=True)
            for _ in range(num_fpn_levels)
        ])
        # Identity init: global path starts at zero contribution
        for fc in self.fusion:
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
            # Restore local identity: first fpn_channels input channels → identity output
            with torch.no_grad():
                fc.weight[:, :fpn_channels, 0, 0] = torch.eye(fpn_channels)

        self._global_feat: Optional[torch.Tensor] = None

    def set_global_image(self, img_tensor: torch.Tensor) -> None:
        """Pre-compute global context feature from downsampled image.
        Must be called before each forward() call (training and inference).
        """
        self._global_feat = self.global_enc(img_tensor)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = self.base(x)

        if self._global_feat is not None:
            keys = list(feats.keys())
            for i, k in enumerate(keys):
                if i < len(self.fusion):
                    gf = F.interpolate(
                        self._global_feat,
                        size=feats[k].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    feats[k] = self.fusion[i](torch.cat([feats[k], gf], dim=1))

        return feats


# =============================================================================
# Model
# =============================================================================

def build_global_retinanet(
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
    global_channels: int = 128,
) -> "RetinaNet":
    """Build a RetinaNet with ResNet50-FPN backbone + GlobalContextEncoder fusion.

    Architecture:
      ResNet50-FPN (base) → P3~P7 (5 FPN levels, 256ch each)
      GlobalContextEncoder (input: 4× downsampled full image) → [B, global_channels, 32, 16]
      Each FPN level: F.interpolate(global_feat) concat local → 1×1 fusion → fused features
      RetinaNet detection head → boxes + scores (det_score only, no ROI reranking)
    """
    if not _TORCHVISION_DET_OK:
        raise RuntimeError(
            "torchvision detection module not available. "
            "Requires torchvision >= 0.11 with detection support."
        )

    # Build ResNet50-FPN backbone
    try:
        base_backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=_IMAGENET_WEIGHTS,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone.")
    except TypeError:
        # Older torchvision API
        base_backbone = resnet_fpn_backbone(  # type: ignore[call-arg]
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
            result = base_backbone.body.load_state_dict(stripped, strict=False)
            missing = len(result.missing_keys) if result is not None else "?"
            unexpected = len(result.unexpected_keys) if result is not None else "?"
            print(f"[Info] Loaded RadImageNet backbone (missing={missing}, unexpected={unexpected}).")
        except Exception as exc:
            print(f"[Warning] Could not load RadImageNet backbone ({exc}). Keeping ImageNet weights.")

    # Adapt conv1 for grayscale-as-3channel (average over input channels)
    try:
        with torch.no_grad():
            mean_w = base_backbone.body.conv1.weight.mean(dim=1, keepdim=True)
            base_backbone.body.conv1.weight.copy_(mean_w.expand_as(base_backbone.body.conv1.weight))
        print("[Info] conv1 weights averaged for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1.")

    # Wrap backbone with GlobalAwareBackbone
    global_encoder = GlobalContextEncoder(in_channels=3, out_channels=global_channels)
    global_backbone = GlobalAwareBackbone(
        base_backbone=base_backbone,
        global_encoder=global_encoder,
        num_fpn_levels=5,
        fpn_channels=256,
        global_channels=global_channels,
    )
    print(f"[Info] GlobalAwareBackbone created (global_channels={global_channels}, identity init).")

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model = RetinaNet(
        backbone=global_backbone,
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

        # Prepare global context: downsample batch 4× and feed to global encoder
        with torch.no_grad():
            imgs_stacked = torch.stack(images)  # [B, 3, H, W]
            small_imgs = F.interpolate(
                imgs_stacked, scale_factor=0.25, mode="bilinear", align_corners=False
            )
        model.backbone.set_global_image(small_imgs)

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

            # Set global context for this single image
            small_img = F.interpolate(
                img_t.unsqueeze(0), scale_factor=0.25, mode="bilinear", align_corners=False
            )
            model.backbone.set_global_image(small_img)

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
        description="Train a RetinaNet-ResNet50-FPN + GlobalContext detection model for VinDr lesion detection (Direction K)."
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
                             "FPN + detection head + global_enc + fusion convs use full --lr.")
    parser.add_argument("--input-h", type=int, default=1024,
                        help="Resize height for model input.")
    parser.add_argument("--input-w", type=int, default=512,
                        help="Resize width for model input.")
    parser.add_argument("--val-iou-threshold", type=float, default=0.1,
                        help="IoU threshold for matching predicted boxes to GT during validation.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cliff-patience-ratio", type=float, default=0.0,
                        help="Cliff-aware patience: if the monitored metric for an epoch falls "
                             "below (best × cliff_patience_ratio), classify that epoch as a "
                             "'cliff' and do NOT increment the patience counter. "
                             "0 = disabled (default). Recommended: 0.6.")
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
                             "4.0 → positive images sampled ~4× more than negatives.")
    parser.add_argument("--anchor-sizes", type=str, default="32,64,128,256,512",
                        help="Comma-separated anchor sizes (one per FPN level). "
                             "Default: 32,64,128,256,512.")
    parser.add_argument("--nms-thresh", type=float, default=0.3,
                        help="NMS IoU threshold applied during inference. Default 0.3.")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Minimum score threshold for reported detections. Default 0.05.")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal Loss foreground weight alpha. Default 0.25 (backward-compatible).")
    parser.add_argument("--monitor-metric", type=str, default="fbeta2",
                        choices=["f1", "recall", "fbeta2", "fbeta2_ref"],
                        help="Metric to monitor for checkpoint saving and early stopping.")
    parser.add_argument("--hide-progress-bar", action="store_true")
    parser.add_argument("--lesion-types", type=str, default=None,
                        help="Comma-separated lesion type names to keep as positive GT boxes. "
                             "Default: None (use all annotated boxes).")
    parser.add_argument("--min-box-side", type=float, default=0.0,
                        help="Minimum box shortest side in resized-image space (pixels). "
                             "0 = no filter (default). Recommended: 24.0.")
    parser.add_argument("--max-box-ar", type=float, default=float("inf"),
                        help="Maximum box aspect ratio (max_side / min_side) to keep as positive GT. "
                             "inf = no filter (default). Recommended: 3.0.")
    parser.add_argument("--global-channels", type=int, default=128,
                        help="Output channels of GlobalContextEncoder. "
                             "Determines the size of the global context vector injected into each "
                             "FPN level. Default 128.")
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
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.K.pth"

    print(f"Start time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    all_samples = load_samples(csv_path, images_root, split_name="training",
                               lesion_types=[t.strip() for t in args.lesion_types.split(",")]
                               if args.lesion_types else None,
                               min_box_side=float(args.min_box_side),
                               max_box_ar=float(args.max_box_ar),
                               input_w=int(args.input_w))
    print(f"Total samples: {len(all_samples)}")
    if args.lesion_types:
        print(f"Lesion type filter: {args.lesion_types}")
    if float(args.min_box_side) > 0.0 or float(args.max_box_ar) < float("inf"):
        print(f"Box detectability filter: min_side≥{args.min_box_side:.1f}px, max_AR≤{args.max_box_ar:.1f}")

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
    print(f"Global channels: {args.global_channels}")

    encoder_lr = float(args.lr) * float(args.encoder_lr_multiplier)
    head_lr = float(args.lr)
    print(
        f"Encoder LR (ResNet50 body): {encoder_lr:.2e} | "
        f"Other LR (FPN/head/global_enc/fusion): {head_lr:.2e} | "
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
    model = build_global_retinanet(
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
        global_channels=int(args.global_channels),
    )
    model.to(device)

    # Differential LR:
    #   - model.backbone.base.body (pretrained ResNet50 body): low LR
    #   - everything else (FPN, detection head, global_enc, fusion convs): high LR
    resnet_body_param_ids = {id(p) for p in model.backbone.base.body.parameters()}
    resnet_body_params = list(model.backbone.base.body.parameters())
    other_params = [p for p in model.parameters() if id(p) not in resnet_body_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": resnet_body_params, "lr": encoder_lr},
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
        cliff_ratio = float(args.cliff_patience_ratio)
        is_cliff = (
            cliff_ratio > 0.0
            and best_metric > 0.0
            and cur_monitor < best_metric * cliff_ratio
        )
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
                    "best_fbeta2": float(val_metrics.get("best_fbeta2", 0.0)),
                    "best_fbeta2_thresh": float(val_metrics.get("best_fbeta2_thresh", 0.3)),
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                    "anchor_sizes": str(args.anchor_sizes),
                    "nms_thresh": float(args.nms_thresh),
                    "focal_alpha": float(args.focal_alpha),
                    "global_channels": int(args.global_channels),
                },
            )
            print(f"  [Checkpoint] Epoch {epoch + 1} | Saved ({monitor_metric_name}={best_metric:.4f}, patience reset) -> {save_path}")
        elif is_cliff:
            print(
                f"  [Cliff] Epoch {epoch + 1}: metric={cur_monitor:.4f} is "
                f"{cur_monitor / best_metric * 100:.0f}% of best={best_metric:.4f} "
                f"(< {cliff_ratio:.0%} threshold) — patience not incremented "
                f"(no_improve={no_improve}/{args.patience})"
            )
        else:
            no_improve += 1
            _pat = int(args.patience)
            if _pat > 0 and no_improve >= _pat:
                print(f"  [EarlyStop] no_improve={no_improve}/{_pat} — early stopping triggered.")
                break
            elif _pat > 0:
                _remaining = _pat - no_improve
                _warn = "  !!!  close to stopping" if _remaining <= 3 else ""
                print(f"  [Patience] no_improve={no_improve}/{_pat} (remaining={_remaining}){_warn}")

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
