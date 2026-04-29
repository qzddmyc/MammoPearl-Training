# 方案 A：RetinaNet + Anchor 级别 Hard Negative Mining (HNM)
# 基于 bbox-train.py（rec_41），新增 anchor 级别 HNM，限制每图参与 loss 计算的背景 anchor 数量。
# 根因：RetinaNet 默认对全部 ~67,500 个 anchor 计算 Focal Loss，当模型陷入退化解
# （所有 anchor score≈0.9）后，背景 anchor 数量比病灶 anchor 多约 22,500 倍，梯度完全
# 被背景淹没，Focal Loss 失效。HNM 将背景 anchor 限制为正样本 anchor 的 N 倍，根本上
# 解决梯度淹没问题。

r"""

Use this to run in Git Bash:

python src/data/bounding-box/bbox-train-A.py \
    --epochs 50 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --lr 0.005 \
    --post-warmup-lr 0.001 \
    --warmup-balanced-epochs 5 \
    --warmup-pos-weight-ratio 3.0 \
    --full-train-pos-weight-ratio 3.0 \
    --freeze-epochs 0 \
    --augment \
    --aug-hflip-prob 0.5 \
    --aug-brightness-delta 0.2 \
    --aug-rotation-max-deg 8.0 \
    --focal-alpha 0.5 \
    --focal-gamma 2.0 \
    --box-fg-iou-thresh 0.4 \
    --box-bg-iou-thresh 0.3 \
    --box-nms-thresh 0.5 \
    --box-score-thresh 0.001 \
    --box-detections-per-img 20 \
    --val-detections-per-img 300 \
    --input-min-size 1200 \
    --classification-loss-scale 1.0 \
    --hnm-neg-pos-ratio 10 \
    --hnm-min-negatives 512 \
    --recall-stop \
    --patience 15 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --hide-progress-bar

本文件介绍：

方案 A — RetinaNet + Anchor 级 Hard Negative Mining（基于 rec_41 修改）

核心改动（相对于 bbox-train.py）：
  1. 新增 _patch_hnm_classification_loss() 函数：通过 monkey-patch 替换
     RetinaNetClassificationHead.compute_loss，在分类 loss 计算时对每张图的
     背景 anchor 做 top-K 困难样本筛选，K = max(num_fg * hnm_neg_pos_ratio,
     hnm_min_negatives)。只有"最难"的 K 个背景 anchor 参与 loss 反向传播。
  2. 新增 --hnm-neg-pos-ratio 参数（默认 None，不激活 HNM）：背景/正样本 anchor 比。
  3. 新增 --hnm-min-negatives 参数（默认 512）：每张图最少保留的背景 anchor 数。
  4. 推荐命令中 pos-weight-ratio 从 10.0 降回 3.0（HNM 在 anchor 层面已处理不平衡）。
  5. classification-loss-scale 从 2.0 降回 1.0（HNM 后不需要额外放大分类 loss）。

预期效果：背景 anchor 从 67,500 降至约 512~3,000 个/图参与 loss，使病灶梯度不再被淹没。

"""

from __future__ import annotations

import os
# Prevent libgomp warning when OMP_NUM_THREADS is set to "" or "0"
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import atexit
import datetime
import math
import random
import re
import signal
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn.functional as F
try:
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
except Exception:  # pragma: no cover
    retinanet_resnet50_fpn_v2 = None  # type: ignore
    RetinaNet_ResNet50_FPN_V2_Weights = None
    RetinaNetClassificationHead = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    """Read an image from a path that may contain unicode characters."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB uint8 with 3 channels.

    For grayscale (2-D) inputs, CLAHE contrast enhancement
    (clipLimit=2.0, tileGridSize=8×8) is applied before RGB conversion.
    """
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
    """Convert RGB uint8 image to a float tensor in [0, 1]."""
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _rotate_box_centers_preserve(
    boxes: torch.Tensor,
    angle_deg: float,
    img_h: int,
    img_w: int,
    pivot_x: float,
    pivot_y: float,
) -> torch.Tensor:
    """Rotate bbox centers around a given pivot; keep original box size.

    Each box center is rotated around (pivot_x, pivot_y).  The box is then
    reconstructed with its original width and height centered on the new
    rotated position.  Coordinates are clamped to [0, W] x [0, H].
    """
    if boxes.numel() == 0:
        return boxes.clone()

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    orig_w = boxes[:, 2] - boxes[:, 0]
    orig_h = boxes[:, 3] - boxes[:, 1]
    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    dx = box_cx - pivot_x
    dy = box_cy - pivot_y
    new_cx = cos_a * dx - sin_a * dy + pivot_x
    new_cy = sin_a * dx + cos_a * dy + pivot_y

    new_x1 = (new_cx - orig_w / 2.0).clamp(0.0, float(img_w))
    new_y1 = (new_cy - orig_h / 2.0).clamp(0.0, float(img_h))
    new_x2 = (new_cx + orig_w / 2.0).clamp(0.0, float(img_w))
    new_y2 = (new_cy + orig_h / 2.0).clamp(0.0, float(img_h))

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


def random_augment_fn(
    img: torch.Tensor,
    target: Dict[str, torch.Tensor],
    hflip_prob: float = 0.5,
    brightness_delta: float = 0.2,
    rotation_max_deg: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply random augmentation to a (image tensor, target) pair.

    img: float tensor in [0, 1] of shape [C, H, W].
    target: dict with 'boxes' in xyxy format and other Faster R-CNN keys.

    Augmentations applied:
    - Random horizontal flip (probability = hflip_prob)
    - Random brightness jitter (uniform in [-brightness_delta, +brightness_delta])
    - Random small-angle rotation using strategy-A (keep original image size, fill
      empty corners with 0; update boxes via rotated-corner AABB)
    """
    _, h, w = img.shape

    # Random horizontal flip
    if random.random() < hflip_prob:
        img = torch.flip(img, [2])
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            flipped_boxes = boxes.clone()
            flipped_boxes[:, 0] = float(w) - boxes[:, 2]
            flipped_boxes[:, 2] = float(w) - boxes[:, 0]
            target = {**target, "boxes": flipped_boxes}

    # Random small-angle rotation (strategy A: keep output size, zero-fill corners)
    # Only applied to single-bbox images to avoid pivot ambiguity for multi-lesion cases.
    if rotation_max_deg > 0.0:
        boxes = target.get("boxes")
        n_boxes = boxes.shape[0] if boxes is not None else 0
        if n_boxes == 1:
            angle = random.uniform(-rotation_max_deg, rotation_max_deg)
            if abs(angle) > 0.1:  # skip near-zero rotations for efficiency
                # Rotation pivot = bbox center
                bx1, by1, bx2, by2 = boxes[0].tolist()
                pivot_x = (bx1 + bx2) / 2.0
                pivot_y = (by1 + by2) / 2.0

                # Rotate image tensor via numpy/cv2 (keeps same H x W)
                img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                M = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle, 1.0)
                rotated_np = cv2.warpAffine(
                    img_np, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                img = torch.from_numpy(rotated_np.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

                # Crop away black corners introduced by rotation.
                # detect_breast_region works on non-zero pixels, so the zero-filled
                # rotation corners are naturally excluded.
                crop_box = detect_breast_region(rotated_np, margin_ratio=0.0)
                rotated_np_cropped, _ = crop_image_and_boxes(rotated_np, np.zeros((0, 4), dtype=np.float32), crop_box)
                img = torch.from_numpy(rotated_np_cropped.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                # Remap h, w to the cropped size for subsequent box clamping
                _, h, w = img.shape
                cx1, cy1, _cx2, _cy2 = crop_box

                # Update bbox: rotate center around pivot, keep original size
                rotated_boxes = _rotate_box_centers_preserve(boxes, angle, h + cy1, w + cx1, pivot_x, pivot_y)
                # Remap rotated bbox to crop-local coordinates
                rotated_boxes[:, 0] -= cx1
                rotated_boxes[:, 2] -= cx1
                rotated_boxes[:, 1] -= cy1
                rotated_boxes[:, 3] -= cy1
                rotated_boxes[:, 0] = rotated_boxes[:, 0].clamp(0.0, float(w))
                rotated_boxes[:, 2] = rotated_boxes[:, 2].clamp(0.0, float(w))
                rotated_boxes[:, 1] = rotated_boxes[:, 1].clamp(0.0, float(h))
                rotated_boxes[:, 3] = rotated_boxes[:, 3].clamp(0.0, float(h))
                # Drop degenerate boxes that collapsed after clamp
                keep = (rotated_boxes[:, 2] > rotated_boxes[:, 0] + 1) & (
                    rotated_boxes[:, 3] > rotated_boxes[:, 1] + 1
                )
                rotated_boxes = rotated_boxes[keep]
                target_labels = target.get("labels")
                target_area = target.get("area")
                target_iscrowd = target.get("iscrowd")
                new_target: Dict[str, torch.Tensor] = {
                    **target,
                    "boxes": rotated_boxes,
                }
                if target_labels is not None:
                    new_target["labels"] = target_labels[keep]
                if target_area is not None:
                    area = (rotated_boxes[:, 2] - rotated_boxes[:, 0]) * (
                        rotated_boxes[:, 3] - rotated_boxes[:, 1]
                    )
                    new_target["area"] = area
                if target_iscrowd is not None:
                    new_target["iscrowd"] = target_iscrowd[keep]
                target = new_target

    # Random brightness/contrast jitter
    if brightness_delta > 0.0:
        factor = 1.0 + random.uniform(-brightness_delta, brightness_delta)
        img = torch.clamp(img * factor, 0.0, 1.0)

    return img, target


def detect_breast_region(
    img: np.ndarray,
    margin_ratio: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Detect a breast-region crop on a processed mammogram.

    The processed images already suppress most background, so a largest-contour
    heuristic is sufficient and keeps the detector focused on coarse lesion
    localization inside the breast region.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0, 0, 0, 0
    if np.max(gray) <= 0:
        return 0, 0, w, h

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(gray > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, w, h
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
    else:
        contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(contour)
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + cw)
        y2 = int(y + ch)

    margin_ratio = max(0.0, float(margin_ratio))
    margin_x = int(round((x2 - x1) * margin_ratio))
    margin_y = int(round((y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def crop_image_and_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    crop_box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and remap boxes to the crop-local coordinate system."""
    x1, y1, x2, y2 = crop_box
    cropped_img = img[y1:y2, x1:x2]
    if boxes.size == 0:
        return cropped_img, np.zeros((0, 4), dtype=np.float32)

    cropped_boxes = boxes.astype(np.float32).copy()
    cropped_boxes[:, [0, 2]] -= float(x1)
    cropped_boxes[:, [1, 3]] -= float(y1)

    crop_h, crop_w = cropped_img.shape[:2]
    cropped_boxes[:, 0] = np.clip(cropped_boxes[:, 0], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 2] = np.clip(cropped_boxes[:, 2], 0, max(crop_w - 1, 0))
    cropped_boxes[:, 1] = np.clip(cropped_boxes[:, 1], 0, max(crop_h - 1, 0))
    cropped_boxes[:, 3] = np.clip(cropped_boxes[:, 3], 0, max(crop_h - 1, 0))

    keep = (cropped_boxes[:, 2] > cropped_boxes[:, 0] + 1) & (cropped_boxes[:, 3] > cropped_boxes[:, 1] + 1)
    return cropped_img, cropped_boxes[keep]


@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray
    orig_size: Tuple[float, float]


class VinDrBboxDataset(Dataset):
    """Dataset grouped at the image level.

    Each item contains one image and all lesion boxes found for that image.
    Images with no lesion boxes are kept as negative samples.
    """

    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        split_name: str,
        positive_only: bool = False,
        crop_breast_region: bool = True,
        breast_crop_margin: float = 0.05,
    ) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name
        self.positive_only = positive_only
        self.crop_breast_region = crop_breast_region
        self.breast_crop_margin = breast_crop_margin

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()

        if df.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        self.samples: List[Sample] = []
        grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=True)

        for (patient_id, _series_id, image_id), group in grouped:
            # Keep all boxes for this image (some images have multiple lesions).
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            boxes: np.ndarray
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)

            image_path = images_root / str(patient_id) / f"{image_id}"

            if boxes.size > 0:
                invalid = np.sum((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1]))
                if invalid > 0:
                    print(f"[Warning] Found {invalid} invalid boxes in {image_path}")

            if positive_only and len(boxes) == 0:
                continue

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            self.samples.append(
                Sample(
                    patient_id=str(patient_id),
                    image_id=str(image_id),
                    image_path=image_path,
                    boxes=boxes,
                    orig_size=(orig_h, orig_w),
                )
            )

        if not self.samples:
            raise ValueError(
                f"No image samples could be built from {csv_path} with split={split_name!r}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")

        img = normalize_image(read_image_unicode(sample.image_path))
        h, w = img.shape[:2]

        boxes = sample.boxes.astype(np.float32)

        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
            # 过滤非法框，左值大于右值，或像素值小于 1。
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        if self.crop_breast_region:
            crop_box = detect_breast_region(img, margin_ratio=self.breast_crop_margin)
            img, boxes = crop_image_and_boxes(img, boxes, crop_box)

        if boxes.size == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.from_numpy(boxes)
            labels_tensor = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                (boxes_tensor[:, 2] - boxes_tensor[:, 0]) *
                (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                if boxes_tensor.numel() > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }
        return image_to_tensor(img), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class TrainAugmentWrapper(torch.utils.data.Dataset):
    """Wraps a Dataset/Subset and applies random augmentation during training.

    This allows the same underlying dataset object to be shared between
    train (with augmentation) and val (without augmentation) without creating
    two separate dataset instances.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        hflip_prob: float = 0.5,
        brightness_delta: float = 0.2,
        rotation_max_deg: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.hflip_prob = float(hflip_prob)
        self.brightness_delta = float(brightness_delta)
        self.rotation_max_deg = float(rotation_max_deg)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, target = self.dataset[index]
        img, target = random_augment_fn(
            img, target,
            hflip_prob=self.hflip_prob,
            brightness_delta=self.brightness_delta,
            rotation_max_deg=self.rotation_max_deg,
        )
        return img, target


class CopyPasteWrapper(torch.utils.data.Dataset):
    """Copy-paste augmentation: paste lesion crops from positive samples onto negative images.

    With probability ``paste_prob``, a negative sample is selected and one
    randomly-chosen positive sample's lesion crop is pasted onto it.  All other
    samples are returned unchanged.  The wrapper is applied *after*
    TrainAugmentWrapper so the pasted crops have already been flipped/rotated.

    Args:
        dataset: The underlying dataset (may already be wrapped by TrainAugmentWrapper).
        positive_indices: Indices in *dataset* that correspond to positive images.
            These are sampled when choosing a crop donor.
        paste_prob: Probability that a negative sample is chosen as the paste target.
        max_pastes: Maximum number of crops to paste per target image (chosen uniformly
            from [1, max_pastes]).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        positive_indices: List[int],
        paste_prob: float = 0.4,
        max_pastes: int = 2,
    ) -> None:
        self.dataset = dataset
        self.positive_indices = list(positive_indices)
        self.paste_prob = float(paste_prob)
        self.max_pastes = int(max_pastes)
        if not self.positive_indices:
            print("[Warning] CopyPasteWrapper: no positive indices supplied; augmentation disabled.")

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, target = self.dataset[index]

        # Only augment negative images (no GT boxes) and only with paste_prob probability.
        if (
            not self.positive_indices
            or target["boxes"].shape[0] > 0
            or random.random() > self.paste_prob
        ):
            return img, target

        # img: [C, H, W] float32 tensor
        _, H, W = img.shape
        img = img.clone()
        new_boxes: List[torch.Tensor] = []

        # Compute tissue bounding box on the target image.
        # Pixels with mean value > 0.05 across channels are considered tissue.
        # Paste positions are restricted to this bounding box to avoid placing
        # lesion crops onto black background regions (would teach wrong prior).
        mean_img = img.mean(dim=0)  # [H, W]
        tissue_mask = mean_img > 0.05
        tissue_rows = tissue_mask.any(dim=1).nonzero(as_tuple=False)
        tissue_cols = tissue_mask.any(dim=0).nonzero(as_tuple=False)
        if tissue_rows.numel() == 0 or tissue_cols.numel() == 0:
            # Fully black image; fall back to full image bounds.
            t_y1, t_x1, t_y2, t_x2 = 0, 0, H, W
        else:
            t_y1 = int(tissue_rows[0].item())
            t_y2 = int(tissue_rows[-1].item()) + 1
            t_x1 = int(tissue_cols[0].item())
            t_x2 = int(tissue_cols[-1].item()) + 1

        n_paste = random.randint(1, self.max_pastes)
        for _ in range(n_paste):
            donor_idx = random.choice(self.positive_indices)
            donor_img, donor_target = self.dataset[donor_idx]
            if donor_target["boxes"].shape[0] == 0:
                continue
            # Pick one random box from the donor.
            bi = random.randrange(donor_target["boxes"].shape[0])
            x1, y1, x2, y2 = donor_target["boxes"][bi].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            # Clamp crop to donor image bounds.
            _, dH, dW = donor_img.shape
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, dW), min(y2, dH)
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = donor_img[:, y1c:y2c, x1c:x2c]  # [C, ch, cw]
            cH, cW = crop.shape[1], crop.shape[2]

            # Build a lesion mask for the crop via largest-connected-component.
            # This removes the rectangular frame of black/near-black pixels
            # surrounding the actual tissue, so only genuine lesion pixels are
            # pasted (no hard rectangular boundary visible in the result).
            crop_gray = (crop.mean(dim=0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Light morphological dilation to avoid over-eroding lesion edges.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            n_labels, labels_map = cv2.connectedComponents(thresh, connectivity=8)
            if n_labels > 1:
                # Label 0 is background; find the largest foreground label.
                label_sizes = np.bincount(labels_map.ravel())
                label_sizes[0] = 0  # exclude background
                largest_label = int(label_sizes.argmax())
                lesion_mask = (labels_map == largest_label).astype(np.uint8)  # HxW uint8
            else:
                # Fallback: use all non-zero pixels.
                lesion_mask = (thresh > 0).astype(np.uint8)
            if lesion_mask.sum() == 0:
                continue

            # Feathered alpha: distance transform gives each mask pixel its
            # distance to the nearest edge, clamped to [0, feather_width].
            # This creates a smooth gradient at the lesion boundary so there
            # is no hard cut and no residual black fringe.
            _feather_w = 5  # pixels; wider = softer edge
            dist_map = cv2.distanceTransform(lesion_mask, cv2.DIST_L2, 3)
            alpha_map = np.clip(dist_map / float(_feather_w), 0.0, 1.0).astype(np.float32)  # [cH,cW]
            alpha_t = torch.from_numpy(alpha_map).unsqueeze(0)  # [1, cH, cW]

            # Paste within the tissue bounding box only.
            avail_w = (t_x2 - t_x1) - cW
            avail_h = (t_y2 - t_y1) - cH
            if avail_w < 0 or avail_h < 0:
                # Tissue region smaller than crop; skip.
                continue
            # Try up to 5 random positions; accept only if ≥70% of the paste
            # region contains tissue (mean pixel > 0.05), preventing lesion
            # crops from being placed on black background areas outside the
            # breast contour (the tissue bbox is a rectangle, not the actual shape).
            placed = False
            for _attempt in range(5):
                px = t_x1 + random.randint(0, avail_w)
                py = t_y1 + random.randint(0, avail_h)
                region_mean = mean_img[py:py + cH, px:px + cW]
                tissue_ratio = float((region_mean > 0.05).float().mean().item())
                if tissue_ratio >= 0.70:
                    placed = True
                    break
            if not placed:
                continue

            # Step A: Brightness alignment — scale crop P95 to match target region P95
            # (mask-weighted pixels only).  Using the 95th-percentile focuses on the
            # brightest tissue pixels and avoids pulling the estimate down via background.
            target_region = img[:, py:py + cH, px:px + cW]  # [C, cH, cW]
            _lesion_px_crop   = crop[:, lesion_mask.astype(bool)].reshape(-1)   # 1-D
            _lesion_px_target = target_region[:, lesion_mask.astype(bool)].reshape(-1)
            crop_p95   = float(torch.quantile(_lesion_px_crop,   0.95)) if _lesion_px_crop.numel() > 0 else 1e-6
            target_p95 = float(torch.quantile(_lesion_px_target, 0.95)) if _lesion_px_target.numel() > 0 else 1e-6
            brightness_scale = float(np.clip(target_p95 / max(crop_p95, 1e-6), 0.5, 2.0))
            crop_adj = (crop * brightness_scale).clamp(0.0, 1.0)

            # Step B: Center darkening — gaussian-shaped gamma map applied to the
            # brightness-adjusted crop. Center gamma < 1 (darken), edge gamma → 1.
            # Gaussian sigma covers ~half the crop so the effect fades to edges.
            # gamma_center = 0.75 means the brightest center pixel becomes pixel^0.75.
            _gamma_center = 0.75
            cy, cx = cH / 2.0, cW / 2.0
            sigma = max(cy, cx) * 0.5
            ys = torch.arange(cH, dtype=torch.float32)
            xs = torch.arange(cW, dtype=torch.float32)
            dist2 = ((ys.unsqueeze(1) - cy) ** 2 + (xs.unsqueeze(0) - cx) ** 2)
            gaussian = torch.exp(-dist2 / (2 * sigma ** 2))  # [cH, cW], max=1 at center
            # gamma_map: center → _gamma_center, edges → 1.0
            gamma_map = 1.0 - gaussian * (1.0 - _gamma_center)  # [cH, cW]
            gamma_map = gamma_map.unsqueeze(0)  # [1, cH, cW]
            # Apply gamma: pixel^gamma_map (only on alpha-weighted region to avoid edge artefacts)
            crop_adj = torch.pow(crop_adj.clamp(1e-6, 1.0), gamma_map)

            # Step C: Feathered paste: result = crop_adj * alpha + background * (1 - alpha)
            # Guard against overlap: check IoU of candidate box with already-pasted boxes.
            # If IoU > 0.3 with any existing box, skip this paste to avoid pixel corruption
            # and duplicate/overlapping supervision signals.
            candidate_box = torch.tensor(
                [float(px), float(py), float(px + cW), float(py + cH)]
            )
            overlap = False
            for prev_box in new_boxes:
                ix1 = max(candidate_box[0], prev_box[0])
                iy1 = max(candidate_box[1], prev_box[1])
                ix2 = min(candidate_box[2], prev_box[2])
                iy2 = min(candidate_box[3], prev_box[3])
                inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
                if inter > 0:
                    area_a = float((candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1]))
                    area_b = float((prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1]))
                    iou = inter / max(area_a + area_b - inter, 1e-6)
                    if iou > 0.3:
                        overlap = True
                        break
            if overlap:
                continue

            blended = crop_adj * alpha_t + target_region * (1.0 - alpha_t)
            img[:, py:py + cH, px:px + cW] = blended
            new_boxes.append(candidate_box)

        if new_boxes:
            new_boxes_t = torch.stack(new_boxes, dim=0)  # [N, 4]
            new_labels = torch.ones((new_boxes_t.shape[0],), dtype=torch.int64)
            new_area = (new_boxes_t[:, 2] - new_boxes_t[:, 0]) * (new_boxes_t[:, 3] - new_boxes_t[:, 1])
            new_crowd = torch.zeros((new_boxes_t.shape[0],), dtype=torch.int64)
            target = {
                "boxes": new_boxes_t,
                "labels": new_labels,
                "image_id": target["image_id"],
                "area": new_area,
                "iscrowd": new_crowd,
            }

        return img, target


def summarize_subset(samples: List[Sample], indices: List[int]) -> Dict[str, Any]:
    """Summarize a dataset subset without changing its distribution."""
    if not indices:
        return {
            "images": 0,
            "patients": 0,
            "positive_images": 0,
            "negative_images": 0,
            "positive_ratio": 0.0,
        }

    patient_ids = {samples[i].patient_id for i in indices}
    positive_images = sum(1 for i in indices if samples[i].boxes.size > 0)
    negative_images = len(indices) - positive_images
    positive_ratio = float(positive_images / max(len(indices), 1))

    return {
        "images": int(len(indices)),
        "patients": int(len(patient_ids)),
        "positive_images": int(positive_images),
        "negative_images": int(negative_images),
        "positive_ratio": float(positive_ratio),
    }


def compute_neg_pos_ratio(summary: Dict[str, Any]) -> float:
    """Return negative-to-positive image ratio for a summarized subset."""
    positive_images = int(summary.get("positive_images", 0))
    negative_images = int(summary.get("negative_images", 0))
    if positive_images <= 0:
        return float("inf") if negative_images > 0 else 0.0
    return float(negative_images / positive_images)


def warn_on_small_epoch_positive_pool(train_summary: Dict[str, Any], only_use: float) -> None:
    """Warn when epoch subsampling leaves too few positive images for stable training."""
    if only_use >= 1.0:
        return

    positive_images = int(train_summary.get("positive_images", 0))
    estimated_positive_images = math.ceil(positive_images * only_use)
    if 0 < estimated_positive_images < 256:
        print(
            f"[Warning] --only-use={only_use:.3f} leaves about {estimated_positive_images} positive images per epoch "
            f"under the current train split. This often weakens lesion learning and can trigger FP rebound; "
            f"prefer --only-use 1.0 for final detector training."
        )


def select_epoch_subset(
    train_indices: List[int],
    samples: List[Sample],
    epoch: int,
    only_use: float,
    seed: int,
) -> List[int]:
    """Select a rotating subset of training indices for one epoch.

    Ensures all images are visited across epochs by cycling through
    positive and negative samples independently.  Positive samples are
    protected to never fall below the original positive ratio of the subset.
    """
    if only_use >= 1.0:
        return train_indices

    pos_all = [i for i in train_indices if samples[i].boxes.size > 0]
    neg_all = [i for i in train_indices if samples[i].boxes.size == 0]

    total_target = max(1, math.ceil(len(train_indices) * only_use))
    original_pos_ratio = len(pos_all) / max(len(train_indices), 1)

    # Protect positive samples: at least the original ratio worth of positives
    n_pos = max(1, math.ceil(total_target * original_pos_ratio)) if pos_all else 0
    n_pos = min(n_pos, len(pos_all))
    n_neg = min(total_target - n_pos, len(neg_all))
    n_neg = max(0, n_neg)

    selected: List[int] = []

    if pos_all and n_pos > 0:
        pos_cycles = max(1, math.ceil(len(pos_all) / n_pos))
        pos_cycle_group = epoch // pos_cycles
        pos_epoch_in_cycle = epoch % pos_cycles
        rng_pos = random.Random(seed + 500 + pos_cycle_group * 31)
        pos_shuffled = pos_all.copy()
        rng_pos.shuffle(pos_shuffled)
        start_p = pos_epoch_in_cycle * n_pos
        selected += [pos_shuffled[j % len(pos_all)] for j in range(start_p, start_p + n_pos)]

    if neg_all and n_neg > 0:
        neg_cycles = max(1, math.ceil(len(neg_all) / n_neg))
        neg_cycle_group = epoch // neg_cycles
        neg_epoch_in_cycle = epoch % neg_cycles
        rng_neg = random.Random(seed + 700 + neg_cycle_group * 37)
        neg_shuffled = neg_all.copy()
        rng_neg.shuffle(neg_shuffled)
        start_n = neg_epoch_in_cycle * n_neg
        selected += [neg_shuffled[j % len(neg_all)] for j in range(start_n, start_n + n_neg)]

    return selected


def split_train_val_by_patient(
    samples: List[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Split training data into train/val at the patient level.

    The split is patient-level to prevent leakage, and it tries to keep the
    original positive/negative distribution approximately stable by selecting
    roughly the same proportion of positive-patient and negative-patient images.
    """
    usable_indices: List[int] = list(range(len(samples)))

    patient_to_records: Dict[str, Dict[str, Any]] = {}
    for idx in usable_indices:
        sample = samples[idx]
        record = patient_to_records.setdefault(
            sample.patient_id,
            {
                "patient_id": sample.patient_id,
                "indices": [],
                "num_images": 0,
                "pos_images": 0,
                "neg_images": 0,
            },
        )
        record["indices"].append(idx)
        record["num_images"] += 1
        if sample.boxes.size > 0:
            record["pos_images"] += 1
        else:
            record["neg_images"] += 1

    records = list(patient_to_records.values())
    if not records:
        raise ValueError("No usable samples remain after filtering bad data.")

    positive_records = [r for r in records if r["pos_images"] > 0]
    negative_records = [r for r in records if r["pos_images"] == 0]

    def choose_val_patients(group_records: List[Dict[str, Any]], ratio: float, seed_offset: int) -> set[str]:
        if not group_records:
            return set()

        group_total_images = sum(r["num_images"] for r in group_records)
        target_images = int(round(group_total_images * ratio))
        if target_images <= 0:
            return set()

        rng = random.Random(seed + seed_offset)
        remaining = group_records.copy()
        rng.shuffle(remaining)

        selected: List[Dict[str, Any]] = []
        current = 0

        while remaining and current < target_images:
            current_diff = abs(current - target_images)
            best_idx = None
            best_key = None

            for i, rec in enumerate(remaining):
                new_current = current + rec["num_images"]
                key = (abs(new_current - target_images), -rec["num_images"])
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = i

            if best_idx is None:
                break

            candidate = remaining[best_idx]
            new_diff = abs((current + candidate["num_images"]) - target_images)

            # Accept the candidate if it improves the target distance,
            # or if we still have very little validation data selected.
            if (not selected) or (new_diff <= current_diff) or (current < target_images * 0.85):
                selected.append(candidate)
                current += candidate["num_images"]
                remaining.pop(best_idx)
            else:
                break

        if not selected:
            selected = [max(group_records, key=lambda r: r["num_images"])]

        return {r["patient_id"] for r in selected}

    val_patient_ids = set()
    val_patient_ids |= choose_val_patients(positive_records, val_ratio, 101)
    val_patient_ids |= choose_val_patients(negative_records, val_ratio, 202)

    train_indices = [idx for idx in usable_indices if samples[idx].patient_id not in val_patient_ids]
    val_indices = [idx for idx in usable_indices if samples[idx].patient_id in val_patient_ids]

    # Fallback: if the patient-level split is empty on one side, keep training usable.
    if not train_indices and usable_indices:
        print("[Warning] Patient-level split produced an empty training split; falling back to using all usable samples for training.")
        train_indices = usable_indices.copy()
        val_indices = []

    if not val_indices and usable_indices:
        print("[Warning] Patient-level split produced an empty validation split; moving one whole patient to validation.")
        fallback_patient = max(records, key=lambda r: r["num_images"])
        val_patient_ids = {fallback_patient["patient_id"]}
        train_indices = [idx for idx in usable_indices if samples[idx].patient_id not in val_patient_ids]
        val_indices = [idx for idx in usable_indices if samples[idx].patient_id in val_patient_ids]

    train_indices.sort()
    val_indices.sort()

    summary = {
        "val_ratio": float(val_ratio),
        "usable_images": int(len(usable_indices)),
        "usable_patients": int(len(records)),
        "train": summarize_subset(samples, train_indices),
        "val": summarize_subset(samples, val_indices),
        "train_patients": int(len({samples[i].patient_id for i in train_indices})),
        "val_patients": int(len({samples[i].patient_id for i in val_indices})),
    }
    return train_indices, val_indices, summary


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix for two box sets in xyxy format."""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)

    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], a_min=0.0, a_max=None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], a_min=0.0, a_max=None
    )

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Greedy one-to-one matching to compute TP / FP / FN for one image."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0

    keep = pred_scores >= float(score_threshold)
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    ious = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = np.zeros((gt_boxes.shape[0],), dtype=bool)

    tp = 0
    for pred_idx in range(pred_boxes.shape[0]):
        unmatched = np.where(~matched_gt)[0]
        if unmatched.size == 0:
            break

        best_rel = int(unmatched[int(np.argmax(ious[pred_idx, unmatched]))])
        best_iou = float(ious[pred_idx, best_rel])

        if best_iou >= float(iou_threshold):
            matched_gt[best_rel] = True
            tp += 1

    fp = int(pred_boxes.shape[0] - tp)
    fn = int(gt_boxes.shape[0] - tp)
    return tp, fp, fn


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    epoch: int,
    epochs: int,
    disable_tqdm: bool = False,
    val_detections_per_img: Optional[int] = None,
) -> Dict[str, float]:
    """Validate one epoch without gradient computation."""
    model.eval()
    # Optionally override detections_per_img so evaluation is not truncated at the
    # training inference cap (e.g. 20 boxes/image).  When val_detections_per_img is set
    # (e.g. 300), the model can return up to that many boxes per image, allowing TP@0.1
    # to be measured without the hard ceiling that distorts recall statistics and
    # early-stopping.  When None, no override is applied (backward-compatible).
    _orig_det_per_img = getattr(model, "detections_per_img", None)
    if val_detections_per_img is not None and _orig_det_per_img is not None and _orig_det_per_img != val_detections_per_img:
        model.detections_per_img = val_detections_per_img

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_raw_preds = 0

    # Multi-threshold tracking
    multi_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    multi_stats: Dict[float, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in multi_thresholds}

    pbar = tqdm(loader, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                total_images += 1

                pred_boxes = output.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu().numpy()
                pred_scores = output.get("scores", torch.zeros((0,), device=device)).detach().cpu().numpy()
                gt_boxes = target["boxes"].detach().cpu().numpy()

                total_gt_boxes += int(gt_boxes.shape[0])
                total_raw_preds += int(pred_boxes.shape[0])

                # Apply the same score threshold used during validation statistics.
                keep = pred_scores >= float(score_threshold)
                total_pred_boxes += int(np.sum(keep))

                tp, fp, fn = match_predictions_to_gt(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Multi-threshold evaluation
                for t in multi_thresholds:
                    tp_t, fp_t, fn_t = match_predictions_to_gt(
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        gt_boxes=gt_boxes,
                        score_threshold=t,
                        iou_threshold=iou_threshold,
                    )
                    multi_stats[t]["tp"] += tp_t
                    multi_stats[t]["fp"] += fp_t
                    multi_stats[t]["fn"] += fn_t

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_tp + total_fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))

    avg_raw_preds = float(total_raw_preds / max(total_images, 1))
    if avg_raw_preds > 200:
        print(f"[Warning] avg_raw_preds_per_image={avg_raw_preds:.1f} > 200, RetinaNet may not be suppressing background anchors effectively")

    # Print multi-threshold summary and compute best-threshold F1
    parts = []
    best_thresh_f1 = -1.0
    best_thresh = float(score_threshold)
    for t in multi_thresholds:
        tp_t = multi_stats[t]["tp"]
        fp_t = multi_stats[t]["fp"]
        fn_t = multi_stats[t]["fn"]
        p_t = float(tp_t / max(tp_t + fp_t, 1))
        r_t = float(tp_t / max(tp_t + fn_t, 1))
        f1_t = float(2.0 * p_t * r_t / max(p_t + r_t, 1e-12))
        parts.append(f"Val@{t}: TP={tp_t}, FP={fp_t} F1={f1_t:.4f}")
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = t
    print(f"  {' | '.join(parts)}")
    print(f"  [BestThresh] F1={best_thresh_f1:.4f} @ threshold={best_thresh}")

    result: Dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_thresh_f1": best_thresh_f1,
        "best_thresh": best_thresh,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "images": float(total_images),
        "gt_boxes": float(total_gt_boxes),
        "pred_boxes": float(total_pred_boxes),
        "raw_preds": float(total_raw_preds),
        "avg_raw_preds_per_image": avg_raw_preds,
    }
    for t in multi_thresholds:
        result[f"tp@{t}"] = float(multi_stats[t]["tp"])
        result[f"fp@{t}"] = float(multi_stats[t]["fp"])
        result[f"fn@{t}"] = float(multi_stats[t]["fn"])
    # Restore original detections_per_img so training inference is unaffected.
    if val_detections_per_img is not None and _orig_det_per_img is not None and _orig_det_per_img != val_detections_per_img:
        model.detections_per_img = _orig_det_per_img
    return result


def _patch_hnm_classification_loss(
    model: torch.nn.Module,
    neg_pos_ratio: float = 10.0,
    min_negatives: int = 512,
) -> None:
    """Monkey-patch RetinaNet's classification head to apply anchor-level HNM.

    Replaces ``RetinaNetClassificationHead.compute_loss`` with a version that:
      1. Keeps ALL foreground anchors (matched to a GT box, matched_idx >= 0).
      2. For background anchors (matched_idx == -1), computes per-anchor focal
         loss and retains only the top-K hardest (highest-loss) ones, where
         K = max(num_fg * neg_pos_ratio, min_negatives).
      3. Ignores anchors with matched_idx < -1 (between IoU thresholds).

    This prevents the overwhelming gradient from ~67,500 background anchors/image
    from drowning out the few (~3) lesion anchor gradients.

    Args:
        model: The RetinaNet model to patch.
        neg_pos_ratio: Max ratio of kept background anchors to foreground anchors
            per image. Recommended: 10.
        min_negatives: Floor on the number of background anchors kept per image
            even when num_fg == 0. Recommended: 512.
    """
    from torchvision.ops import sigmoid_focal_loss as _sfl

    _alpha = float(model.head.classification_head.focal_loss_alpha)
    _gamma = float(model.head.classification_head.focal_loss_gamma)
    _neg_pos_ratio = float(neg_pos_ratio)
    _min_neg = int(min_negatives)

    def _compute_loss_hnm(
        self,
        targets: List[Dict[str, torch.Tensor]],
        head_outputs: Dict[str, torch.Tensor],
        matched_idxs: List[torch.Tensor],
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        cls_logits = head_outputs["cls_logits"]

        for tgt_img, logits_img, match_img in zip(targets, cls_logits, matched_idxs):
            # Foreground anchors: matched to a GT box
            fg_idxs = torch.where(match_img >= 0)[0]
            n_fg = fg_idxs.numel()

            # Background anchors: explicitly assigned as background
            bg_idxs = torch.where(match_img == -1)[0]
            n_bg = bg_idxs.numel()

            # Build per-class one-hot GT targets (shape: [total_anchors, num_classes])
            gt_target = torch.zeros_like(logits_img)
            if n_fg > 0:
                gt_target[
                    fg_idxs,
                    tgt_img["labels"][match_img[fg_idxs]],
                ] = 1.0

            # Determine how many background anchors to keep
            n_keep = max(int(n_fg * _neg_pos_ratio), _min_neg)
            n_keep = min(n_keep, n_bg)

            if n_bg > n_keep and n_keep > 0:
                # Compute per-anchor focal loss for background (sum over classes)
                bg_loss_per_anchor = _sfl(
                    logits_img[bg_idxs],
                    gt_target[bg_idxs],   # all zeros
                    alpha=_alpha,
                    gamma=_gamma,
                    reduction="none",
                ).sum(dim=-1)  # [n_bg]
                # Keep only the top-n_keep hardest background anchors
                _, topk_idx = torch.topk(bg_loss_per_anchor, n_keep)
                selected_bg = bg_idxs[topk_idx]
            elif n_keep > 0:
                selected_bg = bg_idxs
            else:
                selected_bg = bg_idxs[:0]  # empty

            # Combine foreground + selected background
            all_idxs = torch.cat([fg_idxs, selected_bg])
            if all_idxs.numel() == 0:
                losses.append(logits_img.sum() * 0.0)
                continue

            losses.append(
                _sfl(
                    logits_img[all_idxs],
                    gt_target[all_idxs],
                    alpha=_alpha,
                    gamma=_gamma,
                    reduction="sum",
                ) / max(1, n_fg)
            )

        if not losses:
            return head_outputs["cls_logits"].sum() * 0.0
        return sum(losses) / max(len(targets), 1)

    model.head.classification_head.compute_loss = types.MethodType(
        _compute_loss_hnm, model.head.classification_head
    )
    print(
        f"[Info] HNM patch applied: neg_pos_ratio={neg_pos_ratio}, "
        f"min_negatives={min_negatives}"
    )


def build_model(
    num_classes: int = 2,
    anchor_sizes: List[Tuple[int, ...]] | None = None,
    fg_iou_thresh: float = 0.5,
    bg_iou_thresh: float = 0.4,
    score_thresh: float = 0.05,
    detections_per_img: int = 100,
    nms_thresh: float = 0.5,
    focal_loss_alpha: float = 0.75,
    focal_loss_gamma: float = 2.0,
    medical_backbone_path: Optional[str] = None,
    input_min_size: int = 800,
) -> torch.nn.Module:
    """Build RetinaNet (retinanet_resnet50_fpn_v2) with ResNet50-FPN backbone.

    Steps:
      1. Fetch full COCO-pretrained state_dict; strip cls_logits keys (shape
         mismatch: 91-class vs num_classes).
      2. Build model with weights=None, num_classes=num_classes (default
         9 anchors/location when anchor_sizes is None, so COCO head loads cleanly).
      3. Load stripped COCO weights strict=False (backbone + FPN + regression head).
      4. Optionally overwrite backbone.body with RadImageNet weights
         (medical_backbone_path).
      5. Average conv1 channel weights for grayscale-as-3channel input.

    Args:
        anchor_sizes: Per-FPN-level size tuples.  ``None`` keeps torchvision
            default (3 scales x 3 aspects = 9 anchors/location), required for
            COCO head weight compatibility.
        fg_iou_thresh: Anchor IoU threshold to be assigned as positive.
        bg_iou_thresh: Anchor IoU threshold to be assigned as negative.
        score_thresh: Model-internal score threshold for post-NMS filtering.
        detections_per_img: Max detections kept per image after NMS.
        nms_thresh: IoU threshold for NMS.
        focal_loss_alpha: Positive class weight in Focal Loss.
        focal_loss_gamma: Focusing exponent in Focal Loss.
        medical_backbone_path: Path to RadImageNet ResNet50 checkpoint; if
            None, COCO backbone weights are kept.
        input_min_size: Shorter side of the image after RetinaNet's built-in
            resize transform.  Default 800 matches torchvision default.  Raise
            to 1200 to keep small lesions larger relative to anchors.
    """
    if retinanet_resnet50_fpn_v2 is None:
        raise RuntimeError(
            "torchvision RetinaNet v2 not found. Please upgrade torchvision (>=0.14)."
        )

    # When anchor_sizes is None, we let torchvision use its default AnchorGenerator
    # (3 scales × 3 aspect ratios = 9 anchors/location).  This is required for the
    # COCO-pretrained head to load cleanly (its output channels match 9 anchors).
    # Only build a custom AnchorGenerator when the caller explicitly passes sizes.
    if anchor_sizes is None:
        anchor_generator = None  # will be left as-is after factory construction
    else:
        aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))])
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Step 1: Load full COCO-pretrained RetinaNet (backbone + FPN + head).
    # We keep COCO head weights intact so the detection prior is preserved.
    # num_classes=91 is the COCO default; we will NOT rebuild the head here.
    # The anchor_sizes argument is ignored at this step — we patch it below.
    coco_state_dict = None
    if RetinaNet_ResNet50_FPN_V2_Weights is not None:
        try:
            _pretrained = retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            )
            coco_state_dict = _pretrained.state_dict()
            del _pretrained
        except Exception as exc:
            print(f"[Warning] Could not load RetinaNet COCO weights: {exc}")

    # Step 2: Build model with 2 classes and no weights.
    # Note: do NOT pass anchor_generator to retinanet_resnet50_fpn_v2() —
    # current torchvision passes it positionally inside the factory function,
    # so passing it again as a kwarg raises "got multiple values" TypeError.
    # We replace model.anchor_generator after construction instead.
    try:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
        )
    except TypeError:
        # Very old torchvision: retry with minimal args, then patch below.
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        try:
            model.score_thresh = score_thresh
            model.nms_thresh = nms_thresh
            model.detections_per_img = detections_per_img
            model.fg_iou_thresh = fg_iou_thresh
            model.bg_iou_thresh = bg_iou_thresh
        except AttributeError:
            print("[Warning] Could not patch score_thresh/detections_per_img on old torchvision model.")
    # Replace anchor generator only when caller explicitly passed custom sizes.
    # When anchor_generator is None, the torchvision default (9 anchors/location) is kept,
    # which allows the COCO-pretrained head to load with matching dimensions.
    if anchor_generator is not None:
        model.anchor_generator = anchor_generator
    # Override the built-in resize transform min_size so that small lesions
    # stay larger after rescaling and better match the smallest anchors.
    if input_min_size != 800:
        model.transform.min_size = (input_min_size,)
        model.transform.max_size = int(input_min_size * 1333 / 800)
    print(
        f"[Info] Model params: score_thresh={model.score_thresh}, "
        f"nms_thresh={model.nms_thresh}, "
        f"detections_per_img={model.detections_per_img}, "
        f"input_min_size={model.transform.min_size[0]}"
    )

    # Load COCO weights into the 2-class model with strict=False.
    # backbone + FPN + head regression branch will load cleanly.
    # head classification branch (91-class → 2-class) has a shape mismatch:
    # strict=False cannot skip shape mismatches (only missing/unexpected keys).
    # So we manually remove cls_logits keys from coco_state_dict first,
    # then load the rest — classification head stays randomly initialised.
    if coco_state_dict is not None:
        cls_logits_keys = [k for k in coco_state_dict if "cls_logits" in k]
        for k in cls_logits_keys:
            del coco_state_dict[k]
        missing, unexpected = model.load_state_dict(coco_state_dict, strict=False)
        print(
            f"[Info] Loaded COCO weights (strict=False): "
            f"{len(cls_logits_keys)} cls_logits keys removed (shape mismatch, expected), "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    else:
        print("[Warning] No COCO weights loaded; model starts from random initialisation.")


    # Step 3: Overwrite backbone.body with RadImageNet weights (if provided).
    # COCO FPN + head weights loaded in Step 2 are NOT touched here.
    # Only backbone.body (ResNet50 stem + layers) is overwritten.
    if medical_backbone_path is not None:
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            # Accept checkpoint files that store weights under common keys
            if isinstance(ckpt, dict):
                raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                raw_sd = ckpt
            # Strip common prefixes: module., encoder., backbone., body.
            # Then remap RadImageNet-style numeric Sequential indices to
            # torchvision ResNet named layers:
            #   0 -> conv1,  1 -> bn1,  4 -> layer1,  5 -> layer2,
            #   6 -> layer3, 7 -> layer4
            _idx_to_resnet = {
                "0": "conv1", "1": "bn1",
                "4": "layer1", "5": "layer2",
                "6": "layer3", "7": "layer4",
            }
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            missing, unexpected = model.backbone.body.load_state_dict(stripped, strict=False)
            print(f"[Info] Loaded medical backbone from: {medical_backbone_path}")
            if missing:
                print(f"[Info]   Missing keys ({len(missing)}): {missing[:3]}")
            if unexpected:
                print(f"[Info]   Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        except Exception as exc:
            print(f"[Warning] Could not load medical backbone ({exc}). COCO backbone is retained.")

    # Step 3.5: Adapt conv1 for grayscale-as-3channel input.
    # Mammogram images are grayscale loaded as 3 identical channels (R=G=B). ImageNet's
    # conv1 has 3 distinct channel weights optimized for RGB. Averaging them ensures all
    # three input channels start with the same feature detector, which is the correct
    # inductive bias for our single-modality input.
    try:
        conv1 = model.backbone.body.conv1
        with torch.no_grad():
            mean_w = conv1.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            conv1.weight.copy_(mean_w.expand_as(conv1.weight))
        print("[Info] conv1 weights averaged across channels for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1 (unexpected backbone structure).")

    # Step 4: Set Focal Loss parameters on the classification head.
    try:
        model.head.classification_head.focal_loss_alpha = focal_loss_alpha
        model.head.classification_head.focal_loss_gamma = focal_loss_gamma
        print(f"[Info] RetinaNet Focal Loss: alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")
    except AttributeError:
        print("[Warning] Could not set focal_loss_alpha/gamma on classification head; using defaults.")

    return model


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace, base_lr: Optional[float] = None) -> torch.optim.Optimizer:
    """Create SGD optimizer with separate weight decay for biases/BN and others."""
    decay = float(args.weight_decay)
    base_lr = float(base_lr) if base_lr is not None else float(args.lr)

    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith(".bias") or "bn" in lname or "norm" in lname:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    groups = []
    if params_with_decay:
        groups.append({"params": params_with_decay, "weight_decay": decay})
    if params_without_decay:
        groups.append({"params": params_without_decay, "weight_decay": 0.0})

    opt = torch.optim.SGD(groups, lr=base_lr, momentum=float(args.momentum))
    return opt


def freeze_backbone_layers(model: torch.nn.Module) -> None:
    """Freeze ResNet backbone layer1 and layer2 parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = False


def unfreeze_backbone_layers(model: torch.nn.Module) -> None:
    """Unfreeze previously frozen ResNet backbone layers."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = True


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    disable_tqdm: bool = False,
    classification_loss_scale: float = 1.0,
) -> Tuple[float, int, Dict[str, float]]:
    """Train one epoch with gradient accumulation and optional iter-level LinearLR warmup.

    Returns:
        avg_loss: average loss for the epoch
        optimizer_steps: number of times `optimizer.step()` was actually called
    """
    model.train()
    running_loss = 0.0
    count = 0
    optimizer_steps = 0
    bad_keys_count = 0
    # track RetinaNet sub-losses
    subloss_keys = ("classification", "bbox_regression")
    subloss_sums: Dict[str, float] = defaultdict(float)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        bad_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
        if bad_keys:
            bad_keys_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # Apply classification loss scale to strengthen lesion vs background discrimination.
        # bbox_regression loss keeps weight 1.0.
        if classification_loss_scale != 1.0 and "classification" in loss_dict:
            loss_dict = dict(loss_dict)
            loss_dict["classification"] = loss_dict["classification"] * classification_loss_scale
        loss = sum(loss for loss in loss_dict.values())

        # accumulate named sub-losses for reporting
        for k in subloss_keys:
            v = loss_dict.get(k)
            if v is not None and torch.isfinite(v):
                try:
                    subloss_sums[k] += float(v.item())
                except Exception:
                    # fallback if value cannot be .item()'d
                    subloss_sums[k] += float(v)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        # gradient accumulation (scale before backward)
        scaled_loss = loss / float(accumulation_steps)
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            # Only advance the warmup scheduler at the same time as optimizer.step()
            if warmup_scheduler is not None:
                warmup_scheduler.step()

        batch_loss = float(loss.item())
        running_loss += batch_loss
        count += 1
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    print(f"[Sum] count = {count}")
    print(f"[Sum] bad data count = {bad_keys_count}")

    avg_sublosses: Dict[str, float] = {k: (subloss_sums[k] / max(count, 1)) for k in subloss_keys}
    return running_loss / max(count, 1), optimizer_steps, avg_sublosses


def save_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    meta: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(payload, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bbox detector for VinDr.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to vindr_detection_folds.csv",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Root folder containing processed images_png/<patient_id>/<image_id>",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Best checkpoint path (default: models/bbox_resnet50_hnm.A.pth)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-score-threshold", type=float, default=0.5)
    parser.add_argument("--val-iou-threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.005, help="Base learning rate (after warmup)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Compute device, e.g. 'cuda', 'cuda:0', 'cuda:1', 'cpu'. "
            "When omitted, 'cuda' is used if available, else 'cpu'. "
            "Set explicitly to run multiple variants on different GPUs simultaneously."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Drop negative images (images without any bbox).",
    )
    # Gradient accumulation vs direct large-batch mode
    parser.add_argument("--fuck-running", action="store_true", help="When set use provided batch-size directly; when absent use gradient accumulation")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps when not using --fuck-running")

    # Freeze / unfreeze backbone
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Number of epochs to freeze backbone layer1/2 before unfreezing (0 = no freeze)")

    # Anchor tuning
    parser.add_argument("--anchor-sizes", type=str, default="16,32,64,128,256", help="Comma-separated anchor sizes (one per FPN level)")

    # LR scheduling
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-step-size", type=int, default=0, help="StepLR step size; 0 to use CosineAnnealingLR")

    # IoU thresholds for anchor assignment (RetinaNet matcher)
    parser.add_argument("--box-fg-iou-thresh", type=float, default=0.5, help="Foreground IoU threshold for anchor-to-GT matching")
    parser.add_argument("--box-bg-iou-thresh", type=float, default=0.4, help="Background IoU threshold; anchors below this are negatives")

    # Focal Loss parameters (built in to RetinaNet)
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal loss alpha for foreground (lesion) class; set higher than 0.25 default for imbalanced medical data")

    # Positive-only warmup
    parser.add_argument("--warmup-positive-epochs", type=int, default=0, help="Epochs to train with positive-only images before full training (0=disabled)")

    # Balanced sampling warmup (replaces positive-only warmup)
    parser.add_argument("--warmup-balanced-epochs", type=int, default=0, help="Epochs to use balanced (weighted) sampling before full training (0=disabled)")
    parser.add_argument("--warmup-pos-weight-ratio", type=float, default=10.0, help="Positive sample weight relative to negative in balanced warmup")
    parser.add_argument("--post-warmup-lr", type=float, default=None, help="LR used when rebuilding optimizer after balanced warmup ends (defaults to --lr if not set)")

    # Epoch subsampling
    parser.add_argument("--only-use", type=float, default=1.0, help="Fraction of training data to use per epoch (0.0-1.0); rotates across epochs to cover all data")

    # Inference-time box filtering
    parser.add_argument("--box-score-thresh", type=float, default=0.05, help="Score threshold for inference-time box filtering")
    parser.add_argument("--input-min-size", type=int, default=800, help="Shorter side of the image after RetinaNet resize transform (default 800). Raise to 1200 to make small lesions larger relative to anchors.")
    parser.add_argument("--recall-stop", action="store_true", help="Use Recall@0.1 (TP@0.1 / GT boxes) instead of BestThreshF1 as the early-stopping metric. Recommended for recall-first training.")
    parser.add_argument("--box-detections-per-img", type=int, default=100, help="Max detections per image at inference time")
    parser.add_argument("--val-detections-per-img", type=int, default=None, help="Max detections per image during validation (temporarily overrides --box-detections-per-img). When set (e.g. 300), prevents TP@threshold statistics from being truncated at the training inference cap, giving an accurate recall estimate for early-stopping and checkpoint selection. When omitted (default), uses the same value as --box-detections-per-img (backward-compatible).")
    parser.add_argument("--box-nms-thresh", type=float, default=0.5, help="NMS IoU threshold for post-prediction duplicate suppression; lower (e.g. 0.3) removes more overlapping FP boxes")
    parser.add_argument("--classification-loss-scale", type=float, default=1.0, help="Multiply the RetinaNet classification loss by this factor before summing with bbox_regression loss. Values > 1.0 (e.g. 2.0) strengthen the signal for distinguishing lesions from background, counteracting extreme class imbalance.")
    parser.add_argument("--disable-breast-crop", action="store_true", help="Disable breast-region cropping and train on the full processed image")
    parser.add_argument("--breast-crop-margin", type=float, default=0.05, help="Relative padding added around the detected breast crop")
    parser.add_argument("--hide-progress-bar", action="store_true", help="Suppress tqdm progress bars during training and validation")

    # Anchor-level Hard Negative Mining (HNM) — core change in bbox-train-A.py
    parser.add_argument(
        "--hnm-neg-pos-ratio",
        type=float,
        default=None,
        help=(
            "When set (e.g. 10), activates anchor-level Hard Negative Mining. "
            "For each image, background anchors contributing to classification loss are "
            "limited to max(num_fg_anchors * ratio, --hnm-min-negatives) hardest examples "
            "(highest per-anchor focal loss). Prevents the ~67,500 background anchors from "
            "overwhelming the few lesion anchor gradients. None (default) disables HNM "
            "(backward-compatible). Recommended: 10."
        ),
    )
    parser.add_argument(
        "--hnm-min-negatives",
        type=int,
        default=512,
        help=(
            "Minimum number of background anchors to keep per image when HNM is active, "
            "even if num_fg_anchors == 0. Ensures negative images still provide background "
            "suppression signal. Default: 512. Only used when --hnm-neg-pos-ratio is set."
        ),
    )

    # Data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable random data augmentation (hflip + brightness jitter + optional rotation) on the training set")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Probability of random horizontal flip when --augment is set")
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2, help="Magnitude of random brightness jitter (±delta) when --augment is set")
    parser.add_argument(
        "--aug-rotation-max-deg",
        type=float,
        default=0.0,
        help=(
            "Max absolute rotation angle (degrees) for random small-angle rotation augmentation "
            "when --augment is set. 0 disables rotation. Recommended: 5.0-10.0. "
            "Image size is kept identical (strategy-A: zero-fill corners). "
            "Bboxes are updated via rotated-corner AABB and degenerate boxes are dropped."
        ),
    )

    # Copy-paste augmentation
    parser.add_argument("--copy-paste-prob", type=float, default=0.0, help="Probability of applying copy-paste augmentation to a negative training image (0 = disabled, recommended 0.3-0.5)")
    parser.add_argument("--copy-paste-max-pastes", type=int, default=2, help="Max lesion crops to paste per negative image when copy-paste is enabled")

    # Full-training positive sample weight (prevents post-warmup collapse)
    parser.add_argument(
        "--full-train-pos-weight-ratio",
        type=float,
        default=0.0,
        help=(
            "When > 0, keep a mild positive-oversampling via WeightedRandomSampler throughout the "
            "full training phase (after warmup). Positive samples are given this weight relative "
            "to negative samples (e.g. 3.0 means positives are 3x as likely to be sampled). "
            "Helps prevent the model from collapsing to predict-nothing on heavily imbalanced data. "
            "0.0 (default) disables this and uses plain shuffle."
        ),
    )

    # Medical pretrained backbone (rec_37+)
    parser.add_argument(
        "--medical-backbone-path",
        type=str,
        default=None,
        help=(
            "Path to a medical-domain pretrained ResNet50 checkpoint (e.g. RadImageNet). "
            "If provided, backbone weights are loaded from this file instead of COCO pretrained, "
            "shortening the ImageNet→COCO→mammography transfer chain. "
            "Supports checkpoints with state_dict/model keys and common prefixes "
            "(module., encoder., backbone., body.) which are stripped automatically."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox_resnet50_hnm.A.pth")
    crop_breast_region = not bool(args.disable_breast_crop)
    breast_crop_margin = max(0.0, float(args.breast_crop_margin))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    set_seed(args.seed)

    # freeze_epochs / warmup conflict check
    _freeze = int(args.freeze_epochs)
    _wbal = int(args.warmup_balanced_epochs)
    if 0 < _freeze < _wbal:
        print(
            f"[Warning] --freeze-epochs={_freeze} falls inside balanced warmup range (0~{_wbal}). "
            f"This will rebuild the optimizer mid-warmup and may cause FP instability. "
            f"Recommended: set --freeze-epochs=0 or --freeze-epochs>={_wbal}."
        )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the full training split first, then split it into train/validation
    # at the patient level so that the same patient never appears on both sides.
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=False,
        crop_breast_region=crop_breast_region,
        breast_crop_margin=breast_crop_margin,
    )

    usable_indices = list(range(len(train_dataset.samples)))
    pos_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size > 0]
    neg_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size == 0]

    train_indices, val_indices, split_summary = split_train_val_by_patient(
        samples=train_dataset.samples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    # Keep the existing positive-only behavior for training only.
    # Validation must remain untouched so that it keeps the real distribution.
    if args.positive_only:
        positive_train_indices = [i for i in train_indices if train_dataset.samples[i].boxes.size > 0]
        if positive_train_indices:
            train_indices = positive_train_indices
        else:
            print("Warning: positive_only enabled but no positive samples remain in training split; falling back to mixed training split.")

    train_indices.sort()
    val_indices.sort()

    train_summary = summarize_subset(train_dataset.samples, train_indices)
    val_summary = summarize_subset(train_dataset.samples, val_indices)
    train_neg_to_pos_ratio = compute_neg_pos_ratio(train_summary)
    val_neg_to_pos_ratio = compute_neg_pos_ratio(val_summary)

    split_summary["train"] = train_summary
    split_summary["val"] = val_summary
    split_summary["train_patients"] = int(len({train_dataset.samples[i].patient_id for i in train_indices}))
    split_summary["val_patients"] = int(len({train_dataset.samples[i].patient_id for i in val_indices}))

    _base_train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Optionally wrap the training subset with augmentation.
    if args.augment:
        train_subset = TrainAugmentWrapper(
            _base_train_subset,
            hflip_prob=float(args.aug_hflip_prob),
            brightness_delta=float(args.aug_brightness_delta),
            rotation_max_deg=float(args.aug_rotation_max_deg),
        )
        print(
            f"[Info] Training augmentation enabled | hflip_prob={args.aug_hflip_prob} | "
            f"brightness_delta=±{args.aug_brightness_delta} | "
            f"rotation_max_deg=±{args.aug_rotation_max_deg}"
        )
    else:
        train_subset = _base_train_subset

    # Optionally wrap with copy-paste augmentation.
    copy_paste_prob = float(args.copy_paste_prob)
    if copy_paste_prob > 0.0:
        # Find positive indices within train_subset (indices relative to train_subset).
        # train_indices[i] is the index into train_dataset; we need indices into train_subset.
        _pos_in_subset = [
            i for i, idx in enumerate(train_indices)
            if train_dataset.samples[idx].boxes.size > 0
        ]
        train_subset = CopyPasteWrapper(
            train_subset,
            positive_indices=_pos_in_subset,
            paste_prob=copy_paste_prob,
            max_pastes=int(args.copy_paste_max_pastes),
        )
        print(
            f"[Info] Copy-paste augmentation enabled | "
            f"paste_prob={copy_paste_prob} | max_pastes={args.copy_paste_max_pastes} | "
            f"donor_pool={len(_pos_in_subset)} positive images"
        )

    if len(train_subset) == 0:
        raise ValueError("Training split is empty after patient-level split and filtering.")
    if len(val_subset) == 0:
        raise ValueError("Validation split is empty. Please check the CSV and splitting logic.")

    # Parse anchor sizes from args.
    # When the user does not pass --anchor-sizes (default "16,32,64,128,256" keeps legacy CLI),
    # we detect that case and pass None to build_model so the COCO-default 9-anchor generator
    # is retained and the COCO head weights load without shape mismatch.
    _raw_anchor_arg = str(args.anchor_sizes).strip()
    _default_anchor_str = "16,32,64,128,256"
    if _raw_anchor_arg == _default_anchor_str:
        anchor_sizes = None   # use torchvision default (9 anchors/location)
    else:
        anchor_sizes = tuple((int(s.strip()),) for s in _raw_anchor_arg.split(",") if s.strip())

    model = build_model(
        num_classes=2,
        anchor_sizes=anchor_sizes,
        fg_iou_thresh=float(args.box_fg_iou_thresh),
        bg_iou_thresh=float(args.box_bg_iou_thresh),
        score_thresh=float(args.box_score_thresh),
        detections_per_img=int(args.box_detections_per_img),
        nms_thresh=float(args.box_nms_thresh),
        focal_loss_alpha=float(args.focal_alpha),
        focal_loss_gamma=float(args.focal_gamma),
        medical_backbone_path=args.medical_backbone_path if args.medical_backbone_path else None,
        input_min_size=int(args.input_min_size),
    )

    # Apply anchor-level Hard Negative Mining patch (core change of bbox-train-A.py).
    # When --hnm-neg-pos-ratio is set, monkey-patches the classification head's
    # compute_loss to limit background anchor count to max(n_fg * ratio, min_neg).
    if args.hnm_neg_pos_ratio is not None:
        _patch_hnm_classification_loss(
            model,
            neg_pos_ratio=float(args.hnm_neg_pos_ratio),
            min_negatives=int(args.hnm_min_negatives),
        )
    else:
        print("[Info] HNM disabled (--hnm-neg-pos-ratio not set); using original focal loss over all anchors.")

    model.to(device)

    # Freeze low-level backbone layers initially if requested
    if int(args.freeze_epochs) > 0:
        freeze_backbone_layers(model)

    optimizer = create_optimizer(model, args, base_lr=float(args.lr))

    # Choose LR scheduler: StepLR if step size provided, else CosineAnnealingLR
    # T_max excludes balanced warmup epochs so cosine decay aligns with actual training phase
    warmup_balanced_epochs = int(args.warmup_balanced_epochs)
    only_use = float(args.only_use)
    _effective_epochs = max(1, int(args.epochs) - warmup_balanced_epochs)
    if int(args.lr_step_size) > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_effective_epochs, eta_min=1e-6)

    history: List[Dict[str, float]] = []

    print(f"Total usable images: {len(usable_indices)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(
        f"Split summary | train images: {train_summary['images']} (pos={train_summary['positive_images']}, neg={train_summary['negative_images']}) "
        f"| val images: {val_summary['images']} (pos={val_summary['positive_images']}, neg={val_summary['negative_images']})"
    )
    print(
        f"Split summary | train patients: {split_summary['train_patients']} | val patients: {split_summary['val_patients']} | "
        f"val_ratio≈{split_summary['val_ratio']}"
    )
    print(
        f"[Info] Image imbalance | train neg/pos={train_neg_to_pos_ratio:.2f} | "
        f"val neg/pos={val_neg_to_pos_ratio:.2f}"
    )
    print(
        f"[Info] Coarse localization mode | breast_crop_region={crop_breast_region} | "
        f"breast_crop_margin={breast_crop_margin:.3f}"
    )
    print(f"Device: {device}")
    warn_on_small_epoch_positive_pool(train_summary, only_use)

    # Prepare balanced sampling warmup (replaces positive-only warmup)
    warmup_pos_epochs = int(args.warmup_positive_epochs)
    _warmup_weights: List[float] = []
    if warmup_balanced_epochs > 0:
        _train_pos_count = sum(1 for i in train_indices if train_dataset.samples[i].boxes.size > 0)
        if _train_pos_count > 0:
            _pos_weight = float(args.warmup_pos_weight_ratio)
            for j in range(len(train_indices)):
                is_pos = train_dataset.samples[train_indices[j]].boxes.size > 0
                _warmup_weights.append(_pos_weight if is_pos else 1.0)
            print(f"[Info] Balanced warmup: {warmup_balanced_epochs} epochs, pos_weight_ratio={_pos_weight}, train_size={len(train_indices)}")
            _cls_loss_scale = float(args.classification_loss_scale)
            if _cls_loss_scale != 1.0:
                print(f"[Info] classification_loss_scale={_cls_loss_scale} (classification loss will be multiplied by this factor)")
        else:
            warmup_balanced_epochs = 0
            print("[Warning] No positive training samples; disabling balanced warmup")
    # Legacy positive-only warmup subset (kept for backward compat but not recommended)
    warmup_train_subset = None
    if warmup_pos_epochs > 0 and warmup_balanced_epochs == 0 and not args.positive_only:
        _pos_train_idx = [i for i in train_indices if train_dataset.samples[i].boxes.size > 0]
        if _pos_train_idx:
            warmup_train_subset = Subset(train_dataset, _pos_train_idx)
            print(f"[Info] Positive-only warmup: {len(_pos_train_idx)} images for first {warmup_pos_epochs} epochs")
        else:
            warmup_pos_epochs = 0
            print("[Warning] No positive training samples; disabling positive-only warmup")

    best_val_f1 = -float("inf")
    best_recall_at_low_thresh = -float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(int(args.epochs)):
        print(f"\n{'=' * 60} Epoch {epoch + 1}/{int(args.epochs)} start {'=' * 60}")

        # --- Phase detection ---
        is_warmup_balanced = warmup_balanced_epochs > 0 and epoch < warmup_balanced_epochs
        _is_legacy_warmup = (not is_warmup_balanced) and warmup_train_subset is not None and epoch < warmup_pos_epochs

        if is_warmup_balanced and epoch == 0:
            print(f"[Warmup] Starting balanced sampling warmup phase ({warmup_balanced_epochs} epochs)")
        elif _is_legacy_warmup and epoch == 0:
            print(f"[Warmup] Starting positive-only warmup phase ({warmup_pos_epochs} epochs)")

        # --- Reset optimizer/scheduler when transitioning from balanced warmup to full training ---
        rebuilt_warmup = False
        if not is_warmup_balanced and epoch == warmup_balanced_epochs and warmup_balanced_epochs > 0:
            print(f"[Info] Balanced warmup complete, resetting optimizer and switching to full training ({len(train_subset)} images)")
            _post_warmup_lr = float(args.post_warmup_lr) if args.post_warmup_lr is not None else float(args.lr)
            if args.post_warmup_lr is not None:
                print(f"[Info] post-warmup LR set to {_post_warmup_lr} (from --post-warmup-lr)")
            optimizer = create_optimizer(model, args, base_lr=_post_warmup_lr)
            remaining = max(1, int(args.epochs) - epoch)
            if int(args.lr_step_size) > 0:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
            else:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
            rebuilt_warmup = True
        elif not _is_legacy_warmup and epoch == warmup_pos_epochs and warmup_train_subset is not None:
            print(f"[Info] Warmup complete, switching to full training ({len(train_subset)} images)")

        # --- Data selection ---
        if is_warmup_balanced:
            # Balanced warmup: use WeightedRandomSampler on the full training set
            sampler = WeightedRandomSampler(_warmup_weights, num_samples=len(train_indices), replacement=True)
            train_loader = DataLoader(
                train_subset,
                batch_size=int(args.batch_size),
                sampler=sampler,
                num_workers=int(args.num_workers),
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        elif _is_legacy_warmup:
            # Legacy positive-only warmup
            train_loader = DataLoader(
                warmup_train_subset,
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            # Normal training, possibly with --only-use subsampling
            if only_use < 1.0:
                _epoch_offset = epoch - max(warmup_balanced_epochs, warmup_pos_epochs if warmup_train_subset else 0)
                epoch_indices = select_epoch_subset(
                    train_indices, train_dataset.samples, _epoch_offset, only_use, int(args.seed))
                epoch_subset = Subset(train_dataset, epoch_indices)
                if _epoch_offset == 0 or (_epoch_offset % 5 == 0):
                    _ep_pos = sum(1 for i in epoch_indices if train_dataset.samples[i].boxes.size > 0)
                    _ep_neg = len(epoch_indices) - _ep_pos
                    print(f"[Info] Epoch {epoch+1} subset: {len(epoch_indices)} images (pos={_ep_pos}, neg={_ep_neg})")
            else:
                epoch_subset = train_subset

            # Full-training weighted sampler: keep a mild positive-sample bias
            # to prevent the model from collapsing to "predict nothing".
            _full_train_pos_ratio = float(args.full_train_pos_weight_ratio)
            if _full_train_pos_ratio > 0.0:
                # Build per-sample weights based on the current epoch_subset.
                # epoch_subset may be a TrainAugmentWrapper, a Subset, or another Subset;
                # walk down to the underlying train_dataset.samples for the box check.
                _epoch_indices: List[int]
                if hasattr(epoch_subset, "dataset") and hasattr(epoch_subset.dataset, "indices"):
                    # TrainAugmentWrapper -> Subset -> train_dataset
                    _epoch_indices = list(epoch_subset.dataset.indices)  # type: ignore[union-attr]
                elif hasattr(epoch_subset, "indices"):
                    # plain Subset
                    _epoch_indices = list(epoch_subset.indices)  # type: ignore[union-attr]
                else:
                    _epoch_indices = list(range(len(epoch_subset)))
                _ft_weights = [
                    _full_train_pos_ratio if train_dataset.samples[i].boxes.size > 0 else 1.0
                    for i in _epoch_indices
                ]
                _ft_sampler = WeightedRandomSampler(
                    _ft_weights,
                    num_samples=len(_epoch_indices),
                    replacement=True,
                )
                train_loader = DataLoader(
                    epoch_subset,
                    batch_size=int(args.batch_size),
                    sampler=_ft_sampler,
                    num_workers=int(args.num_workers),
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                train_loader = DataLoader(
                    epoch_subset,
                    batch_size=int(args.batch_size),
                    shuffle=True,
                    num_workers=int(args.num_workers),
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )

        # Validation loader must not shuffle and must not use any sampling tricks.
        val_loader = DataLoader(
            val_subset,
            batch_size=max(1, int(args.val_batch_size)),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        # decide accumulation steps (honor --fuck-running / --accumulation-steps)
        accumulation_steps = 1 if args.fuck_running else max(1, int(args.accumulation_steps))

        # iter-level linear warmup only for first epoch (use same heuristic as train-abc.py)
        warmup_scheduler = None
        if epoch == 0:
            warmup_iters = min(1000, len(train_loader) - 1)
            if warmup_iters > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)

        avg_loss, optimizer_steps, avg_sublosses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            int(args.epochs),
            accumulation_steps,
            warmup_scheduler,
            disable_tqdm=args.hide_progress_bar,
            classification_loss_scale=float(args.classification_loss_scale),
        )

        # Unfreeze backbone after configured freeze epochs
        rebuilt_this_epoch = rebuilt_warmup

        if int(args.freeze_epochs) > 0 and (epoch + 1) == int(args.freeze_epochs):
            print(f"[Info] Unfreezing backbone layers after {args.freeze_epochs} epochs and rebuilding optimizer")
            unfreeze_backbone_layers(model)
            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.lr)
            optimizer = create_optimizer(model, args, base_lr=current_lr)
            if int(args.lr_step_size) > 0:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
            else:
                # After unfreezing, schedule for remaining epochs (warmup handled at iter-level)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) - epoch - 1), eta_min=1e-6)
            rebuilt_this_epoch = True

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=float(args.val_score_threshold),
            iou_threshold=float(args.val_iou_threshold),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=args.hide_progress_bar,
            val_detections_per_img=args.val_detections_per_img,  # None means no override (use model's current value)
        )

        # Only step the epoch-level scheduler if we actually performed any optimizer.step()
        # Skip scheduler stepping during balanced warmup phase
        if is_warmup_balanced:
            print(f"[Info] Warmup epoch {epoch + 1}: skipping lr_scheduler.step()")
        elif rebuilt_this_epoch:
            # optimizer/scheduler was just rebuilt this epoch — do not step (this is normal)
            print(f"[Info] Epoch {epoch + 1}: optimizer/scheduler rebuilt this epoch, skipping lr_scheduler.step()")
        elif optimizer_steps > 0:
            print(f"[Info] lr_scheduler.step() at epoch {epoch + 1}.")
            lr_scheduler.step()
        else:
            print(f"[Warning] No optimizer.step() executed in epoch {epoch + 1}; skipping lr_scheduler.step() to avoid PyTorch warning.")

        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "classification": float(avg_sublosses.get("classification", 0.0)),
            "bbox_regression": float(avg_sublosses.get("bbox_regression", 0.0)),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_best_thresh_f1": float(val_metrics.get("best_thresh_f1", val_metrics["f1"])),
            "val_best_thresh": float(val_metrics.get("best_thresh", float(args.val_score_threshold))),
            "val_tp": float(val_metrics["tp"]),
            "val_fp": float(val_metrics["fp"]),
            "val_fn": float(val_metrics["fn"]),
            "val_raw_preds": float(val_metrics.get("raw_preds", 0)),
            "val_avg_raw_preds_per_image": float(val_metrics.get("avg_raw_preds_per_image", 0)),
            "val_recall_at_01": float(val_metrics.get("tp@0.1", val_metrics["tp"])) / max(float(val_metrics.get("gt_boxes", 1)), 1),
            "val_recall_at_03": float(val_metrics.get("tp@0.3", 0)) / max(float(val_metrics.get("gt_boxes", 1)), 1),
            "val_recall_at_05": float(val_metrics.get("tp@0.5", 0)) / max(float(val_metrics.get("gt_boxes", 1)), 1),
        }
        history.append(record)

        # Use best-threshold F1 for checkpoint selection so we don't miss epochs
        # where the model is better at a threshold other than val_score_threshold.
        current_f1 = float(val_metrics.get("best_thresh_f1", val_metrics["f1"]))
        # Recall-first: also track recall@0.1 (TP@0.1 / total GT boxes).
        # When --recall-stop is set, early stopping is driven by recall@0.1 instead of F1.
        gt_boxes_total = float(val_metrics.get("gt_boxes", max(val_metrics["tp"] + val_metrics["fn"], 1)))
        recall_at_01 = float(val_metrics.get("tp@0.1", val_metrics["tp"])) / max(gt_boxes_total, 1)
        current_stop_metric = recall_at_01 if args.recall_stop else current_f1
        best_stop_metric = best_recall_at_low_thresh if args.recall_stop else best_val_f1
        improved = current_stop_metric > (best_stop_metric + float(args.min_delta))

        if improved:
            best_val_f1 = current_f1
            best_recall_at_low_thresh = recall_at_01
            best_epoch = epoch + 1
            no_improve_epochs = 0

            meta = {
                "task": "bbox_detection",
                "num_classes": 2,
                "class_names": ["background", "lesion"],
                "csv_path": str(csv_path),
                "images_root": str(images_root),
                "positive_only": bool(args.positive_only),
                "history": history,
                "torchvision_model": "retinanet_resnet50_fpn_v2",
                "anchor_sizes": str(args.anchor_sizes),
                "focal_loss_alpha": float(args.focal_alpha),
                "focal_loss_gamma": float(args.focal_gamma),
                "val_ratio": float(args.val_ratio),
                "val_score_threshold": float(args.val_score_threshold),
                "val_iou_threshold": float(args.val_iou_threshold),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "best_epoch": int(best_epoch),
                "best_val_precision": float(val_metrics["precision"]),
                "best_val_recall": float(val_metrics["recall"]),
                "best_val_f1": float(val_metrics["f1"]),
                "best_val_recall_at_01": float(recall_at_01),
                "crop_breast_region": bool(crop_breast_region),
                "breast_crop_margin": float(breast_crop_margin),
                "box_fg_iou_thresh": float(args.box_fg_iou_thresh),
                "box_bg_iou_thresh": float(args.box_bg_iou_thresh),
                "warmup_positive_epochs": int(args.warmup_positive_epochs),
                "warmup_balanced_epochs": int(warmup_balanced_epochs),
                "only_use": float(only_use),
                "train_neg_to_pos_ratio": float(train_neg_to_pos_ratio),
                "val_neg_to_pos_ratio": float(val_neg_to_pos_ratio),
                "box_score_thresh": float(args.box_score_thresh),
                "box_nms_thresh": float(args.box_nms_thresh),
                "box_detections_per_img": int(args.box_detections_per_img),
                "input_min_size": int(args.input_min_size),
                "split_summary": split_summary,
            }
            # Save the best checkpoint, not the last one.
            save_checkpoint(save_path, model, meta)
            print(f"[Info] Saved best checkpoint to: {save_path}")
        else:
            no_improve_epochs += 1

        print(
            f"[Milestone] Epoch {epoch + 1:03d}/{int(args.epochs):03d} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_F1={val_metrics['f1']:.4f} | "
            f"val_BestThreshF1={val_metrics.get('best_thresh_f1', val_metrics['f1']):.4f}@{val_metrics.get('best_thresh', args.val_score_threshold)} | "
            f"lr={record['lr']:.6f}"
        )
        print(
            (
                f"  Train sub-losses: classification={record.get('classification', 0.0):.6f}, "
                f"bbox_regression={record.get('bbox_regression', 0.0):.6f}"
            )
        )
        # Recall-at-threshold summary: shows how many GT lesions are found at each threshold.
        # TP / GT boxes = recall. Goal: maximise TP@0.1 (recall with lenient threshold).
        gt_total = int(val_metrics.get("gt_boxes", 0))
        tp_01 = int(val_metrics.get("tp@0.1", 0))
        tp_03 = int(val_metrics.get("tp@0.3", 0))
        tp_05 = int(val_metrics.get("tp@0.5", 0))
        recall_01 = tp_01 / max(gt_total, 1)
        recall_03 = tp_03 / max(gt_total, 1)
        recall_05 = tp_05 / max(gt_total, 1)
        stop_label = "recall@0.1" if args.recall_stop else "BestThreshF1"
        best_stop_val = best_recall_at_low_thresh if args.recall_stop else best_val_f1
        print(
            f"  [Recall] GT_boxes={gt_total} | "
            f"Recall@0.1={recall_01:.3f}({tp_01}/{gt_total}) | "
            f"Recall@0.3={recall_03:.3f}({tp_03}/{gt_total}) | "
            f"Recall@0.5={recall_05:.3f}({tp_05}/{gt_total})"
        )
        print(
            f"  Val counts: TP={int(val_metrics['tp'])}, FP={int(val_metrics['fp'])}, FN={int(val_metrics['fn'])} | "
            f"best_{stop_label}={best_stop_val:.4f} (epoch {best_epoch})"
        )

        if int(args.patience) > 0 and no_improve_epochs >= int(args.patience):
            print(
                f"[EarlyStopping] Stop metric ({stop_label}) has not improved for "
                f"{int(args.patience)} consecutive epochs. Stopping at epoch {epoch + 1}."
            )
            break

    final_meta = {
        "task": "bbox_detection",
        "num_classes": 2,
        "class_names": ["background", "lesion"],
        "csv_path": str(csv_path),
        "images_root": str(images_root),
        "positive_only": bool(args.positive_only),
        "history": history,
        "torchvision_model": "retinanet_resnet50_fpn_v2",
        "anchor_sizes": str(args.anchor_sizes),
        "focal_loss_alpha": float(args.focal_alpha),
        "focal_loss_gamma": float(args.focal_gamma),
        "val_ratio": float(args.val_ratio),
        "val_score_threshold": float(args.val_score_threshold),
        "val_iou_threshold": float(args.val_iou_threshold),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1 if best_val_f1 != -float("inf") else 0.0),
        "best_val_recall_at_01": float(best_recall_at_low_thresh if best_recall_at_low_thresh != -float("inf") else 0.0),
        "recall_stop": bool(args.recall_stop),
        "crop_breast_region": bool(crop_breast_region),
        "breast_crop_margin": float(breast_crop_margin),
        "box_fg_iou_thresh": float(args.box_fg_iou_thresh),
        "box_bg_iou_thresh": float(args.box_bg_iou_thresh),
        "warmup_positive_epochs": int(args.warmup_positive_epochs),
        "warmup_balanced_epochs": int(warmup_balanced_epochs),
        "only_use": float(only_use),
        "train_neg_to_pos_ratio": float(train_neg_to_pos_ratio),
        "val_neg_to_pos_ratio": float(val_neg_to_pos_ratio),
        "box_score_thresh": float(args.box_score_thresh),
        "box_nms_thresh": float(args.box_nms_thresh),
        "box_detections_per_img": int(args.box_detections_per_img),
        "input_min_size": int(args.input_min_size),
        "split_summary": split_summary,
    }

    print(f"Best checkpoint saved at: {save_path}")
    print(f"Best epoch: {best_epoch}, best val_F1: {final_meta['best_val_f1']:.4f}, best Recall@0.1: {final_meta['best_val_recall_at_01']:.4f}")
    print(json.dumps(final_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start time:  {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None):
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
        end_time = time.time()
        if reason:
            print(reason)
        print(f"Exit time:   {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running time: {end_time - start_time:.2f} s.")

    def _handle_signal(signum, _frame):
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"Received {signame}")

    atexit.register(_on_exit)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        main()
    except KeyboardInterrupt as exc:
        reason = "[Info] Training interrupted by user (Ctrl+C)."
        if exc.args and exc.args[0]:
            reason = f"[Info] {exc.args[0]}. Training interrupted."
        _on_exit(reason)
        raise SystemExit(130)
