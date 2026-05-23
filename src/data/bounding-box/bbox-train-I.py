r"""
方向 I：双侧对比检测（Bilateral Multi-View Fusion）

核心思想：对每张乳腺图（primary），将其对侧同视角图（contralateral, same view）
作为参照，将三通道 [primary, contra, |primary-contra|] 拼合后送入 RetinaNet。
病变侧的解剖差异在差分通道中被放大，为模型提供"哪一侧不对称"的明确语义线索。

标准坐标系（Canonical Left-Facing Space）：
  - 左侧 (L) 图像：不翻转，乳头朝右，胸壁在左侧
  - 右侧 (R) 图像：水平翻转，使乳头朝右 → 与左侧坐标一致
  - 因此 |primary-contra| 在解剖上对齐，差异真正反映不对称性
  - 训练时不使用水平翻转增强（会破坏双侧语义）

conv1 初始化策略：
  - primary / contra 通道：继承 RadImageNet 均值权重（与 H 一致）
  - diff 通道：0.1× 初始权重（较小，让模型缓慢学习如何利用差分信号）

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-I.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --input-h 1024 \
    --input-w 512 \
    --patience 10 \
    --monitor-metric fbeta2_ref \
    --medical-backbone-path models/raw/ResNet50.pt \
    --save-path models/bbox_resnet50.I.pth \
    --augment \
    --aug-contrast-range 0.8 1.2 \
    --aug-scale-min 0.85 \
    --focal-alpha 0.25 \
    --min-box-side 24.0 \
    --max-box-ar 3.0 \
    --cliff-patience-ratio 0.6 \
    --hide-progress-bar

─────────────────────────────────────────────────────────────────────────────
改版历史
─────────────────────────────────────────────────────────────────────────────

rec_49（初版）
  - 基于方向 H（bbox-train-H.py upd_6）重构
  - 核心改动：
    · BilateralSample 替换 Sample：新增 contra_path / laterality / view 字段
    · load_bilateral_samples() 负责构建每张图的对侧路径映射
    · BilateralDetectionDataset：加载双图 → 标准化坐标系 → 3 通道拼合
    · build_retinanet_bilateral()：conv1 改为 3 通道，diff 通道权重 0.1×
    · validate_bilateral()：推理时同样构造双侧 3 通道输入
    · 删除 --aug-hflip-prob 和 --aug-brightness-delta（不适用双侧对比场景）
  - 训练样本数：≈16000（每张原始图作为一次 primary，与 H 相同量级）
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


def load_gray_clahe(path: Path) -> Optional[np.ndarray]:
    """Load an image, convert to grayscale, apply CLAHE. Returns [H, W] uint8 or None if failed."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
    except Exception:
        return None

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def _flip_boxes_h(boxes: np.ndarray, width: int) -> np.ndarray:
    """Flip boxes horizontally. boxes: (N, 4) xyxy."""
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def _pad_gray_to_target_ar(
    gray: np.ndarray,
    boxes: Optional[np.ndarray],
    target_ar: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """AR-preserving pad for a single-channel (H, W) grayscale image.

    Pads with zeros so that H/W == target_ar, matching the AR-pad logic in H.
    Returns (padded_gray, padded_boxes).
    """
    orig_h, orig_w = gray.shape[:2]
    actual_ar = orig_h / max(orig_w, 1)
    if actual_ar < target_ar - 1e-6:  # too wide → pad height
        padded_h = int(round(orig_w * target_ar))
        pad_top = (padded_h - orig_h) // 2
        pad_bottom = padded_h - orig_h - pad_top
        gray = np.pad(gray, ((pad_top, pad_bottom), (0, 0)), constant_values=0)
        if boxes is not None and boxes.size > 0:
            boxes = boxes.copy()
            boxes[:, 1] += pad_top
            boxes[:, 3] += pad_top
    elif actual_ar > target_ar + 1e-6:  # too tall → pad width
        padded_w = int(round(orig_h / target_ar))
        pad_left = (padded_w - orig_w) // 2
        pad_right = padded_w - orig_w - pad_left
        gray = np.pad(gray, ((0, 0), (pad_left, pad_right)), constant_values=0)
        if boxes is not None and boxes.size > 0:
            boxes = boxes.copy()
            boxes[:, 0] += pad_left
            boxes[:, 2] += pad_left
    return gray, boxes


def compose_bilateral_tensor(p_gray: np.ndarray, c_gray: np.ndarray) -> torch.Tensor:
    """Compose 3-channel bilateral tensor [primary, contra, |diff|].

    Both inputs must be [H, W] uint8 grayscale already resized to the same shape.
    Returns [3, H, W] float32 in [0, 1].
    """
    p = p_gray.astype(np.float32) / 255.0
    c = c_gray.astype(np.float32) / 255.0
    d = np.abs(p - c)
    arr = np.stack([p, c, d], axis=0)  # [3, H, W]
    return torch.from_numpy(arr).contiguous()


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class BilateralSample:
    patient_id: str
    image_id: str           # primary image ID
    laterality: str         # 'L' or 'R' (primary side)
    view: str               # 'CC' or 'MLO'
    image_path: Path        # primary image path
    contra_path: Path       # contralateral same-view image path
    boxes: np.ndarray       # GT boxes in primary's ORIGINAL (pre-flip) coords, (N, 4) xyxy
    orig_size: Tuple[float, float]  # (H, W) of primary


def load_bilateral_samples(
    csv_path: Path,
    images_root: Path,
    split_name: str,
    lesion_types: Optional[List[str]] = None,
    min_box_side: float = 0.0,
    max_box_ar: float = float("inf"),
    input_w: int = 512,
) -> List[BilateralSample]:
    """Load samples with contralateral image paths for bilateral comparison.

    For each image in the given split, finds its contralateral same-view image
    within the same patient. Images without a valid contralateral are skipped
    with a warning.

    Returns one BilateralSample per primary image (~16000 for training split).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    # Build contra lookup: patient_id → {(laterality, view): image_id}
    # Use first row per (patient_id, series_id, image_id) group for metadata.
    image_meta: Dict[str, Dict[Tuple[str, str], str]] = {}  # pid → {(lat, view): iid}
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        first = group.iloc[0]
        lat = str(first.get("laterality", "")).strip().upper()
        view_raw = str(first.get("view_position", "")).strip().upper()
        if lat not in ("L", "R") or view_raw not in ("CC", "MLO"):
            continue
        pid = str(patient_id)
        if pid not in image_meta:
            image_meta[pid] = {}
        key = (lat, view_raw)
        if key not in image_meta[pid]:
            image_meta[pid][key] = str(image_id)

    samples: List[BilateralSample] = []
    skipped_no_contra = 0

    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        pid = str(patient_id)
        iid = str(image_id)
        first = group.iloc[0]

        lat = str(first.get("laterality", "")).strip().upper()
        view_raw = str(first.get("view_position", "")).strip().upper()
        if lat not in ("L", "R") or view_raw not in ("CC", "MLO"):
            continue

        contra_lat = "R" if lat == "L" else "L"
        contra_key = (contra_lat, view_raw)
        pid_meta = image_meta.get(pid, {})
        if contra_key not in pid_meta:
            skipped_no_contra += 1
            continue
        contra_iid = pid_meta[contra_key]

        # Filter GT boxes by lesion type
        grp = group.copy()
        if lesion_types:
            type_mask = pd.Series(False, index=grp.index)
            for lt in lesion_types:
                if lt in grp.columns:
                    type_mask = type_mask | (grp[lt] == 1)
            grp = grp[type_mask]

        valid = grp[["xmin", "ymin", "xmax", "ymax"]].dropna()
        boxes: np.ndarray = valid.to_numpy(dtype=np.float32) if not valid.empty else np.zeros((0, 4), dtype=np.float32)

        if boxes.size > 0:
            invalid = int(np.sum((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])))
            if invalid > 0:
                print(f"[Warning] {invalid} invalid box(es) in {pid}/{iid}")

        # Detectability filter (in resized-image-space coordinates)
        if boxes.size > 0 and (min_box_side > 0.0 or max_box_ar < float("inf")):
            orig_w_val = float(first["width"]) if pd.notna(first.get("width")) else float(input_w)
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

        orig_h = float(first["height"]) if pd.notna(first.get("height")) else 0.0
        orig_w = float(first["width"]) if pd.notna(first.get("width")) else 0.0

        image_path = images_root / pid / iid
        contra_path = images_root / pid / contra_iid

        samples.append(BilateralSample(
            patient_id=pid,
            image_id=iid,
            laterality=lat,
            view=view_raw,
            image_path=image_path,
            contra_path=contra_path,
            boxes=boxes,
            orig_size=(orig_h, orig_w),
        ))

    if skipped_no_contra > 0:
        print(f"[Warning] Skipped {skipped_no_contra} images with no contralateral same-view image.")

    return samples


def patient_level_split(
    samples: List[BilateralSample],
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

class BilateralDetectionDataset(Dataset):
    """Bilateral full-image detection dataset for torchvision RetinaNet.

    Each item is a 3-channel tensor [primary_canonical, contra_canonical, |diff|]
    in the canonical left-facing coordinate system:
      - L primary: no horizontal flip
      - R primary: flip horizontally (and flip boxes accordingly)

    The contralateral image is loaded, AR-padded, resized, and flipped to the
    same canonical orientation before computing the difference channel.

    Returns (img_tensor [3, H, W], target_dict) where target_dict contains
    'boxes' [N, 4] (xyxy in canonical coords) and 'labels' [N].
    """

    def __init__(
        self,
        samples: List[BilateralSample],
        indices: List[int],
        input_h: int,
        input_w: int,
        augment: bool = False,
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
        self.aug_contrast_min = aug_contrast_min
        self.aug_contrast_max = aug_contrast_max
        self.aug_scale_min = aug_scale_min
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_and_prepare_gray(
        self,
        path: Path,
        target_ar: float,
        boxes: Optional[np.ndarray] = None,
        flip_h: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image, AR-pad, and optionally flip horizontally.

        Returns (gray [H_padded, W_padded], transformed_boxes) or (None, None) if failed.
        boxes are transformed only if not None.
        """
        gray = load_gray_clahe(path)
        if gray is None:
            return None, None

        orig_h, orig_w = gray.shape[:2]

        # Clip boxes before padding
        if boxes is not None and boxes.size > 0:
            boxes = boxes.copy()
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # AR-preserving pad
        gray, boxes = _pad_gray_to_target_ar(gray, boxes, target_ar)
        padded_h, padded_w = gray.shape[:2]

        # Horizontal flip to canonical left-facing space
        if flip_h:
            gray = gray[:, ::-1].copy()
            if boxes is not None and boxes.size > 0:
                boxes = _flip_boxes_h(boxes, padded_w)

        return gray, boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[self.indices[idx]]

        target_ar = self.input_h / max(self.input_w, 1)
        is_right = (sample.laterality == "R")

        # --- Load primary ---
        p_gray, boxes = self._load_and_prepare_gray(
            sample.image_path,
            target_ar=target_ar,
            boxes=sample.boxes.copy(),
            flip_h=is_right,  # R → flip to canonical
        )

        if p_gray is None:
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            target: Dict[str, torch.Tensor] = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            return img_t, target

        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # --- Load contra ---
        # Contra is the opposite side: L contra for R primary (already canonical),
        # R contra for L primary (needs flip).
        contra_flip = not is_right  # contra is R when primary is L → flip contra
        c_gray, _ = self._load_and_prepare_gray(
            sample.contra_path,
            target_ar=target_ar,
            boxes=None,
            flip_h=contra_flip,
        )
        if c_gray is None:
            # Fall back to a black image for contralateral if not available
            c_gray = np.zeros_like(p_gray)

        padded_h, padded_w = p_gray.shape[:2]

        # --- Resize both to input_h × input_w ---
        p_resized = cv2.resize(p_gray, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        c_resized = cv2.resize(c_gray, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # --- Scale boxes ---
        scale_x = self.input_w / max(padded_w, 1)
        scale_y = self.input_h / max(padded_h, 1)
        if boxes.size > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # --- Augmentation (no hflip: would break bilateral semantics) ---
        if self.augment:
            if self.aug_contrast_max > self.aug_contrast_min + 1e-6:
                factor = self.aug_contrast_min + self.rng.random() * (
                    self.aug_contrast_max - self.aug_contrast_min
                )
                # Apply same contrast factor to both channels so the diff channel
                # remains meaningful (|f·p - f·c| = f·|p - c|).
                for arr in (p_resized, c_resized):
                    mean_val = float(arr.mean())
                    arr[:] = np.clip(
                        mean_val + (arr.astype(np.float32) - mean_val) * factor, 0, 255
                    ).astype(np.uint8)
            if self.aug_scale_min < 1.0 - 1e-6:
                scale = self.aug_scale_min + self.rng.random() * (1.0 - self.aug_scale_min)
                if scale < 1.0 - 1e-6:
                    scaled_h = max(int(self.input_h * scale), 1)
                    scaled_w = max(int(self.input_w * scale), 1)
                    pad_top_zo = (self.input_h - scaled_h) // 2
                    pad_left_zo = (self.input_w - scaled_w) // 2

                    p_zoomed = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
                    p_small = cv2.resize(p_resized, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    p_zoomed[pad_top_zo:pad_top_zo + scaled_h, pad_left_zo:pad_left_zo + scaled_w] = p_small
                    p_resized = p_zoomed

                    c_zoomed = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
                    c_small = cv2.resize(c_resized, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    c_zoomed[pad_top_zo:pad_top_zo + scaled_h, pad_left_zo:pad_left_zo + scaled_w] = c_small
                    c_resized = c_zoomed

                    if boxes.size > 0:
                        boxes[:, 0] = boxes[:, 0] * scale + pad_left_zo
                        boxes[:, 2] = boxes[:, 2] * scale + pad_left_zo
                        boxes[:, 1] = boxes[:, 1] * scale + pad_top_zo
                        boxes[:, 3] = boxes[:, 3] * scale + pad_top_zo
                        keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                        boxes = boxes[keep]

        img_t = compose_bilateral_tensor(p_resized, c_resized)  # [3, H, W] float32 in [0, 1]

        if boxes.size > 0:
            target = {
                "boxes": torch.from_numpy(boxes.astype(np.float32)),
                "labels": torch.zeros(boxes.shape[0], dtype=torch.int64),
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
    samples: List[BilateralSample],
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

def build_retinanet_bilateral(
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
    """Build a RetinaNet with ResNet50-FPN backbone for 3-channel bilateral input.

    conv1 initialization:
      - channels 0 (primary) and 1 (contra): inherit the channel-averaged
        RadImageNet weights (same as direction H, each channel sees single-image signal).
      - channel 2 (|diff|): initialized at 0.1× weight to let the model
        gradually learn how to exploit the difference signal.
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

    # Adapt conv1 for 3-channel bilateral input.
    # Primary (ch0) and contra (ch1) get the grayscale-averaged weight;
    # diff (ch2) gets a 0.1× scaled weight to start with a small contribution.
    try:
        with torch.no_grad():
            mean_w = backbone.body.conv1.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            new_w = mean_w.expand(-1, 3, -1, -1).clone()                   # [64, 3, 7, 7]
            new_w[:, 2, :, :].mul_(0.1)                                     # diff channel: 0.1×
            backbone.body.conv1.weight.copy_(new_w)
        print("[Info] conv1 adapted for bilateral 3-channel input (primary/contra/diff).")
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

def validate_bilateral(
    model: "RetinaNet",
    samples: List[BilateralSample],
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
    """Validate with bilateral RetinaNet inference at multiple score thresholds."""
    model.eval()

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

    target_ar = input_h / max(input_w, 1)

    pbar = tqdm(val_indices, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for sample_idx in pbar:
            sample = samples[sample_idx]
            is_right = (sample.laterality == "R")

            # Load primary
            p_gray, gt_boxes = _load_val_bilateral(
                primary_path=sample.image_path,
                contra_path=sample.contra_path,
                boxes=sample.boxes.astype(np.float32).copy(),
                target_ar=target_ar,
                input_h=input_h,
                input_w=input_w,
                is_right=is_right,
            )
            if p_gray is None:
                continue

            if gt_boxes is None:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)

            # Load contra
            contra_flip = not is_right
            c_gray_raw = load_gray_clahe(sample.contra_path)
            if c_gray_raw is not None:
                c_gray_raw, _ = _pad_gray_to_target_ar(c_gray_raw, None, target_ar)
                if contra_flip:
                    c_gray_raw = c_gray_raw[:, ::-1].copy()
                c_resized = cv2.resize(c_gray_raw, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            else:
                c_resized = np.zeros((input_h, input_w), dtype=np.uint8)

            img_t = compose_bilateral_tensor(p_gray, c_resized).to(device)

            outputs = model([img_t])
            pred_boxes = outputs[0]["boxes"].cpu().numpy()
            pred_scores = outputs[0]["scores"].cpu().numpy()

            for thresh in score_thresholds:
                mask = pred_scores >= thresh
                filtered_boxes = pred_boxes[mask]
                tp, fp, fn = compute_iou_matches(filtered_boxes, gt_boxes, iou_threshold)
                stats[thresh]["tp"] += tp
                stats[thresh]["fp"] += fp
                stats[thresh]["fn"] += fn

            total_gt_boxes += int(gt_boxes.shape[0]) if gt_boxes.size > 0 else 0

    model.score_thresh = orig_score_thresh
    model.nms_thresh = orig_nms_thresh
    model.detections_per_img = orig_det_per_img

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
    best_recall_thresh: float = 0.3
    best_recall = recall_per_thresh[best_recall_thresh]
    best_fbeta2_thresh = max(fbeta2_per_thresh, key=lambda t: fbeta2_per_thresh[t])
    best_fbeta2 = fbeta2_per_thresh[best_fbeta2_thresh]
    ref_fbeta2_thresh: float = 0.3
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


def _load_val_bilateral(
    primary_path: Path,
    contra_path: Path,
    boxes: np.ndarray,
    target_ar: float,
    input_h: int,
    input_w: int,
    is_right: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and preprocess primary image + GT boxes for validation.

    Returns (resized_gray [H, W] uint8, transformed_gt_boxes) or (None, None) if failed.
    Boxes are returned in canonical (possibly flipped) resized coordinates.
    """
    p_gray = load_gray_clahe(primary_path)
    if p_gray is None:
        return None, None

    orig_h, orig_w = p_gray.shape[:2]

    gt_boxes = boxes.copy()

    # Clip boxes
    if gt_boxes.size > 0:
        gt_boxes[:, 0] = np.clip(gt_boxes[:, 0], 0, orig_w - 1)
        gt_boxes[:, 2] = np.clip(gt_boxes[:, 2], 0, orig_w - 1)
        gt_boxes[:, 1] = np.clip(gt_boxes[:, 1], 0, orig_h - 1)
        gt_boxes[:, 3] = np.clip(gt_boxes[:, 3], 0, orig_h - 1)
        keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
        gt_boxes = gt_boxes[keep]

    # AR pad
    p_gray, gt_boxes = _pad_gray_to_target_ar(p_gray, gt_boxes if gt_boxes.size > 0 else None, target_ar)
    if gt_boxes is None:
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
    padded_h, padded_w = p_gray.shape[:2]

    # Flip R → canonical
    if is_right:
        p_gray = p_gray[:, ::-1].copy()
        if gt_boxes.size > 0:
            gt_boxes = _flip_boxes_h(gt_boxes, padded_w)

    # Scale boxes
    scale_x = input_w / max(padded_w, 1)
    scale_y = input_h / max(padded_h, 1)
    if gt_boxes.size > 0:
        gt_boxes[:, 0] *= scale_x
        gt_boxes[:, 2] *= scale_x
        gt_boxes[:, 1] *= scale_y
        gt_boxes[:, 3] *= scale_y
        keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
        gt_boxes = gt_boxes[keep]

    p_resized = cv2.resize(p_gray, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    return p_resized, gt_boxes


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
        description="Train a bilateral RetinaNet-ResNet50-FPN for VinDr lesion detection (Direction I)."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--input-h", type=int, default=1024)
    parser.add_argument("--input-w", type=int, default=512)
    parser.add_argument("--val-iou-threshold", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cliff-patience-ratio", type=float, default=0.0,
                        help="Cliff-aware patience ratio (same semantics as direction H). "
                             "0 = disabled (default). Recommended: 0.6.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--augment", action="store_true",
                        help="Enable training augmentation (contrast jitter + zoom-out). "
                             "Note: horizontal flip augmentation is intentionally disabled "
                             "to preserve bilateral symmetry semantics.")
    parser.add_argument("--aug-contrast-range", type=float, nargs=2, default=[1.0, 1.0],
                        metavar=("MIN", "MAX"),
                        help="Contrast jitter range. Same factor applied to primary and "
                             "contra so |diff| stays proportional. Default: 1.0 1.0 (disabled).")
    parser.add_argument("--aug-scale-min", type=float, default=1.0,
                        help="Minimum zoom-out scale. 1.0 = disabled (default).")
    parser.add_argument("--medical-backbone-path", type=Path, default=None)
    parser.add_argument("--pos-oversample-factor", type=float, default=4.0)
    parser.add_argument("--anchor-sizes", type=str, default="32,64,128,256,512")
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--monitor-metric", type=str, default="fbeta2",
                        choices=["f1", "recall", "fbeta2", "fbeta2_ref"])
    parser.add_argument("--hide-progress-bar", action="store_true")
    parser.add_argument("--lesion-types", type=str, default=None)
    parser.add_argument("--min-box-side", type=float, default=0.0)
    parser.add_argument("--max-box-ar", type=float, default=float("inf"))
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
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.I.pth"

    print(f"Start time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Direction:     I (bilateral contralateral comparison) | rec_49")
    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    all_samples = load_bilateral_samples(
        csv_path, images_root, split_name="training",
        lesion_types=[t.strip() for t in args.lesion_types.split(",")]
        if args.lesion_types else None,
        min_box_side=float(args.min_box_side),
        max_box_ar=float(args.max_box_ar),
        input_w=int(args.input_w),
    )
    print(f"Total bilateral samples: {len(all_samples)}")
    if args.lesion_types:
        print(f"Lesion type filter: {args.lesion_types}")
    if float(args.min_box_side) > 0.0 or float(args.max_box_ar) < float("inf"):
        print(f"Box detectability filter: min_side≥{args.min_box_side:.1f}px, max_AR≤{args.max_box_ar:.1f}")

    # Log view distribution
    n_cc = sum(1 for s in all_samples if s.view == "CC")
    n_mlo = sum(1 for s in all_samples if s.view == "MLO")
    n_l = sum(1 for s in all_samples if s.laterality == "L")
    n_r = sum(1 for s in all_samples if s.laterality == "R")
    print(f"View distribution: CC={n_cc} MLO={n_mlo} | Laterality: L={n_l} R={n_r}")

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
    if bool(args.augment):
        print(
            f"Augmentation: contrast=[{args.aug_contrast_range[0]:.2f}, {args.aug_contrast_range[1]:.2f}] "
            f"scale_min={args.aug_scale_min:.2f} | hflip: disabled (bilateral semantics preserved)"
        )
    else:
        print("Augmentation: disabled")
    print(
        f"Cliff patience ratio: {args.cliff_patience_ratio} | "
        f"min_delta: {args.min_delta} | seed: {args.seed}"
    )

    # Parse anchor sizes
    anchor_size_vals = [int(s.strip()) for s in str(args.anchor_sizes).split(",")]
    anchor_sizes = tuple((s,) for s in anchor_size_vals)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size_vals)
    print(f"Anchor sizes: {anchor_sizes} | Aspect ratios: (0.5, 1.0, 2.0) per level")

    # Build model
    model = build_retinanet_bilateral(
        medical_backbone_path=(
            str(args.medical_backbone_path) if args.medical_backbone_path else None
        ),
        num_classes=1,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        min_size=int(args.input_w),
        max_size=int(args.input_h),
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

    # Build training dataset
    train_dataset = BilateralDetectionDataset(
        samples=all_samples,
        indices=train_idx,
        input_h=int(args.input_h),
        input_w=int(args.input_w),
        augment=bool(args.augment),
        aug_contrast_min=float(args.aug_contrast_range[0]),
        aug_contrast_max=float(args.aug_contrast_range[1]),
        aug_scale_min=float(args.aug_scale_min),
        seed=int(args.seed),
    )

    # Weighted sampler: oversample positive images
    sampler_weights = make_oversampling_weights(
        all_samples, train_idx, pos_oversample_factor=float(args.pos_oversample_factor)
    )

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

        val_metrics = validate_bilateral(
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
                    "bilateral": True,
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
