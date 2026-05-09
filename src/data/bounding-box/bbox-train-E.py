r"""
方向 E：U-Net 全图分割检测（使用 segmentation_models_pytorch）

安装依赖（在服务器上运行一次即可）：
    pip install segmentation-models-pytorch

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-E.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --input-h 1024 \
    --input-w 512 \
    --clf-pos-weight 5.0 \
    --val-heatmap-threshold 0.5 \
    --val-heatmap-dilation 30 \
    --val-iou-threshold 0.1 \
    --patience 20 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --hide-progress-bar

与方向 D 的核心差异：
  - 方向 D：patch 分类器（256×256），缺乏全局上下文，F1 天花板 ≈ 0.47
  - 方向 E：U-Net 全图分割（1024×512），模型可见整张乳腺图，能区分
            全局腺体分布与局部病灶，突破 patch 分类器的系统性误差

GT 生成：将 xyxy bbox 转为像素级高斯 blob 热图（σ = 框尺寸 / 6），
缩放到 1024×512 坐标系后写入，用 BCEWithLogitsLoss + Dice loss 联合监督。

后处理复用方向 D 管线：sigmoid → 阈值 → 膨胀 → 连通域 → 外接矩形 → IoU 匹配 GT。
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
# Utilities (adapted from bbox-train-D.py)
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


def heatmap_to_boxes(
    heatmap: np.ndarray,
    threshold: float,
    dilation_size: int = 30,
    min_component_area: int = 50,
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
    return np.array(boxes_list, dtype=np.float32), np.array(scores_list, dtype=np.float32)


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

def make_gaussian_heatmap(
    h: int,
    w: int,
    boxes: np.ndarray,   # (N, 4) xyxy in (h, w) coordinate system
    sigma_ratio: float = 6.0,
) -> np.ndarray:
    """Generate a soft GT heatmap with Gaussian blobs centred on each GT box.

    σ is computed as min(box_w, box_h) / sigma_ratio so larger boxes produce
    wider blobs.  Values are clipped to [0, 1].
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    if boxes.size == 0:
        return heatmap

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    for box in boxes:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        sigma = max(min(x2 - x1, y2 - y1) / sigma_ratio, 1.0)
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        heatmap = np.maximum(heatmap, gauss)

    return np.clip(heatmap, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """Full-image segmentation dataset.

    Each item is (image_tensor [3, H, W], heatmap_tensor [1, H, W]).
    Image is resized to (input_h, input_w).  GT boxes are scaled accordingly,
    then converted to Gaussian blob heatmaps.
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
            # Return zeros on missing file — will produce zero loss contribution
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            hm_t = torch.zeros(1, self.input_h, self.input_w, dtype=torch.float32)
            return img_t, hm_t

        orig_h, orig_w = img.shape[:2]
        boxes = sample.boxes.copy()

        # Filter invalid boxes
        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # Resize image to (input_h, input_w)
        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Scale boxes to resized coordinate system
        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        scaled_boxes = boxes.copy()
        if scaled_boxes.size > 0:
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 3] *= scale_y

        # Augmentation
        if self.augment:
            # Horizontal flip
            if self.rng.random() < self.aug_hflip_prob:
                img_resized = img_resized[:, ::-1, :].copy()
                if scaled_boxes.size > 0:
                    old_x1 = scaled_boxes[:, 0].copy()
                    old_x2 = scaled_boxes[:, 2].copy()
                    scaled_boxes[:, 0] = self.input_w - old_x2
                    scaled_boxes[:, 2] = self.input_w - old_x1
            # Brightness jitter
            if self.aug_brightness_delta > 0:
                delta = (self.rng.random() * 2 - 1) * self.aug_brightness_delta * 255
                img_resized = np.clip(img_resized.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        # Build GT heatmap
        heatmap = make_gaussian_heatmap(self.input_h, self.input_w, scaled_boxes)

        img_t = image_to_tensor(img_resized)
        hm_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
        return img_t, hm_t


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(
    medical_backbone_path: Optional[str] = None,
) -> "torch.nn.Module":
    """Build a U-Net with ResNet50 encoder.

    Requires segmentation_models_pytorch.  The RadImageNet ResNet50 weights
    are loaded into the encoder using the same key-mapping logic as D.
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
            missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
            print(f"[Info] Loaded ImageNet weights into encoder (missing={len(missing)}, unexpected={len(unexpected)})")
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
            missing, unexpected = model.encoder.load_state_dict(stripped, strict=False)
            print(f"[Info] Loaded RadImageNet backbone into encoder (missing={len(missing)}, unexpected={len(unexpected)})")
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

def dice_loss(pred_sigmoid: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss. pred_sigmoid and target are in [0, 1], same shape."""
    pred_flat = pred_sigmoid.reshape(-1)
    tgt_flat = target.reshape(-1)
    intersection = (pred_flat * tgt_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + tgt_flat.sum() + smooth)


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 5.0,
    bce_alpha: float = 0.5,
) -> torch.Tensor:
    """BCE + Dice combined loss.

    bce_alpha: weight of BCE term (1 - bce_alpha applied to Dice).
    """
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    pred_sigmoid = torch.sigmoid(logits)
    dice = dice_loss(pred_sigmoid, target)
    return bce_alpha * bce + (1.0 - bce_alpha) * dice


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
        loss = combined_loss(logits, heatmaps, pos_weight=pos_weight, bce_alpha=bce_alpha)

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
                pred_boxes, _ = heatmap_to_boxes(pred_heatmap_orig, thresh, dilation_size)
                tp, fp, fn = compute_iou_matches(pred_boxes, gt_boxes, iou_threshold)
                multi_stats[thresh]["tp"] += tp
                multi_stats[thresh]["fp"] += fp
                multi_stats[thresh]["fn"] += fn

            total_gt_boxes += int(gt_boxes.shape[0]) if gt_boxes.size > 0 else 0

    # Compute F1 per threshold and pick best
    f1_per_thresh: Dict[float, float] = {}
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_per_thresh[thresh] = f1

    best_thresh = max(f1_per_thresh, key=lambda t: f1_per_thresh[t])
    best_f1 = f1_per_thresh[best_thresh]

    parts = []
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        recall_t = tp / max(tp + fn, 1)
        parts.append(f"@{thresh}: TP={tp} FP={fp} FN={fn} Rec={recall_t:.3f} F1={f1_per_thresh[thresh]:.4f}")
    print(f"  [Val] GT_boxes={total_gt_boxes} | {' | '.join(parts)}")
    print(f"  [BestThresh] F1={best_f1:.4f} @ thresh={best_thresh}")

    result: Dict[str, float] = {
        "best_thresh_f1": float(best_f1),
        "best_thresh": float(best_thresh),
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
    parser = argparse.ArgumentParser(description="Train a U-Net full-image segmentation model for VinDr lesion detection (Direction E).")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-h", type=int, default=1024, help="Resize height for model input")
    parser.add_argument("--input-w", type=int, default=512, help="Resize width for model input")
    parser.add_argument("--clf-pos-weight", type=float, default=5.0, help="BCE pos_weight for foreground pixels")
    parser.add_argument("--bce-alpha", type=float, default=0.5, help="Weight of BCE in combined loss (1-alpha for Dice)")
    parser.add_argument("--val-heatmap-threshold", type=float, default=0.5)
    parser.add_argument("--val-heatmap-dilation", type=int, default=30)
    parser.add_argument("--val-iou-threshold", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1, help="Batch size for validation inference (usually 1 due to variable original sizes)")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2)
    parser.add_argument("--medical-backbone-path", type=Path, default=None)
    parser.add_argument("--hide-progress-bar", action="store_true")
    parser.add_argument("--sigma-ratio", type=float, default=6.0, help="Gaussian blob σ = min(box_w, box_h) / sigma_ratio")
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
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.E.pth"

    print(f"CSV: {csv_path}")
    print(f"Images root: {images_root}")
    print(f"Save path: {save_path}")

    # Load all samples (single CSV — direction E doesn't separate by split column for split;
    # uses the same "training" split as D for consistency).
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
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs} | Patience: {args.patience}")

    model = build_unet(
        medical_backbone_path=str(args.medical_backbone_path) if args.medical_backbone_path else None,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.01
    )

    history: List[Dict[str, float]] = []
    best_metric = 0.0
    best_epoch = 0
    no_improve = 0

    # Exit hook for clean logging
    _exit_state = {"reported": False}

    def _on_exit(reason: Optional[str] = None) -> None:
        if _exit_state["reported"]:
            return
        _exit_state["reported"] = True
        end_time = time.time()
        print(f"\n[Exit] Reason: {reason or 'normal'}")
        print(f"Best BestThreshF1={best_metric:.4f} at epoch {best_epoch}")

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
        print(f"\n{'─' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'─' * 72}")

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
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=bool(args.hide_progress_bar),
        )

        monitor_metric = float(val_metrics.get("best_thresh_f1", 0.0))
        best_thresh = float(val_metrics.get("best_thresh", 0.5))

        row: Dict[str, float] = {"epoch": float(epoch + 1), "loss": float(avg_loss)}
        row.update(val_metrics)
        history.append(row)

        best_tp = int(val_metrics.get(f"tp@{best_thresh}", 0))
        best_fp = int(val_metrics.get(f"fp@{best_thresh}", 0))
        best_fn = int(val_metrics.get(f"fn@{best_thresh}", 0))
        best_recall = best_tp / max(best_tp + best_fn, 1)
        best_prec = best_tp / max(best_tp + best_fp, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"BestThreshF1={monitor_metric:.4f} @ thresh={best_thresh} "
            f"(TP={best_tp} FP={best_fp} FN={best_fn} Recall={best_recall:.3f} Prec={best_prec:.3f}) | "
            f"lr={lr_scheduler.get_last_lr()[0]:.6f}"
        )

        improved = monitor_metric > best_metric + float(args.min_delta)
        if improved:
            best_metric = monitor_metric
            no_improve = 0
            best_epoch = epoch + 1
            save_checkpoint(
                save_path=save_path,
                model=model,
                meta={
                    "epoch": epoch + 1,
                    "best_thresh_f1": monitor_metric,
                    "best_thresh": best_thresh,
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                },
            )
            print(f"  [Checkpoint] Epoch {epoch + 1} | Saved (BestThreshF1={best_metric:.4f}) -> {save_path}")
        else:
            no_improve += 1
            if int(args.patience) > 0 and no_improve >= int(args.patience):
                print(f"Early stopping triggered: no improvement for {no_improve} epochs.")
                break

    print(f"\nTraining complete. Best BestThreshF1={best_metric:.4f} at epoch {best_epoch}.")
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
