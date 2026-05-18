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
    --monitor-metric fbeta2 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --save-path models/bbox_resnet50.H.pth \
    --augment \
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
) -> List[Sample]:
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
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_brightness_delta = aug_brightness_delta
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
            "resnet50",
            weights=_IMAGENET_WEIGHTS,
            trainable_layers=trainable_backbone_layers,
        )
        print("[Info] Loaded ImageNet weights into backbone.")
    except TypeError:
        # Older torchvision API
        backbone = resnet_fpn_backbone(  # type: ignore[call-arg]
            "resnet50",
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
        f"[BestRecall] Recall={best_recall:.4f} @ score={best_recall_thresh} | "
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
    parser.add_argument("--monitor-metric", type=str, default="fbeta2",
                        choices=["f1", "recall", "fbeta2"],
                        help="Metric to monitor for checkpoint saving and early stopping.")
    parser.add_argument("--hide-progress-bar", action="store_true")
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

    all_samples = load_samples(csv_path, images_root, split_name="training")
    print(f"Total samples: {len(all_samples)}")

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
        seed=int(args.seed),
    )

    # Weighted sampler: oversample positive images
    sampler_weights = make_oversampling_weights(
        all_samples, train_idx, pos_oversample_factor=float(args.pos_oversample_factor)
    )
    train_sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
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
                    "input_h": int(args.input_h),
                    "input_w": int(args.input_w),
                    "anchor_sizes": str(args.anchor_sizes),
                    "nms_thresh": float(args.nms_thresh),
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
