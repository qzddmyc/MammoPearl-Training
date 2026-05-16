r"""
方向 G — Stage 2：ROI 级别二分类器（rec_47）

两阶段检测流程：
  Stage 1：U-Net（ResNet50 encoder，方向 F，已训练）生成 heatmap → NMS → 候选框
  Stage 2（本脚本）：对候选框 crop 进行二分类，过滤高置信度 FP

训练策略：
  正样本：训练集每个 GT box 区域（带 crop-expand 倍上下文填充，resize 到 crop-size×crop-size）
  硬负样本：正样本图中随机裁取但不与任何 GT box 重叠（IoU < 0.1）的区域
  易负样本：纯负样本图（无病灶）中随机裁取，数量 = hard_neg × easy-neg-ratio

架构：
  编码器：Stage 1 U-Net 的 ResNet50 encoder（加载 Stage 1 checkpoint 权重）
  分类头：AdaptiveAvgPool2d(1) → Linear(2048, 256) → ReLU → Dropout(0.5) → Linear(256, 1)

验证：
  训练过程中监控 crop 级别分类 F1；训练结束后运行完整 Stage 1 + Stage 2 推理，
  对多个 Stage 2 阈值（0.3–0.9）输出检测指标（TP/FP/FN/Prec/Rec/F1/F2）。

运行命令（Git Bash）：

python src/data/bounding-box/bbox-train-G.py \
    --stage1-model-path models/bbox_resnet50.F.pth \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.01 \
    --crop-expand 1.5 \
    --crop-size 224 \
    --mine-hard-negs-per-image 5 \
    --easy-neg-ratio 0.5 \
    --stage1-threshold 0.3 \
    --val-heatmap-dilation 15 \
    --val-iou-threshold 0.1 \
    --min-detection-area 200 \
    --box-nms-thresh 0.3 \
    --patience 10 \
    --save-path models/bbox_resnet50.G.pth \
    --hide-progress-bar
"""

from __future__ import annotations

import os
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp or not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


# =============================================================================
# Utilities (self-contained; duplicated from bbox-train-F.py for independence)
# =============================================================================

def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray          # (N, 4) xyxy in original image coords
    orig_size: Tuple[float, float] = field(default=(0.0, 0.0))  # (H, W)


def load_samples(csv_path: Path, images_root: Path, split_name: str) -> List[Sample]:
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split_name].copy()
    samples_dict: Dict[str, Sample] = {}
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])
        key = f"{patient_id}/{image_id}"
        image_path = images_root / patient_id / image_id
        if key not in samples_dict:
            samples_dict[key] = Sample(
                patient_id=patient_id,
                image_id=image_id,
                image_path=image_path,
                boxes=np.zeros((0, 4), dtype=np.float32),
                orig_size=(0.0, 0.0),
            )
        s = samples_dict[key]
        # Update orig_size from any row that has valid h/w (includes negative images)
        h_val = row.get("height")
        w_val = row.get("width")
        h = float(h_val) if (h_val is not None and pd.notna(h_val)) else 0.0
        w = float(w_val) if (w_val is not None and pd.notna(w_val)) else 0.0
        if h > 0 and w > 0 and s.orig_size == (0.0, 0.0):
            s.orig_size = (h, w)
        # Add GT box if present and valid
        xmin_v = row.get("xmin")
        ymin_v = row.get("ymin")
        xmax_v = row.get("xmax")
        ymax_v = row.get("ymax")
        if all(v is not None and pd.notna(v) for v in [xmin_v, ymin_v, xmax_v, ymax_v]):
            x1, y1 = float(xmin_v), float(ymin_v)
            x2, y2 = float(xmax_v), float(ymax_v)
            if x2 > x1 + 1 and y2 > y1 + 1:
                nb = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                s.boxes = np.vstack([s.boxes, nb]) if s.boxes.shape[0] > 0 else nb
    return list(samples_dict.values())


def patient_level_split(
    samples: List[Sample], val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int]]:
    patient_ids = list({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_patients = set(patient_ids[:n_val])
    train_idx = [i for i, s in enumerate(samples) if s.patient_id not in val_patients]
    val_idx = [i for i, s in enumerate(samples) if s.patient_id in val_patients]
    return train_idx, val_idx


# =============================================================================
# Stage 1 inference utilities (copied from bbox-train-F.py)
# =============================================================================

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


def heatmap_to_boxes(
    heatmap: np.ndarray,
    threshold: float,
    dilation_size: int = 15,
    min_component_area: int = 200,
    nms_iou_thresh: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if heatmap.size == 0 or float(heatmap.max()) < threshold:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    mask = (heatmap >= threshold).astype(np.uint8)
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        mask = cv2.dilate(mask, kernel)
    n_labels, labels_map = cv2.connectedComponents(mask, connectivity=8)
    boxes_list: List[List[float]] = []
    scores_list: List[float] = []
    for label in range(1, n_labels):
        comp = labels_map == label
        if int(comp.sum()) < min_component_area:
            continue
        ys, xs = np.where(comp)
        boxes_list.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0])
        scores_list.append(float(heatmap[comp].max()))
    if not boxes_list:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    boxes_arr = np.array(boxes_list, dtype=np.float32)
    scores_arr = np.array(scores_list, dtype=np.float32)
    if nms_iou_thresh > 0.0 and len(boxes_arr) > 1:
        keep = _nms_boxes(boxes_arr, scores_arr, nms_iou_thresh)
        boxes_arr = boxes_arr[keep]
        scores_arr = scores_arr[keep]
    return boxes_arr, scores_arr


def compute_iou_matches(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float
) -> Tuple[int, int, int]:
    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    matched_gt = [False] * int(gt_boxes.shape[0])
    matched_pred = [False] * int(pred_boxes.shape[0])
    for pi in range(pred_boxes.shape[0]):
        best_iou, best_gi = 0.0, -1
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
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt[best_gi] = True
            matched_pred[pi] = True
    tp = sum(matched_pred)
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - sum(matched_gt)
    return int(tp), int(fp), int(fn)


# =============================================================================
# Crop utilities
# =============================================================================

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


# (image_path, box_xyxy, orig_h, orig_w)
MiningItem = Tuple[Path, np.ndarray, int, int]


def mine_stage1_fp_crops(
    stage1_model: "nn.Module",
    samples: List[Sample],
    indices: List[int],
    device: torch.device,
    input_h: int,
    input_w: int,
    stage1_threshold: float,
    dilation_size: int,
    min_component_area: int,
    nms_iou_thresh: float,
    max_per_image: int,
    disable_tqdm: bool,
) -> List[MiningItem]:
    """Run Stage 1 on training images and collect FP candidate crops.

    For each image, extracts heatmap candidates and keeps those with
    IoU < 0.1 with all GT boxes (i.e., genuine false positives of Stage 1).
    These are used as hard negatives for Stage 2 training.
    """
    stage1_model.eval()
    mined: List[MiningItem] = []
    pbar = tqdm(indices, desc="Mining Stage1 FPs", leave=False, disable=disable_tqdm)
    with torch.no_grad():
        for idx in pbar:
            s = samples[idx]
            orig_img = normalize_image(read_image_unicode(s.image_path))
            orig_h, orig_w = orig_img.shape[:2]
            img_resized = cv2.resize(orig_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            img_t = image_to_tensor(img_resized).unsqueeze(0).to(device)
            logits = stage1_model(img_t)
            heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()
            heatmap_orig = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            cand_boxes, cand_scores = heatmap_to_boxes(
                heatmap_orig, stage1_threshold, dilation_size,
                min_component_area=min_component_area, nms_iou_thresh=nms_iou_thresh,
            )
            if len(cand_boxes) == 0:
                continue
            gt_boxes = s.boxes
            fp_boxes: List[Tuple[np.ndarray, float]] = []
            for box, score in zip(cand_boxes, cand_scores):
                is_fp = all(box_iou(box, gt) < 0.1 for gt in gt_boxes)
                if is_fp:
                    fp_boxes.append((box, float(score)))
            # Sort by score descending, take top max_per_image
            fp_boxes.sort(key=lambda x: x[1], reverse=True)
            for box, _ in fp_boxes[:max_per_image]:
                mined.append((s.image_path, box, orig_h, orig_w))
    return mined


def make_crop(
    img: np.ndarray,        # HWC uint8
    box: np.ndarray,        # [x1, y1, x2, y2] in img coords
    expand_factor: float,
    crop_size: int,
) -> np.ndarray:
    """Expand box by expand_factor, clamp to image, crop and resize."""
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = bw * expand_factor / 2.0
    half_h = bh * expand_factor / 2.0
    ex1 = max(0, int(cx - half_w))
    ey1 = max(0, int(cy - half_h))
    ex2 = min(img_w, int(cx + half_w) + 1)
    ey2 = min(img_h, int(cy + half_h) + 1)
    if ex2 - ex1 < 2 or ey2 - ey1 < 2:
        ex1 = max(0, int(x1)); ey1 = max(0, int(y1))
        ex2 = min(img_w, int(x2) + 1); ey2 = min(img_h, int(y2) + 1)
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# CropDataset
# =============================================================================

# (image_path, box_xyxy, label, orig_h, orig_w)
CropItem = Tuple[Path, np.ndarray, int, int, int]

_VINDR_DEFAULT_H = 1520
_VINDR_DEFAULT_W = 912


class CropDataset(Dataset):
    """Binary crop classification dataset.

    Positive  : one crop per GT box in positive training images.
    Hard neg  : if mined_hard_negs is given, use Stage 1 FP candidates;
                otherwise, sample hard_neg_per_pos_image random crops per
                positive image that do NOT overlap any GT box (IoU < 0.1).
    Easy neg  : random crops from purely negative images,
                capped at n_hard_neg * easy_neg_ratio.
    """

    def __init__(
        self,
        samples: List[Sample],
        indices: List[int],
        crop_size: int,
        expand_factor: float,
        hard_neg_per_pos_image: int,
        easy_neg_ratio: float,
        augment: bool,
        seed: int = 42,
        mined_hard_negs: Optional[List[MiningItem]] = None,
    ) -> None:
        self.crop_size = crop_size
        self.expand_factor = expand_factor
        self.augment = augment
        self._rng = random.Random(seed)
        self.items: List[CropItem] = []

        pos_indices = [i for i in indices if samples[i].boxes.shape[0] > 0]
        neg_indices = [i for i in indices if samples[i].boxes.shape[0] == 0]

        # ── Positive crops ──────────────────────────────────────────────────
        for idx in pos_indices:
            s = samples[idx]
            orig_h = int(s.orig_size[0]) if s.orig_size[0] > 0 else _VINDR_DEFAULT_H
            orig_w = int(s.orig_size[1]) if s.orig_size[1] > 0 else _VINDR_DEFAULT_W
            for box in s.boxes:
                self.items.append((s.image_path, box.copy(), 1, orig_h, orig_w))

        # ── Hard negatives ──────────────────────────────────────────────────
        n_hard_neg = 0
        if mined_hard_negs is not None:
            # Use Stage 1 FP candidates mined from training images
            for path, box, h, w in mined_hard_negs:
                self.items.append((path, box, 0, h, w))
                n_hard_neg += 1
        else:
            # Fallback: random crops from positive images (not overlapping GT)
            for idx in pos_indices:
                s = samples[idx]
                orig_h = int(s.orig_size[0]) if s.orig_size[0] > 0 else _VINDR_DEFAULT_H
                orig_w = int(s.orig_size[1]) if s.orig_size[1] > 0 else _VINDR_DEFAULT_W
                added, attempts = 0, 0
                max_attempts = hard_neg_per_pos_image * 25
                while added < hard_neg_per_pos_image and attempts < max_attempts:
                    attempts += 1
                    neg_box = self._sample_hard_neg(s.boxes, orig_h, orig_w)
                    if neg_box is not None:
                        self.items.append((s.image_path, neg_box, 0, orig_h, orig_w))
                        added += 1
                        n_hard_neg += 1

        # ── Easy negatives (from pure negative images, capped) ──────────────
        max_easy = int(n_hard_neg * easy_neg_ratio)
        self._rng.shuffle(neg_indices)
        easy_added = 0
        for idx in neg_indices:
            if easy_added >= max_easy:
                break
            s = samples[idx]
            orig_h = int(s.orig_size[0]) if s.orig_size[0] > 0 else _VINDR_DEFAULT_H
            orig_w = int(s.orig_size[1]) if s.orig_size[1] > 0 else _VINDR_DEFAULT_W
            neg_box = self._sample_random_crop(orig_h, orig_w)
            self.items.append((s.image_path, neg_box, 0, orig_h, orig_w))
            easy_added += 1

        n_pos = sum(1 for it in self.items if it[2] == 1)
        n_neg = len(self.items) - n_pos
        hard_src = "mined" if mined_hard_negs is not None else "random"
        print(
            f"  CropDataset: {len(self.items)} items | "
            f"pos={n_pos} neg={n_neg} (hard_neg={n_hard_neg}[{hard_src}] easy_neg={easy_added})"
        )

    def _sample_hard_neg(
        self, gt_boxes: np.ndarray, orig_h: int, orig_w: int
    ) -> Optional[np.ndarray]:
        min_side = max(32, min(orig_h, orig_w) // 20)
        max_side = max(min_side + 2, min(orig_h, orig_w) // 4)
        side = self._rng.randint(min_side, max_side)
        x1 = self._rng.randint(0, max(0, orig_w - side))
        y1 = self._rng.randint(0, max(0, orig_h - side))
        x2, y2 = x1 + side, y1 + side
        neg_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        for gt in gt_boxes:
            if box_iou(neg_box, gt) >= 0.1:
                return None
        return neg_box

    def _sample_random_crop(self, orig_h: int, orig_w: int) -> np.ndarray:
        min_side = max(32, min(orig_h, orig_w) // 20)
        max_side = max(min_side + 2, min(orig_h, orig_w) // 4)
        side = self._rng.randint(min_side, max_side)
        x1 = self._rng.randint(0, max(0, orig_w - side))
        y1 = self._rng.randint(0, max(0, orig_h - side))
        return np.array([x1, y1, x1 + side, y1 + side], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, box, label, orig_h, orig_w = self.items[i]
        img = normalize_image(read_image_unicode(path))
        crop = make_crop(img, box, self.expand_factor, self.crop_size)
        if self.augment:
            if self._rng.random() < 0.5:
                crop = crop[:, ::-1].copy()
            if self._rng.random() < 0.3:
                crop = crop[::-1].copy()
        return image_to_tensor(crop), torch.tensor(float(label))


# =============================================================================
# Stage 2 Model
# =============================================================================

class Stage2Net(nn.Module):
    """Binary ROI classifier built on a Stage 1 U-Net encoder."""

    def __init__(self, encoder: nn.Module, encoder_out_channels: int = 2048) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # list of feature maps; last is deepest
        return self.head(features[-1])  # (B, 1)


def load_stage1_unet(stage1_model_path: Path, device: torch.device) -> "smp.Unet":
    """Load full Stage 1 U-Net from checkpoint."""
    assert HAS_SMP, "segmentation_models_pytorch is required"
    stage1 = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1, activation=None)
    ckpt = torch.load(str(stage1_model_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    stage1.load_state_dict(state_dict)
    return stage1.to(device)


def build_stage2_net(stage1_model_path: Path, device: torch.device) -> Stage2Net:
    """Load Stage 1 U-Net checkpoint, extract encoder, build Stage2Net."""
    stage1 = load_stage1_unet(stage1_model_path, device)
    model = Stage2Net(encoder=stage1.encoder, encoder_out_channels=2048)
    return model.to(device)


# =============================================================================
# Training and crop-level validation
# =============================================================================

def train_epoch(
    model: Stage2Net,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
    disable_tqdm: bool,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    pw = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"train {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_crops(
    model: Stage2Net,
    loader: DataLoader,
    device: torch.device,
    disable_tqdm: bool,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    """Crop-level classification metrics (accuracy, precision, recall, F1)."""
    model.eval()
    all_labels: List[float] = []
    all_scores: List[float] = []
    pbar = tqdm(loader, desc=f"val  {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for imgs, labels in pbar:
        scores = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())
    lbl = np.array(all_labels)
    scr = np.array(all_scores)

    best_f1, best_thresh = 0.0, 0.5
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (scr >= t).astype(int)
        tp = int(((preds == 1) & (lbl == 1)).sum())
        fp = int(((preds == 1) & (lbl == 0)).sum())
        fn = int(((preds == 0) & (lbl == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds05 = (scr >= 0.5).astype(int)
    tp05 = int(((preds05 == 1) & (lbl == 1)).sum())
    fp05 = int(((preds05 == 1) & (lbl == 0)).sum())
    fn05 = int(((preds05 == 0) & (lbl == 1)).sum())
    tn05 = int(((preds05 == 0) & (lbl == 0)).sum())
    acc = (tp05 + tn05) / max(len(lbl), 1)
    prec05 = tp05 / max(tp05 + fp05, 1)
    rec05 = tp05 / max(tp05 + fn05, 1)
    f105 = 2 * prec05 * rec05 / max(prec05 + rec05, 1e-9)
    return {
        "val_acc": acc,
        "val_prec@0.5": prec05,
        "val_rec@0.5": rec05,
        "val_f1@0.5": f105,
        "val_best_f1": best_f1,
        "val_best_thresh": float(best_thresh),
    }


# =============================================================================
# Detection-level validation (Stage 1 + Stage 2 pipeline)
# =============================================================================

# Per-image inference result: (gt_boxes, candidate_boxes, stage2_scores)
DetResult = Tuple[np.ndarray, np.ndarray, np.ndarray]


@torch.no_grad()
def run_detection_inference(
    stage1_model: nn.Module,
    stage2_model: Stage2Net,
    samples: List[Sample],
    val_indices: List[int],
    device: torch.device,
    input_h: int,
    input_w: int,
    stage1_threshold: float,
    dilation_size: int,
    min_component_area: int,
    nms_iou_thresh: float,
    crop_expand: float,
    crop_size: int,
    disable_tqdm: bool,
) -> List[DetResult]:
    """Run Stage 1 inference → Stage 2 scoring per validation image.

    Returns list of (gt_boxes, cand_boxes, stage2_scores) for each image.
    Boxes with no Stage 1 candidates have empty arrays.
    """
    stage1_model.eval()
    stage2_model.eval()
    results: List[DetResult] = []
    pbar = tqdm(val_indices, desc="det_inference", leave=False, disable=disable_tqdm)
    for idx in pbar:
        s = samples[idx]
        orig_img = normalize_image(read_image_unicode(s.image_path))
        orig_h, orig_w = orig_img.shape[:2]
        gt_boxes = s.boxes.astype(np.float32).copy()
        if gt_boxes.size > 0:
            keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
            gt_boxes = gt_boxes[keep]

        # Stage 1: heatmap → candidate boxes
        img_resized = cv2.resize(orig_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img_t = image_to_tensor(img_resized).unsqueeze(0).to(device)
        logits = stage1_model(img_t)
        heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()
        heatmap_orig = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        cand_boxes, _ = heatmap_to_boxes(
            heatmap_orig, stage1_threshold, dilation_size,
            min_component_area=min_component_area, nms_iou_thresh=nms_iou_thresh,
        )

        if len(cand_boxes) == 0:
            results.append((gt_boxes, np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)))
            continue

        # Stage 2: classify each candidate crop
        crops = [image_to_tensor(make_crop(orig_img, box, crop_expand, crop_size)) for box in cand_boxes]
        crop_batch = torch.stack(crops).to(device)
        s2_scores = torch.sigmoid(stage2_model(crop_batch)).squeeze(1).cpu().numpy()
        results.append((gt_boxes, cand_boxes, s2_scores))
    return results


def sweep_detection_thresholds(
    results: List[DetResult],
    s2_thresholds: List[float],
    iou_threshold: float,
    stage1_total_boxes: int,
) -> None:
    """Print detection metrics for each Stage 2 threshold."""
    print(f"\n  [Stage 1 total candidates across val set: {stage1_total_boxes}]")
    print(
        f"  {'s2_thresh':>10} {'TP':>6} {'FP':>7} {'FN':>6} {'GT':>6} "
        f"{'Kept':>6} {'Prec':>6} {'Rec':>6} {'F1':>7} {'F2':>7}"
    )
    for t in s2_thresholds:
        tp, fp, fn, gt_total, kept = 0, 0, 0, 0, 0
        for gt_boxes, cand_boxes, s2_scores in results:
            gt_total += gt_boxes.shape[0]
            if len(cand_boxes) == 0:
                fn += gt_boxes.shape[0]
                continue
            mask = s2_scores >= t
            filtered = cand_boxes[mask]
            kept += int(mask.sum())
            t_, f_, n_ = compute_iou_matches(filtered, gt_boxes, iou_threshold)
            tp += t_; fp += f_; fn += n_
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        print(
            f"  {t:>10.1f} {tp:>6} {fp:>7} {fn:>6} {gt_total:>6} "
            f"{kept:>6} {prec:>6.3f} {rec:>6.3f} {f1:>7.4f} {f2:>7.4f}"
        )


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 ROI classifier for VinDr lesion detection (Direction F)."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--stage1-model-path", type=Path, default=None,
                        help="Path to Stage 1 U-Net checkpoint. Default: models/bbox_resnet50.F.pth")
    parser.add_argument("--save-path", type=Path, default=None,
                        help="Output path for Stage 2 model. Default: models/bbox_resnet50.G.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.01,
                        help="LR multiplier for the Stage 1 encoder inside Stage 2. "
                             "Very small value (0.01) keeps encoder mostly frozen. "
                             "0.0 = head-only training (encoder fully frozen).")
    parser.add_argument("--pos-weight", type=float, default=2.0,
                        help="BCE pos_weight for positive crops in training loss.")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Side length (px) to resize each ROI crop.")
    parser.add_argument("--crop-expand", type=float, default=1.5,
                        help="Context expansion factor around each candidate box. "
                             "1.5 adds 25%% of box width/height as margin on each side.")
    parser.add_argument("--hard-neg-per-pos-image", type=int, default=3,
                        help="Hard negative crops sampled per positive training image "
                             "(used only when --mine-hard-negs-per-image=0).")
    parser.add_argument("--mine-hard-negs-per-image", type=int, default=5,
                        help="Run Stage 1 on training images to mine FP candidates as hard negatives. "
                             "Value = max FP crops per image. Set 0 to disable (use random crops instead).")
    parser.add_argument("--easy-neg-ratio", type=float, default=0.5,
                        help="Ratio of easy negatives (from pure negative images) to hard negatives.")
    parser.add_argument("--input-h", type=int, default=1024,
                        help="Stage 1 input height (must match Stage 1 training).")
    parser.add_argument("--input-w", type=int, default=512,
                        help="Stage 1 input width (must match Stage 1 training).")
    parser.add_argument("--stage1-threshold", type=float, default=0.3,
                        help="Stage 1 heatmap threshold for candidate generation. "
                             "Use a low value (0.3) for high recall; Stage 2 will filter.")
    parser.add_argument("--val-heatmap-dilation", type=int, default=15)
    parser.add_argument("--val-iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-detection-area", type=int, default=200)
    parser.add_argument("--box-nms-thresh", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without crop-level F1 improvement).")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true",
                        help="Apply random horizontal/vertical flips to training crops.")
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
    stage1_path = args.stage1_model_path or repo_root / "models" / "bbox_resnet50.F.pth"
    save_path = args.save_path or repo_root / "models" / "bbox_resnet50.G.pth"

    print(f"Start time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stage 1 model: {stage1_path}")
    print(f"Save path:     {save_path}")

    all_samples = load_samples(csv_path, images_root, split_name="training")
    print(f"Total samples: {len(all_samples)}")

    train_idx, val_idx = patient_level_split(all_samples, val_ratio=0.15, seed=int(args.seed))
    val_pos_idx = [i for i in val_idx if all_samples[i].boxes.shape[0] > 0]
    n_train_pos = sum(1 for i in train_idx if all_samples[i].boxes.shape[0] > 0)
    n_train_neg = len(train_idx) - n_train_pos
    print(f"Train: {len(train_idx)} images (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val: {len(val_idx)} images | Val positive (detection eval): {len(val_pos_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Stage 1 model (used for mining + later reused for Stage 2 encoder) ──
    print(f"\nLoading Stage 1 U-Net from {stage1_path} ...")
    stage1_unet = load_stage1_unet(stage1_path, device)

    # ── Hard negative mining ──────────────────────────────────────────────────
    train_mined_negs: Optional[List[MiningItem]] = None
    mine_n = int(args.mine_hard_negs_per_image)
    if mine_n > 0:
        print(f"Mining Stage 1 FP candidates (max {mine_n}/image, {len(train_idx)} images)...")
        train_mined_negs = mine_stage1_fp_crops(
            stage1_model=stage1_unet,
            samples=all_samples,
            indices=train_idx,
            device=device,
            input_h=int(args.input_h),
            input_w=int(args.input_w),
            stage1_threshold=float(args.stage1_threshold),
            dilation_size=int(args.val_heatmap_dilation),
            min_component_area=int(args.min_detection_area),
            nms_iou_thresh=float(args.box_nms_thresh),
            max_per_image=mine_n,
            disable_tqdm=bool(args.hide_progress_bar),
        )
        print(f"  Mined {len(train_mined_negs)} FP hard negatives from training set.")

    # Build datasets
    print("\nBuilding CropDataset (train)...")
    train_ds = CropDataset(
        all_samples, train_idx,
        crop_size=int(args.crop_size),
        expand_factor=float(args.crop_expand),
        hard_neg_per_pos_image=int(args.hard_neg_per_pos_image),
        easy_neg_ratio=float(args.easy_neg_ratio),
        augment=bool(args.augment),
        seed=int(args.seed),
        mined_hard_negs=train_mined_negs,
    )
    print("Building CropDataset (val)...")
    val_ds = CropDataset(
        all_samples, val_idx,
        crop_size=int(args.crop_size),
        expand_factor=float(args.crop_expand),
        hard_neg_per_pos_image=int(args.hard_neg_per_pos_image),
        easy_neg_ratio=float(args.easy_neg_ratio),
        augment=False,
        seed=int(args.seed),
        mined_hard_negs=None,   # val uses random hard negs (monitoring only)
    )

    # WeightedRandomSampler: force balanced batches (50% pos, 50% neg)
    # regardless of dataset imbalance — eliminates the need to tune pos_weight
    _n_pos = sum(1 for it in train_ds.items if it[2] == 1)
    _n_neg = len(train_ds.items) - _n_pos
    _sample_weights = [
        1.0 / _n_pos if it[2] == 1 else 1.0 / _n_neg
        for it in train_ds.items
    ]
    from torch.utils.data import WeightedRandomSampler
    _sampler = WeightedRandomSampler(
        weights=_sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    print(f"  Balanced sampler: n_pos={_n_pos} n_neg={_n_neg} → each batch ~50% pos/neg")

    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), sampler=_sampler,
        num_workers=int(args.num_workers), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size) * 2, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=True,
    )

    # Build Stage 2 model (reuse already-loaded Stage 1 encoder)
    print("\nBuilding Stage 2 model from Stage 1 encoder...")
    model = Stage2Net(encoder=stage1_unet.encoder, encoder_out_channels=2048).to(device)
    del stage1_unet  # decoder no longer needed; encoder is kept alive via model.encoder

    encoder_lr = float(args.lr) * float(args.encoder_lr_multiplier)
    head_lr = float(args.lr)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=1e-6
    )

    print(
        f"Encoder LR: {encoder_lr:.2e} | Head LR: {head_lr:.2e} | "
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | Patience: {args.patience}"
    )

    # Training loop
    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(int(args.epochs)):
        print(f"\n{'─' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'─' * 72}")

        avg_loss = train_epoch(
            model, train_loader, optimizer, device,
            pos_weight=float(args.pos_weight),
            disable_tqdm=bool(args.hide_progress_bar),
            epoch=epoch, epochs=int(args.epochs),
        )
        lr_scheduler.step()

        val_metrics = validate_crops(
            model, val_loader, device,
            disable_tqdm=bool(args.hide_progress_bar),
            epoch=epoch, epochs=int(args.epochs),
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"acc={val_metrics['val_acc']:.3f} | "
            f"prec@0.5={val_metrics['val_prec@0.5']:.3f} | "
            f"rec@0.5={val_metrics['val_rec@0.5']:.3f} | "
            f"f1@0.5={val_metrics['val_f1@0.5']:.4f} | "
            f"best_f1={val_metrics['val_best_f1']:.4f}@{val_metrics['val_best_thresh']:.1f}"
        )

        cur_f1 = float(val_metrics["val_best_f1"])
        if cur_f1 > best_val_f1 + float(args.min_delta):
            best_val_f1 = cur_f1
            no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "meta": {"epoch": epoch + 1, "crop_f1": cur_f1}},
                save_path,
            )
            print(f"  [Checkpoint] Epoch {epoch + 1} | Saved (crop_f1={cur_f1:.4f}) -> {save_path}")
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                print(f"Early stopping triggered: no improvement for {args.patience} epochs.")
                break

    print(f"\nTraining complete. Best crop F1={best_val_f1:.4f}")
    print(f"Checkpoint: {save_path}")

    # ─── Detection-level validation with the best Stage 2 model ────────────
    print("\nLoading Stage 1 U-Net for detection validation...")
    assert HAS_SMP
    stage1_model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1, activation=None)
    s1_ckpt = torch.load(str(stage1_path), map_location="cpu")
    stage1_model.load_state_dict(s1_ckpt.get("model_state_dict", s1_ckpt))
    stage1_model = stage1_model.to(device)

    best_ckpt = torch.load(str(save_path), map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])

    print("Running Stage 1 + Stage 2 inference on val positive images...")
    det_results = run_detection_inference(
        stage1_model=stage1_model,
        stage2_model=model,
        samples=all_samples,
        val_indices=val_pos_idx,
        device=device,
        input_h=int(args.input_h),
        input_w=int(args.input_w),
        stage1_threshold=float(args.stage1_threshold),
        dilation_size=int(args.val_heatmap_dilation),
        min_component_area=int(args.min_detection_area),
        nms_iou_thresh=float(args.box_nms_thresh),
        crop_expand=float(args.crop_expand),
        crop_size=int(args.crop_size),
        disable_tqdm=bool(args.hide_progress_bar),
    )

    stage1_total = sum(len(r[1]) for r in det_results)
    sweep_detection_thresholds(
        results=det_results,
        s2_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        iou_threshold=float(args.val_iou_threshold),
        stage1_total_boxes=stage1_total,
    )

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nEnd time:    {end_time}")


if __name__ == "__main__":
    main()
