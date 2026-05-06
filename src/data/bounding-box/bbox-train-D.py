r"""

Use this to run in Git Bash:

python src/data/bounding-box/bbox-train-D.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --patch-size 256 \
    --stride 64 \
    --max-pos-per-image 8 \
    --pos-neg-ratio 3.0 \
    --neg-image-patch-count 5000 \
    --clf-pos-weight 2.0 \
    --val-heatmap-threshold 0.5 \
    --val-heatmap-dilation 15 \
    --val-iou-threshold 0.1 \
    --augment \
    --aug-hflip-prob 0.5 \
    --aug-brightness-delta 0.2 \
    --patience 20 \
    --medical-backbone-path models/raw/ResNet50.pt \
    --hide-progress-bar


架构彻底切换：将检测问题重构为 patch 级滑窗二分类（rec_44）：

根因：anchor-based（RetinaNet/FasterRCNN）和 anchor-free（FCOS）框架的共同
瓶颈是每张图产生 60,000–200,000 个 anchor/center point，正负比高达 1:1000；
Focal Loss / HNM 只能缓解梯度不平衡而无法从根本上消除它。彻底放弃检测框架，
将任务重构为：给定一个 patch_size × patch_size 的乳腺区域 patch，判断其中心
1/3 区域是否包含至少一个病灶 GT box 的中心点。
a. 新数据集 PatchDataset：
   - 对每张正样本图像，以步长 stride（默认 64px）在原始坐标系内提取重叠的
     patch_size × patch_size patch（训练阶段不应用乳房裁剪，保持与推理一致）。
   - 标签：至少一个 GT box 中心落在 patch 中心 1/3 区域 → 正（1），否则 → 负（0）。
     中心 1/3 区域定义为局部坐标 [ps/3, 2*ps/3) × [ps/3, 2*ps/3)。
   - 采样策略：每张正样本图最多取 max_pos_per_image（默认 8）个正 patch；
     同时从同一张图随机取 max_pos_per_image × pos_neg_ratio 个负 patch（来自
     无 GT 中心的区域），精确控制正负比 1:pos_neg_ratio（默认 1:3）。
   - 负样本来自正样本图的非病灶区域（难负样本），无需扫描负样本图像。
   - 索引在 __init__ 中无需加载图像即可构建（使用 CSV 的 orig_size 和原始坐标）。
   - 每 epoch 传入不同的 epoch 参数以重新随机采样，保持训练多样性。
b. 新模型 PatchClassifier：
   - ResNet50（可选 RadImageNet 或 ImageNet 预训练初始化）+ 全局平均池化
     （已内置于 ResNet50）+ Dropout(0.5) + FC(2048→1)；BCEWithLogitsLoss +
     pos_weight 可配置（默认 5.0）；conv1 均值适配灰度输入。
   - 去除 RetinaNet / anchor generator / NMS / Focal Loss 全部相关代码。
c. 训练循环：
   - 每 epoch 重新构建 PatchDataset（变化 epoch 参数→不同采样），shuffle=True。
   - BCEWithLogitsLoss 梯度累积，CosineAnnealingLR 调度。
   - 随机水平翻转和亮度扰动作用于 patch 级别。
d. 验证推理（密集滑窗 → 热图 → 伪框）：
   - 在每张验证图像上以步长 stride 直接（无乳房裁剪）运行密集滑窗，sigmoid
     输出构成概率热图（每个 patch 中心点记录最大分数）。
   - 以不同阈值（0.1/0.3/0.5/0.7/0.9）二值化热图 → 形态学膨胀 → 连通域 →
     外接矩形作为预测框，以 IoU@0.5 与 GT 框比较，计算多阈值 BestThreshF1。
   - checkpoint 选择逻辑与之前保持一致（最大化 BestThreshF1）。
参数变化（移除所有 RetinaNet 专有参数，新增）：
- --patch-size（默认 256）：patch 边长（像素）。
- --stride（默认 64）：密集滑窗步长（训练和推理共用）。
- --max-pos-per-image（默认 8）：每张正样本图最多采样的正 patch 数。
- --pos-neg-ratio（默认 3.0）：每张图的负/正 patch 采样比。
- --clf-pos-weight（默认 5.0）：BCEWithLogitsLoss pos_weight。
- --dropout（默认 0.5）：FC 头前的 Dropout 概率。
- --val-heatmap-threshold（默认 0.5）：验证推理时热图二值化基准阈值（multi-
  threshold 评估额外覆盖 0.1/0.3/0.7/0.9）。
- --val-heatmap-dilation（默认 15）：热图连通域膨胀核大小（像素），用于合并
  同一病灶的相邻激活点。
- --accumulation-steps（保留，默认 1）：梯度累积步数。

================

rec_44_upd_1：分析首轮训练（Best F1=0.4817, Recall=46%）后的针对性修复：

问题一（最关键）——热图标注区域错误：
  训练标签定义为"GT中心落在 patch 中心 1/3"，但推理时将 patch 分数写入整个
  256×256 区域，导致热图 blob 约 342px，与典型 100px GT box 的 IoU≈0.07 < 0.1
  阈值，大量本可检出的病灶被迫计为 FN。
  修复：推理时改为写入 patch 中心 1/3（86×86px），blob 约 172px，IoU 提升到 ~0.34。

问题二——负样本来源单一：
  训练负 patch 全来自正样本图的非病灶区域，从未见过真阴性图像（12,413 张）的
  正常乳腺组织，导致 FP=141（每图 0.62 个）。
  修复：新增 --neg-image-patch-count 参数，每 epoch 从阴性图像随机采样 N 个
  patch（标签=0）加入训练，迫使模型学会抑制正常乳腺组织。

问题三——采样方差大导致 F1 振荡：
  每 epoch 重随机负 patch 使 F1 在 0.33–0.48 间剧烈振荡。
  缓解：max-pos-per-image 8→24，增大正样本密度以降低 epoch 间方差。

参数变化：
- --max-pos-per-image 8→24
- --neg-image-patch-count 0→5000（新参数，默认 0 保留旧行为）
- --val-heatmap-dilation 15→5（中心1/3 blob 更小，不需要大膨胀）
- --patience 15→20

================

rec_44_upd_2：分析 rec_44_upd_1（Best F1=0.4265, Recall=55%）后的修复：

问题根因——dilation 过小导致热图 blob 碎裂，FP 从 141 暴增到 326：
  中心 1/3 写入区域为 85×85px，stride=64px，相邻两个 stride 距离之外（128px）
  的 patch 写区域完全不相交（gap=43px）。dilation 从 15→5 后无法桥接碎裂
  blob，同一病灶区域产生 2-4 个独立预测框，大量 FP。修复：恢复 dilation=15。

同时恢复 max-pos-per-image 8，减小训练集体积（每 epoch ~38k vs ~113k patch），
缓解 loss 持续下降但 val F1 不跟随的过拟合现象（epoch 4 触顶即 early stop）。

参数变化（相对 rec_44_upd_1）：
- --max-pos-per-image 24→8（恢复原值）
- --val-heatmap-dilation 5→15（恢复原值）

================

rec_44_upd_3：分析 rec_44_upd_2（Best F1=0.4741, Recall=50.6%, FP=199 @ thresh=0.9）后的修复：

问题根因——clf-pos-weight=5.0 过大导致模型"过度激活"，FP 无法降低：
  BCEWithLogitsLoss 中正样本梯度权重是负样本的 5 倍，强迫模型将正样本分数拉
  到极高水平，同时将决策边界整体上移，导致大量负样本 patch 也"误穿"高阈值。
  表现为：42 个 epoch 中绝大多数 best thresh=0.9，且即使在 thresh=0.9 下仍有
  199 个 FP（正常情况下 thresh=0.5 应能有效过滤误报）。
  修复：将 clf-pos-weight 从 5.0 降至 2.0，让模型更保守、分数分布更均匀。

参数变化（相对 rec_44_upd_2）：
- --clf-pos-weight 5.0→2.0

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
import torch.nn.functional as F
try:
    from torchvision.models import resnet50 as _resnet50, ResNet50_Weights as _ResNet50Weights
except Exception:  # pragma: no cover
    _resnet50 = None  # type: ignore
    _ResNet50Weights = None  # type: ignore

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



# ─────────────────────────────────────────────────────────────────────────────
# Patch-level sliding-window dataset and inference helpers
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(torch.utils.data.Dataset):
    """Patch-level binary classification dataset for sliding-window lesion detection.

    For each positive training image (has GT boxes), overlapping patches of
    size patch_size × patch_size are extracted in the original image coordinate
    system (no breast-crop applied; images are loaded with CLAHE via normalize_image).

    Label assignment:
      - Positive (1): at least one GT box center falls inside the center 1/3 square
        of the patch, i.e. local_cx ∈ [ps/3, 2·ps/3) AND local_cy ∈ [ps/3, 2·ps/3).
      - Negative (0): no GT box center in that region.

    Sampling (at construction time — no image I/O needed):
      - Uses orig_size (from CSV) and sample.boxes (original coords) to enumerate
        patch positions without loading any image.
      - For each positive image: up to max_pos_per_image positive patches are kept;
        up to max_pos_per_image × pos_neg_ratio negative patches are randomly sampled
        from the same image (hard negatives = normal tissue colocated with disease).
      - Additionally, neg_image_patch_count patches are sampled from true negative
        images (no GT boxes) to teach the model normal-tissue appearance and reduce FP.
      - Epoch seed varies the negative sampling across epochs.
    """

    def __init__(
        self,
        samples: "List[Sample]",
        pos_indices: "List[int]",
        neg_indices: "List[int]",
        patch_size: int = 256,
        stride: int = 64,
        max_pos_per_image: int = 8,
        pos_neg_ratio: float = 3.0,
        neg_image_patch_count: int = 0,
        augment: bool = False,
        hflip_prob: float = 0.5,
        brightness_delta: float = 0.2,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        self.samples = samples
        self.patch_size = patch_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.brightness_delta = brightness_delta
        # (sample_idx, px, py, label)
        self._patch_list: "List[Tuple[int, int, int, float]]" = []
        self._build_index(pos_indices, stride, max_pos_per_image, pos_neg_ratio, seed, epoch)
        if neg_image_patch_count > 0 and neg_indices:
            self._add_neg_image_patches(neg_indices, neg_image_patch_count, seed, epoch)

    def _add_neg_image_patches(
        self,
        neg_indices: "List[int]",
        count: int,
        seed: int,
        epoch: int,
    ) -> None:
        """Sample `count` random patches from true negative images (no GT boxes).

        Patches are extracted at random positions using orig_size from CSV (no
        image I/O at index-build time).  All patches are labeled 0.0.
        """
        rng = random.Random(seed + epoch * 1009 + 7)
        ps = self.patch_size
        candidates: "List[Tuple[int, int, int, float]]" = []
        shuffled_neg = neg_indices.copy()
        rng.shuffle(shuffled_neg)
        # Distribute across images; stop once we have enough candidates
        patches_per_image = max(1, count // max(len(shuffled_neg), 1) + 1)
        for sample_idx in shuffled_neg:
            if len(candidates) >= count * 4:  # over-sample, then trim
                break
            sample = self.samples[sample_idx]
            orig_h, orig_w = sample.orig_size
            if orig_h <= 0 or orig_w <= 0:
                continue
            H, W = int(orig_h), int(orig_w)
            if H < ps or W < ps:
                continue
            for _ in range(patches_per_image):
                px = rng.randint(0, W - ps)
                py = rng.randint(0, H - ps)
                candidates.append((sample_idx, px, py, 0.0))
        rng.shuffle(candidates)
        self._patch_list.extend(candidates[:count])
        rng.shuffle(self._patch_list)

    def _build_index(
        self,
        pos_indices: "List[int]",
        stride: int,
        max_pos_per_image: int,
        pos_neg_ratio: float,
        seed: int,
        epoch: int,
    ) -> None:
        rng = random.Random(seed + epoch * 997)
        ps = self.patch_size
        center_lo = ps / 3.0
        center_hi = ps * 2.0 / 3.0

        for sample_idx in pos_indices:
            sample = self.samples[sample_idx]
            orig_h, orig_w = sample.orig_size
            if orig_h <= 0 or orig_w <= 0:
                continue
            H, W = int(orig_h), int(orig_w)
            boxes = sample.boxes.astype(np.float32)
            if boxes.size == 0:
                continue
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]
            if boxes.size == 0:
                continue

            centers = [(float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)) for b in boxes]

            y_positions = list(range(0, max(1, H - ps + 1), stride)) or [0]
            x_positions = list(range(0, max(1, W - ps + 1), stride)) or [0]

            pos_patches: "List[Tuple[int, int, int, float]]" = []
            neg_patches: "List[Tuple[int, int, int, float]]" = []
            for py in y_positions:
                for px in x_positions:
                    is_positive = any(
                        center_lo <= (cx - px) < center_hi and center_lo <= (cy - py) < center_hi
                        for cx, cy in centers
                    )
                    (pos_patches if is_positive else neg_patches).append(
                        (sample_idx, px, py, 1.0 if is_positive else 0.0)
                    )

            rng.shuffle(pos_patches)
            selected_pos = pos_patches[:max_pos_per_image]
            n_neg = max(0, int(round(len(selected_pos) * pos_neg_ratio)))
            rng.shuffle(neg_patches)
            self._patch_list.extend(selected_pos)
            self._patch_list.extend(neg_patches[:n_neg])

        rng.shuffle(self._patch_list)

    def __len__(self) -> int:
        return len(self._patch_list)

    def __getitem__(self, idx: int) -> "Tuple[torch.Tensor, torch.Tensor]":
        sample_idx, px, py, label = self._patch_list[idx]
        sample = self.samples[sample_idx]
        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            return torch.zeros(3, self.patch_size, self.patch_size), torch.tensor(label, dtype=torch.float32)

        h, w = img.shape[:2]
        ps = self.patch_size
        x1, y1 = int(px), int(py)
        x2 = min(x1 + ps, w)
        y2 = min(y1 + ps, h)
        patch_np = img[y1:y2, x1:x2]
        pad_h, pad_w = ps - (y2 - y1), ps - (x2 - x1)
        if pad_h > 0 or pad_w > 0:
            patch_np = np.pad(patch_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        patch = image_to_tensor(patch_np)

        if self.augment:
            if random.random() < self.hflip_prob:
                patch = torch.flip(patch, [2])
            if self.brightness_delta > 0.0:
                patch = torch.clamp(patch * (1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)), 0.0, 1.0)

        return patch, torch.tensor(label, dtype=torch.float32)


def heatmap_to_boxes(
    heatmap: np.ndarray,
    threshold: float,
    dilation_size: int = 15,
    min_component_area: int = 50,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Convert a probability heatmap to bounding boxes via connected components.

    Args:
        heatmap: (H, W) float array with values in [0, 1].
        threshold: Binary threshold applied to heatmap.
        dilation_size: Dilation kernel size (merges nearby activations).
        min_component_area: Minimum connected component area to keep.

    Returns:
        boxes: (N, 4) float32 array in xyxy format.
        scores: (N,) float32 array, max heatmap value per component.
    """
    if heatmap.size == 0 or float(heatmap.max()) < threshold:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mask = (heatmap >= threshold).astype(np.uint8)
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        mask = cv2.dilate(mask, kernel)
    n_labels, labels_map = cv2.connectedComponents(mask, connectivity=8)

    boxes_list: "List[List[float]]" = []
    scores_list: "List[float]" = []
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


def build_patch_classifier(
    medical_backbone_path: Optional[str] = None,
    dropout: float = 0.5,
) -> torch.nn.Module:
    """Build a ResNet50-based binary patch classifier.

    Architecture: ResNet50 (ImageNet pretrained) + AdaptiveAvgPool2d (inside ResNet50)
    + Dropout + FC(2048→1).  Optionally overwrites backbone with a medical-domain
    pretrained checkpoint.  conv1 is adapted for grayscale-as-3channel input.
    """
    if _resnet50 is None:
        raise RuntimeError("torchvision resnet50 not found; please upgrade torchvision.")
    try:
        model = _resnet50(weights=_ResNet50Weights.DEFAULT)
    except Exception:
        model = _resnet50(pretrained=True)  # type: ignore[call-arg]

    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, 1),
    )

    if medical_backbone_path is not None:
        try:
            ckpt = torch.load(medical_backbone_path, map_location="cpu")
            raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
            _idx_to_resnet = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
            stripped: Dict[str, Any] = {}
            for k, v in raw_sd.items():
                k = re.sub(r"^(module\.|encoder\.|backbone\.|body\.)+", "", k)
                m = re.match(r"^(\d+)\.(.*)", k)
                if m and m.group(1) in _idx_to_resnet:
                    k = f"{_idx_to_resnet[m.group(1)]}.{m.group(2)}"
                stripped[k] = v
            missing, unexpected = model.load_state_dict(stripped, strict=False)
            print(f"[Info] Loaded medical backbone from: {medical_backbone_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as exc:
            print(f"[Warning] Could not load medical backbone ({exc}). Using ImageNet weights.")

    try:
        with torch.no_grad():
            mean_w = model.conv1.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(mean_w.expand_as(model.conv1.weight))
        print("[Info] conv1 weights averaged across channels for grayscale-as-3channel input.")
    except AttributeError:
        print("[Warning] Could not adapt conv1.")

    return model


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


def compute_iou_matches(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Simple IoU matching without score threshold (all pred boxes accepted).

    Returns (tp, fp, fn) counts using greedy matching by area overlap.
    """
    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    matched_gt = [False] * int(gt_boxes.shape[0])
    matched_pred = [False] * int(pred_boxes.shape[0])
    # Sort pred boxes by area descending (prefer larger/more confident boxes first)
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


def validate_sliding_window(
    model: torch.nn.Module,
    samples: "List[Sample]",
    val_indices: "List[int]",
    device: torch.device,
    patch_size: int,
    stride: int,
    val_batch_size: int,
    iou_threshold: float,
    dilation_size: int,
    epoch: int,
    epochs: int,
    disable_tqdm: bool = False,
) -> Dict[str, float]:
    """Validate with dense sliding-window inference: heatmap → boxes → F1 @ multi-thresholds."""
    model.eval()

    multi_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    multi_stats: Dict[float, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in multi_thresholds}
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes_at_best = 0

    pbar = tqdm(val_indices, desc=f"val {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for sample_idx in pbar:
            sample = samples[sample_idx]
            gt_boxes = sample.boxes.astype(np.float32)  # (N, 4) xyxy
            # Filter invalid boxes (same criteria as __getitem__ and _build_index)
            if gt_boxes.size > 0:
                keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
                gt_boxes = gt_boxes[keep]

            try:
                img_np = normalize_image(read_image_unicode(sample.image_path))
            except FileNotFoundError:
                continue

            h, w = img_np.shape[:2]
            # Build (H, W) heatmap via dense sliding window.
            # Each patch contributes its score to the ENTIRE patch area (max-pooling),
            # not just the center point. This ensures the resulting blobs are at the
            # correct spatial scale for IoU matching with GT boxes.
            heatmap = np.zeros((h, w), dtype=np.float32)

            ps = patch_size
            y_positions = list(range(0, max(1, h - ps + 1), stride)) or [0]
            x_positions = list(range(0, max(1, w - ps + 1), stride)) or [0]

            # Collect all patches for batch inference
            patch_list: "List[Tuple[int, int, torch.Tensor]]" = []
            for py in y_positions:
                for px in x_positions:
                    y2, x2 = min(py + ps, h), min(px + ps, w)
                    patch_np = img_np[py:y2, px:x2]
                    pad_h, pad_w = ps - (y2 - py), ps - (x2 - px)
                    if pad_h > 0 or pad_w > 0:
                        patch_np = np.pad(patch_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                    patch_list.append((py, px, image_to_tensor(patch_np)))

            for batch_start in range(0, len(patch_list), val_batch_size):
                batch_items = patch_list[batch_start:batch_start + val_batch_size]
                batch_tensor = torch.stack([item[2] for item in batch_items]).to(device)
                logits = model(batch_tensor).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i_b, (py, px, _) in enumerate(batch_items):
                    score = float(probs[i_b])
                    # Write the score to the CENTER 1/3 of the patch only.
                    # This is consistent with the training label definition:
                    # positive = GT center falls in [px+ps/3, px+2ps/3) x [py+ps/3, py+2ps/3).
                    # Full-patch marking (prior) created ~342px blobs; IoU with
                    # a 100px GT box was ~0.07 (<0.1 threshold). Center-1/3 marking
                    # creates ~172px blobs; IoU with a 100px GT box is ~0.34.
                    cy1 = py + ps // 3
                    cy2 = min(py + 2 * ps // 3, h)
                    cx1 = px + ps // 3
                    cx2 = min(px + 2 * ps // 3, w)
                    heatmap[cy1:cy2, cx1:cx2] = np.maximum(heatmap[cy1:cy2, cx1:cx2], score)

            # Evaluate at each threshold
            for thresh in multi_thresholds:
                pred_boxes, _ = heatmap_to_boxes(heatmap, thresh, dilation_size)
                tp, fp, fn = compute_iou_matches(pred_boxes, gt_boxes, iou_threshold)
                multi_stats[thresh]["tp"] += tp
                multi_stats[thresh]["fp"] += fp
                multi_stats[thresh]["fn"] += fn

            total_images += 1
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

    # Print per-threshold summary
    parts = []
    for thresh in multi_thresholds:
        tp = multi_stats[thresh]["tp"]
        fp = multi_stats[thresh]["fp"]
        fn = multi_stats[thresh]["fn"]
        f1 = f1_per_thresh[thresh]
        parts.append(f"@{thresh}: TP={tp} FP={fp} F1={f1:.4f}")
    print(f"  [Val] GT_boxes={total_gt_boxes} | {' | '.join(parts)}")
    print(f"  [BestThresh] F1={best_f1:.4f} @ thresh={best_thresh}")

    result: Dict[str, float] = {
        "best_thresh_f1": float(best_f1),
        "best_thresh": float(best_thresh),
        "val_images": float(total_images),
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int,
    pos_weight: float = 5.0,
    disable_tqdm: bool = False,
) -> Tuple[float, int]:
    """Train one epoch of the patch classifier with BCEWithLogitsLoss.

    Returns:
        avg_loss: average loss for the epoch
        optimizer_steps: number of times optimizer.step() was called
    """
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    running_loss = 0.0
    count = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=disable_tqdm)
    for i, (patches, labels) in enumerate(pbar):
        patches = patches.to(device)
        labels = labels.to(device)
        logits = model(patches).squeeze(1)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaled_loss = loss / float(accumulation_steps)
        scaled_loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item())
        count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(count, 1), optimizer_steps


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
    parser = argparse.ArgumentParser(description="Train a patch-level sliding-window lesion detector for VinDr.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Path to vindr_detection_folds.csv")
    parser.add_argument("--images-root", type=Path, default=None, help="Root folder containing processed images_png/<patient_id>/<image_id>")
    parser.add_argument("--save-path", type=Path, default=None, help="Best checkpoint path (default: models/bbox_resnet50.D.pth)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for patch training DataLoader")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Batch size for sliding-window validation (patch batches per image)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-iou-threshold", type=float, default=0.1, help="IoU threshold for matching predicted boxes to GT (0.1 is appropriate for patch-scale predictions vs lesion GT boxes)")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--only-use", type=float, default=1.0, help="Fraction of positive training images to use per epoch")

    # Patch extraction parameters
    parser.add_argument("--patch-size", type=int, default=256, help="Side length of each extracted patch in pixels")
    parser.add_argument("--stride", type=int, default=64, help="Sliding-window stride in pixels (training extraction and validation inference)")
    parser.add_argument("--max-pos-per-image", type=int, default=8, help="Maximum positive patches to sample per positive training image per epoch")
    parser.add_argument("--pos-neg-ratio", type=float, default=3.0, help="Negative-to-positive patch sampling ratio per image (hard negatives from same image)")
    parser.add_argument("--clf-pos-weight", type=float, default=5.0, help="BCEWithLogitsLoss pos_weight for patch classifier")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability before the final FC layer")
    parser.add_argument("--val-heatmap-threshold", type=float, default=0.5, help="Base binarization threshold for validation heatmap (multi-threshold eval also covers 0.1/0.3/0.7/0.9)")
    parser.add_argument("--val-heatmap-dilation", type=int, default=15, help="Dilation kernel size for merging nearby heatmap activations into bounding boxes")

    # Data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable patch-level augmentation (hflip + brightness jitter)")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Horizontal flip probability when --augment is set")
    parser.add_argument("--aug-brightness-delta", type=float, default=0.2, help="Brightness jitter magnitude (±delta) when --augment is set")

    # Medical pretrained backbone
    parser.add_argument(
        "--medical-backbone-path",
        type=str,
        default=None,
        help=(
            "Path to a medical-domain pretrained ResNet50 checkpoint (e.g. RadImageNet). "
            "Supports checkpoints with state_dict/model keys and common prefixes "
            "(module., encoder., backbone., body.) which are stripped automatically."
        ),
    )

    parser.add_argument("--neg-image-patch-count", type=int, default=0, help="Number of patches to sample from true negative images (no GT boxes) per epoch. 0 disables (legacy behavior).")
    parser.add_argument("--hide-progress-bar", action="store_true", help="Suppress tqdm progress bars")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox_resnet50.D.pth")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the sample list via VinDrBboxDataset (crop_breast_region=False:
    # patch training and inference both operate on the raw preprocessed image).
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=False,
        crop_breast_region=False,
    )

    usable_indices = list(range(len(train_dataset.samples)))
    pos_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size > 0]
    neg_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size == 0]

    train_indices, val_indices, split_summary = split_train_val_by_patient(
        samples=train_dataset.samples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    train_indices.sort()
    val_indices.sort()
    # Only validate on positive val images: negatives have no GT boxes and generate
    # only FP, inflating the denominator and driving F1 toward 0.
    val_pos_indices = [i for i in val_indices if train_dataset.samples[i].boxes.size > 0]

    train_pos_indices = [i for i in train_indices if train_dataset.samples[i].boxes.size > 0]

    train_summary = summarize_subset(train_dataset.samples, train_indices)
    val_summary = summarize_subset(train_dataset.samples, val_indices)
    split_summary["train"] = train_summary
    split_summary["val"] = val_summary
    split_summary["train_patients"] = int(len({train_dataset.samples[i].patient_id for i in train_indices}))
    split_summary["val_patients"] = int(len({train_dataset.samples[i].patient_id for i in val_indices}))

    print(f"Total usable images: {len(usable_indices)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(
        f"Train: {train_summary['images']} images (pos={train_summary['positive_images']}, neg={train_summary['negative_images']}) "
        f"| Val: {val_summary['images']} images (pos={val_summary['positive_images']}, neg={val_summary['negative_images']})"
    )
    print(f"Train patients: {split_summary['train_patients']} | Val patients: {split_summary['val_patients']}")
    train_neg_indices = [i for i in train_indices if train_dataset.samples[i].boxes.size == 0]
    print(f"Positive training images: {len(train_pos_indices)} | Negative training images: {len(train_neg_indices)}")
    print(f"Positive val images (used for validation): {len(val_pos_indices)}")
    if args.neg_image_patch_count > 0:
        print(f"Neg-image patch count per epoch: {args.neg_image_patch_count}")
    print(f"Device: {device}")
    print(f"Patch size: {args.patch_size} | Stride: {args.stride}")
    print(f"Max pos/image: {args.max_pos_per_image} | Pos-neg ratio: {args.pos_neg_ratio}")
    print(f"BatchSize: {args.batch_size} | Accumulation steps: {args.accumulation_steps}")
    print(f"LR: {args.lr} | Epochs: {args.epochs} | Patience: {args.patience}")

    model = build_patch_classifier(
        medical_backbone_path=args.medical_backbone_path if args.medical_backbone_path else None,
        dropout=float(args.dropout),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=1e-6)

    history: List[Dict[str, float]] = []
    best_metric = -1.0
    no_improve = 0
    best_epoch = -1

    for epoch in range(int(args.epochs)):
        print(f"\n{'-' * 72}")
        print(f"Epoch {epoch + 1} / {args.epochs}")
        print(f"{'-' * 72}")
        # Build a new PatchDataset each epoch for varied negative sampling
        patch_dataset = PatchDataset(
            samples=train_dataset.samples,
            pos_indices=train_pos_indices,
            neg_indices=train_neg_indices,
            patch_size=int(args.patch_size),
            stride=int(args.stride),
            max_pos_per_image=int(args.max_pos_per_image),
            pos_neg_ratio=float(args.pos_neg_ratio),
            neg_image_patch_count=int(args.neg_image_patch_count),
            augment=bool(args.augment),
            hflip_prob=float(args.aug_hflip_prob),
            brightness_delta=float(args.aug_brightness_delta),
            seed=int(args.seed),
            epoch=epoch,
        )
        train_loader = DataLoader(
            patch_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

        avg_loss, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=int(args.epochs),
            accumulation_steps=int(args.accumulation_steps),
            pos_weight=float(args.clf_pos_weight),
            disable_tqdm=bool(args.hide_progress_bar),
        )
        lr_scheduler.step()

        val_metrics = validate_sliding_window(
            model=model,
            samples=train_dataset.samples,
            val_indices=val_pos_indices,
            device=device,
            patch_size=int(args.patch_size),
            stride=int(args.stride),
            val_batch_size=int(args.val_batch_size),
            iou_threshold=float(args.val_iou_threshold),
            dilation_size=int(args.val_heatmap_dilation),
            epoch=epoch,
            epochs=int(args.epochs),
            disable_tqdm=bool(args.hide_progress_bar),
        )

        monitor_metric = float(val_metrics.get("best_thresh_f1", 0.0))
        best_thresh = float(val_metrics.get("best_thresh", 0.5))

        row: Dict[str, float] = {"epoch": float(epoch + 1), "loss": float(avg_loss), "optimizer_steps": float(optimizer_steps)}
        row.update(val_metrics)
        history.append(row)

        best_tp = int(val_metrics.get(f"tp@{best_thresh}", 0))
        best_fp = int(val_metrics.get(f"fp@{best_thresh}", 0))
        best_fn = int(val_metrics.get(f"fn@{best_thresh}", 0))
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"BestThreshF1={monitor_metric:.4f} @ thresh={best_thresh} "
            f"(TP={best_tp} FP={best_fp} FN={best_fn}) | "
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
                    "patch_size": int(args.patch_size),
                    "stride": int(args.stride),
                },
            )
            print(f"  [Checkpoint] Saved (BestThreshF1={best_metric:.4f}) -> {save_path}")
        else:
            no_improve += 1
            if int(args.patience) > 0 and no_improve >= int(args.patience):
                print(f"Early stopping triggered: no improvement for {no_improve} epochs.")
                break

    print(f"Training complete. Best BestThreshF1={best_metric:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {save_path}")


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
