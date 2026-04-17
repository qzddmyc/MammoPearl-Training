from __future__ import annotations

"""
修改建议：
先跑一遍此代码，看看是不是 0.9 阈值下，框仍然很多，然后问问 AI下方的策略是否有用。

1.  调整 IoU 阈值： 默认情况下，IoU > 0.5 就会被判定为正样本。对于乳腺病灶这种特征模糊
    的目标，0.5 的阈值可能太低了。尝试将其提高到 0.6 或 0.7。这会强迫模型只学习那些定位
    非常精准的框，减少由于“边界模糊”导致的误检。
2.  Faster R-CNN 结构中，总损失由分类（Classification）和回归（Regression）两部分组成。
    如果 FP 暴涨，说明模型的分类器（Classifier）太“激进”了。
    人为调高分类损失在总 Loss 中的权重（Weight）。当模型把背景认成病灶时，给它一个更大的惩罚值。
    这样模型在给出“0.9”的高分预测时会变得更加谨慎。
3.  尝试完全去掉负样本，只用带有病灶的图像先进行 10(?) 个 Epoch 的“热身训练”，
    热身之后再接着全量样本去训练。
4.  若模型仍然报很多错误框，可以考虑：
      - 加权分类 loss，loss = CE(pos) * 1.0 + CE(neg) * 2.0[or more]，加重负样本的权重
      - 调整 roi-positive-fraction = 0.05 ~ 0.1，减少正样本的比例
5.  核心：
    [概括] 告诉损失函数，背景不值钱，病灶是无价之宝。错认一个病灶的代价，抵得上错认 10 个背景。
    [提示词] 请帮我修改 Faster R-CNN 模型内部 ROI Head 的分类损失函数（Classification Loss）。
        目前模型对背景和病灶的分类损失是同等权重的。我需要你介入 FastRCNNPredictor 或重写 roi_heads 的
        损失计算逻辑，引入带权重的交叉熵损失（Weighted Cross-Entropy Loss）。请为类别赋予明确的
        权重张量（Weight Tensor），例如设定 background 类的权重为 0.1，lesion 类的权重为 1.0 甚至更高。
        以此来成倍增加模型漏报正样本的损失惩罚，削弱大量简单背景被正确识别时带来的 Loss 稀释效应。
  * [概括] 降维打击的Focal Loss：其机制是，如果模型对某个背景有 99% 的把握，
        它产生的那一丁点 Loss 会被直接乘上一个极小的系数（接近归零）；但如果模型对某个病灶犹豫不决，
        它的 Loss 会被完全保留甚至放大。
  * [提示词] 请对现有的 torchvision Faster R-CNN 模型进行深度定制。我要求用 Focal Loss 替换掉 ROI Head 中
        默认的交叉熵分类损失（Cross Entropy Loss）。这是因为当前医疗影像正负样本极度不平衡，模型通过大量输出
        高置信度的‘简单背景（Easy Negatives）’来压低了整体 Loss。请实现一个 Focal Loss 模块，并替换分类头的
        计算逻辑。参数上，请暴露 gamma（建议默认值为 2.0，用于大幅削弱简单背景的 Loss 权重）和 alpha（用于调
        节正负类别的基础平衡比），迫使模型只能通过学习难点（真实的病灶特征）来降低 Loss。
    [概括] 只看错题：既然每次提取的 512 个 ROI 里有 400 个是毫无营养的纯背景，那我们算总分的时候，干脆把
        这 400 个最高分的背景直接扔掉，只计算剩下 112 个最容易混淆的样本的 Loss。
    [提示词] 请在 Faster R-CNN 的分类和回归损失计算中引入在线难例挖掘（OHEM, Online Hard Example Mining）机制。
        目前模型在每个 Batch 中会平均所有采样 ROI 的 Loss，导致大量容易识别的负样本（纯黑背景）稀释了病灶的训练梯度。
        我需要你在 ROI Head 计算损失时，不要使用默认的 mean() 均值。而是先算出所有候选框的独立 Loss，
        对其进行降序排列，只保留 Loss 值排名前 K 个（例如前 20% 或 128 个最困难的样本）的损失进行反向传播，
        将那些 Loss 极低的简单背景样本直接从反向传播的计算中剔除。
"""



# 使用 fasterrcnn_resnet50_fpn 模型对数据集进行训练
# 这个样本中的 bad data 使用 bad_data_record_resnet50.csv，但是跑完筛选程序后，会发现这是一个空集合。

# 使用这个模型去训练会耗费很长的时间，需要注意

# If your computer is GREAT, use this to run fuckingly:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 8 --fuck-running --lr 0.005 --freeze-epochs 4

# Otherwise, use:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4

# try:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 --roi-batch-size-per-image 256 --roi-positive-fraction 0.1

"""Train a breast lesion bounding-box detector from VinDr detection CSV.

This script reads `data/raw/vindr_detection_folds.csv`, matches each row to
`data/processed/images_png/<patient_id>/<image_id>`, and trains a Faster
R-CNN detector to predict lesion bounding boxes (xmin, ymin, xmax, ymax).

Model checkpoint is saved to `models/bbox_resnet50.pth`.


Update prompts:
1.  针对训练后期 Loss 卡在 0.19 左右无法下降的问题，需从学习率策略、
    模型结构和优化器等方面进行系统性干预，打破局部最优解。
2.  引入学习率 Warmup 机制，防止初始训练时因梯度过大破坏预训练权重，并提供合理的初始学习率设定。
3.  增加权重衰减（Weight Decay）的配置参数，通过正则化手段有效防止模型在较小数据集上过拟合。
4.  新增命令行参数 "--fuck-running" 作为算力切换开关：
    当不含此参数时，代码需在 batch_size=2 的前提下通过累积 4 个 step 再执行 optimizer.step()
    来变相实现 batch_size=8 的梯度累积；
    当存在该参数时，直接使用配置的较大 batch_size 进行正常训练，
    --同时两种模式下都必须保持每个 batch 内合理的正负样本混合比例。--(此行需要剔除，逻辑已删)
 *  fix: 在算得正样本时，需要按照比例上取整，以防止正样本丢失。
5.  针对 912x1520 的高分辨率医疗影像数据，修改模型的 AnchorGenerator，为其添加 8 和 16 这
    样更小的 scale 尺寸，以强化微小病灶的检测能力。6. 优化训练策略，除了在 DataLoader 端保持正
    负样本比之外，还需通过调整模型内部的 ROI 采样比例等参数，变相实现 Hard Negative Mining（挖掘难例）。
7.  实现渐冻层训练策略：在训练初期主动冻结 ResNet 的 layer1 和 layer2 层，仅训练 FPN 和检测头；
    在设定的几轮 Epoch 之后，全量解冻这些底层网络进行全局微调。
8.  强制采用 torchvision 中的 fasterrcnn_resnet50_fpn_v2 版本模型，以利用其更先进的
    数据增强策略和优化过的 FPN 特征提取结构。
  * feat: 需要使用 FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 权重。 
-----------
9.  取消对每一个 epoch、batch 的样本强制比例分配，保留原始的正负样本比例。
    即，对每个 epoch 使用全量样本（除了被分为验证集的）直接进行训练，
    另外，每个 epoch 中，在样本内部通过 shuffle=True 来打乱，以防止模型的过拟合。
10. 从 training 中自行划出约 15% 作为验证集；划分优先按 patient_id 进行，避免同一 patient 同时
    出现在 train 和 val；划分后尽量保持原始正负分布；验证集完全不参与训练，不参与反向传播。不做任何
    采样干预，保持真实分布。不使用 shuffle；使用验证集指标作为保存最佳模型的标准，优先推荐 F1；
    每个 epoch 后，如果当前指标优于历史最佳，则保存 best checkpoint；若验证集指标连续 N 个 epoch 没
    有提升，则停止训练，N 由参数控制。
    [注意] 这个版本保存的模型是效果最佳的一轮模型，而不是最终一轮的训练成果。

"""


import argparse
import json
import math
import random
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
try:
    # prefer v2 when available and import associated weights helper
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
except Exception:  # pragma: no cover
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_v2
    FasterRCNN_ResNet50_FPN_V2_Weights = None
# 不推荐使用 fasterrcnn_resnet50_fpn 模型，耗时过长

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
    """Convert image to RGB uint8 with 3 channels."""
    if img.ndim == 2:
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
    ) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name
        self.positive_only = positive_only

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
    bad_set: Optional[set[tuple[str, str]]] = None,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Split training data into train/val at the patient level.

    The split is patient-level to prevent leakage, and it tries to keep the
    original positive/negative distribution approximately stable by selecting
    roughly the same proportion of positive-patient and negative-patient images.
    """
    usable_indices: List[int] = []
    for idx, sample in enumerate(samples):
        if bad_set and (sample.patient_id, sample.image_id) in bad_set:
            continue
        usable_indices.append(idx)

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
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    """Validate one epoch without gradient computation."""
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    pbar = tqdm(loader, desc=f"val {epoch + 1}/{epochs}", leave=False)

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

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_tp + total_fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "images": float(total_images),
        "gt_boxes": float(total_gt_boxes),
        "pred_boxes": float(total_pred_boxes),
    }


def build_model(
    num_classes: int = 2,
    anchor_sizes: List[Tuple[int, ...]] | None = None,
) -> FasterRCNN:
    """Build Faster R-CNN using the v2 factory when available and a custom AnchorGenerator.

    anchor_sizes: sequence of tuples, one tuple per FPN level. Example: ((8,), (16,), (32,), (64,), (128,))
    """
    if anchor_sizes is None:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))

    aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))])
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # prefer to load torchvision v2 default weights when available
    weights = None
    if FasterRCNN_ResNet50_FPN_V2_Weights is not None:
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        # factory may accept a `weights` enum; pass if available
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_anchor_generator=anchor_generator,
        )
    except TypeError:
        # fallback to older factory if v2 signature differs
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn

            # Avoid deprecated `pretrained`/`pretrained_backbone` args by
            # explicitly passing `weights=None` / `weights_backbone=None`.
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, rpn_anchor_generator=anchor_generator)  # type: ignore[call-arg]
        except Exception:
            model = fasterrcnn_resnet50_fpn_v2()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_optimizer(model: FasterRCNN, args: argparse.Namespace, base_lr: Optional[float] = None) -> torch.optim.Optimizer:
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


def freeze_backbone_layers(model: FasterRCNN) -> None:
    """Freeze ResNet backbone layer1 and layer2 parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = False


def unfreeze_backbone_layers(model: FasterRCNN) -> None:
    """Unfreeze previously frozen ResNet backbone layers."""
    for name, param in model.named_parameters():
        if "backbone" in name and ("layer1" in name or "layer2" in name):
            param.requires_grad = True


def train_one_epoch(
    model: FasterRCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    accumulation_steps: int,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
    # track common Faster R-CNN sub-losses
    subloss_keys = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
    subloss_sums: Dict[str, float] = defaultdict(float)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        bad_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
        if bad_keys:
            bad_keys_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

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
    model: FasterRCNN,
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
        help="Best checkpoint path (default: models/bbox_resnet50.pth)",
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
    parser.add_argument("--freeze-epochs", type=int, default=2, help="Number of epochs to freeze backbone layer1/2 before unfreezing")

    # Anchor / ROI / RPN tuning
    parser.add_argument("--anchor-sizes", type=str, default="8,16,32,64,128", help="Comma-separated anchor sizes (one per FPN level ideally)")
    parser.add_argument("--roi-batch-size-per-image", type=int, default=512)
    parser.add_argument("--roi-positive-fraction", type=float, default=0.25)
    parser.add_argument("--rpn-pre-nms-top-n-train", type=int, default=2000)
    parser.add_argument("--rpn-post-nms-top-n-train", type=int, default=1000)

    # LR scheduling
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-step-size", type=int, default=0, help="StepLR step size; 0 to use CosineAnnealingLR")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox_resnet50.pth")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the full training split first, then split it into train/validation
    # at the patient level so that the same patient never appears on both sides.
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=False,
    )

    # Read bad data record (if exists) and build a set of (patient_id,image_id)
    bad_record_path = Path(__file__).resolve().parent / "bad_data_record_resnet50.csv"
    bad_set = set()
    if bad_record_path.exists():
        print("[Info] Bad data file record found.")
        try:
            with bad_record_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("patient_id")
                    iid = row.get("image_id")
                    if pid is not None and iid is not None:
                        bad_set.add((str(pid), str(iid)))
        except Exception:
            bad_set = set()
    else:
        print("[Info] bad data not found, use empty set instead.")

    usable_indices = [i for i, s in enumerate(train_dataset.samples) if (s.patient_id, s.image_id) not in bad_set]
    pos_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size > 0]
    neg_indices = [i for i in usable_indices if train_dataset.samples[i].boxes.size == 0]

    train_indices, val_indices, split_summary = split_train_val_by_patient(
        samples=train_dataset.samples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        bad_set=bad_set,
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

    split_summary["train"] = train_summary
    split_summary["val"] = val_summary
    split_summary["train_patients"] = int(len({train_dataset.samples[i].patient_id for i in train_indices}))
    split_summary["val_patients"] = int(len({train_dataset.samples[i].patient_id for i in val_indices}))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    if len(train_subset) == 0:
        raise ValueError("Training split is empty after patient-level split and filtering.")
    if len(val_subset) == 0:
        raise ValueError("Validation split is empty. Please check the CSV and splitting logic.")

    # Parse anchor sizes from args
    anchor_sizes = tuple((int(s.strip()),) for s in str(args.anchor_sizes).split(",") if s.strip())

    model = build_model(num_classes=2, anchor_sizes=anchor_sizes)
    model.to(device)

    # Freeze low-level backbone layers initially if requested
    if int(args.freeze_epochs) > 0:
        freeze_backbone_layers(model)

    optimizer = create_optimizer(model, args, base_lr=float(args.lr))

    # Choose LR scheduler: StepLR if step size provided, else CosineAnnealingLR
    if int(args.lr_step_size) > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    else:
        # warmup is implemented as iter-level LinearLR on epoch 0; remove dependency on --warmup-epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=1e-6)

    history: List[Dict[str, float]] = []

    # Apply ROI / RPN tuning to encourage hard-negative mining
    try:
        bs = int(args.roi_batch_size_per_image)
        model.roi_heads.batch_size_per_image = bs
        # Use ceiling strategy when computing number of positive ROIs:
        # ensure int(batch_size * positive_fraction) behaves like ceil(batch_size * fraction)
        try:
            pf = float(args.roi_positive_fraction)
        except Exception:
            pf = float(0.25)
        desired_pos = math.ceil(bs * pf) if bs > 0 else 0
        pos_frac = (desired_pos / bs) if bs > 0 else pf
        model.roi_heads.positive_fraction = float(pos_frac)
    except Exception:
        pass
    try:
        setattr(model.rpn, "pre_nms_top_n_train", int(args.rpn_pre_nms_top_n_train))
        setattr(model.rpn, "post_nms_top_n_train", int(args.rpn_post_nms_top_n_train))
    except Exception:
        pass

    print(f"Total usable images: {len(usable_indices)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(
        f"Split summary | train images: {train_summary['images']} (pos={train_summary['positive_images']}, neg={train_summary['negative_images']}) "
        f"| val images: {val_summary['images']} (pos={val_summary['positive_images']}, neg={val_summary['negative_images']})"
    )
    print(
        f"Split summary | train patients: {split_summary['train_patients']} | val patients: {split_summary['val_patients']} | "
        f"val_ratio≈{split_summary['val_ratio']}"
    )
    print(f"Device: {device}")

    best_val_f1 = -float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(int(args.epochs)):
        train_loader = DataLoader(
            train_subset,
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
        )

        # Unfreeze backbone after configured freeze epochs
        rebuilt_this_epoch = False

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
        )

        # Only step the epoch-level scheduler if we actually performed any optimizer.step()
        if optimizer_steps > 0 and not rebuilt_this_epoch:
            print(f"[Info] lr_scheduler.step() at epoch {epoch + 1}.")
            lr_scheduler.step()
        else:
            print(f"[Warning] No optimizer.step() executed in epoch {epoch + 1}; skipping lr_scheduler.step() to avoid PyTorch warning.")

        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss_classifier": float(avg_sublosses.get("loss_classifier", 0.0)),
            "loss_box_reg": float(avg_sublosses.get("loss_box_reg", 0.0)),
            "loss_objectness": float(avg_sublosses.get("loss_objectness", 0.0)),
            "loss_rpn_box_reg": float(avg_sublosses.get("loss_rpn_box_reg", 0.0)),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_tp": float(val_metrics["tp"]),
            "val_fp": float(val_metrics["fp"]),
            "val_fn": float(val_metrics["fn"]),
        }
        history.append(record)

        current_f1 = float(val_metrics["f1"])
        improved = current_f1 > (best_val_f1 + float(args.min_delta))

        if improved:
            best_val_f1 = current_f1
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
                "torchvision_model": "fasterrcnn_resnet50_fpn_v2",
                "anchor_sizes": str(args.anchor_sizes),
                "roi_batch_size_per_image": int(args.roi_batch_size_per_image),
                "roi_positive_fraction": float(args.roi_positive_fraction),
                "val_ratio": float(args.val_ratio),
                "val_score_threshold": float(args.val_score_threshold),
                "val_iou_threshold": float(args.val_iou_threshold),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "best_epoch": int(best_epoch),
                "best_val_precision": float(val_metrics["precision"]),
                "best_val_recall": float(val_metrics["recall"]),
                "best_val_f1": float(val_metrics["f1"]),
                "split_summary": split_summary,
            }
            # Save the best checkpoint, not the last one.
            save_checkpoint(save_path, model, meta)
            print(f"[Info] Saved best checkpoint to: {save_path}")
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch + 1:03d}/{int(args.epochs):03d} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_F1={val_metrics['f1']:.4f} | "
            f"lr={record['lr']:.6f}"
        )
        print(
            (
                f"  Train sub-losses: loss_classifier={record['loss_classifier']:.6f}, "
                f"loss_box_reg={record['loss_box_reg']:.6f}, "
                f"loss_objectness={record['loss_objectness']:.6f}, "
                f"loss_rpn_box_reg={record['loss_rpn_box_reg']:.6f}"
            )
        )
        print(
            f"  Val counts: TP={int(val_metrics['tp'])}, FP={int(val_metrics['fp'])}, FN={int(val_metrics['fn'])} | "
            f"best_F1={best_val_f1:.4f} (epoch {best_epoch})"
        )

        if int(args.patience) > 0 and no_improve_epochs >= int(args.patience):
            print(
                f"[EarlyStopping] val_F1 has not improved for {int(args.patience)} consecutive epochs. "
                f"Stopping at epoch {epoch + 1}."
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
        "torchvision_model": "fasterrcnn_resnet50_fpn_v2",
        "anchor_sizes": str(args.anchor_sizes),
        "roi_batch_size_per_image": int(args.roi_batch_size_per_image),
        "roi_positive_fraction": float(args.roi_positive_fraction),
        "val_ratio": float(args.val_ratio),
        "val_score_threshold": float(args.val_score_threshold),
        "val_iou_threshold": float(args.val_iou_threshold),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1 if best_val_f1 != -float("inf") else 0.0),
        "split_summary": split_summary,
    }

    print(f"Best checkpoint saved at: {save_path}")
    print(f"Best epoch: {best_epoch}, best val_F1: {final_meta['best_val_f1']:.4f}")
    print(json.dumps(final_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Running time: {time.time() - start_time} s.")


r"""log

Here gives the output log:



"""