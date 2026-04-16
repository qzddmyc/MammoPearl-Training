# 使用 fasterrcnn_mobilenet_v3_large_fpn 模型对数据集进行训练

# 样本采用了 neg:pos = 1:1 的采样比例

"""Train a breast lesion bounding-box detector from VinDr detection CSV.

This script reads `data/raw/vindr_detection_folds.csv`, matches each row to
`data/processed/images_png/<patient_id>/<image_id>`, and trains a Faster
R-CNN detector to predict lesion bounding boxes (xmin, ymin, xmax, ymax).

Model checkpoint is saved to `models/bbox.pth`.
"""

from __future__ import annotations

import argparse
import time
import json
import math
import random
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn

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


def build_model(num_classes: int = 2) -> FasterRCNN:
    """Build a Faster R-CNN detector with a single foreground class."""
    try:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
        )
    except TypeError:
        # Compatibility with older torchvision versions.
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)  # type: ignore[call-arg]

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: FasterRCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    bad_keys_count = 0

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # !!! > 筛选无效数据并清除
        bad_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
        if bad_keys:
            bad_keys_count += 1
            # pbar.write(f"[Warning] non-finite loss in {bad_keys}, batch skipped")
            # pbar.write(str(targets))
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(loss):
            pbar.write(f"[Warning] 捕获到 not-Infinite Loss，跳过此 Batch 避免模型崩溃！")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = float(loss.item())
        running_loss += batch_loss
        count += 1
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    print(f"[Sum] count = {count}")
    print(f"[Sum] bad data count = {bad_keys_count}")

    return running_loss / max(count, 1)


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
        help="Output checkpoint path (default: models/bbox.pth)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Drop negative images (images without any bbox).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()

    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox.pth")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=args.positive_only,
    )

    # Read bad data record (if exists) and build a set of (patient_id,image_id)
    bad_record_path = Path(__file__).resolve().parent / "bad_data_record_mobilenet.csv"
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
    # Build index lists of positive / negative images so we can form
    # per-epoch subsets with a fixed negative:positive ratio.
    # Filter out bad samples recorded earlier
    pos_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size > 0 and (s.patient_id, s.image_id) not in bad_set]
    neg_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size == 0 and (s.patient_id, s.image_id) not in bad_set]

    if len(pos_indices) == 0:
        # If no positive samples are present, fall back to using the full dataset.
        print("Warning: no positive samples found in training split; using full dataset")
        pos_indices = []

    model = build_model(num_classes=2)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    history: List[Dict[str, float]] = []

    print(f"Total images: {len(train_dataset)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(f"Device: {device}")
    for epoch in range(args.epochs):
        # For each epoch, build a 1:1 positive:negative subset.
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            rng = random.Random(args.seed + epoch)
            # ensure epoch-level 1:1 ratio by sampling equal counts
            pair_count = min(len(pos_indices), len(neg_indices))
            if pair_count == 0:
                subset = train_dataset
            else:
                pos_sample = rng.sample(pos_indices, pair_count)
                neg_sample = rng.sample(neg_indices, pair_count)
                rng.shuffle(pos_sample)
                rng.shuffle(neg_sample)

                # Build batches that mostly contain both positive and negative
                B = max(1, args.batch_size)
                pos_ptr = 0
                neg_ptr = 0
                epoch_order: List[int] = []
                while pos_ptr < len(pos_sample) or neg_ptr < len(neg_sample):
                    # choose at least one positive if available
                    if pos_ptr < len(pos_sample):
                        p = min(max(1, B // 2), len(pos_sample) - pos_ptr)
                    else:
                        p = 0
                    q = B - p
                    batch = []
                    if p > 0:
                        batch.extend(pos_sample[pos_ptr: pos_ptr + p])
                        pos_ptr += p
                    take_neg = min(q, len(neg_sample) - neg_ptr)
                    if take_neg > 0:
                        batch.extend(neg_sample[neg_ptr: neg_ptr + take_neg])
                        neg_ptr += take_neg
                    # fill remaining slots if any
                    if len(batch) < B:
                        need = B - len(batch)
                        more_neg = min(need, len(neg_sample) - neg_ptr)
                        if more_neg > 0:
                            batch.extend(neg_sample[neg_ptr: neg_ptr + more_neg])
                            neg_ptr += more_neg
                        need = B - len(batch)
                        if need > 0 and pos_ptr < len(pos_sample):
                            more_pos = min(need, len(pos_sample) - pos_ptr)
                            batch.extend(pos_sample[pos_ptr: pos_ptr + more_pos])
                            pos_ptr += more_pos
                    epoch_order.extend(batch)

                subset = torch.utils.data.Subset(train_dataset, epoch_order)
        else:
            # No positives or no negatives -> fall back to full dataset
            subset = train_dataset

        train_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        lr_scheduler.step()

        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} | loss={avg_loss:.4f} | lr={record['lr']:.6f}")

    meta = {
        "task": "bbox_detection",
        "num_classes": 2,
        "class_names": ["background", "lesion"],
        "csv_path": str(csv_path),
        "images_root": str(images_root),
        "positive_only": args.positive_only,
        "history": history,
        "torchvision_model": "fasterrcnn_mobilenet_v3_large_fpn",
    }
    save_checkpoint(save_path, model, meta)
    print(f"Saved checkpoint to: {save_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Running time: {time.time() - start_time} s.")


r"""log

Here gives the output log:

Total images: 16000; positives: 1411; negatives: 8633
Device: cpu
[Sum] count = 1411
[Sum] bad data count = 0
Epoch 001/012 | loss=0.2842 | lr=0.001000
[Sum] count = 1411
[Sum] bad data count = 0
Epoch 002/012 | loss=0.1967 | lr=0.001000
[Sum] count = 1411
[Sum] bad data count = 0
Epoch 003/012 | loss=0.2089 | lr=0.001000
[Sum] count = 1411
[Sum] bad data count = 0
Epoch 004/012 | loss=0.2081 | lr=0.000100
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 005/012 | loss=0.2012 | lr=0.000100
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 006/012 | loss=0.1991 | lr=0.000100
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 007/012 | loss=0.2027 | lr=0.000100
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 008/012 | loss=0.2058 | lr=0.000010
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 009/012 | loss=0.1966 | lr=0.000010
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 010/012 | loss=0.1988 | lr=0.000010
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 011/012 | loss=0.2010 | lr=0.000010
[Sum] count = 1411                                                                                                                           
[Sum] bad data count = 0
Epoch 012/012 | loss=0.2008 | lr=0.000001
Saved checkpoint to: D:\Codes\Github_Repositories\MammoPearl-Training\models\bbox.pth
{
  "task": "bbox_detection",
  "num_classes": 2,
  "class_names": [
    "background",
    "lesion"
  ],
  "csv_path": "D:\\Codes\\Github_Repositories\\MammoPearl-Training\\data\\raw\\vindr_detection_folds.csv",
  "images_root": "D:\\Codes\\Github_Repositories\\MammoPearl-Training\\data\\processed\\images_png",
  "positive_only": false,
  "history": [
    {
      "epoch": 1.0,
      "train_loss": 0.28416168339574255,
      "lr": 0.001
    },
    {
      "epoch": 2.0,
      "train_loss": 0.19672592568873282,
      "lr": 0.001
    },
    {
      "epoch": 3.0,
      "train_loss": 0.20887250005195615,
      "lr": 0.001
    },
    {
      "epoch": 4.0,
      "train_loss": 0.2080930071251901,
      "lr": 0.0001
    },
    {
      "epoch": 5.0,
      "train_loss": 0.20123556787365932,
      "lr": 0.0001
    },
    {
      "epoch": 6.0,
      "train_loss": 0.1991037121590446,
      "lr": 0.0001
    },
    {
      "epoch": 7.0,
      "train_loss": 0.2027248712005734,
      "lr": 0.0001
    },
    {
      "epoch": 8.0,
      "train_loss": 0.2057937145718947,
      "lr": 1e-05
    },
    {
      "epoch": 9.0,
      "train_loss": 0.19655466429799437,
      "lr": 1e-05
    },
    {
      "epoch": 10.0,
      "train_loss": 0.19881717764827259,
      "lr": 1e-05
    },
    {
      "epoch": 11.0,
      "train_loss": 0.20098607952051092,
      "lr": 1e-05
    },
    {
      "epoch": 12.0,
      "train_loss": 0.20081896507257133,
      "lr": 1.0000000000000002e-06
    }
  ],
  "torchvision_model": "fasterrcnn_mobilenet_v3_large_fpn"
}
Running time: 29041.806410312653 s.

"""
