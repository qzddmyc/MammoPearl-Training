# use:
# python src/data/bounding-box/bbox-train-getBad.py --epochs 1 --batch-size 2

# 作用：预跑一遍，将 bad data (loss !== !Infinite) 保存到 csv 文件中

# 当前使用的模型为：fasterrcnn_mobilenet_v3_large_fpn，可以对应 bbox-train-mobilenet.py 训练程序。
# 保存的文件为：./bad_data_record_mobilenet.csv

"""
Here is the output for fasterrcnn_mobilenet_v3_large_fpn model:

Total images: 16000; positives: 1411; negatives: 14589
[Sum] count = 5022                                                                                                                           
[Sum] bad data count = 2978
"""

"""Train script wrapper that records bad data entries when encountered.

Run this script just like `bbox-train.py`. It behaves the same but whenever a
batch produces non-finite losses (bad data), it records the corresponding
`patient_id,image_id` pairs into `bad_data_record(.*)?.csv` in this directory.
"""

from __future__ import annotations

import argparse
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
# or use model: fasterrcnn_mobilenet_v3_large_320_fpn

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def read_image_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
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
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            boxes: np.ndarray
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)

            image_path = images_root / str(patient_id) / f"{image_id}"

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
    try:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
        )
    except TypeError:
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
    bad_record_path: Path,
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

        # Detect non-finite losses
        bad_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
        if bad_keys:
            bad_keys_count += 1
            # analyze each target to try to identify which image(s) contain invalid boxes
            try:
                ds = loader.dataset
                orig = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds

                def analyze_target(t, sample: Sample) -> List[str]:
                    reasons: List[str] = []
                    try:
                        boxes = t.get("boxes")
                        if boxes is None:
                            reasons.append("no_boxes_field")
                            return reasons
                        boxes_np = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
                        if boxes_np.size == 0:
                            return reasons
                        if np.isnan(boxes_np).any():
                            reasons.append("nan_in_boxes")
                        if np.isinf(boxes_np).any():
                            reasons.append("inf_in_boxes")
                        # widths/heights
                        ws = boxes_np[:, 2] - boxes_np[:, 0]
                        hs = boxes_np[:, 3] - boxes_np[:, 1]
                        if (ws <= 0).any() or (hs <= 0).any():
                            reasons.append("non_positive_wh")
                        # check bounds using actual image size
                        try:
                            img = normalize_image(read_image_unicode(sample.image_path))
                            h, w = img.shape[:2]
                            if (boxes_np[:, 0] < 0).any() or (boxes_np[:, 1] < 0).any() or (boxes_np[:, 2] > w).any() or (boxes_np[:, 3] > h).any():
                                reasons.append("out_of_bounds")
                        except Exception:
                            # if image can't be read, note it
                            reasons.append("img_read_error")
                    except Exception:
                        reasons.append("analyze_error")
                    return reasons

                with bad_record_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for t in targets:
                        try:
                            img_idx = int(t["image_id"].item())
                            sample = orig.samples[img_idx]
                            reasons = analyze_target(t, sample)
                            if reasons:
                                writer.writerow([sample.patient_id, sample.image_id, ";".join(reasons)])
                            else:
                                # if we couldn't find per-target issues, still record generic bad_keys
                                writer.writerow([sample.patient_id, sample.image_id, ",".join(bad_keys)])
                        except Exception:
                            continue
            except Exception:
                pass

            optimizer.zero_grad(set_to_none=True)
            continue

        loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(loss):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bbox detector for VinDr (get bad data).")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
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

    bad_record_path = Path(__file__).resolve().parent / "bad_data_record_mobilenet.csv"
    # ensure header exists
    if not bad_record_path.exists():
        with bad_record_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "image_id", "reason"])

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=args.positive_only,
    )

    # Build index lists
    pos_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size > 0]
    neg_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size == 0]

    print(f"Total images: {len(train_dataset)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")

    model = build_model(num_classes=2)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        # Same simple loop as original but primarily this script's job is to record bad data
        subset = train_dataset
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        _ = train_one_epoch(model, loader, optimizer, device, epoch, args.epochs, bad_record_path)


if __name__ == "__main__":
    main()
