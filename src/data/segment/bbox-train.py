"""Train a breast lesion bounding-box detector from VinDr detection CSV.

This script reads `data/raw/vindr_detection_folds.csv`, matches each row to
`data/processed/images_png/<patient_id>/<image_id>`, and trains a Faster
R-CNN detector to predict lesion bounding boxes (xmin, ymin, xmax, ymax).

Model checkpoint is saved to `models/bbox.pth`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
# from torchvision.models.detection import fasterrcnn_resnet50_fpn

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


def scale_boxes_to_image(
    boxes: np.ndarray,
    orig_size: Tuple[float, float],
    new_size: Tuple[int, int],
) -> np.ndarray:
    """Scale bbox coordinates from original CSV size to actual image size."""
    orig_h, orig_w = orig_size
    new_h, new_w = new_size
    if orig_h <= 0 or orig_w <= 0:
        return boxes

    scale_x = float(new_w) / float(orig_w)
    scale_y = float(new_h) / float(orig_h)
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y

    # Clamp to valid image range and remove invalid boxes.
    scaled[:, 0::2] = np.clip(scaled[:, 0::2], 0, max(new_w - 1, 0))
    scaled[:, 1::2] = np.clip(scaled[:, 1::2], 0, max(new_h - 1, 0))
    keep = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
    return scaled[keep]


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
        
        # 排除空值和无病灶（'No Finding'）的数据，不参与训练
        df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
        if "finding_categories" in df.columns:
            df = df[df["finding_categories"] != "['No Finding']"]

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

            if positive_only and len(boxes) == 0:
                continue

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            image_path = images_root / str(patient_id) / f"{image_id}"
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

        boxes = sample.boxes
        if boxes.size > 0:
            boxes = scale_boxes_to_image(boxes, sample.orig_size, (h, w))

        if boxes.size == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.from_numpy(boxes.astype(np.float32))
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


# def build_model(num_classes: int = 2) -> FasterRCNN:
#     try:
#         model = fasterrcnn_resnet50_fpn(
#             weights=None,
#             weights_backbone=None,
#         )
#     except TypeError:
#         model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model


def build_model(num_classes: int = 2) -> FasterRCNN:
    """Build a Faster R-CNN detector with a single foreground class."""
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
        )
    except TypeError:
        # Compatibility with older torchvision versions.
        model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False)  # type: ignore[call-arg]

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

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        running_loss += batch_loss
        count += 1
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

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

    print(f"Training images: {len(train_dataset)}")
    print(f"Device: {device}")
    for epoch in range(args.epochs):
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
        # "torchvision_model": "fasterrcnn_resnet50_fpn",
        "torchvision_model": "fasterrcnn_mobilenet_v3_large_320_fpn",
    }
    save_checkpoint(save_path, model, meta)
    print(f"Saved checkpoint to: {save_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""log

Here gives the output log:

Training images: 1411
Device: cpu
Epoch 001/010 | loss=0.1595 | lr=0.005000                                                                                                    
Epoch 002/010 | loss=0.1239 | lr=0.005000                                                                                                    
Epoch 003/010 | loss=0.1333 | lr=0.000500                                                                                                    
Epoch 004/010 | loss=0.1388 | lr=0.000500                                                                                                    
Epoch 005/010 | loss=0.1379 | lr=0.000500                                                                                                    
Epoch 006/010 | loss=0.1411 | lr=0.000050                                                                                                    
Epoch 007/010 | loss=0.1425 | lr=0.000050                                                                                                    
Epoch 008/010 | loss=0.1426 | lr=0.000050                                                                                                    
Epoch 009/010 | loss=0.1422 | lr=0.000005                                                                                                    
Epoch 010/010 | loss=0.1435 | lr=0.000005                                                                                                    
Saved checkpoint to: @/models/bbox.pth
{
  "task": "bbox_detection",
  "num_classes": 2,
  "class_names": [
    "background",
    "lesion"
  ],
  "csv_path": "@data/raw/vindr_detection_folds.csv",
  "images_root": "@/data/processed/images_png",
  "positive_only": false,
  "history": [
    {
      "epoch": 1.0,
      "train_loss": 0.15954284596839977,
      "lr": 0.005
    },
    {
      "epoch": 2.0,
      "train_loss": 0.12386478695796839,
      "lr": 0.005
    },
    {
      "epoch": 3.0,
      "train_loss": 0.1333375834983908,
      "lr": 0.0005
    },
    {
      "epoch": 4.0,
      "train_loss": 0.13882978673148424,
      "lr": 0.0005
    },
    {
      "epoch": 5.0,
      "train_loss": 0.13788776492789684,
      "lr": 0.0005
    },
    {
      "epoch": 6.0,
      "train_loss": 0.14112772106090113,
      "lr": 5e-05
    },
    {
      "epoch": 7.0,
      "train_loss": 0.14252236584153966,
      "lr": 5e-05
    },
    {
      "epoch": 8.0,
      "train_loss": 0.14263700727616085,
      "lr": 5e-05
    },
    {
      "epoch": 9.0,
      "train_loss": 0.14224379673262494,
      "lr": 5e-06
    },
    {
      "epoch": 10.0,
      "train_loss": 0.14345879639596507,
      "lr": 5e-06
    }
  ],
  "torchvision_model": "fasterrcnn_mobilenet_v3_large_320_fpn"
}

"""