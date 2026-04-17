# use:
# python src/data/bounding-box/bbox-preTrain-getBad-resnet50.py --epochs 1 --batch-size 2

# 作用：预跑一遍，将 bad data (loss !== !Infinite) 保存到 csv 文件中

# 当前使用的模型为：fasterrcnn_resnet50_fpn_v2
# 保存的文件为：./bad_data_record_resnet50.csv

"""
This script is adapted from bbox-preTrain-getBad.py but uses
fasterrcnn_resnet50_fpn_v2 for the ResNet50 training flow.

This script runs one or more epochs over the training split and records any
image-level examples which produce non-finite losses (bad entries) into
`bad_data_record_resnet50.csv` for later inspection / filtering.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import math

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
except Exception:  # pragma: no cover
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_v2
    FasterRCNN_ResNet50_FPN_V2_Weights = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]

from torchvision.models.detection.rpn import AnchorGenerator



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
            raise ValueError(f"No image samples could be built from {csv_path} with split={split_name!r}")

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


def build_model(
    num_classes: int = 2,
    anchor_sizes: List[Tuple[int, ...]] | None = None,
) -> FasterRCNN:
    # prefer to load torchvision v2 default weights when available
    if anchor_sizes is None:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))

    aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(len(anchor_sizes))])
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    weights = None
    if FasterRCNN_ResNet50_FPN_V2_Weights is not None:
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_anchor_generator=anchor_generator,
        )
    except TypeError:
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
    """Create SGD optimizer with separate weight decay for biases/BN and others.

    This mirrors the optimizer grouping used in `bbox-train-resnet50.py` so that
    preTrain runs use the same weight-decay behavior.
    """
    decay = float(args.weight_decay)
    base_lr = float(base_lr) if base_lr is not None else float(args.lr)

    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        # treat biases and batch-norm / norm layers as no-decay
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
    parser = argparse.ArgumentParser(description="Train a bbox detector for VinDr (get bad data) - resnet50 v2.")
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
    # ROI sampling tuning (align with train behavior)
    parser.add_argument("--roi-batch-size-per-image", type=int, default=512)
    parser.add_argument("--roi-positive-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()

    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    save_path = args.save_path or (root / "models" / "bbox.pth")

    bad_record_path = Path(__file__).resolve().parent / "bad_data_record_resnet50.csv"
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

    pos_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size > 0]
    neg_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size == 0]

    print(f"Total images: {len(train_dataset)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")

    model = build_model(num_classes=2)
    model.to(device)

    # Apply ROI / RPN tuning to encourage similar ROI sampling as train
    try:
        bs = int(args.roi_batch_size_per_image)
        model.roi_heads.batch_size_per_image = bs
        pf = float(args.roi_positive_fraction)
        desired_pos = math.ceil(bs * pf) if bs > 0 else 0
        pos_frac = (desired_pos / bs) if bs > 0 else pf
        model.roi_heads.positive_fraction = float(pos_frac)
    except Exception:
        pass

    # Use same optimizer grouping as train for consistent weight-decay behavior
    optimizer = create_optimizer(model, args, base_lr=float(args.lr))

    for epoch in range(args.epochs):
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
