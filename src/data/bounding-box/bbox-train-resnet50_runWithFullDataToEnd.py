# 这个版本没有 validation 集的划分与"提早终止"的代码，但是去除了对单个 epoch 中正负样本 1:1 的分配。

# run:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 --roi-batch-size-per-image 256 --roi-positive-fraction 0.1


from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset
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


# def build_model(num_classes: int = 2) -> FasterRCNN:
#     """Build a Faster R-CNN detector with a single foreground class."""
#     try:
#         model = fasterrcnn_mobilenet_v3_large_320_fpn(
#             weights=None,
#             weights_backbone=None,
#         )
#     except TypeError:
#         # Compatibility with older torchvision versions.
#         model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False)  # type: ignore[call-arg]

#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model


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
        help="Output checkpoint path (default: models/bbox_resnet50.pth)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
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
    train_dataset = VinDrBboxDataset(
        csv_path=csv_path,
        images_root=images_root,
        split_name="training",
        positive_only=args.positive_only,
    )

    # Read bad data record (if exists) and build a set of (patient_id,image_id)
    bad_record_path = Path(__file__).resolve().parent / "bad_data_record_resnet50.csv"
    bad_set = set()
    if bad_record_path.exists():
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

    pos_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size > 0 and (s.patient_id, s.image_id) not in bad_set]
    neg_indices = [i for i, s in enumerate(train_dataset.samples) if s.boxes.size == 0 and (s.patient_id, s.image_id) not in bad_set]

    if len(pos_indices) == 0:
        print("Warning: no positive samples found in training split; using full dataset")
        pos_indices = []

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

    print(f"Total images: {len(train_dataset)}; positives: {len(pos_indices)}; negatives: {len(neg_indices)}")
    print(f"Device: {device}")
    for epoch in range(int(args.epochs)):
        # For each epoch, build a 1:1 positive:negative subset if possible
        # if len(pos_indices) > 0 and len(neg_indices) > 0:
        #     rng = random.Random(int(args.seed) + epoch)
        #     pair_count = min(len(pos_indices), len(neg_indices))
        #     if pair_count == 0:
        #         subset = train_dataset
        #     else:
        #         pos_sample = rng.sample(pos_indices, pair_count)
        #         neg_sample = rng.sample(neg_indices, pair_count)
        #         rng.shuffle(pos_sample)
        #         rng.shuffle(neg_sample)

        #         B = max(1, int(args.batch_size))
        #         pos_ptr = 0
        #         neg_ptr = 0
        #         epoch_order: List[int] = []
        #         while pos_ptr < len(pos_sample) or neg_ptr < len(neg_sample):
        #             if pos_ptr < len(pos_sample):
        #                 p = min(max(1, B // 2), len(pos_sample) - pos_ptr)
        #             else:
        #                 p = 0
        #             q = B - p
        #             batch = []
        #             if p > 0:
        #                 batch.extend(pos_sample[pos_ptr: pos_ptr + p])
        #                 pos_ptr += p
        #             take_neg = min(q, len(neg_sample) - neg_ptr)
        #             if take_neg > 0:
        #                 batch.extend(neg_sample[neg_ptr: neg_ptr + take_neg])
        #                 neg_ptr += take_neg
        #             if len(batch) < B:
        #                 need = B - len(batch)
        #                 more_neg = min(need, len(neg_sample) - neg_ptr)
        #                 if more_neg > 0:
        #                     batch.extend(neg_sample[neg_ptr: neg_ptr + more_neg])
        #                     neg_ptr += more_neg
        #                 need = B - len(batch)
        #                 if need > 0 and pos_ptr < len(pos_sample):
        #                     more_pos = min(need, len(pos_sample) - pos_ptr)
        #                     batch.extend(pos_sample[pos_ptr: pos_ptr + more_pos])
        #                     pos_ptr += more_pos
        #             epoch_order.extend(batch)

        #         subset = torch.utils.data.Subset(train_dataset, epoch_order)
        # else:
        #     subset = train_dataset

        # # If we've built an ordered Subset above, avoid DataLoader shuffling to preserve per-batch pos/neg mixing
        # use_shuffle = not isinstance(subset, torch.utils.data.Subset)

        subset = train_dataset
        use_shuffle = True

        train_loader = DataLoader(
            subset,
            batch_size=int(args.batch_size),
            shuffle=use_shuffle,
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

        avg_loss, optimizer_steps, avg_sublosses = train_one_epoch(model, train_loader, optimizer, device, epoch, int(args.epochs), accumulation_steps, warmup_scheduler)

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
        }
        history.append(record)
        print(f"Epoch {epoch + 1:03d}/{int(args.epochs):03d} | loss={avg_loss:.4f} | lr={record['lr']:.6f}")
        print(
            (
                f"  Sub-losses: loss_classifier={record['loss_classifier']:.6f}, "
                f"loss_box_reg={record['loss_box_reg']:.6f}, "
                f"loss_objectness={record['loss_objectness']:.6f}, "
                f"loss_rpn_box_reg={record['loss_rpn_box_reg']:.6f}"
            )
        )

    meta = {
        "task": "bbox_detection",
        "num_classes": 2,
        "class_names": ["background", "lesion"],
        "csv_path": str(csv_path),
        "images_root": str(images_root),
        "positive_only": args.positive_only,
        "history": history,
        "torchvision_model": "fasterrcnn_resnet50_fpn_v2",
        "anchor_sizes": str(args.anchor_sizes),
        "roi_batch_size_per_image": int(args.roi_batch_size_per_image),
        "roi_positive_fraction": float(args.roi_positive_fraction),
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
root@autodl-container-e0dc46b58e-0f0a0b5b:~/autodl-tmp/MammoPearl-Training# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4 --roi-batch-size-per-image 256 --roi-positive-fraction 0.1
[Warning] Found 1 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/118092d6244fabf9ac376d580aac8cbb/679a6515593f9692eb6a622b7b6b0aa0.png
[Warning] Found 1 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/9870e0438ed1f19cb85aaa32cf1e8830/ea8790b418b754139457d48e5228c077.png
[Warning] Found 1 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/9efd5f5c65ec8c402ec48f0ff7388562/5cd330a56ea7c77a1fa1181712966dbf.png
[Warning] Found 1 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/a8cb9adadef00fc473b1760cd7b513e4/7323314998471cde47c6fba70ae6d32c.png
[Warning] Found 2 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/df7c81d477ed6a29aa8e6e49c1719d03/7be2e5f9ade68ae2bfe4e5edb4968654.png
[Warning] Found 1 invalid boxes in /root/autodl-tmp/MammoPearl-Training/data/processed/images_png/df7c81d477ed6a29aa8e6e49c1719d03/db70c6572ab2f8633feb484aa6d7d38d.png
Total images: 16000; positives: 1411; negatives: 14589
Device: cuda
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 001/012 | loss=0.0647 | lr=0.004915
  Sub-losses: loss_classifier=0.016036, loss_box_reg=0.002185, loss_objectness=0.041585, loss_rpn_box_reg=0.004922
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 002/012 | loss=0.0185 | lr=0.004665
  Sub-losses: loss_classifier=0.006629, loss_box_reg=0.003277, loss_objectness=0.004649, loss_rpn_box_reg=0.003902
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 003/012 | loss=0.0177 | lr=0.004268
  Sub-losses: loss_classifier=0.006542, loss_box_reg=0.003420, loss_objectness=0.003973, loss_rpn_box_reg=0.003738
[Sum] count = 8000
[Sum] bad data count = 0
[Info] Unfreezing backbone layers after 4 epochs and rebuilding optimizer
[Warning] No optimizer.step() executed in epoch 4; skipping lr_scheduler.step() to avoid PyTorch warning.
Epoch 004/012 | loss=0.0176 | lr=0.004268
  Sub-losses: loss_classifier=0.006655, loss_box_reg=0.003638, loss_objectness=0.003820, loss_rpn_box_reg=0.003520
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 005/012 | loss=0.0182 | lr=0.004106
  Sub-losses: loss_classifier=0.006986, loss_box_reg=0.003851, loss_objectness=0.003879, loss_rpn_box_reg=0.003465
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 006/012 | loss=0.0182 | lr=0.003643
  Sub-losses: loss_classifier=0.007007, loss_box_reg=0.003904, loss_objectness=0.003810, loss_rpn_box_reg=0.003459
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 007/012 | loss=0.0181 | lr=0.002951
  Sub-losses: loss_classifier=0.007062, loss_box_reg=0.003964, loss_objectness=0.003727, loss_rpn_box_reg=0.003386
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 008/012 | loss=0.0182 | lr=0.002134
  Sub-losses: loss_classifier=0.007113, loss_box_reg=0.004024, loss_objectness=0.003721, loss_rpn_box_reg=0.003318
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 009/012 | loss=0.0180 | lr=0.001318
  Sub-losses: loss_classifier=0.006972, loss_box_reg=0.004004, loss_objectness=0.003721, loss_rpn_box_reg=0.003278
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 010/012 | loss=0.0179 | lr=0.000626
  Sub-losses: loss_classifier=0.006942, loss_box_reg=0.003949, loss_objectness=0.003731, loss_rpn_box_reg=0.003288
[Sum] bad data count = 0
Epoch 011/012 | loss=0.0179 | lr=0.000163
  Sub-losses: loss_classifier=0.006931, loss_box_reg=0.004060, loss_objectness=0.003640, loss_rpn_box_reg=0.003243
[Sum] count = 8000
[Sum] bad data count = 0
Epoch 012/012 | loss=0.0179 | lr=0.000001
  Sub-losses: loss_classifier=0.006967, loss_box_reg=0.004044, loss_objectness=0.003688, loss_rpn_box_reg=0.003238
Saved checkpoint to: /root/autodl-tmp/MammoPearl-Training/models/bbox_resnet50.pth
{
  "task": "bbox_detection",
  "num_classes": 2,
  "class_names": [
    "background",
    "lesion"
  ],
  "csv_path": "/root/autodl-tmp/MammoPearl-Training/data/raw/vindr_detection_folds.csv",
  "images_root": "/root/autodl-tmp/MammoPearl-Training/data/processed/images_png",
  "positive_only": false,
  "history": [
    {
      "epoch": 1.0,
      "train_loss": 0.06472857511289112,
      "lr": 0.004914831602809505,
      "loss_classifier": 0.016036115038230492,
      "loss_box_reg": 0.0021851819014642613,
      "loss_objectness": 0.04158486384589923,
      "loss_rpn_box_reg": 0.004922414270649369
    },
    {
      "epoch": 2.0,
      "train_loss": 0.0184573096505992,
      "lr": 0.004665130496759185,
      "loss_classifier": 0.006628764671648241,
      "loss_box_reg": 0.003277019441185985,
      "loss_objectness": 0.004649497100574081,
      "loss_rpn_box_reg": 0.003902028408299884
    },
    {
      "epoch": 3.0,
      "train_loss": 0.01767272539901751,
      "lr": 0.004267913399575757,
      "loss_classifier": 0.006541526366305334,
      "loss_box_reg": 0.0034204899241792644,
      "loss_objectness": 0.003972510376355785,
      "loss_rpn_box_reg": 0.0037381986988425523
    },
    {
      "epoch": 4.0,
      "train_loss": 0.01763239026899464,
      "lr": 0.004267913399575757,
      "loss_classifier": 0.00665487932846554,
      "loss_box_reg": 0.0036382419285605466,
      "loss_objectness": 0.0038195468072117363,
      "loss_rpn_box_reg": 0.0035197221993382754
    },
    {
      "epoch": 5.0,
      "train_loss": 0.018180692951746097,
      "lr": 0.004105513678220977,
      "loss_classifier": 0.006985727310020593,
      "loss_box_reg": 0.0038508380654293435,
      "loss_objectness": 0.00387930629272887,
      "loss_rpn_box_reg": 0.003464821274677888
    },
    {
      "epoch": 6.0,
      "train_loss": 0.018180612547472264,
      "lr": 0.00364303839957576,
      "loss_classifier": 0.007007280627460205,
      "loss_box_reg": 0.00390375095771833,
      "loss_objectness": 0.00381049067861386,
      "loss_rpn_box_reg": 0.0034590902723043654
    },
    {
      "epoch": 7.0,
      "train_loss": 0.018139016264714883,
      "lr": 0.0029508952324650015,
      "loss_classifier": 0.0070620254663135715,
      "loss_box_reg": 0.003964037522728404,
      "loss_objectness": 0.0037272323083925585,
      "loss_rpn_box_reg": 0.003385720937035444
    },
    {
      "epoch": 8.0,
      "train_loss": 0.018176369273463933,
      "lr": 0.002134456699787879,
      "loss_classifier": 0.007113481099815544,
      "loss_box_reg": 0.004023658248861579,
      "loss_objectness": 0.003721292563666793,
      "loss_rpn_box_reg": 0.0033179373779717025
    },
    {
      "epoch": 9.0,
      "train_loss": 0.017976241969943657,
      "lr": 0.0013180181671107565,
      "loss_classifier": 0.006972108300677064,
      "loss_box_reg": 0.004004364301841292,
      "loss_objectness": 0.0037212742548763346,
      "loss_rpn_box_reg": 0.0032784950849704727
    },
    {
      "epoch": 10.0,
      "train_loss": 0.017909854239693233,
      "lr": 0.0006258749999999975,
      "loss_classifier": 0.006941743585968652,
      "loss_box_reg": 0.003948992292609831,
      "loss_objectness": 0.003731142747819831,
      "loss_rpn_box_reg": 0.003287975607737053
    },
    {
      "epoch": 11.0,
      "train_loss": 0.017873965756032704,
      "lr": 0.00016339972135478073,
      "loss_classifier": 0.006930558845862834,
      "loss_box_reg": 0.004060271261590572,
      "loss_objectness": 0.0036396742334763987,
      "loss_rpn_box_reg": 0.003243461453119835
    },
    {
      "epoch": 12.0,
      "train_loss": 0.017936786814731022,
      "lr": 1e-06,
      "loss_classifier": 0.006967201272960665,
      "loss_box_reg": 0.00404361561392713,
      "loss_objectness": 0.003687774866917607,
      "loss_rpn_box_reg": 0.003238195093804279
    }
  ],
  "torchvision_model": "fasterrcnn_resnet50_fpn_v2",
  "anchor_sizes": "8,16,32,64,128",
  "roi_batch_size_per_image": 256,
  "roi_positive_fraction": 0.1
}
Running time: 14347.519656181335 s.

"""


r"""test info:

root@autodl-container-e0dc46b58e-0f0a0b5b:~/autodl-tmp/MammoPearl-Training# python src/data/bounding-box/bbox-test-resnet50.py --ckpt-path models/bbox_resnet50.pth --score-threshold 0.9 --anchor-sizes 8,16,32,64,128
{
  "images": 4000,
  "gt_boxes": 447,
  "pred_boxes": 928,
  "tp": 12,
  "fp": 916,
  "fn": 435,
  "precision": 0.01293103448275862,
  "recall": 0.026845637583892617,
  "f1": 0.017454545454545452,
  "image_accuracy": 0.90225,
  "mean_iou": 0.6705907980600992,
  "mean_abs_error": {
    "xmin": 57.101043701171875,
    "ymin": 51.92363357543945,
    "xmax": 60.05610275268555,
    "ymax": 42.688533782958984
  }
}

root@autodl-container-e0dc46b58e-0f0a0b5b:~/autodl-tmp/MammoPearl-Training# python src/data/bounding-box/bbox-test-resnet50.py --ckpt-path models/bbox_resnet50.pth --score-threshold 0.5 --anchor-sizes 8,16,32,64,128
{
  "images": 4000,
  "gt_boxes": 447,
  "pred_boxes": 2018,
  "tp": 54,
  "fp": 1964,
  "fn": 393,
  "precision": 0.026759167492566897,
  "recall": 0.12080536912751678,
  "f1": 0.0438133874239351,
  "image_accuracy": 0.85325,
  "mean_iou": 0.6870159197736669,
  "mean_abs_error": {
    "xmin": 40.7904052734375,
    "ymin": 34.36684036254883,
    "xmax": 38.019290924072266,
    "ymax": 25.974227905273438
  }
}

"""