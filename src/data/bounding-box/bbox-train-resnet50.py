# 使用 fasterrcnn_resnet50_fpn 模型对数据集进行训练
# 这个样本中的 bad data 使用 bad_data_record_resnet50.csv

# 使用这个模型去训练会耗费很长的时间，需要注意

# If your computer is GREAT, use this to run fuckingly:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 8 --fuck-running --lr 0.005 --freeze-epochs 4

# Otherwise, use:
# python src/data/bounding-box/bbox-train-resnet50.py --epochs 12 --batch-size 2 --accumulation-steps 4 --lr 0.005 --freeze-epochs 4

"""Train a breast lesion bounding-box detector from VinDr detection CSV.

This script reads `data/raw/vindr_detection_folds.csv`, matches each row to
`data/processed/images_png/<patient_id>/<image_id>`, and trains a Faster
R-CNN detector to predict lesion bounding boxes (xmin, ymin, xmax, ymax).

Model checkpoint is saved to `models/bbox_resnet50.pth`.


Update prompts:
1. 针对训练后期 Loss 卡在 0.19 左右无法下降的问题，需从学习率策略、
   模型结构和优化器等方面进行系统性干预，打破局部最优解。
2. 引入学习率 Warmup 机制，防止初始训练时因梯度过大破坏预训练权重，并提供合理的初始学习率设定。
3. 增加权重衰减（Weight Decay）的配置参数，通过正则化手段有效防止模型在较小数据集上过拟合。
4. 新增命令行参数 "--fuck-running" 作为算力切换开关：
   当不含此参数时，代码需在 batch_size=2 的前提下通过累积 4 个 step 再执行 optimizer.step()
   来变相实现 batch_size=8 的梯度累积；
   当存在该参数时，直接使用配置的较大 batch_size 进行正常训练，
   同时两种模式下都必须保持每个 batch 内合理的正负样本混合比例。
 * fix: 在算得正样本时，需要按照比例上取整，以防止正样本丢失。
5. 针对 912x1520 的高分辨率医疗影像数据，修改模型的 AnchorGenerator，为其添加 8 和 16 这
   样更小的 scale 尺寸，以强化微小病灶的检测能力。6. 优化训练策略，除了在 DataLoader 端保持正
   负样本比之外，还需通过调整模型内部的 ROI 采样比例等参数，变相实现 Hard Negative Mining（挖掘难例）。
7. 实现渐冻层训练策略：在训练初期主动冻结 ResNet 的 layer1 和 layer2 层，仅训练 FPN 和检测头；
   在设定的几轮 Epoch 之后，全量解冻这些底层网络进行全局微调。
8. 强制采用 torchvision 中的 fasterrcnn_resnet50_fpn_v2 版本模型，以利用其更先进的
   数据增强策略和优化过的 FPN 特征提取结构。
 * feat: 需要使用 FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 权重。 

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
) -> float:
    """Train one epoch with gradient accumulation and optional iter-level LinearLR warmup."""
    model.train()
    running_loss = 0.0
    count = 0
    bad_keys_count = 0

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

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        # gradient accumulation (scale before backward)
        scaled_loss = loss / float(accumulation_steps)
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            rng = random.Random(int(args.seed) + epoch)
            pair_count = min(len(pos_indices), len(neg_indices))
            if pair_count == 0:
                subset = train_dataset
            else:
                pos_sample = rng.sample(pos_indices, pair_count)
                neg_sample = rng.sample(neg_indices, pair_count)
                rng.shuffle(pos_sample)
                rng.shuffle(neg_sample)

                B = max(1, int(args.batch_size))
                pos_ptr = 0
                neg_ptr = 0
                epoch_order: List[int] = []
                while pos_ptr < len(pos_sample) or neg_ptr < len(neg_sample):
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
            subset = train_dataset

        # If we've built an ordered Subset above, avoid DataLoader shuffling to preserve per-batch pos/neg mixing
        use_shuffle = not isinstance(subset, torch.utils.data.Subset)

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

        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, int(args.epochs), accumulation_steps, warmup_scheduler)

        # Unfreeze backbone after configured freeze epochs
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

        lr_scheduler.step()

        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(f"Epoch {epoch + 1:03d}/{int(args.epochs):03d} | loss={avg_loss:.4f} | lr={record['lr']:.6f}")

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
    main()


r"""log

Here gives the output log:



"""