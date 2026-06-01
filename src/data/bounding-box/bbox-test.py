# use:
# python src/data/bounding-box/bbox-test.py \
#     --ckpt-path models/bbox_resnet50.pth \
#     --min-box-side 24.0 --max-box-ar 3.0
#
# * input_h/input_w, anchor_sizes, nms_thresh, focal_alpha/gamma are auto-read
#   from checkpoint meta.  Pass --min-box-side / --max-box-ar to match training.

"""Evaluate a bbox detector on the VinDr test split.

Mirrors the validation logic of bbox-train.py exactly:
  - same AR-preserving pad + resize preprocessing
  - same GT box loading (lesion type filter + min-box-side / max-box-ar)
  - same model architecture (RetinaNet + ResNet50-FPN)
  - F2 (beta=2) at [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] thresholds

Config is read from the checkpoint meta written by bbox-train.py.
Parameters not stored in meta (lesion types, box size filters) must be
supplied on the command line with values matching the training run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    _TORCHVISION_DET_OK = True
except ImportError:
    _TORCHVISION_DET_OK = False

try:
    from torchvision.models import ResNet50_Weights as _ResNet50Weights
    _IMAGENET_WEIGHTS: Any = _ResNet50Weights.DEFAULT
except ImportError:
    _IMAGENET_WEIGHTS = True  # pretrained=True fallback

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore[assignment]


# =============================================================================
# Utilities
# =============================================================================

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



# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Sample:
    patient_id: str
    image_id: str
    image_path: "Path"
    boxes: "np.ndarray"
    orig_size: tuple  # (H, W)


# =============================================================================
# GT loading — mirrors load_samples() in bbox-train.py exactly
# =============================================================================

def load_samples(
    csv_path, images_root, split_name,
    lesion_types=None, min_box_side=0.0, max_box_ar=float("inf"),
    input_w=1024, input_h=1024,
):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

    samples = []
    for (patient_id, _series_id, image_id), group in df.groupby(
        ["patient_id", "series_id", "image_id"], sort=True
    ):
        first = group.iloc[0]
        if lesion_types:
            type_mask = pd.Series(False, index=group.index)
            for lt in lesion_types:
                if lt in group.columns:
                    type_mask = type_mask | (group[lt] == 1)
            group = group[type_mask]
        valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
        boxes = valid.to_numpy(dtype=np.float32) if not valid.empty else np.zeros((0, 4), dtype=np.float32)
        image_path = images_root / str(patient_id) / f"{image_id}"

        if boxes.size > 0 and (min_box_side > 0.0 or max_box_ar < float("inf")):
            orig_w_val = float(first["width"]) if pd.notna(first["width"]) else float(input_w)
            orig_h_val = float(first["height"]) if pd.notna(first["height"]) else float(input_h)
            target_ar = float(input_h) / max(float(input_w), 1.0)
            actual_ar = orig_h_val / max(orig_w_val, 1.0)
            if actual_ar <= target_ar:
                scale = float(input_w) / max(orig_w_val, 1.0)
            else:
                scale = float(input_h) / max(orig_h_val, 1.0)
            bw = (boxes[:, 2] - boxes[:, 0]) * scale
            bh = (boxes[:, 3] - boxes[:, 1]) * scale
            min_sides = np.minimum(bw, bh)
            ars = np.maximum(bw, bh) / np.maximum(min_sides, 1e-3)
            keep = np.ones(len(boxes), dtype=bool)
            if min_box_side > 0.0:
                keep &= (min_sides >= min_box_side)
            if max_box_ar < float("inf"):
                keep &= (ars <= max_box_ar)
            boxes = boxes[keep]

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


# =============================================================================
# Dataset — AR-preserving pad + resize, no augmentation
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, samples, input_h, input_w):
        self.samples = samples
        self.input_h = input_h
        self.input_w = input_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            img_t = torch.zeros(3, self.input_h, self.input_w, dtype=torch.float32)
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros(0, dtype=torch.int64)}
            return img_t, target

        orig_h, orig_w = img.shape[:2]
        boxes = sample.boxes.copy()

        if boxes.size > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        # AR-preserving pad (matches DetectionDataset in bbox-train.py)
        target_ar = self.input_h / max(self.input_w, 1)
        actual_ar = orig_h / max(orig_w, 1)
        if actual_ar < target_ar - 1e-6:
            padded_h = int(round(orig_w * target_ar))
            pad_top = (padded_h - orig_h) // 2
            pad_bottom = padded_h - orig_h - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 1] += pad_top
                boxes[:, 3] += pad_top
            orig_h = padded_h
        elif actual_ar > target_ar + 1e-6:
            padded_w = int(round(orig_h / target_ar))
            pad_left = (padded_w - orig_w) // 2
            pad_right = padded_w - orig_w - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
            if boxes.size > 0:
                boxes[:, 0] += pad_left
                boxes[:, 2] += pad_left
            orig_w = padded_w

        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        scale_x = self.input_w / max(orig_w, 1)
        scale_y = self.input_h / max(orig_h, 1)
        if boxes.size > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]

        boxes_t = torch.from_numpy(boxes.astype(np.float32)) if boxes.size > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.ones(len(boxes_t), dtype=torch.int64)
        return image_to_tensor(img_resized), {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# =============================================================================
# Model — mirrors build_retinanet() in bbox-train.py (no backbone loading)
# =============================================================================

def build_retinanet(
    num_classes=1,
    anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    min_size=1024, max_size=1024,
    nms_thresh=0.3, score_thresh=0.05, detections_per_img=500,
    focal_alpha=0.25, focal_gamma=2.0,
):
    if not _TORCHVISION_DET_OK:
        raise RuntimeError("torchvision detection module not available.")
    try:
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=_IMAGENET_WEIGHTS, trainable_layers=5)
    except TypeError:
        backbone = resnet_fpn_backbone(backbone_name="resnet50", pretrained=True, trainable_layers=5)  # type: ignore

    anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model = RetinaNet(
        backbone=backbone, num_classes=num_classes,
        anchor_generator=anchor_gen,
        min_size=min_size, max_size=max_size,
        nms_thresh=nms_thresh, score_thresh=score_thresh,
        detections_per_img=detections_per_img,
    )
    try:
        model.head.classification_head.focal_loss_alpha = float(focal_alpha)
        model.head.classification_head.focal_loss_gamma = float(focal_gamma)
    except AttributeError:
        pass
    return model


# =============================================================================
# Evaluation
# =============================================================================

def _greedy_match_tp(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    """Return TP count via greedy score-sorted matching."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    order = np.argsort(pred_scores)[::-1]
    used_gt = set()
    tp = 0
    for pi in order:
        pb = pred_boxes[pi]
        best_iou = 0.0
        best_gi = -1
        for gi in range(len(gt_boxes)):
            if gi in used_gt:
                continue
            gb = gt_boxes[gi]
            ix1, iy1 = max(pb[0], gb[0]), max(pb[1], gb[1])
            ix2, iy2 = min(pb[2], gb[2]), min(pb[3], gb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            union = (pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            used_gt.add(best_gi)
            tp += 1
    return tp


def _fbeta(tp, fp, fn, beta=2.0):
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom > 0 else 0.0


def evaluate(model, loader, device, score_thresholds, iou_threshold):
    """Evaluate at multiple score thresholds in one inference pass."""
    model.eval()

    # Collect per-image (pred_boxes, pred_scores, gt_boxes) tuples
    per_image = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                per_image.append({
                    "pred_boxes": output["boxes"].detach().cpu().numpy(),
                    "pred_scores": output["scores"].detach().cpu().numpy(),
                    "gt_boxes": target["boxes"].cpu().numpy(),
                })

    total_gt = sum(len(r["gt_boxes"]) for r in per_image)
    thresh_stats = {}
    for thresh in score_thresholds:
        total_tp = total_fp = 0
        for r in per_image:
            keep = r["pred_scores"] >= thresh
            pb = r["pred_boxes"][keep]
            ps = r["pred_scores"][keep]
            n_gt = len(r["gt_boxes"])
            if len(pb) == 0:
                continue
            if n_gt == 0:
                total_fp += len(pb)
                continue
            tp = _greedy_match_tp(pb, ps, r["gt_boxes"], iou_threshold)
            total_fp += len(pb) - tp
            total_tp += tp
        fn = total_gt - total_tp
        thresh_stats[thresh] = {"tp": total_tp, "fp": total_fp, "fn": fn}

    return {"gt_boxes": total_gt, "total_images": len(per_image), "thresholds": thresh_stats}


# =============================================================================
# Argument parsing + main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Direction-H RetinaNet on VinDr test split."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--ckpt-path", type=Path, default=None,
                        help="Checkpoint path (default: models/bbox_resnet50.pth).")
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                        help="IoU threshold for GT matching (default: 0.1).")
    parser.add_argument("--score-thresholds", type=str, default="0.1,0.2,0.3,0.5,0.7,0.9",
                        help="Comma-separated score thresholds.")
    parser.add_argument("--min-box-side", type=float, default=0.0,
                        help="Min GT box shortest side in resized space (must match training).")
    parser.add_argument("--max-box-ar", type=float, default=float("inf"),
                        help="Max GT box aspect ratio (must match training).")
    parser.add_argument("--lesion-types", type=str, default=None,
                        help="Comma-separated lesion type filter (must match training).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    ckpt_path = args.ckpt_path or (root / "models" / "bbox_resnet50.pth")

    for p, label in [(csv_path, "CSV"), (images_root, "images root"), (ckpt_path, "checkpoint")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    print(f"[Info] Checkpoint meta: {meta}")

    input_h = int(meta.get("input_h", 1024))
    input_w = int(meta.get("input_w", 1024))
    anchor_str = str(meta.get("anchor_sizes", "32,64,128,256,512"))
    anchor_sizes = tuple((int(s.strip()),) for s in anchor_str.split(",") if s.strip())
    nms_thresh = float(meta.get("nms_thresh", 0.3))
    focal_alpha = float(meta.get("focal_alpha", 0.25))
    focal_gamma = float(meta.get("focal_gamma", 2.0))
    print(f"[Info] input={input_h}x{input_w} | anchors={anchor_str} | nms={nms_thresh} | focal a={focal_alpha} g={focal_gamma}")

    score_thresholds = [float(t.strip()) for t in args.score_thresholds.split(",") if t.strip()]
    lesion_types = [t.strip() for t in args.lesion_types.split(",")] if args.lesion_types else None
    print(f"[Info] lesion_types={lesion_types or 'all'} | min_box_side={args.min_box_side} | max_box_ar={args.max_box_ar}")

    print("[Info] Loading test GT samples ...")
    samples = load_samples(
        csv_path=csv_path, images_root=images_root, split_name="test",
        lesion_types=lesion_types, min_box_side=float(args.min_box_side),
        max_box_ar=float(args.max_box_ar), input_w=input_w, input_h=input_h,
    )
    n_pos = sum(1 for s in samples if len(s.boxes) > 0)
    total_gt = sum(len(s.boxes) for s in samples)
    print(f"[Info] test split: {len(samples)} images ({n_pos} positive) | GT boxes: {total_gt}")

    dataset = TestDataset(samples=samples, input_h=input_h, input_w=input_w)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_retinanet(
        num_classes=1,
        anchor_sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
        min_size=min(input_h, input_w), max_size=max(input_h, input_w),
        nms_thresh=nms_thresh, score_thresh=0.05, detections_per_img=500,
        focal_alpha=focal_alpha, focal_gamma=focal_gamma,
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    print("[Info] Model loaded.")

    results = evaluate(
        model=model, loader=loader, device=device,
        score_thresholds=score_thresholds, iou_threshold=float(args.iou_threshold),
    )

    gt_total = results["gt_boxes"]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Test-set evaluation  |  {ckpt_path.name}")
    print(f"GT boxes: {gt_total}  |  IoU threshold: {args.iou_threshold}  |  images: {results['total_images']}")
    print("-" * 72)
    print(f"{'Thresh':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'Recall':>8}  {'Prec':>8}  {'F2':>8}")
    print("-" * 72)
    best_f2, best_thresh = 0.0, score_thresholds[0]
    for thresh in score_thresholds:
        st = results["thresholds"][thresh]
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        recall = tp / gt_total if gt_total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f2 = _fbeta(tp, fp, fn, beta=2.0)
        if f2 > best_f2:
            best_f2, best_thresh = f2, thresh
        print(f"  @{thresh:<5.1f}  {tp:>6}  {fp:>6}  {fn:>6}  {recall:>8.3f}  {prec:>8.3f}  {f2:>8.4f}")
    print("-" * 72)
    print(f"Best F2={best_f2:.4f} @ score_threshold={best_thresh}")
    print(f"{sep}\n")

    if args.save_predictions is not None:
        import json, csv as _csv
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {args.save_predictions}")


if __name__ == "__main__":
    main()
