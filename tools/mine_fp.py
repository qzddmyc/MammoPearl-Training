r"""
False-Positive Mining tool for bbox-train-D.

Run on the TRAINING set to collect high-confidence false-positive patches,
then pass the output CSV to bbox-train-D.py via --fp-mining-csv so those
patches are added as hard negatives in the next training run.

Typical workflow
----------------
# Step 1: run FP mining on training split (after a training run)
python tools/mine_fp.py \
    --model-path models/bbox_resnet50.D.pth \
    --out-csv    tmp/fp_mining.csv \
    --split      train \
    --patch-size 256 \
    --stride     64 \
    --fp-threshold 0.5 \
    --iou-threshold 0.1

# Step 2: use the mined CSV in the next training run
python src/data/bounding-box/bbox-train-D.py \
    ... \
    --fp-mining-csv tmp/fp_mining.csv

Output CSV columns: image_path, px, py, score
  - image_path: absolute path to the image
  - px, py:     top-left corner of the 256x256 patch (in original image coords)
  - score:      model sigmoid output for that patch
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_train_script = REPO_ROOT / "src" / "data" / "bounding-box" / "bbox-train-D.py"
_ns: dict = {}
exec(compile(_train_script.read_text(encoding="utf-8"), str(_train_script), "exec"), _ns)

read_image_unicode   = _ns["read_image_unicode"]
normalize_image      = _ns["normalize_image"]
image_to_tensor      = _ns["image_to_tensor"]
compute_iou_matches  = _ns["compute_iou_matches"]
VinDrBboxDataset     = _ns["VinDrBboxDataset"]
build_patch_classifier = _ns["build_patch_classifier"]


def iou_patch_vs_gt(px: int, py: int, ps: int, gt_boxes: np.ndarray) -> float:
    """Max IoU between a patch box and all GT boxes."""
    if gt_boxes.size == 0:
        return 0.0
    bx1, by1, bx2, by2 = float(px), float(py), float(px + ps), float(py + ps)
    inter_x1 = np.maximum(gt_boxes[:, 0], bx1)
    inter_y1 = np.maximum(gt_boxes[:, 1], by1)
    inter_x2 = np.minimum(gt_boxes[:, 2], bx2)
    inter_y2 = np.minimum(gt_boxes[:, 3], by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    patch_area = float(ps * ps)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = patch_area + gt_areas - inter
    iou = inter / np.maximum(union, 1e-6)
    return float(iou.max())


def mine_image(
    model: torch.nn.Module,
    img_np: np.ndarray,
    gt_boxes: np.ndarray,
    patch_size: int,
    stride: int,
    fp_threshold: float,
    iou_threshold: float,
    device: torch.device,
    batch_size: int = 128,
) -> List[Tuple[int, int, float]]:
    """Return list of (px, py, score) for FP patches in this image."""
    h, w = img_np.shape[:2]
    ps = patch_size

    y_positions = list(range(0, max(1, h - ps + 1), stride)) or [0]
    x_positions = list(range(0, max(1, w - ps + 1), stride)) or [0]

    patch_list: List[Tuple[int, int, torch.Tensor]] = []
    for py in y_positions:
        for px in x_positions:
            y2, x2 = min(py + ps, h), min(px + ps, w)
            patch_np = img_np[py:y2, px:x2]
            pad_h, pad_w = ps - (y2 - py), ps - (x2 - px)
            if pad_h > 0 or pad_w > 0:
                patch_np = np.pad(patch_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            patch_list.append((py, px, image_to_tensor(patch_np)))

    fp_patches: List[Tuple[int, int, float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patch_list), batch_size):
            batch_items = patch_list[start:start + batch_size]
            batch_tensor = torch.stack([it[2] for it in batch_items]).to(device)
            probs = torch.sigmoid(model(batch_tensor).squeeze(1)).cpu().numpy()
            for i_b, (py, px, _) in enumerate(batch_items):
                score = float(probs[i_b])
                if score < fp_threshold:
                    continue
                # FP = high score but no overlap with any GT box
                max_iou = iou_patch_vs_gt(px, py, ps, gt_boxes)
                if max_iou < iou_threshold:
                    fp_patches.append((px, py, score))

    return fp_patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine false-positive patches from training set")
    p.add_argument("--model-path",    type=Path,  default=REPO_ROOT / "models" / "bbox_resnet50.D.pth")
    p.add_argument("--csv-path",      type=Path,  default=REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv")
    p.add_argument("--images-root",   type=Path,  default=REPO_ROOT / "data" / "processed" / "images_png")
    p.add_argument("--out-csv",       type=Path,  default=REPO_ROOT / "tmp" / "fp_mining.csv")
    p.add_argument("--split",         type=str,   default="train")
    p.add_argument("--patch-size",    type=int,   default=256)
    p.add_argument("--stride",        type=int,   default=64)
    p.add_argument("--fp-threshold",  type=float, default=0.5,  help="Min model score to consider a patch as FP candidate")
    p.add_argument("--iou-threshold", type=float, default=0.1,  help="Max IoU with GT to classify as FP (not TP)")
    p.add_argument("--max-fp-per-image", type=int, default=10,  help="Cap FP patches per image to avoid imbalance")
    p.add_argument("--pos-only",      action="store_true",       help="Only mine FP from positive images (default: all images)")
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--batch-size",    type=int,   default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = VinDrBboxDataset(
        csv_path=args.csv_path,
        images_root=args.images_root,
        split_name=args.split,
        positive_only=False,
        crop_breast_region=False,
    )
    samples = dataset.samples
    print(f"Loaded {len(samples)} images from split='{args.split}'")

    if args.pos_only:
        indices = [i for i, s in enumerate(samples) if s.boxes.size > 0]
        print(f"Mining FP from {len(indices)} positive images only")
    else:
        indices = list(range(len(samples)))
        print(f"Mining FP from all {len(indices)} images")

    model = build_patch_classifier(dropout=args.dropout, medical_backbone_path=None).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model_path}")

    total_fp = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "px", "py", "score"])

        for i, sample_idx in enumerate(indices):
            sample = samples[sample_idx]
            try:
                img_np = normalize_image(read_image_unicode(sample.image_path))
            except FileNotFoundError:
                continue

            gt_boxes = sample.boxes.astype(np.float32)
            if gt_boxes.size > 0:
                keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
                gt_boxes = gt_boxes[keep]

            fp_patches = mine_image(
                model=model,
                img_np=img_np,
                gt_boxes=gt_boxes,
                patch_size=args.patch_size,
                stride=args.stride,
                fp_threshold=args.fp_threshold,
                iou_threshold=args.iou_threshold,
                device=device,
                batch_size=args.batch_size,
            )

            # Sort by score descending, cap per image
            fp_patches.sort(key=lambda x: x[2], reverse=True)
            fp_patches = fp_patches[:args.max_fp_per_image]

            for px, py, score in fp_patches:
                writer.writerow([str(sample.image_path), px, py, f"{score:.4f}"])

            total_fp += len(fp_patches)
            if (i + 1) % 500 == 0 or (i + 1) == len(indices):
                print(f"  [{i+1}/{len(indices)}] total FP patches collected: {total_fp}")

    print(f"\nMining complete. {total_fp} FP patches written to: {args.out_csv}")
    print("Pass this file to bbox-train-D.py via --fp-mining-csv <path>")


if __name__ == "__main__":
    main()
