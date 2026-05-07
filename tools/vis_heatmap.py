r"""
Heatmap visualization tool for bbox-train-D inference debugging.

Run from repo root (Git Bash):

python tools/vis_heatmap.py \
    --model-path models/bbox_resnet50.D.pth \
    --out-dir tmp/heatmap_vis \
    --n-images 20 \
    --patch-size 256 \
    --stride 64 \
    --dilation 15 \
    --threshold 0.5

Output per image (saved to --out-dir):
  <image_id>_heatmap.png   -- raw heatmap (hot colormap)
  <image_id>_overlay.png   -- original image + GT boxes (green) + pred boxes (red/orange)

The script loads the same model architecture and inference pipeline as bbox-train-D.py
so results are directly comparable to validation logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── resolve repo root ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Re-use helpers from training script via direct import of functions
# (bbox-train-D.py is not a package, so we exec-import key pieces)
_train_script = REPO_ROOT / "src" / "data" / "bounding-box" / "bbox-train-D.py"
_ns: dict = {}
exec(compile(_train_script.read_text(encoding="utf-8"), str(_train_script), "exec"), _ns)

read_image_unicode = _ns["read_image_unicode"]
normalize_image    = _ns["normalize_image"]
image_to_tensor    = _ns["image_to_tensor"]
heatmap_to_boxes   = _ns["heatmap_to_boxes"]
compute_iou_matches = _ns["compute_iou_matches"]
VinDrBboxDataset   = _ns["VinDrBboxDataset"]
build_patch_classifier = _ns["build_patch_classifier"]


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    out = img_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label and len(boxes):
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def heatmap_to_colored(heatmap: np.ndarray) -> np.ndarray:
    """Convert float [0,1] heatmap to BGR uint8 with COLORMAP_HOT."""
    scaled = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_HOT)


# ── inference ─────────────────────────────────────────────────────────────────

def run_sliding_window(
    model: nn.Module,
    img_np: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return (H, W) float32 heatmap using center-1/3 write (same as training script)."""
    h, w = img_np.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
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

    model.eval()
    with torch.no_grad():
        for start in range(0, len(patch_list), batch_size):
            batch_items = patch_list[start:start + batch_size]
            batch_tensor = torch.stack([it[2] for it in batch_items]).to(device)
            probs = torch.sigmoid(model(batch_tensor).squeeze(1)).cpu().numpy()
            for i_b, (py, px, _) in enumerate(batch_items):
                score = float(probs[i_b])
                cy1 = py + ps // 3
                cy2 = min(py + 2 * ps // 3, h)
                cx1 = px + ps // 3
                cx2 = min(px + 2 * ps // 3, w)
                heatmap[cy1:cy2, cx1:cx2] = np.maximum(heatmap[cy1:cy2, cx1:cx2], score)

    return heatmap


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize sliding-window heatmaps")
    p.add_argument("--model-path", type=Path, default=REPO_ROOT / "models" / "bbox_resnet50.D.pth")
    p.add_argument("--csv-path",   type=Path, default=REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv")
    p.add_argument("--images-root",type=Path, default=REPO_ROOT / "data" / "processed" / "images_png")
    p.add_argument("--out-dir",    type=Path, default=REPO_ROOT / "tmp" / "heatmap_vis")
    p.add_argument("--split",      type=str,  default="test")
    p.add_argument("--n-images",   type=int,  default=20,  help="Number of positive images to visualize")
    p.add_argument("--patch-size", type=int,  default=256)
    p.add_argument("--stride",     type=int,  default=64)
    p.add_argument("--dilation",   type=int,  default=15)
    p.add_argument("--threshold",  type=float,default=0.5, help="Heatmap binarization threshold")
    p.add_argument("--iou-threshold", type=float, default=0.1)
    p.add_argument("--backbone",   type=str,  default="resnet50")
    p.add_argument("--dropout",    type=float,default=0.5)
    p.add_argument("--seed",       type=int,  default=0,   help="Random seed for image selection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load dataset (validation split) ───────────────────────────────────────
    dataset = VinDrBboxDataset(
        csv_path=args.csv_path,
        images_root=args.images_root,
        split_name=args.split,
        positive_only=False,
        crop_breast_region=False,
    )
    samples = dataset.samples

    # Pick positive images only
    pos_indices = [i for i, s in enumerate(samples) if s.boxes.size > 0]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(pos_indices, size=min(args.n_images, len(pos_indices)), replace=False)
    print(f"Visualizing {len(chosen)} positive images from split='{args.split}'")

    # ── load model ─────────────────────────────────────────────────────────────
    model = build_patch_classifier(
        dropout=args.dropout,
        medical_backbone_path=None,
    ).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model_path}")

    # ── per-image inference & visualization ────────────────────────────────────
    total_tp = total_fp = total_fn = 0

    for img_idx in chosen:
        sample = samples[int(img_idx)]

        try:
            img_np = normalize_image(read_image_unicode(sample.image_path))
        except FileNotFoundError:
            print(f"  [SKIP] {sample.image_path} not found")
            continue

        gt_boxes = sample.boxes.astype(np.float32)
        if gt_boxes.size > 0:
            keep = (gt_boxes[:, 2] > gt_boxes[:, 0] + 1) & (gt_boxes[:, 3] > gt_boxes[:, 1] + 1)
            gt_boxes = gt_boxes[keep]

        # Run inference
        heatmap = run_sliding_window(model, img_np, args.patch_size, args.stride, device)
        pred_boxes, _ = heatmap_to_boxes(heatmap, args.threshold, args.dilation)

        tp, fp, fn = compute_iou_matches(pred_boxes, gt_boxes, args.iou_threshold)
        total_tp += tp; total_fp += fp; total_fn += fn

        # ── save heatmap image ───────────────────────────────────────────────
        hm_colored = heatmap_to_colored(heatmap)
        stem = sample.image_id.replace("/", "_").replace("\\", "_")
        cv2.imwrite(str(args.out_dir / f"{stem}_heatmap.png"), hm_colored)

        # ── save overlay image ───────────────────────────────────────────────
        # Convert RGB → BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Blend heatmap
        alpha = 0.4
        img_bgr = cv2.addWeighted(img_bgr, 1 - alpha, hm_colored, alpha, 0)
        # Draw GT boxes (green, thick)
        img_bgr = draw_boxes(img_bgr, gt_boxes, color=(0, 255, 0), thickness=3)
        # Draw pred boxes (red)
        img_bgr = draw_boxes(img_bgr, pred_boxes, color=(0, 0, 255), thickness=2)

        # Annotate stats in corner
        label = f"TP={tp} FP={fp} FN={fn} | thresh={args.threshold}"
        cv2.putText(img_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imwrite(str(args.out_dir / f"{stem}_overlay.png"), img_bgr)
        print(f"  {stem}: GT={len(gt_boxes)} Pred={len(pred_boxes)} TP={tp} FP={fp} FN={fn}")

    # ── summary ───────────────────────────────────────────────────────────────
    prec = total_tp / max(total_tp + total_fp, 1)
    rec  = total_tp / max(total_tp + total_fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nSummary over {len(chosen)} images:")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
    print(f"\nImages saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
