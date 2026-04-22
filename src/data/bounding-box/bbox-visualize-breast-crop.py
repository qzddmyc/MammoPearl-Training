# Visualize breast-region connected-component mask on images that have GT bounding boxes.
#
# Usage:
#   python src/data/bounding-box/bbox-visualize-breast-crop.py
#   python src/data/bounding-box/bbox-visualize-breast-crop.py --num-images 20

"""Pick N images that carry GT bbox annotations, run the same breast-region
detection logic used in the training pipeline, and save a fusion image:
original image with the detected breast connected-component overlaid in
semi-transparent red, plus the GT bounding boxes drawn in green.

Outputs go to tmp/breast_crop_vis/<patient_id>_<image_id>.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


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


def detect_breast_region_with_mask(
    img: np.ndarray,
    margin_ratio: float = 0.05,
) -> Tuple[int, int, int, int, np.ndarray]:
    """Same logic as detect_breast_region in the training pipeline, but also
    returns the filled contour mask (uint8, same H×W as img) so it can be
    overlaid for visualization.

    Returns
    -------
    (x1, y1, x2, y2, filled_mask)
        crop_box coordinates (with margin) and the binary breast mask.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    if h <= 0 or w <= 0:
        return 0, 0, 0, 0, full_mask
    if np.max(gray) <= 0:
        return 0, 0, w, h, full_mask

    # Otsu binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological opening to remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        ys, xs = np.where(gray > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, w, h, full_mask
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        # No contour to fill — use the threshold as the mask
        full_mask = thresh.copy()
    else:
        # Largest contour = breast main body
        contour = max(contours, key=cv2.contourArea)
        # Fill the contour to produce the connected-component mask
        cv2.drawContours(full_mask, [contour], -1, 255, thickness=cv2.FILLED)
        bx, by, cw, ch = cv2.boundingRect(contour)
        x1 = int(bx)
        y1 = int(by)
        x2 = int(bx + cw)
        y2 = int(by + ch)

    margin_ratio = max(0.0, float(margin_ratio))
    margin_x = int(round((x2 - x1) * margin_ratio))
    margin_y = int(round((y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h, full_mask

    return x1, y1, x2, y2, full_mask


def make_fusion_image(
    img_rgb: np.ndarray,
    breast_mask: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    boxes: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay breast mask in red, draw GT boxes in green, crop box in blue.

    Parameters
    ----------
    img_rgb   : H×W×3 uint8 RGB image
    breast_mask : H×W uint8 binary mask (255 = breast region)
    crop_box  : (x1, y1, x2, y2) final crop rectangle (with margin)
    boxes     : (N, 4) float32 array of GT boxes in [xmin, ymin, xmax, ymax]
    alpha     : opacity of the red mask overlay

    Returns
    -------
    BGR uint8 image ready for cv2.imwrite
    """
    vis = img_rgb.astype(np.float32)

    # Red mask overlay on breast region
    red = np.zeros_like(vis)
    red[:, :, 0] = 255.0  # R channel in RGB
    mask_norm = (breast_mask > 0).astype(np.float32)[:, :, np.newaxis]
    vis = vis * (1.0 - alpha * mask_norm) + red * (alpha * mask_norm)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # Convert to BGR for cv2 drawing functions
    out = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # Crop box — blue (BGR: 255,0,0)
    cx1, cy1, cx2, cy2 = crop_box
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (255, 0, 0), 3)

    # GT bounding boxes — green (BGR: 0,255,0)
    for box in boxes:
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

    return out


def load_positive_samples(
    csv_path: Path,
    images_root: Path,
) -> List[dict]:
    """Return records for all images that have at least one valid GT bbox."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only rows with valid bbox coordinates
    bbox_cols = ["xmin", "ymin", "xmax", "ymax"]
    has_bbox = df[bbox_cols].notna().all(axis=1)
    df = df[has_bbox].copy()

    # Drop degenerate boxes
    valid = (df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])
    df = df[valid].copy()

    if df.empty:
        raise ValueError(f"No valid bbox rows found in {csv_path}")

    samples: List[dict] = []
    grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=False)
    for (patient_id, _series_id, image_id), group in grouped:
        boxes = group[bbox_cols].to_numpy(dtype=np.float32)
        image_path = images_root / str(patient_id) / str(image_id)
        if not image_path.exists():
            continue
        samples.append(
            {
                "patient_id": str(patient_id),
                "image_id": str(image_id),
                "image_path": image_path,
                "boxes": boxes,
            }
        )

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize breast-region mask on images that have GT bboxes."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to visualize (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = root / "data" / "raw" / "vindr_detection_folds.csv"
    images_root = root / "data" / "processed" / "images_png"
    out_dir = root / "tmp" / "breast_crop_vis"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Loading positive samples from {csv_path} ...")
    samples = load_positive_samples(csv_path, images_root)
    print(f"[Info] Found {len(samples)} images with valid GT bboxes.")

    n = min(args.num_images, len(samples))
    rng = random.Random(args.seed)
    chosen = rng.sample(samples, n)

    for i, sample in enumerate(chosen, 1):
        img_path: Path = sample["image_path"]
        boxes: np.ndarray = sample["boxes"]

        img = normalize_image(read_image_unicode(img_path))
        h, w = img.shape[:2]

        # Clip boxes to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

        cx1, cy1, cx2, cy2, breast_mask = detect_breast_region_with_mask(img)
        crop_box = (cx1, cy1, cx2, cy2)

        fusion = make_fusion_image(img, breast_mask, crop_box, boxes)

        stem = f"{sample['patient_id']}_{Path(sample['image_id']).stem}"
        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), fusion)
        print(f"[{i:02d}/{n}] Saved: {out_path.relative_to(root)}")

    print(f"\n[Done] {n} fusion images saved to: {out_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
