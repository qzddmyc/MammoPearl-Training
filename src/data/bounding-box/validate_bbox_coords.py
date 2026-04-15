# 可视化 train 与 test 文件中的 bbox 框定位
# 结果保存在 tmp/bbox_validation_out 中

# 时间久远，已经不知道逻辑是否仍然与 train / test 文件同步了

"""
Validate VinDr bbox coordinates by comparing the TRAIN logic against the TEST logic.

This script intentionally mirrors the current source-code behavior:
- Train side: use the same dataset construction logic as bbox-train.py
- Test side:  use the same dataset construction logic as bbox-test.py
- Both sides draw boxes directly from CSV xmin/ymin/xmax/ymax (no scaling)

The output is a side-by-side image:
- Left  : train logic overlay
- Right : test logic overlay

Even if the two are identical, they are rendered independently using the
two code paths so you can confirm they really match.

Usage:
  python ./src/data/bounding-box/validate_bbox_coords.py
* python ./src/data/bounding-box/validate_bbox_coords.py --split test --num-samples 3
  python ./src/data/bounding-box/validate_bbox_coords.py --image-id some_image_id.png
  python ./src/data/bounding-box/validate_bbox_coords.py --output-dir bbox_validation_out
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def image_to_tensor(img: np.ndarray) -> np.ndarray:
    # Kept for parity with the training/testing scripts, though not used for drawing.
    return img.astype(np.float32) / 255.0


def filter_invalid_boxes(boxes: np.ndarray) -> np.ndarray:
    """Remove invalid bounding boxes (non-positive width/height)."""
    if boxes.size == 0:
        return boxes

    keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
    return boxes[keep]

def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    out = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label_prefix}{i + 1}",
            (max(x1, 0), max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] >= height:
        return img
    return cv2.copyMakeBorder(
        img,
        0,
        height - img.shape[0],
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def resolve_image_path(images_root: Path, patient_id: str, image_id: str) -> Path:
    # Mirrors training/testing scripts: images_root / patient_id / image_id
    return images_root / patient_id / image_id


@dataclass
class TrainSample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray
    orig_size: Tuple[float, float]


class TrainLikeDataset:
    """
    Mirrors bbox-train.py dataset behavior.
    Here we keep the same grouping and raw-box extraction logic,
    but do NOT apply scaling.
    """

    def __init__(self, csv_path: Path, images_root: Path, split_name: str) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        self.samples: List[TrainSample] = []
        grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=True)

        for (patient_id, _series_id, image_id), group in grouped:
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)
                boxes = filter_invalid_boxes(boxes)

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            image_path = resolve_image_path(images_root, str(patient_id), str(image_id))
            self.samples.append(
                TrainSample(
                    patient_id=str(patient_id),
                    image_id=str(image_id),
                    image_path=image_path,
                    boxes=boxes,
                    orig_size=(orig_h, orig_w),
                )
            )

        if not self.samples:
            raise ValueError(f"No image samples could be built from {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")

        img = normalize_image(read_image_unicode(sample.image_path))
        boxes = sample.boxes.astype(np.float32)

        # Match train script: boxes are used as-is.
        return image_to_tensor(img), boxes, {
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "image_path": sample.image_path,
            "orig_size": sample.orig_size,
        }


@dataclass
class TestSample:
    patient_id: str
    image_id: str
    image_path: Path
    boxes: np.ndarray
    orig_size: Tuple[float, float]


class TestLikeDataset:
    """
    Mirrors bbox-test.py dataset behavior.
    Again, boxes are used as-is, just like the current test script.
    """

    def __init__(self, csv_path: Path, images_root: Path, split_name: str) -> None:
        self.csv_path = csv_path
        self.images_root = images_root
        self.split_name = split_name

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
        if df.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        self.samples: List[TestSample] = []
        grouped = df.groupby(["patient_id", "series_id", "image_id"], sort=True)

        for (patient_id, _series_id, image_id), group in grouped:
            valid = group[["xmin", "ymin", "xmax", "ymax"]].dropna()
            if valid.empty:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = valid.to_numpy(dtype=np.float32)
                boxes = filter_invalid_boxes(boxes)

            first = group.iloc[0]
            orig_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
            orig_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

            image_path = resolve_image_path(images_root, str(patient_id), str(image_id))
            self.samples.append(
                TestSample(
                    patient_id=str(patient_id),
                    image_id=str(image_id),
                    image_path=image_path,
                    boxes=boxes,
                    orig_size=(orig_h, orig_w),
                )
            )

        if not self.samples:
            raise ValueError(f"No image samples could be built from {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")

        img = normalize_image(read_image_unicode(sample.image_path))
        boxes = sample.boxes.astype(np.float32)

        # Match test script: boxes are used as-is.
        return image_to_tensor(img), boxes, {
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "image_path": sample.image_path,
            "orig_size": sample.orig_size,
        }


def get_samples(df: pd.DataFrame, split: str, image_id: str | None) -> List[pd.DataFrame]:
    d = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    if d.empty:
        raise ValueError(f"No rows found for split={split!r}")

    if image_id:
        mask = d["image_id"].astype(str) == image_id
        if not mask.any():
            stem = Path(image_id).stem
            mask = d["image_id"].astype(str) == stem
        d = d[mask]
        if d.empty:
            raise ValueError(f"No rows found for image_id={image_id!r} in split={split!r}")
        return [d]

    grouped = d.groupby(["patient_id", "series_id", "image_id"], sort=True)
    samples: List[pd.DataFrame] = []
    for _, group in grouped:
        if not group[["xmin", "ymin", "xmax", "ymax"]].dropna().empty:
            samples.append(group)
    if not samples:
        raise ValueError(f"No positive samples found in split={split!r}")
    return samples


def render_train_side(group: pd.DataFrame, images_root: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    first = group.iloc[0]
    patient_id = str(first["patient_id"])
    image_id = str(first["image_id"])

    image_path = resolve_image_path(images_root, patient_id, image_id)
    img = normalize_image(read_image_unicode(image_path))
    boxes = group[["xmin", "ymin", "xmax", "ymax"]].dropna().to_numpy(dtype=np.float32)
    boxes = filter_invalid_boxes(boxes)

    overlay = draw_boxes(img, boxes, (255, 0, 0), "train-")
    overlay = add_title(overlay, "TRAIN logic")
    info = {
        "patient_id": patient_id,
        "image_id": image_id,
        "image_path": image_path,
        "image_size": img.shape[:2],
        "num_boxes": int(boxes.shape[0]),
        "boxes": boxes,
    }
    return overlay, info


def render_test_side(group: pd.DataFrame, images_root: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    first = group.iloc[0]
    patient_id = str(first["patient_id"])
    image_id = str(first["image_id"])

    image_path = resolve_image_path(images_root, patient_id, image_id)
    img = normalize_image(read_image_unicode(image_path))
    boxes = group[["xmin", "ymin", "xmax", "ymax"]].dropna().to_numpy(dtype=np.float32)
    boxes = filter_invalid_boxes(boxes)

    overlay = draw_boxes(img, boxes, (0, 255, 0), "test-")
    overlay = add_title(overlay, "TEST logic")
    info = {
        "patient_id": patient_id,
        "image_id": image_id,
        "image_path": image_path,
        "image_size": img.shape[:2],
        "num_boxes": int(boxes.shape[0]),
        "boxes": boxes,
    }
    return overlay, info


def save_side_by_side(left: np.ndarray, right: np.ndarray, output_path: Path) -> None:
    h = max(left.shape[0], right.shape[0])
    left = pad_to_height(left, h)
    right = pad_to_height(right, h)
    pad = np.full((h, 20, 3), 255, dtype=np.uint8)
    combined = np.concatenate([left, pad, right], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def analyze_sample(
    group: pd.DataFrame,
    images_root: Path,
    output_dir: Path,
    index: int,
) -> None:
    train_overlay, train_info = render_train_side(group, images_root)
    test_overlay, test_info = render_test_side(group, images_root)

    first = group.iloc[0]
    csv_h = float(first["height"]) if pd.notna(first["height"]) else 0.0
    csv_w = float(first["width"]) if pd.notna(first["width"]) else 0.0

    print("=" * 90)
    print(f"[Sample {index}] patient_id={train_info['patient_id']} image_id={train_info['image_id']}")
    print(f"Image file: {train_info['image_path']}")
    print(f"Image size: {train_info['image_size'][0]} x {train_info['image_size'][1]}")
    print(f"CSV height/width: {csv_h} x {csv_w}")
    print(f"Num boxes: {train_info['num_boxes']}")
    print("Boxes (first 3):")
    print(train_info["boxes"][:3])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sample_{index:03d}_{Path(train_info['image_id']).stem}_train_vs_test.png"
    save_side_by_side(train_overlay, test_overlay, out_path)
    print(f"Saved comparison image: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train/test bbox rendering logic.")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image-id", type=str, default=None, help="Optional single image_id to inspect.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_file()
    csv_path = args.csv_path or (root / "data" / "raw" / "vindr_detection_folds.csv")
    images_root = args.images_root or (root / "data" / "processed" / "images_png")
    output_dir = args.output_dir or (root / "tmp" / "bbox_validation_out")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    df = pd.read_csv(csv_path, low_memory=False)
    samples = get_samples(df, split=args.split, image_id=args.image_id)

    # Optional sanity: instantiate both dataset classes so the script truly follows both source paths.
    _train_ds = TrainLikeDataset(csv_path=csv_path, images_root=images_root, split_name=args.split)
    _test_ds = TestLikeDataset(csv_path=csv_path, images_root=images_root, split_name=args.split)
    _ = (_train_ds, _test_ds)

    for idx, group in enumerate(samples[: max(args.num_samples, 1)], start=1):
        analyze_sample(group, images_root=images_root, output_dir=output_dir, index=idx)


if __name__ == "__main__":
    main()
