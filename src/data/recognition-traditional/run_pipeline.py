"""Main pipeline entry point for the traditional breast-cancer classification system.

Usage
-----
# Full pipeline (train → evaluate), no mask generation
python run_pipeline.py

# Generate breast-region masks before training
python run_pipeline.py --generate-mask

# Only run evaluation with pre-trained models
python run_pipeline.py --eval-only

# Choose which stage to run
python run_pipeline.py --stage 1
python run_pipeline.py --stage 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Ensure this directory is on sys.path so sibling modules are importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    FEATURE_CACHE_DIR,
    FINDING_COLS,
    STAGE1_MODEL,
    STAGE2_MODEL,
)
from evaluate import evaluate_pipeline
from features import extract_features_batch
from preprocessing import preprocess_patch
from sampling import build_dataset, generate_masks
from train_stage1 import (
    load_stage1,
    predict_stage1,
    train_stage1,
)
from train_stage2 import (
    load_stage2,
    predict_stage2,
    train_stage2,
)

# ---------------------------------------------------------------------------
# Feature cache helpers
# ---------------------------------------------------------------------------

_TRAIN_FEAT_PATH   = FEATURE_CACHE_DIR / "train_features.npy"
_TRAIN_LABELS_PATH = FEATURE_CACHE_DIR / "train_labels.npy"
_TRAIN_S2_PATH     = FEATURE_CACHE_DIR / "train_stage2_labels.npy"
_TEST_FEAT_PATH    = FEATURE_CACHE_DIR / "test_features.npy"
_TEST_LABELS_PATH  = FEATURE_CACHE_DIR / "test_labels.npy"
_TEST_S2_PATH      = FEATURE_CACHE_DIR / "test_stage2_labels.npy"


def _cache_exists(split: str) -> bool:
    if split == "training":
        return _TRAIN_FEAT_PATH.exists()
    return _TEST_FEAT_PATH.exists()


def _save_cache(split: str, X: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> None:
    if split == "training":
        np.save(_TRAIN_FEAT_PATH, X)
        np.save(_TRAIN_LABELS_PATH, y1)
        np.save(_TRAIN_S2_PATH, y2)
    else:
        np.save(_TEST_FEAT_PATH, X)
        np.save(_TEST_LABELS_PATH, y1)
        np.save(_TEST_S2_PATH, y2)
    print(f"[pipeline] Feature cache saved for split='{split}'.")


def _load_cache(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "training":
        return (
            np.load(_TRAIN_FEAT_PATH),
            np.load(_TRAIN_LABELS_PATH),
            np.load(_TRAIN_S2_PATH),
        )
    return (
        np.load(_TEST_FEAT_PATH),
        np.load(_TEST_LABELS_PATH),
        np.load(_TEST_S2_PATH),
    )


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(split: str, extended: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample patches and extract features for the given split.

    Uses a cached .npy file if available (non-extended baseline features).
    Extended features (Stage-2) are always recomputed from the base patches
    to avoid storing two large caches.

    Returns
    -------
    X  : feature matrix (N, D)
    y1 : Stage-1 binary labels (N,)
    y2 : Stage-2 class labels  (N,)  [-1 for negatives]
    """
    if not extended and _cache_exists(split):
        print(f"[pipeline] Loading cached features for split='{split}' …")
        return _load_cache(split)

    print(f"[pipeline] Building patch dataset for split='{split}' …")
    patches_raw, records = build_dataset(split=split)

    y1 = np.array([r["stage1_label"] for r in records], dtype=np.int32)
    y2 = np.array([r["stage2_label"] for r in records], dtype=np.int32)

    print(f"[pipeline] Preprocessing {len(patches_raw)} patches …")
    patches_norm = [
        preprocess_patch(p)
        for p in tqdm(patches_raw, desc="[pipeline] preprocess", unit="patch")
    ]

    print(f"[pipeline] Extracting features (extended={extended}) …")
    X = extract_features_batch(patches_norm, extended=extended)

    if not extended:
        _save_cache(split, X, y1, y2)

    return X, y1, y2


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_stage1(generate_mask: bool = False, force_retrain: bool = False) -> None:
    """Generate masks (optional) → build features → train Stage-1."""
    if generate_mask:
        generate_masks()

    X_train, y1_train, _ = build_features("training", extended=False)
    pipeline, threshold = train_stage1(X_train, y1_train, model_type=STAGE1_MODEL)

    # Evaluate on test split
    print("\n[pipeline] Evaluating Stage-1 on test split …")
    X_test, y1_test, _ = build_features("test", extended=False)
    labels, proba = predict_stage1(pipeline, X_test, threshold=threshold)
    evaluate_pipeline(y1_test, labels, None, None, proba_s1=proba)


def run_stage2() -> None:
    """Build extended features for positive samples → train Stage-2."""
    pipeline, threshold = load_stage1()

    # Get positive training patches for Stage-2
    print("\n[pipeline] Building Stage-2 training set (positive patches only) …")
    X_train_ext, y1_train, y2_train = build_features("training", extended=True)

    # Filter: only disease-positive samples with a known Stage-2 label
    pos_mask = (y1_train == 1) & (y2_train >= 0)
    X_pos = X_train_ext[pos_mask]
    y_pos = y2_train[pos_mask]

    if len(X_pos) == 0:
        print("[pipeline] No positive samples with Stage-2 labels; skipping Stage-2 training.")
        return

    model, scaler, le = train_stage2(X_pos, y_pos, model_type=STAGE2_MODEL)

    # Evaluate Stage-2 on test split
    print("\n[pipeline] Evaluating Stage-2 on test split …")
    # Stage-1 needs base features (113-dim, loaded from cache)
    X_test_base, y1_test, y2_test = build_features("test", extended=False)
    s1_labels, s1_proba = predict_stage1(pipeline, X_test_base, threshold=threshold)

    # Stage-2 needs extended features (119-dim)
    X_test_ext, _, _ = build_features("test", extended=True)

    # Only test Stage-2 on samples predicted positive by Stage-1
    pred_pos_mask = s1_labels == 1
    true_pos_mask = (y1_test == 1) & (y2_test >= 0)
    eval_mask = pred_pos_mask & true_pos_mask

    if eval_mask.sum() == 0:
        print("[pipeline] No overlapping Stage-2 evaluation samples.")
        return

    s2_labels, _ = predict_stage2(model, scaler, le, X_test_ext[eval_mask])
    evaluate_pipeline(
        y1_test, s1_labels,
        y2_test[eval_mask], s2_labels,
        proba_s1=s1_proba,
    )


def run_eval_only() -> None:
    """Load saved models and evaluate both stages on the test split."""
    pipeline, threshold = load_stage1()
    # Stage-1 needs base features (113-dim)
    X_test_base, y1_test, y2_test = build_features("test", extended=False)
    s1_labels, s1_proba = predict_stage1(pipeline, X_test_base, threshold=threshold)

    try:
        model, scaler, le = load_stage2()
        # Stage-2 needs extended features (119-dim)
        X_test_ext, _, _ = build_features("test", extended=True)
        pred_pos = s1_labels == 1
        true_pos = (y1_test == 1) & (y2_test >= 0)
        eval_mask = pred_pos & true_pos
        if eval_mask.sum() > 0:
            s2_labels, _ = predict_stage2(model, scaler, le, X_test_ext[eval_mask])
            evaluate_pipeline(y1_test, s1_labels, y2_test[eval_mask], s2_labels, s1_proba)
        else:
            evaluate_pipeline(y1_test, s1_labels, None, None, s1_proba)
    except FileNotFoundError:
        print("[pipeline] Stage-2 model not found; evaluating Stage-1 only.")
        evaluate_pipeline(y1_test, s1_labels, None, None, s1_proba)



# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def _nms(boxes: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Non-maximum suppression on a list of detection dicts.

    Each dict must have keys: x1, y1, x2, y2, score.
    Returns the filtered list sorted by score descending.
    """
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept: list[dict] = []

    while boxes_sorted:
        best = boxes_sorted.pop(0)
        kept.append(best)
        remaining = []
        for b in boxes_sorted:
            # Compute IoU
            ix1 = max(best["x1"], b["x1"])
            iy1 = max(best["y1"], b["y1"])
            ix2 = min(best["x2"], b["x2"])
            iy2 = min(best["y2"], b["y2"])
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter = inter_w * inter_h
            area_best = (best["x2"] - best["x1"]) * (best["y2"] - best["y1"])
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            union = area_best + area_b - inter
            iou = inter / (union + 1e-6)
            if iou < iou_threshold:
                remaining.append(b)
        boxes_sorted = remaining

    return kept


def infer_sliding_window(
    image_path: str | Path,
    *,
    stride: int = 32,
    patch_size: int | None = None,
    s1_threshold: float | None = None,
    nms_iou: float = 0.3,
) -> list[dict]:
    """Run full sliding-window detection on a single mammogram image.

    Parameters
    ----------
    image_path:
        Path to a PNG mammogram image.
    stride:
        Step size (pixels) between consecutive windows.  Smaller = more
        candidate windows, higher recall, slower.
    patch_size:
        Window size in pixels.  Defaults to ``config.PATCH_SIZE`` (128).
    s1_threshold:
        Stage-1 probability threshold.  Defaults to the saved threshold.
    nms_iou:
        IoU threshold for NMS; lower = keep more boxes.

    Returns
    -------
    detections : list[dict]
        Each dict has: x1, y1, x2, y2, score (Stage-1 probability),
        stage2_label (int, -1 if Stage-2 not available),
        stage2_name (str).
    """
    import cv2 as _cv2
    from config import PATCH_SIZE, STAGE2_MERGED_NAMES

    ps = patch_size or PATCH_SIZE

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    pipeline, saved_threshold = load_stage1()
    threshold = s1_threshold if s1_threshold is not None else saved_threshold

    try:
        s2_model, s2_scaler, s2_le = load_stage2()
        has_stage2 = True
    except FileNotFoundError:
        s2_model = s2_scaler = s2_le = None
        has_stage2 = False

    # ------------------------------------------------------------------
    # Load and preprocess image
    # ------------------------------------------------------------------
    img_bgr = _cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img_bgr.shape[:2]

    # ------------------------------------------------------------------
    # Collect candidate windows
    # ------------------------------------------------------------------
    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []  # (x1, y1)

    for y in range(0, h - ps + 1, stride):
        for x in range(0, w - ps + 1, stride):
            crop = img_bgr[y : y + ps, x : x + ps]
            patches.append(preprocess_patch(crop))
            coords.append((x, y))

    if not patches:
        print(f"[infer] Image too small for patch_size={ps}; no windows generated.")
        return []

    print(f"[infer] Image {Path(image_path).name}: {h}×{w}, {len(patches)} windows "
          f"(stride={stride}, patch_size={ps})")

    # ------------------------------------------------------------------
    # Stage-1: batch predict
    # ------------------------------------------------------------------
    X_base = extract_features_batch(patches, extended=False)
    # Replace NaN/Inf (from blank/uniform windows) with 0 to avoid PCA failure
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)
    s1_labels, s1_proba = predict_stage1(pipeline, X_base, threshold=threshold)

    pos_indices = np.where(s1_labels == 1)[0]
    print(f"[infer] Stage-1 positives: {len(pos_indices)} / {len(patches)} "
          f"(threshold={threshold:.3f})")

    if len(pos_indices) == 0:
        return []

    # ------------------------------------------------------------------
    # Stage-2: classify positives
    # ------------------------------------------------------------------
    detections: list[dict] = []

    if has_stage2:
        X_ext_pos = extract_features_batch(
            [patches[i] for i in pos_indices], extended=True
        )
        X_ext_pos = np.nan_to_num(X_ext_pos, nan=0.0, posinf=0.0, neginf=0.0)
        s2_labels, s2_names = predict_stage2(s2_model, s2_scaler, s2_le, X_ext_pos)
    else:
        s2_labels = [-1] * len(pos_indices)
        s2_names = ["Unknown"] * len(pos_indices)

    for k, i in enumerate(pos_indices):
        x1, y1 = coords[i]
        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x1 + ps,
            "y2": y1 + ps,
            "score": float(s1_proba[i]),
            "stage2_label": int(s2_labels[k]),
            "stage2_name": str(s2_names[k]),
        })

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------
    before_nms = len(detections)
    detections = _nms(detections, iou_threshold=nms_iou)
    print(f"[infer] After NMS (IoU≤{nms_iou}): {len(detections)} detections "
          f"(from {before_nms})")

    return detections


def run_infer(image_path: str, stride: int = 32, nms_iou: float = 0.3) -> None:
    """CLI wrapper: run sliding-window inference and print results."""
    detections = infer_sliding_window(
        image_path, stride=stride, nms_iou=nms_iou
    )
    if not detections:
        print("[infer] No disease detected.")
        return

    print(f"\n[infer] Detected {len(detections)} region(s):")
    print(f"  {'Rank':<5} {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}  {'Score':>6}  Class")
    print("  " + "-" * 55)
    for rank, det in enumerate(detections, 1):
        print(
            f"  {rank:<5} {det['x1']:>5} {det['y1']:>5} "
            f"{det['x2']:>5} {det['y2']:>5}  {det['score']:.4f}  {det['stage2_name']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional breast-cancer classification pipeline"
    )
    parser.add_argument(
        "--generate-mask",
        action="store_true",
        default=False,
        help="Generate breast-region masks before training (requires src/data/segment/segment.py).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training; only evaluate pre-trained models on the test split.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only a specific stage (1 or 2). Default: run both stages.",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        metavar="IMAGE_PATH",
        help="Run sliding-window inference on a single image (PNG path).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Sliding-window stride in pixels (default: 32). Used with --infer.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.3,
        help="IoU threshold for NMS (default: 0.3). Used with --infer.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.infer:
        run_infer(args.infer, stride=args.stride, nms_iou=args.nms_iou)
        return

    if args.eval_only:
        run_eval_only()
        return

    if args.stage == 1:
        run_stage1(generate_mask=args.generate_mask)
    elif args.stage == 2:
        run_stage2()
    else:
        # Full pipeline
        run_stage1(generate_mask=args.generate_mask)
        run_stage2()


if __name__ == "__main__":
    main()
