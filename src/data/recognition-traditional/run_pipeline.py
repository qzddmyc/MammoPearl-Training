"""Main pipeline entry point for the traditional breast-cancer classification system.

Usage
-----
# Full pipeline (generate masks → train → evaluate)
python run_pipeline.py

# Skip mask generation (masks already exist)
python run_pipeline.py --skip-mask

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

def run_stage1(skip_mask: bool = False, force_retrain: bool = False) -> None:
    """Generate masks → build features → train Stage-1."""
    if not skip_mask:
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
    X_train, y1_train, y2_train = build_features("training", extended=True)

    # Filter: only disease-positive samples with a known Stage-2 label
    pos_mask = (y1_train == 1) & (y2_train >= 0)
    X_pos = X_train[pos_mask]
    y_pos = y2_train[pos_mask]

    if len(X_pos) == 0:
        print("[pipeline] No positive samples with Stage-2 labels; skipping Stage-2 training.")
        return

    model, scaler, le = train_stage2(X_pos, y_pos, model_type=STAGE2_MODEL)

    # Evaluate Stage-2 on test split
    print("\n[pipeline] Evaluating Stage-2 on test split …")
    X_test, y1_test, y2_test = build_features("test", extended=True)
    s1_labels, s1_proba = predict_stage1(pipeline, X_test, threshold=threshold)

    # Only test Stage-2 on samples predicted positive by Stage-1
    pred_pos_mask = s1_labels == 1
    true_pos_mask = (y1_test == 1) & (y2_test >= 0)
    eval_mask = pred_pos_mask & true_pos_mask

    if eval_mask.sum() == 0:
        print("[pipeline] No overlapping Stage-2 evaluation samples.")
        return

    s2_labels, _ = predict_stage2(model, scaler, le, X_test[eval_mask])
    evaluate_pipeline(
        y1_test, s1_labels,
        y2_test[eval_mask], s2_labels,
        proba_s1=s1_proba,
    )


def run_eval_only() -> None:
    """Load saved models and evaluate both stages on the test split."""
    pipeline, threshold = load_stage1()
    X_test, y1_test, y2_test = build_features("test", extended=True)
    s1_labels, s1_proba = predict_stage1(pipeline, X_test, threshold=threshold)

    try:
        model, scaler, le = load_stage2()
        pred_pos = s1_labels == 1
        true_pos = (y1_test == 1) & (y2_test >= 0)
        eval_mask = pred_pos & true_pos
        if eval_mask.sum() > 0:
            s2_labels, _ = predict_stage2(model, scaler, le, X_test[eval_mask])
            evaluate_pipeline(y1_test, s1_labels, y2_test[eval_mask], s2_labels, s1_proba)
        else:
            evaluate_pipeline(y1_test, s1_labels, None, None, s1_proba)
    except FileNotFoundError:
        print("[pipeline] Stage-2 model not found; evaluating Stage-1 only.")
        evaluate_pipeline(y1_test, s1_labels, None, None, s1_proba)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional breast-cancer classification pipeline"
    )
    parser.add_argument(
        "--skip-mask",
        action="store_true",
        default=False,
        help="Skip mask generation (masks already exist in data/segmented/).",
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
        "--force-mask",
        action="store_true",
        default=False,
        help="Force mask regeneration even if masks already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.eval_only:
        run_eval_only()
        return

    skip_mask = args.skip_mask and not args.force_mask

    if args.stage == 1:
        run_stage1(skip_mask=skip_mask)
    elif args.stage == 2:
        run_stage2()
    else:
        # Full pipeline
        run_stage1(skip_mask=skip_mask)
        run_stage2()


if __name__ == "__main__":
    main()
