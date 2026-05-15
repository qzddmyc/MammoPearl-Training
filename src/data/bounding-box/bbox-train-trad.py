"""Traditional sliding-window breast lesion detector — training script.

Trains a binary SVM patch classifier that scores any 128×128 image patch
as "lesion" or "normal tissue".  The resulting model can be applied as a
sliding-window detector at inference time (see bbox-test-trad.py).

Design choices
--------------
* Positive patches  – cropped from GT bbox centers with spatial jitter, so
  the model fires on windows that *partially* overlap a lesion, not only on
  perfectly-centred crops.  Patches at scale 0.5 (256-px equivalent) are
  also generated so the model learns multi-resolution lesion texture.
* Negative patches  – hard negatives mined from the densest non-GT areas via
  an integral-image fast-scan, mimicking the dense glandular regions most
  likely to cause false alarms during sliding-window inference.
* Priority          – Recall > Precision (missed lesions are worse than false
  alarms).  The decision threshold is tuned on the training data to achieve
  Recall >= recall-target (default 0.85).

Output
------
models/bbox_trad/
    svm_pipeline.pkl    sklearn Pipeline: StandardScaler -> PCA -> SVC
    threshold.pkl       probability threshold tuned for high recall
    meta.json           training configuration record

Usage (from repo root in Git Bash):
    python src/data/bounding-box/bbox-train-trad.py
    python src/data/bounding-box/bbox-train-trad.py \\
        --jitter-n 6 --hard-neg 4 --recall-target 0.90
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm
import contextlib
import joblib


@contextlib.contextmanager
def _tqdm_joblib(tqdm_obj):
    """Patch joblib so each completed batch increments a tqdm bar."""
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_obj.close()

# ---------------------------------------------------------------------------
# Repository layout & imports from recognition-traditional
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAD_DIR = REPO_ROOT / "src" / "data" / "recognition-traditional"
if str(_TRAD_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAD_DIR))

from features import extract_features          # noqa: E402  # type: ignore[import]
from preprocessing import apply_clahe, normalize_image, to_gray  # noqa: E402  # type: ignore[import]

CSV_PATH    = REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv"
IMAGES_ROOT = REPO_ROOT / "data" / "processed" / "images_png"
DEFAULT_OUT = REPO_ROOT / "models" / "bbox_trad"

PATCH_SIZE  = 128
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _read_gray(path: Path) -> np.ndarray | None:
    """Load an image from *path* as uint8 grayscale (unicode-safe)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    return img  # None when decode fails


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _crop_centered(
    gray: np.ndarray,
    cx: float,
    cy: float,
    src_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Crop a src_size × src_size region centred at (cx, cy).

    Out-of-bounds areas are filled with BORDER_REFLECT_101.
    The crop is always resized to PATCH_SIZE × PATCH_SIZE.
    """
    h, w = gray.shape[:2]
    half = src_size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x1 + src_size - w)
    pad_b = max(0, y1 + src_size - h)

    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w, x1 + src_size);  y2c = min(h, y1 + src_size)
    patch = gray[y1c:y2c, x1c:x2c]

    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REFLECT_101)
    if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE),
                           interpolation=cv2.INTER_AREA)
    return patch


def _preprocess(patch_gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + normalize to float32 [0, 1]."""
    if patch_gray.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        patch_gray = cv2.resize(patch_gray, (PATCH_SIZE, PATCH_SIZE),
                                interpolation=cv2.INTER_AREA)
    return normalize_image(apply_clahe(patch_gray))


def _gen_positives(
    gray: np.ndarray,
    bboxes: np.ndarray,
    rng: np.random.Generator,
    n_jitter: int,
    jitter_std: float,
    scales: tuple[float, ...] = (1.0,),
) -> list[np.ndarray]:
    """Generate positive patches around each GT bbox.

    For each bbox, produce:
      - One centre crop at each scale.
      - n_jitter randomly jittered crops at each scale.

    At scale s the crop src_size = round(PATCH_SIZE / s) pixels, which
    corresponds to PATCH_SIZE / s pixels in the original image.
    """
    patches: list[np.ndarray] = []
    h, w = gray.shape[:2]

    for scale in scales:
        src_size = max(PATCH_SIZE, int(round(PATCH_SIZE / scale)))
        # Optionally downsample the image first for scales < 1
        if scale < 1.0:
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            work = gray

        for box in bboxes:
            # Map box centre to (possibly scaled) image coordinates
            cx = (box[0] + box[2]) * 0.5 * scale
            cy = (box[1] + box[3]) * 0.5 * scale

            # Centre crop
            patches.append(_preprocess(_crop_centered(work, cx, cy, PATCH_SIZE)))

            # Jittered crops
            jitter_scale = jitter_std * scale
            for _ in range(n_jitter):
                dx = rng.normal(0.0, jitter_scale)
                dy = rng.normal(0.0, jitter_scale)
                patches.append(_preprocess(
                    _crop_centered(work, cx + dx, cy + dy, PATCH_SIZE)))

    return patches


def _gen_random_negatives(
    gray: np.ndarray,
    bboxes: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[np.ndarray]:
    """Sample n random patches from anywhere outside the GT bbox regions.

    This ensures the model learns to reject easy negatives (background,
    air columns, peripheral tissue) that are abundant at sliding-window
    inference time but absent when only hard negatives are used.
    """
    h, w = gray.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE or n <= 0:
        return []

    # Exclusion mask: GT bboxes + one-patch margin
    excl = np.zeros((h, w), dtype=np.uint8)
    for box in bboxes:
        ex1 = int(max(0, box[0] - PATCH_SIZE))
        ey1 = int(max(0, box[1] - PATCH_SIZE))
        ex2 = int(min(w, box[2] + PATCH_SIZE))
        ey2 = int(min(h, box[3] + PATCH_SIZE))
        excl[ey1:ey2, ex1:ex2] = 1

    # Build list of valid (y, x) top-left positions
    ys = np.arange(0, h - PATCH_SIZE + 1, PATCH_SIZE // 2)
    xs = np.arange(0, w - PATCH_SIZE + 1, PATCH_SIZE // 2)
    valid: list[tuple[int, int]] = []
    for y in ys:
        for x in xs:
            if not excl[y:y + PATCH_SIZE, x:x + PATCH_SIZE].any():
                valid.append((y, x))

    if not valid:
        return []

    chosen_idx = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    patches: list[np.ndarray] = []
    for i in chosen_idx:
        y, x = valid[int(i)]
        patches.append(_preprocess(gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()))
    return patches


def _gen_hard_negatives(
    gray: np.ndarray,
    bboxes: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[np.ndarray]:
    """Mine n hard-negative patches from the densest non-GT areas.

    Uses an integral image for O(1) window-sum computation.
    The GT bbox regions (plus a PATCH_SIZE margin) are excluded.
    """
    h, w = gray.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE or n <= 0:
        return []

    # Exclusion mask: GT bboxes + one-patch margin
    excl = np.zeros((h, w), dtype=np.uint8)
    for box in bboxes:
        ex1 = int(max(0, box[0] - PATCH_SIZE))
        ey1 = int(max(0, box[1] - PATCH_SIZE))
        ex2 = int(min(w, box[2] + PATCH_SIZE))
        ey2 = int(min(h, box[3] + PATCH_SIZE))
        excl[ey1:ey2, ex1:ex2] = 1

    integral = cv2.integral(gray.astype(np.float32))
    stride = PATCH_SIZE // 2
    candidates: list[tuple[float, int, int]] = []

    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            if excl[y:y + PATCH_SIZE, x:x + PATCH_SIZE].any():
                continue
            s = (integral[y + PATCH_SIZE, x + PATCH_SIZE]
                 - integral[y, x + PATCH_SIZE]
                 - integral[y + PATCH_SIZE, x]
                 + integral[y, x])
            candidates.append((float(s), y, x))

    if not candidates:
        return []

    candidates.sort(reverse=True)
    pool = candidates[: max(n * 5, 30)]
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)

    patches: list[np.ndarray] = []
    for idx in chosen:
        _, y, x = pool[int(idx)]
        patches.append(_preprocess(gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()))
    return patches


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    split: str,
    n_jitter: int,
    jitter_std: float,
    hard_neg: int,
    hard_neg_nofinding: int,
    rand_neg: int,
    rand_neg_nofinding: int,
    pos_scales: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the feature matrix and binary labels for training.

    Negatives are a mix of:
    * Hard negatives  – densest non-GT regions (mimic difficult FP cases)
    * Random negatives – uniformly sampled non-GT patches (teach the model
      to reject easy regions: background, air, peripheral tissue)

    Returns
    -------
    X : (N, 113) float32  – feature matrix (113-dim base features)
    y : (N,) int32        – 1 = lesion, 0 = normal tissue
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"].str.lower() == split.lower()].copy()
    grouped = list(df.groupby(["patient_id", "image_id"], sort=True))

    all_patches: list[np.ndarray] = []
    all_labels:  list[int]        = []

    for (patient_id, image_id), grp in tqdm(grouped, desc="Building patches", unit="img"):
        img_path = IMAGES_ROOT / str(patient_id) / str(image_id)
        if not img_path.exists():
            continue
        gray = _read_gray(img_path)
        if gray is None:
            continue

        has_finding = int(grp["No_Finding"].iloc[0]) == 0
        valid = grp[["xmin", "ymin", "xmax", "ymax"]].dropna()

        if has_finding and not valid.empty:
            bboxes = valid.to_numpy(dtype=np.float32)
            # Drop degenerate boxes
            bboxes = bboxes[(bboxes[:, 2] > bboxes[:, 0] + 4) &
                            (bboxes[:, 3] > bboxes[:, 1] + 4)]
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        is_lesion = bboxes.size > 0

        # ---- Positive patches ----
        if is_lesion:
            pos_patches = _gen_positives(gray, bboxes, rng, n_jitter,
                                         jitter_std, pos_scales)
            for p in pos_patches:
                all_patches.append(p)
                all_labels.append(1)

        # ---- Hard negatives (dense tissue) ----
        n_hard = hard_neg if is_lesion else hard_neg_nofinding
        for p in _gen_hard_negatives(gray, bboxes, rng, n_hard):
            all_patches.append(p)
            all_labels.append(0)

        # ---- Random negatives (diverse tissue / background) ----
        n_rand = rand_neg if is_lesion else rand_neg_nofinding
        for p in _gen_random_negatives(gray, bboxes, rng, n_rand):
            all_patches.append(p)
            all_labels.append(0)

    n_pos = sum(l == 1 for l in all_labels)
    n_neg = len(all_labels) - n_pos
    print(f"  Patches — positives: {n_pos:,}   negatives: {n_neg:,}")

    print("Extracting features …")
    feats = [
        extract_features(p, extended=False)
        for p in tqdm(all_patches, desc="Features", unit="patch")
    ]
    X = np.array(feats, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def _tune_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    recall_target: float,
) -> float:
    """Sweep thresholds; pick the one maximising F1 subject to Recall >= target."""
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (proba >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f1  = f1_score(y_true, preds, zero_division=0)
        if rec >= recall_target and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    pca_n: int,
    recall_target: float,
) -> tuple[Pipeline, float]:
    print(f"\nTraining SVM  (samples={len(y):,}, pos={int(y.sum()):,}, pca={pca_n}) …")
    nan_count = int(np.isnan(X).sum())
    if nan_count:
        print(f"  [warn] {nan_count} NaN values in feature matrix — will be imputed to 0")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=pca_n, random_state=RANDOM_SEED)),
        ("svm",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    param_grid = {
        "svm__C":     [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", "auto"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    n_fits = 5 * len(param_grid["svm__C"]) * len(param_grid["svm__gamma"])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="recall",
                        n_jobs=-1, verbose=0, refit=True)
    with _tqdm_joblib(tqdm(total=n_fits, desc="GridSearchCV", unit="fit")):
        grid.fit(X, y)
    best = grid.best_estimator_

    print(f"  Best params   : {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")

    # Tune threshold on full training data (in-sample, conservative upper bound)
    proba = best.predict_proba(X)[:, 1]
    threshold = _tune_threshold(y, proba, recall_target)
    preds = (proba >= threshold).astype(int)
    print(f"  Tuned threshold : {threshold:.3f}  "
          f"Recall={recall_score(y, preds):.4f}  F1={f1_score(y, preds):.4f}")
    return best, threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train traditional sliding-window bbox detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split",      default="training",
                   help="CSV split name to train on")
    p.add_argument("--jitter-n",   type=int,   default=4,
                   help="Number of jittered positive patches per GT bbox per scale")
    p.add_argument("--jitter-std", type=float, default=28.0,
                   help="Std-dev (pixels) of position jitter for positive patches")
    p.add_argument("--pos-scales", type=float, nargs="+", default=[1.0, 0.5],
                   help="Image scales for positive patch generation "
                        "(1.0=128px window, 0.5=256px window in original)")
    p.add_argument("--hard-neg",   type=int,   default=3,
                   help="Hard (dense-tissue) negatives per lesion image")
    p.add_argument("--hard-neg-nofinding", type=int, default=1,
                   help="Hard negatives per No-Finding image")
    p.add_argument("--rand-neg",   type=int,   default=2,
                   help="Random negatives per lesion image")
    p.add_argument("--rand-neg-nofinding", type=int, default=4,
                   help="Random negatives per No-Finding image "
                        "(key for reducing false positives on normal tissue)")
    p.add_argument("--max-neg-ratio", type=float, default=3.0,
                   help="Cap total negatives to at most this multiple of positives "
                        "(0 = no cap); applied after feature extraction so cached "
                        "features can be reused at different ratio settings")
    p.add_argument("--pca-n",      type=int,   default=80,
                   help="Number of PCA components")
    p.add_argument("--recall-target", type=float, default=0.85,
                   help="Minimum recall to satisfy when tuning the decision threshold")
    p.add_argument("--output-dir", type=Path,  default=DEFAULT_OUT,
                   help="Directory to save model artefacts")
    p.add_argument("--cache-dir",  type=Path,  default=None,
                   help="Feature cache directory (default: <output-dir>/cache)")
    p.add_argument("--no-cache",   action="store_true",
                   help="Recompute features even if a cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir   = args.output_dir
    cache_dir = args.cache_dir or (out_dir / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scales_tag = "_".join(f"{s:.2f}" for s in sorted(args.pos_scales))
    neg_tag = (f"h{args.hard_neg}r{args.rand_neg}"
               f"_hN{args.hard_neg_nofinding}rN{args.rand_neg_nofinding}")
    cache_x = cache_dir / f"X_{args.split}_s{scales_tag}_{neg_tag}.npy"
    cache_y = cache_dir / f"y_{args.split}_s{scales_tag}_{neg_tag}.npy"

    if not args.no_cache and cache_x.exists() and cache_y.exists():
        print(f"Loading cached features from {cache_dir} …")
        X = np.load(cache_x)
        y = np.load(cache_y)
        print(f"  X={X.shape}  y={y.shape}  "
              f"pos={int(y.sum()):,}  neg={int((y == 0).sum()):,}")
    else:
        X, y = build_dataset(
            split=args.split,
            n_jitter=args.jitter_n,
            jitter_std=args.jitter_std,
            hard_neg=args.hard_neg,
            hard_neg_nofinding=args.hard_neg_nofinding,
            rand_neg=args.rand_neg,
            rand_neg_nofinding=args.rand_neg_nofinding,
            pos_scales=tuple(args.pos_scales),
        )
        np.save(cache_x, X)
        np.save(cache_y, y)
        print(f"Features cached → {cache_dir}")

    # ---- Negative cap (applied after cache load/build) ----
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if args.max_neg_ratio > 0:
        max_neg = int(args.max_neg_ratio * n_pos)
        if n_neg > max_neg:
            _rng = np.random.default_rng(RANDOM_SEED)
            neg_idx  = np.where(y == 0)[0]
            keep_neg = _rng.choice(neg_idx, size=max_neg, replace=False)
            keep_all = np.sort(np.concatenate([np.where(y == 1)[0], keep_neg]))
            X, y = X[keep_all], y[keep_all]
            n_neg = max_neg
            print(f"  Neg cap ({args.max_neg_ratio:.1f}× pos): subsampled → "
                  f"pos={n_pos:,}  neg={n_neg:,}  total={len(y):,}")

    pipeline, threshold = train(X, y, pca_n=args.pca_n,
                                 recall_target=args.recall_target)

    # Persist
    with open(out_dir / "svm_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(out_dir / "threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)

    meta = {
        "split":                args.split,
        "jitter_n":             args.jitter_n,
        "jitter_std":           args.jitter_std,
        "pos_scales":           sorted(args.pos_scales),
        "hard_neg":             args.hard_neg,
        "hard_neg_nofinding":   args.hard_neg_nofinding,
        "rand_neg":             args.rand_neg,
        "rand_neg_nofinding":   args.rand_neg_nofinding,
        "max_neg_ratio":        args.max_neg_ratio,
        "pca_n":                args.pca_n,
        "recall_target":        args.recall_target,
        "threshold":            float(threshold),
        "patch_size":           PATCH_SIZE,
        "feature_dim":          int(X.shape[1]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {out_dir}/")
    print("  svm_pipeline.pkl  threshold.pkl  meta.json")


if __name__ == "__main__":
    main()
