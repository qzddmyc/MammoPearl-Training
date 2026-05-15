"""Traditional sliding-window breast lesion detector — evaluation script.

Applies the SVM trained by bbox-train-trad.py to every image in the test
split using a two-pass sliding-window strategy:

  1. Coarse scan  – slide a PATCH_SIZE window with coarse stride over the
                    image (at each requested scale), collect all windows
                    with probability >= low_thresh.
  2. Refine pass  – for each of the top-K coarse candidates, dense-scan a
                    refine_margin-pixel neighbourhood at fine_stride and
                    update the box to the highest-scoring position found.
                    This corrects for the coarse stride not landing exactly
                    on the lesion centre.
  3. NMS          – merge overlapping boxes, keep top max_det by score.
  4. Evaluate     – IoU-based box precision / recall / F1 and image-level
                    accuracy, reported at multiple score thresholds.

Multi-scale note
----------------
Because the median GT box is ~217 × 149 px (much larger than the 128-px
patch), the script by default scans at two scales:
  scale 1.0  → 128 × 128 window in original image  (small lesions)
  scale 0.5  → 128-px window on 0.5× image ≡ 256 × 256 original  (larger)
Predicted boxes from scale 0.5 are automatically rescaled back to original
image coordinates before NMS.

Performance notes
-----------------
With --stride 128 and 4 000 test images, a full evaluation takes roughly
60–90 minutes on a modern CPU (no GPU required).  Use --max-images N for
a quick sanity check (e.g. --max-images 100).

Usage (from repo root in Git Bash):
    python src/data/bounding-box/bbox-test-trad.py
    python src/data/bounding-box/bbox-test-trad.py \\
        --stride 64 --iou-match 0.2 --max-images 200
    python src/data/bounding-box/bbox-test-trad.py \\
        --stride 128 --scales 1.0 0.5 --save-csv tmp/bbox_trad_preds.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kw: x  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Repository layout & imports from recognition-traditional
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAD_DIR = REPO_ROOT / "src" / "data" / "recognition-traditional"
if str(_TRAD_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAD_DIR))

from features import extract_features          # noqa: E402  # type: ignore[import]
from preprocessing import apply_clahe, normalize_image  # noqa: E402  # type: ignore[import]

CSV_PATH         = REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv"
IMAGES_ROOT      = REPO_ROOT / "data" / "processed" / "images_png"
DEFAULT_MODEL    = REPO_ROOT / "models" / "bbox_trad"

PATCH_SIZE  = 128
RANDOM_SEED = 42

# Type alias
Box = Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original image px


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _read_gray(path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def _preprocess(patch: np.ndarray) -> np.ndarray:
    """CLAHE + normalize to float32 [0, 1].  Resize if needed."""
    if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE),
                           interpolation=cv2.INTER_AREA)
    return normalize_image(apply_clahe(patch))


# ---------------------------------------------------------------------------
# Sliding-window scan
# ---------------------------------------------------------------------------

def _scan(
    gray: np.ndarray,
    pipeline,
    stride: int,
    scale: float,
) -> tuple[list[Box], np.ndarray]:
    """Score all windows on a (possibly scaled) image.

    Returns
    -------
    boxes  : list of (x1, y1, x2, y2) in *original* image coordinates.
    scores : float32 array of SVM positive-class probabilities.
    """
    h, w = gray.shape[:2]

    if scale != 1.0:
        nw = max(PATCH_SIZE, int(round(w * scale)))
        nh = max(PATCH_SIZE, int(round(h * scale)))
        work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        work = gray
        nw, nh = w, h

    # Collect all window patches and their original-space coordinates
    boxes:   list[Box] = []
    patches: list[np.ndarray] = []

    for sy in range(0, nh - PATCH_SIZE + 1, stride):
        for sx in range(0, nw - PATCH_SIZE + 1, stride):
            patch = work[sy: sy + PATCH_SIZE, sx: sx + PATCH_SIZE]
            patches.append(_preprocess(patch.copy()))
            # Map to original image coordinates
            if scale != 1.0:
                ox1 = int(round(sx / scale))
                oy1 = int(round(sy / scale))
                ox2 = int(round((sx + PATCH_SIZE) / scale))
                oy2 = int(round((sy + PATCH_SIZE) / scale))
            else:
                ox1, oy1 = sx, sy
                ox2, oy2 = sx + PATCH_SIZE, sy + PATCH_SIZE
            boxes.append((ox1, oy1, ox2, oy2))

    if not boxes:
        return [], np.array([], dtype=np.float32)

    feats  = np.array([extract_features(p, extended=False) for p in patches],
                      dtype=np.float32)
    scores = pipeline.predict_proba(feats)[:, 1].astype(np.float32)
    return boxes, scores


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _nms(boxes: list[Box], scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    if not boxes:
        return []
    arr  = np.array(boxes, dtype=np.float32)
    idx  = np.argsort(scores)[::-1].tolist()
    keep = []

    while idx:
        best = idx[0];  keep.append(best)
        if len(idx) == 1:
            break
        rest = np.array(idx[1:], dtype=np.int64)

        bx1, by1, bx2, by2 = arr[best]
        rx1 = np.maximum(bx1, arr[rest, 0])
        ry1 = np.maximum(by1, arr[rest, 1])
        rx2 = np.minimum(bx2, arr[rest, 2])
        ry2 = np.minimum(by2, arr[rest, 3])

        inter = np.maximum(0.0, rx2 - rx1) * np.maximum(0.0, ry2 - ry1)
        ab    = (bx2 - bx1) * (by2 - by1)
        ar    = (arr[rest, 2] - arr[rest, 0]) * (arr[rest, 3] - arr[rest, 1])
        union = ab + ar - inter
        iou   = np.where(union > 0, inter / union, 0.0)
        idx   = [rest[i] for i in range(len(rest)) if iou[i] < iou_thresh]

    return keep


# ---------------------------------------------------------------------------
# Refinement pass
# ---------------------------------------------------------------------------

def _refine(
    gray: np.ndarray,
    box: Box,
    pipeline,
    margin: int,
    fine_stride: int,
    coarse_score: float,
) -> tuple[Box, float]:
    """Dense-scan a neighbourhood around *box*; return the best (box, score).

    Scans the rectangle [x1-margin .. x2+margin] × [y1-margin .. y2+margin]
    at *fine_stride* increments.  If no candidate beats *coarse_score* the
    original box and score are returned unchanged.
    """
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = box

    # Search region clipped to valid patch-start positions
    sx0 = max(0, x1 - margin)
    sy0 = max(0, y1 - margin)
    sx1 = min(w - PATCH_SIZE, x2 - PATCH_SIZE + margin)
    sy1 = min(h - PATCH_SIZE, y2 - PATCH_SIZE + margin)

    if sx1 < sx0 or sy1 < sy0:
        return box, coarse_score

    best_score = coarse_score
    best_box   = box

    patches:  list[np.ndarray]   = []
    fine_boxes: list[Box]        = []

    for sy in range(sy0, sy1 + 1, fine_stride):
        for sx in range(sx0, sx1 + 1, fine_stride):
            p = gray[sy: sy + PATCH_SIZE, sx: sx + PATCH_SIZE]
            if p.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue
            patches.append(_preprocess(p.copy()))
            fine_boxes.append((sx, sy, sx + PATCH_SIZE, sy + PATCH_SIZE))

    if not patches:
        return best_box, best_score

    feats  = np.array([extract_features(p, extended=False) for p in patches],
                      dtype=np.float32)
    scores = pipeline.predict_proba(feats)[:, 1]

    best_i = int(np.argmax(scores))
    if scores[best_i] > best_score:
        best_score = float(scores[best_i])
        best_box   = fine_boxes[best_i]

    return best_box, best_score


# ---------------------------------------------------------------------------
# Full detection pipeline for one image
# ---------------------------------------------------------------------------

def detect(
    gray: np.ndarray,
    pipeline,
    coarse_stride: int,
    fine_stride: int,
    refine_margin: int,
    refine_top_k: int,
    low_thresh: float,
    nms_iou: float,
    max_det: int,
    scales: list[float],
    do_refine: bool,
) -> list[Dict]:
    """Run the full two-pass detection pipeline on *gray*.

    Returns a list (up to max_det) of dicts, each with keys:
        'box'   : (x1, y1, x2, y2) in original image pixels
        'score' : float in [0, 1]
    sorted by descending score.
    """
    all_boxes:  list[Box]  = []
    all_scores: list[float] = []

    # ---- Pass 1: coarse multi-scale scan ----
    for scale in scales:
        boxes, scores = _scan(gray, pipeline, coarse_stride, scale)
        for b, s in zip(boxes, scores):
            if s >= low_thresh:
                all_boxes.append(b)
                all_scores.append(float(s))

    if not all_boxes:
        return []

    # ---- Coarse NMS ----
    scores_arr = np.array(all_scores, dtype=np.float32)
    nms_keep   = _nms(all_boxes, scores_arr, nms_iou)
    # Sort kept indices by score (best first) and take top-K for refinement
    nms_keep.sort(key=lambda i: all_scores[i], reverse=True)
    top_k = nms_keep[:refine_top_k]

    # ---- Pass 2: refinement around top-K candidates ----
    if do_refine:
        refined_boxes:  list[Box]   = []
        refined_scores: list[float] = []
        for idx in top_k:
            rb, rs = _refine(
                gray,
                all_boxes[idx],
                pipeline,
                margin=refine_margin,
                fine_stride=fine_stride,
                coarse_score=all_scores[idx],
            )
            refined_boxes.append(rb)
            refined_scores.append(rs)
    else:
        refined_boxes  = [all_boxes[i]  for i in top_k]
        refined_scores = [all_scores[i] for i in top_k]

    if not refined_boxes:
        return []

    # ---- Final NMS + top-max_det ----
    final_scores = np.array(refined_scores, dtype=np.float32)
    final_keep   = _nms(refined_boxes, final_scores, nms_iou)
    final_keep.sort(key=lambda i: refined_scores[i], reverse=True)

    return [
        {"box": refined_boxes[i], "score": refined_scores[i]}
        for i in final_keep[:max_det]
    ]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(
    all_dets:    list[list[Dict]],
    all_gt:      list[np.ndarray],
    threshold:   float,
    iou_match:   float,
) -> dict:
    """Compute box-level and image-level metrics at *threshold*.

    Parameters
    ----------
    all_dets  : per-image list of detection dicts {'box', 'score'}
    all_gt    : per-image GT boxes as (N, 4) float32 arrays
    threshold : score threshold; detections below are ignored
    iou_match : IoU needed to count a detection as a true positive
    """
    box_tp = box_fp = box_fn = 0
    img_tp = img_fp = img_fn = img_tn = 0

    for dets, gt in zip(all_dets, all_gt):
        active  = [d for d in dets if d["score"] >= threshold]
        has_gt  = gt.shape[0] > 0
        has_det = len(active) > 0

        # ---- Image-level ----
        if has_gt and has_det:
            matched = any(
                _iou(d["box"], (int(g[0]), int(g[1]), int(g[2]), int(g[3]))) >= iou_match
                for d in active
                for g in gt
            )
            if matched:
                img_tp += 1
            else:
                img_fp += 1
                img_fn += 1
        elif has_gt:
            img_fn += 1
        elif has_det:
            img_fp += 1
        else:
            img_tn += 1

        # ---- Box-level ----
        if not has_gt:
            box_fp += len(active)
            continue

        gt_matched = [False] * len(gt)
        for d in active:
            best_iou = 0.0;  best_j = -1
            for j, g in enumerate(gt):
                if gt_matched[j]:
                    continue
                v = _iou(d["box"], (int(g[0]), int(g[1]), int(g[2]), int(g[3])))
                if v > best_iou:
                    best_iou = v;  best_j = j
            if best_iou >= iou_match and best_j >= 0:
                box_tp += 1
                gt_matched[best_j] = True
            else:
                box_fp += 1
        box_fn += gt_matched.count(False)

    def _safe(n, d):
        return n / d if d > 0 else 0.0

    bp = _safe(box_tp, box_tp + box_fp)
    br = _safe(box_tp, box_tp + box_fn)
    bf = _safe(2 * bp * br, bp + br)
    ip = _safe(img_tp, img_tp + img_fp)
    ir = _safe(img_tp, img_tp + img_fn)
    if1 = _safe(2 * ip * ir, ip + ir)
    ia = _safe(img_tp + img_tn, img_tp + img_fp + img_fn + img_tn)

    return dict(
        box_tp=box_tp, box_fp=box_fp, box_fn=box_fn,
        box_precision=bp, box_recall=br, box_f1=bf,
        img_tp=img_tp, img_fp=img_fp, img_fn=img_fn, img_tn=img_tn,
        img_precision=ip, img_recall=ir, img_f1=if1, img_accuracy=ia,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate traditional sliding-window bbox detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir",    type=Path,  default=DEFAULT_MODEL)
    p.add_argument("--split",        default="test")
    p.add_argument("--stride",       type=int,   default=128,
                   help="Coarse scan stride in pixels. "
                        "stride=64 gives ~4× more windows (better recall, ~4× slower).")
    p.add_argument("--fine-stride",  type=int,   default=32,
                   help="Stride for the refinement dense scan")
    p.add_argument("--refine-margin",type=int,   default=64,
                   help="Pixels to expand around each coarse candidate for refinement")
    p.add_argument("--refine-top-k", type=int,   default=5,
                   help="Number of coarse candidates to refine per image")
    p.add_argument("--no-refine",    action="store_true",
                   help="Disable the refinement pass (faster, less precise)")
    p.add_argument("--scales",       type=float, nargs="+", default=[1.0, 0.5],
                   help="Image scales for multi-scale detection")
    p.add_argument("--low-thresh",   type=float, default=0.15,
                   help="Minimum probability for a window to enter the candidate pool")
    p.add_argument("--nms-iou",      type=float, default=0.3,
                   help="IoU threshold for NMS")
    p.add_argument("--max-det",      type=int,   default=5,
                   help="Maximum detections to keep per image")
    p.add_argument("--iou-match",    type=float, default=0.3,
                   help="IoU threshold for counting a detection as TP")
    p.add_argument("--score-thresh", type=float, default=None,
                   help="Override the model's saved threshold for evaluation")
    p.add_argument("--max-images",   type=int,   default=None,
                   help="Process only the first N test images (for quick testing)")
    p.add_argument("--save-csv",     type=Path,  default=None,
                   help="Save per-detection results to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load model ----
    model_dir = args.model_dir
    with open(model_dir / "svm_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(model_dir / "threshold.pkl", "rb") as f:
        saved_threshold = float(pickle.load(f))
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)

    eval_threshold = args.score_thresh if args.score_thresh is not None \
                     else saved_threshold

    print(f"Model         : {model_dir}")
    print(f"Patch size    : {meta.get('patch_size', PATCH_SIZE)} px")
    print(f"Train scales  : {meta.get('pos_scales', [1.0])}")
    print(f"Saved thresh  : {saved_threshold:.3f}")
    print(f"Eval thresh   : {eval_threshold:.3f}")
    print(f"Scan scales   : {args.scales}")
    print(f"Coarse stride : {args.stride} px")
    print(f"Refine        : {'disabled (--no-refine)' if args.no_refine else f'top-{args.refine_top_k}, margin={args.refine_margin}, fine_stride={args.fine_stride}'}")
    print(f"IoU match     : >= {args.iou_match}")

    # ---- Load test data ----
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"].str.lower() == args.split.lower()].copy()
    grouped = list(df.groupby(["patient_id", "image_id"], sort=True))

    if args.max_images:
        grouped = grouped[:args.max_images]
        print(f"\n[Note] Evaluating first {args.max_images} images only.")

    print(f"\nRunning detection on {len(grouped)} images …")

    all_dets:    list[list[Dict]]   = []
    all_gt:      list[np.ndarray]   = []
    csv_rows:    list[dict]         = []

    for (patient_id, image_id), grp in tqdm(grouped, unit="img"):
        img_path = IMAGES_ROOT / str(patient_id) / str(image_id)
        if not img_path.exists():
            all_dets.append([])
            all_gt.append(np.zeros((0, 4), dtype=np.float32))
            continue

        gray = _read_gray(img_path)
        if gray is None:
            all_dets.append([])
            all_gt.append(np.zeros((0, 4), dtype=np.float32))
            continue

        # ---- GT boxes ----
        has_finding = int(grp["No_Finding"].iloc[0]) == 0
        valid = grp[["xmin", "ymin", "xmax", "ymax"]].dropna()
        if has_finding and not valid.empty:
            gt = valid.to_numpy(dtype=np.float32)
            gt = gt[(gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1])]
        else:
            gt = np.zeros((0, 4), dtype=np.float32)

        # ---- Detect ----
        dets = detect(
            gray,
            pipeline,
            coarse_stride=args.stride,
            fine_stride=args.fine_stride,
            refine_margin=args.refine_margin,
            refine_top_k=args.refine_top_k,
            low_thresh=args.low_thresh,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
            scales=args.scales,
            do_refine=not args.no_refine,
        )

        all_dets.append(dets)
        all_gt.append(gt)

        if args.save_csv is not None:
            for d in dets:
                csv_rows.append(dict(
                    patient_id=patient_id, image_id=image_id,
                    score=d["score"],
                    x1=d["box"][0], y1=d["box"][1],
                    x2=d["box"][2], y2=d["box"][3],
                ))

    # ---- Multi-threshold evaluation ----
    thresholds = sorted({0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, eval_threshold})
    print(f"\n=== Evaluation (IoU-match >= {args.iou_match}) ===")
    header = (f"{'Thresh':>7} | {'Box-P':>7} {'Box-R':>7} {'Box-F1':>8} |"
              f" {'Img-P':>7} {'Img-R':>7} {'Img-F1':>8} {'Img-Acc':>9}")
    print(header)
    print("-" * len(header))

    for t in thresholds:
        m = evaluate(all_dets, all_gt, t, args.iou_match)
        mark = " ←" if abs(t - eval_threshold) < 1e-5 else ""
        print(
            f"  {t:.3f}  |"
            f" {m['box_precision']:7.4f} {m['box_recall']:7.4f} {m['box_f1']:8.4f} |"
            f" {m['img_precision']:7.4f} {m['img_recall']:7.4f} {m['img_f1']:8.4f}"
            f" {m['img_accuracy']:9.4f}{mark}"
        )

    # ---- Detailed breakdown ----
    m = evaluate(all_dets, all_gt, eval_threshold, args.iou_match)
    print(f"\n=== Details at threshold={eval_threshold:.3f} ===")
    print(f"  Box  TP={m['box_tp']} FP={m['box_fp']} FN={m['box_fn']}")
    print(f"  Img  TP={m['img_tp']} FP={m['img_fp']} FN={m['img_fn']} TN={m['img_tn']}")
    print(f"  Box  Precision={m['box_precision']:.4f}  Recall={m['box_recall']:.4f}"
          f"  F1={m['box_f1']:.4f}")
    print(f"  Img  Precision={m['img_precision']:.4f}  Recall={m['img_recall']:.4f}"
          f"  F1={m['img_f1']:.4f}  Accuracy={m['img_accuracy']:.4f}")

    if args.save_csv and csv_rows:
        out_df = pd.DataFrame(csv_rows)
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save_csv, index=False)
        print(f"\nDetections saved → {args.save_csv}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
