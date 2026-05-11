"""Patch sampling utilities.

Responsibilities
----------------
1. generate_masks()   – call build_training_masks_from_csv once to create
                        data/segmented/mask/** if not already done.
2. build_dataset()    – iterate the CSV and collect:
                        * positive patches  (from annotated bbox regions)
                        * hard-negative patches (from densest non-ROI regions)
   Returns two parallel lists: patches (np.ndarray) and records (dict with
   metadata including Stage-1 and Stage-2 labels).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow importing from src/data/segment without installing the package
_SEGMENT_DIR = Path(__file__).resolve().parents[1] / "segment"
if str(_SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGMENT_DIR))

from segment import build_training_masks_from_csv  # noqa: E402

from config import (
    COL_IMAGE,
    COL_NO_FINDING,
    COL_PATIENT,
    COL_SPLIT,
    COL_XMAX,
    COL_XMIN,
    COL_YMAX,
    COL_YMIN,
    CSV_PATH,
    FINDING_COLS,
    HARD_NEG_PER_IMAGE,
    IMAGES_ROOT,
    MASK_ROOT,
    PATCH_SIZE,
    RANDOM_SEED,
    SEGMENTED_ROOT,
    STAGE2_MERGE_MAP,
)


# ---------------------------------------------------------------------------
# Step 0 helper
# ---------------------------------------------------------------------------

def generate_masks(force: bool = False) -> int:
    """Generate segmentation masks for the training split.

    Parameters
    ----------
    force:
        If False and masks already exist, skip generation.
    Returns the number of masks written (0 if skipped).
    """
    existing = list(MASK_ROOT.rglob("*.png"))
    if existing and not force:
        print(f"[sampling] Masks already exist ({len(existing)} files), skipping generation.")
        return 0
    print("[sampling] Generating segmentation masks …")
    n = build_training_masks_from_csv(
        csv_path=CSV_PATH,
        images_root=IMAGES_ROOT,
        segmented_root=SEGMENTED_ROOT,
    )
    print(f"[sampling] Generated {n} masks.")
    return n


# ---------------------------------------------------------------------------
# Hard-negative mining helper
# ---------------------------------------------------------------------------

def _hard_negative_patches(
    gray: np.ndarray,
    bbox_rows: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample n hard-negative patches from the densest non-ROI regions.

    Strategy: compute the cumulative brightness map, mask out all bbox
    regions, and pick the top-n brightest windows of size PATCH_SIZE.
    """
    h, w = gray.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        return []

    # Build exclusion mask (all bbox regions are excluded)
    excl = np.zeros((h, w), dtype=np.uint8)
    for _, row in bbox_rows.iterrows():
        x1 = int(max(0, min(w - 1, round(float(row[COL_XMIN])))))
        y1 = int(max(0, min(h - 1, round(float(row[COL_YMIN])))))
        x2 = int(max(0, min(w, round(float(row[COL_XMAX])))))
        y2 = int(max(0, min(h, round(float(row[COL_YMAX])))))
        excl[y1:y2, x1:x2] = 1

    # Compute integral image for fast window-sum computation
    float_gray = gray.astype(np.float32)
    integral = cv2.integral(float_gray)

    stride = PATCH_SIZE // 2
    best: list[tuple[float, int, int]] = []

    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            # Skip windows that overlap with any bbox
            if excl[y:y + PATCH_SIZE, x:x + PATCH_SIZE].any():
                continue
            # Window sum from integral image
            win_sum = (
                integral[y + PATCH_SIZE, x + PATCH_SIZE]
                - integral[y, x + PATCH_SIZE]
                - integral[y + PATCH_SIZE, x]
                + integral[y, x]
            )
            best.append((win_sum, y, x))

    if not best:
        return []

    best.sort(key=lambda t: t[0], reverse=True)
    top = best[: max(n * 3, 20)]  # candidate pool
    chosen = rng.choice(len(top), size=min(n, len(top)), replace=False)

    patches = []
    for idx in chosen:
        _, y, x = top[int(idx)]
        patch = gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        patches.append(patch)
    return patches


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _load_image(patient_id: str, image_id: str) -> np.ndarray | None:
    """Load a processed image as a grayscale numpy array."""
    path = IMAGES_ROOT / patient_id / image_id
    if not path.exists():
        return None
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        pass
    else:
        return None
    return img


def _stage2_label(row: pd.Series) -> int:
    """Return the merged Stage-2 class index for a disease row.

    Returns the STAGE2_MERGE_MAP value for the first active FINDING_COLS
    entry, or -1 for 'No Finding'.
    """
    if row.get(COL_NO_FINDING, 1) == 1:
        return -1
    for i, col in enumerate(FINDING_COLS):
        if row.get(col, 0) == 1:
            return STAGE2_MERGE_MAP.get(i, -1)
    return -1  # no known finding column active


def build_dataset(split: str = "training") -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Build the patch dataset for the given split.

    Parameters
    ----------
    split:
        One of ``"training"`` or ``"test"``.

    Returns
    -------
    patches : list of np.ndarray
        Raw grayscale patches of shape (PATCH_SIZE, PATCH_SIZE).
    records : list of dict
        Metadata per patch:
        * patient_id, image_id
        * stage1_label : int  (1 = disease, 0 = no finding)
        * stage2_label : int  (index into FINDING_COLS, or -1)
        * is_hard_neg  : bool
    """
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    split_df = df[df[COL_SPLIT].astype(str).str.lower() == split.lower()].copy()

    patches: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    grouped = split_df.groupby([COL_PATIENT, COL_IMAGE], sort=True)
    total = len(grouped)

    for (patient_id, image_id), group in tqdm(
        grouped, total=total, desc="[sampling] images", unit="img"
    ):
        patient_id = str(patient_id)
        image_id = str(image_id)

        gray = _load_image(patient_id, image_id)
        if gray is None:
            continue

        h, w = gray.shape[:2]

        # ------------------------------------------------------------------
        # Determine per-image No_Finding status (use first row)
        # ------------------------------------------------------------------
        first_row = group.iloc[0]
        no_finding = int(first_row.get(COL_NO_FINDING, 0))

        # ------------------------------------------------------------------
        # Positive patches: crop from each annotated bbox
        # ------------------------------------------------------------------
        valid_bbox = group[[COL_XMIN, COL_YMIN, COL_XMAX, COL_YMAX]].dropna()

        if no_finding == 0 and not valid_bbox.empty:
            for _, bbox_row in valid_bbox.iterrows():
                x1 = int(max(0, min(w - 1, round(float(bbox_row[COL_XMIN])))))
                y1 = int(max(0, min(h - 1, round(float(bbox_row[COL_YMIN])))))
                x2 = int(max(0, min(w, round(float(bbox_row[COL_XMAX])))))
                y2 = int(max(0, min(h, round(float(bbox_row[COL_YMAX])))))

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = gray[y1:y2, x1:x2]
                roi_resized = cv2.resize(
                    roi, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA
                )

                # Determine Stage-2 label from the row that owns this bbox
                full_row = group[
                    (group[COL_XMIN] == bbox_row[COL_XMIN])
                    & (group[COL_YMIN] == bbox_row[COL_YMIN])
                ]
                s2 = _stage2_label(full_row.iloc[0] if not full_row.empty else first_row)

                patches.append(roi_resized)
                records.append(
                    dict(
                        patient_id=patient_id,
                        image_id=image_id,
                        stage1_label=1,
                        stage2_label=s2,
                        is_hard_neg=False,
                    )
                )

        # ------------------------------------------------------------------
        # Hard-negative patches: densest non-ROI windows
        # ------------------------------------------------------------------
        neg_patches = _hard_negative_patches(gray, valid_bbox, HARD_NEG_PER_IMAGE, rng)
        for np_patch in neg_patches:
            patches.append(np_patch)
            records.append(
                dict(
                    patient_id=patient_id,
                    image_id=image_id,
                    stage1_label=0,
                    stage2_label=-1,
                    is_hard_neg=True,
                )
            )

    print(f"[sampling] Done. Total patches: {len(patches)}")
    return patches, records
