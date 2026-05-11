"""Global configuration for the traditional breast-cancer classification pipeline."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (three levels up from this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CSV_PATH = REPO_ROOT / "data" / "raw" / "vindr_detection_folds.csv"
IMAGES_ROOT = REPO_ROOT / "data" / "processed" / "images_png"
SEGMENTED_ROOT = REPO_ROOT / "data" / "segmented"
MASK_ROOT = SEGMENTED_ROOT / "mask"
BASE_ROOT = SEGMENTED_ROOT / "base"

# ---------------------------------------------------------------------------
# Output paths (models, features, reports)
# ---------------------------------------------------------------------------
OUTPUT_ROOT = REPO_ROOT / "src" / "data" / "recognition-traditional" / "output"
FEATURE_CACHE_DIR = OUTPUT_ROOT / "features"
MODEL_DIR = OUTPUT_ROOT / "models"
REPORT_DIR = OUTPUT_ROOT / "reports"

for _d in (FEATURE_CACHE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------
COL_PATIENT = "patient_id"
COL_IMAGE = "image_id"
COL_SPLIT = "split"
COL_XMIN = "xmin"
COL_YMIN = "ymin"
COL_XMAX = "xmax"
COL_YMAX = "ymax"
COL_NO_FINDING = "No_Finding"

# All disease-type one-hot columns (used for Stage-2 label)
FINDING_COLS = [
    "Architectural_Distortion",
    "Asymmetry",
    "Focal_Asymmetry",
    "Global_Asymmetry",
    "Mass",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
    "Suspicious_Lymph_Node",
]

# ---------------------------------------------------------------------------
# Patch / sampling parameters
# ---------------------------------------------------------------------------
PATCH_SIZE = 128          # pixels; positive and negative patches are resized to this
HARD_NEG_PER_IMAGE = 3   # number of hard negative patches to mine per image
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ---------------------------------------------------------------------------
# Feature extraction parameters
# ---------------------------------------------------------------------------
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, 0.785, 1.571, 2.356]  # 0, 45, 90, 135 degrees in radians

LBP_RADIUS = 3
LBP_N_POINTS = 24          # 8 * radius

WAVELET = "db4"
WAVELET_LEVEL = 2

GABOR_FREQUENCIES = (0.1, 0.2, 0.4)
GABOR_THETAS = (0, 0.524, 1.047, 1.571)   # 0, 30, 60, 90 degrees

PCA_N_COMPONENTS = 80

# ---------------------------------------------------------------------------
# Stage-1 (binary) training parameters
# ---------------------------------------------------------------------------
STAGE1_MODEL = "svm"            # "svm" | "rf"
STAGE1_DECISION_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Stage-2 (multi-class) training parameters
# ---------------------------------------------------------------------------
STAGE2_MODEL = "xgboost"       # "xgboost" | "lightgbm"

# Merge mapping: original 10-class index → 4 merged class index.
# Rationale: classes with very few training samples cannot be learned
# reliably; grouping by clinical similarity improves model coverage.
# Focal_Asymmetry is kept with other asymmetry types because separating
# it (50 vs 52 samples) hurt Macro F1 significantly (0.568 → 0.449).
#
# Original index → FINDING_COLS[index]
#   0  Architectural_Distortion  ┐
#   1  Asymmetry                 ├─→ 0  Asymmetry_Distortion  (~102 test)
#   2  Focal_Asymmetry           │
#   3  Global_Asymmetry          ┘
#   4  Mass                      ──→ 1  Mass                  (~232 test)
#   5  Nipple_Retraction         ┐
#   6  Skin_Retraction           ├─→ 2  Skin_Other            (~20 test)
#   7  Skin_Thickening           │
#   9  Suspicious_Lymph_Node     ┘
#   8  Suspicious_Calcification  ──→ 3  Suspicious_Calcification (~86 test)
STAGE2_MERGE_MAP: dict[int, int] = {
    0: 0,  # Architectural_Distortion  → Asymmetry_Distortion
    1: 0,  # Asymmetry                 → Asymmetry_Distortion
    2: 0,  # Focal_Asymmetry           → Asymmetry_Distortion
    3: 0,  # Global_Asymmetry          → Asymmetry_Distortion
    4: 1,  # Mass                      → Mass
    5: 2,  # Nipple_Retraction         → Skin_Other
    6: 2,  # Skin_Retraction           → Skin_Other
    7: 2,  # Skin_Thickening           → Skin_Other
    8: 3,  # Suspicious_Calcification  → Suspicious_Calcification
    9: 2,  # Suspicious_Lymph_Node     → Skin_Other
}

STAGE2_MERGED_NAMES: list[str] = [
    "Asymmetry_Distortion",     # 0
    "Mass",                     # 1
    "Skin_Other",               # 2
    "Suspicious_Calcification", # 3
]
