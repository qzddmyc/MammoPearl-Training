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
