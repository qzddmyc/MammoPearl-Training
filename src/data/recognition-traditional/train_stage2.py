"""Stage-2 multi-class classifier: identify the type of finding.

Only samples predicted as positive by Stage-1 are fed here.

Pipeline
--------
StandardScaler → XGBoost (or LightGBM) multi-class classifier.
One-vs-Rest wrapper is applied when ``ovo=False`` (default = OvR which
is natively supported by both libraries via ``objective="multi:softmax"``).
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    FINDING_COLS,
    MODEL_DIR,
    RANDOM_SEED,
    STAGE2_MODEL,
    STAGE2_MERGED_NAMES,
)

_STAGE2_MODEL_PATH = MODEL_DIR / "stage2_model.pkl"
_STAGE2_SCALER_PATH = MODEL_DIR / "stage2_scaler.pkl"
_STAGE2_ENCODER_PATH = MODEL_DIR / "stage2_label_encoder.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_xgb(n_classes: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for STAGE2_MODEL='xgboost'") from exc

    return XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _build_lgbm(n_classes: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required for STAGE2_MODEL='lightgbm'") from exc

    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_stage2(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = STAGE2_MODEL,
    n_splits: int = 5,
) -> tuple[object, StandardScaler, LabelEncoder]:
    """Train the Stage-2 multi-class classifier.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Extended feature matrix (includes calcification/mass features).
    y : np.ndarray, shape (N,)
        Stage-2 class indices (values in ``range(len(FINDING_COLS))``).
        Samples with label ``-1`` are excluded by the caller before passing
        here.
    model_type : str
        ``"xgboost"`` or ``"lightgbm"``.
    n_splits : int
        Number of stratified folds for cross-validation reporting.

    Returns
    -------
    model, scaler, label_encoder
    """
    print(f"[stage2] Training {model_type.upper()} multi-class classifier …")
    print(f"[stage2] X shape: {X.shape}, classes present: {np.unique(y)}")

    # Encode labels to contiguous integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"[stage2] Number of classes: {n_classes}  ({[STAGE2_MERGED_NAMES[c] for c in le.classes_]})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "xgboost":
        model = _build_xgb(n_classes)
    elif model_type == "lightgbm":
        model = _build_lgbm(n_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Cross-validated accuracy report
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y_enc), dtype=np.int64)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_scaled, y_enc)):
        model.fit(X_scaled[tr_idx], y_enc[tr_idx])
        oof_preds[val_idx] = model.predict(X_scaled[val_idx])
        acc = accuracy_score(y_enc[val_idx], oof_preds[val_idx])
        print(f"[stage2]  fold {fold + 1}/{n_splits}  val accuracy: {acc:.4f}")

    oof_acc = accuracy_score(y_enc, oof_preds)
    print(f"[stage2] OOF accuracy: {oof_acc:.4f}")

    # Refit on all data
    model.fit(X_scaled, y_enc)

    # Persist
    with open(_STAGE2_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(_STAGE2_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(_STAGE2_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[stage2] Model saved to {_STAGE2_MODEL_PATH}")

    return model, scaler, le


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_stage2() -> tuple[object, StandardScaler, LabelEncoder]:
    """Load previously saved Stage-2 artefacts."""
    for p in (_STAGE2_MODEL_PATH, _STAGE2_SCALER_PATH, _STAGE2_ENCODER_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Stage-2 artefact not found: {p}")
    with open(_STAGE2_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(_STAGE2_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(_STAGE2_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, scaler, le


def predict_stage2(
    model,
    scaler: StandardScaler,
    le: LabelEncoder,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage-2 inference.

    Returns
    -------
    labels : np.ndarray of int
        Original class indices (into ``FINDING_COLS``).
    names : np.ndarray of str
        Human-readable finding names.
    """
    X_scaled = scaler.transform(X)
    y_enc = model.predict(X_scaled)
    labels = le.inverse_transform(y_enc)
    names = np.array([FINDING_COLS[i] for i in labels])
    return labels, names
