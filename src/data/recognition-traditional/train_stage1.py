"""Stage-1 binary classifier: disease vs. no-finding.

Pipeline
--------
1. StandardScaler  →  PCA  →  SVM (RBF) or RandomForest
2. GridSearchCV with Recall-oriented scoring.
3. Decision threshold tuning on a held-out validation fold.
4. Persist the trained pipeline + threshold to MODEL_DIR.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    MODEL_DIR,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
    STAGE1_DECISION_THRESHOLD,
    STAGE1_MODEL,
)

_STAGE1_MODEL_PATH = MODEL_DIR / "stage1_pipeline.pkl"
_STAGE1_THRESHOLD_PATH = MODEL_DIR / "stage1_threshold.pkl"


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _build_svm_pipeline() -> tuple[Pipeline, dict[str, Any]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=RANDOM_SEED)),
    ])
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
    }
    return pipe, param_grid


def _build_rf_pipeline() -> tuple[Pipeline, dict[str, Any]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    param_grid = {
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 5],
    }
    return pipe, param_grid


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_stage1(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = STAGE1_MODEL,
    n_splits: int = 5,
) -> tuple[Pipeline, float]:
    """Train the Stage-1 binary classifier.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Binary labels: 1 = disease, 0 = no-finding.
    model_type : str
        ``"svm"`` or ``"rf"``.
    n_splits : int
        Number of CV folds for GridSearchCV.

    Returns
    -------
    best_pipeline : sklearn Pipeline
        Fitted pipeline (scaler → PCA → classifier).
    threshold : float
        Decision-probability threshold tuned for high recall.
    """
    print(f"[stage1] Training {model_type.upper()} classifier …")
    print(f"[stage1] X shape: {X.shape}, positives: {y.sum()}, negatives: {(y == 0).sum()}")

    if model_type == "svm":
        pipe, param_grid = _build_svm_pipeline()
    elif model_type == "rf":
        pipe, param_grid = _build_rf_pipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X, y)
    best = grid.best_estimator_
    print(f"[stage1] Best params: {grid.best_params_}")
    print(f"[stage1] Best CV recall: {grid.best_score_:.4f}")

    # ------------------------------------------------------------------
    # Threshold tuning on OOF predictions from the best fold
    # ------------------------------------------------------------------
    proba = grid.best_estimator_.predict_proba(X)[:, 1]
    threshold = _tune_threshold(y, proba)
    print(f"[stage1] Tuned decision threshold: {threshold:.3f}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    with open(_STAGE1_MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    with open(_STAGE1_THRESHOLD_PATH, "wb") as f:
        pickle.dump(threshold, f)
    print(f"[stage1] Model saved to {_STAGE1_MODEL_PATH}")

    return best, threshold


def _tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Choose the probability threshold that maximises F1 while keeping
    recall >= 0.90 on the training data.  Falls back to the configured
    default if the constraint cannot be satisfied.
    """
    thresholds = np.linspace(0.1, 0.7, 61)
    best_thresh = STAGE1_DECISION_THRESHOLD
    best_f1 = 0.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        if rec >= 0.90 and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_stage1() -> tuple[Pipeline, float]:
    """Load a previously saved Stage-1 pipeline and threshold."""
    if not _STAGE1_MODEL_PATH.exists():
        raise FileNotFoundError(f"Stage-1 model not found: {_STAGE1_MODEL_PATH}")
    with open(_STAGE1_MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    with open(_STAGE1_THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)
    return pipe, threshold


def predict_stage1(
    pipe: Pipeline,
    X: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage-1 inference.

    Returns
    -------
    labels : np.ndarray of int
        0 = negative, 1 = positive (disease suspected).
    proba : np.ndarray of float
        Probability of the positive class.
    """
    if threshold is None:
        threshold = STAGE1_DECISION_THRESHOLD
    proba = pipe.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return labels, proba
