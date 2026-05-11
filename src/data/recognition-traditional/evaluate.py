"""Evaluation utilities for Stage-1, Stage-2, and the end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FINDING_COLS, REPORT_DIR


# ---------------------------------------------------------------------------
# Stage-1 evaluation
# ---------------------------------------------------------------------------

def evaluate_stage1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    tag: str = "stage1",
) -> dict:
    """Compute and log Stage-1 binary classification metrics.

    Parameters
    ----------
    y_true : binary ground-truth labels.
    y_pred : binary predicted labels (after threshold).
    proba  : positive-class probabilities (optional; enables ROC-AUC).
    tag    : label for the saved report file.

    Returns a metrics dict.
    """
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))

    report_str = classification_report(y_true, y_pred, target_names=["No Finding", "Disease"])

    print(f"\n=== {tag.upper()} Evaluation ===")
    print(report_str)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save
    _save_report(tag, metrics, report_str)
    return metrics


# ---------------------------------------------------------------------------
# Stage-2 evaluation
# ---------------------------------------------------------------------------

def evaluate_stage2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    tag: str = "stage2",
) -> dict:
    """Compute and log Stage-2 multi-class classification metrics.

    Parameters
    ----------
    y_true : ground-truth class indices (into FINDING_COLS).
    y_pred : predicted class indices.
    class_names : optional list of class name strings.
    tag : label for the saved report file.
    """
    if class_names is None:
        all_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        class_names = [FINDING_COLS[c] if 0 <= c < len(FINDING_COLS) else str(c)
                       for c in all_classes]

    kappa = float(cohen_kappa_score(y_true, y_pred))
    acc   = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report_str = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "kappa": kappa,
        "macro_f1": macro_f1,
    }

    print(f"\n=== {tag.upper()} Evaluation ===")
    print(report_str)
    print(f"  Cohen Kappa : {kappa:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")

    _save_report(tag, metrics, report_str)
    _save_confusion_matrix(y_true, y_pred, class_names, tag)
    return metrics


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    y_true_s1: np.ndarray,
    y_pred_s1: np.ndarray,
    y_true_s2: np.ndarray | None,
    y_pred_s2: np.ndarray | None,
    proba_s1: np.ndarray | None = None,
) -> dict:
    """Combined end-to-end evaluation report."""
    s1_metrics = evaluate_stage1(y_true_s1, y_pred_s1, proba_s1, tag="pipeline_stage1")
    s2_metrics: dict = {}
    if y_true_s2 is not None and y_pred_s2 is not None and len(y_true_s2) > 0:
        s2_metrics = evaluate_stage2(y_true_s2, y_pred_s2, tag="pipeline_stage2")
    return {"stage1": s1_metrics, "stage2": s2_metrics}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_report(tag: str, metrics: dict, report_str: str) -> None:
    out_json = REPORT_DIR / f"{tag}_metrics.json"
    out_txt  = REPORT_DIR / f"{tag}_report.txt"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_txt, "w") as f:
        f.write(report_str)
    print(f"  → Report saved to {out_txt}")


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    tag: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"Confusion Matrix – {tag}")
        plt.tight_layout()
        out = REPORT_DIR / f"{tag}_confusion_matrix.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  → Confusion matrix saved to {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [evaluate] Could not save confusion matrix: {exc}")
