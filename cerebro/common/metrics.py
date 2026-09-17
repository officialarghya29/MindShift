"""Metric helpers — single implementation used by baselines, ablations and the final report."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)


def classification_metrics(y_true, y_pred, labels=None, average="macro") -> dict:
    """Accuracy / precision / recall / macro-F1 / weighted-F1 in one call."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        f"f1_{average}": round(float(f1_score(y_true, y_pred, labels=labels,
                                              average=average, zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "r2": round(float(1 - (err ** 2).sum() /
                          ((y_true - y_true.mean()) ** 2).sum()), 4),
    }


def probability_metrics(y_true, p_pred) -> dict:
    """Calibration + ranking quality for binary heads (PS-01 §37)."""
    y_true, p_pred = np.asarray(y_true, int), np.clip(np.asarray(p_pred, float), 1e-6, 1 - 1e-6)
    out = {"roc_auc": None, "pr_auc": None, "brier": round(float(brier_score_loss(y_true, p_pred)), 4)}
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, p_pred)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, p_pred)), 4)
    return out
