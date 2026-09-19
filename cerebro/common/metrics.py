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


def calibration_metrics(correct, confidence, n_bins: int = 10,
                        strategy: str = "quantile") -> dict:
    """Top-1 calibration of a predictor's own confidence (PS-01 §28, §34).

    ``correct`` is a 0/1 array of whether the top-1 label was right and
    ``confidence`` is the probability the model assigned to that top-1 label.
    Returns ECE (expected calibration error), MCE (maximum calibration error) and
    the mean confidence, so a well-calibrated head sits at ECE ~ 0 and
    mean_confidence ~ accuracy.

    Quantile binning is the default because it keeps every bin populated when the
    confidence distribution is skewed — the fixed-width alternative produces
    empty bins whose "calibration" is meaningless.
    """
    correct = np.asarray(correct, float)
    confidence = np.asarray(confidence, float)
    n = len(correct)
    if n == 0:
        return {"ece": None, "mce": None, "mean_confidence": None, "n": 0}
    if strategy == "quantile":
        edges = np.unique(np.quantile(confidence, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    # digitize assigns every point to exactly one bin. A `lo < c <= hi` mask would
    # silently drop the points sitting exactly on the lowest edge (common when a
    # head predicts a repeated value), which would flatter the reported ECE.
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, len(edges) - 2)
    ece, mce = 0.0, 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        gap = abs(float(correct[sel].mean()) - float(confidence[sel].mean()))
        ece += gap * float(sel.mean())
        mce = max(mce, gap)
    return {"ece": round(float(ece), 4), "mce": round(float(mce), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
            "accuracy": round(float(correct.mean()), 4),
            "n_bins_used": int(len(edges) - 1), "n": int(n)}


def cohen_kappa(a, b) -> dict:
    """Cohen's κ between two annotators over paired labels (PS-01 §7)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return {"kappa": None, "n": 0}
    po = float((a == b).mean())
    cats = np.unique(np.concatenate([a, b]))
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    kappa = 0.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return {"observed_agreement": round(po, 4), "expected_agreement": round(float(pe), 4),
            "kappa": round(float(kappa), 4), "n": int(n)}


def krippendorff_alpha_nominal(a, b) -> dict:
    """Krippendorff's α (nominal) for two coders.

    With two coders the nominal-scale α reduces to Scott's π, i.e. it uses the
    *pooled* marginals rather than each coder's own (the difference from Cohen's
    κ). Both are reported because reviewers expect κ and α to agree closely when
    the coders are symmetric.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return {"alpha": None, "n": 0}
    po = float((a == b).mean())
    cats = np.unique(np.concatenate([a, b]))
    pooled = np.concatenate([a, b])
    pe = sum(float((pooled == c).mean()) ** 2 for c in cats)
    alpha = 0.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return {"alpha": round(float(alpha), 4), "observed_agreement": round(po, 4),
            "n": int(n)}


def probability_metrics(y_true, p_pred) -> dict:
    """Calibration + ranking quality for binary heads (PS-01 §37)."""
    y_true, p_pred = np.asarray(y_true, int), np.clip(np.asarray(p_pred, float), 1e-6, 1 - 1e-6)
    out = {"roc_auc": None, "pr_auc": None, "brier": round(float(brier_score_loss(y_true, p_pred)), 4)}
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, p_pred)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, p_pred)), 4)
    return out
