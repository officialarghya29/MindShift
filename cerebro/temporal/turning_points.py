"""Turning-point detector (PS-01 §21).

A turning point = statistically meaningful shift in the conversation's
emotional state. Model-estimated, never claimed as causal ground truth.
"""
from __future__ import annotations

import numpy as np

from cerebro.temporal.arc import INTENSITY


def detect_turning_points(results: list[dict], z_threshold: float = 1.6,
                          min_delta_tension: float = 15.0) -> list[dict]:
    """Detect significant state shifts via robust z-score on tension deltas."""
    if len(results) < 5:
        return []
    ts = np.array([r["tension"] for r in results])
    deltas = np.diff(ts)
    med = np.median(deltas)
    mad = np.median(np.abs(deltas - med))
    scale = 1.4826 * mad
    if scale < 1e-3:                     # degenerate (piecewise-constant) series
        scale = max(float(deltas.std()), 1.0)
    robust_z = (deltas - med) / scale
    out = []
    for i in np.argsort(-np.abs(robust_z))[:len(deltas)]:
        if abs(deltas[i]) < 1e-9:        # no actual change → never a turning point
            continue
        if abs(robust_z[i]) < z_threshold and abs(deltas[i]) < min_delta_tension:
            continue
        i = int(i) + 1  # message AFTER which the shift is observed
        if any(tp["message_id"] == i for tp in out):
            continue
        prev, cur = results[i-1], results[min(i, len(results)-1)]
        before_state = {
            "emotion": prev["emotion"]["label"],
            "tension": prev["tension"],
            "tone": prev["tone"]["label"],
        }
        after_state = {
            "emotion": cur["emotion"]["label"],
            "tension": cur["tension"],
            "tone": cur["tone"]["label"],
        }
        out.append({
            "message_id": cur["message_id"],
            "before": before_state,
            "after": after_state,
            "tension_change": round(float(cur["tension"] - prev["tension"]), 1),
            "robust_z": round(float(robust_z[i-1]), 2),
            "trigger_text": cur["text"][:140],
            "speaker": cur["speaker_id"],
            "note": "model-estimated turning point, not a causal claim",
        })
        if len(out) >= 5:
            break
    return sorted(out, key=lambda t: t["message_id"])
