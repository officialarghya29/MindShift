"""Temporal emotion engine (PS-01 §18, §20): emotion trajectory + arc metrics."""
from __future__ import annotations

import numpy as np

# emotion → intensity in [0,1] (calm → hot); used as the arc's Y-axis
INTENSITY = {"joy": .35, "affection": .25, "excitement": .45, "relief": .18,
             "neutral": .10, "confusion": .28, "surprise": .35, "sadness": .40,
             "anxiety": .50, "fear": .55, "frustration": .62, "anger": .82,
             "disgust": .70}


def emotion_intensity(res: dict) -> float:
    return INTENSITY.get(res["emotion"]["label"], .1)


def build_arc(results: list[dict]) -> dict:
    """Emotional arc + intensity/direction/stability metrics (PS-01 §18)."""
    ys = np.array([emotion_intensity(r) for r in results])
    ts = np.array([r["tension"] for r in results])
    peaks_idx = _peaks(ys, prominence=0.18)
    drops_idx = _drops(ys, prominence=0.18)
    arc = [{
        "message_id": r["message_id"],
        "emotion": r["emotion"]["label"],
        "emotion_confidence": r["emotion"]["confidence"],
        "intensity": round(float(y), 3),
        "tension": r["tension"],
    } for r, y in zip(results, ys)]
    # stability: 1 - normalized variance
    stability = 1 - min(float(np.std(ys)) / 0.4, 1.0)
    recovery = _recovery(ys)
    return {
        "arc": arc,
        "mean_intensity": round(float(ys.mean()), 3),
        "peak_intensity": round(float(ys.max()), 3),
        "stability": round(stability, 3),
        "direction": ("de-escalating" if ys[-1] < ys[: max(1, len(ys)//3)].mean() - 0.08
                      else "escalating" if ys[-1] > ys[: max(1, len(ys)//3)].mean() + 0.08
                      else "stable"),
        "peaks": peaks_idx,
        "drops": drops_idx,
        "recovered": recovery,
        "tension_curve": [round(float(t), 1) for t in ts],
    }


def _peaks(ys: np.ndarray, prominence=0.15) -> list[int]:
    out = []
    for i in range(1, len(ys) - 1):
        if ys[i] - max(ys[i-1], ys[i+1]) >= -1e-9 and \
           (ys[i] - min(ys[max(0, i-3):i+1].min() if i >= 3 else ys.min(), ys[i+1:i+4].min() if i+4 <= len(ys) else ys.min()) >= prominence):
            out.append(i)
    # dedupe near-adjacent peaks keeping the highest
    dedup = []
    for i in out:
        if dedup and i - dedup[-1] <= 2:
            if ys[i] > ys[dedup[-1]]:
                dedup[-1] = i
        else:
            dedup.append(i)
    return dedup


def _drops(ys: np.ndarray, prominence=0.15) -> list[int]:
    return _peaks(-ys, prominence)


def _recovery(ys: np.ndarray) -> bool:
    if len(ys) < 6:
        return False
    hot = ys.max()
    if hot < 0.55:
        return False
    peak_i = int(np.argmax(ys))
    tail = ys[peak_i + 1:]
    return bool(len(tail) >= 2 and tail.mean() < hot - 0.25)
