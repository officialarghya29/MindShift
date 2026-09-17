"""Escalation engine (PS-01 §22): tension trajectory classification + phases."""
from __future__ import annotations

import numpy as np


def classify_trajectory(results: list[dict]) -> dict:
    ts = np.array([r["tension"] for r in results], dtype=float)
    n = len(ts)
    if n < 4:
        return {"trajectory": "stable", "note": "conversation too short for phase analysis"}
    thirds = (ts[:n//3], ts[n//3: 2*n//3], ts[2*n//3:])
    start_mean, mid_mean, end_mean = (float(t.mean()) for t in thirds)
    slope = float(np.polyfit(np.arange(n), ts, 1)[0])
    peak_i = int(np.argmax(ts))
    drop_after_peak = float(ts[peak_i:].min() if n - peak_i > 2 else ts[peak_i])

    if end_mean - start_mean > 12 and slope > 0.35:
        traj = "escalating"
    elif start_mean - end_mean > 12 and slope < -0.35:
        traj = "de-escalating"
    elif ts.max() - ts.min() > 40:
        traj = "volatile"
    else:
        traj = "stable"

    escalation_start = None
    running = []
    for i, t in enumerate(ts):
        running.append(t)
        if len(running) >= 4 and np.all(np.diff(running[-4:]) > 0) and t > start_mean + 8:
            escalation_start = results[i - 3]["message_id"]
            break

    return {
        "trajectory": traj,
        "escalation_start_message": escalation_start,
        "escalation_rate": round(float(slope), 3),
        "peak_tension": round(float(ts.max()), 1),
        "peak_message": results[peak_i]["message_id"],
        "deescalation_point": (results[peak_i:][int(np.argmin(ts[peak_i:]))]["message_id"]
                               if drop_after_peak < ts.max() - 20 and n - peak_i > 2 else None),
        "phase_means": {"start": round(start_mean, 1), "middle": round(mid_mean, 1),
                        "end": round(end_mean, 1)},
        "confidence": round(min(0.99, abs(slope) / 2.5 + (ts.max()-ts.min())/200), 3),
    }


def escalation_flags(results: list[dict]) -> list[dict]:
    """Per-message escalation phase labels for the dashboard timeline."""
    out = []
    traj = classify_trajectory(results)
    start = traj.get("escalation_start_message")
    peak = traj.get("peak_message")
    for r in results:
        phase = "neutral"
        if start and r["message_id"] >= start:
            phase = "escalating"
        if peak and r["message_id"] == peak:
            phase = "peak"
        elif peak and start and start < r["message_id"] < peak:
            phase = "escalating"
        elif peak and r["message_id"] > peak and traj["trajectory"] in ("de-escalating", "volatile"):
            phase = "de-escalating"
        out.append({"message_id": r["message_id"], "phase": phase})
    return out
