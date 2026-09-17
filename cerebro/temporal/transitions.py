"""Emotion transition engine (PS-01 §19): transitions, magnitudes, matrix."""
from __future__ import annotations

import numpy as np

from cerebro.temporal.arc import INTENSITY


def track_transitions(results: list[dict], tension_delta: float = 12.0,
                      emotion_change_bonus: float = 0.22) -> list[dict]:
    """Every significant emotion change: prev → new, magnitude, trigger message."""
    transitions = []
    for i in range(1, len(results)):
        prev, cur = results[i-1], results[i]
        e0, e1 = prev["emotion"]["label"], cur["emotion"]["label"]
        if e0 == e1:
            continue
        d_tension = cur["tension"] - prev["tension"]
        magnitude = abs(INTENSITY.get(e1, .1) - INTENSITY.get(e0, .1))
        significance = (magnitude > emotion_change_bonus or
                        abs(d_tension) > tension_delta)
        if not significance:
            continue
        transitions.append({
            "at_message": cur["message_id"],
            "speaker": cur["speaker_id"],
            "from_emotion": e0,
            "to_emotion": e1,
            "tension_before": prev["tension"],
            "tension_after": cur["tension"],
            "tension_delta": round(d_tension, 1),
            "magnitude": round(magnitude, 3),
            "trigger_text": cur["text"][:120],
            "confidence": round(float(cur["emotion"]["confidence"]), 4),
        })
    return transitions


def transition_matrix(results: list[dict]) -> dict:
    """Empirical transition counts between emotion classes."""
    from collections import Counter
    counts = Counter()
    for i in range(1, len(results)):
        counts[(results[i-1]["emotion"]["label"], results[i]["emotion"]["label"])] += 1
    return {f"{a} -> {b}": n for (a, b), n in counts.most_common()}
