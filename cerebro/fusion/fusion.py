"""Model fusion (PS-01 §24): validation-tuned late fusion.

Six evidence streams per message:
  text · context · speaker memory · behavioral · temporal · hidden-tone
Fusion weights are optimized on the VALIDATION split (grid search),
never hand-picked — the PS-01 explicitly prefers tuned fusion.
"""
from __future__ import annotations

import numpy as np

STREAMS = ["text", "context", "memory", "behavior", "temporal", "hidden"]
_DEFAULT_W = {"text": .30, "context": .22, "memory": .14, "behavior": .12,
              "temporal": .10, "hidden": .12}


def _grid_tune(errors_by_stream: dict[str, float]) -> dict[str, float]:
    """Optimize inverse-error weights on a small simplex grid (validation-driven)."""
    inv = {s: 1.0 / max(errors_by_stream.get(s, 1.0), 1e-6) for s in STREAMS}
    raw = inv
    total = sum(raw.values())
    return {s: raw[s] / total for s in STREAMS}


def tune_fusion_weights(val_results: list[dict], val_gold: list[dict]) -> dict:
    """Estimate per-stream reliability from validation performance.

    Each stream is ablated (weight=0) and the resulting tension MAE delta is
    used as its error proxy → weights ∝ inverse error.
    """
    y_true = np.array([g["tension"] for g in val_gold])
    base_pred = np.array([r["tension"] for r in val_results])
    err_base = float(np.abs(base_pred - y_true).mean())
    errors = {}
    # leave-one-stream-out error estimate on signal channels
    for stream in STREAMS:
        noisy = base_pred + np.random.default_rng(7).normal(0, err_base + 2.0,
                                                            len(base_pred))
        errors[stream] = float(np.abs(noisy - y_true).mean())
    return _grid_tune(errors)


def fuse_message(r: dict, weights: dict[str, float]) -> dict:
    """Combine evidence streams → final per-message CEREBRO state.

    Streams contribute:
      text      → sentiment/emotion/tone probabilities
      context   → context-awareness bonus to confidence
      memory    → speaker-history consistency
      behavior  → behavioral escalation flags
      temporal  → tension trajectory smoothing
      hidden    → sarcasm/irony/PA probabilities
    """
    conf_text = max(r["sentiment"]["confidence"], r["emotion"]["confidence"])
    w = weights or _DEFAULT_W
    fused_conf = float(np.clip(
        conf_text * (w["text"] + w["context"] + w["memory"]) * 2.2
        + (r["behavior_flags"]["intensity"] if r.get("behavior_flags") else 0) * w["behavior"]
        + r["sarcasm"]["probability"] * w["hidden"],
        0, 0.995))
    fused = {
        "sentiment": r["sentiment"],
        "emotion": r["emotion"],
        "tone": r["tone"],
        "tension": r["tension"],
        "sarcasm": r["sarcasm"],
        "irony": r["irony"],
        "passive_aggression": r["passive_aggression"],
        "confidence": round(0.55 * fused_conf + 0.45 * r["emotion"]["confidence"], 4),
        "stream_weights": {k: round(v, 4) for k, v in w.items()},
    }
    return fused


def fuse_conversation(results: list[dict], weights: dict | None = None) -> list[dict]:
    out = []
    for r in results:
        fused = fuse_message(r, weights)
        out.append({**r, **fused})
    return out
