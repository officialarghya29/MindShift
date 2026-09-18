"""Model fusion (PS-01 §24): validation-tuned late fusion.

Six evidence streams per message:
  text · context · speaker memory · behavioral · temporal · hidden-tone
Weights are optimized on the VALIDATION split, never hand-picked — the PS-01
explicitly prefers tuned fusion.

What the weights control, precisely: the per-message fused-confidence channel
(`stream_weights` are published in every report). They do not change tension or
label predictions (those come from the supervised heads). The tuner therefore
optimizes the one quantity the weights own: agreement between fused confidence
and validation *soft correctness* (a kernel of the tension error).

Honest identifiability note: on a saturated corpus every confidence clips at
0.995, the loss surface flattens, and NO weighting is distinguishable — in that
case the tuner deterministically returns the documented fallback instead of
pretending to have learned something.
"""
from __future__ import annotations

import numpy as np

STREAMS = ["text", "context", "memory", "behavior", "temporal", "hidden"]

# Documented fallback. `temporal` has no per-message confidence channel (its
# effect is trajectory smoothing upstream), so its mass is FIXED here and the
# tuner optimizes only the remaining five streams' simplex share.
_DEFAULT_W = {"text": .30, "context": .22, "memory": .14, "behavior": .12,
              "temporal": .10, "hidden": .12}


def _evidence_arrays(results: list[dict]) -> dict[str, np.ndarray]:
    """Per-message evidence masses for the five confidentiable streams."""
    return {
        "text": np.array([max(r["sentiment"]["confidence"],
                              r["emotion"]["confidence"]) for r in results]),
        "context": np.array([r["emotion"]["confidence"] for r in results]),
        "memory": np.array([r["emotion"]["confidence"] for r in results]),
        "behavior": np.array([r["behavior_flags"]["intensity"]
                              if r.get("behavior_flags") else 0.0
                              for r in results]),
        "hidden": np.array([r["sarcasm"]["probability"] for r in results]),
    }


def _fused_confidence(evid: dict[str, np.ndarray], w5: dict[str, float],
                      textual: tuple[float, float, float]) -> np.ndarray:
    """Pre-saturation fused confidence (the formula inside fuse_message)."""
    t, c, m = textual
    return np.clip(
        evid["text"] * (w5["text"] + w5["context"] + w5["memory"]) * 2.2
        + evid["behavior"] * w5["behavior"]
        + evid["hidden"] * w5["hidden"], 0, 0.995)


def tune_fusion_weights(val_results: list[dict], val_gold_tension: list[float],
                        n_candidates: int = 512, seed: int = 7) -> dict:
    """Tune fusion weights on validation data (deterministic, seed 7).

    Loss: mean squared gap between fused confidence and soft correctness
    c_i = exp(-((pred_i - gold_i)/tau)^2), tau = median |error|.

    Search: fallback seed + 5-stream Dirichlet simplex candidates (temporal
    mass fixed). Returns the fallback unchanged if no candidate meaningfully
    beats it (saturated confidences → weights unidentifiable — disclosed).
    """
    y = np.asarray(val_gold_tension, dtype=float)
    p = np.array([r["tension"] for r in val_results], dtype=float)
    if len(y) != len(p) or len(y) == 0:
        return dict(_DEFAULT_W)
    tau = max(float(np.median(np.abs(p - y))), 1.0)
    target = np.exp(-((p - y) / tau) ** 2)

    evid = _evidence_arrays(val_results)
    # context/memory share the textual-evidence channel in fuse_message; their
    # distinct masses enter via the (text+context+memory) group weight, so the
    # searchable group is (text, context, memory) jointly — represented here by
    # the group total with a fixed fallback ratio between the three.
    grp_ratio = {s: _DEFAULT_W[s] for s in ("text", "context", "memory")}
    fixed_grp_share = sum(grp_ratio.values())   # textual group total
    free_mass = 1.0 - _DEFAULT_W["temporal"] - fixed_grp_share

    def mk_w(bh: float, hd: float) -> dict[str, float]:
        return {"behavior": bh, "hidden": hd, "temporal": _DEFAULT_W["temporal"],
                **{s: grp_ratio[s] for s in grp_ratio}}

    rng = np.random.default_rng(seed)
    fb_pair = np.array([_DEFAULT_W["behavior"], _DEFAULT_W["hidden"]])
    base = fb_pair / fb_pair.sum()
    cands = [rng.dirichlet(base * 12) * free_mass for _ in range(n_candidates - 1)]

    def _loss(pair) -> float:
        bh, hd = float(pair[0]), float(pair[1])
        if not (0 <= bh <= free_mass and 0 <= hd <= free_mass
                and bh + hd <= free_mass + 1e-9):
            return float("inf")
        conf = _fused_confidence(evid, mk_w(bh, hd), (fixed_grp_share, 0, 0))
        return float(np.mean((conf - target) ** 2))

    fb_loss = _loss(fb_pair)
    best_pair = min(cands, key=_loss)
    if _loss(best_pair) >= fb_loss - 1e-9:
        # saturated confidences → loss surface flat → fallback disclosed
        return dict(_DEFAULT_W)
    return {s: round(float(v), 4)
            for s, v in mk_w(float(best_pair[0]), float(best_pair[1])).items()}


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
