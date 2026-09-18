"""CEREBRO analysis pipeline — orchestrates the full stack (PS-01 §46).

    engine heads → hidden-signal fusion → temporal engines → model fusion
    → explainability artifacts (WHY / WHAT CHANGED / speaker profiles)
"""
from __future__ import annotations

import numpy as np

from cerebro.models.engines import MultiTaskEngine
from cerebro.models.hidden_signals import apply_hidden_signals
from cerebro.fusion.fusion import fuse_conversation, _DEFAULT_W
from cerebro.temporal.arc import build_arc
from cerebro.temporal.transitions import track_transitions, transition_matrix
from cerebro.temporal.turning_points import detect_turning_points
from cerebro.temporal.escalation import classify_trajectory, escalation_flags
from cerebro.explain.explanation_engine import (
    explain_message, what_changed, speaker_profiles, behavior_flags)


def _add_behavior_flags(results):
    for r in results:
        r["behavior_flags"] = behavior_flags(np.array(_vec_from_signals(r), dtype=float))
    return results


def _vec_from_signals(r) -> list[float]:
    """Reconstruct the behavioral vector fields needed by behavior_flags."""
    s = r["signals"]
    return [s["n_words"], len(r["text"]), s["exclam"], 0, 0, 0,
            len(s["pos_hits"]), len(s["neg_hits"]), len(s["exaggeration"]),
            s["emoji_polarity"], len(s["emoji_sarc"]), int(s["laugh"]),
            s["ellipsis"], s["quoted_echo"] + s["swear"], int(s["is_short"]), 0]


class CerebroPipeline:
    def __init__(self, engine: MultiTaskEngine, fusion_weights: dict | None = None):
        self.engine = engine
        self.fusion_weights = fusion_weights or _DEFAULT_W

    @classmethod
    def from_trained(cls, engine: MultiTaskEngine, val_convs=None,
                     fusion_weights: dict | None = None):
        """Build the deployment pipeline.

        Fusion-weight resolution order (PS-01 §24 — tuned over hand-picked):
          1. explicit `fusion_weights` argument
          2. weights tuned on `val_convs` via real leave-one-stream-out
          3. weights persisted inside the engine (tuned at fit time)
          4. documented fallback
        """
        if fusion_weights is None and val_convs:
            from cerebro.fusion.fusion import tune_fusion_weights
            from cerebro.models.pipeline import _add_behavior_flags as _abf
            probe = cls(engine, None)
            all_fused, all_gold = [], []
            for conv in val_convs[:10]:          # pool validation conversations
                msgs = [dict(m) for m in conv]
                res = probe.engine.predict_conversation(msgs)
                apply_hidden_signals(res)
                _abf(res)
                all_fused.extend(fuse_conversation(res, None))
                all_gold.extend(g["tension"] for g in conv)
            fusion_weights = tune_fusion_weights(all_fused, all_gold)
        return cls(engine, fusion_weights or engine.fusion_weights)

    def analyze(self, messages: list[dict], conversation_id="conv_uploaded") -> dict:
        """Full analysis → dashboard-ready report dict (PS-01 §30)."""
        if not messages:                      # degenerate-input guard (fuzz-tested)
            return {
                "summary": {"conversation_id": conversation_id, "n_messages": 0,
                            "n_speakers": 0, "dominant_emotion": [],
                            "sentiment_distribution": {}, "tone_distribution": {},
                            "sarcasm_level": 0.0, "irony_level": 0.0,
                            "passive_aggression_level": 0.0, "mean_tension": 0.0,
                            "peak_tension": 0.0, "trajectory": "stable"},
                "messages": [], "emotional_arc": None, "emotion_transitions": [],
                "transition_matrix": None, "turning_points": [],
                "escalation": {"trajectory": "stable"}, "escalation_phases": [],
                "speaker_profiles": [], "explanations": [], "topics": None,
                "disclaimer": ("All outputs are model-estimated with calibrated confidence; "
                               "turning points are associations, not causal claims."),
            }
        for i, m in enumerate(messages):
            m.setdefault("message_id", i + 1)
            m.setdefault("conversation_id", conversation_id)
            m.setdefault("speaker_id", m.get("speaker", f"speaker_{i % 2 + 1}"))
            m.setdefault("timestamp", None)
            m.setdefault("platform", "generic")

        results = self.engine.predict_conversation(messages)
        apply_hidden_signals(results)
        _add_behavior_flags(results)
        results = fuse_conversation(results, self.fusion_weights)

        arc = build_arc(results)
        transitions = track_transitions(results)
        tmat = transition_matrix(results)
        turning = detect_turning_points(results)
        escalation = classify_trajectory(results)
        phases = escalation_flags(results)
        profiles = speaker_profiles(results)
        explanations = [explain_message(r, results[i-1] if i else None)
                        for i, r in enumerate(results)]

        # summary-level aggregates (report §30, items 1-9)
        from collections import Counter
        n = len(results)
        summary = {
            "conversation_id": conversation_id,
            "n_messages": n,
            "n_speakers": len({r["speaker_id"] for r in results}),
            "dominant_emotion": Counter(r["emotion"]["label"] for r in results).most_common(3),
            "sentiment_distribution": dict(Counter(r["sentiment"]["label"] for r in results)),
            "tone_distribution": dict(Counter(r["tone"]["label"] for r in results).most_common(6)),
            "sarcasm_level": round(float(np.mean([r["sarcasm"]["probability"] for r in results])), 3),
            "irony_level": round(float(np.mean([r["irony"]["probability"] for r in results])), 3),
            "passive_aggression_level": round(float(np.mean([r["passive_aggression"]["probability"] for r in results])), 3),
            "mean_tension": round(float(np.mean([r["tension"] for r in results])), 1),
            "peak_tension": round(float(np.max([r["tension"] for r in results])), 1),
            "trajectory": escalation["trajectory"],
        }

        return {
            "summary": summary,
            "messages": results,
            "emotional_arc": arc,
            "emotion_transitions": transitions,
            "transition_matrix": tmat,
            "turning_points": turning,
            "escalation": escalation,
            "escalation_phases": phases,
            "speaker_profiles": profiles,
            "explanations": explanations,
            "topics": None,  # filled by caller via segmentation if desired
            "disclaimer": ("All outputs are model-estimated with calibrated confidence; "
                           "turning points are associations, not causal claims."),
        }

    def why(self, report: dict, message_id: int) -> dict:
        ex = next((e for e in report["explanations"]
                   if e["message_id"] == message_id), None)
        return ex or {"error": f"no explanation for message {message_id}"}

    def what_changed(self, report: dict, message_id: int) -> dict:
        return what_changed(report["messages"], message_id)
