"""Temporal + explainability engine tests on synthetic per-message states."""
import sys
sys.path.insert(0, ".")

from cerebro.temporal.escalation import classify_trajectory, escalation_flags
from cerebro.temporal.transitions import track_transitions
from cerebro.temporal.turning_points import detect_turning_points
from cerebro.explain.explanation_engine import what_changed, speaker_profiles, build_digest


def _mk(n=12, escalate=True):
    """Synthetic per-message states. Escalating = calm plateau then a sharp
    discontinuity at message 5 (a genuine turning point)."""
    msgs = []
    for i in range(n):
        t = (18 if i < 4 else 55 + (i - 4) * 6) if escalate else 18
        msgs.append({
            "message_id": i + 1, "speaker_id": "A" if i % 2 == 0 else "B",
            "text": f"message {i}",
            "sentiment": {"label": "negative" if t > 50 else "neutral",
                          "confidence": .8, "probabilities": {}},
            "emotion": {"label": "anger" if t > 60 else "frustration" if t > 40 else "neutral",
                        "confidence": .8, "probabilities": {}},
            "tone": {"label": "aggressive" if t > 60 else "frustrated" if t > 40 else "casual",
                     "confidence": .8, "probabilities": {}},
            "tension": float(min(t, 98)),
            "sarcasm": {"probability": .1, "confidence": .2, "supporting_signals": []},
            "irony": {"probability": .1, "confidence": .2, "supporting_signals": []},
            "passive_aggression": {"probability": .2, "confidence": .2, "supporting_signals": []},
            "signals": {"n_words": 4, "pos_hits": [], "neg_hits": [], "sarc_words": [],
                        "irony_words": [], "pa_phrase": None, "exaggeration": [],
                        "interjections": [], "praise_minus_neg": 0, "exclam": 0,
                        "ellipsis": 0, "emoji_sarc": [], "emoji_polarity": 0.0,
                        "laugh": False, "swear": 0, "quoted_echo": False, "is_short": False},
            "context_text": "ctx", "speaker_state_before": {},
        })
    return msgs


def test_escalation_trajectory_detected():
    es = classify_trajectory(_mk(escalate=True))
    assert es["trajectory"] == "escalating"
    assert es["peak_message"] == 12
    es2 = classify_trajectory(_mk(escalate=False))
    assert es2["trajectory"] == "stable"


def test_escalation_start_found():
    es = classify_trajectory(_mk(escalate=True))
    assert es["escalation_start_message"] is not None
    assert 3 <= es["escalation_start_message"] <= 6


def test_escalation_flags_align():
    flags = escalation_flags(_mk(escalate=True))
    assert flags[0]["phase"] == "neutral"
    assert any(f["phase"] == "peak" for f in flags)


def test_transitions_captured():
    tr = track_transitions(_mk())
    assert tr, "expected at least one transition"
    t0 = tr[0]
    assert t0["from_emotion"] != t0["to_emotion"]
    assert "tension_delta" in t0


def test_turning_points_found_at_discontinuity():
    tp = detect_turning_points(_mk(escalate=True))
    assert tp, "expected turning points at the tension jump"
    ids = [t["message_id"] for t in tp]
    assert 4 <= min(ids) <= 6
    for t in tp:
        assert "before" in t and "after" in t and "tension_change" in t
        assert t["note"].startswith("model-estimated")


def test_turning_points_absent_on_plateau():
    assert detect_turning_points(_mk(escalate=False)) == []


def test_what_changed_panel():
    msgs = _mk()
    wc = what_changed(msgs, 6)
    assert wc["before"]["tension"] < wc["after"]["tension"]
    assert "tension_delta" in wc["change"]


def test_speaker_profiles():
    prof = speaker_profiles(_mk())
    assert set(prof.keys()) == {"A", "B"}
    a = prof["A"]
    assert a["messages"] == 6
    assert 0 <= a["avg_tension"] <= 100
    assert "not a psychological profile" in a["note"]


_ESC_KEYS = {"trajectory", "escalation_start_message", "escalation_rate",
             "peak_tension", "peak_message", "deescalation_point",
             "phase_means", "confidence"}


def test_escalation_schema_complete_for_every_length():
    """Short/degenerate conversations must emit the SAME escalation schema as
    full ones — report consumers rely on a stable key set (regression for the
    '<4 messages' shape mismatch)."""
    for n in (0, 1, 3):
        es = classify_trajectory(_mk(n))
        assert _ESC_KEYS <= set(es), f"missing escalation keys for n={n}"
        assert es["trajectory"] == "stable" and es["note"]
        assert es["peak_message"] is None or isinstance(es["peak_message"], int)
    full = classify_trajectory(_mk(12, escalate=False))
    assert _ESC_KEYS <= set(full) and full["trajectory"] == "stable"
    assert full["note"] is None


def test_escalation_flags_short_conversation():
    flags = escalation_flags(_mk(2))
    assert len(flags) == 2 and flags[0]["phase"] in ("neutral", "peak")


def test_analyze_empty_report_full_escalation_schema():
    from cerebro.models.pipeline import CerebroPipeline
    rep = CerebroPipeline(engine=None).analyze([])
    assert "escalation_rate" in rep["escalation"]
    assert _ESC_KEYS <= set(rep["escalation"])
    assert rep["escalation"]["trajectory"] == "stable"
    assert rep["topics"] == []


def _report(msgs):
    from cerebro.temporal.arc import build_arc
    from cerebro.temporal.transitions import track_transitions
    from cerebro.explain.explanation_engine import speaker_profiles, explain_message
    return {
        "summary": {"conversation_id": "conv_x", "n_messages": len(msgs)},
        "messages": msgs,
        "emotional_arc": build_arc(msgs),
        "emotion_transitions": track_transitions(msgs),
        "turning_points": detect_turning_points(msgs),
        "escalation": classify_trajectory(msgs),
        "escalation_phases": escalation_flags(msgs),
        "speaker_profiles": speaker_profiles(msgs),
        "explanations": [explain_message(r, None) for r in msgs],
    }


def test_digest_aggregates_report():
    d = build_digest(_report(_mk(escalate=True)))
    assert d["conversation_id"] == "conv_x" and d["n_messages"] == 12
    assert d["headline_finding"] is not None
    assert d["headline_finding"]["direction"] == "tension spike"
    assert d["trajectory"]["trajectory"] == "escalating"
    assert d["text"]["overview"] and "turning point" in d["text"]["overview"]
    assert d["turning_points"][0]["from_to"]


def test_digest_empty_and_short_reports():
    from cerebro.models.pipeline import CerebroPipeline
    d_empty = build_digest(CerebroPipeline(engine=None).analyze([]))
    assert d_empty["n_messages"] == 0 and d_empty["headline_finding"] is None
    assert d_empty["trajectory"]["trajectory"] == "stable"
    d_short = build_digest(_report(_mk(2)))
    assert d_short["n_messages"] == 2 and d_short["trajectory"]["trajectory"] == "stable"
    assert d_short["speakers"]
    for s in d_short["speakers"]:
        assert "avg_tension" in s and "high_tension_share" in s and "note" not in s
