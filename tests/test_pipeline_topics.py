"""Pipeline report-consistency tests: topics, transition matrix, digest.

Uses a tiny real MultiTaskEngine (no big model, no network) so the full
analyze() path is exercised rather than a stub.
"""
import sys
sys.path.insert(0, ".")

from cerebro.models.engines import MultiTaskEngine
from cerebro.models.pipeline import CerebroPipeline
from cerebro.explain.explanation_engine import build_digest

_EMO = ["joy", "frustration", "anger", "relief", "neutral", "anxiety"]
_TONE = ["casual", "frustrated", "aggressive", "calm", "neutral", "anxious"]


def _mk_texts():
    """Two conversations with an emotion shift in the middle."""
    a = ["hey how are you", "good thanks you", "great to hear",
         "did you finish the report", "not yet sorry",
         "this is ridiculous", "fine i will do it myself"]
    b = ["meeting moved to tomorrow", "works for me", "see you then"]
    return a, b


def _train():
    texts = sum(_mk_texts(), [])
    convs = []
    for group in _mk_texts():
        convs.append([
            {"text": t, "message_id": i + 1, "timestamp": None,
             "speaker_id": "A" if i % 2 else "B",
             "emotion": _EMO[i % len(_EMO)], "tone": _TONE[i % len(_TONE)],
             "sentiment": "negative" if i % 4 == 1 else "positive",
             "tension": float(20 + 10 * i), "sarcasm": i % 5 == 0,
             "irony": 0, "passive_aggression": i % 6 == 0}
            for i, t in enumerate(group)])
    eng = MultiTaskEngine(seed=42)
    eng.fit(convs, use_context=False, use_memory=False, use_behavior=True)
    return eng


def test_pipeline_report_topics_are_lists():
    """Every report shape (even degenerate) carries a list, not None."""
    empty = CerebroPipeline(engine=None).analyze([])
    assert empty["topics"] == []

    pipe = CerebroPipeline.from_trained(_train())
    msgs = [{"text": t, "speaker_id": "A" if i % 2 else "B"}
            for i, t in enumerate(_mk_texts()[0])]
    rep = pipe.analyze(msgs)
    assert isinstance(rep["topics"], list) and rep["topics"]
    for seg in rep["topics"]:
        for k in ("segment_id", "start", "end", "method", "messages_analyzed"):
            assert k in seg, f"topic segment missing {k}"


def test_pipeline_transition_matrix_and_digest():
    pipe = CerebroPipeline.from_trained(_train())
    msgs = [{"text": t, "speaker_id": "A" if i % 2 else "B"}
            for i, t in enumerate(_mk_texts()[0])]
    rep = pipe.analyze(msgs)
    assert rep["transition_matrix"], "expected emotion transitions"
    d = build_digest(rep)
    assert d["conversation_id"] == rep["summary"]["conversation_id"]
    assert d["n_messages"] == len(msgs)
    assert isinstance(d["turning_points"], list)
    for s in d["speakers"]:
        assert "avg_tension" in s and "high_tension_share" in s