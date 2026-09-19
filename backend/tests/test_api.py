"""API tests with a stubbed pipeline (no trained model required)."""
import io
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import main  # noqa: E402


class _StubPipeline:
    def analyze(self, messages, conversation_id="conv_stub"):
        from cerebro.explain.explanation_engine import explain_message
        results = []
        for i, m in enumerate(messages):
            r = {
                "message_id": m["message_id"], "speaker_id": m["speaker_id"],
                "text": m["text"],
                "sentiment": {"label": "neutral", "confidence": .6,
                              "probabilities": {"neutral": .6}},
                "emotion": {"label": "neutral", "confidence": .7,
                            "probabilities": {"neutral": .7}},
                "tone": {"label": "casual", "confidence": .5,
                         "probabilities": {"casual": .5}},
                "tension": 20.0,
                "sarcasm": {"probability": .1, "confidence": .2, "supporting_signals": []},
                "irony": {"probability": .1, "confidence": .2, "supporting_signals": []},
                "passive_aggression": {"probability": .2, "confidence": .2,
                                       "supporting_signals": []},
                "signals": {"n_words": 3, "pos_hits": [], "neg_hits": [],
                            "sarc_words": [], "irony_words": [], "pa_phrase": None,
                            "exaggeration": [], "interjections": [], "praise_minus_neg": 0,
                            "exclam": 0, "ellipsis": 0, "emoji_sarc": [],
                            "emoji_polarity": 0.0, "laugh": False, "swear": 0,
                            "quoted_echo": False, "is_short": False},
                "context_text": "", "speaker_state_before": {},
            }
            results.append(r)
        from cerebro.models.pipeline import _add_behavior_flags
        _add_behavior_flags(results)
        from cerebro.fusion.fusion import fuse_conversation
        results = fuse_conversation(results)
        from cerebro.temporal.arc import build_arc
        from cerebro.temporal.transitions import track_transitions
        from cerebro.temporal.turning_points import detect_turning_points
        from cerebro.temporal.escalation import classify_trajectory, escalation_flags
        from cerebro.explain.explanation_engine import speaker_profiles

        prev = None
        explanations = []
        for r in results:
            explanations.append(explain_message(r, prev))
            prev = r
        return {
            "summary": {"conversation_id": conversation_id,
                        "n_messages": len(results),
                        "dominant_emotion": [("neutral", len(results))],
                        "trajectory": "stable",
                        "mean_tension": 20.0, "peak_tension": 20.0},
            "messages": results,
            "emotional_arc": build_arc(results),
            "emotion_transitions": track_transitions(results),
            "transition_matrix": {},
            "turning_points": detect_turning_points(results),
            "escalation": classify_trajectory(results),
            "escalation_phases": escalation_flags(results),
            "speaker_profiles": speaker_profiles(results),
            "explanations": explanations,
            "topics": None,
            "disclaimer": "stub",
        }


def _client():
    main._STATE["pipeline"] = _StubPipeline()
    return TestClient(main.app)


def test_upload_parses_generic():
    c = _client()
    r = c.post("/upload", files={"file": ("chat.txt", io.BytesIO(b"Aarav: hi\nMeera: hello"), "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_messages"] == 2 and body["platform_detected"] == "generic"


def test_analyze_and_full_flow():
    c = _client()
    r = c.post("/analyze", files={"file": ("chat.txt", io.BytesIO(b"A: fine.\nB: ok then"), "text/plain")})
    assert r.status_code == 200
    conv = r.json()["summary"]["conversation_id"]
    # stored conversation retrievable
    assert c.get(f"/conversation/{conv}").status_code == 200
    assert c.get(f"/conversation/{conv}/timeline").status_code == 200
    tp = c.get(f"/conversation/{conv}/turning-points").json()
    assert "turning_points" in tp
    assert c.get(f"/conversation/{conv}/speakers").status_code == 200
    why = c.get(f"/why/{conv}/1")
    assert why.status_code == 200 and "prediction" in why.json()
    wc = c.get(f"/what-changed/{conv}/2")
    assert wc.status_code == 200 and "change" in wc.json()
    assert c.get("/healthz").json()["model_loaded"] is True


def test_404_for_unknown_conversation():
    c = _client()
    assert c.get("/conversation/does_not_exist").status_code == 404
    assert c.get("/conversation/does_not_exist/digest").status_code == 404


def test_short_conversation_escalation_schema_and_digest():
    c = _client()
    r = c.post("/analyze", files={"file": ("chat.txt", io.BytesIO(b"A: fine.\nB: ok then\n"), "text/plain")})
    body = r.json()
    esc = body["escalation"]
    for k in ("escalation_start_message", "escalation_rate", "peak_tension",
              "peak_message", "deescalation_point", "phase_means", "confidence"):
        assert k in esc, f"escalation missing key {k} for short conversation"
    assert esc["trajectory"] == "stable"
    conv = body["summary"]["conversation_id"]
    d = c.get(f"/conversation/{conv}/digest").json()
    assert d["conversation_id"] == conv and d["n_messages"] == 2
    assert d["trajectory"]["trajectory"] == "stable"
    assert d["headline_finding"] is None
    assert d["speakers"] and d["text"]["overview"]


def test_upload_rejects_garbage():
    c = _client()
    r = c.post("/upload", files={"file": ("x.txt", io.BytesIO(b""), "text/plain")})
    assert r.status_code == 422


def test_pdf_report_endpoint():
    """PDF export returns a valid PDF document (PS-01 §30 P2 item)."""
    c = _client()
    r = c.post("/analyze", files={
        "file": ("t.txt", b"Aarav: Hey!\nMeera: Fine. Do what you want then.\n")})
    cid = r.json()["summary"]["conversation_id"]
    p = c.get(f"/conversation/{cid}/report.pdf")
    assert p.status_code == 200
    assert p.headers["content-type"] == "application/pdf"
    assert p.content[:4] == b"%PDF"
