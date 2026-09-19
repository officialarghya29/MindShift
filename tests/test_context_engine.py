"""Context-window tests (PS-01 §8)."""
from cerebro.context.context_engine import ContextWindow


def test_context_window_ranks_older_by_tension():
    cw = ContextWindow(k=2)
    msgs = [{"message_id": i + 1, "text": f"m{i + 1}", "tension": t}
            for i, t in enumerate([1, 9, 3, 8, 4, 5, 6, 2])]
    # i=7 → recent verbatim = m6,m7; older pool = m1..m5, ranked by tension
    ctx = cw.context_text(msgs, 7)
    assert "m1" not in ctx and "m3" not in ctx   # lowest tensions, ranked out
    assert "m2" in ctx and "m4" in ctx and "m5" in ctx  # highest, ranked in


def test_context_window_without_tension_falls_back_to_verbatim():
    cw = ContextWindow(k=4)
    msgs = [{"message_id": i + 1, "text": f"m{i + 1}"} for i in range(3)]
    assert cw.context_text(msgs, 3) == "m1 || m2 || m3"