"""Feature-engine tests: behavioral vector, micro signals, segmentation."""
import numpy as np

from cerebro.features.preprocess import (behavioral_vector, process_text,
                                         micro_signals, BEHAVIORAL_DIMS)
from cerebro.features.segmentation import segment_conversation


def test_behavioral_vector_dims_and_values():
    v = behavioral_vector("WHY did this happen AGAIN!!!", None, None, None, "A")
    assert len(v) == BEHAVIORAL_DIMS
    assert v[2] == 3            # exclamation count
    assert v[5] > 0             # caps ratio > 0 ("AGAIN", "WHY")
    assert v[7] >= 1            # "again" is in the negative lexicon


def test_process_text_preserves_meaning_masks_urls():
    out = process_text("check https://x.com/a?b=1 now!!!")
    assert "https" not in out and "link" in out
    assert "!!" in out and "!!!" not in out


def test_micro_signals_sarcasm_markers():
    s = micro_signals("Oh great, PERFECT timing 🙄")
    assert s["sarc_words"] or s["irony_words"]
    assert s["emoji_sarc"] == ["🙄"]


def test_micro_signals_pa_phrases():
    s2 = micro_signals("Fine. Whatever.")
    # "fine." is the classic concessive opener; both are in the lexicon
    assert s2["pa_phrase"] in ("fine.", "whatever.")


def test_response_gap_computed():
    v = behavioral_vector("hey", "2025-01-01T10:00:00", "2025-01-01T12:00:00", "A", "B")
    assert abs(v[15] - 2.0) < 1e-6            # 2-hour gap in hours


def test_segmentation_time_gap():
    msgs = [{"message_id": 1, "text": "morning plan talk", "timestamp": "2025-01-01T09:00:00"},
            {"message_id": 2, "text": "yes let us start", "timestamp": "2025-01-01T09:05:00"},
            {"message_id": 3, "text": "completely new subject now", "timestamp": "2025-01-01T13:00:00"}]
    segs = segment_conversation(msgs)
    assert len(segs) == 2 and segs[1]["start"] == 3


def test_segmentation_cohesion_short_returns_single():
    msgs = [{"message_id": i + 1, "text": f"msg {i}", "timestamp": None} for i in range(4)]
    assert len(segment_conversation(msgs)) == 1
