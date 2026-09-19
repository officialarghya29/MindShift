"""Feature-engine tests: behavioral vector, micro signals, segmentation."""
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


def test_sarcasm_markers_are_sense_disambiguated():
    """Regression: plain certainty/agreement must not count as irony evidence.

    "sure" is a dismissive marker in "Sure." but ordinary certainty in "I was
    sure ..."; matching by token alone produced false positives on sincere
    messages. Found by error analysis, fixed in SARC_SENSE_BLOCKERS.
    """
    from cerebro.features.preprocess import micro_signals
    assert "sure" not in micro_signals("I was sure the deadline was next month.")["sarc_words"]
    assert "sure" not in micro_signals("Make sure the file is attached.")["sarc_words"]
    assert "right" not in micro_signals("That's right, well spotted.")["sarc_words"]
    # the ironic senses must survive the blocker
    assert "sure" in micro_signals("Sure.")["sarc_words"]
    assert "right" in micro_signals("Right, and I'm the villain of course.")["sarc_words"]


def test_belief_update_markers_damp_contradiction_but_not_true_sarcasm():
    from cerebro.features.preprocess import micro_signals
    from cerebro.models.hidden_signals import sarcasm_score
    sig = micro_signals("Wait, you finished the whole thing already? That's amazing!")
    assert sig["belief_update"] is True
    sincere = sarcasm_score(0.05, {"probabilities": {"positive": .9, "negative": .01}},
                            "it crashed again and we are late", sig, 80.0, 84.0)
    plain = micro_signals("That's amazing, congratulations on the launch!")
    no_update = sarcasm_score(0.05, {"probabilities": {"positive": .9, "negative": .01}},
                              "it crashed again and we are late", plain, 80.0, 84.0)
    # belief revision lowers the score relative to the same wording without it
    assert sincere["probability"] < no_update["probability"]
    # a real echoic barb carries no belief-update marker and stays flagged
    barb = micro_signals("Oh, we're doing this again tonight? Wonderful.")
    assert barb["belief_update"] is False
    assert sarcasm_score(0.05, {"probabilities": {"positive": .9, "negative": .01}},
                         "it crashed again and we are late", barb, 80.0, 86.0)["probability"] > 0.5
