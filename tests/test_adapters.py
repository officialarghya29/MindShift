"""Public-dataset adapter tests: format conversion + schema validation."""
import sys
sys.path.insert(0, ".")

import pytest

from cerebro.data.adapters import (
    SCHEMA_FIELDS, SENTIMENTS, EMOTIONS,
    goemotions_records, sarc_records, dailydialog_conversations,
    validate_conversation, validate_cerebro_record,
)


# ---------------------------------------------------------------- GoEmotions
GOE_LINES = [
    '{"text": "I love this so much!", "labels": [18, 15]}',          # love, gratitude
    '{"text": "This is absolutely unacceptable.", "labels": [2, 10]}',  # anger, disapproval
    '{"text": "Wait, what? How does that even work?", "labels": [6]}',   # confusion
    '{"text": "", "labels": [0]}',                                    # dropped (empty)
    'not json at all',                                                # raises
]


def test_goemotions_basic():
    convs = goemotions_records(l for l in GOE_LINES if l != "not json at all")
    assert len(convs) == 3                          # empty text dropped
    rec = convs[0][0]
    assert rec["emotion"] == "affection"            # love > gratitude priority
    assert rec["sentiment"] == "positive"
    assert all(f in rec for f in SCHEMA_FIELDS)
    assert convs[1][0]["emotion"] == "anger"
    assert convs[2][0]["emotion"] == "confusion"


def test_goemotions_bad_json_raises():
    with pytest.raises(json.JSONDecodeError):
        goemotions_records(["not json at all"])


# ---------------------------------------------------------------------- SARC
SARC_LINES = [
    '{"label": 1, "comment": "Oh yeah, GREAT idea, genius.", "author": "alice"}',
    '{"label": 0, "comment": "Thanks, that helps a lot.", "author": "alice"}',
    '{"label": 0, "comment": "Sure, whatever you say.", "author": "bob"}',
    '{"label": 1, "comment": "", "author": "bob"}',   # empty dropped
]


def test_sarc_groups_by_author():
    convs = sarc_records(SARC_LINES)
    assert len(convs) == 2                          # alice + bob
    alice = convs[0]
    assert len(alice) == 2
    assert alice[0]["sarcasm"] == 1
    assert alice[0]["tone"] == "sarcastic"
    assert alice[1]["sarcasm"] == 0
    assert [m["message_id"] for m in alice] == [1, 2]
    # one-turn fallback for missing authors
    solo = sarc_records(['{"label": 0, "comment": "hi there"}'])
    assert len(solo) == 1 and len(solo[0]) == 1


# ---------------------------------------------------------------- DailyDialog
DD_UTT = ["hi , how are you doing ?__eou__ i 'm fine , thanks .__eou__\n",
          "what time is it ?__eou__ about three .__eou__\n"]
DD_EMO = ["4 0\n", "0 0\n"]                          # joy, neutral / neutral, neutral
DD_ACT = ["1 2\n", "1 3\n"]


def test_dailydialog_roundtrip():
    convs = dailydialog_conversations(DD_UTT, DD_EMO, DD_ACT)
    assert len(convs) == 2
    c0 = convs[0]
    assert [m["speaker_id"] for m in c0] == ["A", "B"]
    assert c0[0]["emotion"] == "joy"
    assert c0[1]["emotion"] == "neutral"
    assert c0[1]["tone"] == "casual"


# ------------------------------------------------------------- schema guards
def test_validator_accepts_good_records():
    convs = goemotions_records(l for l in GOE_LINES if l != "not json at all")
    out = validate_conversation(convs)
    assert len(out) == 3


def test_validator_rejects_bad_tension():
    bad = {"conversation_id": "x", "message_id": 1, "speaker_id": "A",
           "timestamp": None, "text": "hi", "sentiment": "neutral",
           "emotion": "neutral", "tone": "casual", "sarcasm": 0, "irony": 0,
           "passive_aggression": 0, "tension": 250}
    with pytest.raises(ValueError):
        validate_cerebro_record(bad)


def test_validator_rejects_bad_emotion():
    bad = {"conversation_id": "x", "message_id": 1, "speaker_id": "A",
           "timestamp": None, "text": "hi", "sentiment": "neutral",
           "emotion": "ecstatic", "tone": "casual", "sarcasm": 0, "irony": 0,
           "passive_aggression": 0, "tension": 10}
    with pytest.raises(ValueError):
        validate_cerebro_record(bad)


def test_validator_rejects_duplicate_ids():
    dup = [[{"conversation_id": "x", "message_id": 1, "speaker_id": "A",
             "timestamp": None, "text": "a", "sentiment": "neutral",
             "emotion": "neutral", "tone": "casual", "sarcasm": 0,
             "irony": 0, "passive_aggression": 0, "tension": 10}]] * 1
    dup[0].append(dict(dup[0][0], text="b"))
    with pytest.raises(ValueError):
        validate_conversation(dup)


import json  # noqa: E402  (used in test_goemotions_bad_json_raises)
