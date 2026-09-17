"""Public-dataset adapters → unified CEREBRO conversation format (PS-01 §3).

Adapters are *lossy label mappers*: they import native labels verbatim where a
CEREBRO label exists and fall back to documented heuristics otherwise. They do
NOT re-annotate text. All functions accept iterables (file handles, lists of
lines, streamed records) so nothing needs to be loaded wholesale.

Sources and licenses (verify against the official distribution before any
production/redistribution use):
  • GoEmotions (Google Research, Reddit) — per-message emotion labels; the
    authors distribute it for research use. No conversation structure:
    each message becomes a 1-turn conversation.
  • SARC (Reddit self-referential sarcasm corpus) — binary sarcasm labels on
    comments; author field (when present) groups comments into conversations.
  • DailyDialog — multi-turn dialogues with per-utterance emotion and act
    labels; research/non-commercial license.

All outputs validate against `validate_conversation`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

# --------------------------------------------------------------------------
# unified schema
# --------------------------------------------------------------------------
SCHEMA_FIELDS = ("conversation_id", "message_id", "speaker_id", "timestamp",
                 "text", "sentiment", "emotion", "tone", "sarcasm", "irony",
                 "passive_aggression", "tension")

SENTIMENTS = ("positive", "neutral", "negative")
EMOTIONS = ("joy", "affection", "excitement", "relief", "neutral", "surprise",
            "confusion", "frustration", "anxiety", "fear", "disgust",
            "sadness", "anger")

# emotion → (sentiment, tension heuristic)
_EMOTION_VALENCE = {
    "joy": ("positive", 8), "affection": ("positive", 8),
    "excitement": ("positive", 14), "relief": ("positive", 8),
    "neutral": ("neutral", 12), "surprise": ("neutral", 22),
    "confusion": ("neutral", 26), "frustration": ("negative", 45),
    "anxiety": ("negative", 48), "fear": ("negative", 52),
    "disgust": ("negative", 50), "sadness": ("negative", 42),
    "anger": ("negative", 58),
}


def validate_cerebro_record(rec: dict) -> None:
    missing = [f for f in SCHEMA_FIELDS if f not in rec]
    if missing:
        raise ValueError(f"record missing fields {missing}: {rec}")
    if rec["sentiment"] not in SENTIMENTS:
        raise ValueError(f"bad sentiment {rec['sentiment']!r}")
    if rec["emotion"] not in EMOTIONS:
        raise ValueError(f"bad emotion {rec['emotion']!r}")
    for f in ("sarcasm", "irony", "passive_aggression"):
        if rec[f] not in (0, 1, True, False):
            raise ValueError(f"{f} must be 0/1, got {rec[f]!r}")
    if not (0.0 <= float(rec["tension"]) <= 100.0):
        raise ValueError(f"tension out of [0,100]: {rec['tension']}")
    if not str(rec["text"]).strip():
        raise ValueError("empty text")


def validate_conversation(convs: Iterable[list[dict]]) -> list[list[dict]]:
    """Validate and return as list; also enforces conversation uniqueness
    of message ids and 1..n message ids."""
    out = []
    for conv in convs:
        ids = [m["message_id"] for m in conv]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate message_id in conversation: {ids}")
        if ids != sorted(ids):
            raise ValueError(f"message_ids not ordered: {ids}")
        for m in conv:
            validate_cerebro_record(m)
        out.append(conv)
    return out


def _mk(conv_id: str, mid: int, speaker: str, text: str, emotion: str,
        tone: str, sarcasm: int, irony: int, pa: int) -> dict:
    # sentiment = surface valence of the emotion (PS-01 §11: sentiment ≠
    # sarcasm — the hidden twist lives in the sarcasm/irony labels)
    sentiment, tension = _EMOTION_VALENCE[emotion]
    tension = float(min(tension + sarcasm * 15 + pa * 12, 100))
    return {
        "conversation_id": conv_id, "message_id": mid, "speaker_id": speaker,
        "timestamp": None, "text": text, "sentiment": sentiment,
        "emotion": emotion, "tone": tone, "sarcasm": int(sarcasm),
        "irony": int(irony), "passive_aggression": int(pa),
        "tension": tension,
    }


# --------------------------------------------------------------------------
# GoEmotions (Reddit) — JSONL: {"text": ..., "labels": [ids...]}, one line/msg
# --------------------------------------------------------------------------
GOEMOTIONS_ID2LABEL = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness", 20: "optimism",
    21: "realization", 22: "relief", 23: "remorse", 24: "sadness",
    25: "surprise", 26: "neutral",
}
# native → CEREBRO 13-emotion set, priority order for multi-label collapse
_GO_NATIVE2CEB = [
    ("joy", "joy"), ("amusement", "joy"), ("gratitude", "affection"),
    ("love", "affection"), ("caring", "affection"), ("admiration", "affection"),
    ("optimism", "excitement"), ("excitement", "excitement"),
    ("desire", "excitement"), ("relief", "relief"),
    ("anger", "anger"), ("annoyance", "frustration"),
    ("disappointment", "frustration"), ("disapproval", "frustration"),
    ("nervousness", "anxiety"), ("embarrassment", "anxiety"),
    ("fear", "fear"), ("grief", "sadness"), ("sadness", "sadness"),
    ("remorse", "sadness"), ("disgust", "disgust"),
    ("surprise", "surprise"), ("confusion", "confusion"),
    ("realization", "surprise"), ("approval", "neutral"),
    ("curiosity", "neutral"), ("neutral", "neutral"),
]
_CE_LEX_POS = ("good", "great", "thanks", "love", "nice", "awesome", "best")
_CE_LEX_NEG = ("bad", "hate", "awful", "terrible", "worst", "stupid", "wrong")


def goemotions_records(lines: Iterable[str], source_id: str = "goe") -> list[list[dict]]:
    """JSONL lines → one 1-turn conversation per message."""
    convs = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        labels = [GOEMOTIONS_ID2LABEL.get(int(l)) for l in rec.get("labels", [])]
        labels = [l for l in labels if l]
        emotion = next((ceb for native, ceb in _GO_NATIVE2CEB if native in labels),
                       "neutral")
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        convs.append([_mk(f"{source_id}_{i}", 1, "anon", text, emotion,
                          "casual", 0, 0, 0)])
    return convs


# --------------------------------------------------------------------------
# SARC (Reddit sarcasm) — JSONL: {"label": 0/1, "comment": ..., "author": ...}
# --------------------------------------------------------------------------
def sarc_records(lines: Iterable[str], source_id: str = "sarc") -> list[list[dict]]:
    """JSONL lines → conversations grouped by author (falls back to one
    1-turn conversation per message when no author is given)."""
    convs: dict[str, list[dict]] = {}
    order: list[str] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = str(rec.get("comment", "")).strip()
        if not text:
            continue
        low = text.lower()
        pos = sum(w in low for w in _CE_LEX_POS)
        neg = sum(w in low for w in _CE_LEX_NEG)
        if neg > pos:
            emotion = "frustration"
        elif pos > neg:
            emotion = "joy"
        else:
            emotion = "neutral"
        tone = "sarcastic" if int(rec.get("label", 0)) == 1 else "casual"
        author = str(rec.get("author", f"anon{i}"))
        key = f"{source_id}_{author}"
        if key not in convs:
            convs[key] = []
            order.append(key)
        convs[key].append(_mk(key, len(convs[key]) + 1, author, text, emotion,
                              tone, int(rec.get("label", 0)), 0, 0))
    return [convs[k] for k in order]


# --------------------------------------------------------------------------
# DailyDialog — dialogue/emotion/act files, utterances split by __eou__
# --------------------------------------------------------------------------
_DD_EMOTION = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
               4: "joy", 5: "sadness", 6: "surprise"}
_DD_EMOTION2CEB = {"anger": "anger", "disgust": "disgust", "fear": "fear",
                   "joy": "joy", "sadness": "sadness", "surprise": "surprise",
                   "neutral": "neutral"}


def dailydialog_conversations(utterance_lines: Iterable[str],
                              emotion_lines: Iterable[str],
                              act_lines: Iterable[str] = (),
                              source_id: str = "dd") -> list[list[dict]]:
    """DailyDialog → conversations. Speaker alternates A/B per turn;
    per-utterance emotion mapped natively; tone derived from emotion valence
    (documented heuristic, DailyDialog has no tone labels)."""
    acts = list(act_lines)
    convs = []
    for d, (uline, eline) in enumerate(zip(utterance_lines, emotion_lines)):
        utterances = [u.strip() for u in uline.split("__eou__") if u.strip()]
        emotions = [int(e) for e in eline.split()]
        conv = []
        for t, (text, eidx) in enumerate(zip(utterances, emotions)):
            emotion = _DD_EMOTION2CEB[_DD_EMOTION.get(eidx, 0)]
            tone = {"positive": "friendly", "negative": "frustrated",
                    "neutral": "casual"}[_EMOTION_VALENCE[emotion][0]]
            conv.append(_mk(f"{source_id}_d{d}", t + 1,
                            "A" if t % 2 == 0 else "B", text, emotion, tone,
                            0, 0, 0))
        if conv:
            convs.append(conv)
    return convs
