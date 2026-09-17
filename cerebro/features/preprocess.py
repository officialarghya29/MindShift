"""Preprocessing + behavioral feature extraction (PS-01 §5, §17).

Preserves raw_text; produces processed_text plus a dense interpretable vector.
"""
from __future__ import annotations

import re

from cerebro.features.lexicons import (
    POS_LEX, NEG_LEX, SARC_MARKERS, IRONY_MARKERS, PA_PHRASES,
    EXAG_WORDS, EMOJI_SENTIMENT, EMOJI_SARC_HINT, LAUGHTER_RE, ELLIPSIS, SWEAR_RE,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\U0000FE0F\U0000200D]+")
REPEAT_PUNCT_RE = re.compile(r"([!?])\1{1,}")
CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
SWEAR_RE = SWEAR_RE

BEHAVIORAL_DIMS = 16  # documented dimensionality of the behavioral vector


def emoji_split(text: str) -> tuple[str, list[str]]:
    emojis = EMOJI_RE.findall(text)
    return EMOJI_RE.sub(" ", text), emojis


def behavioral_vector(text: str, prev_ts, ts, prev_speaker: str | None,
                      speaker: str) -> list[float]:
    """16-dim interpretable behavioral vector (PS-01 §17). Order is fixed and
    documented — the fusion layer and explanations depend on this ordering."""
    base, emojis = emoji_split(text)
    words = base.split()
    n_words = max(len(words), 1)
    excl = text.count("!")
    ques = text.count("?")
    rep_punct = len(REPEAT_PUNCT_RE.findall(text))
    caps_ratio = sum(len(w) for w in CAPS_WORD_RE.findall(text)) / max(len(text), 1)
    pos_hits = sum(1 for w in words if w.lower().strip(".,!?") in POS_LEX)
    neg_hits = sum(1 for w in words if w.lower().strip(".,!?") in NEG_LEX)
    exag = sum(1 for w in words if w.lower().strip(".,!?") in EXAG_WORDS)
    emoji_polarity = sum(EMOJI_SENTIMENT.get(e, 0.0) for e in emojis)
    emoji_sarc = sum(1 for e in emojis if e in EMOJI_SARC_HINT)
    laugh = 1.0 if LAUGHTER_RE.search(text) else 0.0
    ellipsis = len(ELLIPSIS.findall(text))
    quotes = text.count('"') % 2  # unbalanced quote → echo/mocking use
    swear = len(SWEAR_RE.findall(text))
    short = 1.0 if len(words) <= 2 else 0.0
    if prev_ts and ts:
        try:
            from datetime import datetime
            if isinstance(prev_ts, str):
                t0 = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
            else:
                t0 = prev_ts
            t1 = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                  if isinstance(ts, str) else ts)
            gap = min(max((t1 - t0).total_seconds(), 0), 86400) / 3600.0
        except (ValueError, TypeError):
            gap = 0.0
    else:
        gap = 0.0
    return [len(words), len(text), excl, ques, rep_punct, caps_ratio,
            pos_hits, neg_hits, exag, emoji_polarity, emoji_sarc,
            laugh, ellipsis, quotes + swear, short, gap]


def process_text(text: str) -> str:
    """Normalized view: URL masking, slang expansion, whitespace cleanup.
    Raw text is never modified (PS-01 §5: 'preserve both')."""
    t = URL_RE.sub(" link ", text)
    t = re.sub(r"([!?])\1+", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def micro_signals(text: str) -> dict:
    """Sparse evidence signals for the hidden-tone engines (§14–16) and the
    explainability layer (§26). Evidence only — never a verdict."""
    base, emojis = emoji_split(text)
    low = base.lower().strip()
    words = [w.strip(".,!?\"'") for w in low.split()]
    pos_hits = [w for w in words if w in POS_LEX]
    neg_hits = [w for w in words if w in NEG_LEX]
    sarc_words = [w for w in words if w in SARC_MARKERS]
    irony_words = [w for w in words if w in IRONY_MARKERS]
    pa_phrase = next((p for p in PA_PHRASES if low.startswith(p)), None)
    exag = [w for w in words if w in EXAG_WORDS]
    praise_minus_context = len(pos_hits) - len(neg_hits)
    return {
        "n_words": len(words),
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "sarc_words": sarc_words,
        "irony_words": irony_words,
        "pa_phrase": pa_phrase,
        "exaggeration": exag,
        "interjections": [w for w in ("wow", "oh", "ah") if w in words],
        "praise_minus_neg": praise_minus_context,
        "exclam": text.count("!"),
        "ellipsis": len(ELLIPSIS.findall(text)),
        "emoji_sarc": [e for e in emojis if e in EMOJI_SARC_HINT],
        "emoji_polarity": sum(EMOJI_SENTIMENT.get(e, 0.0) for e in emojis),
        "laugh": bool(LAUGHTER_RE.search(text)),
        "swear": len(SWEAR_RE.findall(text)),
        "quoted_echo": '"' in base,
        "is_short": len(words) <= 2,
    }
