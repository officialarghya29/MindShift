"""Conversation segmentation (PS-01 §23).

Two levels:
  (a) time-gap splits from real timestamps;
  (b) lexical-cohesion segmentation (TextTiling-style similarity dips) for
      conversations without reliable timestamps.
"""
from __future__ import annotations

import re
from datetime import datetime

from cerebro.features.lexicons import SLANG_MAP

GAP_MINUTES = 90          # (a): a silence > 90 min starts a new segment
SIMILARITY_DROP = 0.28    # (b): cosine drop vs. previous window starts a segment
WINDOW = 5                # words per side in the cohesion dip test


def expand_slang(tokens: list[str]) -> list[str]:
    out = []
    for w in tokens:
        out.extend(SLANG_MAP.get(w, w).split())
    return out


def _tokens(text: str) -> list[str]:
    return expand_slang(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def segment_by_time(messages: list[dict]) -> list[list[dict]]:
    segs, cur = [], []
    last_ts = None
    for m in messages:
        ts = _parse_ts(m.get("timestamp"))
        if ts and last_ts and (ts - last_ts).total_seconds() > GAP_MINUTES * 60:
            segs.append(cur)
            cur = []
        cur.append(m)
        if ts:
            last_ts = ts
    if cur:
        segs.append(cur)
    return segs


def segment_by_cohesion(messages: list[dict]) -> list[list[dict]]:
    """Sliding cosine similarity between adjacent word windows; a sharp dip
    marks a topic boundary (TextTiling, Hearst 1997)."""
    if len(messages) < 6:
        return [messages]
    toks = [" ".join(_tokens(m["text"])) for m in messages]
    bounds = []
    for i in range(1, len(messages)):
        a = set(" ".join(toks[max(0, i - WINDOW):i]).split())
        b = set(" ".join(toks[i:i + WINDOW]).split())
        if not a or not b:
            continue
        inter = len(a & b)
        cos = inter / ((len(a) * len(b)) ** 0.5)
        prev = bounds[-1][1] if bounds else None
        bounds.append((i, cos))
    cuts = []
    for j, (i, cos) in enumerate(bounds):
        left = bounds[max(0, j - 2):j] + bounds[j + 1:j + 3]
        if len(left) < 2:
            continue
        avg = sum(c for _, c in left) / len(left)
        if avg - cos > SIMILARITY_DROP:
            if not cuts or i - cuts[-1] >= 3:
                cuts.append(i)
    segs = []
    start = 0
    for c in cuts:
        segs.append(messages[start:c])
        start = c
    segs.append(messages[start:])
    return [s for s in segs if s]


def segment_conversation(messages: list[dict]) -> list[dict]:
    """Unified entry: returns [{segment_id, start, end, messages, method}]."""
    segs = (segment_by_time(messages) if messages and messages[0].get("timestamp")
            else segment_by_cohesion(messages))
    out = []
    for sid, seg in enumerate(segs):
        out.append({"segment_id": sid, "start": seg[0]["message_id"],
                    "end": seg[-1]["message_id"], "method":
                    "time-gap" if messages[0].get("timestamp") else "cohesion",
                    "messages": seg})
    return out


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
