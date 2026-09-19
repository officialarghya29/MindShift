"""Unified CEREBRO dataset construction (PS-01 §3, §4).

Synthetic-but-annotated conversational corpus: 6 domains × 7 arc patterns,
weak-supervision labels applied at generation time, conversation-level
train/val/test splits (no context leakage).

Every message carries the full blueprint §6 unified schema:
  conversation_id, message_id, speaker_id, timestamp, text, sentiment,
  emotion, tone, sarcasm, irony, passive_aggression, tension,
  escalation (derived: tension ≥ ESCALATION_TENSION_THRESHOLD), topic

Run:  python -m cerebro.data.generator
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from cerebro.common.labels import ESCALATION_TENSION_THRESHOLD
from cerebro.data.domains import DOMAINS

ARC_PATTERNS = ["calm", "positive", "friction", "escalation", "sarcasm",
                "passive", "mixed"]
SPLITS = {"train": 0.70, "val": 0.15, "whole conversations": 1.0}

TENSION_NOISE = 3.5      # gaussian noise on tension (realistic measurement)
LABEL_NOISE_RATE = 0.015 # small controlled label noise (annotator disagreement)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pick(rng, seq):
    return rng.choice(seq)


def _surface(rng, domain, i):
    topic = _pick(rng, domain["seed_topics"])
    tpl = _pick(rng, domain["surface"])
    s = tpl.format(topic=topic)
    if i % 3 == 0:
        return f"quick one {s}"
    if i % 3 == 1:
        return f"we should talk {s}"
    return s


def _arc_sequence(arc: str, n_turns: int, rng) -> list[str]:
    """Return per-turn band names following one of 7 narrative arcs."""
    seq = []
    if arc == "calm":
        seq = ["calm"] * n_turns
    elif arc == "positive":
        seq = ["calm"] * n_turns
    elif arc == "friction":
        seq = (["calm"] * (n_turns // 2) + ["strained"] * (n_turns - n_turns // 2))
    halfway = n_turns // 2
    if arc == "escalation":
        seq = (["calm"] * halfway + ["strained"] * max(1, n_turns // 4) +
               ["hot"] * max(1, n_turns // 4))
    if arc == "sarcasm":
        seq = (["calm"] * (n_turns // 3) + ["hot"] * max(1, n_turns // 3) +
               ["calm"] * (n_turns - n_turns // 3 - max(1, n_turns // 3)))
    if arc == "passive":
        seq = (["calm"] * (n_turns // 2) + ["strained"] * (n_turns - n_turns // 2))
    if arc == "mixed":
        bands = ["calm", "strained", "hot", "calm"]
        seq = [bands[i % 4] for i in range(n_turns)]
    return seq or ["calm"] * n_turns


def _deescalate(seq):
    """Append de-escalation tail to escalation arcs (PS-01 §22)."""
    n = max(2, len(seq) // 5)
    return seq + ["strained"] * min(2, n) + ["calm"] * n


def _make_conversation(dom_name, domain, arc, conv_id, rng, n_turns=None) -> list[dict]:
    calm, strained, hot = domain["bands"]
    n = n_turns or rng.randint(10, 26)
    seq = _arc_sequence(arc, n, rng)
    if arc == "escalation":
        seq = _deescalate(seq)
        n = len(seq)
    a, b = domain["cast"]
    speakers = [a, b] if rng.random() < 0.5 else [b, a]
    ts = datetime(2025, rng.randint(1, 12), rng.randint(1, 28), 9, 0) \
        + timedelta(minutes=rng.randint(0, 720))
    convo = []
    used = {"calm": [], "strained": [], "hot": []}
    for i, band in enumerate(seq):
        pool = {"calm": calm, "strained": strained, "hot": hot}[band]
        # prefer unused templates to reduce repetition
        avail = [k for k in range(len(pool)) if k not in used[band]]
        if not avail:
            used[band] = []
            avail = list(range(len(pool)))
        k = _pick(rng, avail)
        used[band].append(k)
        text, sent, emo, tone, sarc, iron, pa, ten = pool[k]
        ten = max(0.0, min(100.0, rng.gauss(ten, TENSION_NOISE)))
        if rng.random() < 0.25:  # occasionally reference the seed topic in-text
            text = f"{text} ({_surface(rng, domain, i)})"
        labels = dict(sentiment=sent, emotion=emo, tone=tone,
                      sarcasm=sarc, irony=iron, passive_aggression=pa)
        if rng.random() < LABEL_NOISE_RATE:      # controlled annotator-noise
            flippable = [lbl for lbl, v in labels.items() if v in (0, 1)]
            if flippable:
                lbl = _pick(rng, flippable)
                labels[lbl] = 1 - labels[lbl]
        convo.append({
            "conversation_id": conv_id,
            "message_id": i + 1,
            "speaker_id": speakers[i % 2],
            "timestamp": ts.isoformat(),
            "text": text,
            "platform": "synthetic",
            **labels,
            "tension": round(ten, 1),
            # blueprint §6 unified schema: escalation is derived from tension
            # (never an independent judgement), topic is the conversation's
            # situational domain
            "escalation": int(round(ten, 1) >= ESCALATION_TENSION_THRESHOLD),
            "topic": dom_name,
        })
        ts = ts + timedelta(seconds=rng.randint(10, 900))
    return convo


def generate_corpus(seed: int = 42, convs_per_cell: int = 14) -> list[list[dict]]:
    """6 domains × 7 arcs × convs_per_cell conversations (deterministic)."""
    convs = []
    conv_idx = 0
    for dom_name, domain in DOMAINS.items():
        for arc in ARC_PATTERNS:
            for _ in range(convs_per_cell):
                rng = _rng(seed * 1000 + conv_idx)
                conv_id = f"conv_{conv_idx:04d}"
                convs.append(_make_conversation(dom_name, domain, arc, conv_id, rng))
                conv_idx += 1
    return convs


def split_conversations(convs: list[list[dict]], seed: int = 42):
    """Split by whole conversation (never by message) — prevents context leakage."""
    def key(c):
        return int(hashlib.md5(c[0]["conversation_id"].encode()).hexdigest(), 16) % 10**8
    ranked = sorted(convs, key=lambda c: (key(c), c[0]["conversation_id"]))
    n = len(ranked)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {"train": ranked[:n_train],
            "val": ranked[n_train:n_train + n_val],
            "test": ranked[n_train + n_val:]}


def flatten(convs: list[list[dict]]) -> list[dict]:
    return [m for c in convs for m in c]


def corpus_stats(convs):
    msgs = flatten(convs)
    n = len(msgs)
    if n == 0:
        return {"conversations": 0, "messages": 0}
    from collections import Counter
    return {
        "conversations": len(convs),
        "messages": n,
        "speakers_per_conversation": round(sum(len({m['speaker_id'] for m in c})
                                               for c in convs) / len(convs), 2),
        "avg_messages_per_conversation": round(n / len(convs), 2),
        "sarcasm_positive_rate": round(sum(m["sarcasm"] for m in msgs) / n, 4),
        "irony_positive_rate": round(sum(m["irony"] for m in msgs) / n, 4),
        "passive_aggression_positive_rate": round(sum(m["passive_aggression"] for m in msgs) / n, 4),
        "sentiment_distribution": dict(Counter(m["sentiment"] for m in msgs)),
        "tension_mean": round(sum(m["tension"] for m in msgs) / n, 2),
    }


if __name__ == "__main__":
    corpus = generate_corpus()
    splits = split_conversations(corpus)
    print("total conversations:", len(corpus))
    for name, part in splits.items():
        print(f"  {name}: {len(part)} conversations, {len(flatten(part))} messages")
    import json
    print(json.dumps(corpus_stats(corpus), indent=2))
