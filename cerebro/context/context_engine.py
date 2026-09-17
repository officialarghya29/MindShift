"""Sliding context window + long-range summary embedding (PS-01 §8)."""
from __future__ import annotations


class ContextWindow:
    """Previous K turns (verbatim) + tension-ranked selection of older turns.
    `decay` is kept for API compatibility with the documented long-range
    weighting; the selection itself is salience-based."""

    def __init__(self, k: int = 4, decay: float = 0.85):
        self.k = k
        self.decay = decay

    def context_text(self, messages: list[dict], i: int) -> str:
        lo = max(0, i - self.k)
        recent = " || ".join(m["text"] for m in messages[lo:i])
        older = messages[:lo]
        if older:
            top = sorted(older, key=lambda m: -abs(m.get("tension", 0)))[:3]
            recent = " :: ".join(m["text"] for m in top) + " || " + recent
        return recent or ""

    def context_vector(self, vec, messages: list[dict], i: int):
        from scipy.sparse import csr_matrix
        txt = self.context_text(messages, i)
        return csr_matrix(vec.transform([txt])) if txt else \
            csr_matrix((1, len(vec.get_feature_names_out())))


def conversation_summary(messages: list[dict], max_chars: int = 600) -> str:
    """Cheap extractive summary: highest-tension message per 10-turn block."""
    picks = []
    for start in range(0, len(messages), 10):
        block = messages[start:start + 10]
        best = max(block, key=lambda m: abs(m.get("tension", 0)))
        picks.append(best["text"])
    s = " ".join(picks)
    return s[:max_chars]
