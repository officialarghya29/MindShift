"""Hidden-signal detectors (PS-01 §14–16): sarcasm, irony, passive-aggression.

Architecture: learned head probability ⊕ symbolic contradiction evidence.
Context decides — never the phrase alone (PS-01 §16 constraint).
"""
from __future__ import annotations

import numpy as np

from cerebro.features.preprocess import micro_signals


def _sentiment_polarity(sent: dict) -> float:
    probs = sent.get("probabilities", {})
    return probs.get("positive", 0) - probs.get("negative", 0)


def _prior_contradiction(sent, context_text: str, sig: dict) -> float:
    """Evidence score for: literal-positive message inside negative context."""
    polarity = _sentiment_polarity(sent)
    ctx_neg = 0.0
    if context_text:
        ctx_low = context_text.lower()
        ctx_neg_words = ("crashed", "again", "late", "broken", "wrong", "failed",
                         "sorry", "unfortunately", "problem", "issue", "stuck",
                         "delay", "angry", "annoyed", "frustrated")
        hits = sum(1 for w in ctx_neg_words if w in ctx_low)
        ctx_neg = min(hits / 3.0, 1.0)
    lit_pos = polarity > 0.15
    return (0.30 * lit_pos * ctx_neg +
            0.20 * lit_pos * (1 if sig["exclam"] >= 1 else 0) +
            0.15 * lit_pos * min(len(sig["sarc_words"]) / 2.0, 1.0))


def sarcasm_score(learned: float, sent: dict, context_text: str, sig: dict,
                  prev_tension: float, tension: float) -> dict:
    """Learned probability ⊕ prior-based contradiction evidence → combined score."""
    contradiction = _prior_contradiction(sent, context_text, sig)
    spike = max(0.0, tension - prev_tension) / 100.0
    combined = 1 - (1 - learned) * (1 - min(contradiction, 0.95)) * (1 - 0.30 * spike)
    combined = float(np.clip(combined, 0, 1))
    supporting = []
    if contradiction > 0.3:
        supporting.append("positive wording in negative context")
    if sig["sarc_words"]:
        supporting.append(f"marker words: {', '.join(sig['sarc_words'][:3])}")
    if sig["emoji_sarc"]:
        supporting.append(f"sarcasm-typical emoji: {', '.join(sig['emoji_sarc'][:2])}")
    if sig["exclam"]:
        supporting.append("exclamation emphasis")
    if spike > 0.05:
        supporting.append(f"tension spike (+{spike*100:.0f} after this message)")
    return {"probability": round(combined, 4),
            "confidence": round(abs(combined - 0.5) * 2, 4),
            "supporting_signals": supporting,
            "learned_p": round(learned, 4),
            "contradiction_evidence": round(contradiction, 4)}


def irony_score(learned: float, sent: dict, context_text: str, sig: dict) -> dict:
    """Literal vs. contextual meaning mismatch (PS-01 §15)."""
    contradiction = _prior_contradiction(sent, context_text, sig)
    interjection = 0.15 if sig["interjections"] else 0.0
    quote_echo = 0.2 if sig["quoted_echo"] else 0.0
    combined = 1 - (1 - learned) * (1 - min(contradiction + interjection + quote_echo, 0.95))
    supporting = []
    if contradiction > 0.3:
        supporting.append("literal meaning opposes situational context")
    if sig["interjections"]:
        supporting.append("evaluative interjection ('wow'/'oh')")
    if sig["quoted_echo"]:
        supporting.append("quoted wording may echo context")
    return {"probability": round(min(combined, 1.0), 4),
            "confidence": round(abs(combined - 0.5) * 2, 4),
            "supporting_signals": supporting,
            "learned_p": round(learned, 4),
            "mismatch_evidence": round(contradiction, 4)}


def pa_score(learned: float, sig: dict, prev_tension: float, tension: float,
             speaker_state: dict) -> dict:
    """Concessive phrases + coldness + tension trajectory (context-gated)."""
    phrase = 0.35 if sig["pa_phrase"] else 0.0
    cold = 0.15 if sig["is_short"] and sig["exclam"] == 0 and sig["emoji_polarity"] <= 0 else 0.0
    prev_t = speaker_state.get("recent_tension_mean") or prev_tension
    trajectory = 0.2 if (tension - (prev_t or 0)) > 8 else 0.0
    combined = 1 - (1 - learned) * (1 - min(phrase + cold + trajectory, 0.95))
    supporting = []
    if phrase:
        supporting.append(f"concessive phrase: \"{sig['pa_phrase']}\"")
    if cold:
        supporting.append("short cold response pattern")
    if trajectory:
        supporting.append("rising tension trajectory")
    return {"probability": round(min(combined, 1.0), 4),
            "confidence": round(abs(combined - 0.5) * 2, 4),
            "supporting_signals": supporting,
            "learned_p": round(learned, 4)}


def apply_hidden_signals(results: list[dict]) -> list[dict]:
    """Post-process engine output: fuse learned + symbolic evidence in place."""
    prev_t = 0.0
    for r in results:
        r["sarcasm"] = sarcasm_score(r["sarcasm"]["probability"], r["sentiment"],
                                     r["context_text"], r["signals"], prev_t,
                                     r["tension"])
        r["irony"] = irony_score(r["irony"]["probability"], r["sentiment"],
                                 r["context_text"], r["signals"])
        r["passive_aggression"] = pa_score(r["passive_aggression"]["probability"],
                                           r["signals"], prev_t, r["tension"],
                                           r["speaker_state_before"])
        prev_t = r["tension"]
    return results
