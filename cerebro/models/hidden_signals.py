"""Hidden-signal detectors (PS-01 §14–16): sarcasm, irony, passive-aggression.

Architecture: learned head probability ⊕ symbolic contradiction evidence.
Context decides — never the phrase alone (PS-01 §16 constraint).
"""
from __future__ import annotations

import numpy as np


def _sentiment_polarity(sent: dict) -> float:
    probs = sent.get("probabilities", {})
    return probs.get("positive", 0) - probs.get("negative", 0)


def _prior_contradiction(sent, context_text: str, sig: dict,
                         prev_tension: float = 0.0) -> float:
    """Evidence score for: literal-positive message inside a negative context.

    Negative context = lexically negative window OR high ambient tension
    (a heated conversation is a negative situational context even when its
    words are neutral — the Sperber–Wilson echoic gap). Interjections add
    echoic evidence, always gated on literal positivity.
    """
    polarity = _sentiment_polarity(sent)
    ctx_neg = 0.0
    if context_text:
        ctx_low = context_text.lower()
        ctx_neg_words = ("crashed", "again", "late", "broken", "wrong", "failed",
                         "sorry", "unfortunately", "problem", "issue", "stuck",
                         "delay", "angry", "annoyed", "frustrated")
        hits = sum(1 for w in ctx_neg_words if w in ctx_low)
        ctx_neg = min(hits / 3.0, 1.0)
    # ambient-tension heat: (prev_tension-35)/45 capped, contributing up to 0.6
    ctx_heat = min(max(prev_tension - 35.0, 0.0) / 45.0, 1.0) * 0.6
    ctx_neg = max(ctx_neg, ctx_heat)
    lit_pos = polarity > 0.10
    interjection = 0.15 if sig["interjections"] else 0.0
    return (0.30 * lit_pos * ctx_neg +
            0.20 * lit_pos * (1 if sig["exclam"] >= 1 else 0) +
            0.15 * lit_pos * min(len(sig["sarc_words"]) / 2.0, 1.0) +
            interjection * lit_pos)


def sarcasm_score(learned: float, sent: dict, context_text: str, sig: dict,
                  prev_tension: float, tension: float) -> dict:
    """Learned probability ⊕ prior-based contradiction evidence → combined score."""
    contradiction = _prior_contradiction(sent, context_text, sig, prev_tension)
    repetition = min(len([w for w in sig.get("repeated_words", [])
                          if w in sig.get("pos_hits", [])
                          or w in sig.get("sarc_words", [])]) / 2.0, 1.0) * 0.25
    # sincerity markers ("thanks", "I'll review", ...) resolve the ambiguity
    # toward the literal reading: shrink contradiction evidence AND grow the
    # sincere residual so the combined score drops (sign-checked).
    is_coop = bool(sig.get("cooperative"))
    contradiction = contradiction * (0.25 if is_coop else 1.0)
    spike = max(0.0, tension - prev_tension) / 100.0
    residual = (1 - learned) * (1 - min(contradiction + repetition, 0.95)) \
        * (1 - 0.30 * spike) * (2.5 if is_coop else 1.0)
    combined = float(np.clip(1 - min(residual, 1.0), 0, 1))
    supporting = []
    if contradiction > 0.3:
        supporting.append("positive wording in negative context")
    if sig["sarc_words"]:
        supporting.append(f"marker words: {', '.join(sig['sarc_words'][:3])}")
    if sig.get("repeated_words"):
        supporting.append(f"repeated wording: {', '.join(sig['repeated_words'][:2])}")
    if sig.get("cooperative"):
        supporting.append("cooperative intent markers present (argues against sarcasm)")
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


def irony_score(learned: float, sent: dict, context_text: str, sig: dict,
                prev_tension: float = 0.0) -> dict:
    """Literal vs. contextual meaning mismatch (PS-01 §15)."""
    contradiction = _prior_contradiction(sent, context_text, sig, prev_tension)
    # cooperative sincerity markers argue for the literal reading here too
    contradiction = contradiction * (0.25 if sig.get("cooperative") else 1.0)
    quote_echo = 0.2 if sig["quoted_echo"] else 0.0
    combined = 1 - (1 - learned) * (1 - min(contradiction + quote_echo, 0.95))
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
                                 r["context_text"], r["signals"], prev_t)
        r["passive_aggression"] = pa_score(r["passive_aggression"]["probability"],
                                           r["signals"], prev_t, r["tension"],
                                           r["speaker_state_before"])
        prev_t = r["tension"]
    return results
