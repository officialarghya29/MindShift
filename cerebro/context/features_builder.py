"""Context-augmented feature construction.

For every message i in a conversation:
    [TF-IDF(text_i) ⊕ TF-IDF(context_i) ⊕ behavioral_i ⊕ memory_i]

Training uses gold history; inference re-feeds the model's own predictions
(teacher-forcing gap is documented in docs/methodology).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from cerebro.features.preprocess import process_text, behavioral_vector
from cerebro.context.context_engine import ContextWindow
from cerebro.context.speaker_memory import SpeakerMemory

SENT_CODE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
EMO_AROUSAL = {"joy": .6, "affection": .3, "excitement": .8, "relief": -.2,
               "neutral": 0, "confusion": .1, "surprise": .5, "sadness": -.5,
               "anxiety": .3, "fear": .4, "frustration": .6, "anger": .9,
               "disgust": .7}
EMO_VALENCE = {"joy": .8, "affection": .7, "excitement": .7, "relief": .5,
               "neutral": 0, "confusion": -.1, "surprise": .1, "sadness": -.7,
               "anxiety": -.5, "fear": -.6, "frustration": -.6, "anger": -.8,
               "disgust": -.7}
TONE_WARMTH = {"friendly": .8, "supportive": .9, "apologetic": .4, "concerned": .3,
               "casual": .5, "humorous": .7, "professional": .2, "sarcastic": -.5,
               "defensive": -.3, "dismissive": -.6, "cold": -.7, "frustrated": -.6,
               "aggressive": -.9, "passive-aggressive": -.6}
MEMORY_DIMS = 10  # 5 raw + 5 interactions (documented order)


def _memory_vector(mem: SpeakerMemory, speaker: str, behav: np.ndarray) -> np.ndarray:
    st = mem.state(speaker)
    last_emo_arousal = EMO_AROUSAL.get(st["last_emotion"], 0.0)
    last_emo_valence = EMO_VALENCE.get(st["last_emotion"], 0.0)
    last_tone_warmth = TONE_WARMTH.get(st["last_tone"], 0.0)
    last_sent = SENT_CODE.get(st["last_sentiment"], 0.0)
    recent_tension = st["recent_tension_mean"] if st["recent_tension_mean"] is not None else 0.0
    base = np.array([
        last_emo_arousal, last_emo_valence, last_tone_warmth, last_sent,
        recent_tension / 100.0,
    ])
    # interactions with current behavioral signals (index map: see preprocess.py)
    cur_neg = behav[7] / 3.0        # neg_hits
    cur_excl = behav[2] / 3.0       # exclamation count
    cur_short = behav[14]           # short-response flag
    cur_gap = behav[15] / 24.0      # response delay (hours, capped)
    inter = np.array([
        last_emo_valence * cur_neg,
        last_tone_warmth * cur_excl,
        recent_tension / 100.0 * cur_neg,
        cur_short * (1.0 - (last_tone_warmth + 1.0) / 2.0),
        cur_gap * (recent_tension / 100.0),
    ])
    return np.concatenate([base, inter])


def build_conversation_matrix(vec, messages: list[dict], use_context: bool = True,
                              use_memory: bool = True, use_behavior: bool = True):
    """Returns (X, meta) where meta carries per-row context text for explainability."""
    cw = ContextWindow()
    mem = SpeakerMemory()
    X_rows, meta = [], []
    behavior_cache = []
    for i, m in enumerate(messages):
        behav = np.asarray(behavioral_vector(
            m["text"], messages[i - 1]["timestamp"] if i else None,
            m.get("timestamp"), messages[i - 1]["speaker_id"] if i else None,
            m["speaker_id"]), dtype=float)
        behavior_cache.append(behav)

        text_part = vec.transform([process_text(m["text"])])
        parts = [text_part]
        if use_context:
            ctx_txt = cw.context_text(messages, i)
            if ctx_txt:
                parts.append(vec.transform([process_text(ctx_txt)]))
            else:
                parts.append(csr_matrix(text_part.shape))
        behav_part = csr_matrix(behav.reshape(1, -1)) if use_behavior else None
        if use_behavior:
            parts.append(behav_part)
        if use_memory:
            mem_vec = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
            parts.append(csr_matrix(mem_vec))
        X_rows.append(hstack(parts).tocsr())

        st = mem.state(m["speaker_id"])
        meta.append({"row": i, "context_text": cw.context_text(messages, i) if use_context else "",
                     "speaker_state": dict(st)})
        # teacher forcing: advance memory with gold labels during training data prep
        mem.observe(m["speaker_id"], m.get("emotion", "neutral"),
                    m.get("tone", "neutral"), m.get("sentiment", "neutral"),
                    m.get("tension", 0.0), bool(behav[14]))
    X = vstack(X_rows).tocsr()
    return X, meta


def infer_mode(use_context: bool, use_memory: bool, use_behavior: bool) -> str:
    if not (use_context or use_memory or use_behavior):
        return "A"
    if use_context and not (use_memory or use_behavior):
        return "B"
    if use_context and use_memory and not use_behavior:
        return "C"
    if use_context and use_memory and use_behavior:
        return "E"
    return "D"
