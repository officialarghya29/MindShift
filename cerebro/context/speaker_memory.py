"""Per-speaker analytical memory (PS-01 §9).

Tracks each speaker's recent emotional state so context-dependent messages
("Fine.", "Whatever.") are interpreted against personal history, not in isolation.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class SpeakerMemory:
    def __init__(self, window: int = 5):
        self.window = window
        self.emo_hist = defaultdict(lambda: deque(maxlen=window))
        self.tone_hist = defaultdict(lambda: deque(maxlen=window))
        self.sent_hist = defaultdict(lambda: deque(maxlen=window))
        self.tension_hist = defaultdict(lambda: deque(maxlen=window))
        self.short_streak = defaultdict(int)

    def observe(self, speaker: str, emotion: str, tone: str,
                sentiment: str, tension: float, short_msg: bool) -> None:
        self.emo_hist[speaker].append(emotion)
        self.tone_hist[speaker].append(tone)
        self.sent_hist[speaker].append(sentiment)
        self.tension_hist[speaker].append(float(tension))
        self.short_streak[speaker] = self.short_streak[speaker] + 1 if short_msg else 0

    def state(self, speaker: str) -> dict:
        t = list(self.tension_hist[speaker])
        return {
            "last_emotion": self.emo_hist[speaker][-1] if self.emo_hist[speaker] else None,
            "last_tone": self.tone_hist[speaker][-1] if self.tone_hist[speaker] else None,
            "last_sentiment": self.sent_hist[speaker][-1] if self.sent_hist[speaker] else None,
            "recent_tension_mean": float(np.mean(t)) if t else None,
            "short_streak": self.short_streak[speaker],
        }

    def reset(self) -> None:
        self.emo_hist.clear()
        self.tone_hist.clear()
        self.sent_hist.clear()
        self.tension_hist.clear()
        self.short_streak.clear()
