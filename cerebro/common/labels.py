"""CEREBRO canonical label space (PS-01 §4). Single source of truth for every head."""

SENTIMENT_LABELS = ["positive", "neutral", "negative"]

EMOTION_LABELS = [
    "joy", "affection", "excitement", "relief",          # positive valence
    "neutral", "confusion", "surprise",                  # middle
    "sadness", "anxiety", "fear", "frustration",         # negative
    "anger", "disgust",                                  # high arousal negative
]

TONE_LABELS = [
    "friendly", "professional", "casual", "humorous", "supportive",
    "apologetic", "concerned", "sarcastic", "defensive",
    "dismissive", "cold", "frustrated", "aggressive", "passive-aggressive",
]

# How each signal is predicted
HEADS = {
    "sentiment": {"type": "multiclass", "labels": SENTIMENT_LABELS},
    "emotion":   {"type": "multiclass", "labels": EMOTION_LABELS},
    "tone":      {"type": "multiclass", "labels": TONE_LABELS},
    "tension":   {"type": "regression", "range": [0, 100]},
    "sarcasm":   {"type": "binary"},
    "irony":     {"type": "binary"},
    "passive_aggression": {"type": "binary"},
}

SCHEMA_VERSION = "1.0.0"
