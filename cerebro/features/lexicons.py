"""Hand-built lexicons used by the interpretable feature layer (PS-01 §5, §14–16).

Design note: lexicons provide *evidence*, never verdicts — context decides.
"""
from __future__ import annotations

import re

POS_LEX = {
    "good", "great", "awesome", "nice", "love", "loved", "perfect", "thanks", "thank",
    "happy", "glad", "cool", "excellent", "amazing", "wonderful", "best", "sweet",
    "excited", "fantastic", "brilliant", "enjoy", "enjoyed", "fun", "yay", "win",
    "appreciate", "appreciated", "helpful", "works", "smooth", "beautiful", "solid",
}
NEG_LEX = {
    "bad", "worst", "hate", "hated", "terrible", "awful", "horrible", "angry", "annoyed",
    "annoying", "frustrated", "frustrating", "ridiculous", "stupid", "pathetic", "useless",
    "broken", "crashed", "fail", "failed", "failure", "never", "wrong", "waste", "late",
    "again", "still", "sick", "tired", "exhausted", "unbelievable", "overwhelmed",
    "screwed", "messed", "liar", "lie", "lying", "betrayed", "unfair", "ignore", "ignored",
}
SARC_MARKERS = {"yeah", "sure", "great", "perfect", "obviously", "totally", "brilliant",
                "fantastic", "wonderful", "amazing", "right", "exactly", "congrats", "thanks"}
IRONY_MARKERS = {"wow", "ah", "oh", "great", "perfect", "just", "exactly", "clearly"}
PA_PHRASES = [
    "fine.", "whatever.", "do what you want", "okay then.", "ok then.", "no problem.",
    "as you wish", "i'm not angry", "i'm not mad", "do whatever you want", "suit yourself",
    "if you say so", "i guess it's fine", "sure, go ahead", "it's fine, really",
]
EXAG_WORDS = {"literally", "never", "always", "everyone", "nobody", "everything", "nothing"}

EMOJI_SENTIMENT = {
    "😀": .8, "😃": .8, "😄": .8, "😁": .7, "😊": .7, "🙂": .5, "😍": .8, "🥰": .8,
    "😂": .6, "🤣": .6, "😉": .4, "👍": .6, "❤️": .8, "🔥": .5, "🎉": .7, "✅": .4,
    "😞": -.7, "😢": -.7, "😭": -.7, "😡": -.9, "🤬": -.9, "😠": -.8, "😑": -.3,
    "🙄": -.5, "😒": -.6, "😞": -.7, "😔": -.6, "☹️": -.6, "😞": -.7, "👎": -.6,
    "😬": -.3, "😩": -.6, "🤦": -.5, "💀": -.3, "🫠": -.4, "🥲": -.5, "😅": -.1,
}
EMOJI_SARC_HINT = {"🙄", "😒", "🫠", "🥲", "😅", "🙃", "😌", "🤡"}

LAUGHTER_RE = re.compile(r"\b(ha)+\b|\blol+\b|\blmao+\b|\brote?fl\b", re.I)
ELLIPSIS = re.compile(r"\.{3,}|…")
SWEAR_RE = re.compile(
    r"\b(fuck|shit|bitch|bastard|asshole|damn|hell|crap|dumb|idiot|moron)s?\b", re.I)
SLANG_MAP = {
    "u": "you", "ur": "your", "r": "are", "ya": "yes", "yep": "yes", "nope": "no",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "dunno": "do not know",
    "k": "ok", "kk": "ok", "thx": "thanks", "pls": "please", "plz": "please",
    "omg": "oh my god", "smh": "shaking my head", "brb": "be right back",
    "idc": "i do not care", "idk": "i do not know", "nvm": "never mind",
    "btw": "by the way", "tbh": "to be honest", "smgdh": "shake my god damn head",
}
CONCESSIVES = {"ok", "okay", "fine", "sure", "whatever", "right", "yeah", "yes"}
