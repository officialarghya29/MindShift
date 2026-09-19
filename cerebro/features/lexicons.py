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

# Several SARC_MARKERS words are ironic in only ONE of their senses. Token-level
# membership therefore fires on sincere messages: "I was sure the deadline was next
# month" is plain certainty, not the dismissive "Sure.", and "that's right" is
# agreement, not the confrontational "Right.". Error analysis surfaced both as
# false positives at p ~ 0.52-0.58 on sincere messages in heated windows.
# A marker is suppressed when any of its non-ironic frames appears in the message.
# (Same idea as PA_PHRASES, which matches positionally rather than by token.)
SARC_SENSE_BLOCKERS = {
    "sure": ("i was sure", "i'm sure", "im sure", "make sure", "for sure",
             "not sure", "wasn't sure", "be sure", "sure that", "sure if",
             "sure about", "sure to", "sure thing", "sure you"),
    "right": ("that's right", "that is right", "you're right", "you are right",
              "all right", "right now", "right away", "right there",
              "right back", "isn't right", "not right"),
    "great": ("great to hear", "great news", "that's great", "feel great"),
    "thanks": ("thanks for", "thanks so much", "thanks a lot", "no thanks"),
}

# Explicit belief-revision markers: the speaker is updating on new information
# rather than echoing a prior grievance. Sarcasm requires an echoic gap between
# literal praise and a known negative fact (PS-01 §14); genuine surprise is the
# opposite — the speaker did not know the fact yet. Used to dampen contradiction
# evidence, exactly as the `cooperative` markers do, but more gently.
BELIEF_UPDATE_MARKERS = ("wait", "no way", "i was sure", "i thought", "really?",
                         "for real", "you did", "did you", "already",
                         "wasn't expecting", "didn't expect", "turns out")
PA_PHRASES = [
    "fine.", "whatever.", "do what you want", "okay then.", "ok then.", "no problem.",
    "as you wish", "i'm not angry", "i'm not mad", "do whatever you want", "suit yourself",
    "if you say so", "i guess it's fine", "sure, go ahead", "it's fine, really",
]
EXAG_WORDS = {"literally", "never", "always", "everyone", "nobody", "everything", "nothing"}

EMOJI_SENTIMENT = {
    "😀": .8, "😃": .8, "😄": .8, "😁": .7, "😊": .7, "🙂": .5, "😍": .8, "🥰": .8,
    "😂": .6, "🤣": .6, "😉": .4, "👍": .6, "❤️": .8, "🔥": .5, "🎉": .7, "✅": .4,
    "😢": -.7, "😭": -.7, "😡": -.9, "🤬": -.9, "😠": -.8, "😑": -.3,
    "🙄": -.5, "😒": -.6, "😞": -.7, "😔": -.6, "☹️": -.6, "👎": -.6,
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
