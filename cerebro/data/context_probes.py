"""Designed context-dependence benchmark (blueprint §3, §12, §36).

Why this module exists
----------------------
The generated corpus of §3 cannot answer the blueprint's headline research
question — *"can contextual and temporal information improve the understanding of
emotion and tone in multi-turn conversations compared with message-only
analysis?"* — because in that corpus every surface utterance carries a **fixed**
label. A text-only model is therefore already Bayes-optimal and no amount of
context can add information. The measured ablation confirms exactly that.

This module supplies the complementary, **controlled** experiment: a probe corpus
in which the text is deliberately uninformative and *only the preceding turns*
disambiguate it. That is the setting §12 describes in prose ("Fine.", "Okay.",
"Sure.", "Whatever." change meaning with history) and the setting the blueprint
requires us to prove.

Design guarantees (this is an experiment, not a scrape)
-------------------------------------------------------
1. **Utterance-level label balance.** Every probe utterance appears with every
   one of its gold labels equally often, so the mutual information between the
   probe text and its label is 0 by construction. A text-only model cannot beat
   the majority-class rate on the probes; any measured lift is attributable to
   context / speaker memory, not to lexical leakage.
2. **Matched histories.** Each probe is instantiated over benign and tense
   histories drawn from the *same* vocabulary as the main corpus bands, so the
   two conditions differ only in their history.
3. **Unambiguous controls.** A set of probes whose label does *not* depend on
   history is included, so we can show context does not degrade easy turns.
4. **Conversation-level split, stratified by (utterance, condition)** so both
   polarities are present in every split — no context leakage, no imbalance.

The polarity (benign / tense) is *not* injected as a label; it is only ever
observable through the history turns, which is what makes the probe informative.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from cerebro.common.labels import ESCALATION_TENSION_THRESHOLD

# ---------------------------------------------------------------------------
# history scripts — 5 turns each, drawn from the main corpus's vocabulary
# ---------------------------------------------------------------------------
# turn: (text, sentiment, emotion, tone, sarcasm, irony, pa, tension)

BENIGN_HISTORIES = [
    [
        ("Morning! Did you get a chance to look at the plan?", "neutral", "neutral", "friendly", 0, 0, 0, 12),
        ("Yeah, I went through it last night. Looks solid.", "positive", "relief", "casual", 0, 0, 0, 10),
        ("Great. I'll finish my part tonight.", "positive", "neutral", "friendly", 0, 0, 0, 10),
        ("Perfect, thanks! Let me know if anything blocks you.", "positive", "joy", "supportive", 0, 0, 0, 8),
        ("Will do. Honestly this is going well.", "positive", "joy", "casual", 0, 0, 0, 8),
    ],
    [
        ("Hey, how did the review go?", "neutral", "concerned", "friendly", 0, 0, 0, 10),
        ("Really well, they liked the structure.", "positive", "joy", "casual", 0, 0, 0, 10),
        ("That's great to hear. Nothing blocking then?", "positive", "relief", "friendly", 0, 0, 0, 8),
        ("Nothing at all, we're actually ahead.", "positive", "excitement", "casual", 0, 0, 0, 8),
        ("Love that. Ping me if that changes.", "positive", "affection", "supportive", 0, 0, 0, 6),
    ],
    [
        ("Just finished the setup, everything works now.", "positive", "relief", "casual", 0, 0, 0, 12),
        ("Nice, that was quick.", "positive", "joy", "friendly", 0, 0, 0, 8),
        ("Yeah, it clicked once I read the docs properly.", "positive", "relief", "casual", 0, 0, 0, 8),
        ("Good. I'll send the summary in a bit.", "neutral", "neutral", "professional", 0, 0, 0, 12),
        ("Thanks, that was fast.", "positive", "affection", "friendly", 0, 0, 0, 8),
    ],
]

TENSE_HISTORIES = [
    [
        ("You said you'd send it yesterday. It's still not here.", "negative", "frustration", "frustrated", 0, 0, 0, 52),
        ("I know, I'm sorry. The day got away from me.", "negative", "sadness", "apologetic", 0, 0, 0, 44),
        ("That's the second time this week. I keep waiting on you.", "negative", "frustration", "cold", 0, 0, 0, 62),
        ("And now I'm the one chasing it again.", "negative", "anger", "aggressive", 0, 0, 0, 74),
        ("Every single time. You never change.", "negative", "anger", "aggressive", 0, 0, 0, 84),
    ],
    [
        ("We keep going in circles on this.", "negative", "frustration", "dismissive", 0, 0, 0, 55),
        ("Because nobody actually reads what I write.", "negative", "sadness", "defensive", 0, 0, 0, 50),
        ("That's not fair, I read everything.", "negative", "anger", "defensive", 0, 0, 0, 58),
        ("Then why is it still wrong?", "negative", "frustration", "cold", 0, 0, 0, 66),
        ("I don't have the energy to argue about this again.", "negative", "sadness", "cold", 0, 0, 0, 70),
    ],
    [
        ("The file you sent is the old one. Again.", "negative", "frustration", "frustrated", 0, 0, 0, 54),
        ("Are you serious? I checked it twice.", "negative", "confusion", "defensive", 0, 0, 0, 58),
        ("Then check a third time, because it's wrong.", "negative", "anger", "aggressive", 0, 0, 0, 76),
        ("Fine. Blame me like you always do.", "negative", "frustration", "passive-aggressive", 0, 0, 1, 72),
        ("I'm not blaming you, I'm telling you it's broken.", "negative", "anger", "aggressive", 0, 0, 0, 80),
    ],
]

# ---------------------------------------------------------------------------
# probes — the label of the *same* utterance under a benign vs a tense history
# ---------------------------------------------------------------------------
# Each probe is instantiated over three benign and three tense history scripts.
# The splitter holds the third script's wording out of training, so the reported
# context gain cannot be explained by memorising the history text.
# (id, text, benign label set, tense label set)
# label set = (sentiment, emotion, tone, sarcasm, irony, passive_aggression, tension)
# probe ("fine", polarity) is instantiated over a history of the matching polarity.

CONTEXT_DEPENDENT_PROBES = [
    ("fine", "Fine.",
     ("neutral", "neutral", "casual", 0, 0, 0, 15),
     ("negative", "frustration", "cold", 0, 0, 1, 62)),
    ("sure", "Sure.",
     ("positive", "neutral", "friendly", 0, 0, 0, 14),
     ("negative", "frustration", "dismissive", 0, 0, 1, 58)),
    ("okay", "Okay.",
     ("neutral", "neutral", "casual", 0, 0, 0, 12),
     ("negative", "frustration", "dismissive", 0, 0, 1, 56)),
    ("whatever", "Whatever.",
     ("neutral", "neutral", "casual", 0, 0, 0, 20),
     ("negative", "anger", "passive-aggressive", 0, 0, 1, 70)),
    ("no_problem", "No problem.",
     ("positive", "affection", "supportive", 0, 0, 0, 10),
     ("negative", "frustration", "passive-aggressive", 0, 0, 1, 60)),
    ("thanks", "Thanks.",
     ("positive", "affection", "friendly", 0, 0, 0, 10),
     ("negative", "disgust", "sarcastic", 1, 0, 0, 64)),
    ("great", "Great.",
     ("positive", "joy", "friendly", 0, 0, 0, 12),
     ("negative", "anger", "sarcastic", 1, 0, 0, 72)),
    ("perfect_timing", "Wow, perfect timing.",
     ("positive", "joy", "friendly", 0, 0, 0, 12),
     ("negative", "frustration", "sarcastic", 1, 1, 0, 74)),
    ("said_yesterday", "You said that yesterday too.",
     ("neutral", "confusion", "concerned", 0, 0, 0, 30),
     ("negative", "frustration", "frustrated", 0, 0, 0, 62)),
    ("tonight", "I'll do it tonight.",
     ("positive", "neutral", "professional", 0, 0, 0, 12),
     ("negative", "anxiety", "defensive", 0, 0, 1, 58)),
    ("its_fine", "It's fine.",
     ("neutral", "relief", "casual", 0, 0, 0, 14),
     ("negative", "sadness", "cold", 0, 0, 1, 60)),
    ("right", "Right.",
     ("positive", "neutral", "casual", 0, 0, 0, 12),
     ("negative", "anger", "sarcastic", 1, 0, 0, 66)),
    ("im_sorry", "I'm sorry.",
     ("positive", "relief", "apologetic", 0, 0, 0, 20),
     ("negative", "anxiety", "defensive", 0, 0, 0, 52)),
    ("okay_then", "Okay then.",
     ("neutral", "neutral", "casual", 0, 0, 0, 16),
     ("negative", "frustration", "passive-aggressive", 0, 0, 1, 64)),
    ("nice", "Nice.",
     ("positive", "joy", "friendly", 0, 0, 0, 12),
     ("negative", "disgust", "sarcastic", 1, 0, 0, 68)),
    ("do_what_you_want", "Do what you want.",
     ("neutral", "neutral", "casual", 0, 0, 0, 18),
     ("negative", "anger", "passive-aggressive", 0, 0, 1, 74)),
]

# unambiguous controls: identical label under either history, so a text-only
# model should get them right and we can show context does no harm
CONTROL_PROBES = [
    ("ctl_best_news", "Honestly this is the best news I've had all week!",
     ("positive", "joy", "friendly", 0, 0, 0, 8),
     ("positive", "joy", "friendly", 0, 0, 0, 8)),
    ("ctl_furious", "I am absolutely furious about how this was handled.",
     ("negative", "anger", "aggressive", 0, 0, 0, 88),
     ("negative", "anger", "aggressive", 0, 0, 0, 88)),
    ("ctl_send_file", "Could you send me the updated file when you get a sec?",
     ("neutral", "neutral", "professional", 0, 0, 0, 14),
     ("neutral", "neutral", "professional", 0, 0, 0, 14)),
    ("ctl_thank_you", "Thank you so much, that really helped me out.",
     ("positive", "affection", "supportive", 0, 0, 0, 8),
     ("positive", "affection", "supportive", 0, 0, 0, 8)),
    ("ctl_mess", "This is a mess and I don't even know where to start.",
     ("negative", "anxiety", "concerned", 0, 0, 0, 56),
     ("negative", "anxiety", "concerned", 0, 0, 0, 56)),
    ("ctl_surprised", "Wait, it actually worked first try? That's amazing.",
     ("positive", "surprise", "friendly", 0, 0, 0, 14),
     ("positive", "surprise", "friendly", 0, 0, 0, 14)),
]

ALL_PROBES = CONTEXT_DEPENDENT_PROBES + CONTROL_PROBES

_SPEAKERS = ["Aarav", "Meera"]
PROBE_TURN_INDEX_OFFSET = len(BENIGN_HISTORIES[0])   # history length


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 31))


def _turn(m, i, conv_id, speaker, ts, topic, is_probe):
    text, sent, emo, tone, sarc, iron, pa, ten = m
    return {
        "conversation_id": conv_id,
        "message_id": i + 1,
        "speaker_id": speaker,
        "timestamp": ts.isoformat(),
        "text": text,
        "platform": "probe",
        "sentiment": sent,
        "emotion": emo,
        "tone": tone,
        "sarcasm": sarc,
        "irony": iron,
        "passive_aggression": pa,
        "tension": float(ten),
        "escalation": int(float(ten) >= ESCALATION_TENSION_THRESHOLD),
        "topic": topic,
        "is_probe": is_probe,
        "probe_id": None,
        "probe_condition": None,
    }


REPS_PER_SCRIPT = 4


def build_probe_corpus(reps_per_script: int = REPS_PER_SCRIPT) -> list[list[dict]]:
    """One conversation per (probe, condition, history script, repetition).

    Every conversation is 5 history turns + 1 probe turn. The probe turn is
    flagged with ``is_probe`` and carries ``history_index`` so the split can hold
    a whole history *wording* out of training.
    """
    convs = []
    for probe_id, text, benign, tense in ALL_PROBES:
        for cond, label in (("benign", benign), ("tense", tense)):
            histories = BENIGN_HISTORIES if cond == "benign" else TENSE_HISTORIES
            for hidx in range(len(histories)):
                for k in range(reps_per_script):
                    conv_id = f"probe_{probe_id}_{cond}_h{hidx}_k{k}"
                    rng = _rng(conv_id)
                    history = histories[hidx]
                    topic = "probe:" + probe_id
                    ts = (datetime(2025, 6, 1, 9, 0)
                          + timedelta(minutes=rng.randint(0, 600)))
                    # the probe speaker answers the other speaker's last turn
                    history_speakers = [_SPEAKERS[i % 2] for i in range(len(history))]
                    probe_speaker = _SPEAKERS[len(history) % 2]
                    convo = []
                    for i, h in enumerate(history):
                        convo.append(_turn(h, i, conv_id, history_speakers[i], ts,
                                           topic, False))
                        ts = ts + timedelta(seconds=rng.randint(20, 400))
                    probe = _turn((text, *label), len(history), conv_id,
                                  probe_speaker, ts, topic, True)
                    probe["probe_id"] = probe_id
                    probe["probe_condition"] = cond
                    probe["history_index"] = hidx
                    probe["rep"] = k
                    convo.append(probe)
                    convs.append(convo)
    return convs


def split_probes(convs: list[list[dict]], seed: int = 42):
    """Conversation-level split, stratified by (probe_id, condition).

    Two TEST regimes, so the effect can be separated from memorised history
    wording:
      * ``test``        — histories whose *wording* is held out entirely
                          (scripts 2 only). The model must generalise the
                          contextual mapping to unseen phrasing.
      * ``test_seen``   — histories whose wording also occurs in TRAIN. An upper
                          reference: it isolates "does it use the state" from
                          "does it transfer the phrasing".

    Train uses scripts 0-1 only, so script 2 never appears in training.
    Stratifying on (probe_id, condition) guarantees every probe utterance appears
    with both of its gold labels inside TRAIN — the property that pins the
    text-only ceiling to the majority-class rate.
    """
    groups: dict[tuple[str, str], dict[int, list[list[dict]]]] = {}
    for c in convs:
        probe = c[-1]
        groups.setdefault((probe["probe_id"], probe["probe_condition"]),
                          {}).setdefault(probe["history_index"], []).append(c)

    train, val, test, test_seen = [], [], [], []
    for key in sorted(groups):
        by_script = groups[key]
        for script, items in sorted(by_script.items()):
            items = sorted(items, key=lambda c: (c[-1]["rep"], c[0]["conversation_id"]))
            if script <= 1:
                # scripts 0-1 are train material; the final repetition of
                # script 1 becomes the seen-wording test reference
                train.extend(items[:-2])
                val.extend(items[-2:-1])
                test_seen.extend(items[-1:])
            else:
                test.extend(items)
    return {"train": train, "val": val, "test": test, "test_seen": test_seen}


def probe_stats(convs) -> dict:
    probes = [c[-1] for c in convs]
    n = len(probes)
    out = {"conversations": len(convs), "probe_turns": n,
           "history_turns": sum(len(c) - 1 for c in convs),
           "history_scripts_per_condition": len(BENIGN_HISTORIES),
           "context_dependent_probes": len(CONTEXT_DEPENDENT_PROBES),
           "control_probes": len(CONTROL_PROBES)}
    # label balance of the ambiguous probes: every utterance must carry each of
    # its gold labels equally often, otherwise text-only analysis could cheat
    from collections import Counter
    per_utt = Counter((p["probe_id"], p["sentiment"]) for p in probes
                      if p["probe_id"] and not p["probe_id"].startswith("ctl_"))
    counts = Counter()
    for (_pid, _s), c in per_utt.items():
        counts[c] += 1
    out["utterance_labels_balanced"] = len(counts) == 1 or set(counts) <= {1}
    out["label_count_distribution"] = {str(k): v for k, v in sorted(counts.items())}
    return out


if __name__ == "__main__":
    import json
    corpus = build_probe_corpus()
    splits = split_probes(corpus)
    print("probe conversations:", len(corpus))
    for name, part in splits.items():
        print(f"  {name}: {len(part)} conversations")
    print(json.dumps(probe_stats(corpus), indent=2))
