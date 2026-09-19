"""Explainability engine (PS-01 §26–29).

Every important prediction ships with evidence, not vibes:
  - WHY?      → prediction + confidence + detected signals + inferred reading
  - WHAT CHANGED? → before/after state deltas + contributing signals
  - speaker profiles → distributions per participant (not a diagnosis)

Detected evidence and inferred interpretation are kept strictly separate.
"""
from __future__ import annotations

import numpy as np

TONE_DISPLAY = {
    "passive-aggressive": "Passive-aggressive", "passive_aggressive": "Passive-aggressive",
}


def _display(x: str) -> str:
    return TONE_DISPLAY.get(x, x.replace("_", " ").capitalize() if x else x)


def behavior_flags(behav: np.ndarray) -> dict:
    """Human-readable behavioral signals from the 16-dim vector (§17)."""
    (n_words, n_chars, excl, ques, rep_punct, caps_ratio, pos_hits, neg_hits,
     exag, emoji_pol, emoji_sarc, laugh, ellipsis, quotes_swear, short,
     gap) = behav
    intensity = float(np.clip((excl * .18 + rep_punct * .22 + caps_ratio * 2.2 +
                               neg_hits * .15 + quotes_swear * .3 + exag * .1), 0, 1))
    return {
        "intensity": round(intensity, 3),
        "short_response": bool(short),
        "negative_lexicon": int(neg_hits),
        "positive_lexicon": int(pos_hits),
        "exaggeration_words": int(exag),
        "emoji_polarity": round(float(emoji_pol), 2),
        "response_gap_hours": round(float(gap), 2),
        "caps_ratio": round(float(caps_ratio), 3),
    }


def explain_message(r: dict, prev: dict | None) -> dict:
    """The WHY? panel (PS-01 §28): evidence list + inferred interpretation."""
    detected, inferred = [], []
    sig = r["signals"]
    bf = r.get("behavior_flags", {})

    # --- detected evidence (measurable, in the text itself) ---
    if sig["neg_hits"]:
        detected.append(f"negative wording: {', '.join(sig['neg_hits'][:3])}")
    if sig["pos_hits"]:
        detected.append(f"positive wording: {', '.join(sig['pos_hits'][:3])}")
    if sig["exclam"]:
        detected.append(f"intensified punctuation ({sig['exclam']}× '!')")
    if sig["ellipsis"]:
        detected.append("trailing ellipsis (unfinished thought)")
    if sig["exaggeration"]:
        detected.append(f"absolute/exaggerated words: {', '.join(sig['exaggeration'][:3])}")
    if sig["emoji_polarity"]:
        detected.append(f"emoji polarity {sig['emoji_polarity']:+.1f}")
    if sig["sarc_words"]:
        detected.append(f"irony-prone markers: {', '.join(sig['sarc_words'][:3])}")
    if sig["pa_phrase"]:
        detected.append(f"concessive phrase: \"{sig['pa_phrase']}\"")
    if sig["swear"]:
        detected.append(f"strong language ({sig['swear']})")
    if bf.get("caps_ratio", 0) > 0.12:
        detected.append("elevated CAPS ratio (shouting-style emphasis)")
    if bf.get("short_response"):
        detected.append("minimal-length response")

    # --- inferred interpretation (model's reading, clearly labeled) ---
    if prev is not None:
        d_t = r["tension"] - prev["tension"]
        if d_t > 8:
            inferred.append(f"tension rose {d_t:+.0f} vs previous turn")
        elif d_t < -8:
            inferred.append(f"tension eased {d_t:+.0f} vs previous turn")
        if prev["emotion"]["label"] != r["emotion"]["label"]:
            inferred.append(f"emotional shift {prev['emotion']['label']} → {r['emotion']['label']}")
        if prev["sentiment"]["label"] != r["sentiment"]["label"]:
            inferred.append(f"sentiment shift {prev['sentiment']['label']} → {r['sentiment']['label']}")
    st = r.get("speaker_state_before") or {}
    if st.get("last_emotion") and st["last_emotion"] in ("frustration", "anger", "sadness"):
        inferred.append(f"speaker's previous state was {st['last_emotion']}")
    if r["sarcasm"]["supporting_signals"]:
        inferred.append("sarcasm evidence: " + "; ".join(r["sarcasm"]["supporting_signals"][:2]))
    if r["passive_aggression"]["supporting_signals"]:
        inferred.append("passive-aggression evidence: " +
                        "; ".join(r["passive_aggression"]["supporting_signals"][:2]))
    if r.get("context_text"):
        inferred.append("interpreted against the preceding conversation window")

    return {
        "message_id": r["message_id"],
        "prediction": {
            "sentiment": {"label": _display(r["sentiment"]["label"]),
                          "confidence": r["sentiment"]["confidence"]},
            "emotion": {"label": _display(r["emotion"]["label"]),
                        "confidence": r["emotion"]["confidence"]},
            "tone": {"label": _display(r["tone"]["label"]),
                     "confidence": r["tone"]["confidence"]},
            "sarcasm_probability": r["sarcasm"]["probability"],
            "passive_aggression_probability": r["passive_aggression"]["probability"],
            "tension": r["tension"],
        },
        "detected_evidence": detected,
        "inferred_interpretation": inferred,
        "model_confidence": r.get("confidence"),
        "disclaimer": "Model-estimated confidence, not human certainty. "
                      "Evidence is detected; interpretation is inferred.",
    }


def what_changed(results: list[dict], at_message: int, window: int = 1) -> dict:
    """The WHAT CHANGED? panel (PS-01 §27): before vs after at a boundary."""
    i = next((k for k, r in enumerate(results) if r["message_id"] == at_message), None)
    if i is None or i == 0:
        return {"error": "message not found or no previous state"}
    before, after = results[i-1], results[i]
    d_tension = round(after["tension"] - before["tension"], 1)
    contributing = []
    sig = after["signals"]
    if sig["neg_hits"]:
        contributing.append("negative wording")
    if sig["exclam"] or sig["swear"]:
        contributing.append("increased linguistic intensity")
    if sig["pa_phrase"] or after["passive_aggression"]["probability"] > .5:
        contributing.append("concessive/cold phrasing")
    if after["sarcasm"]["probability"] > .5:
        contributing.append("sarcasm at the boundary")
    if (after.get("speaker_state_before") or {}).get("last_emotion") in \
            ("frustration", "anger"):
        contributing.append("carry-over of speaker's prior emotional state")
    if (after.get("behavior_flags") or {}).get("short_response"):
        contributing.append("abrupt short-response pattern")
    return {
        "at_message": at_message,
        "before": {"emotion": before["emotion"]["label"], "tone": before["tone"]["label"],
                   "sentiment": before["sentiment"]["label"], "tension": before["tension"]},
        "after": {"emotion": after["emotion"]["label"], "tone": after["tone"]["label"],
                  "sentiment": after["sentiment"]["label"], "tension": after["tension"]},
        "change": {"tension_delta": d_tension,
                   "emotion_shift": (before["emotion"]["label"] + " → " +
                                     after["emotion"]["label"])},
        "potential_contributing_signals": contributing or ["no dominant single signal; cumulative context shift"],
    }


def speaker_profiles(results: list[dict]) -> dict:
    """Per-speaker analysis (PS-01 §29). Distributions, not diagnoses."""
    from collections import Counter, defaultdict
    prof = defaultdict(lambda: {"n": 0, "emotion": Counter(), "tone": Counter(),
                                "sentiment": Counter(), "tension": [],
                                "sarc": 0, "pa": 0, "escalation_msgs": 0})
    for r in results:
        p = prof[r["speaker_id"]]
        p["n"] += 1
        p["emotion"][r["emotion"]["label"]] += 1
        p["tone"][r["tone"]["label"]] += 1
        p["sentiment"][r["sentiment"]["label"]] += 1
        p["tension"].append(r["tension"])
        p["sarc"] += r["sarcasm"]["probability"] >= .5
        p["pa"] += r["passive_aggression"]["probability"] >= .5
        if r["tension"] >= 60:
            p["escalation_msgs"] += 1
    out = {}
    for spk, p in prof.items():
        n = p["n"]
        out[spk] = {
            "messages": n,
            "dominant_emotions": {k: round(v / n, 3) for k, v in p["emotion"].most_common(4)},
            "tone_distribution": {k: round(v / n, 3) for k, v in p["tone"].most_common(4)},
            "sentiment_distribution": {k: round(v / n, 3) for k, v in p["sentiment"].most_common()},
            "avg_tension": round(float(np.mean(p["tension"])), 1),
            "max_tension": round(float(np.max(p["tension"])), 1),
            "sarcastic_share": round(p["sarc"] / n, 3),
            "passive_aggressive_share": round(p["pa"] / n, 3),
            "high_tension_share": round(p["escalation_msgs"] / n, 3),
            "note": "communication-pattern summary, not a psychological profile",
        }
    return out


def build_digest(report: dict) -> dict:
    """Executive conversation digest — aggregates an existing report into the
    headline finding, trajectory reading, and per-speaker risk summary.

    Deterministic and strictly derived from already-computed model outputs:
    it adds no new claims, and per-speaker notes keep the 'patterns, not
    diagnoses' stance of speaker_profiles() (PS-01 §29–30).
    """
    summary = report.get("summary") or {}
    msgs = report.get("messages") or []
    esc = report.get("escalation") or {}
    arc = report.get("emotional_arc") or {}
    tps = report.get("turning_points") or []
    profiles = report.get("speaker_profiles") or {}
    n = len(msgs)

    def _from_to(tp: dict) -> str | None:
        if not tp.get("before") or not tp.get("after"):
            return None
        return f"{tp['before']['emotion']} → {tp['after']['emotion']}"

    # --- headline finding: the single most salient event ---
    headline = None
    if tps:
        strongest = max(tps, key=lambda t: abs(float(t.get("tension_change", 0))))
        if abs(float(strongest.get("tension_change", 0))) >= 5.0:
            headline = {
                "kind": "turning_point",
                "message_id": strongest.get("message_id"),
                "tension_change": strongest.get("tension_change"),
                "direction": ("tension spike" if strongest.get("tension_change", 0) > 0
                              else "tension dip"),
                "from_to": _from_to(strongest),
                "trigger": strongest.get("trigger_text"),
                "speaker": strongest.get("speaker"),
            }
    if headline is None and esc.get("trajectory") == "escalating":
        peak_msg = next((m for m in msgs
                         if m["message_id"] == esc.get("peak_message")), None)
        headline = {
            "kind": "escalation_peak",
            "message_id": esc.get("peak_message"),
            "peak_tension": esc.get("peak_tension"),
            "trigger": peak_msg["text"][:140] if peak_msg else None,
            "speaker": peak_msg.get("speaker_id") if peak_msg else None,
        }

    # --- per-speaker risk summary (aggregated, not diagnostic) ---
    speakers = []
    for spk, p in sorted(profiles.items()):
        speakers.append({
            "speaker_id": spk,
            "messages": p["messages"],
            "avg_tension": p["avg_tension"],
            "max_tension": p["max_tension"],
            "high_tension_share": p["high_tension_share"],
            "sarcastic_share": p["sarcastic_share"],
            "passive_aggressive_share": p["passive_aggressive_share"],
            "dominant_emotions": p["dominant_emotions"],
        })
    flagged = [s for s in speakers
               if s["high_tension_share"] > 0.5 or s["sarcastic_share"] > 0.25]

    # --- trajectory reading ---
    trajectory = {
        "trajectory": esc.get("trajectory"),
        "escalation_rate": esc.get("escalation_rate"),
        "start_middle_end": esc.get("phase_means"),
        "peak_tension": esc.get("peak_tension"),
        "n_turning_points": len(tps),
        "recovered": bool(arc.get("recovered") or esc.get("deescalation_point") is not None),
    }

    # --- templated, deterministic prose ---
    overview = (f"{n} messages, {len(speakers)} participant(s): trajectory is "
                f"{trajectory['trajectory']}, peak tension "
                f"{trajectory['peak_tension']}/100, "
                f"{trajectory['n_turning_points']} turning point(s) detected.")
    if trajectory["recovered"]:
        overview += " The conversation de-escalated after its peak."
    if flagged:
        names = ", ".join(s["speaker_id"] for s in flagged)
        overview += f" Elevated-tension/sarcasm share: {names}."
    headline_text = overview
    if headline:
        headline_text = ("Detected " + headline["direction"] +
                         (f" after message {headline['message_id']}" if headline.get("message_id")
                          else ""))
        if headline.get("from_to"):
            headline_text += f" ({headline['from_to']})"
        if headline.get("peak_tension") is not None:
            headline_text += f" — peak tension {headline['peak_tension']}."
        else:
            headline_text += "."

    return {
        "conversation_id": summary.get("conversation_id"),
        "n_messages": n,
        "headline_finding": headline,
        "trajectory": trajectory,
        "turning_points": [{"message_id": t.get("message_id"),
                            "tension_change": t.get("tension_change"),
                            "from_to": _from_to(t)}
                           for t in tps[:5]],
        "speakers": speakers,
        "text": {"overview": overview, "headline": headline_text},
        "disclaimer": "Aggregated model estimates — associations, not causation. "
                      "See per-message explanations for the underlying evidence.",
    }
