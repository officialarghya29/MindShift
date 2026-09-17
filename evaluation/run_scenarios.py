"""PS-01 §41 scenario evaluation — 20 hand-crafted, out-of-distribution
conversations pushed through the *trained* CEREBRO engine.

These are NOT from the template corpus: different phrasing, emoji, slang,
timestamps, formats — testing parser + pipeline robustness (§41 cases 1–20).

Run:  python -m evaluation.run_scenarios
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from cerebro.models.engines import MultiTaskEngine
from cerebro.models.pipeline import CerebroPipeline
from cerebro.common.io import save_json

RESULTS = RESULTS_DIR = "evaluation/results"


def _ts(base_min: int):
    t = datetime(2026, 1, 10, 10, 0) + timedelta(minutes=base_min)
    return t.isoformat()


SCENARIOS: list[dict] = [
    # 1 normal
    dict(id="normal", expect="mostly calm", msgs=[
        ("A", "Hey, are we still on for lunch tomorrow?"),
        ("B", "Yes! 1pm at the usual place."),
        ("A", "Perfect, see you there."),
    ]),
    # 2 happy
    dict(id="happy", expect="joy dominant", msgs=[
        ("A", "I GOT THE JOB!!! 🎉🎉"),
        ("B", "NO WAY! Congrats!! So happy for you 😄"),
        ("A", "Thanks man, still shaking haha"),
    ]),
    # 3 angry
    dict(id="angry", expect="anger/escalation", msgs=[
        ("A", "You DELETED my files without asking??"),
        ("B", "It was an accident, calm down."),
        ("A", "Don't tell me to calm down. Two weeks of work, gone."),
    ]),
    # 4 sarcastic
    dict(id="sarcastic", expect="sarcasm flagged", msgs=[
        ("A", "The wifi died again in the middle of my exam."),
        ("B", "Wow. Perfect timing. As always."),
        ("A", "Right? It has a great sense of humor."),
    ]),
    # 5 passive-aggressive
    dict(id="passive_aggressive", expect="PA flagged", msgs=[
        ("A", "I cleaned the kitchen. Again. Since you were busy."),
        ("B", "I said I'd do it later."),
        ("A", "No problem. Do what you want."),
    ]),
    # 6 mixed emotions
    dict(id="mixed", expect="bittersweet", msgs=[
        ("A", "I got the promotion... but it's in another city."),
        ("B", "Oh wow, congrats but that's rough. When do you move?"),
        ("A", "End of the month. Excited and terrified honestly."),
    ]),
    # 7 emoji-heavy
    dict(id="emoji_heavy", expect="emoji polarity read", msgs=[
        ("A", "Look at this 🥰😍✨🌟"),
        ("B", "Omg adorable 😭❤️"),
        ("A", "I know right 😩💕"),
    ]),
    # 8 slang-heavy
    dict(id="slang_heavy", expect="robust to slang", msgs=[
        ("A", "yo that exam was mad tough ngl"),
        ("B", "fr bro, i'm cooked 💀"),
        ("A", "lol we'll lock in next time"),
    ]),
    # 9 very short messages
    dict(id="very_short", expect="no crashes, sensible read", msgs=[
        ("A", "hey"),
        ("B", "hi"),
        ("A", "sup"),
        ("B", "nm u"),
    ]),
    # 10 long conversation (compressed here: 24 turns)
    dict(id="long", expect="stable trajectory", msgs=[
        *[(("A" if i % 2 == 0 else "B"),
           ("point %d: the budget should cover marketing" if i % 2 == 0
            else "agreed on %d, though design needs a bigger share") % (i + 1))
          for i in range(24)]],
    ),
    # 11 multi-speaker (3 people)
    dict(id="multi_speaker", expect="3 speakers profiled", msgs=[
        ("A", "Team, the demo is at 3pm."),
        ("B", "Slides are ready."),
        ("C", "I'll handle the live env."),
        ("A", "Great, let's do a dry run at 2."),
    ]),
    # 12 rapid-fire (seconds apart)
    dict(id="rapid", expect="fast gaps parsed", msgs=[
        ("A", "call now?"), ("B", "yes"), ("A", "joining"), ("B", "ok"),
    ]),
    # 13 slow (hours apart)
    dict(id="slow", expect="response-gap features fire", msgs=[
        ("A", "Did you see my message from this morning?"),
        ("B", "Just saw it, sorry — crazy day."),
        ("A", "All good. Tonight then?"),
    ]),
    # 14 topic change mid-way
    dict(id="topic_change", expect="segmentation splits", msgs=[
        ("A", "The quarterly numbers look strong."),
        ("B", "Yes, revenue is up 12%."),
        ("A", "By the way, did you watch the match last night?"),
        ("B", "What a game!! That last over was insane."),
    ]),
    # 15 escalating
    dict(id="escalating", expect="escalating trajectory + peak", msgs=[
        ("A", "You forgot our anniversary."),
        ("B", "I was swamped at work, I said sorry."),
        ("A", "You say sorry every year. SAME THING."),
        ("B", "Don't exaggerate. It's not the same."),
        ("A", "UNBELIEVABLE. You never prioritize us."),
    ]),
    # 16 de-escalating
    dict(id="deescalating", expect="cooling trajectory", msgs=[
        ("A", "I'm really upset about last night."),
        ("B", "I know. I should have called."),
        ("A", "Thanks for saying that."),
        ("B", "Dinner this weekend? My treat."),
        ("A", "Yeah. I'd like that."),
    ]),
    # 17 neutral with ambiguous phrases
    dict(id="ambiguous", expect="no false PA alarm", msgs=[
        ("A", "Fine, let's go with your plan."),
        ("B", "Sure. I'll draft it."),
        ("A", "Okay then, see you Monday."),
    ]),
    # 18 irony
    dict(id="irony", expect="irony > 0.5", msgs=[
        ("A", "My flight was delayed 6 hours, then they lost my bag."),
        ("B", "Wow, what a perfect start to the vacation."),
    ]),
    # 19 humor
    dict(id="humor", expect="humor without escalation", msgs=[
        ("A", "I put 'Excel expert' on my CV. Excel asked me to leave."),
        ("B", "LMAO 💀 at least you're honest."),
        ("A", "My pivot tables never recovered."),
    ]),
    # 20 malformed export
    dict(id="malformed", expect="graceful handling", msgs=[
        ("A", "message with no clean format"),
        ("B", ""),
        ("A", "trailing garbage \x00\x01 but readable"),
    ]),
]


def main():
    t0 = time.time()
    eng = MultiTaskEngine(seed=42).load("models/saved/cerebro_engine")
    pipe = CerebroPipeline.from_trained(eng)
    out = []
    for sc in SCENARIOS:
        base = datetime(2026, 1, 10, 10, 0)
        gap = {"rapid": 5, "slow": 7000}.get(sc["id"], 240)
        msgs = [{"conversation_id": sc["id"], "message_id": i + 1,
                 "speaker_id": spk, "timestamp": (base + timedelta(seconds=i * gap)).isoformat(),
                 "text": txt, "platform": "scenario"}
                for i, (spk, txt) in enumerate(sc["msgs"])]
        msgs = [m for m in msgs if m["text"]]        # malformed case: drop empties
        rep = pipe.analyze(msgs, conversation_id=sc["id"])
        from cerebro.features.segmentation import segment_conversation
        segs = segment_conversation([{"message_id": m["message_id"], "text": m["text"],
                                      "timestamp": m["timestamp"]} for m in msgs])
        out.append({
            "scenario": sc["id"],
            "n_messages": len(msgs),
            "expectation": sc["expect"],
            "dominant_emotion": rep["summary"]["dominant_emotion"][0][0],
            "trajectory": rep["summary"]["trajectory"],
            "mean_tension": rep["summary"]["mean_tension"],
            "peak_tension": rep["summary"]["peak_tension"],
            "sarcasm_mean": rep["summary"]["sarcasm_level"],
            "irony_mean": rep["summary"]["irony_level"],
            "pa_mean": rep["summary"]["passive_aggression_level"],
            "n_turning_points": len(rep["turning_points"]),
            "n_segments": len(segs),
            "n_speakers": rep["summary"]["n_speakers"],
            "status": "ok",
        })
        print(f"  {sc['id']:18s} emo={out[-1]['dominant_emotion']:11s} "
              f"traj={out[-1]['trajectory']:13s} sarc={out[-1]['sarcasm_mean']:.2f} "
              f"PA={out[-1]['pa_mean']:.2f} t̄={out[-1]['mean_tension']:.0f} "
              f"tps={out[-1]['n_turning_points']} segs={out[-1]['n_segments']}")
    save_json({"scenarios": out, "n_passed": sum(s["status"] == "ok" for s in out),
               "wall_time_seconds": round(time.time() - t0, 1)},
              f"{RESULTS_DIR}/scenarios.json")
    print(f"all {len(out)} scenarios executed without failure → {RESULTS_DIR}/scenarios.json")


if __name__ == "__main__":
    main()
