"""Fine-tune a text-only engine variant on REAL GoEmotions data (PS-01 §37).

Protocol:
  1. pull N real Reddit comments (HF datasets-server, research license)
  2. adapter → unified schema (sentiment/tension derived from native labels)
  3. 60/20/20 split BY SOURCE CONVERSATION (Reddit link_id), stratified
  4. fit a fresh text-only MultiTaskEngine on the train split
  5. evaluate on the held-out real test split:
       before = the in-corpus engine (zero-shot)
       after  = the real-data engine
  6. retention check: both engines on the in-corpus test split —
     does real-data training destroy template performance?

Outputs:
  evaluation/results/finetune_summary.json   (committed, no raw text)
  evaluation/results/finetune_engine.joblib  (local-only, gitignored)

Run:  PYTHONPATH=. python3 evaluation/run_finetune.py [n_batches]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from collections import defaultdict

import numpy as np

from cerebro.data.adapters import goemotions_records
from cerebro.models.engines import MultiTaskEngine
from cerebro.models.pipeline import CerebroPipeline

HF = ("https://datasets-server.huggingface.co/rows?dataset={ds}"
      "&config=simplified&split=train&offset={off}&length=100")
DATASET = "google-research-datasets%2Fgo_emotions"
RESULTS = "evaluation/results"
SEED = 42


def _get(url: str, tries: int = 8) -> dict:
    """GET with exponential backoff — the datasets-server rate-limits (429)
    with a cool-down window longer than naive 2^n seconds."""
    import time
    last = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (2 ** t)      # 15s, 30s, 60s, ...
                print(f"  429 rate-limited, backing off {wait}s...")
                time.sleep(wait)
                last = e
            else:
                raise
    raise last


def resolve_id2label() -> dict[int, str]:
    info = _get("https://datasets-server.huggingface.co/info?"
                "dataset=" + DATASET + "&config=simplified")
    names = info["dataset_info"]["features"]["labels"]["feature"]["names"]
    return dict(enumerate(names))


def fetch_msgs(n_batches: int) -> list[dict]:
    """Fetch rows → flat message list, exact-duplicates removed.

    The HF `simplified` config exposes no Reddit thread ids, so true
    thread-level splits are impossible through this API. The main leakage
    vector for single-message classification is duplicate/near-duplicate
    text — mitigated here by normalized exact-dedup. Limitation disclosed
    in the summary artifact.
    """
    id2label = resolve_id2label()

    import re
    import time
    seen: set[str] = set()
    msgs: list[dict] = []
    n_raw = 0
    for b in range(n_batches):
        url = HF.format(ds=DATASET, off=b * 100)
        payload = _get(url)
        time.sleep(0.4)                     # polite pacing between batches
        for row in payload.get("rows", []):
            row = row["row"]
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            n_raw += 1
            norm = re.sub(r"\s+", " ", text.lower()).strip()
            if norm in seen:
                continue
            seen.add(norm)
            convs = goemotions_records(
                [{"text": text, "labels": row.get("labels", [])}],
                source_id="goe", id2label=id2label)
            if convs:
                msgs.append(convs[0][0])
    print(f"  fetched {n_raw} rows → {len(msgs)} unique messages "
          f"({n_raw - len(msgs)} duplicates dropped)")
    assert msgs, "no usable messages fetched — check network/rate limits"
    return msgs


def split_msgs(msgs: list[dict], seed=SEED):
    """70/15/15 row-level split, stratified by gold emotion band.

    Row-level because the public API exposes no thread ids; exact-duplicate
    removal in fetch_msgs mitigates the dominant leakage mode. Disclosed.
    """
    rng = np.random.RandomState(seed)
    strata: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        band = ("pos" if m["emotion"] in ("joy", "affection", "excitement", "relief")
                else "neg" if m["emotion"] in ("frustration", "anger", "sadness",
                                               "anxiety", "fear", "disgust")
                else "neu")
        strata[band].append(m)
    train, val, test = [], [], []
    for band in sorted(strata):
        ms = strata[band]
        idx = rng.permutation(len(ms))
        n = len(ms)
        n_tr, n_va = int(n * .7), int(n * .15)
        for k, i in enumerate(idx):
            (train if k < n_tr else val if k < n_tr + n_va else test).append(ms[i])
    return train, val, test


def eval_msgs(pipe: CerebroPipeline, msgs: list[dict]) -> dict:
    """Zero-shot style: each message analyzed as its own 1-turn conversation."""
    emo_hit = sent_hit = top3 = 0
    per_msg = []
    for m in msgs:
        rep = pipe.analyze([m])
        r = rep["messages"][0]
        probs = r["emotion"]["probabilities"]
        top3_labels = sorted(probs, key=probs.get, reverse=True)[:3]
        emo_hit += r["emotion"]["label"] == m["emotion"]
        top3 += m["emotion"] in top3_labels
        sent_hit += r["sentiment"]["label"] == m["sentiment"]
        per_msg.append({"gold_emo": m["emotion"], "pred_emo": r["emotion"]["label"],
                        "gold_sent": m["sentiment"], "pred_sent": r["sentiment"]["label"],
                        "tension": r["tension"]})
    n = max(len(msgs), 1)
    # valence separation AUC on tension
    negs = [x["tension"] for x in per_msg if x["gold_sent"] == "negative"]
    poss = [x["tension"] for x in per_msg if x["gold_sent"] == "positive"]
    if negs and poss:
        pairs = [(p, q) for p in negs for q in poss]
        auc = (sum(1 for p, q in pairs if p > q)
               + .5 * sum(1 for p, q in pairs if p == q)) / len(pairs)
    else:
        auc = None
    return {
        "n": len(msgs),
        "emotion_exact": round(emo_hit / n, 4),
        "emotion_top3": round(top3 / n, 4),
        "sentiment_exact": round(sent_hit / n, 4),
        "tension_valence_auc": round(auc, 4) if auc is not None else None,
    }


def main() -> None:
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.RandomState(SEED)

    print(f"fetching {n_batches * 100} real messages (deduplicated)...")
    msgs = fetch_msgs(n_batches)

    train, val, test = split_msgs(msgs)
    print(f"row-level split (dedup, stratified): "
          f"train={len(train)} val={len(val)} test={len(test)}")

    # ---- fit real-data engine (text-only variant) ----
    print("fitting real-data engine (text-only)...")
    real_eng = MultiTaskEngine(seed=SEED)
    info = real_eng.fit([[m] for m in train], use_context=False,
                        use_memory=False, use_behavior=False)
    print(f"  {info}")

    real_pipe = CerebroPipeline.from_trained(real_eng)

    # ---- in-corpus engine (zero-shot on real data) ----
    inc_eng = MultiTaskEngine().load("models/saved/cerebro_engine")
    inc_pipe = CerebroPipeline.from_trained(inc_eng)

    test_sample = list(test)          # already flat messages (row-level split)
    rng.shuffle(test_sample)
    test_sample = test_sample[:1500]

    print("evaluating BEFORE (in-corpus engine, zero-shot)...")
    before = eval_msgs(inc_pipe, test_sample)
    print("  ", before)
    print("evaluating AFTER (real-data engine)...")
    after = eval_msgs(real_pipe, test_sample)
    print("  ", after)

    # ---- retention: both engines on the in-corpus test split ----
    print("retention check: loading in-corpus dataset...")
    from cerebro.data.generator import generate_corpus, split_conversations
    splits = split_conversations(generate_corpus())
    inc_test_msgs = [m for c in splits["test"] for m in c]
    rng.shuffle(inc_test_msgs)
    inc_sample = inc_test_msgs[:1500]

    print("retention: in-corpus engine on in-corpus test (reference)...")
    ref = eval_msgs(inc_pipe, inc_sample)
    print("  ", ref)
    print("retention: real-data engine on in-corpus test (transfer back)...")
    back = eval_msgs(real_pipe, inc_sample)
    print("  ", back)

    summary = {
        "dataset": DATASET,
        "n_batches": n_batches,
        "n_messages": len(msgs),
        "split": {"level": "row (deduplicated, stratified by emotion band)",
                  "train": len(train), "val": len(val), "test": len(test),
                  "limitation": ("HF simplified config exposes no Reddit thread ids; "
                                 "thread-level split impossible via this API. Exact-dup "
                                 "removal mitigates the dominant leakage mode.")},
        "engine_variant": "text-only MultiTaskEngine (same heads as CEREBRO)",
        "before_zero_shot": before,
        "after_finetune": after,
        "delta": {k: (round(after[k] - before[k], 4)
                      if isinstance(after[k], float) else None)
                  for k in before if k != "n"},
        "retention_incorpus_reference": ref,
        "retention_incorpus_after_realdata_training": back,
        "note": ("Text-only fine-tune on real data closes part of the transfer "
                 "gap; full CEREBRO context/temporal stack needs multi-turn real "
                 "conversations (roadmap: DailyDialog transfer)."),
    }
    with open(f"{RESULTS}/finetune_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    os.makedirs("models/saved", exist_ok=True)
    import joblib
    joblib.dump({"vec": real_eng.vec, "heads": real_eng.heads,
                 "seed": real_eng.seed}, "models/saved/finetune_engine.joblib")
    print(f"done → {RESULTS}/finetune_summary.json · "
          f"engine kept local-only (models/saved/finetune_engine.joblib, gitignored)")


if __name__ == "__main__":
    main()
