"""Zero-shot transfer evaluation on REAL GoEmotions data (PS-01 §3, §37).

Pulls the official Google Research GoEmotions corpus (Reddit comments,
research license — see docs/dataset/DATASET_CARD.md) through the HuggingFace
datasets-server, converts it with the public-dataset adapter, and evaluates
the trained CEREBRO engine against its native emotion labels.

This is an out-of-distribution stress test: the training corpus is template-
composed; these are real, messy human comments. Published metrics are the
honest cross-corpus numbers, not the saturated in-corpus ones.

Outputs:
  evaluation/results/transfer_goemotions.json
  evaluation/results/transfer_logsafe.json   (graph-safe, no raw text)

Run:  PYTHONPATH=. python3 evaluation/run_transfer.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from collections import Counter

import numpy as np

from cerebro.data.adapters import goemotions_records
from cerebro.models.engines import MultiTaskEngine
from cerebro.models.pipeline import CerebroPipeline

HF = ("https://datasets-server.huggingface.co/rows?dataset={ds}"
      "&config=simplified&split=train&offset={off}&length=100")
DATASET = "google-research-datasets%2Fgo_emotions"
N_BATCHES = 30          # 30 × 100 = 3,000 real messages
RESULTS = "evaluation/results"


def fetch_batches() -> list[dict]:
    rows = []
    for b in range(N_BATCHES):
        url = HF.format(ds=DATASET, off=b * 100)
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read().decode("utf-8"))
        rows.extend(r["row"] for r in payload.get("rows", []))
    return rows


def resolve_id2label() -> dict[int, str]:
    url = ("https://datasets-server.huggingface.co/info?"
           "dataset=" + DATASET + "&config=simplified")
    with urllib.request.urlopen(url, timeout=30) as r:
        info = json.loads(r.read().decode("utf-8"))
    names = info["dataset_info"]["features"]["labels"]["feature"]["names"]
    return dict(enumerate(names))


def main() -> dict:
    print("resolving GoEmotions label layout from HF datasets-server...")
    id2label = resolve_id2label()
    print(f"  {len(id2label)} classes resolved")

    print(f"fetching {N_BATCHES * 100} real messages...")
    rows = fetch_batches()
    print(f"  {len(rows)} rows received")

    # convert with the native-label priority map (adapter does the collapsing)
    recs = []
    for i, row in enumerate(rows):
        convs = goemotions_records([{"text": row.get("text", ""),
                                     "labels": row.get("labels", [])}],
                                   source_id="goe", id2label=id2label)
        recs.extend(convs)
    msgs = [c[0] for c in recs]
    print(f"  {len(msgs)} usable messages after conversion")

    engine = MultiTaskEngine().load("models/saved/cerebro_engine")
    pipe = CerebroPipeline.from_trained(engine)

    per_msg = []
    top3 = 0
    for m in msgs:
        rep = pipe.analyze([dict(m)])
        r = rep["messages"][0]
        probs = r["emotion"]["probabilities"]
        top3_labels = sorted(probs, key=probs.get, reverse=True)[:3]
        if m["emotion"] in top3_labels:
            top3 += 1
        per_msg.append({
            "text": m["text"],
            "gold_emotion": m["emotion"],
            "gold_sentiment": m["sentiment"],
            "pred_emotion": r["emotion"]["label"],
            "pred_sentiment": r["sentiment"]["label"],
            "pred_tone": r["tone"]["label"],
            "sarcasm_p": r["sarcasm"]["probability"],
            "irony_p": r["irony"]["probability"],
            "pa_p": r["passive_aggression"]["probability"],
            "tension": r["tension"],
        })

    n = len(per_msg)
    emo_correct = sum(1 for x in per_msg if x["gold_emotion"] == x["pred_emotion"])
    sent_correct = sum(1 for x in per_msg if x["gold_sentiment"] == x["pred_sentiment"])

    # valence separation: predicted tension ranks negative-valence gold messages
    # above positive-valence ones (Mann–Whitney AUC)
    pos_scores = [x["tension"] for x in per_msg if x["gold_sentiment"] == "negative"]
    neg_scores = [x["tension"] for x in per_msg if x["gold_sentiment"] == "positive"]
    if pos_scores and neg_scores:
        pairs = [(p, q) for p in pos_scores for q in neg_scores]
        sep = sum(1 for p, q in pairs if p > q) + 0.5 * sum(1 for p, q in pairs if p == q)
        tension_auc = sep / len(pairs)
    else:
        tension_auc = None

    summary = {
        "dataset": "google-research-datasets/go_emotions (simplified), research license",
        "n_messages": n,
        "source": "real Reddit comments via HF datasets-server (batches 0-29)",
        "zero_shot_emotion_accuracy": round(emo_correct / n, 4),
        "zero_shot_emotion_top3": round(top3 / n, 4),
        "zero_shot_sentiment_accuracy": round(sent_correct / n, 4),
        "tension_separates_valence_auc": round(tension_auc, 4) if tension_auc else None,
        "pred_emotion_distribution": dict(Counter(x["pred_emotion"] for x in per_msg).most_common()),
        "gold_emotion_distribution": dict(Counter(x["gold_emotion"] for x in per_msg).most_common()),
        "mean_sarcasm_p": round(float(np.mean([x["sarcasm_p"] for x in per_msg])), 4),
        "mean_irony_p": round(float(np.mean([x["irony_p"] for x in per_msg])), 4),
        "mean_pa_p": round(float(np.mean([x["pa_p"] for x in per_msg])), 4),
        "mean_tension": round(float(np.mean([x["tension"] for x in per_msg])), 2),
        "note": ("Cross-corpus zero-shot numbers on real human text; not comparable "
                 "to the in-corpus table. Publishing honest transfer metrics is the "
                 "PS-01 §39 spirit: measure, disclose, fix."),
    }

    # confusion: gold × pred top pairs
    confusion = Counter((x["gold_emotion"], x["pred_emotion"]) for x in per_msg)
    summary["top_confusions"] = [
        {"gold": g, "pred": p, "n": c} for (g, p), c in confusion.most_common(8)
    ]

    with open(f"{RESULTS}/transfer_goemotions.json", "w") as f:
        json.dump({"summary": summary, "per_message": per_msg}, f, indent=1)
    # log-safe graph payload: distributions + metrics only, no raw text
    with open(f"{RESULTS}/transfer_logsafe.json", "w") as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(summary, indent=1)[:1400])
    print(f"done → {RESULTS}/transfer_goemotions.json")
    return summary


if __name__ == "__main__":
    main()
