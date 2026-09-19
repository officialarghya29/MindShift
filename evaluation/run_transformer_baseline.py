"""Pretrained-transformer baselines B2 / B3 (blueprint §9), evaluated end-to-end.

  B2  frozen MiniLM embeddings (utterance only) + task heads
  B3  frozen MiniLM embeddings (utterance ⊕ mean history) + task heads

Run on two corpora so the result is interpretable:

  * the **context-dependence probe** — here the conclusion is the headline:
    a strong pretrained encoder still cannot beat the lexical ceiling without
    context, which is precisely the blueprint §3 claim;
  * the **main corpus test set** — where the template design makes every method
    saturate, reported only so the comparison is complete.

Frozen weights, no fine-tuning: §41 requires the innovation to come from context,
temporal reasoning and fusion, not from pre-training. Weights are fetched once to
`models/cache/minilm/` and reused.

Run:  python -m evaluation.run_transformer_baseline
"""
from __future__ import annotations

import time

import numpy as np

from cerebro.common.io import save_json
from cerebro.common.metrics import (calibration_metrics, classification_metrics,
                                    probability_metrics)
from cerebro.context.context_engine import ContextWindow
from cerebro.data.context_probes import build_probe_corpus, split_probes
from cerebro.data.generator import generate_corpus, split_conversations
from cerebro.models.transformer_baseline import MiniLMEmbedder, fit_transformer_heads

RESULTS_DIR = "evaluation/results"
SEED = 42
HEADS_CLASS = ["sentiment", "emotion", "tone"]
HEADS_BINARY = ["sarcasm", "irony", "passive_aggression"]


def rows_from_convs(convs):
    """Flatten conversations → (text, context_text, gold, is_probe)."""
    texts, ctxs, gold, is_probe = [], [], [], []
    for c in convs:
        cw = ContextWindow()
        for i, m in enumerate(c):
            texts.append(m["text"])
            ctxs.append(cw.context_text(c, i) or "")
            gold.append({h: m[h] for h in
                         HEADS_CLASS + HEADS_BINARY + ["tension"]})
            is_probe.append(bool(m.get("is_probe")))
    return texts, ctxs, gold, is_probe


def score_heads(heads, X, gold, is_probe, only_probe: bool) -> dict:
    idx = [i for i, p in enumerate(is_probe) if (p or not only_probe)]
    y_true = {h: [] for h in HEADS_CLASS + HEADS_BINARY + ["tension"]}
    y_pred = {h: [] for h in y_true}
    p_proba = {h: [] for h in HEADS_BINARY}
    for i in idx:
        g = gold[i]
        for head in HEADS_CLASS:
            y_true[head].append(str(g[head]))
            y_pred[head].append(str(heads[head].predict(X[i:i + 1])[0]))
        for head in HEADS_BINARY:
            y_true[head].append(int(g[head]))
            y_pred[head].append(int(heads[head].predict(X[i:i + 1])[0]))
            p_proba[head].append(float(heads[head].predict_proba(X[i:i + 1])[0][1]))
        y_true["tension"].append(float(g["tension"]))
        y_pred["tension"].append(
            float(np.clip(heads["tension"].predict(X[i:i + 1])[0], 0, 100)))
    out = {}
    for head in HEADS_CLASS:
        out[head] = classification_metrics(y_true[head], y_pred[head])
    for head in HEADS_BINARY:
        out[head] = classification_metrics(
            [int(v) for v in y_true[head]], [int(v) for v in y_pred[head]])
        p = np.asarray(p_proba[head], float)
        yb = np.asarray(y_true[head], int)
        if len(np.unique(yb)) == 2:
            out[head].update(probability_metrics(yb, p))
        out[head]["ece"] = calibration_metrics(yb, np.clip(p, 1e-9, 1 - 1e-9))["ece"]
    err = np.abs(np.asarray(y_pred["tension"]) - np.asarray(y_true["tension"]))
    out["tension"] = {"mae": round(float(err.mean()), 3),
                      "rmse": round(float(np.sqrt((err ** 2).mean())), 3)}
    out["n_scored"] = len(idx)
    return out


def probe_experiment(embedder) -> dict:
    corpus = build_probe_corpus()
    splits = split_probes(corpus)
    train = splits["train"]

    t0 = time.time()
    tr_text, tr_ctx, tr_gold, tr_probe = rows_from_convs(train)
    print(f"  encoding {len(tr_text)} train rows (utterance + history) …")
    Xtr_u = embedder.encode(tr_text)
    Xtr_c = embedder.encode([t if t else " " for t in tr_ctx])
    print(f"  encoded in {time.time() - t0:.1f}s")

    heads_b2 = fit_transformer_heads(Xtr_u, tr_gold, seed=SEED)
    heads_b3 = fit_transformer_heads(np.hstack([Xtr_u, Xtr_c]), tr_gold, seed=SEED)

    out = {}
    for regime in ["test", "test_seen"]:
        convs = splits[regime]
        te_text, te_ctx, te_gold, te_probe = rows_from_convs(convs)
        Xte_u = embedder.encode(te_text)
        Xte_c = embedder.encode([t if t else " " for t in te_ctx])
        only = bool(te_probe and not all(te_probe))
        out[regime] = {
            "n_scored": sum(1 for p in te_probe if p) if only else len(te_probe),
            "B2_transformer_text_only": score_heads(heads_b2, Xte_u, te_gold, te_probe, only),
            "B3_transformer_plus_context": score_heads(
                heads_b3, np.hstack([Xte_u, Xte_c]), te_gold, te_probe, only),
        }
    return out


def main_corpus_experiment(embedder) -> dict:
    splits = split_conversations(generate_corpus(convs_per_cell=14))
    tr_text, tr_ctx, tr_gold, tr_probe = rows_from_convs(splits["train"])
    print(f"  encoding {len(tr_text)} main-corpus train rows …")
    Xtr_u = embedder.encode(tr_text)
    Xtr_c = embedder.encode([t if t else " " for t in tr_ctx])
    heads_b2 = fit_transformer_heads(Xtr_u, tr_gold, seed=SEED)
    heads_b3 = fit_transformer_heads(np.hstack([Xtr_u, Xtr_c]), tr_gold, seed=SEED)

    te_text, te_ctx, te_gold, te_probe = rows_from_convs(splits["test"])
    Xte_u = embedder.encode(te_text)
    Xte_c = embedder.encode([t if t else " " for t in te_ctx])
    return {
        "n_train": len(tr_text),
        "n_test": len(te_text),
        "B2_transformer_text_only": score_heads(heads_b2, Xte_u, te_gold, te_probe, False),
        "B3_transformer_plus_context": score_heads(
            heads_b3, np.hstack([Xte_u, Xte_c]), te_gold, te_probe, False),
    }


def run():
    t0 = time.time()
    print("loading frozen MiniLM (onnxruntime) …")
    embedder = MiniLMEmbedder(verbose=True).load()
    print("probe corpus:")
    probes = probe_experiment(embedder)
    print("main corpus (template-saturated; reported for completeness):")
    main = main_corpus_experiment(embedder)

    payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2 (frozen, ONNX, CPU)",
        "runtime": "onnxruntime + stdlib BERT WordPiece tokenizer (no new dependency)",
        "fine_tuned": False,
        "probe_experiment": probes,
        "main_corpus_experiment": main,
        "wall_time_seconds": round(time.time() - t0, 1),
        "note": ("Frozen encoder + the same head set as CEREBRO, so the comparison "
                 "isolates context rather than capacity. B2 looks only at the utterance; "
                 "B3 adds the mean-pooled history embedding (§11)."),
    }
    save_json(payload, f"{RESULTS_DIR}/transformer_baselines.json")
    print(f"\nsaved → {RESULTS_DIR}/transformer_baselines.json in {payload['wall_time_seconds']}s")
    return payload


if __name__ == "__main__":
    run()
