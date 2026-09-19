"""Controlled proof of the blueprint's core claim (PS-01 §3, §36).

The in-corpus ablation cannot show a context benefit: in a corpus where every
utterance carries a fixed label, a text-only model is already Bayes-optimal. This
runner performs the complementary *controlled* experiment on the probe corpus
(`cerebro.data.context_probes`), where the text is deliberately uninformative and
only the history disambiguates it.

Protocol
--------
  1. fit the vectorizer on probe-corpus TRAIN text
  2. build the wide [text ⊕ context ⊕ behavior ⊕ memory] design matrix on TRAIN
     (teacher-forced history), slice per ablation variant (§36 A/B/C/D)
  3. evaluate sequentially on TEST with predicted history (deployment-faithful)
  4. variant E = engine D + hidden-signal fusion (full CEREBRO)
  5. report two references so the lift is interpretable:
       * text-only lexical ceiling — the majority label per probe utterance, i.e.
         the best any context-free model can do on these probes
       * condition oracle — knowing the history's polarity, which is 1.0 by
         construction and therefore the ceiling context *can* reach
  6. significance: exact McNemar (A vs E) per head + bootstrap 95% CI on the
     paired accuracy delta

Run:  python -m evaluation.run_context_proof
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import binomtest

from cerebro.common.io import save_json
from cerebro.common.metrics import classification_metrics
from cerebro.context.context_engine import ContextWindow
from cerebro.context.features_builder import _memory_vector
from cerebro.context.speaker_memory import SpeakerMemory
from cerebro.data.context_probes import (ALL_PROBES, CONTEXT_DEPENDENT_PROBES,
                                         build_probe_corpus, probe_stats,
                                         split_probes)
from cerebro.data.generator import flatten
from cerebro.features.featurizer import build_vectorizer
from cerebro.features.preprocess import behavioral_vector, process_text
from cerebro.models.engines import MultiTaskEngine
from cerebro.models.hidden_signals import apply_hidden_signals
# the design-matrix / head-fitting protocol is shared with the main evaluation so
# the two experiments are directly comparable, not merely similar
from evaluation.run_full import build_wide_matrices, train_variant_heads

RESULTS_DIR = "evaluation/results"
SEED = 42
HEAD_NAMES = ["sentiment", "emotion", "tone", "sarcasm", "irony",
              "passive_aggression", "tension"]
N_BOOT = 2000


# ---------------------------------------------------------------------------
# text-only lexical ceiling: the best a context-free model can possibly do
# ---------------------------------------------------------------------------
def text_only_ceiling(train_convs, test_convs) -> dict:
    """Predict, for each probe utterance, its majority label in TRAIN.

    Because the probe design balances every utterance across its gold labels,
    this ceiling sits at the majority-class rate — no lexical model can beat it.
    """
    from collections import Counter
    per_utt = {h: {} for h in HEAD_NAMES}
    for c in train_convs:
        p = c[-1]
        for h in HEAD_NAMES:
            per_utt[h].setdefault(p["probe_id"], Counter())[str(p[h])] += 1
    best = {h: {u: cnt.most_common(1)[0][0] for u, cnt in d.items()}
            for h, d in per_utt.items()}

    y_true = {h: [] for h in HEAD_NAMES}
    y_pred = {h: [] for h in HEAD_NAMES}
    for c in test_convs:
        p = c[-1]
        for h in HEAD_NAMES:
            y_true[h].append(str(p[h]))
            y_pred[h].append(best[h].get(p["probe_id"], "neutral"))
    out = {}
    for h in HEAD_NAMES:
        m = classification_metrics(y_true[h], y_pred[h])
        if h == "tension":           # categorical proxy is meaningless for a score
            m = {"note": "continuous target — lexical ceiling not applicable"}
        out[h] = m
    return out


# ---------------------------------------------------------------------------
# sequential probe-turn evaluation (A/B/C/D)
# ---------------------------------------------------------------------------
def eval_probe_variant(heads_variant, vec, convs, variant, n_text):
    """Score only the flagged probe turn of each conversation."""
    y_true = {h: [] for h in HEAD_NAMES}
    y_pred = {h: [] for h in HEAD_NAMES}
    correct = {h: [] for h in ["sentiment", "emotion", "tone"]}
    latencies = []

    for c in convs:
        cw = ContextWindow()
        mem = SpeakerMemory()
        for i, m in enumerate(c):
            prev = c[i - 1] if i else None
            behav = np.asarray(behavioral_vector(
                m["text"], prev["timestamp"] if prev else None, m.get("timestamp"),
                prev["speaker_id"] if prev else None, m["speaker_id"]), dtype=float)
            t0 = time.perf_counter()
            text_part = vec.transform([process_text(m["text"])])
            ctx = cw.context_text(c, i)
            ctx_part = (vec.transform([process_text(ctx)]) if ctx
                        else csr_matrix((1, n_text)))
            if variant == "A":
                X = text_part
            elif variant == "B":
                X = hstack([text_part, ctx_part]).tocsr()
            elif variant == "C":
                mv = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
                X = hstack([text_part, ctx_part, csr_matrix(mv)]).tocsr()
            else:
                mv = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
                X = hstack([text_part, ctx_part,
                            csr_matrix(behav.reshape(1, -1)),
                            csr_matrix(mv)]).tocsr()

            pred = {}
            for head in ["sentiment", "emotion", "tone"]:
                p = heads_variant[head].predict_proba(X)[0]
                cls = list(heads_variant[head].classes_)
                pred[head] = str(cls[int(np.argmax(p))])
            pred["tension"] = float(np.clip(heads_variant["tension"].predict(X)[0], 0, 100))
            for head in ["sarcasm", "irony", "passive_aggression"]:
                pred[head] = str(int(float(heads_variant[head].predict_proba(X)[0][1]) >= .5))
            latencies.append(time.perf_counter() - t0)

            mem.observe(m["speaker_id"], pred["emotion"], pred["tone"],
                        pred["sentiment"], pred["tension"], bool(behav[14]))

            if not m.get("is_probe"):
                continue
            for head in HEAD_NAMES:
                gt = float(m[head]) if head == "tension" else str(m[head])
                y_true[head].append(gt)
                y_pred[head].append(pred[head])
            for head in ["sentiment", "emotion", "tone"]:
                correct[head].append(int(str(m[head]) == pred[head]))

    out = {}
    for head in HEAD_NAMES:
        if head == "tension":
            err = np.abs(np.asarray(y_pred[head], float) - np.asarray(y_true[head], float))
            out[head] = {"mae": round(float(err.mean()), 3),
                         "rmse": round(float(np.sqrt((err ** 2).mean())), 3)}
        else:
            out[head] = classification_metrics(y_true[head], y_pred[head])
    out["_correct"] = correct
    out["latency_ms"] = round(float(np.mean(latencies)) * 1000, 3)
    return out


def eval_probe_full(engine, convs):
    """Variant E: engine D + hidden-signal fusion."""
    y_true = {h: [] for h in HEAD_NAMES}
    y_pred = {h: [] for h in HEAD_NAMES}
    correct = {h: [] for h in ["sentiment", "emotion", "tone"]}
    for c in convs:
        results = engine.predict_conversation(c)
        apply_hidden_signals(results)
        for r, m in zip(results, c):
            if not m.get("is_probe"):
                continue
            for head in ["sentiment", "emotion", "tone"]:
                y_true[head].append(str(m[head]))
                y_pred[head].append(r[head]["label"])
                correct[head].append(int(str(m[head]) == r[head]["label"]))
            for head in ["sarcasm", "irony", "passive_aggression"]:
                y_true[head].append(str(int(m[head])))
                y_pred[head].append(str(int(r[head]["probability"] >= .5)))
            y_true["tension"].append(float(m["tension"]))
            y_pred["tension"].append(float(r["tension"]))
    out = {}
    for head in HEAD_NAMES:
        if head == "tension":
            err = np.abs(np.asarray(y_pred[head], float) - np.asarray(y_true[head], float))
            out[head] = {"mae": round(float(err.mean()), 3),
                         "rmse": round(float(np.sqrt((err ** 2).mean())), 3)}
        else:
            out[head] = classification_metrics(y_true[head], y_pred[head])
    out["_correct"] = correct
    return out


# ---------------------------------------------------------------------------
# significance
# ---------------------------------------------------------------------------
def mcnemar_exact(a_correct, b_correct) -> dict:
    a = np.asarray(a_correct, int)
    b = np.asarray(b_correct, int)
    b_only = int(((a == 0) & (b == 1)).sum())   # A wrong, B right
    c_only = int(((a == 1) & (b == 0)).sum())   # A right, B wrong
    n = b_only + c_only
    p = 1.0 if n == 0 else float(binomtest(min(b_only, c_only), n, 0.5).pvalue * 2)
    return {"a_wrong_b_right": b_only, "a_right_b_wrong": c_only,
            "discordant": n, "p_value": round(min(1.0, p), 6)}


def bootstrap_delta(a_correct, b_correct, n_boot: int = N_BOOT) -> dict:
    a = np.asarray(a_correct, float)
    b = np.asarray(b_correct, float)
    rng = np.random.default_rng(SEED)
    n = len(a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = b[idx].mean() - a[idx].mean()
    return {"delta": round(float(b.mean() - a.mean()), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                     round(float(np.percentile(deltas, 97.5)), 4)],
            "n_boot": n_boot}


def flip_rates(train_convs) -> dict:
    """Share of probe utterances whose gold label differs between conditions —
    the share on which context is not merely helpful but *required*."""
    from collections import defaultdict
    per_utt = defaultdict(dict)
    for c in train_convs:
        p = c[-1]
        per_utt[p["probe_id"]][p["probe_condition"]] = p
    rates = {}
    for head in ["sentiment", "emotion", "tone", "sarcasm", "irony",
                 "passive_aggression"]:
        n_flip = sum(1 for u, d in per_utt.items()
                     if len(d) == 2 and str(d["benign"][head]) != str(d["tense"][head]))
        rates[head] = round(n_flip / len(per_utt), 4)
    rates["n_utterances"] = len(per_utt)
    rates["context_dependent_utterances"] = len(CONTEXT_DEPENDENT_PROBES)
    rates["control_utterances"] = len(ALL_PROBES) - len(CONTEXT_DEPENDENT_PROBES)
    return rates


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _public(variants):
    out = {}
    for v, m in variants.items():
        out[v] = {h: m[h] for h in m if h != "_correct"}
    return out


def run():
    t0 = time.time()
    corpus = build_probe_corpus()
    splits = split_probes(corpus)
    stats = probe_stats(corpus)
    print(f"probe corpus: {len(corpus)} conversations (train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])} "
          f"test_seen={len(splits['test_seen'])})")

    msgs_train = flatten(splits["train"])
    vec = build_vectorizer([m["text"] for m in msgs_train])
    print("building wide design matrix + fitting A/B/C/D...")
    X_wide, _ = build_wide_matrices(splits["train"], vec)
    n_text = len(vec.get_feature_names_out())
    heads = train_variant_heads(X_wide, msgs_train, ["A", "B", "C", "D"], n_text)

    engine = MultiTaskEngine(seed=SEED)
    engine.vec = vec
    engine.heads = heads["D"]
    engine.trained = True

    regimes = {"unseen_history_wording": splits["test"],
               "seen_history_wording": splits["test_seen"]}
    per_regime = {}
    for regime, convs in regimes.items():
        print(f"\n-- test regime: {regime} ({len(convs)} probe turns)")
        variants = {}
        for v in ["A", "B", "C", "D"]:
            variants[v] = eval_probe_variant(heads[v], vec, convs, v, n_text)
        variants["E"] = eval_probe_full(engine, convs)
        for v in ["A", "B", "C", "D", "E"]:
            m = variants[v]
            print(f"   {v}: sentiment={m['sentiment']['accuracy']:<7} "
                  f"emotion={m['emotion']['accuracy']:<7} "
                  f"tone={m['tone']['accuracy']:<7} "
                  f"pa-f1={m['passive_aggression']['f1_macro']:<7} "
                  f"tension-mae={m['tension']['mae']}")
        ceiling = text_only_ceiling(splits["train"], convs)
        print(f"   ceiling: sentiment={ceiling['sentiment']['accuracy']:<7} "
              f"emotion={ceiling['emotion']['accuracy']:<7} "
              f"tone={ceiling['tone']['accuracy']}")
        significance = {}
        for head in ["sentiment", "emotion", "tone"]:
            significance[head] = {
                "A_to_B": {**mcnemar_exact(variants["A"]["_correct"][head],
                                           variants["B"]["_correct"][head]),
                           **bootstrap_delta(variants["A"]["_correct"][head],
                                             variants["B"]["_correct"][head])},
                "A_to_E": {**mcnemar_exact(variants["A"]["_correct"][head],
                                           variants["E"]["_correct"][head]),
                           **bootstrap_delta(variants["A"]["_correct"][head],
                                             variants["E"]["_correct"][head])},
            }
        per_regime[regime] = {
            "n_probe_turns": len(convs),
            "text_only_lexical_ceiling": ceiling,
            "variants": _public(variants),
            "significance": significance,
            "headline_gain_pp": {
                head: {"text_only": variants["A"][head]["accuracy"],
                       "full_cerebro": variants["E"][head]["accuracy"],
                       "gain_pp": round((variants["E"][head]["accuracy"]
                                         - variants["A"][head]["accuracy"]) * 100, 1)}
                for head in ["sentiment", "emotion", "tone"]},
        }
        del variants

    payload = {
        "seed": SEED,
        "design": ("controlled context-dependence probe: identical surface text under "
                   "benign vs tense histories; utterance-level label balance pins "
                   "I(text;label)=0, so any context-free model is bounded by the "
                   "lexical ceiling reported alongside every number"),
        "corpus": stats,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "flip_rates": flip_rates(splits["train"]),
        "regimes": per_regime,
        "primary_regime": "unseen_history_wording",
        "wall_time_seconds": round(time.time() - t0, 1),
        "note": ("The probe corpus is constructed, not sampled — blueprint §12 names "
                 "these exact ambiguous forms. It measures whether context and speaker "
                 "memory ARE USED when they are the only available evidence. It is not "
                 "a natural-distribution benchmark; see the transfer evaluation for that."),
    }
    save_json(payload, f"{RESULTS_DIR}/context_proof.json")
    print(f"\nsaved → {RESULTS_DIR}/context_proof.json in {payload['wall_time_seconds']}s")
    return payload


if __name__ == "__main__":
    run()
