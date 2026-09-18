"""Full evaluation runner (PS-01 §6, §37, §38, §39).

Protocol:
  1. one wide design matrix on TRAIN (text ⊕ context ⊕ behavior ⊕ memory,
     teacher-forced history) → column-sliced per ablation variant
  2. fit baselines (B1 TF-IDF+LR, B2 TF-IDF+LinearSVC, B3 TF-IDF+context)
  3. fit ablation engines A/B/C/D, evaluate on TEST sequentially
     (predicted-history speaker memory — same as deployment)
  4. E = full CEREBRO: engine D + hidden-signal fusion + temporal engines
  5. metrics + calibration + error analysis → results/*.json

Run:  python -m evaluation.run_full
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
from scipy.sparse import hstack, csr_matrix, vstack

from cerebro.common.metrics import (classification_metrics, regression_metrics,
                                    probability_metrics)
from cerebro.data.generator import generate_corpus, split_conversations, corpus_stats
from cerebro.features.featurizer import build_vectorizer
from cerebro.features.preprocess import process_text, behavioral_vector
from cerebro.context.context_engine import ContextWindow
from cerebro.context.features_builder import _memory_vector
from cerebro.context.speaker_memory import SpeakerMemory
from cerebro.models.baselines import Baseline
from cerebro.models.engines import MultiTaskEngine
from cerebro.models.hidden_signals import apply_hidden_signals
from cerebro.common.io import save_json

RESULTS_DIR = "evaluation/results"
SEED = 42
TENSION_EDGE = 60  # escalation threshold for binary escalation F1 (§37)

HEADS_CLASS = ["sentiment", "emotion", "tone"]
HEADS_BINARY = ["sarcasm", "irony", "passive_aggression"]


# ----------------------------------------------------------------------------
# wide design matrix: build once, slice per variant
# ----------------------------------------------------------------------------
def build_wide_matrices(convs, vec):
    """Per conversation: [text ⊕ context ⊕ behavior ⊕ memory] with teacher-forced
    history (labels known during TRAINING matrix construction only)."""
    Xs, all_meta = [], []
    n_text = len(vec.get_feature_names_out())
    for c in convs:
        rows, meta = [], []
        cw = ContextWindow()
        mem = SpeakerMemory()
        for i, m in enumerate(c):
            behav = np.asarray(behavioral_vector(
                m["text"], c[i-1]["timestamp"] if i else None, m.get("timestamp"),
                c[i-1]["speaker_id"] if i else None, m["speaker_id"]), dtype=float)
            text_part = vec.transform([process_text(m["text"])])
            ctx = cw.context_text(c, i)
            ctx_part = vec.transform([process_text(ctx)]) if ctx else csr_matrix((1, n_text))
            parts = [text_part, ctx_part, csr_matrix(behav.reshape(1, -1))]
            mv = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
            parts.append(csr_matrix(mv))
            rows.append(hstack(parts).tocsr())
            meta.append({"conv": c[0]["conversation_id"], "row": i})
            mem.observe(m["speaker_id"], m.get("emotion", "neutral"),
                        m.get("tone", "neutral"), m.get("sentiment", "neutral"),
                        m.get("tension", 0.0), bool(behav[14]))
        Xs.append(vstack(rows))
        all_meta.extend(meta)
    return vstack(Xs).tocsr(), all_meta


# column layout of the wide matrix:
#   [0, n_text)               text
#   [n_text, 2*n_text)        context
#   [2*n_text, 2*n_text+16)   behavior
#   [2*n_text+16, end)        memory (10 dims)
def _slice_variant(X, n_text, variant: str):
    behav_hi = 2 * n_text + 16
    mem_lo, mem_hi = behav_hi, X.shape[1]
    ctx_hi = 2 * n_text
    if variant == "A":   # text only
        return X[:, :n_text]
    if variant == "B":   # text + context
        return X[:, :ctx_hi]
    if variant == "C":   # text + context + memory
        keep = np.concatenate([np.arange(0, ctx_hi), np.arange(mem_lo, mem_hi)])
        return X[:, keep]
    if variant == "D":   # full: text + context + memory + behavior
        return X
    raise ValueError(variant)


def train_variant_heads(X, msgs, variants, n_text):
    """Fit all heads for each variant (column-sliced)."""
    y = {
        "sentiment": [m["sentiment"] for m in msgs],
        "emotion": [m["emotion"] for m in msgs],
        "tone": [m["tone"] for m in msgs],
        "tension": [float(m["tension"]) for m in msgs],
        "sarcasm": [int(m["sarcasm"]) for m in msgs],
        "irony": [int(m["irony"]) for m in msgs],
        "passive_aggression": [int(m["passive_aggression"]) for m in msgs],
    }
    heads = {}
    for variant in variants:
        Xv = _slice_variant(X, n_text, variant)
        vh = {}
        for head, labels in y.items():
            if head == "tension":
                from sklearn.linear_model import Ridge
                vh[head] = Ridge(alpha=1.0, random_state=SEED).fit(Xv, labels)
            elif head in HEADS_BINARY:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.linear_model import LogisticRegression
                base = LogisticRegression(max_iter=1500, C=1.6,
                                          class_weight="balanced", random_state=SEED)
                vh[head] = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(Xv, labels)
            else:
                from sklearn.linear_model import LogisticRegression
                vh[head] = LogisticRegression(max_iter=1500, C=4.0,
                                              class_weight="balanced",
                                              random_state=SEED).fit(Xv, labels)
        heads[variant] = vh
    return heads


# ----------------------------------------------------------------------------
# sequential test evaluation (predicted history — deployment-faithful)
# ----------------------------------------------------------------------------
def eval_variant_sequential(heads_variant, vec, convs, variant, n_text):
    y_true = {h: [] for h in HEADS_CLASS + HEADS_BINARY + ["tension", "escalation"]}
    y_pred = {h: [] for h in y_true}
    p_proba = {h: [] for h in HEADS_BINARY}
    latencies = []

    for c in convs:
        cw = ContextWindow()
        mem = SpeakerMemory()
        for i, m in enumerate(c):
            behav = np.asarray(behavioral_vector(
                m["text"], c[i-1]["timestamp"] if i else None, m.get("timestamp"),
                c[i-1]["speaker_id"] if i else None, m["speaker_id"]), dtype=float)
            t0 = time.perf_counter()
            text_part = vec.transform([process_text(m["text"])])
            ctx = cw.context_text(c, i)
            ctx_part = vec.transform([process_text(ctx)]) if ctx else csr_matrix((1, n_text))
            if variant == "A":
                X = text_part
            elif variant == "B":
                X = hstack([text_part, ctx_part]).tocsr()
            elif variant == "C":
                mv = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
                X = hstack([text_part, ctx_part, csr_matrix(mv)]).tocsr()
            else:
                mv = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
                X = hstack([text_part, ctx_part, csr_matrix(behav.reshape(1, -1)),
                            csr_matrix(mv)]).tocsr()
            pred = {}
            for head in HEADS_CLASS:
                p = heads_variant[head].predict_proba(X)[0]
                cls = list(heads_variant[head].classes_)
                pred[head] = {"label": str(cls[int(np.argmax(p))]),
                              "confidence": float(np.max(p)),
                              "probabilities": {str(k): float(v) for k, v in zip(cls, p)}}
            pred["tension"] = float(np.clip(heads_variant["tension"].predict(X)[0], 0, 100))
            for head in HEADS_BINARY:
                p1 = float(heads_variant[head].predict_proba(X)[0][1])
                pred[head] = {"probability": p1, "prediction": int(p1 >= .5)}
            latencies.append(time.perf_counter() - t0)

            # gold
            y_true["sentiment"].append(m["sentiment"])
            y_true["emotion"].append(m["emotion"])
            y_true["tone"].append(m["tone"])
            y_true["tension"].append(float(m["tension"]))
            for head in HEADS_BINARY:
                y_true[head].append(int(m[head]))
                y_pred[head].append(pred[head]["prediction"])
                p_proba[head].append(pred[head]["probability"])

            for head in HEADS_CLASS:
                y_pred[head].append(pred[head]["label"])
            y_pred["tension"].append(pred["tension"])

            # escalation flag: tension crossing the edge threshold
            y_true["escalation"].append(int(float(m["tension"]) >= TENSION_EDGE))
            y_pred["escalation"].append(int(pred["tension"] >= TENSION_EDGE))

            # advance memory with predictions (deployment-faithful)
            mem.observe(m["speaker_id"], pred["emotion"]["label"],
                        pred["tone"]["label"], pred["sentiment"]["label"],
                        pred["tension"], bool(behav[14]))

    metrics = {}
    for head in HEADS_CLASS:
        metrics[head] = classification_metrics(y_true[head], y_pred[head])
    for head in HEADS_BINARY:
        metrics[head] = classification_metrics(y_true[head], y_pred[head])
        pm = probability_metrics(y_true[head], p_proba[head])
        metrics[head].update({"roc_auc": pm["roc_auc"], "pr_auc": pm["pr_auc"],
                              "brier": pm["brier"]})
    metrics["tension"] = regression_metrics(y_true["tension"],
                                            [p for p in y_pred["tension"]])
    metrics["escalation"] = classification_metrics(y_true["escalation"], y_pred["escalation"])
    metrics["latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    return metrics


# ----------------------------------------------------------------------------
# full CEREBRO (variant E): engine D + hidden fusion + temporal
# ----------------------------------------------------------------------------
def eval_full_cerebro(engine: MultiTaskEngine, convs):
    """Engine predictions → hidden-signal fusion → temporal engines.
    Message-level metrics use tension/emotion/sarcasm from the fused state."""
    y_true = {h: [] for h in HEADS_CLASS + HEADS_BINARY + ["tension", "escalation"]}
    y_pred = {h: [] for h in y_true}
    p_proba = {h: [] for h in HEADS_BINARY}
    latencies = []

    from cerebro.models.pipeline import _add_behavior_flags
    for c in convs:
        t0c = time.perf_counter()
        results = engine.predict_conversation(c)
        apply_hidden_signals(results)
        _add_behavior_flags(results)
        latencies.append((time.perf_counter() - t0c) / len(c))

        for r, m in zip(results, c):
            y_true["sentiment"].append(m["sentiment"])
            y_true["emotion"].append(m["emotion"])
            y_true["tone"].append(m["tone"])
            y_true["tension"].append(float(m["tension"]))
            y_true["escalation"].append(int(float(m["tension"]) >= TENSION_EDGE))
            y_pred["escalation"].append(int(r["tension"] >= TENSION_EDGE))
            for head in HEADS_CLASS:
                y_pred[head].append(r[head]["label"])
            for head in HEADS_BINARY:
                y_true[head].append(int(m[head]))
                y_pred[head].append(int(r[head]["probability"] >= .5))
                p_proba[head].append(r[head]["probability"])
            y_pred["tension"].append(r["tension"])

    metrics = {}
    for head in HEADS_CLASS:
        metrics[head] = classification_metrics(y_true[head], y_pred[head])
    for head in HEADS_BINARY:
        metrics[head] = classification_metrics(y_true[head], y_pred[head])
        pm = probability_metrics(y_true[head], p_proba[head])
        metrics[head].update({"roc_auc": pm["roc_auc"], "pr_auc": pm["pr_auc"],
                              "brier": pm["brier"]})
    metrics["tension"] = regression_metrics(y_true["tension"], y_pred["tension"])
    metrics["escalation"] = classification_metrics(y_true["escalation"], y_pred["escalation"])
    metrics["latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    return metrics


# ----------------------------------------------------------------------------
# error analysis (PS-01 §39)
# ----------------------------------------------------------------------------
def _design_label_lookup() -> dict[str, tuple[int, int, int]]:
    """Reverse lookup: template text → design (sarcasm, irony, PA) labels.
    Used to attribute gold-label disagreements to the injected annotator noise."""
    from cerebro.data.domains import DOMAINS
    import re as _re
    lookup = {}
    for dom in DOMAINS.values():
        for band in dom["bands"]:
            for tpl in band:
                text, _, _, _, sarc, iron, pa, _ = tpl
                key = _re.sub(r"\s+", " ", text.lower().strip())
                lookup[key] = (int(sarc), int(iron), int(pa))
    return lookup


def _clean_text(t: str) -> str:
    import re as _re
    t = _re.sub(r"\s*\([^)]*\)\s*", " ", t)   # strip injected topic references
    return _re.sub(r"\s+", " ", t.lower().strip())


def error_analysis(engine: MultiTaskEngine, convs, max_rows: int = 400) -> dict:
    rows = []
    err_types = Counter()
    fn_fp = Counter()
    noise_attributed = Counter()
    design = _design_label_lookup()
    for c in convs:
        results = engine.predict_conversation(c)
        apply_hidden_signals(results)
        from cerebro.models.pipeline import _add_behavior_flags
        _add_behavior_flags(results)
        for r, m in zip(results, c):
            heads_to_check = [("sentiment", m["sentiment"], r["sentiment"]["label"],
                               r["sentiment"]["confidence"]),
                              ("emotion", m["emotion"], r["emotion"]["label"],
                               r["emotion"]["confidence"]),
                              ("tone", m["tone"], r["tone"]["label"],
                               r["tone"]["confidence"])]
            for head, gt, pred_label, conf in heads_to_check:
                if pred_label == gt:
                    continue
                sig = r["signals"]
                if sig["sarc_words"] or r["sarcasm"]["probability"] > .5:
                    et = "sarcasm interference"
                elif sig["is_short"]:
                    et = "short-message ambiguity"
                elif sig["emoji_polarity"] != 0:
                    et = "emoji polarity conflict"
                elif len(sig["neg_hits"]) and len(sig["pos_hits"]):
                    et = "mixed polarity"
                else:
                    et = "context disagreement"
                err_types[et] += 1
                if len(rows) < max_rows:
                    rows.append({
                        "conversation": m["conversation_id"],
                        "message_id": m["message_id"],
                        "head": head,
                        "text": m["text"][:100],
                        "ground_truth": str(gt),
                        "prediction": str(pred_label),
                        "confidence": conf,
                        "error_type": et,
                    })
            for head_idx, head in enumerate(HEADS_BINARY):
                gold = int(m[head])
                pred = int(r[head]["probability"] >= .5)
                if gold == pred:
                    continue
                kind = "false_negative" if gold == 1 else "false_positive"
                fn_fp[f"{head}:{kind}"] += 1
                err_types[f"{head}:{kind}"] += 1
                sig = r["signals"]
                # noise attribution: does the gold label contradict the template design?
                d_label = design.get(_clean_text(m["text"]), (None, None, None))[head_idx]
                is_noise = d_label is not None and d_label != gold
                if is_noise:
                    noise_attributed[f"{head}:{kind}"] += 1
                    cause = ("gold label contradicts the template design — attributable "
                             "to the injected 1.5% annotator noise (irreducible error)")
                elif gold == 1 and pred == 0:
                    cause = ("symbolic evidence below threshold (subtle contradiction)"
                             if r[head]["learned_p"] > .4
                             else "literal reading dominated — no contextual trigger")
                else:
                    cause = ("tension trajectory primed the prior"
                             if r[head]["probability"] - r[head].get("learned_p", 0) > .2
                             else "marker words present without ironic intent")
                if len(rows) < max_rows:
                    rows.append({
                        "conversation": m["conversation_id"],
                        "message_id": m["message_id"],
                        "head": head,
                        "text": m["text"][:100],
                        "ground_truth": str(gold),
                        "prediction": f"{pred} (p={r[head]['probability']:.2f})",
                        "confidence": r[head]["probability"],
                        "error_type": f"{head}:{kind}",
                        "attributed_to_noise": bool(is_noise),
                        "template_design_label": d_label,
                        "possible_cause": cause,
                    })
    real_errors = {k: v - noise_attributed.get(k, 0) for k, v in fn_fp.items()}
    save_json({"counts": dict(err_types), "binary_breakdown": dict(fn_fp),
               "noise_attribution": dict(noise_attributed),
               "real_errors_after_noise_removal": real_errors,
               "samples": rows},
              f"{RESULTS_DIR}/error_analysis.json")
    return {"counts": dict(err_types), "binary_breakdown": dict(fn_fp),
            "noise_attribution": dict(noise_attributed),
            "real_errors_after_noise_removal": real_errors,
            "n_samples": len(rows)}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def fit_stage(convs_per_cell: int = 14) -> str:
    """Stage 1: fit everything and persist to disk (survives between runs)."""
    import joblib
    t0 = time.time()
    print("== STAGE 1: fit ==")
    corpus = generate_corpus(convs_per_cell=convs_per_cell)
    splits = split_conversations(corpus)
    print(f"corpus: {len(corpus)} conversations; "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    msgs_train = [m for c in splits['train'] for m in c]
    print("building vectorizer + wide design matrix...")
    vec = build_vectorizer([m["text"] for m in msgs_train])
    X_wide, _ = build_wide_matrices(splits["train"], vec)
    n_text = len(vec.get_feature_names_out())
    print(f"wide matrix: {X_wide.shape} (text={n_text})")

    print("fitting ablation variants A/B/C/D...")
    heads = train_variant_heads(X_wide, msgs_train, ["A", "B", "C", "D"], n_text)
    print("fitting baselines B1/B2/B3...")
    baselines = {
        "B1_tfidf_logreg": Baseline("B1", use_context=False).fit(splits["train"]),
        "B2_tfidf_svc": Baseline("B2", use_context=False, linear_svc=True).fit(splits["train"]),
        "B3_tfidf_context": Baseline("B3", use_context=True).fit(splits["train"]),
    }
    joblib.dump({"vec": vec, "heads": heads, "n_text": n_text,
                 "baselines": baselines}, "models/saved/eval_state.joblib")
    print(f"stage 1 done in {round(time.time()-t0,1)}s → models/saved/eval_state.joblib")
    return "models/saved/eval_state.joblib"


def eval_stage():
    """Stage 2: load fitted state, run all test evaluations, write results."""
    import joblib
    t_start = time.time()
    print("== STAGE 2: evaluate ==")
    state = joblib.load("models/saved/eval_state.joblib")
    vec, heads, n_text = state["vec"], state["heads"], state["n_text"]
    baselines = state["baselines"]

    corpus = generate_corpus(convs_per_cell=14)
    splits = split_conversations(corpus)
    stats = corpus_stats(corpus)
    print(f"test conversations: {len(splits['test'])}")

    print("evaluating baselines...")
    baseline_metrics = {}
    for name, b in baselines.items():
        baseline_metrics[name] = eval_variant_sequential(
            b.heads, b.vec, splits["test"], "A" if not b.use_context else "B",
            len(b.vec.get_feature_names_out()))
        bm = baseline_metrics[name]
        print(f"  {name}: sent-f1={bm['sentiment']['f1_macro']} "
              f"emo-f1={bm['emotion']['f1_macro']} tension-mae={bm['tension']['mae']}")
    save_json(baseline_metrics, f"{RESULTS_DIR}/baselines.json")

    print("evaluating ablations A-D (sequential, predicted history)...")
    ablation_metrics = {}
    for variant in ["A", "B", "C", "D"]:
        ablation_metrics[variant] = eval_variant_sequential(
            heads[variant], vec, splits["test"], variant, n_text)
        am = ablation_metrics[variant]
        print(f"  {variant}: sent-f1={am['sentiment']['f1_macro']} "
              f"emo-f1={am['emotion']['f1_macro']} tone-f1={am['tone']['f1_macro']} "
              f"sarc-auc={am['sarcasm']['roc_auc']} tension-mae={am['tension']['mae']}")
    save_json(ablation_metrics, f"{RESULTS_DIR}/ablations.json")

    print("full CEREBRO (E)...")
    engine = MultiTaskEngine(seed=SEED)
    engine.vec = vec
    engine.heads = heads["D"]
    engine.trained = True
    full_metrics = eval_full_cerebro(engine, splits["test"])
    print(f"  E: sent-f1={full_metrics['sentiment']['f1_macro']} "
          f"emo-f1={full_metrics['emotion']['f1_macro']} "
          f"sarc-auc={full_metrics['sarcasm']['roc_auc']} "
          f"tension-mae={full_metrics['tension']['mae']} "
          f"latency={full_metrics['latency_ms']}ms/msg")

    print("error analysis (40 test conversations)...")
    errs = error_analysis(engine, splits["test"][:40])

    print("tuning fusion weights on validation (calibration loss, PS-01 §24)...")
    from cerebro.models.pipeline import CerebroPipeline
    pipe = CerebroPipeline.from_trained(engine, val_convs=splits["val"])
    engine.fusion_weights = pipe.fusion_weights
    print(f"  tuned: {engine.fusion_weights}")

    engine.save("models/saved/cerebro_engine")
    print("engine saved → models/saved/cerebro_engine.joblib")

    demo = pipe.analyze([dict(m) for m in splits["test"][0]])
    save_json(demo, f"{RESULTS_DIR}/demo_report.json")
    print(f"demo report: {demo['summary']['n_messages']} msgs, "
          f"trajectory={demo['summary']['trajectory']}, "
          f"turning_points={len(demo['turning_points'])}")

    save_json({
        "seed": SEED,
        "n_train_conversations": len(splits["train"]),
        "n_val_conversations": len(splits["val"]),
        "n_test_conversations": len(splits["test"]),
        "n_test_messages": sum(len(c) for c in splits["test"]),
        "n_features_total": state["heads"]["D"]["tension"].n_features_in_,
        "n_features_text": n_text,
        "corpus_stats": stats,
        "full_metrics": full_metrics,
        "fusion_weights": engine.fusion_weights,
        "fusion_tuning_status": (
            "tuned on validation" if engine.fusion_weights !=
            __import__("cerebro.fusion.fusion", fromlist=["_DEFAULT_W"])._DEFAULT_W
            else "fallback (validation confidences saturated — weights unidentifiable, disclosed)"),
        "error_analysis": errs,
        "wall_time_seconds": round(time.time() - t_start, 1),
    }, f"{RESULTS_DIR}/summary.json")
    print(f"done in {round(time.time()-t_start,1)}s → {RESULTS_DIR}/")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "fit":
        fit_stage()
    elif stage == "eval":
        eval_stage()
    else:
        fit_stage()
        eval_stage()
