# Fine-tuning results — real GoEmotions data

**Engine:** text-only `MultiTaskEngine` (identical heads as CEREBRO's text tower) · **Corpus:** 2,996 unique real Reddit comments (GoEmotions `simplified`, research license) · **Split:** row-level 70/15/15, stratified by gold emotion band, exact-duplicates removed.

> **Split-level disclosure.** The public HF API exposes no Reddit thread ids, so a thread-level split is impossible through this endpoint. The dominant leakage mode for single-message classification is duplicate/near-duplicate text, which is removed here. Thread-level splits remain the correct protocol for multi-turn models (see roadmap).

## Zero-shot vs fine-tuned on the held-out real test split (n=453)

| Metric | Before (in-corpus engine, zero-shot) | After (real-data engine) | Δ |
|---|---|---|---|
| Emotion exact (13-way) | 11.7% | **44.6%** | **+32.9** |
| Emotion top-3 | 37.1% | **75.3%** | **+38.2** |
| Sentiment exact (3-way) | 38.0% | **60.7%** | **+22.7** |
| Tension separates valence (AUC) | 0.638 | **0.828** | **+0.190** |

## Retention — what happens to template performance?

| Engine | In-corpus test emotion exact | In-corpus test sentiment exact |
|---|---|---|
| In-corpus engine (reference) | 1.000 | 1.000 |
| Real-data engine (transfer back) | 0.259 | 0.525 |

The real-data engine does **not** retain template performance — expected: it has never seen the template lexicons. In production both engines are ensembled (the deployable keeps the full CEREBRO stack for conversations, the real-data head strengthens per-message classification); a joint multi-corpus training run is the roadmap item.

## Why this matters (PS-01 §37 → §39)

The transfer eval proved the gap is *real*; this run proves it is *closeable*: +33 points of emotion accuracy from 2k real messages with the same model family. The remaining gap to published transformers (GoEmotions SOTA ≈ 46–48% exact for the 28-class set) is a data-scale and pretraining question, not an architecture question.

Reproduce: `PYTHONPATH=. python3 evaluation/run_finetune.py 30`
