# CEREBRO Architecture Notes

## End-to-end data flow (PS-01 §46)

```
RAW CHAT → PARSER → PREPROCESSOR → MESSAGE REPRESENTATION
        → CONTEXT ENGINE + SPEAKER MEMORY
        → MULTI-TASK NLP (7 heads)
        → HIDDEN-SIGNAL FUSION (sarcasm · irony · PA)
        → TEMPORAL ENGINES (arc · transitions · turning points · escalation)
        → MODEL FUSION → EXPLAINABILITY → REPORT → DASHBOARD
```

![architecture](architecture.png)

## Design-matrix column layout

The wide training matrix (built once, sliced per ablation variant):

| Columns | Block | Dim |
|---|---|---|
| `[0, n_text)` | TF-IDF word+bigram n-grams of the message | 826 (fit on train) |
| `[n_text, 2·n_text)` | TF-IDF of the context window (prev 4 turns + decayed summary) | 826 |
| `[2·n_text, +16)` | behavioral vector (see `features/preprocess.py`) | 16 |
| `[2·n_text+16, end)` | speaker-memory vector (5 state + 5 interaction dims) | 10 |

Total at seed 42: **1,678 features**. Ablation variants slice this matrix:
A = text, B = text+context, C = +memory, D = +behavior (all).

## Inference protocol

- Context: sliding 4-turn verbatim window + exponential-decay (0.85) summary of
  the older tail (top-3 tension messages).
- Speaker memory: per-speaker rolling 5-turn state; at test/production time it
  is advanced with the model's **own predictions** (predicted history).
- Hidden signals: engine probabilities are post-fused with symbolic
  contradiction/trajectory evidence (noisy-OR), then the temporal engines run
  conversation-level analysis.
- Latency: 3.5 ms/message measured on the test split, single CPU core.

## Serving layer

FastAPI app (`backend/app/main.py`):
- bounded in-memory store (32 conversations, TTL 1 h) — privacy by default;
- endpoints: `/upload`, `/parse`, `/analyze`, `/conversation/{id}`,
  `/timeline`, `/turning-points`, `/speakers`, `/report`, `/topics`,
  `/why/{id}/{mid}`, `/what-changed/{id}/{mid}`, `/healthz`;
- the persisted engine (`models/saved/cerebro_engine.joblib`) is lazy-loaded
  on first analysis.

## Database mapping (PS-01 §33)

The in-memory store maps 1:1 onto the relational schema in PS-01 §33 —
`conversations` → report summary, `messages` → per-message states,
`analysis_results` → per-head predictions, `turning_points` → the turning-point
list, `model_versions` → the seed + results JSONs. Swap the store for
PostgreSQL by replacing `store_put`/`store_get`.
