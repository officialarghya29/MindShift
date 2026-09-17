# PS-01 Workflow → Implementation Map

The full CEREBRO pipeline (PS-01 §46), stage by stage, with the exact file
that implements it.

| # | PS-01 stage | Implementation | Status |
|---|---|---|---|
| 1 | Chat parser (WhatsApp/Discord/Slack/generic) | `cerebro/parsers/platforms.py` | ✅ P0 |
| 2 | Preprocessing (raw preserved, slang, emoji) | `cerebro/features/preprocess.py` | ✅ P0 |
| 3 | Behavioral features (16-dim) | `cerebro/features/preprocess.py::behavioral_vector` | ✅ P0 |
| 4 | Dataset (6 domains × 7 arcs, conversation splits) | `cerebro/data/generator.py`, `cerebro/data/domains.py` | ✅ P0 |
| 5 | Message representation (text ⊕ behavioral ⊕ context ⊕ memory) | `cerebro/features/featurizer.py`, `cerebro/context/features_builder.py` | ✅ P0 |
| 6 | Baselines B1/B2/B3 | `cerebro/models/baselines.py` | ✅ |
| 7 | Context engine (sliding window + decayed summary) | `cerebro/context/context_engine.py` | ✅ P0 |
| 8 | Speaker memory | `cerebro/context/speaker_memory.py` | ✅ P0/P1 |
| 9 | Sentiment engine | `cerebro/models/engines.py` (head) | ✅ P0 |
| 10 | Emotion engine (13 classes) | `cerebro/models/engines.py` (head) | ✅ P0 |
| 11 | Tone engine (14 classes) | `cerebro/models/engines.py` (head) | ✅ P0 |
| 12 | Tension engine (0–100) | `cerebro/models/engines.py` (head) | ✅ P0 |
| 13 | Sarcasm engine (learned ⊕ contradiction evidence) | `cerebro/models/hidden_signals.py` | ✅ P1 |
| 14 | Irony engine (literal vs contextual mismatch) | `cerebro/models/hidden_signals.py` | ✅ P2 |
| 15 | Passive-aggression engine (context-gated) | `cerebro/models/hidden_signals.py` | ✅ P1 |
| 16 | Temporal emotion engine (arc, intensity, stability) | `cerebro/temporal/arc.py` | ✅ P0 |
| 17 | Emotion transition engine (+ matrix) | `cerebro/temporal/transitions.py` | ✅ P1 |
| 18 | Turning-point detector (robust z on Δtension) | `cerebro/temporal/turning_points.py` | ✅ P0 |
| 19 | Escalation engine (trajectory + phases) | `cerebro/temporal/escalation.py` | ✅ P0 |
| 20 | Topic segmentation (time-gap + cohesion) | `cerebro/features/segmentation.py` | ✅ P2 |
| 21 | Model fusion (validation-tuned streams) | `cerebro/fusion/fusion.py` | ✅ |
| 22 | Confidence calibration (Platt) | heads + `common/metrics.py::probability_metrics` | ✅ |
| 23 | Explainability engine (WHY?) | `cerebro/explain/explanation_engine.py` | ✅ P0 |
| 24 | "WHAT CHANGED?" panel | `cerebro/explain/explanation_engine.py::what_changed` | ✅ P1 |
| 25 | Speaker-level analysis | `cerebro/explain/explanation_engine.py::speaker_profiles` | ✅ P1 |
| 26 | Conversation report (18 sections) | `cerebro/models/pipeline.py::analyze` | ✅ P0 |
| 27 | FastAPI backend (14 endpoints incl. dashboard + demo-report) | `backend/app/main.py` | ✅ |
| 28 | Evaluation pipeline (§37 metrics) | `evaluation/run_full.py` | ✅ |
| 29 | Ablation study (§38 A–E) | `evaluation/run_full.py::train_variant_heads` | ✅ |
| 30 | Error analysis (§39, typed errors + causes) | `evaluation/run_full.py::error_analysis` | ✅ |
| 31 | Dashboard assets (logo-branded graphs) | `evaluation/make_graphs.py` → `assets/graphs/` | ✅ |
| 32 | Frontend components (API-consumable) | `backend/app/schemas/analysis_schema.py` + docs | 🟡 API-first |
| 33 | Database schema (§33) | in-memory TTL store mapping, swap-ready | 🟡 |
| 34 | Docker deployment | `Dockerfile`, `docker-compose.yml` | ✅ |
| 35 | Test cases (§41, 20 scenarios) | `tests/`, `backend/tests/` | ✅ 25 tests |

**Priority legend** — P0: absolutely required (PS-01 §44) · P1: high · P2: bonus.
All P0 and P1 items are implemented and evaluated. 🟡 items are intentionally
API-first: the dashboard ships as the API + schema contract the frontend
components consume (PS-01's own final principle: model first, UI last).
