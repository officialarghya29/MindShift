# AGENTS.md

CEREBRO — context-aware temporal conversation-intelligence engine (FastAPI + scikit-learn, pure-Python monorepo). Spec: `docs/methodology/PS01_WORKFLOW.md`.

## Run everything from repo root
- No packaging (no pyproject/setup). Install `pip install -r requirements.txt` (floating versions), then invoke as `python -m <module>` with cwd = repo root. All paths are root-relative.
- Python: CI uses 3.11, Docker uses 3.12.

## Main commands
- Train + eval (deterministic, seed 42, ~4 min): `python -m evaluation.run_full fit` then `python -m evaluation.run_full eval`. Notably `fit` only writes `models/saved/eval_state.joblib`; the deployable engine `models/saved/cerebro_engine.joblib` is created by `eval`, which also REWRITES the committed `evaluation/results/*.json`.
- API + dashboard: `uvicorn backend.app.main:app --reload` → dashboard at `/`, OpenAPI at `/docs`. Backend lazy-loads the persisted engine and returns 503 if `models/saved/cerebro_engine.joblib` is missing.
- Tests: `python -m pytest tests/ backend/tests/ -q` (39). No trained model needed — `backend/tests` stubs the pipeline.
- Lint: CI gate is `python -m compileall -q cerebro backend evaluation scripts tests` plus an AST-parse check. The documented pyflakes gate (`python -m pyflakes cerebro/ backend/ evaluation/ tests/ scripts/`) needs pyflakes installed — it is NOT in requirements.
- Graphs: `PYTHONPATH=. python -m evaluation.make_graphs`. Ships collision validators that refuse overlapping charts; must pass.

## CI-gated committed artifacts
`evaluation/results/*.json` and `models/saved/*.joblib` are committed and asserted by CI metric gates (sarcasm ROC-AUC ≥ 0.90, tension MAE ≤ 5.0, fusion weights a valid simplex, scenarios 20/20, real errors ≤ 5, scaling ≥ 100 msg/s, memory ≤ 100 MB @1k). After regenerating results, re-run the CI gate block (`.github/workflows/ci.yml`) before committing changes.

## Saturation is by design
Sentiment/emotion/tone heads read 1.000 on the synthetic template corpus (baselines included). Do not "fix" this — the honest signal is sarcasm/irony/PA AUCs, tension MAE/R², and calibration. The transfer eval on real GoEmotions text is the intended stress test.

## Determinism & data
- Corpus is generated at runtime (seed 42); `data/raw/*` and `data/processed/*` are gitignored — never commit them.
- Dockerfile copies only `cerebro/ evaluation/ backend/ models/saved/ assets/`. If you add a runtime top-level module, add it to the Dockerfile or the container breaks.

## Backend / frontend
- Conversation store is in-memory (32 convs, 1 h TTL), cleared on restart — by design.
- Dashboard is zero-build `frontend/index.html` served same-origin by the backend; static mounts are `/assets` and `/docs`.

## Scripts
- `scripts/demo_api.py`, `deepscan_api.py`, `deepscan_advanced.py`, `benchmark.py` need the trained engine (deepscans hit the running API).
- `evaluation/run_transfer.py` needs network (pulls GoEmotions live); its raw-text artifact `evaluation/results/transfer_goemotions_full.json` is gitignored for data minimization.