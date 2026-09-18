"""Efficiency benchmark of the full CEREBRO pipeline (PS-01 §42).

Measures, with the REAL persisted engine:
  1. model cold-load time
  2. per-stage latency (parse → features → heads → hidden → temporal → explain)
  3. end-to-end latency scaling: 10/50/200/500/1000-message conversations
  4. throughput (messages/sec) and per-message cost
  5. peak RSS memory during a 1000-message analysis
  6. API-level latency through the FastAPI stack (same as deployment)

Writes evaluation/results/benchmarks.json and prints a summary table.
Run:  python scripts/benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, ".")
sys.path.insert(0, "backend")

warnings_off = True
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from cerebro.data.generator import generate_corpus
from cerebro.parsers.platforms import parse_generic
from cerebro.features.preprocess import process_text, behavioral_vector
from cerebro.models.engines import MultiTaskEngine
from cerebro.models.pipeline import CerebroPipeline
from cerebro.models.hidden_signals import apply_hidden_signals
from cerebro.temporal.arc import build_arc
from cerebro.temporal.escalation import escalation_flags
from cerebro.explain.explanation_engine import explain_message
from cerebro.features.segmentation import segment_conversation

R = {"seed": 42, "engine": "models/saved/cerebro_engine"}


def _conv(n: int) -> list[dict]:
    """A realistic n-message conversation mixing domains from the corpus."""
    corpus = generate_corpus(convs_per_cell=4)
    pool = [m for c in corpus for m in c]
    out, i = [], 0
    while len(out) < n:
        m = dict(pool[i % len(pool)])
        m["message_id"] = len(out) + 1
        m["timestamp"] = None
        out.append(m)
        i += 1
    return out


def bench_load() -> None:
    t0 = time.perf_counter()
    eng = MultiTaskEngine(seed=42).load("models/saved/cerebro_engine")
    R["load_seconds"] = round(time.perf_counter() - t0, 3)
    R["engine_heads"] = sorted(eng.heads.keys()) if hasattr(eng, "heads") else "n/a"
    global PIPE
    PIPE = CerebroPipeline.from_trained(eng)


def bench_stages() -> None:
    conv = _conv(50)
    texts = [m["text"] for m in conv]

    def t(fn, reps=3):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return round(best * 1000 / len(texts), 3)  # ms per message

    R["stage_latency_ms_per_message"] = {
        "parse": t(lambda: parse_generic("\n".join(f"S{i%2}: {t}" for i, t in enumerate(texts)), "bench")),
        "preprocess+features": t(lambda: [(process_text(x), behavioral_vector(x, None, None, None, "S")) for x in texts]),
        "ml_heads": t(lambda: PIPE.engine.predict_conversation(conv)),
        "hidden_signals": t(lambda: apply_hidden_signals(PIPE.engine.predict_conversation(conv))),
        "temporal": t(lambda: (build_arc(PIPE.engine.predict_conversation(conv)),
                               escalation_flags(PIPE.engine.predict_conversation(conv)))),
        "explainability": t(lambda: _explain_cost(conv)),
        "segmentation": t(lambda: segment_conversation(conv)),
    }


def _explain_cost(conv: list[dict]) -> None:
    """Explain 5 messages (WHY? panel cost) at correct per-message scale."""
    res = PIPE.engine.predict_conversation(conv)
    apply_hidden_signals(res)
    for i in range(5):
        explain_message(res[i], res[i - 1] if i else None)


def bench_scaling() -> None:
    rows = []
    for n in (10, 50, 200, 500, 1000):
        conv = _conv(n)
        t0 = time.perf_counter()
        report = PIPE.analyze(conv, conversation_id="bench")
        dt = time.perf_counter() - t0
        assert len(report["messages"]) == n
        rows.append({"n_messages": n, "total_s": round(dt, 3),
                     "ms_per_message": round(dt * 1000 / n, 2),
                     "messages_per_second": round(n / dt, 1)})
        print(f"    {n:>5} msgs → {dt:6.2f}s  ({rows[-1]['ms_per_message']:.1f} ms/msg, "
              f"{rows[-1]['messages_per_second']:.0f} msg/s)")
    R["scaling"] = rows


def bench_memory() -> None:
    conv = _conv(1000)
    tracemalloc.start()
    report = PIPE.analyze(conv, conversation_id="bench_mem")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    R["memory_1000msg"] = {"peak_mb": round(peak / 1e6, 1),
                           "report_size_kb": round(len(json.dumps(report)) / 1024, 1)}


def bench_api() -> None:
    """API-level latency including HTTP + JSON (the number users feel)."""
    from fastapi.testclient import TestClient
    from backend.app.main import app
    client = TestClient(app)
    conv = _conv(50)
    lines = "\n".join(f"{m['speaker_id']}: {m['text']}" for m in conv)
    files = {"file": ("chat.txt", lines.encode())}
    # warm-up (model lazy-load)
    client.post("/analyze", files=files)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = client.post("/analyze", files=files)
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    R["api_50msg"] = {"mean_s": round(float(np.mean(times)), 3),
                      "p95_s": round(float(np.percentile(times, 95)), 3)}


if __name__ == "__main__":
    print("CEREBRO efficiency benchmark — real persisted engine, seed 42")
    print("  1 · model load…")
    bench_load()
    print(f"     loaded in {R['load_seconds']}s")
    print("  2 · per-stage latency…")
    bench_stages()
    for k, v in R["stage_latency_ms_per_message"].items():
        print(f"     {k:<22} {v:>7.2f} ms/msg")
    print("  3 · end-to-end scaling…")
    bench_scaling()
    print("  4 · memory…")
    bench_memory()
    print(f"     peak {R['memory_1000msg']['peak_mb']} MB for 1,000 messages")
    print("  5 · API latency (50 msgs, incl. HTTP)…")
    bench_api()
    print(f"     mean {R['api_50msg']['mean_s']}s · p95 {R['api_50msg']['p95_s']}s")
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    print("done → evaluation/results/benchmarks.json")
