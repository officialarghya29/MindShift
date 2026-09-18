"""FastAPI application (PS-01 §31, §34).

Endpoints:
  POST /upload            → raw chat export → parsed messages + id
  POST /parse             → parse + preview without analysis
  POST /analyze           → full CEREBRO analysis
  GET  /conversation/{id}                 → stored report
  GET  /conversation/{id}/timeline        → per-message states
  GET  /conversation/{id}/turning-points  → detected shifts
  GET  /conversation/{id}/speakers        → speaker profiles
  GET  /conversation/{id}/report          → full report
  GET  /healthz               → liveness + model status
  GET  /                      → the futuristic dashboard (frontend/index.html)
  GET  /demo-report           → persisted report from the held-out demo run

Privacy (§40): conversations live in a bounded in-memory store with
TTL eviction; nothing touches disk unless explicitly requested.
"""
from __future__ import annotations

import io
import csv
import json
import time
import uuid
from collections import OrderedDict

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from cerebro.parsers import auto_parse
from cerebro.features.segmentation import segment_conversation
from cerebro.models.pipeline import CerebroPipeline
from cerebro.models.engines import MultiTaskEngine

MAX_STORED = 32
TTL_SECONDS = 3600
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(
    title="CEREBRO — Conversation Intelligence API",
    version="1.0.0",
    description="Context-aware temporal conversation intelligence: sentiment, emotion, "
                "tone, sarcasm, irony, passive-aggression, tension, turning points, "
                "escalation — with explanations.",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_STATE = {"pipeline": None, "loaded_at": None}
_STORE: OrderedDict[str, tuple[dict, float]] = OrderedDict()


def get_pipeline() -> CerebroPipeline:
    if _STATE["pipeline"] is None:
        eng = MultiTaskEngine(seed=42)
        try:
            eng.load("models/saved/cerebro_engine")
        except FileNotFoundError:
            raise HTTPException(503, "model not trained yet — run `python -m evaluation.run_full` first")
        _STATE["pipeline"] = CerebroPipeline.from_trained(eng)
        _STATE["loaded_at"] = time.time()
    return _STATE["pipeline"]


def store_put(key: str, report: dict) -> None:
    _STORE[key] = (report, time.time())
    _STORE.move_to_end(key)
    while len(_STORE) > MAX_STORED:
        _STORE.popitem(last=False)


def store_get(key: str) -> dict:
    item = _STORE.get(key)
    if not item:
        raise HTTPException(404, f"conversation '{key}' not found (expired or never analyzed)")
    report, ts = item
    if time.time() - ts > TTL_SECONDS:
        _STORE.pop(key, None)
        raise HTTPException(404, "conversation expired (TTL)")
    return report


def _read_upload(file: UploadFile) -> str:
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "file too large (max 8 MB)")
    return raw.decode("utf-8", errors="replace")


def _messages_from_upload(text: str, platform: str | None, conv_id: str) -> list[dict]:
    if platform == "csv" or (platform is None and text.lstrip().startswith(("speaker,", "timestamp,"))):
        rows = list(csv.DictReader(io.StringIO(text)))
        return [{"conversation_id": conv_id, "message_id": i + 1,
                 "speaker_id": str(r.get("speaker") or r.get("speaker_id") or "user"),
                 "timestamp": r.get("timestamp"), "text": str(r.get("text", "")),
                 "platform": "csv"} for i, r in enumerate(rows)]
    if platform == "json" or (platform is None and text.lstrip().startswith(("[", "{"))):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "messages" in data:
                data = data["messages"]
            if isinstance(data, list) and data and isinstance(data[0], dict) \
                    and "text" in data[0]:
                return [{"conversation_id": conv_id, "message_id": i + 1,
                         "speaker_id": str(m.get("speaker") or m.get("speaker_id") or "user"),
                         "timestamp": m.get("timestamp"),
                         "text": str(m.get("text", "")), "platform": "json"}
                        for i, m in enumerate(data)]
        except json.JSONDecodeError:
            pass
    return auto_parse(text, conv_id, platform)


@app.post("/upload")
async def upload(file: UploadFile = File(...), platform: str | None = Form(None)):
    """Upload a chat export; returns parsed preview + conversation_id."""
    conv_id = f"conv_{uuid.uuid4().hex[:10]}"
    text = _read_upload(file)
    msgs = _messages_from_upload(text, platform, conv_id)
    if not msgs:
        raise HTTPException(422, "could not parse any messages from the upload")
    return {"conversation_id": conv_id, "n_messages": len(msgs),
            "n_speakers": len({m['speaker_id'] for m in msgs}),
            "platform_detected": msgs[0]["platform"],
            "preview": msgs[:10], "messages": msgs}


@app.post("/parse")
async def parse(file: UploadFile = File(...), platform: str | None = Form(None)):
    """Parse without analysis (preview only)."""
    text = _read_upload(file)
    msgs = _messages_from_upload(text, None, "conv_preview")
    return {"n_messages": len(msgs), "preview": msgs[:10]}


@app.post("/analyze", response_model=None)
async def analyze(file: UploadFile = File(...), platform: str | None = Form(None)):
    """Upload + full CEREBRO analysis in one call. Returns the complete report."""
    t0 = time.time()
    conv_id = f"conv_{uuid.uuid4().hex[:10]}"
    text = _read_upload(file)
    msgs = _messages_from_upload(text, platform, conv_id)
    if not msgs:
        raise HTTPException(422, "could not parse any messages from the upload")
    pipe = get_pipeline()
    report = pipe.analyze(msgs, conversation_id=conv_id)
    report["processing_ms"] = round((time.time() - t0) * 1000, 1)
    store_put(conv_id, report)
    return report


@app.get("/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    return store_get(conv_id)


@app.get("/conversation/{conv_id}/timeline")
async def get_timeline(conv_id: str):
    rep = store_get(conv_id)
    return {"conversation_id": conv_id, "timeline": [
        {"message_id": m["message_id"], "speaker_id": m["speaker_id"],
         "emotion": m["emotion"]["label"], "tension": m["tension"],
         "sentiment": m["sentiment"]["label"], "tone": m["tone"]["label"],
         "sarcasm": m["sarcasm"]["probability"]}
        for m in rep["messages"]]}


@app.get("/conversation/{conv_id}/turning-points")
async def get_turning_points(conv_id: str):
    rep = store_get(conv_id)
    return {"conversation_id": conv_id, "turning_points": rep["turning_points"]}


@app.get("/conversation/{conv_id}/speakers")
async def get_speakers(conv_id: str):
    rep = store_get(conv_id)
    return {"conversation_id": conv_id, "speaker_profiles": rep["speaker_profiles"]}


@app.get("/conversation/{conv_id}/report")
async def get_report(conv_id: str):
    return store_get(conv_id)


@app.get("/conversation/{conv_id}/topics")
async def get_topics(conv_id: str):
    rep = store_get(conv_id)
    msgs = [{"message_id": m["message_id"], "text": m["text"],
             "timestamp": None} for m in rep["messages"]]
    segs = segment_conversation(msgs)
    return {"conversation_id": conv_id, "topics": [
        {"segment_id": s["segment_id"], "start": s["start"], "end": s["end"],
         "method": s["method"],
         "messages_analyzed": len(s["messages"])} for s in segs]}


@app.get("/why/{conv_id}/{message_id}")
async def get_why(conv_id: str, message_id: int):
    rep = store_get(conv_id)
    ex = next((e for e in rep["explanations"] if e["message_id"] == message_id), None)
    if not ex:
        raise HTTPException(404, f"message {message_id} not found")
    return ex


@app.get("/what-changed/{conv_id}/{message_id}")
async def get_what_changed(conv_id: str, message_id: int):
    rep = store_get(conv_id)
    from cerebro.explain.explanation_engine import what_changed
    return what_changed(rep["messages"], message_id)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_loaded": _STATE["pipeline"] is not None,
            "stored_conversations": len(_STORE), "time": time.time()}


_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the futuristic dashboard (frontend/index.html)."""
    if not _FRONTEND.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(_FRONTEND, media_type="text/html")


@app.get("/demo-report", include_in_schema=False)
async def demo_report():
    """Persisted report from the held-out demo run (no raw-text guarantee:
    contents come from evaluation/results/demo_report.json)."""
    p = Path(__file__).resolve().parents[2] / "evaluation" / "results" / "demo_report.json"
    if not p.exists():
        raise HTTPException(404, "demo_report.json not found — run evaluation")
    return FileResponse(p, media_type="application/json")


@app.get("/conversation/{conv_id}/report.pdf")
async def conversation_pdf(conv_id: str):
    """PDF export of the stored conversation report (PS-01 §30).
    Requires the optional fpdf2 dependency."""
    rep = store_get(conv_id)
    import tempfile
    from cerebro.reports.pdf_report import build_pdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        build_pdf(rep, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f'inline; filename="{conv_id}_cerebro_report.pdf"'})
