"""Deterministic JSON/JSONL IO used across the pipeline."""
from __future__ import annotations

import json
import os
import tempfile


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path) or ".",
                                     delete=False, suffix=".tmp") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        tmp = f.name
    os.replace(tmp, path)
    return path


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(items, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
