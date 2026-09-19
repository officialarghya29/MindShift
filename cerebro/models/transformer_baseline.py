"""Pretrained-transformer baseline (blueprint §9, baselines B2 and B3).

Blueprint §9 asks for three reference points before the advanced system:

  B1  TF-IDF + Logistic Regression        → `cerebro/models/baselines.py`
  B2  Pretrained Transformer + head       → here (`MiniLMEmbedder`)
  B3  Transformer + conversation context  → here (context-concatenated head)

Design notes
------------
* **Why MiniLM.** §41 forbids training a large model from scratch and asks that
  the innovation live in context modelling, temporal reasoning and fusion — not
  in pre-training. A small frozen sentence encoder is exactly the reference
  point §9 wants: strong general-purpose language understanding, no fine-tuning.
* **Why ONNX + stdlib tokenizer.** The project ships a CPU-only container and
  `requirements.txt` stays light (no torch/transformers in production). This
  module therefore fetches the public ONNX export of
  `sentence-transformers/all-MiniLM-L6-v2` and runs it through `onnxruntime`,
  tokenising with a self-contained BERT WordPiece implementation. No new runtime
  dependency is added to the service; `onnxruntime` is imported lazily so that a
  missing runtime or an offline host degrades to a clear, non-fatal error.
* **Weights are downloaded, never trained.** They land in `models/cache/minilm/`
  (git-ignored). Delete the directory to force a re-fetch.

Usage:
    emb = MiniLMEmbedder()
    X = emb.encode(["Fine.", "Yeah I'll do it tonight."])   # (n, 384) float32
"""
from __future__ import annotations

import json
import os
import unicodedata
import urllib.request

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"
CACHE_DIR = os.path.join("models", "cache", "minilm")
FILES = {
    "model.onnx": f"{BASE_URL}/onnx/model.onnx",
    "vocab.txt": f"{BASE_URL}/vocab.txt",
    "tokenizer_config.json": f"{BASE_URL}/tokenizer_config.json",
}
MAX_LEN = 128
MAX_CHARS_PER_WORD = 100


# ---------------------------------------------------------------------------
# BERT WordPiece tokenizer (uncased)
# ---------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S")


def _basic_tokens(text: str) -> list[str]:
    """Whitespace split with punctuation broken out as its own token."""
    out = []
    for chunk in text.split():
        cur = ""
        for ch in chunk:
            if _is_punct(ch):
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(ch)
            else:
                cur += ch
        if cur:
            out.append(cur)
    return out


class WordPieceTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, encoding="utf-8") as fh:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(fh)}
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > MAX_CHARS_PER_WORD:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        text = _strip_accents(text.lower()).strip()
        ids = [self.cls_id]
        for tok in _basic_tokens(text):
            ids.extend(self._wordpiece(tok))
            if len(ids) >= MAX_LEN - 1:
                break
        ids = ids[: MAX_LEN - 1] + [self.sep_id]
        return ids, [1] * len(ids)


# ---------------------------------------------------------------------------
# embedder
# ---------------------------------------------------------------------------
class MiniLMEmbedder:
    """Frozen pretrained encoder → L2-normalised 384-d sentence embeddings."""

    def __init__(self, cache_dir: str = CACHE_DIR, verbose: bool = False):
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._session = None
        self._tok = None
        self._input_names: set[str] = set()

    # -- weights ---------------------------------------------------------
    def _ensure_files(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        for name, url in FILES.items():
            path = os.path.join(self.cache_dir, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            if self.verbose:
                print(f"  fetching {name} …")
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            os.replace(tmp, path)

    def load(self) -> "MiniLMEmbedder":
        if self._session is not None:
            return self
        try:
            import onnxruntime as ort
        except ImportError as exc:            # pragma: no cover - env dependent
            raise RuntimeError(
                "onnxruntime is required for the pretrained-transformer baseline "
                "(pip install onnxruntime)") from exc
        self._ensure_files()
        self._tok = WordPieceTokenizer(os.path.join(self.cache_dir, "vocab.txt"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self._session = ort.InferenceSession(
            os.path.join(self.cache_dir, "model.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        return self

    # -- inference -------------------------------------------------------
    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        self.load()
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = [self._tok.encode(t if t else " ") for t in chunk]
            width = max(len(e[0]) for e in enc)
            ids = np.zeros((len(enc), width), dtype=np.int64)
            mask = np.zeros((len(enc), width), dtype=np.int64)
            for row, (i, m) in enumerate(enc):
                ids[row, :len(i)] = i
                mask[row, :len(m)] = m
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            if hidden.ndim != 3:                      # already pooled
                out.append(np.asarray(hidden))
                continue
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append((pooled / np.clip(norm, 1e-9, None)).astype(np.float32))
        return np.vstack(out)


# ---------------------------------------------------------------------------
# baseline heads (B2 / B3)
# ---------------------------------------------------------------------------
def fit_transformer_heads(X_train: np.ndarray, msgs: list[dict], seed: int = 42) -> dict:
    """Fit the same head set as the main engine on frozen embeddings (§9)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, Ridge

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
    for head, labels in y.items():
        if head == "tension":
            heads[head] = Ridge(alpha=1.0, random_state=seed).fit(X_train, labels)
        elif head in ("sarcasm", "irony", "passive_aggression"):
            base = LogisticRegression(max_iter=1500, C=1.6, class_weight="balanced",
                                      random_state=seed)
            heads[head] = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(
                X_train, labels)
        else:
            heads[head] = LogisticRegression(max_iter=2000, C=4.0,
                                             class_weight="balanced",
                                             random_state=seed).fit(X_train, labels)
    return heads


def context_embeddings(texts: list[str], context_texts: list[str],
                       embedder: MiniLMEmbedder) -> np.ndarray:
    """B3: [utterance embedding ⊕ mean history embedding] (blueprint §11)."""
    u = embedder.encode(texts)
    c = embedder.encode([t if t else " " for t in context_texts])
    return np.hstack([u, c])


if __name__ == "__main__":                        # smoke test
    emb = MiniLMEmbedder(verbose=True).load()
    vecs = emb.encode(["Fine.", "This is fine, thanks!",
                       "I am absolutely furious about this.", "Wow, perfect timing."])
    print("shape:", vecs.shape)
    print("Fine. vs furious  :", round(float(vecs[0] @ vecs[2]), 3))
    print("Fine. vs thanking :", round(float(vecs[0] @ vecs[1]), 3))
    print(json.dumps({"ok": vecs.shape[1] == 384}, indent=2))
