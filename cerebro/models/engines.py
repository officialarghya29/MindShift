"""CEREBRO multi-task NLP engine (PS-01 §10–16).

One shared contextual representation → seven specialized heads:
  sentiment (3-class) · emotion (13-class) · tone (14-class) · tension (0-100)
  sarcasm / irony / passive-aggression (binary, probability outputs)

Linear models + tree ensembles on contextual features: fast, calibrated,
and fully explainable via linear coefficients (no black-box requirement).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from cerebro.features.featurizer import (
    build_vectorizer, vectorize, featurize_messages, CachedVectorizer)
from cerebro.features.preprocess import micro_signals
from cerebro.context.features_builder import build_conversation_matrix


def _jsonable(d: dict) -> dict:
    """numpy scalars/strings → plain python (JSON-safe reports)."""
    def _cv(v):
        if hasattr(v, "item") and not isinstance(v, str):
            return v.item()
        if isinstance(v, dict):
            return {str(k): _cv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_cv(x) for x in v]
        if isinstance(v, str):
            return str(v)
        return v
    return {str(k): _cv(v) for k, v in d.items()}


class MultiTaskEngine:
    """All seven heads over one vectorizer. fit() is offline; predict_conversation()
    runs sequentially, re-feeding its own predictions (speaker memory)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.vec = None
        self.heads = {}
        self.trained = False
        self.cfg: dict | None = None   # modality flags the engine was trained with
        self.fusion_weights: dict | None = None  # validation-tuned (PS-01 §24)

    # ---------------- training ----------------
    def fit(self, convs: list[list[dict]], use_context=True, use_memory=True,
            use_behavior=True) -> dict:
        msgs = [m for c in convs for m in c]
        self.vec = build_vectorizer([m["text"] for m in msgs])

        if use_context or use_memory:
            X, _ = self._ctx_matrix(convs, use_context, use_memory, use_behavior)
        else:
            behav, _ = self._behavior(convs)
            X = vectorize(self.vec, msgs, behav, use_behavior=use_behavior)

        self._fit_heads(X, msgs)
        self.trained = True
        self.cfg = {"use_context": use_context, "use_memory": use_memory,
                    "use_behavior": use_behavior}
        return {"n_messages": len(msgs), "n_features": X.shape[1]}

    _NEUTRAL_BINARY = {"probability": 0.0, "confidence": 0.0, "prediction": 0}

    def _fit_heads(self, X, msgs):
        y = {
            "sentiment": [m["sentiment"] for m in msgs],
            "emotion": [m["emotion"] for m in msgs],
            "tone": [m["tone"] for m in msgs],
            "tension": [float(m["tension"]) for m in msgs],
            "sarcasm": [int(m["sarcasm"]) for m in msgs],
            "irony": [int(m["irony"]) for m in msgs],
            "passive_aggression": [int(m["passive_aggression"]) for m in msgs],
        }
        for head, labels in y.items():
            self._fit_one(head, X, labels)

        # tension is a regression (no classes) — handle separately so the
        # single-class skip above can't drop it when all labels are equal
        if "tension" not in self.heads:
            from sklearn.linear_model import Ridge
            self.heads["tension"] = Ridge(alpha=1.0, random_state=self.seed)
            self.heads["tension"].fit(X, y["tension"])

    # ---------------- persistence ----------------
    def save(self, path: str) -> str:
        import joblib
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"vec": self.vec, "heads": self.heads, "seed": self.seed,
                     "cfg": self.cfg, "fusion_weights": self.fusion_weights},
                    path + ".joblib")
        return path + ".joblib"

    def load(self, path: str) -> "MultiTaskEngine":
        import joblib
        blob = joblib.load(path + ".joblib")
        self.vec = blob["vec"]
        self.heads = blob["heads"]
        self.seed = blob["seed"]
        self.cfg = blob.get("cfg")
        self.fusion_weights = blob.get("fusion_weights")
        self.trained = True
        return self

    def _make_head(self, head: str, labels: list):
        seed = self.seed
        if head == "tension":
            # Ridge: empirically beats RF on MAE for this task and calibrates well
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=seed)
        if head in ("sarcasm", "irony", "passive_aggression"):
            # rare positives → class-balanced, calibrated probabilities
            base = LogisticRegression(max_iter=1500, C=1.6,
                                      class_weight="balanced", random_state=seed)
            return CalibratedClassifierCV(base, method="sigmoid", cv=3).fit
        if head in ("sentiment", "emotion", "tone"):
            return LogisticRegression(max_iter=1500, C=4.0,
                                      class_weight="balanced", random_state=seed)
        raise KeyError(head)

    def _fit_one(self, head, X, labels):
        # heads with a single class in the training data (e.g. real corpora
        # without sarcasm/irony/PA/tone annotations) are skipped — the head
        # stays absent and predict_conversation serves a neutral prior
        if len(set(map(str, labels))) < 2:
            self.heads.pop(head, None)
            return None
        model = self._make_head(head, labels)
        if head in ("sarcasm", "irony", "passive_aggression"):
            self.heads[head] = model(X, labels)      # CalibratedClassifierCV.fit
        else:
            self.heads[head] = model.fit(X, labels)
        return self.heads[head]

    # helpers ---------------------------------------------------------------
    def _ctx_matrix(self, convs, use_context, use_memory, use_behavior):
        Xs = []
        for c in convs:
            X, meta = build_conversation_matrix(self.vec, c, use_context,
                                                use_memory, use_behavior)
            Xs.append(X)
        from scipy.sparse import vstack
        return vstack(Xs).tocsr(), meta

    def _behavior(self, convs):
        return featurize_messages([m for c in convs for m in c])

    # ---------------- inference ----------------
    def predict_conversation(self, messages: list[dict], use_context=None,
                             use_memory=None, use_behavior=None) -> list[dict]:
        """Sequential analysis of one conversation. Speaker memory is updated
        with the model's OWN predictions (no gold labels at inference).

        Modality flags default to whatever the engine was trained with
        (self.cfg) so a text-only engine is never queried with feature
        blocks it has never seen."""
        assert self.trained, "engine not trained"
        cfg = self.cfg or {"use_context": True, "use_memory": True,
                           "use_behavior": True}
        use_context = cfg["use_context"] if use_context is None else use_context
        use_memory = cfg["use_memory"] if use_memory is None else use_memory
        use_behavior = cfg["use_behavior"] if use_behavior is None else use_behavior
        from cerebro.context.speaker_memory import SpeakerMemory
        from cerebro.context.context_engine import ContextWindow
        from cerebro.features.preprocess import behavioral_vector
        from scipy.sparse import hstack, csr_matrix

        cw = ContextWindow()
        mem = SpeakerMemory()
        cached = CachedVectorizer(self.vec)
        n_out = len(self.vec.get_feature_names_out())
        # Writable copies: predicted tension is fed back into the context
        # window's "tension-ranked older turns" selection, so uploads (which
        # carry no tension labels) get salience-ranked context exactly like
        # gold-labeled training conversations (PS-01 §8).
        ctx_messages = [dict(m) for m in messages]
        results = []
        for i, m in enumerate(messages):
            behav = np.asarray(behavioral_vector(
                m["text"], messages[i - 1]["timestamp"] if i else None,
                m.get("timestamp"), messages[i - 1]["speaker_id"] if i else None,
                m["speaker_id"]), dtype=float)
            parts = [cached.transform_one(m["text"])]
            ctx = cw.context_text(ctx_messages, i)
            if use_context:
                parts.append(cached.transform_one(ctx)
                             if ctx else csr_matrix((1, n_out)))
            if use_behavior:
                parts.append(csr_matrix(behav.reshape(1, -1)))
            if use_memory:
                from cerebro.context.features_builder import _memory_vector
                mem_vec = _memory_vector(mem, m["speaker_id"], behav).reshape(1, -1)
                parts.append(csr_matrix(mem_vec))
            X = hstack(parts).tocsr()

            sent = self._proba("sentiment", X, use_memory or use_context or use_behavior)
            emo = self._proba("emotion", X)
            tone = self._proba("tone", X)
            tens = float(np.clip(self.heads["tension"].predict(X)[0], 0, 100))
            ctx_messages[i]["tension"] = tens
            sarc = self._binary("sarcasm", X)
            iron = self._binary("irony", X)
            pa = self._binary("passive_aggression", X)

            res = {
                "message_id": m["message_id"],
                "speaker_id": m["speaker_id"],
                "text": m["text"],
                "sentiment": _jsonable(sent),
                "emotion": _jsonable(emo),
                "tone": _jsonable(tone),
                "tension": round(tens, 1),
                "sarcasm": sarc,
                "irony": iron,
                "passive_aggression": pa,
                "signals": micro_signals(m["text"]),
                "context_text": ctx,
                "speaker_state_before": mem.state(m["speaker_id"]),
            }
            results.append(res)
            # advance memory with predicted labels (inference-mode memory)
            mem.observe(m["speaker_id"], emo["label"], tone["label"],
                        sent["label"], tens, bool(behav[14]))
        return results

    def _proba(self, head, X, *_):
        if head not in self.heads:
            return {"label": "neutral", "confidence": 0.0, "probabilities": {}}
        model = self.heads[head]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
        else:  # LinearSVC fallback (decision margin → softmax)
            d = model.decision_function(X)[0]
            e = np.exp(d - d.max())
            p = e / e.sum()
        classes = list(model.classes_)
        idx = int(np.argmax(p))
        top = sorted(zip(classes, p), key=lambda t: -t[1])[:3]
        return {"label": classes[idx], "confidence": round(float(p[idx]), 4),
                "probabilities": {c: round(float(v), 4) for c, v in top}}

    def _binary(self, head, X):
        if head not in self.heads:
            return dict(self._NEUTRAL_BINARY)
        model = self.heads[head]
        p1 = float(model.predict_proba(X)[0][1])
        return {"probability": round(p1, 4),
                "confidence": round(abs(p1 - 0.5) * 2, 4),
                "prediction": int(p1 >= 0.5)}
