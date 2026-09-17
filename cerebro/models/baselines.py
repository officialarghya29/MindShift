"""Baselines (PS-01 §6): measurable references for the full CEREBRO engine.

  B1: TF-IDF + LogisticRegression        (text-only, message-isolated)
  B2: TF-IDF + LinearSVC                 (strong linear text-only)
  B3: TF-IDF + context-window features   (text + conversation context)

All heads share the same metric protocol as the full engine.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from cerebro.features.featurizer import build_vectorizer, vectorize, featurize_messages
from cerebro.context.features_builder import build_conversation_matrix


def _binary_head(seed=42):
    base = LogisticRegression(max_iter=1500, C=1.6, class_weight="balanced",
                              random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


class Baseline:
    """One baseline configuration, all 7 heads."""

    def __init__(self, name: str, use_context=False, linear_svc=False, seed=42):
        self.name = name
        self.use_context = use_context
        self.linear_svc = linear_svc
        self.seed = seed
        self.vec = None
        self.heads = {}

    def _design(self, convs, train: bool):
        msgs = [m for c in convs for m in c]
        if train:
            self.vec = build_vectorizer([m["text"] for m in msgs])
        if self.use_context:
            from scipy.sparse import vstack
            Xs = [build_conversation_matrix(self.vec, c, use_context=True,
                                            use_memory=False, use_behavior=False)[0]
                  for c in convs]
            return vstack(Xs).tocsr(), msgs
        behav, _ = featurize_messages(msgs)
        return vectorize(self.vec, msgs, behav, use_behavior=False), msgs

    def fit(self, train_convs):
        X, msgs = self._design(train_convs, train=True)
        y = self._targets(msgs)
        for head, labels in y.items():
            if head == "tension":
                from sklearn.linear_model import Ridge
                self.heads[head] = Ridge(alpha=1.0, random_state=self.seed).fit(X, labels)
            elif head in ("sarcasm", "irony", "passive_aggression"):
                model = _binary_head(self.seed) if not self.linear_svc else \
                    CalibratedClassifierCV(LinearSVC(C=1.0, random_state=self.seed),
                                           method="sigmoid", cv=3)
                self.heads[head] = model.fit(X, labels)
            else:
                if self.linear_svc:
                    model = CalibratedClassifierCV(
                        LinearSVC(C=1.0, random_state=self.seed, dual="auto"),
                        method="sigmoid", cv=3)
                    self.heads[head] = model.fit(X, labels)
                else:
                    model = LogisticRegression(max_iter=1500, C=4.0,
                                               class_weight="balanced",
                                               random_state=self.seed, n_jobs=-1)
                    self.heads[head] = model.fit(X, labels)
        return self

    def _targets(self, msgs):
        return {
            "sentiment": [m["sentiment"] for m in msgs],
            "emotion": [m["emotion"] for m in msgs],
            "tone": [m["tone"] for m in msgs],
            "tension": [float(m["tension"]) for m in msgs],
            "sarcasm": [int(m["sarcasm"]) for m in msgs],
            "irony": [int(m["irony"]) for m in msgs],
            "passive_aggression": [int(m["passive_aggression"]) for m in msgs],
        }

    def predict_labels(self, convs) -> tuple[dict, list[dict]]:
        """Returns (gold_by_head, pred_by_head) for metric computation."""
        X, msgs = self._design(convs, train=False)
        gold = self._targets(msgs)
        pred = {}
        for head in gold:
            model = self.heads[head]
            if head == "tension":
                pred[head] = list(model.predict(X))
            else:
                pred[head] = list(model.predict(X))
        return gold, pred

    def predict_proba_binary(self, convs) -> dict:
        """Probabilities for ROC/PR-AUC + Brier on binary heads."""
        X, msgs = self._design(convs, train=False)
        out = {}
        for head in ("sarcasm", "irony", "passive_aggression"):
            model = self.heads[head]
            p1 = model.predict_proba(X)[:, 1]
            out[head] = {"p": p1, "y": np.array([int(m[head]) for m in msgs])}
        return out
