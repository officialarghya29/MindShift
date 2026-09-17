"""Message featurization (PS-01 §7): text n-grams + behavioral vector."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from cerebro.features.preprocess import process_text, behavioral_vector, BEHAVIORAL_DIMS


def _prev_ts(messages, i):
    return messages[i - 1]["timestamp"] if i else None


def featurize_messages(messages: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Dense behavioral features only (context engines consume these)."""
    feats = []
    for i, m in enumerate(messages):
        feats.append(behavioral_vector(m["text"], _prev_ts(messages, i),
                                       m.get("timestamp"),
                                       messages[i - 1]["speaker_id"] if i else None,
                                       m["speaker_id"]))
    return np.asarray(feats, dtype=float), [m["text"] for m in messages]


def build_vectorizer(texts: list[str], ngram=(1, 2), min_df=2, max_features=12000) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features,
                          sublinear_tf=True, lowercase=True)
    vec.fit([process_text(t) for t in texts])
    return vec


class CachedVectorizer:
    """Memoizes transform() per raw string — context windows overlap heavily,
    so consecutive messages re-transform nearly identical context text."""

    def __init__(self, vec: TfidfVectorizer, cache_size: int = 200_000):
        self.vec = vec
        self._cache: dict[str, object] = {}
        self._cache_size = cache_size

    def transform_one(self, text: str):
        key = text
        hit = self._cache.get(key)
        if hit is None:
            hit = self.vec.transform([process_text(key)])
            if len(self._cache) < self._cache_size:
                self._cache[key] = hit
        return hit

    def transform(self, texts: list[str]):
        from scipy.sparse import vstack
        if isinstance(texts, str):
            texts = [texts]
        return vstack([self.transform_one(t) for t in texts])

    def transform_many(self, texts: list[str]):
        return self.transform(texts)

    def get_feature_names_out(self, *args, **kwargs):
        return self.vec.get_feature_names_out(*args, **kwargs)

    @property
    def n_features(self) -> int:
        return len(self.vec.get_feature_names_out())


def vectorize(vec: TfidfVectorizer, messages: list[dict], behavioral: np.ndarray,
              use_behavior: bool = True):
    """TF-IDF text features ⊕ behavioral vector → design matrix."""
    X_text = vec.transform([process_text(m["text"]) for m in messages])
    if use_behavior:
        behav = csr_matrix(behavioral)
        return hstack([X_text, behav]).tocsr()
    return X_text.tocsr()


def feature_names(vec: TfidfVectorizer) -> list[str]:
    return list(vec.get_feature_names_in()) + [f"behavioral_{i}" for i in range(BEHAVIORAL_DIMS)]
