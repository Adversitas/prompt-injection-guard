"""Learned detectors and a stacked hybrid (classifier + rules)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .rules import RuleDetector

SEED = 13


def build_text_model(kind: str = "char") -> Pipeline:
    """TF-IDF + logistic regression. ``kind`` in {word, char, word+char}."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    if kind == "word":
        features = word
    elif kind == "char":
        features = char
    elif kind == "word+char":
        features = FeatureUnion([("word", word), ("char", char)])
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    clf = LogisticRegression(C=8.0, max_iter=4000, class_weight="balanced", random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


class HybridDetector:
    """Stacks the out-of-fold probability of a text model with the rule score.

    The meta-model is a 2-feature logistic regression trained on out-of-fold predictions, so the
    weight given to each layer is learned from data instead of hand-tuned.
    """

    def __init__(self, kind: str = "word+char", folds: int = 5) -> None:
        self.kind = kind
        self.folds = folds
        self.text_model = build_text_model(kind)
        self.rules = RuleDetector()
        self.meta = LogisticRegression(class_weight="balanced", random_state=SEED)

    def _stack(self, p_text: np.ndarray, texts) -> np.ndarray:
        return np.column_stack([p_text, np.asarray(self.rules.score(texts))])

    def fit(self, texts, labels) -> HybridDetector:
        texts = list(texts)
        labels = np.asarray(labels)
        cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=SEED)
        oof = cross_val_predict(
            build_text_model(self.kind), texts, labels, cv=cv, method="predict_proba"
        )[:, 1]
        self.meta.fit(self._stack(oof, texts), labels)
        self.text_model.fit(texts, labels)
        return self

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        p_text = self.text_model.predict_proba(texts)[:, 1]
        return self.meta.predict_proba(self._stack(p_text, texts))[:, 1]

    @property
    def meta_weights(self) -> dict[str, float]:
        coef = self.meta.coef_[0]
        return {"text_model": float(coef[0]), "rules": float(coef[1])}


class TextDetector:
    """Plain text model exposing the same ``score`` interface as the other detectors."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.model = build_text_model(kind)

    def fit(self, texts, labels) -> TextDetector:
        self.model.fit(list(texts), np.asarray(labels))
        return self

    def score(self, texts) -> np.ndarray:
        return self.model.predict_proba(list(texts))[:, 1]


def save(detector, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)


def load(path: Path | str):
    return joblib.load(path)
