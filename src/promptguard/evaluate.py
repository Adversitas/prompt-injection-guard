"""Benchmark of every detector on the official test split + cross-validation on train."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import load_split
from .model import SEED, HybridDetector, TextDetector
from .rules import RuleDetector

DETECTORS = {
    "rules only": lambda: RuleDetector(),
    "tfidf word + logreg": lambda: TextDetector("word"),
    "tfidf char + logreg": lambda: TextDetector("char"),
    "tfidf word+char + logreg": lambda: TextDetector("word+char"),
    "hybrid (word+char stacked with rules)": lambda: HybridDetector("word+char"),
}


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    false_positive_rate: float


def compute_scores(y_true, scores, threshold: float = 0.5) -> Scores:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    negatives = y_true == 0
    fpr = float(y_pred[negatives].mean()) if negatives.any() else 0.0
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else float("nan")
    return Scores(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=float(auc),
        false_positive_rate=fpr,
    )


def cross_validate(make, texts, labels, folds: int = 5) -> dict[str, float]:
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    f1s, aucs = [], []
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, va in cv.split(texts, labels):
        det = make().fit(list(texts[tr]), labels[tr])
        s = compute_scores(labels[va], det.score(list(texts[va])))
        f1s.append(s.f1)
        aucs.append(s.roc_auc)
    return {
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_auc_mean": float(np.mean(aucs)),
    }


def latency_ms(det, texts, repeats: int = 3) -> dict[str, float]:
    """Single-text latency (the guard is called once per user message)."""
    timings = []
    for _ in range(repeats):
        for t in texts:
            start = time.perf_counter()
            det.score([t])
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def run_benchmark(data_dir=None, out_dir: Path | str = "reports") -> dict:
    kwargs = {"data_dir": data_dir} if data_dir else {}
    train, test = load_split("train", **kwargs), load_split("test", **kwargs)
    results = {
        "dataset": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_injection_share": float(train.label.mean()),
            "test_injection_share": float(test.label.mean()),
        },
        "detectors": {},
    }
    for name, make in DETECTORS.items():
        det = make().fit(train.text, train.label)
        test_scores = compute_scores(test.label, det.score(test.text))
        row = {"test": asdict(test_scores)}
        row.update(cross_validate(make, train.text, train.label))
        row.update(latency_ms(det, test.text.tolist()[:50]))
        if isinstance(det, HybridDetector):
            row["meta_weights"] = det.meta_weights
        results["detectors"][name] = row

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(results), encoding="utf-8")
    return results


def to_markdown(results: dict) -> str:
    lines = [
        "| Detector | Test P | Test R | Test F1 | Test FPR | Test AUC | CV F1 (train, 5-fold) "
        "| p50 latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results["detectors"].items():
        t = r["test"]
        lines.append(
            f"| {name} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | "
            f"{t['false_positive_rate']:.3f} | {t['roc_auc']:.3f} | "
            f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f} | {r['p50_ms']:.2f} ms |"
        )
    return "\n".join(lines) + "\n"
