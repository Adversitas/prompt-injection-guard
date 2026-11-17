"""Runtime guard: verdict + reasons + latency, with optional JSONL tracing for observability."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .data import normalize
from .rules import matched_rules

MAX_CHARS = 8000


@dataclass
class Verdict:
    blocked: bool
    score: float
    threshold: float
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    truncated: bool = False


class Guard:
    """Wraps any detector exposing ``score(list[str])``.

    Raw user text is never written to the trace: only a SHA-256 prefix, its length and the
    verdict, so the log can be shipped to a monitoring stack without leaking user data.
    """

    def __init__(self, detector, threshold: float = 0.5, trace_path: Path | str | None = None):
        self.detector = detector
        self.threshold = threshold
        self.trace_path = Path(trace_path) if trace_path else None

    def check(self, text: str) -> Verdict:
        start = time.perf_counter()
        clean = normalize(text)
        truncated = len(clean) > MAX_CHARS
        clean = clean[:MAX_CHARS]
        score = float(self.detector.score([clean])[0])
        reasons = [r.name for r in matched_rules(clean)]
        verdict = Verdict(
            blocked=score >= self.threshold,
            score=round(score, 4),
            threshold=self.threshold,
            reasons=reasons,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            truncated=truncated,
        )
        self._trace(clean, verdict)
        return verdict

    def _trace(self, text: str, verdict: Verdict) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "chars": len(text),
            **asdict(verdict),
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
