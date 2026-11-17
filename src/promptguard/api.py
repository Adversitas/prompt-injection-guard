"""FastAPI service placed in front of an LLM / RAG endpoint."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .guard import Guard


class GuardRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


def create_app(guard: Guard) -> FastAPI:
    app = FastAPI(title="promptguard", version="0.1.0")
    counters: Counter[str] = Counter()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "threshold": guard.threshold}

    @app.post("/guard")
    def check(req: GuardRequest) -> dict:
        verdict = guard.check(req.text)
        counters["requests"] += 1
        counters["blocked"] += int(verdict.blocked)
        return asdict(verdict)

    @app.get("/metrics")
    def metrics() -> dict:
        total = counters["requests"]
        return {
            "requests": total,
            "blocked": counters["blocked"],
            "block_rate": counters["blocked"] / total if total else 0.0,
        }

    return app


def app_from_env() -> FastAPI:  # pragma: no cover - used by `uvicorn --factory`
    from .model import load

    detector = load(os.environ.get("PROMPTGUARD_MODEL", "artifacts/hybrid.joblib"))
    threshold = float(os.environ.get("PROMPTGUARD_THRESHOLD", "0.5"))
    return create_app(Guard(detector, threshold, os.environ.get("PROMPTGUARD_TRACE")))
