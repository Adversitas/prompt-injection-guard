"""Command line: ``promptguard {benchmark,train,check,serve}``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="promptguard", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="evaluate every detector, write reports/results.*")
    b.add_argument("--out", default="reports")

    t = sub.add_parser("train", help="train the hybrid detector on the train split")
    t.add_argument("--out", default="artifacts/hybrid.joblib")

    c = sub.add_parser("check", help="score one text with a trained detector")
    c.add_argument("text")
    c.add_argument("--model", default="artifacts/hybrid.joblib")
    c.add_argument("--threshold", type=float, default=0.5)

    s = sub.add_parser("serve", help="run the FastAPI guard")
    s.add_argument("--model", default="artifacts/hybrid.joblib")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--trace", default="artifacts/trace.jsonl")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.cmd == "benchmark":
        from .evaluate import run_benchmark, to_markdown

        print(to_markdown(run_benchmark(out_dir=args.out)))
    elif args.cmd == "train":
        from .data import load_split
        from .model import HybridDetector, save

        train = load_split("train")
        det = HybridDetector().fit(train.text, train.label)
        save(det, args.out)
        print(json.dumps({"saved": args.out, "meta_weights": det.meta_weights}))
    elif args.cmd == "check":
        from .guard import Guard
        from .model import load

        verdict = Guard(load(args.model), args.threshold).check(args.text)
        print(json.dumps(asdict(verdict), ensure_ascii=False))
        return 2 if verdict.blocked else 0
    elif args.cmd == "serve":  # pragma: no cover
        import uvicorn

        from .api import create_app
        from .guard import Guard
        from .model import load

        app = create_app(Guard(load(args.model), args.threshold, args.trace))
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
