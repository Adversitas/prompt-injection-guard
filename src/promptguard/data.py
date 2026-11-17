"""Loading of the deepset/prompt-injections benchmark (Apache-2.0, committed under data/)."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPLITS = ("train", "test")


def normalize(text: str) -> str:
    """Unicode-normalise and collapse whitespace so that cosmetic tricks don't change the score."""
    text = unicodedata.normalize("NFKC", str(text))
    # zero-width characters are a classic way to split trigger words
    for zw in ("​", "‌", "‍", "﻿", "⁠"):
        text = text.replace(zw, "")
    return " ".join(text.split())


def load_split(split: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame with columns ``text`` (normalised) and ``label`` (1 = injection)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{split}.parquet is missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {"text": df["text"].map(normalize), "label": df["label"].astype(int)}
    ).reset_index(drop=True)
    return out
