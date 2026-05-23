"""Data loading + simple discovery helpers.

The on-disk format is one Parquet file per symbol, written by
`scripts/fetch_data.py` (`wbtc fetch`). This module provides agent-friendly
read helpers — no surprises, no network calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATA_DIR",
    "available_symbols",
    "load_returns",
    "load_ohlcv",
    "DataInfo",
    "data_info",
]


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def _parquet_for(symbol: str) -> Path:
    return DATA_DIR / f"{_slug(symbol)}_1d.parquet"


def available_symbols() -> list[str]:
    """List symbols that already have cached parquet data on disk."""
    out: list[str] = []
    for p in sorted(DATA_DIR.glob("*_1d.parquet")):
        stem = p.stem.removesuffix("_1d")
        # heuristic: insert "/" before "usdt"/"usdc"/"btc"/"eth" at the tail
        for quote in ("usdt", "usdc", "busd", "btc", "eth"):
            if stem.endswith(quote) and len(stem) > len(quote):
                base = stem[: -len(quote)]
                out.append(f"{base.upper()}/{quote.upper()}")
                break
        else:
            out.append(stem.upper())
    return out


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load the full OHLCV history for `symbol`. Raises FileNotFoundError if missing."""
    pq = _parquet_for(symbol)
    if not pq.exists():
        raise FileNotFoundError(
            f"no cached data for {symbol} at {pq}. Run: wbtc fetch {symbol}"
        )
    df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
    return df


def load_returns(symbol: str) -> pd.DataFrame:
    """Load returns frame for `symbol`. Columns: ts, close, log_close, r."""
    df = load_ohlcv(symbol)
    df["log_close"] = np.log(df["close"].astype(float))
    df["r"] = df["log_close"].diff()
    return df.dropna(subset=["r"]).reset_index(drop=True)


@dataclass
class DataInfo:
    symbol: str
    n_rows: int
    first_date: str
    last_date: str
    sha256_8: str  # first 8 hex chars of file sha256 — a stable provenance tag
    path: str


def data_info(symbol: str) -> DataInfo:
    pq = _parquet_for(symbol)
    if not pq.exists():
        raise FileNotFoundError(f"{pq} does not exist; run `wbtc fetch {symbol}`")
    df = load_ohlcv(symbol)
    h = hashlib.sha256(pq.read_bytes()).hexdigest()[:8]
    return DataInfo(
        symbol=symbol,
        n_rows=len(df),
        first_date=str(df["ts"].iloc[0].date()),
        last_date=str(df["ts"].iloc[-1].date()),
        sha256_8=h,
        path=str(pq),
    )
