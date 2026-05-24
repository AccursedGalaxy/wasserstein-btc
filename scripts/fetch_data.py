"""Fetch daily OHLCV for one or more USDT pairs from Binance via ccxt.

Idempotent: re-runs only fetch missing tail bars.
Usage:
    python scripts/fetch_data.py                  # default: BTC, ETH, SOL
    python scripts/fetch_data.py BTC/USDT ADA/USDT
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAME = "1d"
LIMIT = 1000  # Binance daily cap per call

# When each symbol was listed on Binance (give plenty of cushion).
LISTED_SINCE = {
    "BTC/USDT": "2017-08-17T00:00:00Z",
    "ETH/USDT": "2017-08-17T00:00:00Z",
    "SOL/USDT": "2020-08-11T00:00:00Z",
    "BNB/USDT": "2017-11-06T00:00:00Z",
    "ADA/USDT": "2018-04-17T00:00:00Z",
    "XRP/USDT": "2018-05-04T00:00:00Z",
}


def slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def parquet_for(symbol: str) -> Path:
    return DATA_DIR / f"{slug(symbol)}_1d.parquet"


def fetch_all(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> pd.DataFrame:
    rows: list[list[float]] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=cursor, limit=LIMIT)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 24 * 60 * 60 * 1000  # next day in ms
        time.sleep(exchange.rateLimit / 1000.0)
        if cursor > exchange.milliseconds():
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def fetch_one(exchange: ccxt.Exchange, symbol: str) -> None:
    pq = parquet_for(symbol)
    if pq.exists():
        existing = pd.read_parquet(pq)
        last_ts = existing["ts"].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        print(f"[fetch] {symbol}: resuming from {last_ts.isoformat()}")
        new = fetch_all(exchange, symbol, since_ms)
        if not new.empty:
            df = (
                pd.concat([existing, new], ignore_index=True)
                .drop_duplicates(subset="ts")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        else:
            df = existing
    else:
        listed = LISTED_SINCE.get(symbol, "2017-08-17T00:00:00Z")
        print(f"[fetch] {symbol}: cold start from {listed}")
        since_ms = exchange.parse8601(listed)
        df = fetch_all(exchange, symbol, since_ms)

    df.to_parquet(pq, index=False)
    print(
        f"[fetch] {symbol}: wrote {pq.name} "
        f"rows={len(df)} {df['ts'].min().date()} → {df['ts'].max().date()}"
    )


def main(argv: list[str]) -> int:
    symbols = argv if argv else DEFAULT_SYMBOLS
    exchange = ccxt.binance({"enableRateLimit": True})
    for s in symbols:
        fetch_one(exchange, s)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
