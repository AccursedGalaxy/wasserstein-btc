"""Fetch BTC/USDT daily OHLCV from Binance via ccxt and cache as parquet.

Idempotent: a second run only fetches missing tail bars.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = DATA_DIR / "btcusdt_1d.parquet"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
# Binance BTC/USDT listed 2017-08-17. Use Aug 17 as ms-epoch start.
SINCE_MS = ccxt.binance().parse8601("2017-08-17T00:00:00Z")
LIMIT = 1000  # Binance daily cap per call


def fetch_all(exchange: ccxt.Exchange, since_ms: int) -> pd.DataFrame:
    rows: list[list[float]] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=cursor, limit=LIMIT)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 24 * 60 * 60 * 1000  # next day in ms
        time.sleep(exchange.rateLimit / 1000.0)
        # avoid runaway loop
        if cursor > exchange.milliseconds():
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def main() -> int:
    exchange = ccxt.binance({"enableRateLimit": True})
    if PARQUET.exists():
        existing = pd.read_parquet(PARQUET)
        last_ts = existing["ts"].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        print(f"[fetch] resuming from {last_ts.isoformat()}")
        new = fetch_all(exchange, since_ms)
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
        print(f"[fetch] cold start from 2017-08-17")
        df = fetch_all(exchange, SINCE_MS)

    df.to_parquet(PARQUET, index=False)
    print(
        f"[fetch] wrote {PARQUET} "
        f"rows={len(df)} from {df['ts'].min().date()} to {df['ts'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
