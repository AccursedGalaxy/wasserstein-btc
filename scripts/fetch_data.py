"""Fetch OHLCV for one or more USDT pairs from Binance via ccxt.

Idempotent: re-runs only fetch missing tail bars.
Usage:
    python scripts/fetch_data.py                        # default: 1d, all panel symbols
    python scripts/fetch_data.py BTC/USDT ADA/USDT
    python scripts/fetch_data.py --timeframe 5m BTC/USDT   # intraday (data/btcusdt_5m.parquet)
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
DEFAULT_TIMEFRAME = "1d"
LIMIT = 1000  # Binance cap per call

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 3_600_000,
    "4h": 4 * 3_600_000,
    "1d": 86_400_000,
}

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


def parquet_for(symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> Path:
    return DATA_DIR / f"{slug(symbol)}_{timeframe}.parquet"


def fetch_all(
    exchange: ccxt.Exchange, symbol: str, since_ms: int, timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    step_ms = TIMEFRAME_MS[timeframe]
    rows: list[list[float]] = []
    cursor = since_ms
    n_calls = 0
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=LIMIT)
        n_calls += 1
        if n_calls % 100 == 0:
            print(f"[fetch] {symbol} {timeframe}: {n_calls} calls, {len(rows)} rows", flush=True)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + step_ms  # next bar in ms
        time.sleep(exchange.rateLimit / 1000.0)
        if cursor > exchange.milliseconds():
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def drop_unclosed(df: pd.DataFrame, timeframe: str, now_ms: int) -> pd.DataFrame:
    """Drop the bar that is still open: ``ts`` is the bar *open* time, so a
    bar is closed only when ``ts + step <= now``. Writing an open bar freezes
    a mid-bar snapshot as if it were the close."""
    step = pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe])
    now = pd.Timestamp(now_ms, unit="ms", tz="UTC")
    return df[df["ts"] + step <= now].reset_index(drop=True)


def fetch_one(
    exchange: ccxt.Exchange, symbol: str, timeframe: str = DEFAULT_TIMEFRAME
) -> None:
    pq = parquet_for(symbol, timeframe)
    if pq.exists():
        existing = pd.read_parquet(pq)
        last_ts = existing["ts"].max()
        # Re-fetch the last two bars so a bar that was still open when it was
        # last written gets replaced by its finalised version (keep="last").
        since_ms = int(last_ts.timestamp() * 1000) - 2 * TIMEFRAME_MS[timeframe]
        print(f"[fetch] {symbol}: resuming from {last_ts.isoformat()} (minus 2 bars)")
        new = fetch_all(exchange, symbol, since_ms, timeframe)
        if not new.empty:
            df = (
                pd.concat([existing, new], ignore_index=True)
                .drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        else:
            df = existing
    else:
        listed = LISTED_SINCE.get(symbol, "2017-08-17T00:00:00Z")
        print(f"[fetch] {symbol}: cold start from {listed}")
        since_ms = exchange.parse8601(listed)
        df = fetch_all(exchange, symbol, since_ms, timeframe)

    df = drop_unclosed(df, timeframe, exchange.milliseconds())
    df.to_parquet(pq, index=False)
    print(
        f"[fetch] {symbol}: wrote {pq.name} "
        f"rows={len(df)} {df['ts'].min().date()} → {df['ts'].max().date()}"
    )


def main(argv: list[str]) -> int:
    timeframe = DEFAULT_TIMEFRAME
    args = list(argv)
    if "--timeframe" in args:
        i = args.index("--timeframe")
        timeframe = args[i + 1]
        del args[i : i + 2]
    if timeframe not in TIMEFRAME_MS:
        print(f"unsupported timeframe {timeframe!r}; choose from {sorted(TIMEFRAME_MS)}")
        return 2
    symbols = args if args else DEFAULT_SYMBOLS
    exchange = ccxt.binance({"enableRateLimit": True})
    for s in symbols:
        fetch_one(exchange, s, timeframe)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
