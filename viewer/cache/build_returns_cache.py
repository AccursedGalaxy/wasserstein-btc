"""Generate viewer/cache/{returns,ohlcv}.json from the parquets in data/.

Two small JSON caches, used by the viewer when the parquet files
themselves are unavailable (CI runners, where Binance returns HTTP 451):

* ``returns.json`` (~200 KB): daily log-returns per asset. Without it, the
  residualised-DM controls |y|, y² are dropped on CI, costing three h=21
  cells in the headline DM count (8/12 → 5/12).
* ``ohlcv.json`` (~700 KB): columnar daily OHLCV per asset, feeding the
  "underlying markets" candlestick + return charts. Without it those
  charts render empty on the deployed site.

Run locally whenever you refresh the parquets:

    uv run wbtc fetch BTC/USDT ETH/USDT SOL/USDT BNB/USDT
    uv run python viewer/cache/build_returns_cache.py
    git add viewer/cache/returns.json && git commit -m "viewer: refresh returns cache"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "returns.json"
OUT_OHLCV = Path(__file__).resolve().parent / "ohlcv.json"

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def _sig(x: float, digits: int) -> float:
    """Round to ``digits`` significant figures so the JSON stays compact."""
    return float(f"{float(x):.{digits}g}")


def main() -> None:
    cache: dict[str, list[float]] = {}
    ohlcv: dict[str, dict[str, list]] = {}
    for sym in SYMBOLS:
        pq = DATA / f"{_slug(sym)}_1d.parquet"
        if not pq.exists():
            print(f"  skipping {sym} (no parquet)")
            continue
        df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
        # Match _log_returns_from_parquet in build_data.py: first-difference
        # of log close, length N-1.
        log_close = np.log(df["close"].astype(float).to_numpy())
        r = np.diff(log_close)
        cache[sym] = [round(float(x), 8) for x in r]
        # Columnar OHLCV. 8 significant figures keeps BTC to the cent and
        # sub-dollar assets (XRP) to well below a tick; volume needs less.
        ohlcv[sym] = {
            "t": [str(pd.to_datetime(t).date()) for t in df["ts"].tolist()],
            **{
                k: [_sig(x, 8) for x in df[col].to_numpy()]
                for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))
            },
            "v": [_sig(x, 6) for x in df["volume"].to_numpy()],
        }
        print(f"  {sym}: {len(r)} daily returns, {len(df)} candles")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB, {len(cache)} assets)")

    OUT_OHLCV.write_text(json.dumps(ohlcv, separators=(",", ":")))
    size_kb = OUT_OHLCV.stat().st_size / 1024
    print(f"wrote {OUT_OHLCV} ({size_kb:.1f} KB, {len(ohlcv)} assets)")


if __name__ == "__main__":
    main()
