"""Generate viewer/cache/returns.json from the parquets in data/.

A small (~200 KB) JSON cache of daily log-returns per asset, used by the
viewer when the parquet files themselves are unavailable (CI runners,
where Binance returns HTTP 451). Without this cache, the residualised-DM
controls |y|, y² are dropped on CI, costing three h=21 cells in the
headline DM count (8/12 → 5/12).

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

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def main() -> None:
    cache: dict[str, list[float]] = {}
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
        print(f"  {sym}: {len(r)} daily returns")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB, {len(cache)} assets)")


if __name__ == "__main__":
    main()
