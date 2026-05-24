"""Produce a compact v0.3 summary from the per-step JSON in results/.

Reads results/long_<symbol>_h{h}.json (which the long backtest writes),
computes mean CRPS per method, the best WGeo-family member, the best
baseline, the pairwise DM p-value, and prints a Markdown table + a
plaintext one-line-per-row form suitable for Telegram.

Run after `wbtc backtest-long` has populated results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wbtc.report import slug
from wbtc.scoring import diebold_mariano

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

WGEO_VARIANTS = [
    "WGeo",
    "WGeo-Gated",
    "WGeo-TheilSen",
    "WGeo-EWMA",
    "WGeo-Hetero",
    "WGeo-GARCH-Ens",
]
BASELINE_VARIANTS = [
    "Static",
    "RW-Drift",
    "HS-Bootstrap",
    "GARCH-N",
    "GARCH-t",
    "GJR-GARCH-t",
]

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
HORIZONS = [1, 5, 21]


def summarize() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for h in HORIZONS:
            pth = RESULTS / f"long_{slug(sym)}_h{h}.json"
            if not pth.exists():
                continue
            d = json.loads(pth.read_text())
            losses = {k: np.array(v) for k, v in d.items() if k != "t_idx"}
            mean_crps = {k: float(np.mean(v)) for k, v in losses.items()}
            wgeo_best = min(WGEO_VARIANTS, key=lambda n: mean_crps[n])
            base_best = min(BASELINE_VARIANTS, key=lambda n: mean_crps[n])
            stat, p = diebold_mariano(losses[wgeo_best], losses[base_best], h=h)
            rows.append(
                {
                    "symbol": sym,
                    "h": h,
                    "n": len(losses[wgeo_best]),
                    "best_wgeo": wgeo_best,
                    "best_baseline": base_best,
                    "wgeo_crps": mean_crps[wgeo_best],
                    "baseline_crps": mean_crps[base_best],
                    "improvement_pct": (mean_crps[wgeo_best] - mean_crps[base_best])
                    / mean_crps[base_best]
                    * 100.0,
                    "dm_p": p,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = summarize()
    if df.empty:
        print("no results yet")
        return
    print("\n=== v0.3 headline ===\n")
    print(df.to_markdown(index=False, floatfmt=".5f"))
    # compact Telegram form
    print("\n=== telegram lines ===\n")
    for _, r in df.iterrows():
        sig = "**" if r["dm_p"] < 0.05 else ""
        print(
            f"{sig}{r['symbol']:<9} h={int(r['h']):>2}: {r['best_wgeo']:<16}"
            f" beat {r['best_baseline']:<14} by {r['improvement_pct']:+.2f}% "
            f"(DM p={r['dm_p']:.3g}, n={int(r['n'])}){sig}"
        )


if __name__ == "__main__":
    main()
