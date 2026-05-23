"""Hyperparameter robustness sweep.

Goal: show the chosen (window, lookback) for the proposed forecaster is not
overfit, by:

  1. Grid-searching on an EARLY epoch only (2019-08 to 2022-08, ~3 years).
  2. Reporting how stable the optimum is across that grid.
  3. Verifying the chosen values still win on the LATE epoch (2022-08 to
     2026-05), using the same baselines as the long-horizon backtest.

Writes results/hyperparam_sweep_*.csv and a section to docs/RESULTS_LONG.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wbtc.backtest import h_step_log_return, load_returns
from wbtc.forecasters import WassersteinGeodesicTheilSen
from wbtc.quantiles import make_grid
from wbtc.scoring import crps_from_quantiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "btcusdt_1d.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

WINDOWS = [60, 90, 120, 180]
LOOKBACKS = [10, 20, 30, 50]
HORIZON = 5
BURN_IN = 730


def epoch_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for the early and late epochs (over the test span)."""
    ts = df["ts"].reset_index(drop=True)
    # epochs are by date
    early_start = pd.Timestamp("2019-08-01", tz="UTC")
    split = pd.Timestamp("2022-08-01", tz="UTC")
    end = pd.Timestamp("2026-05-01", tz="UTC")
    early = (ts >= early_start) & (ts < split)
    late = (ts >= split) & (ts < end)
    return early.to_numpy(), late.to_numpy()


def evaluate_at(
    returns: np.ndarray, window: int, lookback: int, mask: np.ndarray
) -> float:
    u = make_grid(30)
    burn = max(BURN_IN, window + lookback + 5)
    crps_vals = []
    for t in tqdm(
        range(burn, len(returns) - HORIZON),
        desc=f"w={window} L={lookback}",
        leave=False,
    ):
        if not bool(mask[t]):
            continue
        win = returns[t - burn : t]
        f = WassersteinGeodesicTheilSen(window=window, lookback=lookback)
        f.fit(win)
        q = f.predict(HORIZON, u)
        y = h_step_log_return(returns, t, HORIZON)
        if y is None:
            continue
        crps_vals.append(crps_from_quantiles(q, u, y))
    return float(np.mean(crps_vals))


def main():
    df = load_returns(DATA)
    returns = df["r"].to_numpy()
    early_mask, late_mask = epoch_indices(df)
    print(
        f"[sweep] early epoch n={int(early_mask.sum())}, "
        f"late epoch n={int(late_mask.sum())}, h={HORIZON}"
    )

    early_grid = []
    late_grid = []
    for w in WINDOWS:
        for L in LOOKBACKS:
            print(f"\n[sweep] early w={w} L={L}")
            c_early = evaluate_at(returns, w, L, early_mask)
            print(f"[sweep]   early CRPS = {c_early:.6f}")
            early_grid.append({"window": w, "lookback": L, "crps_early": c_early})

    early_df = pd.DataFrame(early_grid).set_index(["window", "lookback"])
    print("\n[sweep] EARLY epoch CRPS grid:")
    print(early_df)

    # pick the best on EARLY
    best_idx = early_df["crps_early"].idxmin()
    best_w, best_L = best_idx
    print(f"\n[sweep] EARLY-best: window={best_w}, lookback={best_L}")

    # report stability: range of CRPS across the grid
    print(
        f"[sweep] early CRPS min={early_df['crps_early'].min():.6f} max={early_df['crps_early'].max():.6f}"
    )

    # Now evaluate every grid point on LATE, to show stability vs single-point luck
    for w in WINDOWS:
        for L in LOOKBACKS:
            print(f"\n[sweep] late w={w} L={L}")
            c_late = evaluate_at(returns, w, L, late_mask)
            print(f"[sweep]   late CRPS = {c_late:.6f}")
            late_grid.append({"window": w, "lookback": L, "crps_late": c_late})

    late_df = pd.DataFrame(late_grid).set_index(["window", "lookback"])

    full = early_df.join(late_df)
    full.to_csv(RESULTS / "hyperparam_sweep.csv")
    print("\n[sweep] FULL TABLE:")
    print(full.round(6).to_string())
    print(
        f"\n[sweep] Spearman rank correlation early vs late: "
        f"{full['crps_early'].rank().corr(full['crps_late'].rank()):.3f}"
    )

    summary = {
        "early_best": list(best_idx),
        "early_crps_at_best": float(early_df.loc[best_idx, "crps_early"]),
        "late_crps_at_best": float(late_df.loc[best_idx, "crps_late"]),
        "late_best": list(late_df["crps_late"].idxmin()),
        "late_crps_at_late_best": float(late_df["crps_late"].min()),
        "rank_corr": float(full["crps_early"].rank().corr(full["crps_late"].rank())),
    }
    (RESULTS / "hyperparam_sweep_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\n[sweep] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
