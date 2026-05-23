"""Long-horizon backtest with per-year and per-regime breakdowns.

Unlike `backtest.compare_methods`, this evaluates on ALL days after the
initial 730-day burn-in (no separate holdout), so the "test set" is multi-
year and lets us measure stability across regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtest import h_step_log_return
from .quantiles import make_grid
from .scoring import crps_from_quantiles, diebold_mariano, stationary_bootstrap_ci

__all__ = ["LongRunResult", "run_long_horizon", "tag_regimes", "by_year", "by_regime"]


@dataclass
class LongRunResult:
    per_step: dict[str, pd.DataFrame]
    aligned_idx: list[int]
    aligned_losses: dict[str, np.ndarray]
    realised: np.ndarray


def run_long_horizon(
    returns: np.ndarray,
    timestamps: pd.Series,  # one per return
    method_factories: dict[str, Callable[[], object]],
    burn_in: int,
    horizon: int,
    K: int = 30,
) -> LongRunResult:
    u = make_grid(K)
    per_step: dict[str, pd.DataFrame] = {}
    for name, fac in method_factories.items():
        rows = []
        for t in tqdm(range(burn_in, len(returns) - horizon), desc=name, leave=False):
            window = returns[:t] if t < burn_in + 1 else returns[t - burn_in : t]
            f = fac()
            try:
                f.fit(window)  # type: ignore[attr-defined]
                q = f.predict(horizon, u)  # type: ignore[attr-defined]
            except Exception:
                rows.append({"t": t, "crps": np.nan, "y": np.nan})
                continue
            y = h_step_log_return(returns, t, horizon)
            if y is None:
                continue
            rows.append({"t": t, "crps": crps_from_quantiles(q, u, y), "y": y})
        per_step[name] = pd.DataFrame(rows)

    # align
    common = None
    for df in per_step.values():
        idx = set(df["t"].tolist())
        common = idx if common is None else common & idx
    common = sorted(common or set())
    aligned = {
        name: df.set_index("t").loc[common, "crps"].to_numpy()
        for name, df in per_step.items()
    }
    realised = next(iter(per_step.values())).set_index("t").loc[common, "y"].to_numpy()
    return LongRunResult(
        per_step=per_step, aligned_idx=common, aligned_losses=aligned, realised=realised
    )


def by_year(
    result: LongRunResult,
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Per-year mean CRPS per method, plus row counts."""
    ts = timestamps.iloc[result.aligned_idx].reset_index(drop=True)
    years = ts.dt.year.to_numpy()
    rows = []
    for y in sorted(set(years)):
        mask = years == y
        if mask.sum() < 30:
            continue
        row = {"year": int(y), "n": int(mask.sum())}
        for name, losses in result.aligned_losses.items():
            row[name] = float(np.mean(losses[mask]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def tag_regimes(returns: np.ndarray, lookback: int = 60) -> np.ndarray:
    """Tag each step as 'crash', 'rally', 'low-vol', 'high-vol', or 'neutral'.

    Computed from the trailing `lookback`-day window of returns:
      - cumulative return < -25% over lookback -> crash
      - cumulative return > +25% over lookback -> rally
      - else realised vol percentile >= 80% -> high-vol
      - else realised vol percentile <= 20% -> low-vol
      - else neutral
    """
    n = len(returns)
    tags = np.array(["neutral"] * n, dtype=object)
    # realised vol at each t
    vol = np.full(n, np.nan, dtype=float)
    cum = np.full(n, np.nan, dtype=float)
    for t in range(lookback, n):
        w = returns[t - lookback : t]
        vol[t] = float(np.std(w) * np.sqrt(252))
        cum[t] = float(w.sum())
    vol_lo = np.nanpercentile(vol[lookback:], 20)
    vol_hi = np.nanpercentile(vol[lookback:], 80)
    for t in range(lookback, n):
        if cum[t] < -0.25:
            tags[t] = "crash"
        elif cum[t] > 0.25:
            tags[t] = "rally"
        elif vol[t] >= vol_hi:
            tags[t] = "high-vol"
        elif vol[t] <= vol_lo:
            tags[t] = "low-vol"
        else:
            tags[t] = "neutral"
    return tags


def by_regime(
    result: LongRunResult,
    regime_tags: np.ndarray,
) -> pd.DataFrame:
    """Per-regime mean CRPS per method."""
    tags = regime_tags[result.aligned_idx]
    rows = []
    for regime in ["crash", "high-vol", "neutral", "low-vol", "rally"]:
        mask = tags == regime
        if mask.sum() < 20:
            continue
        row = {"regime": regime, "n": int(mask.sum())}
        for name, losses in result.aligned_losses.items():
            row[name] = float(np.mean(losses[mask]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def pairwise_dm(
    result: LongRunResult, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(result.aligned_losses.keys())
    dm_stat = pd.DataFrame(index=names, columns=names, dtype=float)
    dm_p = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                dm_stat.loc[a, b] = 0.0
                dm_p.loc[a, b] = 1.0
                continue
            stat, p = diebold_mariano(
                result.aligned_losses[a], result.aligned_losses[b], h=horizon
            )
            dm_stat.loc[a, b] = stat
            dm_p.loc[a, b] = p
    return dm_stat, dm_p


def overall_summary(result: LongRunResult, horizon: int) -> pd.DataFrame:
    rows = []
    for name, losses in result.aligned_losses.items():
        mean, lo, hi = stationary_bootstrap_ci(losses, block_mean=max(2.0, horizon))
        rows.append(
            {
                "method": name,
                "n": len(losses),
                "mean_crps": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows).set_index("method")
