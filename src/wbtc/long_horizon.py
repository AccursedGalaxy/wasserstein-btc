"""Long-horizon backtest with per-year and per-regime breakdowns.

Unlike :func:`wbtc.backtest.compare_methods`, this evaluates on ALL days
after the initial 730-day burn-in (no separate holdout), so the "test set"
is multi-year and lets us measure stability across regimes. The inner
walk-forward loop is the same one used by ``compare_methods`` — see
:func:`wbtc.backtest._walk_forward_one`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import _align_methods, _walk_forward_one
from .forecasters import Forecaster
from .quantiles import make_grid
from .scoring import (
    diebold_mariano,
    diebold_mariano_residualised,
    stationary_bootstrap_ci,
)

__all__ = [
    "LongRunResult",
    "run_long_horizon",
    "tag_regimes",
    "by_year",
    "by_regime",
    "pairwise_dm",
    "pairwise_dm_residualised",
    "regime_conditional_dm",
    "overall_summary",
]


@dataclass
class LongRunResult:
    per_step: dict[str, pd.DataFrame]
    aligned_idx: list[int]
    aligned_losses: dict[str, np.ndarray]
    realised: np.ndarray


def run_long_horizon(
    returns: np.ndarray,
    timestamps: pd.Series,  # one per return
    method_factories: dict[str, Callable[[], Forecaster]],
    burn_in: int,
    horizon: int,
    K: int = 30,
    n_jobs: int = 1,
    stride: int = 1,
) -> LongRunResult:
    u = make_grid(K)
    if n_jobs == 1:
        per_step = {
            name: _walk_forward_one(
                returns,
                fac,
                train_window=burn_in,
                horizon=horizon,
                u=u,
                stride=stride,
                show_progress=True,
                label=name,
            )
            for name, fac in method_factories.items()
        }
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_walk_forward_one)(
                returns,
                fac,
                train_window=burn_in,
                horizon=horizon,
                u=u,
                stride=stride,
                show_progress=False,
                label=name,
            )
            for name, fac in method_factories.items()
        )
        per_step = dict(zip(method_factories.keys(), results))

    common_idx, aligned, realised = _align_methods(per_step)
    return LongRunResult(
        per_step=per_step,
        aligned_idx=common_idx,
        aligned_losses=aligned,
        realised=realised,
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


def pairwise_dm_residualised(
    result: LongRunResult,
    horizon: int,
    controls: list[np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise residualised DM table.

    Uses :func:`diebold_mariano_residualised` to project out shared
    volatility-clustering noise from each loss differential before computing
    the HAC variance, materially raising power versus vanilla DM at long
    horizons where the lag-(h-1) HAC estimator inflates SE substantially.

    Controls
    --------
    If ``controls`` is None, uses two sets of natural common-noise covariates:
    (a) ``|y|`` and ``y²`` (realised return moments — every method's loss
    co-moves with the day's absolute return size), and (b) per-method losses
    other than the pair under test (panel covariates capturing common
    forecast-error structure across methods). The combination dominates
    cell-vol-noise variance reduction without affecting the test mean.
    """
    names = list(result.aligned_losses.keys())
    dm_stat = pd.DataFrame(index=names, columns=names, dtype=float)
    dm_p = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                dm_stat.loc[a, b] = 0.0
                dm_p.loc[a, b] = 1.0
                continue
            if controls is None:
                # peer-method losses (excl. the two under test) + realised moments
                peer_ctrls = [
                    result.aligned_losses[k] for k in names if k != a and k != b
                ][:4]
                y = np.asarray(result.realised, dtype=float)
                ctrls = peer_ctrls + [np.abs(y), y * y]
            else:
                ctrls = controls
            stat, p = diebold_mariano_residualised(
                result.aligned_losses[a],
                result.aligned_losses[b],
                ctrls,
                h=horizon,
            )
            dm_stat.loc[a, b] = stat
            dm_p.loc[a, b] = p
    return dm_stat, dm_p


def regime_conditional_dm(
    result: LongRunResult,
    regime_tags: np.ndarray,
    method_a: str,
    method_b: str,
    horizon: int,
) -> pd.DataFrame:
    """DM stat + p-value for ``method_a`` vs ``method_b`` per regime.

    The aggregate DM at long horizons averages over very different regimes —
    WGeo dominates GARCH families in crash/rally regimes by 5-10% but ties in
    neutral/low-vol regimes (where Static is already near-optimal). The
    per-regime DM separates the signal from the noise and shows where the
    distributional method has its real advantage.

    Returns a small DataFrame with columns ``n, mean_a, mean_b, delta_pct,
    dm, p``, indexed by regime.
    """
    tags = regime_tags[result.aligned_idx]
    la = result.aligned_losses[method_a]
    lb = result.aligned_losses[method_b]
    rows = []
    for regime in ["crash", "high-vol", "neutral", "low-vol", "rally"]:
        mask = tags == regime
        if mask.sum() < max(30, horizon * 2):
            continue
        la_r = la[mask]
        lb_r = lb[mask]
        dm, p = diebold_mariano(la_r, lb_r, h=horizon)
        rows.append(
            {
                "regime": regime,
                "n": int(mask.sum()),
                "mean_a": float(la_r.mean()),
                "mean_b": float(lb_r.mean()),
                "delta_pct": float((la_r.mean() - lb_r.mean()) / lb_r.mean() * 100),
                "dm": float(dm),
                "p": float(p),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


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
