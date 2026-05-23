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
    "build_dm_controls",
    "headline_dm_sensitivity",
    "regime_conditional_dm",
    "overall_summary",
    "garch_fallback_rate",
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


def build_dm_controls(
    result: LongRunResult,
    method_a: str,
    method_b: str,
    *,
    control_set: str,
) -> list[np.ndarray]:
    """Construct the residualised-DM control list for one of three named sets.

    ``control_set`` picks between three pre-registered Giacomini-White augmented
    control sets, ordered from least to most powerful. Reported alongside the
    headline residualised DM as a sensitivity column in v0.5 — the
    pre-registered falsification threshold is anchored to ``"vol"`` so the bar
    does not depend on peer-loss correlations.

    - ``"none"``  → empty list (residualised DM with no controls is vanilla DM).
    - ``"vol"``   → ``[y, |y|, y²]`` — direction, magnitude, kurtosis-like
                   moments of the realised return. Three covariates, all
                   predictable at time t and uncorrelated with the EPA mean.
    - ``"full"``  → ``"vol"`` ∪ four peer-method loss series (excluding the
                   two methods under test). The peer losses are admissible
                   under GW (predictable at time t) but rhetorically more
                   endogenous than vol controls — separate column so the
                   reader can see their incremental contribution.
    """
    if control_set == "none":
        return []
    y = np.asarray(result.realised, dtype=float)
    vol = [y, np.abs(y), y * y]
    if control_set == "vol":
        return vol
    if control_set == "full":
        names = list(result.aligned_losses.keys())
        peers = [
            result.aligned_losses[k] for k in names if k != method_a and k != method_b
        ][:4]
        return vol + peers
    raise ValueError(
        f"control_set must be 'none' | 'vol' | 'full', got {control_set!r}"
    )


def pairwise_dm_residualised(
    result: LongRunResult,
    horizon: int,
    controls: list[np.ndarray] | None = None,
    *,
    control_set: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise residualised DM table.

    Uses :func:`diebold_mariano_residualised` to project out shared
    volatility-clustering noise from each loss differential before computing
    the HAC variance, materially raising power versus vanilla DM at long
    horizons where the lag-(h-1) HAC estimator inflates SE substantially.

    Controls
    --------
    Specify the controls in exactly one of three ways:
    - ``controls=<list>`` for an explicit covariate list (legacy interface);
    - ``control_set="none" | "vol" | "full"`` for one of the three pre-
      registered control sets (see :func:`build_dm_controls` for details);
    - leave both as None → defaults to ``control_set="full"`` (vol + peer
      losses), matching the v0.4 / v0.5 headline residualised DM.
    """
    if controls is not None and control_set is not None:
        raise ValueError("pass either controls= or control_set=, not both")
    if controls is None and control_set is None:
        control_set = "full"
    cs: str = control_set or ""  # narrows the type for the inner loop

    names = list(result.aligned_losses.keys())
    dm_stat = pd.DataFrame(index=names, columns=names, dtype=float)
    dm_p = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                dm_stat.loc[a, b] = 0.0
                dm_p.loc[a, b] = 1.0
                continue
            if controls is not None:
                ctrls = controls
            else:
                ctrls = build_dm_controls(result, a, b, control_set=cs)
            if not ctrls:
                # no controls = vanilla DM
                stat, p = diebold_mariano(
                    result.aligned_losses[a],
                    result.aligned_losses[b],
                    h=horizon,
                )
            else:
                stat, p = diebold_mariano_residualised(
                    result.aligned_losses[a],
                    result.aligned_losses[b],
                    ctrls,
                    h=horizon,
                )
            dm_stat.loc[a, b] = stat
            dm_p.loc[a, b] = p
    return dm_stat, dm_p


def headline_dm_sensitivity(
    result: LongRunResult,
    horizon: int,
    head: str,
    baselines: list[str],
) -> pd.DataFrame:
    """Per-cell DM p-values under the three pre-registered control sets.

    For each baseline in ``baselines`` returns one row with the vanilla DM
    statistic and the residualised DM under ``control_set ∈ {"none", "vol",
    "full"}`` (with "none" being identical to vanilla — kept as a column for
    visual alignment). The reader can scan across the row to see how much of
    the residualised-DM lift is driven by vol controls vs. peer losses.
    """
    rows = []
    for ref in baselines:
        row = {"baseline": ref}
        for cs in ("none", "vol", "full"):
            ctrls = build_dm_controls(result, head, ref, control_set=cs)
            if not ctrls:
                stat, p = diebold_mariano(
                    result.aligned_losses[head],
                    result.aligned_losses[ref],
                    h=horizon,
                )
            else:
                stat, p = diebold_mariano_residualised(
                    result.aligned_losses[head],
                    result.aligned_losses[ref],
                    ctrls,
                    h=horizon,
                )
            row[f"dm_stat_{cs}"] = stat
            row[f"dm_p_{cs}"] = p
        rows.append(row)
    return pd.DataFrame(rows).set_index("baseline")


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
    fb_rates = garch_fallback_rate(result)
    for name, losses in result.aligned_losses.items():
        mean, lo, hi = stationary_bootstrap_ci(losses, block_mean=max(2.0, horizon))
        rows.append(
            {
                "method": name,
                "n": len(losses),
                "mean_crps": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "garch_fallback": fb_rates.get(name, float("nan")),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def garch_fallback_rate(result: LongRunResult) -> dict[str, float]:
    """Per-method fraction of walk-forward steps that fell back from GARCH.

    Only methods whose ``_walk_forward_one`` rows carry a ``garch_fallback``
    column (i.e. forecasters that opted into GARCH-conditioned dispersion)
    appear in the output. For others the key is absent. ``WGeo-Hetero`` is
    the headline consumer of this rate — the falsification floor in
    docs/THEORY.md §4 is conditional on the rate being small (otherwise we
    would be measuring the unconditional sqrt(h) scaling, not the GARCH
    contribution).
    """
    rates: dict[str, float] = {}
    for name, df in result.per_step.items():
        if "garch_fallback" not in df.columns:
            continue
        col = df["garch_fallback"].dropna()
        if len(col) == 0:
            continue
        rates[name] = float(col.astype(bool).mean())
    return rates
