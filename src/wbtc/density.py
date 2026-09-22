"""Intraday-density data object: one quantile vector per UTC day.

This module builds the *data object* of the intraday-density track described
in ``docs/PREREG.md`` (protocol (i)): for each UTC day, a Gaussian kernel
density estimate of that day's 5-minute log-returns, converted to a length-K
quantile vector on a common support. It is deliberately independent of every
forecaster in :mod:`wbtc.forecasters` — pre-reg v1.0 says no forecaster is
built until Gate 1 passes, and this module contains none.

The construction constants below are the "current candidate" values listed
under *Things this pre-reg does NOT freeze* in ``docs/PREREG.md`` v1.0. They
are frozen by the v1.1 amendment in the same document; edit them only with a
dated amendment.

Pipeline per day
----------------
1. ``r_i = log(close_i) - log(close_{i-1})`` for consecutive candles inside
   the UTC day (no overnight gap; the first candle of the day contributes no
   return). A full day has 288 candles → 287 returns.
2. Days with fewer than ``MIN_OBS_PER_DAY`` returns (exchange outage, partial
   listing day, partial last day) are flagged ``is_excluded`` with a reason.
   Their quantile vector is still computed when at least 2 returns exist so
   the row is inspectable, but downstream scripts drop excluded rows.
3. Gaussian KDE with Silverman's rule-of-thumb bandwidth
   ``h = 0.9 * min(sd, IQR / 1.34) * n^(-1/5)`` (falls back to ``sd`` when
   the IQR is zero, and to ``MIN_BANDWIDTH`` when both are zero).
4. The KDE CDF is evaluated in closed form on a fine grid over the common
   support ``[SUPPORT_LO, SUPPORT_HI]``, renormalised to that support, and
   inverted by linear interpolation at the mid-point probability grid
   ``u_k = (k - 0.5) / K``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

__all__ = [
    "K_QUANTILES",
    "SUPPORT_LO",
    "SUPPORT_HI",
    "GRID_POINTS",
    "MIN_OBS_PER_DAY",
    "MIN_BANDWIDTH",
    "DensitySpec",
    "DEFAULT_SPEC",
    "silverman_bandwidth",
    "kde_quantiles",
    "intraday_log_returns",
    "build_daily_densities",
]

K_QUANTILES = 100
SUPPORT_LO = -0.20
SUPPORT_HI = 0.20
GRID_POINTS = 8001  # step 5e-5 on [-0.2, 0.2]; ~1/4 of a typical 5-min bandwidth
MIN_OBS_PER_DAY = 230  # 80% of the 287 returns a complete 5-min day yields
MIN_BANDWIDTH = 1e-6  # floor when a day's returns are (numerically) constant


@dataclass(frozen=True)
class DensitySpec:
    k: int = K_QUANTILES
    support_lo: float = SUPPORT_LO
    support_hi: float = SUPPORT_HI
    grid_points: int = GRID_POINTS
    min_obs_per_day: int = MIN_OBS_PER_DAY
    min_bandwidth: float = MIN_BANDWIDTH

    def u_grid(self) -> np.ndarray:
        return (np.arange(self.k) + 0.5) / self.k

    def x_grid(self) -> np.ndarray:
        return np.linspace(self.support_lo, self.support_hi, self.grid_points)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "support": [self.support_lo, self.support_hi],
            "grid_points": self.grid_points,
            "min_obs_per_day": self.min_obs_per_day,
            "min_bandwidth": self.min_bandwidth,
            "bandwidth_rule": "silverman: 0.9 * min(sd, IQR/1.34) * n^(-1/5)",
            "kernel": "gaussian",
            "u_grid": "(k - 0.5) / K, k = 1..K",
        }


DEFAULT_SPEC = DensitySpec()


def silverman_bandwidth(x: np.ndarray, min_bandwidth: float = MIN_BANDWIDTH) -> float:
    """Silverman (1986) rule-of-thumb bandwidth for a Gaussian kernel."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return min_bandwidth
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25) / 1.34
    scale = min(sd, iqr) if iqr > 0 else sd
    h = 0.9 * scale * n ** (-0.2)
    return max(h, min_bandwidth)


def kde_quantiles(
    x: np.ndarray, spec: DensitySpec = DEFAULT_SPEC, bandwidth: float | None = None
) -> tuple[np.ndarray, float]:
    """Quantile vector of the Gaussian KDE of ``x`` on the common support.

    Returns ``(q, h)`` where ``q`` has length ``spec.k`` and is non-decreasing,
    and ``h`` is the bandwidth used. The KDE CDF is renormalised to the
    support, so the quantiles are those of the KDE *conditional on* the
    support; mass outside ``[support_lo, support_hi]`` is discarded.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 observations for a KDE")
    h = silverman_bandwidth(x, spec.min_bandwidth) if bandwidth is None else bandwidth
    xg = spec.x_grid()
    # closed-form KDE CDF: mean_i Phi((x_grid - x_i) / h), chunked to bound memory
    cdf = np.zeros_like(xg)
    chunk = 4096
    for start in range(0, x.size, chunk):
        xi = x[start : start + chunk]
        cdf += ndtr((xg[:, None] - xi[None, :]) / h).sum(axis=1)
    cdf /= x.size
    lo, hi = cdf[0], cdf[-1]
    if hi - lo <= 0:
        raise ValueError("KDE has no mass on the common support")
    cdf = (cdf - lo) / (hi - lo)
    # make strictly increasing for interpolation (ties can occur in flat tails)
    cdf = np.maximum.accumulate(cdf)
    u = spec.u_grid()
    q = np.interp(u, cdf, xg)
    q = np.maximum.accumulate(q)
    return q, float(h)


def intraday_log_returns(
    ohlcv: pd.DataFrame, ts_col: str = "ts", close_col: str = "close"
) -> pd.DataFrame:
    """Within-day log-returns of consecutive candles, keyed by UTC day.

    Returns a frame with columns ``day`` (``datetime.date``) and ``r``. The
    first candle of each day contributes no return (no overnight gap).
    """
    df = ohlcv[[ts_col, close_col]].dropna().sort_values(ts_col)
    ts = pd.to_datetime(df[ts_col], utc=True)
    day = ts.dt.date.to_numpy()
    logc = np.log(df[close_col].to_numpy(dtype=float))
    r = np.diff(logc)
    same_day = day[1:] == day[:-1]
    return pd.DataFrame({"day": day[1:][same_day], "r": r[same_day]})


def build_daily_densities(
    ohlcv: pd.DataFrame,
    spec: DensitySpec = DEFAULT_SPEC,
    exclude_days: frozenset | None = None,
) -> pd.DataFrame:
    """One row per UTC day present in ``ohlcv``.

    Columns: ``day``, ``n_obs``, ``bandwidth``, ``quantile`` (length-K array),
    ``is_excluded``, ``exclusion_reason``. ``exclude_days`` is an optional
    calendar of days to flag regardless of coverage (e.g. a depeg calendar).
    """
    exclude_days = exclude_days or frozenset()
    rets = intraday_log_returns(ohlcv)
    # days with only one candle yield zero returns but must still appear
    all_days = sorted(set(pd.to_datetime(ohlcv["ts"], utc=True).dt.date))
    grouped = {d: g["r"].to_numpy() for d, g in rets.groupby("day")}
    rows = []
    for d in all_days:
        r = grouped.get(d, np.empty(0))
        n = int(r.size)
        reason = ""
        if d in exclude_days:
            reason = "calendar"
        elif n < spec.min_obs_per_day:
            reason = f"short_day:{n}<{spec.min_obs_per_day}"
        if n >= 2:
            q, h = kde_quantiles(r, spec)
        else:
            q, h = np.full(spec.k, np.nan), float("nan")
        rows.append(
            {
                "day": d,
                "n_obs": n,
                "bandwidth": h,
                "quantile": q,
                "is_excluded": bool(reason),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)
