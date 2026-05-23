"""Distributional forecasters.

All forecasters expose:

    fit(returns: np.ndarray) -> None        # train on a window of past returns
    predict(h: int, u: np.ndarray) -> np.ndarray   # h-step quantile forecast on grid u

A "forecast" is therefore a quantile vector on a fixed grid, encoding a
1D probability measure under the W_2 identification used in `quantiles.py`.

Note: `fit` is called fresh at each walk-forward step, so the forecasters
intentionally do not maintain state across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arch import arch_model
from scipy.stats import norm, t as student_t

from .quantiles import empirical_quantiles, isotonic_project

__all__ = [
    "StaticEmpirical",
    "RandomWalkDrift",
    "GarchNormal",
    "GarchStudentT",
    "WassersteinGeodesic",
    "WassersteinGeodesicGated",
]


# ----------------------------- baselines -------------------------------


@dataclass
class StaticEmpirical:
    """B1: future window distribution == current window distribution."""

    _returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        self._returns = np.asarray(returns, dtype=float)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._returns is not None
        # h-day return ~ sum of h i.i.d. draws from the empirical 1-day dist.
        # Under StaticEmpirical we approximate by scaling quantiles: this is
        # not exact for non-Gaussian, but is the canonical naive baseline.
        q1 = empirical_quantiles(self._returns, u)
        mu = self._returns.mean()
        # mean scales linearly, deviations scale as sqrt(h) (Gaussian-like)
        return mu * h + (q1 - mu) * np.sqrt(h)


@dataclass
class RandomWalkDrift:
    """B2: empirical 1-day distribution, mean shifted by h * drift."""

    _returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        self._returns = np.asarray(returns, dtype=float)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._returns is not None
        q1 = empirical_quantiles(self._returns, u)
        mu = self._returns.mean()
        return mu * h + (q1 - mu) * np.sqrt(h)


@dataclass
class GarchNormal:
    """B3: GARCH(1,1) with Gaussian innovations, fit by MLE."""

    _result = None
    _last_returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float) * 100.0  # arch wants percent
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        self._result = am.fit(disp="off", show_warning=False)
        self._last_returns = r

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._result is not None
        f = self._result.forecast(horizon=h, reindex=False)
        # mean of h-step cumulative return = h * mu
        mu_1 = float(self._result.params["mu"])
        # variance of h-step cumulative return ≈ sum of forecast variances
        var_h = float(f.variance.values[-1, :].sum())
        sigma_h = np.sqrt(var_h)
        # quantiles in percent, then convert back to log-return
        q_pct = mu_1 * h + sigma_h * norm.ppf(u)
        return q_pct / 100.0


@dataclass
class GarchStudentT:
    """B4: GARCH(1,1) with Student-t innovations."""

    _result = None

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float) * 100.0
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        self._result = am.fit(disp="off", show_warning=False)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._result is not None
        f = self._result.forecast(horizon=h, reindex=False)
        mu_1 = float(self._result.params["mu"])
        nu = float(self._result.params["nu"])
        var_h = float(f.variance.values[-1, :].sum())
        sigma_h = np.sqrt(var_h)
        # quantile of standardised Student-t (variance = nu/(nu-2)), so rescale
        # arch parameterises so the innovations have unit variance
        q_pct = mu_1 * h + sigma_h * student_t.ppf(u, df=nu)
        return q_pct / 100.0


# ----------------------- the proposed method ---------------------------


@dataclass
class WassersteinGeodesic:
    """Pure tangent-space W2-geodesic forecaster (no regime gating).

    Parameters
    ----------
    window
        Length n of the rolling window used to estimate each empirical
        quantile vector q_t.
    lookback
        Number of past quantile vectors used to estimate the tangent slope.
    """

    window: int = 90
    lookback: int = 30

    _returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        if len(returns) < self.window + self.lookback:
            raise ValueError(
                f"need >= {self.window + self.lookback} returns, got {len(returns)}"
            )
        self._returns = np.asarray(returns, dtype=float)

    def _quantile_history(self, u: np.ndarray) -> np.ndarray:
        """Build the (lookback, K) matrix of rolling quantile vectors ending at t."""
        assert self._returns is not None
        n = self.window
        r = self._returns
        Q = np.empty((self.lookback, len(u)), dtype=float)
        # most recent quantile vector uses last `n` returns; previous one uses
        # the window shifted back by 1, etc.
        for j in range(self.lookback):
            start = len(r) - n - j
            stop = len(r) - j
            Q[self.lookback - 1 - j] = empirical_quantiles(r[start:stop], u)
        return Q  # rows ordered oldest -> newest

    def _slope(self, Q: np.ndarray) -> np.ndarray:
        """OLS slope of each column of Q against time index s = 0..L-1."""
        L = Q.shape[0]
        s = np.arange(L, dtype=float)
        s_mean = s.mean()
        s_var = ((s - s_mean) ** 2).sum()
        Q_mean = Q.mean(axis=0, keepdims=True)
        return ((s - s_mean)[:, None] * (Q - Q_mean)).sum(axis=0) / s_var

    def _h_step_scale(self, h: int) -> float:
        """Scale forecast spread to h-day returns under sqrt-time aggregation.

        The geodesic-velocity slope `beta` is in units of (return / day) per day,
        i.e. how much each quantile moves per day. For an h-step forecast we
        accumulate h days of drift but rescale dispersion by sqrt(h) under the
        weak assumption that 1-day shocks are roughly uncorrelated.
        """
        return float(h)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        Q = self._quantile_history(u)
        beta = self._slope(Q)
        q_now = Q[-1]
        median_now = float(np.median(q_now))
        # split into level (median) and shape; drift the level by h*beta_median,
        # widen the shape by sqrt(h).
        beta_median = float(np.median(beta))
        center = q_now - median_now
        q_pred = (
            (median_now + h * beta_median)
            + center * np.sqrt(h)
            + h * (beta - beta_median)
        )
        return isotonic_project(q_pred)


@dataclass
class WassersteinGeodesicGated(WassersteinGeodesic):
    """The proposed method: WassersteinGeodesic + regime-curvature gate.

    Parameters
    ----------
    kappa_star
        Curvature threshold. Above this the forecast is blended with the
        static-empirical fallback. Continuous gating with linear ramp.
    tau
        Lag (in steps) used when measuring two consecutive tangent vectors
        for curvature estimation. tau=5 -> weekly-ish tangents.
    """

    kappa_star: float = 0.6
    tau: int = 5

    def _curvature(self, u: np.ndarray) -> float:
        assert self._returns is not None
        r = self._returns
        n = self.window
        tau = self.tau
        if len(r) < n + 2 * tau:
            return 0.0
        q_now = empirical_quantiles(r[-n:], u)
        q_mid = empirical_quantiles(r[-n - tau : -tau], u)
        q_old = empirical_quantiles(r[-n - 2 * tau : -2 * tau], u)
        v1 = q_now - q_mid
        v2 = q_mid - q_old
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos = float(np.dot(v1, v2) / (n1 * n2))
        return 1.0 - cos  # in [0, 2]

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        q_geo = super().predict(h, u)
        # static-empirical fallback (sqrt-time scaled)
        q_now = empirical_quantiles(self._returns, u)  # type: ignore[arg-type]
        mu = float(self._returns.mean())  # type: ignore[union-attr]
        q_static = mu * h + (q_now - mu) * np.sqrt(h)
        # blend weight
        kappa = self._curvature(u)
        w = max(0.0, min(1.0, kappa / self.kappa_star - 1.0))
        q_pred = (1.0 - w) * q_geo + w * q_static
        return isotonic_project(q_pred)
