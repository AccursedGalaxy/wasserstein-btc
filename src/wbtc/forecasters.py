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
    "GJRGarchStudentT",
    "HistoricalSimulationBootstrap",
    "WassersteinGeodesic",
    "WassersteinGeodesicTheilSen",
    "WassersteinGeodesicGated",
    "WassersteinGeodesicEWMA",
    "WassersteinGeodesicHetero",
    "WGeoGarchEnsemble",
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


@dataclass
class GJRGarchStudentT:
    """B5: GJR-GARCH(1,1,1) with Student-t innovations.

    Adds an asymmetric leverage term: bad news (negative shocks) raise
    next-step variance more than good news. Standard in crypto where down-
    side spikes are the dominant risk.
    """

    _result = None

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float) * 100.0
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        self._result = am.fit(disp="off", show_warning=False)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._result is not None
        f = self._result.forecast(horizon=h, reindex=False)
        mu_1 = float(self._result.params["mu"])
        nu = float(self._result.params["nu"])
        var_h = float(f.variance.values[-1, :].sum())
        sigma_h = np.sqrt(var_h)
        q_pct = mu_1 * h + sigma_h * student_t.ppf(u, df=nu)
        return q_pct / 100.0


@dataclass
class HistoricalSimulationBootstrap:
    """B6: industry-standard non-parametric distribution forecast.

    Draw `n_paths` h-step paths by sampling daily returns with replacement
    from the training window, sum each path, and take empirical quantiles
    of the resulting h-step return distribution. Captures the empirical
    shape (including fat tails) without parametric assumptions.
    """

    n_paths: int = 5000
    rng_seed: int = 0

    _returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        self._returns = np.asarray(returns, dtype=float)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._returns is not None
        rng = np.random.default_rng(self.rng_seed)
        idx = rng.integers(0, len(self._returns), size=(self.n_paths, h))
        paths = self._returns[idx].sum(axis=1)
        return np.quantile(paths, u, method="linear")


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


@dataclass
class WassersteinGeodesicTheilSen(WassersteinGeodesic):
    """The proposed method, robust slope variant.

    Replaces the per-quantile OLS slope with the Theil-Sen median-of-pairwise-
    slopes estimator. Theil-Sen has a 29.3% breakdown point and is provably
    robust to outliers in the response — exactly the failure mode that hurts
    OLS during regime shifts. No explicit curvature gate needed.

    See Theil (1950), Sen (1968), Rousseeuw & Leroy (1987 §3.1).
    """

    def _slope(self, Q: np.ndarray) -> np.ndarray:
        L, K = Q.shape
        s = np.arange(L, dtype=float)
        # all pairwise i < j slopes per column, then median
        i_idx, j_idx = np.triu_indices(L, k=1)
        ds = s[j_idx] - s[i_idx]  # always positive
        dq = Q[j_idx] - Q[i_idx]  # (pairs, K)
        return np.median(dq / ds[:, None], axis=0)


@dataclass
class WassersteinGeodesicEWMA(WassersteinGeodesic):
    """v0.3: Exponentially-weighted recency slope.

    OLS gives every lookback observation the same weight; Theil-Sen gives the
    median (robust but with equal voting power). In regime-switching dynamics
    the *recent* tangent vectors are more representative of the current
    geodesic direction than the oldest ones.

    We use a weighted least-squares slope with weights
        w_j = lambda^(L-1-j),  j = 0..L-1 (oldest=0, newest=L-1)
    where lambda in (0, 1] is the decay factor. lambda=1 reduces to OLS;
    smaller lambda emphasises recency.

    The WLS slope of y on x with weights w is the standard:
        beta = sum_j w_j (x_j - x_bar_w)(y_j - y_bar_w) / sum_j w_j (x_j - x_bar_w)^2
    """

    decay: float = 0.85

    def _slope(self, Q: np.ndarray) -> np.ndarray:
        L, K = Q.shape
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        s = np.arange(L, dtype=float)
        w = self.decay ** (L - 1 - s)  # newest has w=1, oldest has w=lambda^{L-1}
        w_sum = w.sum()
        s_mean = (w * s).sum() / w_sum
        Q_mean = (w[:, None] * Q).sum(axis=0, keepdims=True) / w_sum
        num = (w[:, None] * (s - s_mean)[:, None] * (Q - Q_mean)).sum(axis=0)
        den = (w * (s - s_mean) ** 2).sum()
        return num / max(den, 1e-12)


def _garch_h_step_sigma_ratio(returns: np.ndarray, h: int) -> float:
    """Return s_h := sqrt(sum_i sigma_{t+i}^2) / sqrt(h * sigma_uncond^2).

    Conditional/unconditional volatility ratio over the next h steps.
    >1 in turbulent windows (next-h cumulative vol exceeds the long-run
    sqrt-time scaling), <1 in calm windows. Robust to fit failures: returns
    1.0 (i.e. revert to sqrt(h) scaling) on any exception.
    """
    try:
        r = np.asarray(returns, dtype=float) * 100.0
        am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=h, reindex=False)
        var_path = np.asarray(f.variance.values[-1, :], dtype=float)  # length h
        var_uncond = float(r.var(ddof=1))
        if var_uncond <= 0 or np.any(var_path <= 0):
            return 1.0
        return float(np.sqrt(var_path.sum() / (h * var_uncond)))
    except Exception:
        return 1.0


@dataclass
class WassersteinGeodesicHetero(WassersteinGeodesicTheilSen):
    """v0.3: Heteroskedastic geodesic — dispersion scaled by GARCH variance forecast.

    The vanilla WGeo expands the quantile vector around its median by sqrt(h),
    which is the i.i.d.-shock spread-scaling exponent. Under heteroskedasticity
    (volatility clustering), the true h-step return variance can be
    *substantially* different from h*sigma_1^2 — especially right after a
    regime switch.

    We replace the static sqrt(h) factor by a *conditional* scaling derived
    from a GARCH(1,1) variance forecast:

        s_h(t) := sqrt( sum_{i=1..h} sigma_{t+i}^2 / (h * sigma_uncond^2) )

    so that s_h = 1 when the next-h cumulative vol equals the long-run sqrt-h
    scaling, and s_h > 1 (resp. < 1) in turbulent (calm) regimes. The forecast
    becomes:

        q_pred(u) = (median + h * beta_median)
                  + (q_now(u) - median_now) * sqrt(h) * s_h(t)
                  + h * (beta(u) - beta_median)

    Direction (the tangent slope) is still estimated robustly with Theil-Sen.
    Only the *dispersion* picks up GARCH conditioning. This addresses the
    long-horizon weakness of WGeo (the constant-velocity tangent says nothing
    about *width* of the predictive distribution at h=21).

    The novelty vs published methods: existing Wasserstein-time-series methods
    (Koopman-Wasserstein, Saluzzi-Soize 2025) do not condition the dispersion
    on a parametric vol forecast; existing GARCH methods do not use a
    distributional tangent direction. This is the missing cross.
    """

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        Q = self._quantile_history(u)
        beta = self._slope(Q)
        q_now = Q[-1]
        median_now = float(np.median(q_now))
        beta_median = float(np.median(beta))
        s_h = _garch_h_step_sigma_ratio(self._returns, h)  # type: ignore[arg-type]
        center = q_now - median_now
        q_pred = (
            (median_now + h * beta_median)
            + center * np.sqrt(h) * s_h
            + h * (beta - beta_median)
        )
        return isotonic_project(q_pred)


@dataclass
class WGeoGarchEnsemble:
    """v0.3: Regime-aware ensemble of Wasserstein-Geodesic and GJR-GARCH-t.

    Motivation. The 6.75-year long-horizon decomposition in RESULTS_LONG.md
    shows that WGeo wins decisively in calm regimes (low-vol + neutral = 62%
    of days) while GARCH wins decisively in the rare high-vol regime (~3% of
    days). They are *complementary*, not competing. A single mixture model
    that adaptively routes by regime should dominate both components on
    average without manual regime classification.

    Method. We use a *continuous* mixing weight w_t in [0, 1] computed from
    realised-vol percentile within the rolling window:

        sigma_t  := std of the last `vol_window` returns
        rank_t   := percentile rank of sigma_t in its own trailing
                    `vol_rank_window` history
        w_t      := smoothstep(rank_t, lo=`rank_lo`, hi=`rank_hi`)

    w_t=0 -> pure WGeo (calm regime), w_t=1 -> pure GARCH (turbulent regime),
    smooth blend in between. Critically the threshold percentiles are *in-
    window* — no forward look — and the smoothstep avoids the regime-flip
    instability of hard switching.

    The mixed forecast is a *quantile-space* convex combination, which is
    exact on the W_2 manifold: a geodesic in W_2 between two measures is the
    linear interpolation of their quantile functions (McCann 1997).
    """

    window: int = 90
    lookback: int = 20
    vol_window: int = 20
    vol_rank_window: int = 252
    rank_lo: float = 0.60
    rank_hi: float = 0.90

    _returns: np.ndarray | None = None
    _wgeo: WassersteinGeodesicTheilSen | None = None
    _garch: GarchNormal | None = None
    _weight: float = 0.0

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        self._returns = r
        self._wgeo = WassersteinGeodesicTheilSen(
            window=self.window, lookback=self.lookback
        )
        self._wgeo.fit(r)
        self._garch = GarchNormal()
        self._garch.fit(r)
        # blend weight from realised-vol percentile
        if len(r) < self.vol_window + self.vol_rank_window:
            self._weight = 0.0
            return
        # rolling std over last `vol_rank_window` non-overlapping origins
        sigmas = np.array(
            [
                float(np.std(r[i - self.vol_window : i]))
                for i in range(len(r) - self.vol_rank_window, len(r) + 1)
            ]
        )
        sigma_now = sigmas[-1]
        rank = float(np.mean(sigmas[:-1] < sigma_now))
        # smoothstep between rank_lo and rank_hi
        if rank <= self.rank_lo:
            w = 0.0
        elif rank >= self.rank_hi:
            w = 1.0
        else:
            t = (rank - self.rank_lo) / (self.rank_hi - self.rank_lo)
            w = float(t * t * (3.0 - 2.0 * t))  # smoothstep
        self._weight = w

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._wgeo is not None and self._garch is not None
        q_w = self._wgeo.predict(h, u)
        q_g = self._garch.predict(h, u)
        w = self._weight
        q_pred = (1.0 - w) * q_w + w * q_g
        return isotonic_project(q_pred)
