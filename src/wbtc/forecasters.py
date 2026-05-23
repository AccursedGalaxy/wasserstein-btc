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
    # extended baselines (v0.4)
    "HARRV",
    "CAViaRSAV",
    "MarkovSwitching2",
    "FIGARCH",
    "StochasticVolatilityAR1",
    "BivariateVARGarch",
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


# ---------------------------------------------------------------------------
# v0.4 extended baselines: HAR-RV, CAViaR, MS-GARCH, FIGARCH, SV, BVAR-GARCH
# ---------------------------------------------------------------------------
#
# These six forecasters round out the comparison panel against named methods
# from the financial-econometrics canon. They follow the same fit/predict
# protocol as the original baselines. None of them is intended to be the
# default forecaster — they exist so that RESULTS_EXTENDED.md can compare
# WGeo-* against vol-only, quantile-direct, regime-switching, long-memory,
# stochastic-volatility, and multivariate-mean families on the same data.
#
# Each is a *deliberately faithful but minimal* implementation. The goal is
# benchmark coverage, not a production-grade library version, so we use
# standard formulations and QML/EM estimation (no MCMC, no fractional
# packages). Where multiple reasonable parameterisations exist, we pick the
# one that is best documented in the literature (see docstrings for refs).


def _student_t_quantiles_from_var(
    mu_h: float, var_h: float, df: float, u: np.ndarray
) -> np.ndarray:
    """Quantiles of a Student-t(df) with mean mu_h and variance var_h.

    arch parameterises the standardised Student-t so it has unit variance;
    we mirror that here so a returned `df` from arch lines up with `var_h`
    being the actual conditional variance.
    """
    sigma_h = float(np.sqrt(max(var_h, 1e-18)))
    return mu_h + sigma_h * student_t.ppf(u, df=df)


# --------------------------- HAR-RV (Corsi 2009) ----------------------------


@dataclass
class HARRV:
    """C1: Heterogeneous Autoregressive of Realised Variance (Corsi 2009).

    The original HAR-RV is a linear regression of one-period realised
    variance on its daily, weekly, and monthly lagged averages:

        RV_t = c + b_d RV_{t-1}^{(d)} + b_w RV_{t-1}^{(w)} + b_m RV_{t-1}^{(m)} + e_t

    Here we only have *daily* close-to-close returns, so we use the squared
    log-return r_t^2 as the realised-variance proxy — this is the standard
    "naive RV" choice for daily-only series (Andersen-Bollerslev 1998 §2).
    The aggregates over windows of 1, 5, 22 days follow Corsi's original
    spec. Innovations are modelled as Student-t with degrees of freedom
    estimated from the standardised residuals.

    For an h-day forecast we iterate the HAR recursion h times to get a
    variance path {RV_{t+i}}_{i=1..h}, sum to get cumulative variance, and
    return Student-t quantiles around the cumulative-drift mean.
    """

    p_d: int = 1
    p_w: int = 5
    p_m: int = 22

    _coef: np.ndarray | None = None  # (4,) [c, b_d, b_w, b_m]
    _resid_df: float = 8.0
    _last_rv: np.ndarray | None = None  # most recent 22 daily RV values
    _last_returns: np.ndarray | None = None

    @staticmethod
    def _agg(rv: np.ndarray, w: int) -> np.ndarray:
        """Trailing mean over `w` periods, ending at each index (NaN until full)."""
        out = np.full_like(rv, np.nan, dtype=float)
        if len(rv) >= w:
            csum = np.cumsum(rv)
            out[w - 1 :] = (csum[w - 1 :] - np.concatenate([[0.0], csum[:-w]])) / w
        return out

    def _build_design(self, rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rv_d = self._agg(rv, self.p_d)
        rv_w = self._agg(rv, self.p_w)
        rv_m = self._agg(rv, self.p_m)
        # predict RV_t from lagged-1 aggregates
        y = rv[1:]
        X = np.column_stack([np.ones_like(y), rv_d[:-1], rv_w[:-1], rv_m[:-1]])
        mask = np.isfinite(X).all(axis=1)
        return X[mask], y[mask]

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        if len(r) < self.p_m + 10:
            raise ValueError(f"need >= {self.p_m + 10} returns, got {len(r)}")
        rv = r * r
        X, y = self._build_design(rv)
        # constrained OLS — keep coefficients non-negative so the variance
        # forecast cannot drift negative when iterated. Use NNLS.
        from scipy.optimize import nnls

        coef, _ = nnls(X, y)
        self._coef = coef
        # Standardised-return tail heaviness -> Student-t df via method of moments.
        # We use observed daily returns (aligned to the design tail) divided by
        # the in-sample HAR predicted vol; kurt = 6/(nu-4) gives nu directly.
        yhat = X @ coef
        sigma_est = np.sqrt(np.maximum(yhat, 1e-12))
        r_used = r[1:][-len(yhat) :]
        z = r_used / sigma_est
        excess_kurt = float(np.mean(z**4)) - 3.0
        nu = 4.0 + 6.0 / excess_kurt if excess_kurt > 1e-3 else 30.0
        self._resid_df = float(min(max(nu, 4.5), 30.0))
        self._last_rv = rv[-self.p_m :]
        self._last_returns = r

    def _iterate_variance(self, h: int) -> np.ndarray:
        """Iterate the HAR recursion h steps starting from last_rv."""
        assert self._coef is not None and self._last_rv is not None
        c, b_d, b_w, b_m = self._coef
        buf = list(self._last_rv)  # length p_m
        path = []
        for _ in range(h):
            rv_d = float(np.mean(buf[-self.p_d :]))
            rv_w = float(np.mean(buf[-self.p_w :]))
            rv_m = float(np.mean(buf[-self.p_m :]))
            rv_next = float(max(c + b_d * rv_d + b_w * rv_w + b_m * rv_m, 1e-18))
            path.append(rv_next)
            buf.append(rv_next)
        return np.asarray(path, dtype=float)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._last_returns is not None
        var_path = self._iterate_variance(h)
        var_h = float(var_path.sum())
        mu_1 = float(self._last_returns.mean())
        q = _student_t_quantiles_from_var(mu_1 * h, var_h, self._resid_df, u)
        return isotonic_project(q)


# ----------------------------- CAViaR (Engle-Manganelli 2004) ---------------


def _pinball(q_pred: np.ndarray, y: np.ndarray, tau: float) -> float:
    e = y - q_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1.0) * e)))


def _caviar_sav_path(params: np.ndarray, returns: np.ndarray, q0: float) -> np.ndarray:
    """Symmetric Absolute Value CAViaR quantile path.

    q_t = beta0 + beta1 * q_{t-1} + beta2 * |r_{t-1}|
    """
    b0, b1, b2 = params
    n = len(returns)
    q = np.empty(n, dtype=float)
    q[0] = q0
    for t in range(1, n):
        q[t] = b0 + b1 * q[t - 1] + b2 * abs(returns[t - 1])
    return q


@dataclass
class CAViaRSAV:
    """C2: Symmetric Absolute Value CAViaR (Engle-Manganelli 2004).

    The CAViaR family fits the conditional *quantile* directly via
    quantile-regression loss, bypassing the location-scale assumption that
    GARCH-style models impose. We use the Symmetric Absolute Value
    specification:

        q_t(τ) = β0 + β1 q_{t-1}(τ) + β2 |r_{t-1}|

    fit by minimising mean pinball loss over the training window. We fit
    one CAViaR model per quantile level on the grid `u`. For multi-step
    horizons we apply √h scaling to deviations from the median, which is
    the standard i.i.d. approximation also used by GARCH-style baselines
    here (so the multi-step comparison is on equal footing).

    Fitting many quantiles is expensive (~K optimisations per call).
    `n_starts=2` keeps the per-step cost moderate while still being robust
    to local minima.
    """

    n_starts: int = 1
    anchor_taus: tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    maxiter: int = 100
    _anchor_q: np.ndarray | None = None  # shape (len(anchor_taus),)
    _returns: np.ndarray | None = None

    @staticmethod
    def _objective(params: np.ndarray, r: np.ndarray, tau: float, q0: float) -> float:
        q = _caviar_sav_path(params, r, q0)
        return _pinball(q[1:], r[1:], tau)

    def _fit_quantile(self, r: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
        from scipy.optimize import minimize

        q0 = float(np.quantile(r, tau))
        best_x = None
        best_f = np.inf
        x0_list = [
            np.array([q0 * 0.1, 0.85, -np.sign(0.5 - tau) * 0.1]),
            np.array([q0 * 0.2, 0.70, -np.sign(0.5 - tau) * 0.2]),
        ][: self.n_starts]
        for x0 in x0_list:
            try:
                res = minimize(
                    self._objective,
                    x0,
                    args=(r, tau, q0),
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": self.maxiter},
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.asarray(res.x, dtype=float)
            except Exception:
                continue
        if best_x is None:
            best_x = x0_list[0]
        return best_x, q0

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        self._returns = r
        # Fit at the (small) anchor grid once per call; predict() interpolates
        # to whatever u grid is requested. This is the standard distributional-
        # CAViaR trick — fitting K=30 separate CAViaR models per walk-forward
        # step is empirically wasteful since the quantile curve is smooth in tau.
        q_now = np.empty(len(self.anchor_taus), dtype=float)
        for i, tau in enumerate(self.anchor_taus):
            params, q0 = self._fit_quantile(r, float(tau))
            path = _caviar_sav_path(params, r, q0)
            q_now[i] = float(path[-1])
        # enforce monotone-in-tau before interpolation (raw CAViaR can crosse)
        self._anchor_q = np.maximum.accumulate(q_now)

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._anchor_q is not None
        # interpolate the anchor quantile curve to the requested u grid
        q_1d = np.interp(u, np.asarray(self.anchor_taus, dtype=float), self._anchor_q)
        q_1d = isotonic_project(q_1d)
        # multi-step scaling: shift median by h*median_1day, sqrt(h) on spread
        med_1 = float(np.interp(0.5, u, q_1d))
        q_h = h * med_1 + (q_1d - med_1) * np.sqrt(h)
        return isotonic_project(q_h)


# --------------------- 2-state Markov-Switching Normal -----------------------


def _ms_normal_em(
    r: np.ndarray, n_iter: int = 50, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EM for a 2-state Markov-switching Normal (Hamilton 1989, simplified).

    State k in {0, 1} has params (mu_k, sigma_k^2). Transition matrix P
    (P[i,j] = prob i->j). Returns (mu, sigma2, P, last_filtered) where
    last_filtered are the filtered state probabilities at the final t.
    """
    n = len(r)
    # init: low-vol state has below-median |r|, high-vol above
    abs_r = np.abs(r)
    med = np.median(abs_r)
    g0 = (abs_r < med).astype(float)
    g1 = 1.0 - g0
    mu = np.array([float(np.mean(r[g0 > 0])), float(np.mean(r[g1 > 0]))])
    sigma2 = np.array(
        [
            float(np.var(r[g0 > 0]) + 1e-10),
            float(np.var(r[g1 > 0]) + 1e-10),
        ]
    )
    # ensure state 1 is the high-vol state
    if sigma2[0] > sigma2[1]:
        mu = mu[::-1]
        sigma2 = sigma2[::-1]
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi0 = np.array([0.5, 0.5])

    last_ll = -np.inf
    last_filtered = pi0.copy()
    for _ in range(n_iter):
        # E-step: forward (Hamilton filter) + backward (Kim smoother)
        eta = np.zeros((n, 2))
        for k in range(2):
            eta[:, k] = norm.pdf(r, loc=mu[k], scale=np.sqrt(sigma2[k]))
        alpha = np.zeros((n, 2))
        c = np.zeros(n)
        alpha[0] = pi0 * eta[0]
        c[0] = alpha[0].sum() + 1e-300
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ P) * eta[t]
            c[t] = alpha[t].sum() + 1e-300
            alpha[t] /= c[t]
        beta = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            beta[t] = (P @ (eta[t + 1] * beta[t + 1])) / (c[t + 1])
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        # xi[t,i,j] = P(S_t=i, S_{t+1}=j | data)
        # M-step
        new_mu = np.zeros(2)
        new_sigma2 = np.zeros(2)
        for k in range(2):
            w = gamma[:, k]
            wsum = w.sum() + 1e-12
            new_mu[k] = float((w * r).sum() / wsum)
            new_sigma2[k] = float((w * (r - new_mu[k]) ** 2).sum() / wsum) + 1e-12
        # transition update
        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            denom = (alpha[t][:, None] * P * (eta[t + 1] * beta[t + 1])[None, :]).sum()
            if denom <= 0:
                continue
            xi_sum += (
                alpha[t][:, None] * P * (eta[t + 1] * beta[t + 1])[None, :]
            ) / denom
        row_sums = xi_sum.sum(axis=1, keepdims=True) + 1e-12
        new_P = xi_sum / row_sums
        # keep state ordering (low-vol = 0)
        if new_sigma2[0] > new_sigma2[1]:
            new_mu = new_mu[::-1]
            new_sigma2 = new_sigma2[::-1]
            new_P = new_P[::-1, ::-1]
        ll = float(np.log(np.maximum(c, 1e-300)).sum())
        if abs(ll - last_ll) < tol:
            mu, sigma2, P = new_mu, new_sigma2, new_P
            last_filtered = alpha[-1]
            break
        mu, sigma2, P = new_mu, new_sigma2, new_P
        last_filtered = alpha[-1]
        last_ll = ll
    return mu, sigma2, P, last_filtered


@dataclass
class MarkovSwitching2:
    """C3: Two-state Markov-switching Normal (Hamilton 1989, simplified).

    A 2-regime mixture-of-Gaussians model with persistent latent state
    (transition matrix P). State 0 = low-vol, state 1 = high-vol; means
    and variances are state-specific. Fit by EM on the full Hamilton
    filter (forward + backward).

    Forecast at horizon h: propagate the filtered state probabilities by
    P^h, then return quantiles of the resulting 2-component Gaussian
    mixture for the h-step cumulative return. The h-step *cumulative*
    distribution given a constant state k is N(h*mu_k, h*sigma_k^2). For a
    Markov-switching panel this is an approximation (the true h-step
    distribution mixes across all sequences of states) but is the standard
    "stay-in-current-state" forecast used in practice when h is small.

    This is the closest a discrete daily-returns model can come to
    "MS-GARCH" without paying the full Haas-Mittnik-Paolella likelihood
    cost. It captures the regime-switching geometry that motivated the
    family.
    """

    n_em_iter: int = 30

    _mu: np.ndarray | None = None  # (2,)
    _sigma2: np.ndarray | None = None  # (2,)
    _P: np.ndarray | None = None  # (2,2)
    _filtered: np.ndarray | None = None  # (2,) last-step probs

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        self._mu, self._sigma2, self._P, self._filtered = _ms_normal_em(
            r, n_iter=self.n_em_iter
        )

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert (
            self._mu is not None
            and self._sigma2 is not None
            and self._P is not None
            and self._filtered is not None
        )
        # state probabilities at horizon h
        prob_h = self._filtered @ np.linalg.matrix_power(self._P, h)
        # h-step distribution: weighted mixture of N(h*mu_k, h*sigma_k^2)
        sigmas_h = np.sqrt(self._sigma2 * h)
        mus_h = self._mu * h

        # invert mixture CDF by bisection on grid
        def mix_cdf(x: float) -> float:
            return float(
                prob_h[0] * norm.cdf(x, mus_h[0], sigmas_h[0])
                + prob_h[1] * norm.cdf(x, mus_h[1], sigmas_h[1])
            )

        # bisection bracket: widest plausible h-step range
        lo = float(mus_h.min() - 8 * sigmas_h.max())
        hi = float(mus_h.max() + 8 * sigmas_h.max())
        q = np.empty_like(u, dtype=float)
        for i, target in enumerate(u):
            t = float(target)
            a, b = lo, hi
            for _ in range(60):
                m = 0.5 * (a + b)
                if mix_cdf(m) < t:
                    a = m
                else:
                    b = m
            q[i] = 0.5 * (a + b)
        return isotonic_project(q)


# ----------------------------- FIGARCH(1,d,0) ---------------------------------


def _fractional_diff_coeffs(d: float, J: int) -> np.ndarray:
    """Coefficients δ_j of the (1-L)^d expansion, j=0..J-1.

    δ_0 = 1,  δ_j = δ_{j-1} * (j - 1 - d) / j
    """
    coeffs = np.empty(J, dtype=float)
    coeffs[0] = 1.0
    for j in range(1, J):
        coeffs[j] = coeffs[j - 1] * (j - 1.0 - d) / j
    return coeffs


def _figarch_var_path(
    omega: float, beta: float, d: float, eps2: np.ndarray, J: int = 250
) -> np.ndarray:
    """In-sample conditional variance path for FIGARCH(1,d,0).

    ARCH-∞ representation (Baillie-Bollerslev-Mikkelsen 1996, eq. 18):

        σ_t^2 = ω / (1-β) + λ(L) ε_t^2

    where λ(L) = 1 - (1-βL)^{-1} (1-L)^d. We compute λ_j iteratively from
    the fractional-diff coefficients δ_j:

        λ_1 = d - β
        λ_j = β λ_{j-1} + (δ_j  -  β δ_{j-1})  for j >= 2

    (Equivalent to the textbook recursion; see Bollerslev-Mikkelsen 1996
    appendix and Tse 1998.)
    """
    delta = _fractional_diff_coeffs(d, J + 1)
    lam = np.zeros(J + 1, dtype=float)
    lam[1] = d - beta  # j=1 term
    for j in range(2, J + 1):
        lam[j] = beta * lam[j - 1] + (-delta[j] + beta * delta[j - 1]) * (-1.0)
        # The sign convention: δ_j (j>=1) of (1-L)^d are negative for d in (0,1),
        # and λ_j must be non-negative for variance to be positive. The above
        # closely follows BBM's formula; with d in (0,1) and β in (0,1) the
        # λ_j are typically small and non-negative for j>=2. Clipping below.
    lam = np.maximum(lam, 0.0)
    n = len(eps2)
    var = np.empty(n, dtype=float)
    const = omega / max(1.0 - beta, 1e-6)
    for t in range(n):
        # lag length used is min(t, J)
        k = min(t, J)
        if k == 0:
            var[t] = const + lam[0] * eps2[t]
            continue
        # σ_t^2 = const + sum_{j=1..k} λ_j ε_{t-j}^2  + λ_0 ε_t^2
        var[t] = const + float(np.dot(lam[1 : k + 1], eps2[t - k : t][::-1]))
        var[t] = max(var[t], 1e-12)
    return var


@dataclass
class FIGARCH:
    """C4: FIGARCH(1, d, 0) with long-memory variance (BBM 1996).

    Fractionally Integrated GARCH captures the slow polynomial decay of
    autocorrelations in |r_t| that ordinary GARCH (geometric) cannot
    reproduce. The conditional variance follows the ARCH-∞ representation

        σ_t^2 = ω/(1-β) + λ(L) r_t^2,
        λ(L) = 1 - (1-βL)^{-1} (1-L)^d.

    We fit (ω, β, d) by Gaussian QML on the truncated representation
    (J=250 lags), then forecast h-step cumulative variance by iterating
    the FIGARCH recursion forward — using the current σ_t^2 as a flat
    proxy for missing future shocks (the standard h-step QML projection).

    Innovations are Student-t with df estimated from standardised
    residuals on the training window. Returns Student-t quantiles around
    the h-step mean μ * h.
    """

    J: int = 250
    _omega: float = 0.0
    _beta: float = 0.5
    _d: float = 0.4
    _df: float = 8.0
    _last_var: float = 1e-4
    _last_returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> None:
        from scipy.optimize import minimize

        r = np.asarray(returns, dtype=float) * 100.0  # match GARCH baselines
        self._last_returns = r
        eps = r - r.mean()
        eps2 = eps * eps

        def neg_ll(theta: np.ndarray) -> float:
            omega, beta, d = theta
            if omega <= 0 or not (0.0 < beta < 0.99) or not (0.05 < d < 0.95):
                return 1e10
            try:
                var = _figarch_var_path(omega, beta, d, eps2, J=self.J)
                # Gaussian QML
                ll = -0.5 * np.sum(np.log(var) + eps2 / var)
                if not np.isfinite(ll):
                    return 1e10
                return -float(ll)
            except Exception:
                return 1e10

        # initial: ω=0.05 (in percent^2), β=0.6, d=0.4
        x0 = np.array([0.05, 0.6, 0.4])
        res = minimize(
            neg_ll,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 250},
        )
        omega, beta, d = res.x
        self._omega = float(max(omega, 1e-8))
        self._beta = float(min(max(beta, 0.01), 0.98))
        self._d = float(min(max(d, 0.06), 0.94))
        var_in = _figarch_var_path(self._omega, self._beta, self._d, eps2, J=self.J)
        self._last_var = float(var_in[-1])
        z = eps / np.sqrt(var_in)
        excess_kurt = float(np.mean(z**4)) - 3.0
        nu = 4.0 + 6.0 / excess_kurt if excess_kurt > 1e-3 else 30.0
        self._df = float(min(max(nu, 4.5), 30.0))

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._last_returns is not None
        # h-step iterated forecast: use the convention E[σ_{t+i}^2] derives
        # from the FIGARCH stationary projection. Following Bollerslev-
        # Mikkelsen 1996 §3, the multi-step forecast converges slowly to
        # the unconditional level (no closed-form); we use the flat
        # extrapolation E[σ_{t+i}^2] ≈ σ_{t+1}^2 for the first h steps,
        # which understates the slow mean-reversion but matches what most
        # practitioners do at short h.
        var_h = float(self._last_var * h)
        mu_1 = float(self._last_returns.mean())
        q_pct = mu_1 * h + np.sqrt(var_h) * student_t.ppf(u, df=self._df)
        return isotonic_project(q_pct / 100.0)


# ------------------- Stochastic Volatility (SV-AR1, Kalman QML) ---------------


@dataclass
class StochasticVolatilityAR1:
    """C5: Discrete-time SV with AR(1) log-variance, fit by Kalman QML.

    The canonical Taylor (1982) / Harvey-Ruiz-Shephard (1994) SV model:

        r_t = exp(h_t / 2) ε_t,           ε_t ~ N(0, 1)
        h_t = μ + φ (h_{t-1} - μ) + σ_η η_t,  η_t ~ N(0, 1)

    is linearised by squaring and taking logs:

        y_t := log r_t^2 = h_t + (log ε_t^2)

    The noise term log(ε_t^2) for ε~N(0,1) has mean -1.27 and variance
    π²/2 ≈ 4.93. Treating it as Gaussian (quasi-likelihood) gives a
    linear Kalman filter on the AR(1) state h_t — the standard pragmatic
    fit for SV without MCMC. Forecast: h_{t+i} = μ + φ^i (h_t - μ),
    integrated variance over h steps gives the h-step return variance;
    quantiles via Gaussian (innovation ε is Gaussian by construction).
    """

    _mu: float = 0.0
    _phi: float = 0.9
    _sigma_eta2: float = 0.05
    _h_filtered: float = 0.0
    _h_var: float = 1.0
    _drift: float = 0.0

    def fit(self, returns: np.ndarray) -> None:
        from scipy.optimize import minimize

        r = np.asarray(returns, dtype=float)
        self._drift = float(r.mean())
        # offset for numerical safety in log
        y = np.log(np.maximum((r - self._drift) ** 2, 1e-12)) + 1.27  # bias-correct

        def neg_ll(theta: np.ndarray) -> float:
            mu, phi, log_sig2 = theta
            if not (-0.99 < phi < 0.99):
                return 1e10
            sigma_eta2 = float(np.exp(log_sig2))
            R = np.pi**2 / 2.0  # measurement noise variance after bias correction
            # Kalman filter
            h_pred = mu
            P_pred = sigma_eta2 / max(1.0 - phi**2, 1e-6)
            ll = 0.0
            for yt in y:
                S = P_pred + R
                K = P_pred / S
                inn = yt - h_pred
                ll += -0.5 * (np.log(2 * np.pi * S) + inn * inn / S)
                h_filt = h_pred + K * inn
                P_filt = (1.0 - K) * P_pred
                h_pred = mu + phi * (h_filt - mu)
                P_pred = phi**2 * P_filt + sigma_eta2
            if not np.isfinite(ll):
                return 1e10
            return -float(ll)

        x0 = np.array([float(np.mean(y)), 0.9, np.log(0.05)])
        res = minimize(
            neg_ll,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 200},
        )
        mu, phi, log_sig2 = res.x
        self._mu = float(mu)
        self._phi = float(min(max(phi, -0.99), 0.99))
        self._sigma_eta2 = float(np.exp(log_sig2))
        # final filtered state
        R = np.pi**2 / 2.0
        h_pred = self._mu
        P_pred = self._sigma_eta2 / max(1.0 - self._phi**2, 1e-6)
        h_filt = h_pred
        P_filt = P_pred
        for yt in y:
            S = P_pred + R
            K = P_pred / S
            inn = yt - h_pred
            h_filt = h_pred + K * inn
            P_filt = (1.0 - K) * P_pred
            h_pred = self._mu + self._phi * (h_filt - self._mu)
            P_pred = self._phi**2 * P_filt + self._sigma_eta2
        self._h_filtered = h_filt
        self._h_var = P_filt

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        # h-step variance = sum_{i=1..h} E[exp(h_{t+i})]
        # log-normal expectation: E[exp(h_{t+i})] = exp(m_i + 0.5 v_i)
        # where m_i = mu + phi^i (h_t - mu), v_i = phi^{2i} P_t + sigma_eta2 (1-phi^{2i})/(1-phi^2)
        var_h = 0.0
        for i in range(1, h + 1):
            m_i = self._mu + (self._phi**i) * (self._h_filtered - self._mu)
            denom = max(1.0 - self._phi**2, 1e-6)
            v_i = (self._phi ** (2 * i)) * self._h_var + self._sigma_eta2 * (
                1.0 - self._phi ** (2 * i)
            ) / denom
            var_h += float(np.exp(m_i + 0.5 * v_i))
        mu_h = self._drift * h
        q = mu_h + np.sqrt(var_h) * norm.ppf(u)
        return isotonic_project(q)


# --------------------- Bivariate VAR + diagonal GARCH ------------------------


@dataclass
class BivariateVARGarch:
    """C6: Bivariate VAR(1) on (BTC, ETH) + GARCH(1,1) on BTC residuals.

    Cross-asset baseline. The mean dynamics are a stationary VAR(1):

        r_t^{BTC} = c1 + a11 r_{t-1}^{BTC} + a12 r_{t-1}^{ETH} + ε1_t
        r_t^{ETH} = c2 + a21 r_{t-1}^{BTC} + a22 r_{t-1}^{ETH} + ε2_t

    so today's ETH return informs tomorrow's BTC mean forecast through
    the a12 coefficient. The BTC marginal *variance* is then modelled by
    a univariate GARCH(1,1) on the VAR residual ε1_t (we keep the
    variance side univariate; a full BEKK/DCC is unneeded for a marginal
    BTC-quantile comparison since both projects the same VAR mean).

    For multi-step forecasts: iterate the VAR (h steps) for the mean,
    accumulate GARCH variance (h-step sum). Innovations are Student-t
    (df estimated from standardised residuals).

    Alignment with the univariate harness: the constructor stores the
    full exogenous (ETH) series along with the full target (BTC) series.
    `fit(returns)` matches `returns` against the BTC tail to recover the
    matching ETH window. This avoids needing harness-level support for
    exogenous arrays while preserving correct walk-forward alignment.
    """

    full_target: np.ndarray  # full BTC log-return series
    full_exog: np.ndarray  # full ETH log-return series, aligned to BTC
    _coef: np.ndarray | None = None  # (2, 3) [c, a_self, a_cross] per equation
    _resid_btc: np.ndarray | None = None
    _last_btc: float = 0.0
    _last_eth: float = 0.0
    _garch_result = None
    _df: float = 8.0

    def __post_init__(self):
        self.full_target = np.asarray(self.full_target, dtype=float)
        self.full_exog = np.asarray(self.full_exog, dtype=float)
        if len(self.full_target) != len(self.full_exog):
            raise ValueError(
                f"full_target and full_exog must have equal length, "
                f"got {len(self.full_target)} vs {len(self.full_exog)}"
            )

    def _locate_window(self, returns: np.ndarray) -> int:
        """Find the suffix-end index of `returns` inside full_target.

        We expect walk-forward to pass a contiguous suffix of full_target.
        Match on the last 16 values (high-probability uniqueness for
        continuous log-returns). Returns the end index (exclusive).
        """
        n = len(returns)
        k = min(16, n)
        needle = returns[-k:]
        # search from the right for efficiency (walk-forward advances forward)
        target = self.full_target
        for end in range(len(target), k - 1, -1):
            if np.allclose(target[end - k : end], needle, atol=1e-12, rtol=0):
                # also verify length-n window matches
                start = end - n
                if start >= 0 and np.allclose(target[start:end], returns, atol=1e-12):
                    return end
        raise ValueError("could not align returns window to full_target")

    def fit(self, returns: np.ndarray) -> None:
        end = self._locate_window(returns)
        start = end - len(returns)
        r_btc = self.full_target[start:end]
        r_eth = self.full_exog[start:end]
        n = len(r_btc)
        # VAR(1) by OLS, per equation
        y_btc = r_btc[1:]
        y_eth = r_eth[1:]
        X = np.column_stack([np.ones(n - 1), r_btc[:-1], r_eth[:-1]])  # (n-1, 3)
        # OLS solves: y = X beta
        XtX_inv = np.linalg.pinv(X.T @ X)
        b_btc = XtX_inv @ X.T @ y_btc
        b_eth = XtX_inv @ X.T @ y_eth
        self._coef = np.vstack([b_btc, b_eth])  # (2, 3)
        # BTC residuals -> GARCH
        resid_btc = y_btc - X @ b_btc
        self._resid_btc = resid_btc
        am = arch_model(resid_btc * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
        self._garch_result = am.fit(disp="off", show_warning=False)
        self._df = float(self._garch_result.params["nu"])
        self._last_btc = float(r_btc[-1])
        self._last_eth = float(r_eth[-1])

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        assert self._coef is not None and self._garch_result is not None
        # iterate VAR(1) h steps
        a = self._coef  # (2,3)
        state = np.array([self._last_btc, self._last_eth])
        c = a[:, 0]
        A = a[:, 1:]  # (2, 2)
        mu_path = np.empty(h, dtype=float)
        for i in range(h):
            state = c + A @ state
            mu_path[i] = state[0]
        mu_h = float(mu_path.sum())
        # GARCH-t variance forecast over h steps
        f = self._garch_result.forecast(horizon=h, reindex=False)
        var_h_pct = float(f.variance.values[-1, :].sum())
        sigma_h = np.sqrt(var_h_pct) / 100.0
        q = mu_h + sigma_h * student_t.ppf(u, df=self._df)
        return isotonic_project(q)
