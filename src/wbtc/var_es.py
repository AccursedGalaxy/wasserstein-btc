"""VaR / Expected-Shortfall backtests on distributional forecasts.

The forecasters in :mod:`wbtc.forecasters` already return full quantile
vectors, so VaR and ES at any tail level α are a free by-product. This
module turns those into the four standard tail-calibration tests used by
regulators and risk desks:

- :func:`kupiec_pof_test` — Kupiec (1995) unconditional-coverage LR
  (binomial: does the empirical violation rate match α?). χ²(1).
- :func:`christoffersen_independence_test` — Christoffersen (1998) Markov-
  chain independence LR (are violations clustered?). χ²(1).
- :func:`christoffersen_cc_test` — combined conditional coverage
  LR_uc + LR_ind. χ²(2).
- :func:`acerbi_szekely_z1`, :func:`acerbi_szekely_z2` — the two
  Acerbi-Szekely (2014) Expected-Shortfall tests (Z1 conditions on
  exceedances; Z2 averages unconditionally). Both have non-standard null
  distributions; we supply Monte-Carlo p-values by drawing synthetic
  realised returns from each step's own predictive quantile function.

Sign convention. ``returns`` carries log-returns (can be negative).
``var_returns`` and ``es_returns`` are *signed* — the α-quantile of returns
and the conditional mean of returns in the lower α tail respectively — so
both are negative for the tail levels we care about (α ≤ 0.05). The
classic positive-loss VaR_loss / ES_loss = -var_returns / -es_returns
shows up only inside the AS test bodies where the formulas are most
readable in the loss convention.

Limitations
-----------
- Tests assume non-overlapping forecasts. For h>1 you should pass a
  strided subset (every h-th step) or interpret χ² critical values with
  caution — overlap inflates LR_ind by autocorrelating the hit series.
- LR_ind is a 1st-order Markov test; higher-order clustering (e.g.,
  one violation in a week, then five in two days) is not detected. See
  :func:`christoffersen_independence_test` for the literature pointer to
  the Engle-Manganelli (2004) Dynamic Quantile test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

__all__ = [
    "extract_var_es",
    "kupiec_pof_test",
    "christoffersen_independence_test",
    "christoffersen_cc_test",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "acerbi_szekely_mc_pvalues",
    "VarEsBacktest",
    "var_es_panel",
]


# ---------------------------- VaR / ES extraction ---------------------------


def extract_var_es(
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Return ``(var_return, es_return)`` at tail level ``alpha`` from a forecast.

    ``var_return = Q(alpha)`` interpolated on the forecast grid; this is
    the α-quantile of the *return* distribution (negative for α ≤ 0.5).

    ``es_return = (1/α) ∫_0^α Q(u) du`` computed by the trapezoidal rule
    on the part of the grid in [0, α], padded with the point ``(α, VaR)``
    so the integral reaches exactly α regardless of where the grid hits.

    The returned ES_return is the conditional **mean** of returns in the
    lower α tail — it is more negative than VaR_return whenever the
    forecast has a non-degenerate left tail.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    q = np.asarray(quantile_values, dtype=float)
    u = np.asarray(quantile_levels, dtype=float)
    if q.shape != u.shape:
        raise ValueError("quantile_values and quantile_levels must have equal shape")
    order = np.argsort(u)
    u = u[order]
    q = q[order]
    var_return = float(np.interp(alpha, u, q))
    mask = u <= alpha
    if not mask.any():
        # Grid does not reach alpha at all; fall back to the smallest-u
        # grid point as the only tail observation.
        return var_return, float(q[0])
    u_int = np.concatenate([u[mask], [alpha]])
    q_int = np.concatenate([q[mask], [var_return]])
    # Anchor the integral at u=0 by reflecting the first segment slope —
    # i.e. assume the quantile function is locally linear between 0 and
    # the smallest grid point. This is the same convention used by
    # numpy.trapz on a left-padded grid.
    if u_int[0] > 0.0:
        # linear extrapolation back to 0 using first two points
        if len(u_int) >= 2:
            slope = (q_int[1] - q_int[0]) / (u_int[1] - u_int[0])
            q0 = q_int[0] - slope * u_int[0]
        else:
            q0 = q_int[0]
        u_int = np.concatenate([[0.0], u_int])
        q_int = np.concatenate([[q0], q_int])
    es_return = float(np.trapezoid(q_int, u_int) / alpha)
    return var_return, es_return


# ---------------------------- Kupiec POF (LR_uc) ----------------------------


@dataclass
class CoverageResult:
    """One-row test result for Kupiec / Christoffersen tests.

    ``stat`` is the LR statistic; ``p_value`` its χ²(``df``) right-tail
    p-value. ``empirical_rate`` is ``n1 / n`` for diagnostic display.
    """

    n: int
    n_violations: int
    empirical_rate: float
    expected_rate: float
    stat: float
    df: int
    p_value: float


def kupiec_pof_test(hits: np.ndarray, alpha: float) -> CoverageResult:
    """Kupiec (1995) Proportion-of-Failures unconditional-coverage LR test.

    Tests ``H_0: π = α`` against ``H_1: π ≠ α`` where ``π = P(r_t < VaR_α,t)``
    is the empirical violation rate. The likelihood ratio

        LR_uc = -2 [ N_0 log(1-α) + N_1 log(α)
                   - N_0 log(1-π̂) - N_1 log(π̂) ]

    is χ²(1) under H_0. ``hits`` must be a 0/1 array of violation
    indicators (one per non-overlapping forecast).
    """
    h = np.asarray(hits, dtype=int)
    if h.ndim != 1:
        raise ValueError("hits must be 1D")
    n = int(len(h))
    n1 = int(h.sum())
    n0 = n - n1
    if n == 0:
        return CoverageResult(0, 0, float("nan"), alpha, float("nan"), 1, float("nan"))
    pi_hat = n1 / n
    # Degenerate cases — return LR=0 and p=1 (no power) rather than div-by-0.
    if n1 == 0:
        # purely the (1-α)^n vs 1 ratio
        ll_null = n * np.log(1.0 - alpha)
        ll_alt = 0.0  # n0 log(1) + n1 log(...) with n1=0
        lr = -2.0 * (ll_null - ll_alt)
        p = float(1.0 - chi2.cdf(lr, df=1))
        return CoverageResult(n, n1, pi_hat, alpha, float(lr), 1, p)
    if n1 == n:
        ll_null = n * np.log(alpha)
        ll_alt = 0.0
        lr = -2.0 * (ll_null - ll_alt)
        p = float(1.0 - chi2.cdf(lr, df=1))
        return CoverageResult(n, n1, pi_hat, alpha, float(lr), 1, p)
    ll_null = n0 * np.log(1.0 - alpha) + n1 * np.log(alpha)
    ll_alt = n0 * np.log(1.0 - pi_hat) + n1 * np.log(pi_hat)
    lr = float(-2.0 * (ll_null - ll_alt))
    p = float(1.0 - chi2.cdf(lr, df=1))
    return CoverageResult(n, n1, pi_hat, alpha, lr, 1, p)


# -------------------- Christoffersen independence (LR_ind) ------------------


def christoffersen_independence_test(hits: np.ndarray) -> CoverageResult:
    """Christoffersen (1998) Markov-chain independence LR test.

    Tests whether the sequence of violations forms an iid Bernoulli
    process against the alternative of first-order Markov dependence
    (clustering). Let ``n_{ij}`` count the transitions from state i to
    state j (i, j ∈ {0, 1}) and define

        π_01 = n_{01} / (n_{00} + n_{01})       (P[hit | no hit])
        π_11 = n_{11} / (n_{10} + n_{11})       (P[hit | hit])
        π    = (n_{01} + n_{11}) / N

    Then LR_ind = -2 [ (n_{00}+n_{10}) log(1-π) + (n_{01}+n_{11}) log(π)
                     - n_{00} log(1-π_01) - n_{01} log(π_01)
                     - n_{10} log(1-π_11) - n_{11} log(π_11) ]
    is χ²(1) under H_0: π_01 = π_11 (no clustering).

    NB. This is a 1st-order Markov test; higher-order clustering goes
    undetected. For finer resolution see the Engle-Manganelli (2004)
    Dynamic Quantile (DQ) test, which regresses hits on lagged hits and
    lagged VaR — left as an extension here for parsimony.
    """
    h = np.asarray(hits, dtype=int)
    if h.ndim != 1:
        raise ValueError("hits must be 1D")
    n = int(len(h))
    if n < 2:
        return CoverageResult(
            n, int(h.sum()), float("nan"), float("nan"), float("nan"), 1, float("nan")
        )
    # transition counts
    h_prev = h[:-1]
    h_curr = h[1:]
    n00 = int(((h_prev == 0) & (h_curr == 0)).sum())
    n01 = int(((h_prev == 0) & (h_curr == 1)).sum())
    n10 = int(((h_prev == 1) & (h_curr == 0)).sum())
    n11 = int(((h_prev == 1) & (h_curr == 1)).sum())
    n_total = n00 + n01 + n10 + n11
    n1 = int(h.sum())
    pi_uncond = (n01 + n11) / max(n_total, 1)
    # Degenerate: any column or row zero -> Markov MLE is on the boundary
    # and LR is 0 (no power against the iid null). Return a finite stat.
    if (n00 + n01) == 0 or (n10 + n11) == 0 or pi_uncond <= 0 or pi_uncond >= 1:
        return CoverageResult(n, n1, n1 / n, float("nan"), 0.0, 1, 1.0)
    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    # ll under null (iid)
    ll_null = (n00 + n10) * np.log(1.0 - pi_uncond) + (n01 + n11) * np.log(pi_uncond)

    # ll under alt (Markov); guard log(0) — 0*log(0) := 0
    def _safe(x: int, p: float) -> float:
        if x == 0:
            return 0.0
        return float(x * np.log(p))

    ll_alt = (
        _safe(n00, 1.0 - pi_01)
        + _safe(n01, pi_01)
        + _safe(n10, 1.0 - pi_11)
        + _safe(n11, pi_11)
    )
    lr = float(-2.0 * (ll_null - ll_alt))
    # numerical safety
    if not np.isfinite(lr) or lr < 0:
        lr = max(0.0, lr) if np.isfinite(lr) else 0.0
    p = float(1.0 - chi2.cdf(lr, df=1))
    return CoverageResult(n, n1, n1 / n, float("nan"), lr, 1, p)


def christoffersen_cc_test(hits: np.ndarray, alpha: float) -> CoverageResult:
    """Christoffersen (1998) combined conditional-coverage LR test.

    LR_cc = LR_uc + LR_ind, χ²(2) under the joint null (correct
    unconditional rate **and** no clustering). The two component LRs are
    asymptotically independent under H_0, so the sum is the right joint
    statistic.
    """
    uc = kupiec_pof_test(hits, alpha)
    ind = christoffersen_independence_test(hits)
    uc_stat = uc.stat if np.isfinite(uc.stat) else 0.0
    ind_stat = ind.stat if np.isfinite(ind.stat) else 0.0
    lr = float(uc_stat + ind_stat)
    p = float(1.0 - chi2.cdf(lr, df=2))
    return CoverageResult(
        n=uc.n,
        n_violations=uc.n_violations,
        empirical_rate=uc.empirical_rate,
        expected_rate=alpha,
        stat=lr,
        df=2,
        p_value=p,
    )


# ---------------------------- Acerbi-Szekely ES -----------------------------


def acerbi_szekely_z1(
    returns: np.ndarray,
    var_returns: np.ndarray,
    es_returns: np.ndarray,
) -> float:
    """Acerbi-Szekely (2014) Z1 — conditional ES-magnitude statistic.

        Z1 = (1/N_T) Σ_t [ X_t · I_t / ES_t ]  -  1

    where ``X_t = -returns_t`` is the positive loss, ``I_t`` indicates
    a VaR exceedance, ``ES_t = -es_returns_t`` is the positive predicted
    expected shortfall, and ``N_T = max(1, Σ I_t)``.

    Sign interpretation. Under the null (model correctly specifies the
    conditional tail) E[X_t | I_t=1] = ES_t, so the ratio averages 1 and
    Z1 = 0. **Z1 > 0** means realised exceedance losses are *larger* than
    the model's ES — the model **underestimates** the tail. Z1 < 0
    means the model is conservative.

    Returns the statistic only; for the p-value use
    :func:`acerbi_szekely_mc_pvalues` which draws synthetic returns from
    each step's own predictive quantile function (the per-model null).
    """
    r = np.asarray(returns, dtype=float)
    var_r = np.asarray(var_returns, dtype=float)
    es_r = np.asarray(es_returns, dtype=float)
    if r.shape != var_r.shape or r.shape != es_r.shape:
        raise ValueError("returns, var_returns, es_returns must have equal shape")
    hits = (r < var_r).astype(float)
    n_excess = float(hits.sum())
    if n_excess == 0:
        # No exceedances — the conditional statistic is undefined. By
        # convention return 0 (no evidence of mis-specification in the
        # tail magnitude; UC test catches the rate side).
        return 0.0
    losses = -r
    es_loss = -es_r
    # guard against zero/positive es_loss (would indicate a degenerate
    # forecast); clip away from zero.
    es_loss = np.where(es_loss > 1e-12, es_loss, 1e-12)
    z = float(np.sum(hits * losses / es_loss) / n_excess - 1.0)
    return z


def acerbi_szekely_z2(
    returns: np.ndarray,
    var_returns: np.ndarray,
    es_returns: np.ndarray,
    alpha: float,
) -> float:
    """Acerbi-Szekely (2014) Z2 — unconditional ES statistic.

        Z2 = (1/(T·α)) Σ_t [ X_t · I_t / ES_t ]  -  1

    Same exceedance / loss / ES definitions as :func:`acerbi_szekely_z1`,
    but the sum is averaged over **all** T steps rather than only
    exceedance days. Z2 jointly tests the violation rate *and* the tail
    magnitude — failure can come from either side. The test does not
    require an explicit VaR forecast for the population statistic but our
    finite-sample version uses I_t for symmetry with Z1.

    Z2 > 0 means underestimation (more or worse losses than predicted);
    Z2 < 0 conservative.
    """
    r = np.asarray(returns, dtype=float)
    var_r = np.asarray(var_returns, dtype=float)
    es_r = np.asarray(es_returns, dtype=float)
    if r.shape != var_r.shape or r.shape != es_r.shape:
        raise ValueError("returns, var_returns, es_returns must have equal shape")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    hits = (r < var_r).astype(float)
    losses = -r
    es_loss = -es_r
    es_loss = np.where(es_loss > 1e-12, es_loss, 1e-12)
    T = float(len(r))
    z = float(np.sum(hits * losses / es_loss) / (T * alpha) - 1.0)
    return z


def _sample_from_quantile_grid(
    q_matrix: np.ndarray,
    u: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_samples`` synthetic return paths of length T from each row's
    predictive quantile function.

    ``q_matrix`` has shape (T, K). Returns array shape (n_samples, T) where
    row b column t is a single draw r*_{b,t} ~ Q_t (the inverse CDF
    transform of u ~ Uniform(0,1)).
    """
    T = q_matrix.shape[0]
    # draw uniforms then interpolate per t
    us = rng.uniform(0.0, 1.0, size=(n_samples, T))
    out = np.empty_like(us, dtype=float)
    # vectorised interp per timestep; T loop is cheap vs. per-call numpy.interp
    for t in range(T):
        out[:, t] = np.interp(us[:, t], u, q_matrix[t])
    return out


def acerbi_szekely_mc_pvalues(
    returns: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
    alpha: float,
    n_mc: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Monte-Carlo p-values for the AS Z1 and Z2 statistics.

    Algorithm (the standard Acerbi-Szekely procedure):

    1. Compute observed Z1, Z2 from the realised returns vs the
       (VaR, ES) extracted from each step's forecast.
    2. For b = 1..B, draw synthetic returns r*_b from each step's own
       predictive quantile function (the per-step null), recompute Z1*_b
       and Z2*_b using the **same** (VaR, ES) (these depend only on the
       forecast).
    3. One-sided p-value: fraction of bootstrap samples ≥ the observed
       statistic. We use the upper tail because the diagnostically
       meaningful failure mode in finance is *underestimation* of the
       tail (Z > 0).

    Returns
    -------
    dict with keys ``z1, p_z1, z2, p_z2, n_violations``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    r = np.asarray(returns, dtype=float)
    Q = np.asarray(quantile_matrix, dtype=float)
    u = np.asarray(quantile_levels, dtype=float)
    if Q.shape[0] != len(r):
        raise ValueError(
            f"quantile_matrix has {Q.shape[0]} rows but returns has {len(r)}"
        )
    # Extract VaR and ES per step.
    var_arr = np.empty(len(r), dtype=float)
    es_arr = np.empty(len(r), dtype=float)
    for t in range(len(r)):
        var_arr[t], es_arr[t] = extract_var_es(Q[t], u, alpha)
    z1_obs = acerbi_szekely_z1(r, var_arr, es_arr)
    z2_obs = acerbi_szekely_z2(r, var_arr, es_arr, alpha)
    n_viol = int((r < var_arr).sum())
    # Simulate.
    sim = _sample_from_quantile_grid(Q, u, n_mc, rng)  # (B, T)
    z1_sim = np.empty(n_mc, dtype=float)
    z2_sim = np.empty(n_mc, dtype=float)
    for b in range(n_mc):
        z1_sim[b] = acerbi_szekely_z1(sim[b], var_arr, es_arr)
        z2_sim[b] = acerbi_szekely_z2(sim[b], var_arr, es_arr, alpha)
    # One-sided p-value: how often does the null reach as extreme as observed?
    p_z1 = (
        float((z1_sim >= z1_obs).mean())
        if z1_obs >= 0
        else float((z1_sim <= z1_obs).mean())
    )
    p_z2 = (
        float((z2_sim >= z2_obs).mean())
        if z2_obs >= 0
        else float((z2_sim <= z2_obs).mean())
    )
    return {
        "z1": float(z1_obs),
        "p_z1": float(p_z1),
        "z2": float(z2_obs),
        "p_z2": float(p_z2),
        "n_violations": n_viol,
    }


# ----------------------------- panel wrapper --------------------------------


@dataclass
class VarEsBacktest:
    """All four tail tests at a single ``alpha`` for one (method, asset) cell.

    Fields are the natural display order: violation rate, then the three
    coverage LRs (Kupiec, Christoffersen indep + cc), then the two
    Acerbi-Szekely ES statistics with MC p-values.
    """

    alpha: float
    n: int
    n_violations: int
    empirical_rate: float
    kupiec_stat: float
    kupiec_p: float
    indep_stat: float
    indep_p: float
    cc_stat: float
    cc_p: float
    z1: float
    z1_p: float
    z2: float
    z2_p: float


def var_es_panel(
    returns: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
    alpha: float,
    n_mc: int = 1000,
    rng: np.random.Generator | None = None,
) -> VarEsBacktest:
    """Run all four tail tests at level ``alpha`` for one method's forecasts.

    The single entry point used by the driver script: takes realised
    returns aligned with a (T, K) quantile matrix and returns the full
    test panel as a :class:`VarEsBacktest` dataclass.
    """
    r = np.asarray(returns, dtype=float)
    Q = np.asarray(quantile_matrix, dtype=float)
    u = np.asarray(quantile_levels, dtype=float)
    var_arr = np.empty(len(r), dtype=float)
    for t in range(len(r)):
        var_arr[t], _ = extract_var_es(Q[t], u, alpha)
    hits = (r < var_arr).astype(int)
    uc = kupiec_pof_test(hits, alpha)
    ind = christoffersen_independence_test(hits)
    cc = christoffersen_cc_test(hits, alpha)
    as_res = acerbi_szekely_mc_pvalues(r, Q, u, alpha, n_mc=n_mc, rng=rng)
    return VarEsBacktest(
        alpha=alpha,
        n=int(uc.n),
        n_violations=int(uc.n_violations),
        empirical_rate=float(uc.empirical_rate),
        kupiec_stat=float(uc.stat),
        kupiec_p=float(uc.p_value),
        indep_stat=float(ind.stat),
        indep_p=float(ind.p_value),
        cc_stat=float(cc.stat),
        cc_p=float(cc.p_value),
        z1=float(as_res["z1"]),
        z1_p=float(as_res["p_z1"]),
        z2=float(as_res["z2"]),
        z2_p=float(as_res["p_z2"]),
    )
