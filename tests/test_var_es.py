"""Unit tests for the VaR / Expected-Shortfall backtests.

The four tests in :mod:`wbtc.var_es` each have a textbook-known behaviour
under their null and alternative; this file checks the headline ones with
synthetic data where the answer is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from wbtc.quantiles import make_grid
from wbtc.var_es import (
    acerbi_szekely_mc_pvalues,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    christoffersen_cc_test,
    christoffersen_independence_test,
    extract_var_es,
    kupiec_pof_test,
    var_es_panel,
)


# --------------------------- extract_var_es ---------------------------------


def test_extract_var_es_matches_normal_closed_form():
    """For N(0,1) at alpha=0.05: VaR=-1.6449, ES=-2.0627 (closed form)."""
    u = make_grid(200)
    q = norm.ppf(u)
    var_r, es_r = extract_var_es(q, u, alpha=0.05)
    # closed forms: VaR_α = norm.ppf(α); ES_α = -phi(z_α)/α for N(0,1)
    expected_var = float(norm.ppf(0.05))
    expected_es = -float(norm.pdf(norm.ppf(0.05))) / 0.05
    assert abs(var_r - expected_var) < 0.01, f"VaR off: {var_r} vs {expected_var}"
    assert abs(es_r - expected_es) < 0.02, f"ES off: {es_r} vs {expected_es}"


def test_extract_var_es_es_below_var():
    """For any non-degenerate left tail, ES is more negative than VaR."""
    rng = np.random.default_rng(0)
    sample = rng.normal(size=10000)
    u = make_grid(100)
    q = np.quantile(sample, u)
    var_r, es_r = extract_var_es(q, u, alpha=0.05)
    assert es_r < var_r, "ES should be below VaR for a tail"
    assert var_r < 0, "VaR at 5% should be negative for centred returns"


# --------------------------- Kupiec POF -------------------------------------


def test_kupiec_accepts_when_rate_matches():
    """Hit rate ≈ α should not reject."""
    rng = np.random.default_rng(0)
    n = 2000
    alpha = 0.05
    hits = (rng.uniform(size=n) < alpha).astype(int)
    res = kupiec_pof_test(hits, alpha)
    # under H_0, p-value uniform; failure rate ~5% — most seeds should not reject
    assert res.p_value > 0.05, f"false positive at seed=0: p={res.p_value}"


def test_kupiec_rejects_biased_hit_rate():
    """A doubled hit rate at α=0.05 should reject at any reasonable n."""
    rng = np.random.default_rng(1)
    n = 2000
    alpha = 0.05
    hits = (rng.uniform(size=n) < 2 * alpha).astype(int)  # actual rate ~10%
    res = kupiec_pof_test(hits, alpha)
    assert res.p_value < 0.001, f"failed to reject 2× rate: p={res.p_value}"
    assert res.empirical_rate > 0.08
    assert res.df == 1


def test_kupiec_zero_violations_does_not_crash():
    hits = np.zeros(500, dtype=int)
    res = kupiec_pof_test(hits, alpha=0.05)
    assert np.isfinite(res.stat)
    # 500 days at α=5% with zero violations is incredibly unlikely under null
    assert res.p_value < 1e-6


# ------------------- Christoffersen independence ----------------------------


def test_christoffersen_indep_accepts_iid():
    """Bernoulli iid hits should pass the independence test on average."""
    rng = np.random.default_rng(2)
    n = 2000
    hits = (rng.uniform(size=n) < 0.05).astype(int)
    res = christoffersen_independence_test(hits)
    assert res.p_value > 0.05, f"false positive on iid hits: p={res.p_value}"


def test_christoffersen_indep_rejects_clustering():
    """A clustered hit series should reject the iid null.

    Construction: persistent regime where, after a hit, P(hit|hit)=0.5
    (way above the unconditional rate). LR_ind should detect this.
    """
    rng = np.random.default_rng(3)
    n = 3000
    hits = np.zeros(n, dtype=int)
    for t in range(1, n):
        if hits[t - 1] == 1:
            hits[t] = int(rng.uniform() < 0.5)  # P[hit|hit] = 0.5
        else:
            hits[t] = int(rng.uniform() < 0.04)  # P[hit|no hit] = 0.04
    res = christoffersen_independence_test(hits)
    assert res.p_value < 0.01, f"failed to detect clustering: p={res.p_value}"


def test_christoffersen_cc_combines_both():
    """CC test should reject when *either* unconditional rate is wrong or
    clustering is present — confirms LR_cc = LR_uc + LR_ind is doing both."""
    rng = np.random.default_rng(4)
    n = 2000
    alpha = 0.05
    # clustered + wrong rate
    hits = np.zeros(n, dtype=int)
    for t in range(1, n):
        if hits[t - 1] == 1:
            hits[t] = int(rng.uniform() < 0.4)
        else:
            hits[t] = int(rng.uniform() < 0.08)
    res = christoffersen_cc_test(hits, alpha)
    assert res.df == 2
    assert res.p_value < 0.01


# ----------------------- Acerbi-Szekely Z1 / Z2 -----------------------------


def test_acerbi_szekely_z1_zero_when_well_specified():
    """Under correctly-specified Gaussian, Z1 should be close to 0."""
    rng = np.random.default_rng(5)
    n = 5000
    alpha = 0.05
    sigma = 0.02
    r = rng.normal(0, sigma, size=n)
    var_r = np.full(n, sigma * norm.ppf(alpha))
    es_r = np.full(n, -sigma * norm.pdf(norm.ppf(alpha)) / alpha)
    z1 = acerbi_szekely_z1(r, var_r, es_r)
    # exceedance count ~ 250, conditional mean of r/es ≈ 1 → z1 ≈ 0
    assert abs(z1) < 0.10, f"|Z1|={z1} too large for well-specified case"


def test_acerbi_szekely_z1_positive_when_es_underestimated():
    """Model says σ=0.02 but actual is σ=0.04 — Z1 should be strongly positive."""
    rng = np.random.default_rng(6)
    n = 5000
    alpha = 0.05
    model_sigma = 0.02
    true_sigma = 0.04
    r = rng.normal(0, true_sigma, size=n)
    var_r = np.full(n, model_sigma * norm.ppf(alpha))
    es_r = np.full(n, -model_sigma * norm.pdf(norm.ppf(alpha)) / alpha)
    z1 = acerbi_szekely_z1(r, var_r, es_r)
    # closed form for doubling σ on N(0,σ): E[r|r<VaR_model] / ES_model - 1 ≈ 0.34
    assert z1 > 0.2, f"Z1={z1} should be positive when σ doubled"


def test_acerbi_szekely_z2_matches_sign_of_z1():
    """Z2 should have the same sign as Z1 under monotone mis-specification."""
    rng = np.random.default_rng(7)
    n = 5000
    alpha = 0.05
    r = rng.normal(0, 0.04, size=n)
    var_r = np.full(n, 0.02 * norm.ppf(alpha))
    es_r = np.full(n, -0.02 * norm.pdf(norm.ppf(alpha)) / alpha)
    z1 = acerbi_szekely_z1(r, var_r, es_r)
    z2 = acerbi_szekely_z2(r, var_r, es_r, alpha)
    assert (z1 > 0) == (z2 > 0)
    assert z2 > 0.1


# ----------------------- MC p-values ---------------------------------------


def test_mc_pvalue_high_when_well_specified():
    """Under correctly-specified forecasts the observed Z1/Z2 should be
    indistinguishable from the bootstrap null (high p-value)."""
    rng = np.random.default_rng(8)
    n = 800
    alpha = 0.05
    sigma = 0.02
    u = make_grid(50)
    # All forecasts are the same N(0, σ) quantile vector
    q_template = sigma * norm.ppf(u)
    Q = np.tile(q_template, (n, 1))
    # Realised returns drawn from the same N(0, σ)
    r = rng.normal(0, sigma, size=n)
    out = acerbi_szekely_mc_pvalues(r, Q, u, alpha, n_mc=200, rng=rng)
    # MC p-value should not be tiny
    assert out["p_z1"] > 0.05, f"false positive Z1: p={out['p_z1']}"
    assert out["p_z2"] > 0.05, f"false positive Z2: p={out['p_z2']}"


def test_mc_pvalue_low_when_es_underestimated():
    """When the true σ is double the model σ, MC p-value should be small."""
    rng = np.random.default_rng(9)
    n = 800
    alpha = 0.05
    model_sigma = 0.02
    true_sigma = 0.04
    u = make_grid(50)
    Q = np.tile(model_sigma * norm.ppf(u), (n, 1))
    r = rng.normal(0, true_sigma, size=n)
    out = acerbi_szekely_mc_pvalues(r, Q, u, alpha, n_mc=200, rng=rng)
    assert out["p_z2"] < 0.05, f"failed to reject misspecified Z2: p={out['p_z2']}"


# ----------------------- panel wrapper --------------------------------------


def test_var_es_panel_runs_end_to_end():
    """Smoke test the full panel wrapper on a small synthetic case."""
    rng = np.random.default_rng(10)
    n = 400
    alpha = 0.05
    u = make_grid(50)
    sigma = 0.02
    Q = np.tile(sigma * norm.ppf(u), (n, 1))
    r = rng.normal(0, sigma, size=n)
    res = var_es_panel(r, Q, u, alpha, n_mc=100, rng=rng)
    assert res.n == n
    assert 0 < res.empirical_rate < 0.2
    # all p-values in [0, 1]
    for p in (res.kupiec_p, res.indep_p, res.cc_p, res.z1_p, res.z2_p):
        assert 0.0 <= p <= 1.0


def test_extract_var_es_alpha_validation():
    u = make_grid(20)
    q = np.linspace(-0.05, 0.05, 20)
    with pytest.raises(ValueError):
        extract_var_es(q, u, alpha=0.0)
    with pytest.raises(ValueError):
        extract_var_es(q, u, alpha=1.5)
