import numpy as np

from wbtc.quantiles import empirical_quantiles, make_grid
from wbtc.scoring import (
    crps_from_quantiles,
    diebold_mariano,
    diebold_mariano_residualised,
)


def test_crps_zero_when_distribution_is_dirac_at_y():
    # CDF is a step at y -> CRPS = 0.
    u = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    y = 1.23
    q = np.full_like(u, y, dtype=float)
    crps = crps_from_quantiles(q, u, y)
    # tiny numerical residual from the tail padding
    assert crps < 1e-4


def test_crps_equals_mae_for_dirac_offset():
    # Dirac forecast at y_hat, realised y; CRPS = |y - y_hat|.
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_hat = 2.0
    q = np.full_like(u, y_hat, dtype=float)
    for y in [-1.0, 1.5, 5.0]:
        crps = crps_from_quantiles(q, u, y)
        assert abs(crps - abs(y - y_hat)) < 1e-3


def test_crps_proper_normal_known_value():
    """CRPS of N(0,1) at y=0 has closed form 1/sqrt(pi) - 1/sqrt(2*pi) ≈ 0.2336."""
    from scipy.stats import norm

    u = np.linspace(0.001, 0.999, 2000)
    q = norm.ppf(u)
    crps = crps_from_quantiles(q, u, 0.0)
    expected = 2 * norm.pdf(0.0) - 1 / np.sqrt(np.pi)  # known closed form
    assert abs(crps - expected) < 0.01


def test_dm_detects_a_strictly_better():
    rng = np.random.default_rng(42)
    n = 500
    loss_b = rng.uniform(0, 1, size=n)
    loss_a = loss_b - 0.1  # A is uniformly better
    dm, p = diebold_mariano(loss_a, loss_b, h=1)
    assert dm < -3
    assert p < 0.001


def test_dm_finds_no_diff_when_equal():
    rng = np.random.default_rng(0)
    n = 500
    loss = rng.uniform(0, 1, size=n)
    dm, p = diebold_mariano(loss, loss, h=1)
    assert abs(dm) < 1e-9 or np.isnan(dm)
    # p should be high
    if not np.isnan(p):
        assert p > 0.5


def test_residualised_dm_reduces_to_dm_with_uncorrelated_control():
    """When the control has zero correlation with the loss differential the
    OLS slope is ≈0, the residualisation is a no-op, and the residualised DM
    statistic must equal the vanilla DM up to numerical noise."""
    rng = np.random.default_rng(101)
    n = 800
    loss_b = rng.uniform(0, 1, size=n)
    loss_a = loss_b - 0.05 + rng.normal(0, 0.05, size=n)
    # independent control
    control = rng.normal(0, 1.0, size=n)
    dm, p = diebold_mariano(loss_a, loss_b, h=1)
    dm_r, p_r = diebold_mariano_residualised(loss_a, loss_b, control, h=1)
    # Slope is ~zero so residuals are essentially the original differential.
    assert abs(dm_r - dm) < 0.1
    assert abs(p_r - p) < 0.05


def test_residualised_dm_more_powerful_when_control_explains_noise():
    """When the control is correlated with the *shared* component of the two
    losses (e.g. a common volatility shock), the regression absorbs it and
    the residualised DM has a strictly larger |stat| / smaller p-value than
    vanilla DM, while preserving the same point estimate of the mean
    differential."""
    rng = np.random.default_rng(102)
    n = 2000
    # shared "vol" noise: large, autocorrelated
    eta = rng.normal(0, 1.0, size=n)
    common = np.convolve(eta, np.ones(20) / 20, mode="same")  # smoothed vol
    # Each forecaster's loss reacts to the common vol with a slightly different
    # gain — so the *differential* still inherits a non-trivial chunk of the
    # common noise, which the control can absorb.
    base = rng.normal(0, 0.1, size=n)
    private_a = rng.normal(0, 0.05, size=n)
    private_b = rng.normal(0, 0.05, size=n)
    loss_b = base + 0.50 * common + private_b
    loss_a = base + 0.45 * common + private_a - 0.01
    # the control loss is dominated by the same common noise
    loss_c = rng.normal(0, 0.1, size=n) + 1.0 * common
    dm, p = diebold_mariano(loss_a, loss_b, h=1)
    dm_r, p_r = diebold_mariano_residualised(loss_a, loss_b, loss_c, h=1)
    # Mean differential is preserved (test only changes variance)
    assert abs(dm_r) > abs(dm), f"residualised DM not more powerful: {dm_r} vs {dm}"
    assert p_r < p, f"residualised p not lower: {p_r} vs {p}"


def test_residualised_dm_handles_multiple_controls():
    rng = np.random.default_rng(103)
    n = 500
    loss_b = rng.uniform(0, 1, size=n)
    loss_a = loss_b - 0.03
    # two controls, both noise — answer should match the no-control DM
    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)
    dm, _ = diebold_mariano(loss_a, loss_b, h=1)
    dm_r, _ = diebold_mariano_residualised(loss_a, loss_b, [c1, c2], h=1)
    assert abs(dm_r - dm) < 0.3


def test_residualised_dm_with_hac_lag_for_long_horizon():
    """Sanity check: the test runs without numerical issues at h>1 and
    returns a finite stat."""
    rng = np.random.default_rng(104)
    n = 600
    loss_b = rng.normal(0, 1, size=n)
    loss_a = loss_b - 0.02 + rng.normal(0, 0.5, size=n)
    control = rng.normal(0, 1, size=n)
    dm, p = diebold_mariano_residualised(loss_a, loss_b, control, h=21)
    assert np.isfinite(dm)
    assert 0.0 <= p <= 1.0
