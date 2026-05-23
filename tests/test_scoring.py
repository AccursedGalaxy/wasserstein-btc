import numpy as np

from wbtc.quantiles import empirical_quantiles, make_grid
from wbtc.scoring import crps_from_quantiles, diebold_mariano


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
