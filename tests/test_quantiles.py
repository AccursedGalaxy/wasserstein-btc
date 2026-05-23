import numpy as np

from wbtc.quantiles import (
    empirical_quantiles,
    isotonic_project,
    make_grid,
    w2_distance,
    weighted_quantiles,
)


def test_grid_in_open_interval():
    u = make_grid(10)
    assert (u > 0).all() and (u < 1).all()
    assert u.shape == (10,)


def test_empirical_quantiles_uniform_recovers_grid():
    # Quantiles of Uniform(0,1) at grid u should be ≈ u in the limit.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=200_000)
    u = make_grid(20)
    q = empirical_quantiles(x, u)
    assert np.max(np.abs(q - u)) < 0.01


def test_w2_distance_zero_for_same():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10_000)
    u = make_grid(50)
    q = empirical_quantiles(x, u)
    assert w2_distance(q, q) == 0.0


def test_w2_distance_increases_with_shift():
    rng = np.random.default_rng(2)
    x = rng.normal(size=10_000)
    u = make_grid(50)
    q0 = empirical_quantiles(x, u)
    q1 = empirical_quantiles(x + 1.0, u)
    q2 = empirical_quantiles(x + 2.0, u)
    # A pure translation by t shifts every quantile by t, so W_2 == |t|.
    assert abs(w2_distance(q0, q1) - 1.0) < 0.05
    assert abs(w2_distance(q0, q2) - 2.0) < 0.05


def test_weighted_quantiles_uniform_weights_match_empirical():
    """Uniform weights must reproduce the unweighted empirical quantile.

    Hazen plotting positions (used by ``weighted_quantiles``) differ from
    numpy's type-7 only at the extreme tails by O(1/n); both are consistent
    estimators of the population quantile (Hyndman-Fan 1996). We therefore
    test agreement on the interior of the grid plus an asymptotic-style check
    that the largest discrepancy is O(1/sqrt(n)) on a Gaussian sample.
    """
    rng = np.random.default_rng(101)
    n = 2000
    x = rng.normal(size=n)
    u = make_grid(40)
    q_un = empirical_quantiles(x, u)
    q_w = weighted_quantiles(x, u, np.ones_like(x))
    # Interior of the grid (drop top + bottom level) must agree closely.
    interior_diff = np.max(np.abs(q_un[1:-1] - q_w[1:-1]))
    assert interior_diff < 1e-2, f"interior diff too large: {interior_diff}"
    # Tails differ by at most O(1/sqrt(n)) for a Gaussian sample.
    boundary_diff = np.max(np.abs(q_un - q_w))
    assert boundary_diff < 3.0 / np.sqrt(n), (
        f"tail diff too large: {boundary_diff} vs 3/sqrt(n)={3.0 / np.sqrt(n):.4f}"
    )


def test_weighted_quantiles_zero_weight_observation_is_ignored():
    """Putting a huge outlier with zero weight should not move any quantile."""
    rng = np.random.default_rng(102)
    x = rng.normal(size=500)
    x_with_outlier = np.append(x, 1e6)
    w = np.append(np.ones_like(x), 0.0)
    u = make_grid(20)
    q_clean = weighted_quantiles(x, u, np.ones_like(x))
    q_outlier = weighted_quantiles(x_with_outlier, u, w)
    np.testing.assert_allclose(q_clean, q_outlier, atol=1e-10)


def test_weighted_quantiles_concentrated_weight_picks_window():
    """Exponential decay should make recent observations dominate."""
    # First half is a calm regime (small std), second half is turbulent.
    rng = np.random.default_rng(103)
    n = 600
    half = n // 2
    calm = rng.normal(0, 0.005, size=half)
    turb = rng.normal(0, 0.05, size=half)
    x = np.concatenate([calm, turb])
    u = make_grid(20)
    # Decay weights favouring the recent (turbulent) half: λ^(N-1-i)
    lam = 0.95
    w = lam ** np.arange(n)[::-1]
    q_uni = weighted_quantiles(x, u, np.ones_like(x))
    q_dec = weighted_quantiles(x, u, w)
    spread_uni = q_uni.max() - q_uni.min()
    spread_dec = q_dec.max() - q_dec.min()
    # The recency-weighted quantile must be substantially wider — it reflects
    # the turbulent regime; the unweighted version is blurred by the calm half.
    assert spread_dec > 1.5 * spread_uni


def test_weighted_quantiles_rejects_bad_weights():
    x = np.array([0.0, 1.0, 2.0])
    u = make_grid(5)
    try:
        weighted_quantiles(x, u, np.array([-1.0, 1.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative weight")
    try:
        weighted_quantiles(x, u, np.zeros_like(x))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for all-zero weights")


def test_isotonic_project_is_monotone_and_close():
    q_bad = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0])
    q_proj = isotonic_project(q_bad)
    assert (np.diff(q_proj) >= -1e-12).all()
    # projection is idempotent
    assert np.allclose(isotonic_project(q_proj), q_proj)
    # already-monotone passes through
    q_good = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert np.allclose(isotonic_project(q_good), q_good)
