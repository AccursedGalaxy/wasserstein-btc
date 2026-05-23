import numpy as np

from wbtc.quantiles import empirical_quantiles, isotonic_project, make_grid, w2_distance


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


def test_isotonic_project_is_monotone_and_close():
    q_bad = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0])
    q_proj = isotonic_project(q_bad)
    assert (np.diff(q_proj) >= -1e-12).all()
    # projection is idempotent
    assert np.allclose(isotonic_project(q_proj), q_proj)
    # already-monotone passes through
    q_good = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert np.allclose(isotonic_project(q_good), q_good)
