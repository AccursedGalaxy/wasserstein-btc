"""Tests for the split-conformal calibration layer (wbtc.conformal)."""

import numpy as np

from wbtc.conformal import (
    ConformalCalibrator,
    conformal_offsets,
    conformalize,
    conformalize_path,
    empirical_coverage,
    pit_values,
)
from wbtc.forecasters import StaticEmpirical
from wbtc.quantiles import make_grid


def test_offsets_are_the_conformal_order_statistic():
    rng = np.random.default_rng(0)
    n, K = 99, 4
    u = np.array([0.05, 0.5, 0.9, 0.99])
    E = rng.normal(size=(n, K))
    d = conformal_offsets(E, u)
    for k in range(K):
        srt = np.sort(E[:, k])
        idx = min(int(np.ceil((n + 1) * u[k])) - 1, n - 1)
        assert d[k] == srt[idx]
    # u=0.99 with n=99 clips to the sample max
    assert d[3] == E[:, 3].max()


def test_conformalize_output_is_monotone():
    q = np.linspace(-0.1, 0.1, 30)
    # offsets that would break monotonicity if applied naively
    d = np.where(np.arange(30) % 2 == 0, 0.02, -0.02)
    qc = conformalize(q, d)
    assert (np.diff(qc) >= -1e-12).all()


def test_pit_values_uniform_for_calibrated_forecast():
    rng = np.random.default_rng(1)
    u = make_grid(50)
    T = 4000
    y = rng.normal(size=T)
    from scipy.stats import norm

    Q = np.tile(norm.ppf(u), (T, 1))
    p = pit_values(Q, u, y)
    hist, _ = np.histogram(p, bins=[0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(hist / T, 0.25, atol=0.03)
    cov = empirical_coverage(Q, u, y, levels=np.array([0.05, 0.5, 0.95]))
    assert abs(cov[0.05] - 0.05) < 0.02
    assert abs(cov[0.95] - 0.95) < 0.02


def test_path_restores_coverage_of_a_too_narrow_forecast():
    """A base forecast that is 2x too narrow is badly under-covered; after the
    rolling conformal correction the empirical coverage is close to nominal."""
    rng = np.random.default_rng(2)
    u = make_grid(50)
    T = 3000
    from scipy.stats import norm

    y = rng.standard_t(4, size=T) * 0.02
    Q = np.tile(norm.ppf(u) * 0.01, (T, 1))  # far too narrow
    Q_cal, active = conformalize_path(Q, y, u, lag=1, calib_window=250, min_calib=50)
    assert active.sum() > T - 100
    before = empirical_coverage(Q[active], u, y[active], levels=np.array([0.05, 0.95]))
    after = empirical_coverage(
        Q_cal[active], u, y[active], levels=np.array([0.05, 0.95])
    )
    assert before[0.05] > 0.12 and before[0.95] < 0.88  # base is broken
    assert abs(after[0.05] - 0.05) < 0.02
    assert abs(after[0.95] - 0.95) < 0.02
    # monotone on every row
    assert (np.diff(Q_cal, axis=1) >= -1e-12).all()


def test_path_respects_lag_and_min_calib():
    u = make_grid(10)
    T = 120
    Q = np.zeros((T, 10))
    y = np.linspace(-1, 1, T)
    _, active = conformalize_path(Q, y, u, lag=5, calib_window=50, min_calib=20)
    # first active row t satisfies t - 5 + 1 >= 20  ->  t >= 24
    assert not active[:24].any()
    assert active[24:].all()


def test_online_calibrator_matches_offline_path():
    """Sliding a window through the series with the online wrapper must give
    the same calibrated quantiles as the vectorised offline function."""
    rng = np.random.default_rng(3)
    r = rng.standard_t(5, size=900) * 0.02
    W, h = 300, 3
    u = make_grid(20)
    cal = ConformalCalibrator(StaticEmpirical(), calib_window=100, min_calib=30)
    online, base_path, ys = [], [], []
    origins = range(W, len(r) - h + 1)  # window r[t-W:t], target r[t:t+h]
    for t in origins:
        window = r[t - W : t]
        cal.fit(window)
        online.append(cal.predict(h, u))
        b = StaticEmpirical()
        b.fit(window)
        base_path.append(b.predict(h, u))
        ys.append(r[t : t + h].sum())
    online = np.array(online)
    Q = np.array(base_path)
    y = np.array(ys)
    off, active = conformalize_path(Q, y, u, lag=h, calib_window=100, min_calib=30)
    assert active.sum() > 400
    np.testing.assert_allclose(online[active], off[active], rtol=0, atol=1e-12)
    np.testing.assert_allclose(online[~active], Q[~active], rtol=0, atol=1e-12)
    assert cal.last_active


def test_online_calibrator_resets_on_non_contiguous_window():
    rng = np.random.default_rng(4)
    r = rng.normal(size=500)
    u = make_grid(10)
    cal = ConformalCalibrator(StaticEmpirical(), calib_window=50, min_calib=10)
    for t in range(200, 260):
        cal.fit(r[t - 200 : t])
        cal.predict(1, u)
    assert cal.n_scores(1) > 10
    cal.fit(r[100:300])  # jump -> not a shift of the previous window
    assert cal.n_scores(1) == 0
    q = cal.predict(1, u)
    assert not cal.last_active
    assert (np.diff(q) >= -1e-12).all()
