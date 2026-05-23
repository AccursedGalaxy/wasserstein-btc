import numpy as np

from wbtc.forecasters import (
    GarchNormal,
    StaticEmpirical,
    WassersteinGeodesic,
    WassersteinGeodesicGated,
)
from wbtc.quantiles import make_grid


def _synth_returns(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.02, size=n)


def test_static_empirical_predict_returns_monotone_grid():
    r = _synth_returns()
    u = make_grid(20)
    f = StaticEmpirical()
    f.fit(r)
    q = f.predict(h=1, u=u)
    assert (np.diff(q) >= -1e-9).all()


def test_garch_normal_runs_and_widens_with_horizon():
    r = _synth_returns(n=800)
    u = make_grid(20)
    f = GarchNormal()
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q5 = f.predict(h=5, u=u)
    # spread should grow with horizon
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())


def test_wasserstein_geodesic_recovers_constant_drift():
    """Returns whose location drifts linearly -> the geodesic forecaster
    should pick up the drift in its median."""
    rng = np.random.default_rng(7)
    n = 600
    t = np.arange(n)
    drift = 1e-4 * t
    r = drift + rng.normal(0, 0.01, size=n)
    u = make_grid(40)
    f = WassersteinGeodesic(window=90, lookback=30)
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q10 = f.predict(h=10, u=u)
    # median should shift by ~9 * 1e-4
    shift = float(np.median(q10) - np.median(q1))
    assert 0.0005 < shift < 0.0015


def test_gated_predict_is_finite_and_monotone():
    r = _synth_returns(n=500)
    u = make_grid(20)
    f = WassersteinGeodesicGated(window=90, lookback=30)
    f.fit(r)
    q = f.predict(h=5, u=u)
    assert np.isfinite(q).all()
    assert (np.diff(q) >= -1e-9).all()


def test_historical_simulation_bootstrap_shape_and_finite():
    from wbtc.forecasters import HistoricalSimulationBootstrap

    r = _synth_returns(n=400)
    u = make_grid(20)
    f = HistoricalSimulationBootstrap(n_paths=2000, rng_seed=0)
    f.fit(r)
    q = f.predict(h=5, u=u)
    assert q.shape == u.shape
    assert np.isfinite(q).all()
    assert (np.diff(q) >= -1e-9).all()


def test_gjr_garch_t_runs_and_widens_with_horizon():
    from wbtc.forecasters import GJRGarchStudentT

    r = _synth_returns(n=800)
    u = make_grid(20)
    f = GJRGarchStudentT()
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q5 = f.predict(h=5, u=u)
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())


def test_theil_sen_recovers_constant_drift():
    """Same drift recovery test as OLS variant, but with outliers added."""
    from wbtc.forecasters import WassersteinGeodesicTheilSen

    rng = np.random.default_rng(11)
    n = 600
    t = np.arange(n)
    r = 1e-4 * t + rng.normal(0, 0.01, size=n)
    # inject 2% outliers — OLS would be perturbed, Theil-Sen should not
    outlier_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    r[outlier_idx] = rng.choice([-0.2, 0.2], size=len(outlier_idx))
    u = make_grid(40)
    f = WassersteinGeodesicTheilSen(window=90, lookback=30)
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q10 = f.predict(h=10, u=u)
    shift = float(np.median(q10) - np.median(q1))
    assert 0.0005 < shift < 0.0015
