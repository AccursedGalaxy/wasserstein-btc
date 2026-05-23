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
