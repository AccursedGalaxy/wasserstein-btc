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


def test_ewma_recovers_constant_drift():
    from wbtc.forecasters import WassersteinGeodesicEWMA

    rng = np.random.default_rng(13)
    n = 600
    t = np.arange(n)
    r = 1e-4 * t + rng.normal(0, 0.01, size=n)
    u = make_grid(40)
    f = WassersteinGeodesicEWMA(window=90, lookback=30, decay=0.85)
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q10 = f.predict(h=10, u=u)
    shift = float(np.median(q10) - np.median(q1))
    # Drift is constant across the series, so weighted slope ~ 1e-4 regardless.
    assert 0.0005 < shift < 0.0015


def test_ewma_decay_one_matches_ols():
    """decay=1 must equal the OLS WassersteinGeodesic slope, by construction."""
    from wbtc.forecasters import WassersteinGeodesicEWMA

    rng = np.random.default_rng(17)
    r = rng.normal(0, 0.02, size=600)
    u = make_grid(20)
    f_ols = WassersteinGeodesic(window=90, lookback=30)
    f_ewma = WassersteinGeodesicEWMA(window=90, lookback=30, decay=1.0)
    f_ols.fit(r)
    f_ewma.fit(r)
    q_ols = f_ols.predict(h=5, u=u)
    q_ewma = f_ewma.predict(h=5, u=u)
    np.testing.assert_allclose(q_ols, q_ewma, rtol=1e-9, atol=1e-12)


def test_hetero_runs_and_is_monotone():
    from wbtc.forecasters import WassersteinGeodesicHetero

    r = _synth_returns(n=800)
    u = make_grid(30)
    f = WassersteinGeodesicHetero(window=90, lookback=20)
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q21 = f.predict(h=21, u=u)
    assert np.isfinite(q1).all()
    assert np.isfinite(q21).all()
    assert (np.diff(q1) >= -1e-9).all()
    assert (np.diff(q21) >= -1e-9).all()
    assert (q21.max() - q21.min()) > (q1.max() - q1.min())


def test_ensemble_extremes_match_components():
    """Forcing the smoothstep into its saturated ends collapses the ensemble
    onto its respective component, which validates the weighting algebra."""
    from wbtc.forecasters import (
        GarchNormal,
        WassersteinGeodesicTheilSen,
        WGeoGarchEnsemble,
    )

    rng = np.random.default_rng(23)
    r = rng.normal(0, 0.02, size=800)
    u = make_grid(20)

    # Force w=1 (rank always above rank_hi). Ensemble must equal pure GARCH.
    f_g = GarchNormal()
    f_g.fit(r)
    q_g = f_g.predict(h=5, u=u)
    f_pure_garch = WGeoGarchEnsemble(
        window=90,
        lookback=20,
        vol_window=20,
        vol_rank_window=252,
        rank_lo=-1.0,
        rank_hi=-0.5,
    )
    f_pure_garch.fit(r)
    q_e = f_pure_garch.predict(h=5, u=u)
    np.testing.assert_allclose(q_e, q_g, rtol=1e-9, atol=1e-12)

    # Force w=0 (rank always below rank_lo). Ensemble must equal pure WGeo.
    f_w = WassersteinGeodesicTheilSen(window=90, lookback=20)
    f_w.fit(r)
    q_w = f_w.predict(h=5, u=u)
    f_pure_wgeo = WGeoGarchEnsemble(
        window=90,
        lookback=20,
        vol_window=20,
        vol_rank_window=252,
        rank_lo=1.5,
        rank_hi=2.0,
    )
    f_pure_wgeo.fit(r)
    q_e = f_pure_wgeo.predict(h=5, u=u)
    np.testing.assert_allclose(q_e, q_w, rtol=1e-9, atol=1e-12)


def test_ensemble_is_monotone_and_finite():
    from wbtc.forecasters import WGeoGarchEnsemble

    r = _synth_returns(n=800)
    u = make_grid(20)
    f = WGeoGarchEnsemble(window=90, lookback=20)
    f.fit(r)
    q = f.predict(h=5, u=u)
    assert np.isfinite(q).all()
    assert (np.diff(q) >= -1e-9).all()


# --------------------- v0.4 extended baselines ------------------------------


def test_harrv_monotone_and_widens_with_horizon():
    from wbtc.forecasters import HARRV

    r = _synth_returns(n=500)
    u = make_grid(20)
    f = HARRV()
    f.fit(r)
    q1 = f.predict(1, u)
    q5 = f.predict(5, u)
    assert np.isfinite(q1).all() and np.isfinite(q5).all()
    assert (np.diff(q1) >= -1e-9).all()
    assert (np.diff(q5) >= -1e-9).all()
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())


def test_caviar_monotone_and_finite():
    from wbtc.forecasters import CAViaRSAV

    r = _synth_returns(n=300)
    u = make_grid(15)
    f = CAViaRSAV(n_starts=1)
    f.fit(r)
    q = f.predict(1, u)
    assert np.isfinite(q).all()
    assert (np.diff(q) >= -1e-9).all()


def test_ms_two_state_recovers_known_vol_split():
    """A returns series with a clear high-vol second half should land state 1
    (high-vol) with high filtered probability at the final timestep."""
    from wbtc.forecasters import MarkovSwitching2

    rng = np.random.default_rng(42)
    r = np.concatenate([rng.normal(0, 0.005, size=300), rng.normal(0, 0.04, size=300)])
    u = make_grid(20)
    f = MarkovSwitching2()
    f.fit(r)
    q5 = f.predict(5, u)
    assert np.isfinite(q5).all()
    assert (np.diff(q5) >= -1e-9).all()
    assert f._filtered is not None  # type: ignore[attr-defined]
    # high-vol state filtered prob at the end should dominate
    assert f._filtered[1] > 0.7  # type: ignore[index]


def test_figarch_monotone_and_widens_with_horizon():
    from wbtc.forecasters import FIGARCH

    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.02, size=500)
    u = make_grid(20)
    f = FIGARCH()
    f.fit(r)
    q1 = f.predict(1, u)
    q5 = f.predict(5, u)
    assert np.isfinite(q1).all() and np.isfinite(q5).all()
    assert (np.diff(q1) >= -1e-9).all()
    assert (np.diff(q5) >= -1e-9).all()
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())


def test_sv_ar1_monotone_and_widens_with_horizon():
    from wbtc.forecasters import StochasticVolatilityAR1

    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.02, size=500)
    u = make_grid(20)
    f = StochasticVolatilityAR1()
    f.fit(r)
    q1 = f.predict(1, u)
    q5 = f.predict(5, u)
    assert np.isfinite(q1).all() and np.isfinite(q5).all()
    assert (np.diff(q1) >= -1e-9).all()
    assert (np.diff(q5) >= -1e-9).all()
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())


def test_bivariate_var_garch_uses_exog_for_mean():
    """If ETH = -2 * BTC contemporaneous and BTC is AR(1), the VAR fit must
    pick up a non-trivial cross-coefficient — i.e. the multivariate model
    is genuinely using the second series."""
    from wbtc.forecasters import BivariateVARGarch

    rng = np.random.default_rng(11)
    n = 600
    btc = np.zeros(n)
    for t in range(1, n):
        btc[t] = 0.2 * btc[t - 1] + rng.normal(0, 0.02)
    # ETH leads BTC slightly: ETH_t informs BTC_{t+1} via a cross term
    eth = np.zeros(n)
    eth[0] = rng.normal(0, 0.02)
    for t in range(1, n):
        eth[t] = 0.5 * btc[t - 1] + rng.normal(0, 0.02)
    # now make BTC's mean depend on lagged ETH too, so the VAR should learn it
    btc2 = btc.copy()
    for t in range(1, n):
        btc2[t] += 0.3 * eth[t - 1]
    u = make_grid(20)
    f = BivariateVARGarch(full_target=btc2, full_exog=eth)
    f.fit(btc2)
    q = f.predict(5, u)
    assert np.isfinite(q).all()
    assert (np.diff(q) >= -1e-9).all()
    # cross coefficient (BTC equation, ETH lag) should be non-trivial
    assert f._coef is not None  # type: ignore[attr-defined]
    cross = float(f._coef[0, 2])  # type: ignore[index]
    assert abs(cross) > 0.05, f"expected non-trivial cross-coef, got {cross:.4f}"
