import numpy as np
import pytest

from wbtc.forecasters import (
    Forecaster,
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


def test_adaptive_cumulative_drift_scales_linearly():
    """Adaptive uses the Static-convention cumulative-drift level (μ_now·h)
    rather than the OLS-convention (median_now + h·β). This is the correct
    mean of an h-step cumulative return under stationary returns and is the
    convention used by the StaticEmpirical and GARCH families. The test
    therefore verifies linear scaling of the predicted median with horizon
    on a trending series.
    """
    from wbtc.forecasters import WassersteinGeodesicAdaptive

    rng = np.random.default_rng(31)
    n = 800
    t = np.arange(n)
    drift = 1e-4
    r = drift * t + rng.normal(0, 0.01, size=n)
    u = make_grid(40)
    f = WassersteinGeodesicAdaptive(window=90, lookback=30)
    f.fit(r)
    q1 = f.predict(h=1, u=u)
    q10 = f.predict(h=10, u=u)
    shift = float(np.median(q10) - np.median(q1))
    # Adaptive inherits the OLS-convention level (median_now + h·β̄), so the
    # h=1 → h=10 increment is ≈ 9·β̄ ≈ 9·1e-4 with the recency-weighted base
    # quantile contributing a small √h-induced perturbation around it.
    assert 0.0005 < shift < 0.0015, f"shift={shift}"


def test_wgeo_ensemble_default_is_w2_barycentre_of_v03_trio():
    """Default ensemble is the equal-weight quantile-function average of
    TheilSen + EWMA + Gated. Numerically this must equal the mean of the
    three components' predict() outputs (modulo the final isotonic projection,
    which is a no-op on monotone inputs)."""
    from wbtc.forecasters import (
        WGeoEnsemble,
        WassersteinGeodesicEWMA,
        WassersteinGeodesicGated,
        WassersteinGeodesicTheilSen,
    )

    rng = np.random.default_rng(41)
    r = rng.normal(0, 0.02, size=800)
    u = make_grid(30)
    f_ts = WassersteinGeodesicTheilSen(window=90, lookback=20)
    f_ew = WassersteinGeodesicEWMA(window=90, lookback=20, decay=0.85)
    f_gt = WassersteinGeodesicGated(window=90, lookback=20, kappa_star=0.6, tau=5)
    for f in (f_ts, f_ew, f_gt):
        f.fit(r)
    q_avg = (f_ts.predict(5, u) + f_ew.predict(5, u) + f_gt.predict(5, u)) / 3.0
    f_ens = WGeoEnsemble()
    f_ens.fit(r)
    q_ens = f_ens.predict(5, u)
    np.testing.assert_allclose(q_ens, q_avg, atol=1e-9)


def test_wgeo_ensemble_jensen_inequality_holds():
    """CRPS is convex in the forecast CDF, so the ensemble's per-step CRPS
    must be ≤ the mean of components' per-step CRPS on every forecast (with
    inequality being strict whenever components disagree). Test on a single
    forecast horizon and several realised outcomes."""
    from wbtc.forecasters import WGeoEnsemble
    from wbtc.scoring import crps_from_quantiles

    rng = np.random.default_rng(42)
    r = rng.normal(0, 0.02, size=800)
    u = make_grid(30)
    f_ens = WGeoEnsemble()
    f_ens.fit(r)
    q_ens = f_ens.predict(5, u)
    q_components = [f.predict(5, u) for f in f_ens._fitted]
    for y in (-0.1, -0.02, 0.0, 0.02, 0.1):
        crps_ens = crps_from_quantiles(q_ens, u, y)
        crps_mean = float(np.mean([crps_from_quantiles(q, u, y) for q in q_components]))
        # Jensen on CRPS (convex in the forecast) — equality only if components
        # are identical.
        assert crps_ens <= crps_mean + 1e-9, (
            f"Jensen violated at y={y}: ensemble {crps_ens} vs mean {crps_mean}"
        )


def test_wgeo_ensemble_weights_renormalise():
    """Weights are normalised before use, so any positive scaling is the
    same ensemble. Catches a class of subtle bugs in weight handling."""
    from wbtc.forecasters import WGeoEnsemble

    rng = np.random.default_rng(43)
    r = rng.normal(0, 0.02, size=400)
    u = make_grid(20)
    f_a = WGeoEnsemble(weights=[1.0, 1.0, 1.0])
    f_b = WGeoEnsemble(weights=[5.0, 5.0, 5.0])
    f_a.fit(r)
    f_b.fit(r)
    np.testing.assert_allclose(f_a.predict(5, u), f_b.predict(5, u), atol=1e-12)


def test_wgeo_ensemble_rejects_bad_weights():
    from wbtc.forecasters import WGeoEnsemble

    try:
        WGeoEnsemble(weights=[1.0, -1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative weight")
    try:
        WGeoEnsemble(weights=[0.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for all-zero weights")


def test_adaptive_decay_one_matches_ewma():
    """With decay_quantile=1.0 the recency-weighted base quantile equals the
    equal-weighted empirical quantile (up to Hazen-vs-type-7 tail O(1/n)),
    so Adaptive must collapse to :class:`WassersteinGeodesicEWMA` on the
    interior of the grid."""
    from wbtc.forecasters import WassersteinGeodesicAdaptive, WassersteinGeodesicEWMA

    rng = np.random.default_rng(35)
    r = rng.normal(0, 0.02, size=600)
    u = make_grid(30)
    f_ewma = WassersteinGeodesicEWMA(window=90, lookback=30, decay=0.85)
    f_adp = WassersteinGeodesicAdaptive(
        window=90, lookback=30, decay=0.85, decay_quantile=1.0
    )
    f_ewma.fit(r)
    f_adp.fit(r)
    q_ewma = f_ewma.predict(h=5, u=u)
    q_adp = f_adp.predict(h=5, u=u)
    interior = np.s_[1:-1]
    np.testing.assert_allclose(q_adp[interior], q_ewma[interior], atol=2e-3)


def test_adaptive_monotone_and_finite_and_widens():
    from wbtc.forecasters import WassersteinGeodesicAdaptive

    rng = np.random.default_rng(32)
    r = rng.normal(0, 0.02, size=800)
    u = make_grid(30)
    f = WassersteinGeodesicAdaptive(window=90, lookback=20)
    f.fit(r)
    q1 = f.predict(1, u)
    q5 = f.predict(5, u)
    q21 = f.predict(21, u)
    for q in (q1, q5, q21):
        assert np.isfinite(q).all()
        assert (np.diff(q) >= -1e-9).all()
    assert (q5.max() - q5.min()) > (q1.max() - q1.min())
    assert (q21.max() - q21.min()) > (q5.max() - q5.min())


def test_adaptive_scale_responds_to_recent_vol_regime():
    """Vol response: if the recent regime is turbulent the predicted spread
    must be substantially wider than if the recent regime is calm — even when
    the long-window shape is identical."""
    from wbtc.forecasters import WassersteinGeodesicAdaptive

    rng = np.random.default_rng(33)
    # 700 days of moderate vol, then either calm or turbulent tail
    base = rng.normal(0, 0.02, size=700)
    calm_tail = rng.normal(0, 0.005, size=100)
    turb_tail = rng.normal(0, 0.05, size=100)
    r_calm = np.concatenate([base, calm_tail])
    r_turb = np.concatenate([base, turb_tail])
    u = make_grid(30)
    f_calm = WassersteinGeodesicAdaptive(window=90, lookback=20)
    f_turb = WassersteinGeodesicAdaptive(window=90, lookback=20)
    f_calm.fit(r_calm)
    f_turb.fit(r_turb)
    q_calm = f_calm.predict(1, u)
    q_turb = f_turb.predict(1, u)
    spread_calm = q_calm.max() - q_calm.min()
    spread_turb = q_turb.max() - q_turb.min()
    assert spread_turb > 2.5 * spread_calm, (
        f"adaptive scale failed to widen: calm spread={spread_calm:.4f}, "
        f"turb spread={spread_turb:.4f}"
    )


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


def test_bivariate_var_garch_rejects_non_view_returns():
    """The harness passes a numpy slice of full_target; passing a separately-
    allocated array with matching values must fail loudly rather than fall
    back to a float-suffix heuristic that could silently misalign."""
    from wbtc.forecasters import BivariateVARGarch

    rng = np.random.default_rng(0)
    n = 60
    btc = rng.normal(0, 0.02, n)
    eth = rng.normal(0, 0.02, n)
    f = BivariateVARGarch(full_target=btc, full_exog=eth)

    # Fresh buffer, same values: rejected.
    with pytest.raises(ValueError, match="numpy slice"):
        f.fit(btc.copy())

    # Actual slice view: accepted (no exception).
    f.fit(btc[:50])


def test_bivariate_var_garch_aligns_arbitrary_window():
    """fit() on a non-tail slice (e.g. middle of full_target) recovers the
    correct end index and uses the matching ETH window — not the last
    `len(returns)` ETH values."""
    from wbtc.forecasters import BivariateVARGarch

    rng = np.random.default_rng(3)
    n = 200
    btc = rng.normal(0, 0.02, n)
    # Make ETH have a strong, position-dependent signal so that wrong
    # alignment (e.g. always using the ETH tail) would shift the fitted
    # cross-coefficient noticeably.
    eth = np.linspace(-0.05, 0.05, n) + rng.normal(0, 0.005, n)
    for t in range(1, n):
        btc[t] += 0.4 * eth[t - 1]

    f = BivariateVARGarch(full_target=btc, full_exog=eth)
    # Fit on a window in the middle, not the tail.
    f.fit(btc[20:120])
    assert f._coef is not None  # type: ignore[attr-defined]
    cross_mid = float(f._coef[0, 2])  # type: ignore[index]

    # Compare to a fit where full_target == the same window (so the
    # alignment is trivially correct). Coefficients should match closely.
    btc_mid = btc[20:120].copy()
    eth_mid = eth[20:120].copy()
    f2 = BivariateVARGarch(full_target=btc_mid, full_exog=eth_mid)
    f2.fit(btc_mid)
    cross_trivial = float(f2._coef[0, 2])  # type: ignore[index]
    assert abs(cross_mid - cross_trivial) < 1e-10, (
        f"middle-slice fit drifted from trivial fit: {cross_mid} vs {cross_trivial}"
    )


# ---------------------------------------------------------------------------
# Protocol conformance — every exported forecaster must implement Forecaster
# ---------------------------------------------------------------------------


def _forecaster_factories():
    """One factory per exported forecaster class, including the multivariate
    BivariateVARGarch which requires a paired exogenous series at construction.
    """
    from wbtc import forecasters as fc

    n = 32
    zeros = np.zeros(n, dtype=float)
    return [
        ("StaticEmpirical", fc.StaticEmpirical),
        ("RandomWalkDrift", fc.RandomWalkDrift),
        ("GarchNormal", fc.GarchNormal),
        ("GarchStudentT", fc.GarchStudentT),
        ("GJRGarchStudentT", fc.GJRGarchStudentT),
        ("HistoricalSimulationBootstrap", fc.HistoricalSimulationBootstrap),
        ("WassersteinGeodesic", fc.WassersteinGeodesic),
        ("WassersteinGeodesicTheilSen", fc.WassersteinGeodesicTheilSen),
        ("WassersteinGeodesicGated", fc.WassersteinGeodesicGated),
        ("WassersteinGeodesicEWMA", fc.WassersteinGeodesicEWMA),
        ("WassersteinGeodesicHetero", fc.WassersteinGeodesicHetero),
        ("WassersteinGeodesicAdaptive", fc.WassersteinGeodesicAdaptive),
        ("WassersteinGeodesicCondShape", fc.WassersteinGeodesicCondShape),
        ("WGeoEnsemble", fc.WGeoEnsemble),
        ("WGeoGarchEnsemble", fc.WGeoGarchEnsemble),
        ("HARRV", fc.HARRV),
        ("CAViaRSAV", fc.CAViaRSAV),
        ("MarkovSwitching2", fc.MarkovSwitching2),
        ("FIGARCH", fc.FIGARCH),
        ("StochasticVolatilityAR1", fc.StochasticVolatilityAR1),
        (
            "BivariateVARGarch",
            lambda: fc.BivariateVARGarch(full_target=zeros, full_exog=zeros),
        ),
    ]


@pytest.mark.parametrize(
    "name,factory",
    _forecaster_factories(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_forecaster_protocol_conformance(name, factory):
    """Every exported forecaster satisfies the Forecaster Protocol.

    runtime_checkable Protocols only verify method *existence*, not
    signature — that's good enough to catch the drift case the candidate
    cares about (a new forecaster missing fit or predict). Signature
    enforcement is the static-type-checker's job.
    """
    instance = factory()
    assert isinstance(instance, Forecaster), (
        f"{name} does not satisfy the Forecaster Protocol"
    )
