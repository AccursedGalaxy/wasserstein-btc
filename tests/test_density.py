import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from wbtc.density import (
    DEFAULT_SPEC,
    DensitySpec,
    build_daily_densities,
    intraday_log_returns,
    kde_quantiles,
    silverman_bandwidth,
)


def _ohlcv(days: int, per_day: int = 288, seed: int = 0, sigma: float = 0.003):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-01", periods=days * per_day, freq="5min", tz="UTC")
    logp = np.cumsum(rng.normal(0.0, sigma, size=ts.size))
    return pd.DataFrame({"ts": ts, "close": np.exp(logp)})


def test_silverman_bandwidth_scales_with_spread_and_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    h1 = silverman_bandwidth(x)
    assert silverman_bandwidth(2.0 * x) == pytest.approx(2.0 * h1)
    assert silverman_bandwidth(rng.normal(size=5000)) < h1
    assert silverman_bandwidth(np.zeros(50)) == DEFAULT_SPEC.min_bandwidth


def test_kde_quantiles_monotone_and_recover_normal():
    rng = np.random.default_rng(2)
    sigma = 0.004
    x = rng.normal(0.0, sigma, size=20_000)
    q, h = kde_quantiles(x)
    assert q.shape == (DEFAULT_SPEC.k,)
    assert np.all(np.diff(q) >= 0)
    assert h > 0
    # mid-body quantiles should match the Normal within a small tolerance;
    # KDE smoothing widens the tails slightly, so compare the 10%-90% range.
    u = DEFAULT_SPEC.u_grid()
    body = (u > 0.1) & (u < 0.9)
    np.testing.assert_allclose(q[body], norm.ppf(u[body], 0.0, sigma), atol=2e-4)


def test_kde_quantiles_truncate_to_support():
    spec = DensitySpec(support_lo=-0.01, support_hi=0.01)
    x = np.array([-0.05, 0.0, 0.05, 0.2, -0.2])
    q, _ = kde_quantiles(x, spec)
    assert q.min() >= spec.support_lo and q.max() <= spec.support_hi


def test_kde_quantiles_deterministic():
    rng = np.random.default_rng(3)
    x = rng.standard_t(3, size=287) * 0.003
    q1, h1 = kde_quantiles(x)
    q2, h2 = kde_quantiles(x)
    np.testing.assert_array_equal(q1, q2)
    assert h1 == h2


def test_intraday_log_returns_drop_overnight_gap():
    df = _ohlcv(days=3)
    r = intraday_log_returns(df)
    assert len(r) == 3 * 287
    assert r.groupby("day").size().tolist() == [287, 287, 287]
    # manual check of the first within-day return
    expected = np.log(df["close"].iloc[1]) - np.log(df["close"].iloc[0])
    assert r["r"].iloc[0] == pytest.approx(expected)


def test_build_daily_densities_flags_short_and_calendar_days():
    df = _ohlcv(days=4)
    # amputate day 2 to 100 candles, and calendar-exclude day 4
    day2 = pd.Timestamp("2021-01-02", tz="UTC").date()
    keep = ~((df["ts"].dt.date == day2) & (df["ts"].dt.hour >= 8))
    df = df[keep].reset_index(drop=True)
    day4 = pd.Timestamp("2021-01-04", tz="UTC").date()
    out = build_daily_densities(df, exclude_days=frozenset({day4}))
    assert out["day"].tolist() == sorted(out["day"].tolist())
    assert len(out) == 4
    by_day = out.set_index("day")
    assert not by_day.loc[pd.Timestamp("2021-01-01").date(), "is_excluded"]
    assert by_day.loc[day2, "is_excluded"]
    assert by_day.loc[day2, "exclusion_reason"].startswith("short_day:")
    assert by_day.loc[day4, "is_excluded"]
    assert by_day.loc[day4, "exclusion_reason"] == "calendar"
    # every row carries a stackable, monotone quantile vector
    Q = np.stack(out["quantile"].to_numpy())
    assert Q.shape == (4, DEFAULT_SPEC.k)
    assert np.all(np.diff(Q, axis=1) >= 0)
    assert (out["n_obs"] == [287, 95, 287, 287]).all()
