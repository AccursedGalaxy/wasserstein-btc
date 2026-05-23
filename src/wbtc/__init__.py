"""Tangent-space Wasserstein-geodesic distributional forecasting for crypto returns.

# Quick start

    from wbtc import forecast, load_returns, available_symbols

    available_symbols()                       # ['BTC/USDT', 'ETH/USDT', ...]
    df = load_returns("BTC/USDT")             # ts, close, log_close, r
    fc = forecast("BTC/USDT", horizon=5)      # ForecastResult
    fc.median                                  # central forecast log-return
    fc.quantiles                               # dict {u -> log-return}
    fc.to_dict()                               # JSON-safe summary

# What the default forecaster is

`default_forecaster()` returns a `WassersteinGeodesicTheilSen(window=90,
lookback=20)` for `h >= 5` and a `WassersteinGeodesicGated(window=90,
lookback=20, kappa_star=0.6, tau=5)` for `h == 1`. These are the
hyperparameters validated in `docs/RESULTS_LONG.md`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    DATA_DIR,
    DataInfo,
    available_symbols,
    data_info,
    load_ohlcv,
    load_returns,
)
from .forecasters import (
    GJRGarchStudentT,
    GarchNormal,
    GarchStudentT,
    HistoricalSimulationBootstrap,
    RandomWalkDrift,
    StaticEmpirical,
    WassersteinGeodesic,
    WassersteinGeodesicGated,
    WassersteinGeodesicTheilSen,
)
from .quantiles import make_grid
from .scoring import crps_from_quantiles, diebold_mariano

__version__ = "0.2.0"

__all__ = [
    # data
    "DATA_DIR",
    "DataInfo",
    "available_symbols",
    "data_info",
    "load_ohlcv",
    "load_returns",
    # forecasters
    "StaticEmpirical",
    "RandomWalkDrift",
    "HistoricalSimulationBootstrap",
    "GarchNormal",
    "GarchStudentT",
    "GJRGarchStudentT",
    "WassersteinGeodesic",
    "WassersteinGeodesicGated",
    "WassersteinGeodesicTheilSen",
    # high-level API
    "ForecastResult",
    "default_forecaster",
    "forecast",
    # scoring
    "make_grid",
    "crps_from_quantiles",
    "diebold_mariano",
    "__version__",
]


DEFAULT_K = 50  # quantile grid size for the public API


@dataclass
class ForecastResult:
    """Container for a single distributional forecast."""

    symbol: str
    asof: pd.Timestamp
    horizon: int
    method: str
    quantile_levels: np.ndarray
    quantile_values: np.ndarray
    train_window_days: int
    train_data_first: pd.Timestamp
    train_data_last: pd.Timestamp

    @property
    def median(self) -> float:
        """Forecast median log-return over [asof, asof + horizon days]."""
        return float(np.interp(0.5, self.quantile_levels, self.quantile_values))

    @property
    def mean(self) -> float:
        """Forecast mean log-return (trapezoidal integral over quantile grid)."""
        return float(np.trapezoid(self.quantile_values, self.quantile_levels) / 1.0)

    def quantile(self, u: float) -> float:
        """Interpolated forecast quantile at level u in (0, 1)."""
        return float(np.interp(u, self.quantile_levels, self.quantile_values))

    @property
    def quantiles(self) -> dict[float, float]:
        return {
            float(u): float(q)
            for u, q in zip(self.quantile_levels, self.quantile_values)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "horizon_days": self.horizon,
            "method": self.method,
            "train_window_days": self.train_window_days,
            "train_data_first": self.train_data_first.isoformat(),
            "train_data_last": self.train_data_last.isoformat(),
            "median": self.median,
            "mean": self.mean,
            "quantile_levels": self.quantile_levels.tolist(),
            "quantile_values": self.quantile_values.tolist(),
            "notable_quantiles": {
                "0.05": self.quantile(0.05),
                "0.25": self.quantile(0.25),
                "0.50": self.median,
                "0.75": self.quantile(0.75),
                "0.95": self.quantile(0.95),
            },
        }


def default_forecaster(horizon: int):
    """Return the recommended forecaster instance for a given horizon.

    Justification: per `docs/RESULTS_LONG.md`, the curvature-gate variant wins
    at h=1 and the Theil-Sen robust-slope variant wins at h >= 5.
    """
    if horizon <= 1:
        return WassersteinGeodesicGated(window=90, lookback=20, kappa_star=0.6, tau=5)
    return WassersteinGeodesicTheilSen(window=90, lookback=20)


def forecast(
    symbol: str,
    horizon: int = 5,
    asof: pd.Timestamp | str | None = None,
    train_window_days: int = 730,
    K: int = DEFAULT_K,
    forecaster: object | None = None,
) -> ForecastResult:
    """Return a distributional forecast for `symbol` from the cached data.

    Parameters
    ----------
    symbol
        e.g. "BTC/USDT" — must have been fetched via `wbtc fetch`.
    horizon
        Forecast horizon in days.
    asof
        Cutoff timestamp — the forecast uses only data with `ts <= asof`. If
        None, uses the latest available day.
    train_window_days
        Length of the rolling training window passed to the forecaster.
    K
        Number of quantile grid points.
    forecaster
        Pre-built forecaster object (e.g. for an ablation). If None, uses
        `default_forecaster(horizon)`.
    """
    df = load_returns(symbol)
    if asof is not None:
        cutoff = pd.Timestamp(asof, tz="UTC")
        df = df[df["ts"] <= cutoff].reset_index(drop=True)
    if len(df) < train_window_days:
        raise ValueError(
            f"only {len(df)} returns available for {symbol}, need {train_window_days}"
        )
    window = df["r"].to_numpy()[-train_window_days:]
    u = make_grid(K)

    fc = forecaster if forecaster is not None else default_forecaster(horizon)
    fc.fit(window)  # type: ignore[attr-defined]
    q = fc.predict(horizon, u)  # type: ignore[attr-defined]

    return ForecastResult(
        symbol=symbol,
        asof=df["ts"].iloc[-1],
        horizon=horizon,
        method=type(fc).__name__,
        quantile_levels=np.asarray(u, dtype=float),
        quantile_values=np.asarray(q, dtype=float),
        train_window_days=train_window_days,
        train_data_first=df["ts"].iloc[-train_window_days],
        train_data_last=df["ts"].iloc[-1],
    )
