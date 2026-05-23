"""Walk-forward backtest of distributional forecasters on BTC log-returns.

This module is the single source of truth for the walk-forward loop. Both
the short-horizon harness (:func:`compare_methods`, holdout-on-tail) and
the long-horizon harness (:func:`wbtc.long_horizon.run_long_horizon`,
stride + parallel) call :func:`_walk_forward_one` for the inner loop and
:func:`_align_methods` for the multi-method index alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .forecasters import Forecaster
from .quantiles import make_grid
from .scoring import crps_from_quantiles, diebold_mariano, stationary_bootstrap_ci

__all__ = ["BacktestConfig", "load_returns", "walk_forward", "compare_methods"]


@dataclass
class BacktestConfig:
    train_window: int = 730  # ~2y of daily bars
    horizon: int = 1
    K: int = 50  # quantile grid size
    test_holdout: int = 365  # last N days are STRICT test set


def load_returns(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path).sort_values("ts").reset_index(drop=True)
    df["log_close"] = np.log(df["close"].astype(float))
    df["r"] = df["log_close"].diff()
    return df.dropna(subset=["r"]).reset_index(drop=True)


def h_step_log_return(returns: np.ndarray, t: int, h: int) -> float | None:
    """Realised cumulative log-return from end of day t+1 to end of day t+h."""
    if t + h >= len(returns):
        return None
    return float(returns[t + 1 : t + h + 1].sum())


# ---------------------------------------------------------------------------
# Unified walk-forward primitives
# ---------------------------------------------------------------------------


def _walk_forward_one(
    returns: np.ndarray,
    make_forecaster: Callable[[], Forecaster],
    *,
    train_window: int,
    horizon: int,
    u: np.ndarray,
    stride: int = 1,
    show_progress: bool = False,
    label: str = "",
) -> pd.DataFrame:
    """Canonical walk-forward loop for one method.

    Iterates ``t`` over ``[train_window, len(returns) - horizon)``. The
    forecaster is refit on the trailing ``train_window`` returns every
    ``stride`` steps and reused for intermediate predictions. ``stride=1``
    is the methodologically conservative refit-every-step setting used by
    the headline panel; ``stride>1`` is the "weekly-refit" approximation
    used by long-horizon production-risk runs.

    Returns
    -------
    DataFrame with columns ``t``, ``crps``, ``y``; rows where fit or
    predict raised carry NaN losses and an ``err`` field with the first 80
    chars of the exception. Rows where the h-step realised return is
    unavailable (end-of-series) are skipped.
    """
    rows: list[dict] = []
    steps = range(train_window, len(returns) - horizon)
    iterator = (
        tqdm(steps, desc=label or "method", leave=False) if show_progress else steps
    )
    f: Forecaster | None = None
    for i, t in enumerate(iterator):
        if (i % stride == 0) or (f is None):
            window = returns[t - train_window : t]
            f = make_forecaster()
            try:
                f.fit(window)
            except Exception as e:
                rows.append({"t": t, "crps": np.nan, "y": np.nan, "err": str(e)[:80]})
                f = None
                continue
        try:
            q = f.predict(horizon, u)
        except Exception as e:
            rows.append({"t": t, "crps": np.nan, "y": np.nan, "err": str(e)[:80]})
            continue
        y = h_step_log_return(returns, t, horizon)
        if y is None:
            continue
        row = {"t": t, "crps": crps_from_quantiles(q, u, y), "y": y}
        # Forecasters that use GARCH conditioning set _garch_fallback on each
        # predict (None when GARCH is disabled). Surface the flag so the
        # long-horizon harness can report the fallback rate per method.
        fb = getattr(f, "_garch_fallback", None)
        if fb is not None:
            row["garch_fallback"] = bool(fb)
        rows.append(row)
    return pd.DataFrame(rows)


def _align_methods(
    per_step: dict[str, pd.DataFrame],
) -> tuple[list[int], dict[str, np.ndarray], np.ndarray]:
    """Intersect per-method ``t`` indices and stack aligned loss vectors.

    Returns
    -------
    common_idx
        Sorted list of timesteps present in every method's per-step frame.
    aligned
        Dict ``method_name -> losses[common_idx]``.
    realised
        The shared realised h-step returns at ``common_idx`` (taken from
        the first method — the realised series does not depend on method).
    """
    common: set[int] | None = None
    for df in per_step.values():
        idx = set(df["t"].tolist())
        common = idx if common is None else common & idx
    common_idx = sorted(common or set())
    aligned = {
        name: df.set_index("t").loc[common_idx, "crps"].to_numpy()
        for name, df in per_step.items()
    }
    if per_step:
        realised = (
            next(iter(per_step.values())).set_index("t").loc[common_idx, "y"].to_numpy()
        )
    else:
        realised = np.empty(0, dtype=float)
    return common_idx, aligned, realised


# ---------------------------------------------------------------------------
# Public single-method / short-horizon API
# ---------------------------------------------------------------------------


def walk_forward(
    returns: np.ndarray,
    make_forecaster: Callable[[], Forecaster],
    cfg: BacktestConfig,
    label: str = "method",
) -> pd.DataFrame:
    """Run a walk-forward backtest. Returns per-step CRPS + realised return."""
    u = make_grid(cfg.K)
    return _walk_forward_one(
        returns,
        make_forecaster,
        train_window=cfg.train_window,
        horizon=cfg.horizon,
        u=u,
        stride=1,
        show_progress=True,
        label=label,
    )


def compare_methods(
    returns: np.ndarray,
    method_factories: dict[str, Callable[[], Forecaster]],
    cfg: BacktestConfig,
) -> dict:
    """Run backtest for every method and compute pairwise DM tests."""
    u = make_grid(cfg.K)
    results = {
        name: _walk_forward_one(
            returns,
            fac,
            train_window=cfg.train_window,
            horizon=cfg.horizon,
            u=u,
            stride=1,
            show_progress=True,
            label=name,
        )
        for name, fac in method_factories.items()
    }

    common_idx, aligned, _ = _align_methods(results)
    test_cut = max(common_idx) - cfg.test_holdout if common_idx else 0
    test_mask = np.array([t > test_cut for t in common_idx])

    summary_rows = []
    for name, losses in aligned.items():
        test_losses = losses[test_mask]
        mean, lo, hi = stationary_bootstrap_ci(
            test_losses, block_mean=max(2.0, cfg.horizon)
        )
        summary_rows.append(
            {
                "method": name,
                "n_test": int(test_mask.sum()),
                "mean_crps": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("method")

    # pairwise DM table (using test-period losses)
    methods = list(aligned.keys())
    dm_table = pd.DataFrame(index=methods, columns=methods, dtype=float)
    p_table = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            if a == b:
                dm_table.loc[a, b] = 0.0
                p_table.loc[a, b] = 1.0
                continue
            dm, p = diebold_mariano(
                aligned[a][test_mask], aligned[b][test_mask], h=cfg.horizon
            )
            dm_table.loc[a, b] = dm
            p_table.loc[a, b] = p

    return {
        "per_step": results,
        "aligned": aligned,
        "test_mask": test_mask,
        "summary": summary,
        "dm_stat": dm_table,
        "dm_p": p_table,
        "test_cut_t": test_cut,
    }
