"""Walk-forward backtest of distributional forecasters on BTC log-returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def walk_forward(
    returns: np.ndarray,
    make_forecaster: Callable[[], object],
    cfg: BacktestConfig,
    label: str = "method",
) -> pd.DataFrame:
    """Run a walk-forward backtest. Returns per-step CRPS + realised return."""
    u = make_grid(cfg.K)
    n = cfg.train_window
    rows: list[dict] = []
    iterator = range(n, len(returns) - cfg.horizon)
    for t in tqdm(iterator, desc=label, leave=False):
        window = returns[t - n : t]
        f = make_forecaster()
        try:
            f.fit(window)  # type: ignore[attr-defined]
            q = f.predict(cfg.horizon, u)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - defensive
            rows.append({"t": t, "crps": np.nan, "y": np.nan, "err": str(e)[:80]})
            continue
        y = h_step_log_return(returns, t, cfg.horizon)
        if y is None:
            continue
        crps = crps_from_quantiles(q, u, y)
        rows.append({"t": t, "crps": crps, "y": y})
    return pd.DataFrame(rows)


def compare_methods(
    returns: np.ndarray,
    method_factories: dict[str, Callable[[], object]],
    cfg: BacktestConfig,
) -> dict:
    """Run backtest for every method and compute pairwise DM tests."""
    results = {
        name: walk_forward(returns, fac, cfg, label=name)
        for name, fac in method_factories.items()
    }

    # Align all methods to the common index
    common_idx = None
    for df in results.values():
        idx = set(df["t"].tolist())
        common_idx = idx if common_idx is None else common_idx & idx
    common_idx = sorted(common_idx or set())

    aligned = {
        name: df.set_index("t").loc[common_idx, "crps"].to_numpy()
        for name, df in results.items()
    }
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
