"""Fast-iteration harness for new forecaster variants.

Loads the per-step CRPS arrays already saved under ``results/long_{symbol}_h{h}.json``
by the long-horizon backtest, then runs ONE new method on the same walk-forward
indices, and reports:

  - Mean CRPS per cell (asset x horizon).
  - DM p-value vs every saved method.
  - The current "best non-WGeo baseline" per cell + new-method DM vs it.

This avoids re-running the entire 12-method panel just to evaluate a new
variant. The new method must follow the standard fit/predict protocol used by
the long-horizon harness; it is constructed by a factory callable passed on the
command line via ``--method``.

Usage:
    uv run python scripts/score_new_method.py --method WGeo-Adaptive
    uv run python scripts/score_new_method.py --method WGeo-Adaptive --symbols BTC/USDT ETH/USDT --horizons 5 21
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wbtc.backtest import h_step_log_return, load_returns
from wbtc.forecasters import (
    WassersteinGeodesic,
    WassersteinGeodesicEWMA,
    WassersteinGeodesicTheilSen,
)
from wbtc.quantiles import make_grid
from wbtc.scoring import (
    crps_from_quantiles,
    diebold_mariano,
    diebold_mariano_residualised,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

WGEO_WINDOW = 90
WGEO_LOOKBACK = 20
BURN_IN = 730

NON_WGEO_NAMES = {
    "Static",
    "RW-Drift",
    "HS-Bootstrap",
    "GARCH-N",
    "GARCH-t",
    "GJR-GARCH-t",
}


def slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def _resolve_factory(name: str):
    """Map a method name to a no-arg factory.

    Recognised names live in this module; alternatively pass a fully-qualified
    ``module:attr`` string to load a custom factory.
    """
    if ":" in name:
        mod, attr = name.split(":")
        m = importlib.import_module(mod)
        return getattr(m, attr)
    if name in _BUILTIN_FACTORIES:
        return _BUILTIN_FACTORIES[name]
    raise ValueError(f"unknown method: {name}")


def _make_adaptive():
    from wbtc.forecasters import WassersteinGeodesicAdaptive

    return WassersteinGeodesicAdaptive(window=WGEO_WINDOW, lookback=WGEO_LOOKBACK)


def _make_condshape():
    from wbtc.forecasters import WassersteinGeodesicCondShape

    return WassersteinGeodesicCondShape(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, shape_window=500
    )


def _make_ensemble():
    from wbtc.forecasters import WGeoEnsemble

    return WGeoEnsemble()


_BUILTIN_FACTORIES = {
    # sanity-checking baselines
    "WGeo": lambda: WassersteinGeodesic(window=WGEO_WINDOW, lookback=WGEO_LOOKBACK),
    "WGeo-TheilSen": lambda: WassersteinGeodesicTheilSen(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
    "WGeo-EWMA": lambda: WassersteinGeodesicEWMA(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, decay=0.85
    ),
    # candidates under evaluation
    "WGeo-Adaptive": _make_adaptive,
    "WGeo-CondShape": _make_condshape,
    "WGeo-Ensemble": _make_ensemble,
}


def run_method_on_indices(
    returns: np.ndarray, factory, t_indices: list[int], horizon: int, K: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward CRPS at exactly the indices found in the saved JSON.

    Returns
    -------
    losses : (N,) array of per-step CRPS (NaN where the prediction failed)
    ys     : (N,) array of realised h-step returns
    """
    u = make_grid(K)
    losses = np.full(len(t_indices), np.nan, dtype=float)
    ys = np.full(len(t_indices), np.nan, dtype=float)
    for i, t in enumerate(t_indices):
        window = returns[t - BURN_IN : t]
        f = factory()
        try:
            f.fit(window)
            q = f.predict(horizon, u)
        except Exception:
            continue
        y = h_step_log_return(returns, t, horizon)
        if y is None:
            continue
        losses[i] = crps_from_quantiles(q, u, y)
        ys[i] = y
    return losses, ys


def score_cell(method_name: str, symbol: str, horizon: int, K: int = 30) -> dict:
    sym_slug = slug(symbol)
    saved_path = RESULTS / f"long_{sym_slug}_h{horizon}.json"
    if not saved_path.exists():
        raise FileNotFoundError(
            f"saved losses for {symbol} h={horizon} not found at {saved_path}"
        )
    saved = json.loads(saved_path.read_text())
    t_indices = list(saved["t_idx"])

    df = load_returns(DATA / f"{sym_slug}_1d.parquet")
    returns = df["r"].to_numpy()

    factory = _resolve_factory(method_name)
    new_losses, _ = run_method_on_indices(returns, factory, t_indices, horizon, K=K)

    # Drop NaN steps consistently across all methods being compared
    saved_losses = {
        k: np.asarray(v, dtype=float) for k, v in saved.items() if k != "t_idx"
    }
    keep = np.isfinite(new_losses)
    for arr in saved_losses.values():
        keep &= np.isfinite(arr)
    if not keep.any():
        return {
            "symbol": symbol,
            "h": horizon,
            "n": 0,
            "mean_crps": float("nan"),
            "best_non_wgeo": None,
            "best_non_wgeo_crps": float("nan"),
            "dm_vs_best_non_wgeo": float("nan"),
            "p_vs_best_non_wgeo": float("nan"),
        }
    new_losses = new_losses[keep]
    saved_losses = {k: v[keep] for k, v in saved_losses.items()}

    mean_crps = float(new_losses.mean())
    # best non-WGeo baseline by mean CRPS
    non_wgeo = {k: v for k, v in saved_losses.items() if k in NON_WGEO_NAMES}
    best_non_wgeo = min(non_wgeo, key=lambda k: non_wgeo[k].mean())
    best_non_wgeo_crps = float(non_wgeo[best_non_wgeo].mean())
    dm, p = diebold_mariano(new_losses, non_wgeo[best_non_wgeo], h=horizon)
    # Residualised DM: control for shared volatility noise via |y|, y², y plus
    # a panel of NON-best non-WGeo losses (each captures its own volatility-
    # tracking error; the regression projects out their common component).
    df_full = load_returns(DATA / f"{slug(symbol)}_1d.parquet")
    rfull = df_full["r"].to_numpy()
    y_arr = np.array(
        [h_step_log_return(rfull, t, horizon) for t in t_indices], dtype=float
    )
    y_arr = y_arr[keep]
    non_best_losses = [v for k, v in non_wgeo.items() if k != best_non_wgeo][:4]
    controls = non_best_losses + [np.abs(y_arr), y_arr * y_arr]
    dm_r, p_r = diebold_mariano_residualised(
        new_losses, non_wgeo[best_non_wgeo], controls, h=horizon
    )

    # also DM vs every saved method, for inspection
    dm_table = {}
    for k, v in saved_losses.items():
        s, pp = diebold_mariano(new_losses, v, h=horizon)
        dm_table[k] = {
            "mean_crps_other": float(v.mean()),
            "dm": float(s),
            "p": float(pp),
        }

    return {
        "symbol": symbol,
        "h": horizon,
        "n": int(keep.sum()),
        "mean_crps": mean_crps,
        "best_non_wgeo": best_non_wgeo,
        "best_non_wgeo_crps": best_non_wgeo_crps,
        "improvement_pct": (mean_crps - best_non_wgeo_crps) / best_non_wgeo_crps * 100,
        "dm_vs_best_non_wgeo": float(dm),
        "p_vs_best_non_wgeo": float(p),
        "dm_r_vs_best_non_wgeo": float(dm_r),
        "p_r_vs_best_non_wgeo": float(p_r),
        "by_method": dm_table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="method name or module:attr")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    )
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 21])
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = []
    for symbol in args.symbols:
        for h in args.horizons:
            print(f"[score] {args.method} {symbol} h={h} ...", flush=True)
            row = score_cell(args.method, symbol, h, K=args.K)
            print(
                f"  mean_crps={row['mean_crps']:.6f}  "
                f"best_non_wgeo={row['best_non_wgeo']} ({row['best_non_wgeo_crps']:.6f})  "
                f"Δ={row['improvement_pct']:+.2f}%  "
                f"DM={row['dm_vs_best_non_wgeo']:+.2f} p={row['p_vs_best_non_wgeo']:.4f}  "
                f"| DM_r={row['dm_r_vs_best_non_wgeo']:+.2f} p_r={row['p_r_vs_best_non_wgeo']:.4f}",
                flush=True,
            )
            if args.verbose:
                for k, info in row["by_method"].items():
                    print(
                        f"    vs {k:14s}: Δmean={(row['mean_crps'] - info['mean_crps_other']):+.6f} "
                        f"DM={info['dm']:+.2f}  p={info['p']:.4f}"
                    )
            rows.append(row)

    n_wins = sum(
        1
        for r in rows
        if r["p_vs_best_non_wgeo"] < 0.05 and r["mean_crps"] < r["best_non_wgeo_crps"]
    )
    n_wins_r = sum(
        1
        for r in rows
        if r["p_r_vs_best_non_wgeo"] < 0.05 and r["mean_crps"] < r["best_non_wgeo_crps"]
    )
    n_total = len(rows)
    print(
        f"\n[score] {args.method}: vanilla DM {n_wins}/{n_total} cells p<0.05; "
        f"residualised DM {n_wins_r}/{n_total} cells p<0.05."
    )

    # compact summary table
    summary_rows = [
        {
            "symbol": r["symbol"],
            "h": r["h"],
            "n": r["n"],
            "mean_crps": r["mean_crps"],
            "best_baseline": r["best_non_wgeo"],
            "baseline_crps": r["best_non_wgeo_crps"],
            "Δ%": r["improvement_pct"],
            "DM": r["dm_vs_best_non_wgeo"],
            "p": r["p_vs_best_non_wgeo"],
            "DM_r": r["dm_r_vs_best_non_wgeo"],
            "p_r": r["p_r_vs_best_non_wgeo"],
        }
        for r in rows
    ]
    df = pd.DataFrame(summary_rows)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
