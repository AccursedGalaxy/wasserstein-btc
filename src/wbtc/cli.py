"""`wbtc` command-line interface.

Exposed via [project.scripts] in pyproject.toml, so after `uv sync` you get:

    wbtc fetch BTC/USDT ETH/USDT
    wbtc info
    wbtc forecast BTC/USDT --horizon 5
    wbtc forecast BTC/USDT --horizon 5 --json
    wbtc backtest --symbol BTC/USDT --horizon 5 --quick
    wbtc backtest-long
    wbtc sweep
    wbtc test

The CLI is intentionally thin: it dispatches to the same library functions
documented in `src/wbtc/__init__.py`. Use those directly from Python for
programmatic access.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import (
    __version__,
    available_symbols,
    data_info,
    default_forecaster,
    forecast as api_forecast,
    load_returns,
)
from .data import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module in-process.

    ``scripts/`` is not a Python package (and the wheel doesn't ship it), so
    we load by path under a private namespace. Source checkouts only — pip-
    installed wheel users get a clear ImportError instead of a silent
    FileNotFoundError from the previous subprocess-call approach.
    """
    path = ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — this command requires the source checkout."
        )
    spec = importlib.util.spec_from_file_location(f"_wbtc_scripts.{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- subcommands ----------


def cmd_info(args: argparse.Namespace) -> int:
    """Print a one-screen overview of the repo state."""
    syms = available_symbols()
    if not syms:
        print(
            "No cached data. Run `wbtc fetch BTC/USDT ETH/USDT` to download.",
            file=sys.stderr,
        )
        return 1
    print(f"wbtc {__version__}    data dir: {DATA_DIR}")
    print()
    print(f"{'symbol':<14} {'rows':>6} {'first':<12} {'last':<12} {'sha8':<9}")
    print("-" * 60)
    for s in syms:
        try:
            info = data_info(s)
            print(
                f"{info.symbol:<14} {info.n_rows:>6} {info.first_date:<12} {info.last_date:<12} {info.sha256_8:<9}"
            )
        except Exception as e:
            print(f"{s:<14} <error: {e}>")
    print()
    print("Default forecaster (h=1):", type(default_forecaster(1)).__name__)
    print("Default forecaster (h=5):", type(default_forecaster(5)).__name__)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Delegate to scripts/fetch_data.py — it already does the right thing."""
    mod = _load_script("fetch_data")
    return int(mod.main(args.symbols) or 0)


def _fan_chart_png(
    df, fc, png_path: Path, history_days: int = 60, title: str | None = None
) -> None:
    """Draw a fan chart: recent close prices + forecast quantile envelope."""
    recent = df.tail(history_days).reset_index(drop=True)
    last_close = float(df["close"].iloc[-1])
    last_ts = df["ts"].iloc[-1]
    # forecast quantiles are LOG-returns over horizon h
    fc_close = last_close * np.exp(fc.quantile_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recent["ts"], recent["close"], "k-", lw=1.4, label="realised close")
    fc_ts = last_ts + np.timedelta64(fc.horizon, "D")
    # plot interior fan: 5/25/50/75/95 envelopes as vertical bars at the forecast time
    for lo, hi, alpha, label in [
        (0.05, 0.95, 0.15, "5–95% band"),
        (0.25, 0.75, 0.30, "25–75% band"),
    ]:
        q_lo = float(np.interp(lo, fc.quantile_levels, fc.quantile_values))
        q_hi = float(np.interp(hi, fc.quantile_levels, fc.quantile_values))
        ax.fill_between(
            [last_ts, fc_ts],
            [last_close, last_close * np.exp(q_lo)],
            [last_close, last_close * np.exp(q_hi)],
            alpha=alpha,
            color="C0",
            label=label,
        )
    ax.plot(
        [last_ts, fc_ts],
        [last_close, last_close * np.exp(fc.median)],
        "C0--",
        lw=1.4,
        label="forecast median",
    )
    ax.scatter([fc_ts], [last_close * np.exp(fc.median)], color="C0", s=30, zorder=5)
    ax.axvline(last_ts, color="k", lw=0.6, alpha=0.3)
    ax.set_title(
        title or f"{fc.symbol} — {fc.method}, h={fc.horizon}d, asof {last_ts.date()}"
    )
    ax.set_ylabel("close (USDT)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def cmd_forecast(args: argparse.Namespace) -> int:
    fc = api_forecast(
        args.symbol, horizon=args.horizon, train_window_days=args.train_window
    )
    df = load_returns(args.symbol)
    if args.json:
        print(json.dumps(fc.to_dict(), indent=2))
    else:
        print(
            f"{args.symbol}  asof {fc.asof.date()}  h={fc.horizon}d  method={fc.method}"
        )
        print(
            f"  trained on {args.train_window} days "
            f"({fc.train_data_first.date()} → {fc.train_data_last.date()})"
        )
        print(
            f"  median log-return: {fc.median:+.4f}  ({(np.exp(fc.median) - 1) * 100:+.2f}%)"
        )
        for u in [0.05, 0.25, 0.5, 0.75, 0.95]:
            q = fc.quantile(u)
            pct = (np.exp(q) - 1) * 100
            bar = "█" * max(1, int(abs(pct) * 1.5))
            side = "+" if pct >= 0 else "-"
            print(f"  q{u:>4.2f}  log {q:+.4f}  ({side}{abs(pct):5.2f}%)  {bar}")
    if args.plot:
        png = (
            RESULTS
            / f"forecast_{args.symbol.lower().replace('/', '')}_h{args.horizon}.png"
        )
        _fan_chart_png(df, fc, png)
        print(f"\nfan chart -> {png}", file=sys.stderr)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Single-symbol single-horizon backtest, optionally --quick."""
    from .backtest import BacktestConfig, compare_methods
    from .forecasters import (
        GarchNormal,
        StaticEmpirical,
        WassersteinGeodesicGated,
        WassersteinGeodesicTheilSen,
    )

    df = load_returns(args.symbol)
    returns = df["r"].to_numpy()
    cfg = BacktestConfig(
        train_window=args.train_window,
        horizon=args.horizon,
        K=args.K,
        test_holdout=180 if args.quick else 365,
    )
    methods = {
        "Static": StaticEmpirical,
        "GARCH-N": GarchNormal,
        "WGeo-Gated": lambda: WassersteinGeodesicGated(window=90, lookback=20),
        "WGeo-TheilSen": lambda: WassersteinGeodesicTheilSen(window=90, lookback=20),
    }
    out = compare_methods(returns, methods, cfg)
    summary = out["summary"]
    print(f"{args.symbol}  h={args.horizon}  n_test={summary['n_test'].iloc[0]}")
    for name, row in summary.iterrows():
        print(
            f"  {name:<18}  mean_crps {row['mean_crps']:.6f}  "
            f"[{row['ci_lo']:.6f}, {row['ci_hi']:.6f}]"
        )
    return 0


def cmd_backtest_long(args: argparse.Namespace) -> int:
    mod = _load_script("run_long_horizon")
    return int(mod.main() or 0)


def cmd_var_es(args: argparse.Namespace) -> int:
    """VaR / Expected-Shortfall multi-asset panel (Kupiec, Christoffersen, AS)."""
    mod = _load_script("run_var_es_backtest")
    return int(mod.main() or 0)


def cmd_extended_baselines(args: argparse.Namespace) -> int:
    """Extended econometric baselines (HAR-RV, CAViaR, MS, FIGARCH, SV, BVAR)."""
    mod = _load_script("run_extended_baselines")
    argv: list[str] = []
    if args.parallel != 1:
        argv += ["--parallel", str(args.parallel)]
    if args.stride != 1:
        argv += ["--stride", str(args.stride)]
    if args.out is not None:
        argv += ["--out", str(args.out)]
    return int(mod.main(argv) or 0)


def cmd_sweep(args: argparse.Namespace) -> int:
    mod = _load_script("hyperparam_sweep")
    return int(mod.main() or 0)


def cmd_test(args: argparse.Namespace) -> int:
    import subprocess

    return subprocess.call([sys.executable, "-m", "pytest", *args.pytest_args])


# ---------- argparse wiring ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wbtc",
        description="Wasserstein-geodesic distributional forecasting toolkit.",
    )
    p.add_argument("-V", "--version", action="version", version=f"wbtc {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show cached data inventory + defaults.")
    p_info.set_defaults(fn=cmd_info)

    p_fetch = sub.add_parser(
        "fetch", help="Fetch / update OHLCV from Binance via ccxt."
    )
    p_fetch.add_argument(
        "symbols", nargs="*", help="e.g. BTC/USDT ETH/USDT (default: BTC, ETH, SOL)"
    )
    p_fetch.set_defaults(fn=cmd_fetch)

    p_fc = sub.add_parser("forecast", help="Produce today's distributional forecast.")
    p_fc.add_argument("symbol", help="e.g. BTC/USDT")
    p_fc.add_argument(
        "--horizon", "-H", type=int, default=5, help="Days ahead (default 5)."
    )
    p_fc.add_argument(
        "--train-window", type=int, default=730, help="Training window length in days."
    )
    p_fc.add_argument(
        "--json", action="store_true", help="Emit JSON to stdout (good for piping)."
    )
    p_fc.add_argument(
        "--plot", action="store_true", help="Also write a fan chart PNG to results/."
    )
    p_fc.set_defaults(fn=cmd_forecast)

    p_bt = sub.add_parser(
        "backtest", help="Quick single-symbol single-horizon backtest."
    )
    p_bt.add_argument("--symbol", default="BTC/USDT")
    p_bt.add_argument("--horizon", "-H", type=int, default=5)
    p_bt.add_argument("--train-window", type=int, default=730)
    p_bt.add_argument("--K", type=int, default=50)
    p_bt.add_argument(
        "--quick",
        action="store_true",
        help="Smaller test holdout (180d instead of 365d).",
    )
    p_bt.set_defaults(fn=cmd_backtest)

    p_btl = sub.add_parser(
        "backtest-long", help="Run the full long-horizon multi-asset backtest."
    )
    p_btl.set_defaults(fn=cmd_backtest_long)

    p_vares = sub.add_parser(
        "var-es",
        help="VaR / ES tail-calibration backtest panel (Kupiec, Christoffersen, AS).",
    )
    p_vares.set_defaults(fn=cmd_var_es)

    p_ext = sub.add_parser(
        "extended-baselines",
        help="Run the extended econometric-baseline comparison (HAR-RV, CAViaR, MS, FIGARCH, SV, BVAR).",
    )
    p_ext.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Worker processes for the method loop (1=sequential, -1=all cores).",
    )
    p_ext.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Refit every K steps (1=every step, 5=weekly-refit risk-system style).",
    )
    p_ext.add_argument(
        "--out",
        default=None,
        help="Override output markdown path (default: docs/RESULTS_EXTENDED.md).",
    )
    p_ext.set_defaults(fn=cmd_extended_baselines)

    p_sw = sub.add_parser("sweep", help="Hyperparameter robustness sweep.")
    p_sw.set_defaults(fn=cmd_sweep)

    p_t = sub.add_parser("test", help="Run pytest on the library.")
    p_t.add_argument("pytest_args", nargs="*", help="Forwarded to pytest.")
    p_t.set_defaults(fn=cmd_test)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
