"""Long-horizon (multi-year) backtest with per-year + per-regime tables.

For each (symbol, horizon) writes:
  results/long_<symbol>_h{h}.json   per-step CRPS arrays
  docs/RESULTS_LONG.md              human-readable report (overwritten)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wbtc.backtest import load_returns
from wbtc.forecasters import (
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
from wbtc.long_horizon import (
    by_regime,
    by_year,
    overall_summary,
    pairwise_dm,
    run_long_horizon,
    tag_regimes,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DOC = ROOT / "docs" / "RESULTS_LONG.md"

WGEO_WINDOW = 90
WGEO_LOOKBACK = 20
KAPPA_STAR = 0.6
TAU = 5

METHODS = {
    "Static": StaticEmpirical,
    "RW-Drift": RandomWalkDrift,
    "HS-Bootstrap": lambda: HistoricalSimulationBootstrap(n_paths=3000, rng_seed=0),
    "GARCH-N": GarchNormal,
    "GARCH-t": GarchStudentT,
    "GJR-GARCH-t": GJRGarchStudentT,
    "WGeo": lambda: WassersteinGeodesic(window=WGEO_WINDOW, lookback=WGEO_LOOKBACK),
    "WGeo-Gated": lambda: WassersteinGeodesicGated(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, kappa_star=KAPPA_STAR, tau=TAU
    ),
    "WGeo-TheilSen": lambda: WassersteinGeodesicTheilSen(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
}

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
HORIZONS = [1, 5, 21]
BURN_IN = 730


def slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def fmt_pct_diff(a: float, b: float) -> str:
    if b == 0:
        return "—"
    return f"{(a - b) / b * 100:+.1f}%"


def main():
    from wbtc.manifest import write_manifest

    md = [
        "# Long-Horizon Results — Multi-Year, Multi-Asset Validation",
        "",
        "Goal: prove the Wasserstein-Geodesic forecaster works over a *long* time horizon.",
        "Train: rolling 730-day window. Test: every day after burn-in (no separate holdout).",
        "Scoring: CRPS (lower better, strictly proper).",
        "",
    ]

    headline_rows: list[dict] = []

    for symbol in SYMBOLS:
        sym_slug = slug(symbol)
        df = load_returns(DATA / f"{sym_slug}_1d.parquet")
        returns = df["r"].to_numpy()
        timestamps = df["ts"]
        regime_tags = tag_regimes(returns, lookback=60)
        md += [
            f"## {symbol}",
            "",
            f"_{len(df)} days from {df['ts'].min().date()} to {df['ts'].max().date()}_",
            "",
        ]

        for h in HORIZONS:
            print(f"\n[long-run] {symbol} h={h}")
            res = run_long_horizon(
                returns, timestamps, METHODS, burn_in=BURN_IN, horizon=h, K=30
            )
            # save raw
            out = {name: losses.tolist() for name, losses in res.aligned_losses.items()}
            out["t_idx"] = res.aligned_idx
            (RESULTS / f"long_{sym_slug}_h{h}.json").write_text(json.dumps(out))

            summary = overall_summary(res, horizon=h)
            yr = by_year(res, timestamps)
            reg = by_regime(res, regime_tags)
            dm_stat, dm_p = pairwise_dm(res, horizon=h)

            # headline row: WGeo (best of three variants) vs best of GARCH family
            wgeo_variants = ["WGeo", "WGeo-Gated", "WGeo-TheilSen"]
            wgeo_best_name = summary.loc[wgeo_variants, "mean_crps"].idxmin()
            garch_variants = ["GARCH-N", "GARCH-t", "GJR-GARCH-t"]
            garch_best_name = summary.loc[garch_variants, "mean_crps"].idxmin()
            wbest = float(summary.loc[wgeo_best_name, "mean_crps"])
            gbest = float(summary.loc[garch_best_name, "mean_crps"])
            stat, p = (
                float(dm_stat.loc[wgeo_best_name, garch_best_name]),
                float(dm_p.loc[wgeo_best_name, garch_best_name]),
            )
            headline_rows.append(
                {
                    "symbol": symbol,
                    "h": h,
                    "n_test": int(summary["n"].iloc[0]),
                    "best_wgeo": wgeo_best_name,
                    "best_garch": garch_best_name,
                    "wgeo_crps": wbest,
                    "garch_crps": gbest,
                    "improvement": fmt_pct_diff(wbest, gbest),
                    "dm_stat": stat,
                    "dm_p": p,
                }
            )

            md += [
                f"### Horizon h = {h} day(s)",
                "",
                "**Overall mean CRPS on the full test span (bootstrap 95% CI):**",
                "",
                summary.map(
                    lambda x: f"{x:.6f}" if isinstance(x, float) else x
                ).to_markdown(),
                "",
                "**Per-year mean CRPS:**",
                "",
                yr.map(
                    lambda x: f"{x:.5f}" if isinstance(x, float) else x
                ).to_markdown(),
                "",
                "**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**",
                "",
                reg.map(
                    lambda x: f"{x:.5f}" if isinstance(x, float) else x
                ).to_markdown(),
                "",
                "**Diebold-Mariano p-values vs WGeo-TheilSen** (lower = WGeo-TheilSen significantly better):",
                "",
                dm_p["WGeo-TheilSen"]
                .round(4)
                .to_frame(name="p_vs_WGeo-TheilSen")
                .to_markdown(),
                "",
            ]

            # cumulative CRPS plot
            fig, ax = plt.subplots(figsize=(10, 4))
            xs = np.arange(len(res.aligned_idx))
            for name, losses in res.aligned_losses.items():
                ax.plot(xs, np.cumsum(losses), label=name, lw=1.0)
            ax.set_title(f"Cumulative CRPS — {symbol} h={h}d ({len(xs)} test steps)")
            ax.set_xlabel("step")
            ax.set_ylabel("cumulative CRPS")
            ax.legend(loc="upper left", fontsize=7, ncols=2)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            png = RESULTS / f"long_cum_crps_{sym_slug}_h{h}.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            md += [f"![cumulative CRPS](../results/{png.name})", ""]

    md = [
        "# Long-Horizon Results — Multi-Year, Multi-Asset Validation",
        "",
        "Goal: prove the Wasserstein-Geodesic forecaster works over a *long* time horizon.",
        "Train: rolling 730-day window. Test: every day after burn-in (no separate holdout).",
        "Scoring: CRPS (lower better, strictly proper).",
        "",
        "## Headline — best WGeo variant vs best GARCH variant",
        "",
        pd.DataFrame(headline_rows)
        .assign(
            wgeo_crps=lambda d: d["wgeo_crps"].map(lambda x: f"{x:.6f}"),
            garch_crps=lambda d: d["garch_crps"].map(lambda x: f"{x:.6f}"),
            dm_stat=lambda d: d["dm_stat"].map(lambda x: f"{x:+.2f}"),
            dm_p=lambda d: d["dm_p"].map(lambda x: f"{x:.4f}"),
        )
        .to_markdown(index=False),
        "",
    ] + md[6:]
    DOC.write_text("\n".join(md))
    print(f"\n[long-run] wrote {DOC}")
    print(json.dumps(headline_rows, indent=2, default=str))
    manifest_path = write_manifest(
        "run_long_horizon.py", extra={"headline": headline_rows}
    )
    print(f"[long-run] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
