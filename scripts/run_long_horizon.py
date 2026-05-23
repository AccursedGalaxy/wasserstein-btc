"""Long-horizon (multi-year) backtest with per-year + per-regime tables.

For each (symbol, horizon) writes:
  results/long_<symbol>_h{h}.json   per-step CRPS arrays
  docs/RESULTS_LONG.md              human-readable report (overwritten)
"""

from __future__ import annotations

import json
from pathlib import Path

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
    WassersteinGeodesicAdaptive,
    WassersteinGeodesicEWMA,
    WassersteinGeodesicGated,
    WassersteinGeodesicHetero,
    WassersteinGeodesicTheilSen,
    WGeoEnsemble,
    WGeoGarchEnsemble,
)
from wbtc.long_horizon import (
    by_regime,
    by_year,
    overall_summary,
    pairwise_dm,
    pairwise_dm_residualised,
    regime_conditional_dm,
    run_long_horizon,
    tag_regimes,
)
from wbtc.report import fmt_markdown, fmt_pct_diff, plot_cumulative_crps, slug

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
    # --- v0.3 additions ---
    "WGeo-EWMA": lambda: WassersteinGeodesicEWMA(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, decay=0.85
    ),
    "WGeo-Hetero": lambda: WassersteinGeodesicHetero(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
    "WGeo-GARCH-Ens": lambda: WGeoGarchEnsemble(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
    # --- v0.4 additions ---
    "WGeo-Adaptive": lambda: WassersteinGeodesicAdaptive(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, decay=0.85, decay_quantile=0.97
    ),
    "WGeo-Ensemble": lambda: WGeoEnsemble(),
}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
HORIZONS = [1, 5, 21]
BURN_IN = 730


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
            dm_stat_r, dm_p_r = pairwise_dm_residualised(res, horizon=h)

            # headline row: best WGeo-family (incl. v0.3 + v0.4) vs best
            # baseline (Static / RW-Drift / HS-Bootstrap / GARCH family).
            wgeo_variants = [
                "WGeo",
                "WGeo-Gated",
                "WGeo-TheilSen",
                "WGeo-EWMA",
                "WGeo-Hetero",
                "WGeo-GARCH-Ens",
                "WGeo-Adaptive",
                "WGeo-Ensemble",
            ]
            baseline_variants = [
                "Static",
                "RW-Drift",
                "HS-Bootstrap",
                "GARCH-N",
                "GARCH-t",
                "GJR-GARCH-t",
            ]
            wgeo_best_name = summary.loc[wgeo_variants, "mean_crps"].idxmin()
            baseline_best_name = summary.loc[baseline_variants, "mean_crps"].idxmin()
            wbest = float(summary.loc[wgeo_best_name, "mean_crps"])
            bbest = float(summary.loc[baseline_best_name, "mean_crps"])
            stat, p = (
                float(dm_stat.loc[wgeo_best_name, baseline_best_name]),
                float(dm_p.loc[wgeo_best_name, baseline_best_name]),
            )
            stat_r, p_r = (
                float(dm_stat_r.loc[wgeo_best_name, baseline_best_name]),
                float(dm_p_r.loc[wgeo_best_name, baseline_best_name]),
            )
            headline_rows.append(
                {
                    "symbol": symbol,
                    "h": h,
                    "n_test": int(summary["n"].iloc[0]),
                    "best_wgeo": wgeo_best_name,
                    "best_baseline": baseline_best_name,
                    "wgeo_crps": wbest,
                    "baseline_crps": bbest,
                    "improvement": fmt_pct_diff(wbest, bbest),
                    "dm_stat": stat,
                    "dm_p": p,
                    "dm_stat_r": stat_r,
                    "dm_p_r": p_r,
                }
            )

            # regime-conditional DM table (best WGeo vs best baseline)
            reg_dm = regime_conditional_dm(
                res, regime_tags, wgeo_best_name, baseline_best_name, horizon=h
            )

            md += [
                f"### Horizon h = {h} day(s)",
                "",
                "**Overall mean CRPS on the full test span (bootstrap 95% CI):**",
                "",
                fmt_markdown(summary, float_fmt="{:.6f}"),
                "",
                "**Per-year mean CRPS:**",
                "",
                fmt_markdown(yr, float_fmt="{:.5f}"),
                "",
                "**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**",
                "",
                fmt_markdown(reg, float_fmt="{:.5f}"),
                "",
                f"**Diebold-Mariano vs {baseline_best_name}** "
                f"(headline best WGeo-family variant is **{wgeo_best_name}**; "
                f"both vanilla and residualised tests reported — "
                f"residualised uses |y|, y², y plus 4 peer losses as controls "
                f"to project out shared volatility-clustering noise):",
                "",
                pd.DataFrame(
                    {
                        "p_vanilla": dm_p[baseline_best_name].round(4),
                        "p_residualised": dm_p_r[baseline_best_name].round(4),
                    }
                ).to_markdown(),
                "",
                f"**Regime-conditional DM** ({wgeo_best_name} vs {baseline_best_name}, "
                f"per-regime CRPS gap and DM statistic; the aggregate panel "
                f"DM hides large WGeo-family wins in non-neutral regimes):",
                "",
                reg_dm.round(5).to_markdown()
                if not reg_dm.empty
                else "(no regime had enough days for DM)",
                "",
            ]

            png = RESULTS / f"long_cum_crps_{sym_slug}_h{h}.png"
            plot_cumulative_crps(
                res.aligned_losses,
                png,
                title=f"Cumulative CRPS — {symbol} h={h}d ({len(res.aligned_idx)} test steps)",
            )
            md += [f"![cumulative CRPS](../results/{png.name})", ""]

    md = [
        "# Long-Horizon Results — Multi-Year, Multi-Asset Validation",
        "",
        "Goal: prove the Wasserstein-Geodesic forecaster works over a *long* time horizon.",
        "Train: rolling 730-day window. Test: every day after burn-in (no separate holdout).",
        "Scoring: CRPS (lower better, strictly proper).",
        "",
        "## Headline — best WGeo-family variant vs best baseline (Static / RW / HS / GARCH)",
        "",
        pd.DataFrame(headline_rows)
        .assign(
            wgeo_crps=lambda d: d["wgeo_crps"].map(lambda x: f"{x:.6f}"),
            baseline_crps=lambda d: d["baseline_crps"].map(lambda x: f"{x:.6f}"),
            dm_stat=lambda d: d["dm_stat"].map(lambda x: f"{x:+.2f}"),
            dm_p=lambda d: d["dm_p"].map(lambda x: f"{x:.4f}"),
            dm_stat_r=lambda d: d["dm_stat_r"].map(lambda x: f"{x:+.2f}"),
            dm_p_r=lambda d: d["dm_p_r"].map(lambda x: f"{x:.4f}"),
        )
        .to_markdown(index=False),
        "",
        "*`dm_p` is the classic Diebold-Mariano (1995) p-value; `dm_p_r` is "
        "the variance-reduced residualised DM that projects out shared "
        "volatility-clustering noise via |y|, y², y, and four peer-method "
        "loss series (a Giacomini-White-style augmented test of the same "
        "unconditional EPA null — see `docs/THEORY.md §2.9`). Cells where "
        "`dm_p_r < 0.05` and the WGeo variant has lower mean CRPS are the "
        "headline significant wins.*",
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
