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
    garch_fallback_rate,
    headline_dm_sensitivity,
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

    # Pre-registered headline forecaster (see v0.5 PREREGISTRATION). All
    # headline DM tests are WGeo-Ensemble vs a fixed reference baseline; the
    # "best-of-family vs best-of-baseline" rows are demoted to a robustness
    # appendix because the implicit max-over-comparators inflates type-I error.
    PREREG_HEADLINE = "WGeo-Ensemble"
    PREREG_BASELINES = ("Static", "GARCH-N")

    headline_ensemble_rows: list[dict] = []  # WGeo-Ensemble vs Static + GARCH-N
    # Per-cell residualised-DM p-values under {none, vol, full} controls. The
    # cross-cell aggregate of this table is the falsification floor used by
    # PREREGISTRATION.md — pre-registered to the "vol" column so the bar does
    # not depend on peer-loss correlations.
    sensitivity_rows: list[dict] = []
    robustness_best_rows: list[dict] = []  # legacy best-of-family vs best-of-baseline

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
            fallback_rates = garch_fallback_rate(res)

            # ---- pre-registered headline: WGeo-Ensemble vs Static + GARCH-N
            wgeo_crps = float(summary.loc[PREREG_HEADLINE, "mean_crps"])
            sensitivity = headline_dm_sensitivity(
                res, horizon=h, head=PREREG_HEADLINE, baselines=list(PREREG_BASELINES)
            )
            for ref in PREREG_BASELINES:
                ref_crps = float(summary.loc[ref, "mean_crps"])
                row = {
                    "symbol": symbol,
                    "h": h,
                    "n_test": int(summary["n"].iloc[0]),
                    "baseline": ref,
                    "ensemble_crps": wgeo_crps,
                    "baseline_crps": ref_crps,
                    "improvement": fmt_pct_diff(wgeo_crps, ref_crps),
                    "dm_stat": float(dm_stat.loc[PREREG_HEADLINE, ref]),
                    "dm_p": float(dm_p.loc[PREREG_HEADLINE, ref]),
                    "dm_stat_r": float(dm_stat_r.loc[PREREG_HEADLINE, ref]),
                    "dm_p_r": float(dm_p_r.loc[PREREG_HEADLINE, ref]),
                }
                headline_ensemble_rows.append(row)
                sensitivity_rows.append(
                    {
                        "symbol": symbol,
                        "h": h,
                        "baseline": ref,
                        "dm_p_none": float(sensitivity.loc[ref, "dm_p_none"]),
                        "dm_p_vol": float(sensitivity.loc[ref, "dm_p_vol"]),
                        "dm_p_full": float(sensitivity.loc[ref, "dm_p_full"]),
                        "dm_stat_none": float(sensitivity.loc[ref, "dm_stat_none"]),
                        "dm_stat_vol": float(sensitivity.loc[ref, "dm_stat_vol"]),
                        "dm_stat_full": float(sensitivity.loc[ref, "dm_stat_full"]),
                    }
                )

            # ---- robustness: best WGeo-family vs best non-WGeo baseline (legacy)
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
            robustness_best_rows.append(
                {
                    "symbol": symbol,
                    "h": h,
                    "n_test": int(summary["n"].iloc[0]),
                    "best_wgeo": wgeo_best_name,
                    "best_baseline": baseline_best_name,
                    "wgeo_crps": wbest,
                    "baseline_crps": bbest,
                    "improvement": fmt_pct_diff(wbest, bbest),
                    "dm_stat": float(dm_stat.loc[wgeo_best_name, baseline_best_name]),
                    "dm_p": float(dm_p.loc[wgeo_best_name, baseline_best_name]),
                    "dm_stat_r": float(
                        dm_stat_r.loc[wgeo_best_name, baseline_best_name]
                    ),
                    "dm_p_r": float(dm_p_r.loc[wgeo_best_name, baseline_best_name]),
                    # GARCH-fallback rate of WGeo-Hetero in this cell. The §4
                    # falsification floor for Hetero is conditional on a
                    # small rate.
                    "hetero_garch_fallback": fallback_rates.get(
                        "WGeo-Hetero", float("nan")
                    ),
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
                "**GARCH-fallback rate** (fraction of walk-forward steps where "
                "the GARCH fit raised or produced degenerate variances, "
                "forcing the WGeo-Hetero / CondShape variants back onto the "
                "unconditional √h scaling). The §4 falsification floor for "
                "Hetero is conditional on this rate being small — otherwise "
                "the headline is measuring √h scaling, not the GARCH "
                "contribution:",
                "",
                pd.DataFrame(
                    [
                        {"method": name, "fallback_rate": rate}
                        for name, rate in sorted(fallback_rates.items())
                    ]
                ).to_markdown(index=False)
                if fallback_rates
                else "(no GARCH-conditioned methods in this panel)",
                "",
            ]

            png = RESULTS / f"long_cum_crps_{sym_slug}_h{h}.png"
            plot_cumulative_crps(
                res.aligned_losses,
                png,
                title=f"Cumulative CRPS — {symbol} h={h}d ({len(res.aligned_idx)} test steps)",
            )
            md += [f"![cumulative CRPS](../results/{png.name})", ""]

    def _fmt_ensemble_table(rows: list[dict]) -> str:
        return (
            pd.DataFrame(rows)
            .assign(
                ensemble_crps=lambda d: d["ensemble_crps"].map(lambda x: f"{x:.6f}"),
                baseline_crps=lambda d: d["baseline_crps"].map(lambda x: f"{x:.6f}"),
                dm_stat=lambda d: d["dm_stat"].map(lambda x: f"{x:+.2f}"),
                dm_p=lambda d: d["dm_p"].map(lambda x: f"{x:.4f}"),
                dm_stat_r=lambda d: d["dm_stat_r"].map(lambda x: f"{x:+.2f}"),
                dm_p_r=lambda d: d["dm_p_r"].map(lambda x: f"{x:.4f}"),
            )
            .to_markdown(index=False)
        )

    rows_vs_static = [r for r in headline_ensemble_rows if r["baseline"] == "Static"]
    rows_vs_garchn = [r for r in headline_ensemble_rows if r["baseline"] == "GARCH-N"]

    # Aggregate sensitivity table: per pre-registered baseline, count cells
    # where WGeo-Ensemble has both lower CRPS and DM p-value < 0.05 under each
    # of the three control sets. The pre-registered falsification threshold
    # is set against the "vol" column (PREREGISTRATION.md), so peer losses
    # are reported as a power-only bonus.
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_summary_rows = []
    for ref in PREREG_BASELINES:
        sub = sens_df[sens_df["baseline"] == ref]
        n_cells = len(sub)
        sens_summary_rows.append(
            {
                "baseline": ref,
                "cells": f"{n_cells}",
                "no_controls": f"{int(((sub['dm_p_none'] < 0.05) & (sub['dm_stat_none'] < 0)).sum())}/{n_cells}",
                "vol_only": f"{int(((sub['dm_p_vol'] < 0.05) & (sub['dm_stat_vol'] < 0)).sum())}/{n_cells}",
                "vol_plus_peers": f"{int(((sub['dm_p_full'] < 0.05) & (sub['dm_stat_full'] < 0)).sum())}/{n_cells}",
            }
        )
    sens_summary_md = pd.DataFrame(sens_summary_rows).to_markdown(index=False)

    # Per-cell sensitivity p-values (one row per cell × baseline).
    sens_detail_md = (
        sens_df.assign(
            dm_p_none=lambda d: d["dm_p_none"].map(lambda x: f"{x:.4f}"),
            dm_p_vol=lambda d: d["dm_p_vol"].map(lambda x: f"{x:.4f}"),
            dm_p_full=lambda d: d["dm_p_full"].map(lambda x: f"{x:.4f}"),
        )
        .drop(columns=["dm_stat_none", "dm_stat_vol", "dm_stat_full"])
        .to_markdown(index=False)
    )

    md = [
        "# Long-Horizon Results — Multi-Year, Multi-Asset Validation",
        "",
        "Goal: prove the Wasserstein-Geodesic forecaster works over a *long* time horizon.",
        "Train: rolling 730-day window. Test: every day after burn-in (no separate holdout).",
        "Scoring: CRPS (lower better, strictly proper).",
        "",
        "## Pre-registration",
        "",
        f"**Pre-registered headline forecaster:** `{PREREG_HEADLINE}` (the equal-weight "
        "W₂ barycentre of `WGeo-TheilSen`, `WGeo-EWMA`, `WGeo-Gated` — see "
        "`THEORY.md §2.9`). All headline DM tests below are `WGeo-Ensemble` "
        "against a fixed reference baseline. The previous reporting style — "
        "*best-of-family vs best-of-baseline* — is retained as a robustness "
        "appendix because the implicit max-over-comparators inflates type-I "
        "error and is not a valid pre-committed test.",
        "",
        f"**Pre-registered reference baselines:** {', '.join(f'`{b}`' for b in PREREG_BASELINES)}. "
        "`Static` is the most naive distributional baseline (the current "
        "empirical quantile, √h-scaled); `GARCH-N` is the standard parametric "
        "vol baseline from the econometrics canon. A win against both is the "
        "minimum bar for the v0.5 claim.",
        "",
        f"## Headline 1 — {PREREG_HEADLINE} vs Static (pre-registered)",
        "",
        _fmt_ensemble_table(rows_vs_static),
        "",
        f"## Headline 2 — {PREREG_HEADLINE} vs GARCH-N (pre-registered)",
        "",
        _fmt_ensemble_table(rows_vs_garchn),
        "",
        "*`dm_p` is the classic Diebold-Mariano (1995) p-value; `dm_p_r` is "
        "the variance-reduced residualised DM with the `full` control set "
        "(vol moments + four peer-method loss series) — a Giacomini-White-"
        "style augmented test of the same unconditional EPA null. See the "
        "sensitivity table below for the breakdown by control set, and "
        "`docs/THEORY.md §2.10` for the math.*",
        "",
        "## Residualised-DM sensitivity to control set",
        "",
        "The residualised DM test admits any covariate predictable at time t. "
        "Three control sets are reported, ordered from least to most powerful: "
        "`none` (= vanilla DM), `vol` (`[y, |y|, y²]` — sign, magnitude, and "
        "kurtosis-like moment of the realised return), and `full` (`vol` plus "
        "up to four peer-method loss series). Peer-loss controls are admissible "
        "under Giacomini-White but rhetorically more endogenous; the table "
        "below decomposes the residualised lift so the reader can see how "
        "much is driven by vol controls alone vs. peer losses.",
        "",
        "**Aggregate — cells with `dm_p < 0.05` and `WGeo-Ensemble` lower CRPS:**",
        "",
        sens_summary_md,
        "",
        "The pre-registered falsification threshold in `PREREGISTRATION.md` is "
        "anchored to the `vol_only` column so the v0.5 bar does not depend on "
        "peer-loss correlations — peer losses are reported as a power-only "
        "bonus, not a contributor to the headline claim.",
        "",
        "**Per-cell residualised-DM p-values:**",
        "",
        sens_detail_md,
        "",
        "## Robustness — best WGeo-family vs best non-WGeo baseline (legacy)",
        "",
        "Retained for continuity with v0.3 / v0.4 reporting. Both sides are "
        "selected by minimum cell CRPS, so the implicit multiple comparison "
        "(8 WGeo variants × 6 baselines = 48 implicit pairs per cell) inflates "
        "type-I error. Use Headlines 1–2 above for inference; this table is a "
        "robustness check that the pre-registered headline does not depend on "
        "the choice of WGeo variant.",
        "",
        pd.DataFrame(robustness_best_rows)
        .assign(
            wgeo_crps=lambda d: d["wgeo_crps"].map(lambda x: f"{x:.6f}"),
            baseline_crps=lambda d: d["baseline_crps"].map(lambda x: f"{x:.6f}"),
            dm_stat=lambda d: d["dm_stat"].map(lambda x: f"{x:+.2f}"),
            dm_p=lambda d: d["dm_p"].map(lambda x: f"{x:.4f}"),
            dm_stat_r=lambda d: d["dm_stat_r"].map(lambda x: f"{x:+.2f}"),
            dm_p_r=lambda d: d["dm_p_r"].map(lambda x: f"{x:.4f}"),
            hetero_garch_fallback=lambda d: d["hetero_garch_fallback"].map(
                lambda x: "n/a" if (x != x) else f"{x:.1%}"
            ),
        )
        .to_markdown(index=False),
        "",
    ] + md[6:]
    DOC.write_text("\n".join(md))
    print(f"\n[long-run] wrote {DOC}")
    print(json.dumps(headline_ensemble_rows, indent=2, default=str))
    manifest_path = write_manifest(
        "run_long_horizon.py",
        extra={
            "headline_ensemble": headline_ensemble_rows,
            "robustness_best": robustness_best_rows,
        },
    )
    print(f"[long-run] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
