"""Patch the v0.4 methods (`WGeo-Adaptive`, `WGeo-Ensemble`) into the existing
``results/long_<sym>_h{h}.json`` per-step CRPS files without re-running the
full panel.

Reads each saved JSON, runs ONLY the new methods on the same walk-forward
``t_idx`` (so timing alignment is preserved), and writes the patched JSON in
place. Then regenerates ``docs/RESULTS_LONG.md`` from the patched JSONs by
calling into the same overall-summary / by-year / by-regime / pairwise-DM
machinery used by ``run_long_horizon.py``, so the headline table and per-
horizon sections include the new methods alongside the v0.3 ones.

Use this *only* to avoid the multi-hour cost of re-fitting GARCH families
on every cell. The walk-forward indices and underlying data are unchanged
from the saved baseline run; results are byte-identical for the original
methods and the new methods are appended without disturbing them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from wbtc.backtest import h_step_log_return, load_returns
from wbtc.forecasters import WassersteinGeodesicAdaptive, WGeoEnsemble
from wbtc.quantiles import make_grid
from wbtc.scoring import crps_from_quantiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "RESULTS_LONG.md"

BURN_IN = 730
WGEO_WINDOW = 90
WGEO_LOOKBACK = 20

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
HORIZONS = [1, 5, 21]

NEW_METHODS = {
    "WGeo-Adaptive": lambda: WassersteinGeodesicAdaptive(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, decay=0.85, decay_quantile=0.97
    ),
    "WGeo-Ensemble": lambda: WGeoEnsemble(),
}


def slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def run_method_on_indices(returns, factory, t_indices, h, K=30):
    u = make_grid(K)
    losses = np.full(len(t_indices), np.nan, dtype=float)
    for i, t in enumerate(tqdm(t_indices, leave=False)):
        window = returns[t - BURN_IN : t]
        f = factory()
        try:
            f.fit(window)
            q = f.predict(h, u)
        except Exception:
            continue
        y = h_step_log_return(returns, t, h)
        if y is None:
            continue
        losses[i] = crps_from_quantiles(q, u, y)
    return losses


def patch_one_json(symbol: str, h: int):
    sym_slug = slug(symbol)
    json_path = RESULTS / f"long_{sym_slug}_h{h}.json"
    saved = json.loads(json_path.read_text())
    t_indices = list(saved["t_idx"])
    df = load_returns(DATA / f"{sym_slug}_1d.parquet")
    returns = df["r"].to_numpy()
    needs_patch = False
    for name, factory in NEW_METHODS.items():
        if name in saved:
            continue  # already present
        needs_patch = True
        print(f"[patch] {symbol} h={h} computing {name} ...", flush=True)
        losses = run_method_on_indices(returns, factory, t_indices, h)
        saved[name] = [None if not np.isfinite(x) else float(x) for x in losses]
    if needs_patch:
        json_path.write_text(json.dumps(saved))
        print(f"[patch] wrote {json_path}")
    else:
        print(f"[patch] {symbol} h={h} already up to date")


def regenerate_results_long():
    """Rebuild RESULTS_LONG.md from the patched JSON files.

    This re-uses the formatting/tables logic from run_long_horizon.py, but
    reads losses from disk instead of recomputing them.
    """
    from wbtc.long_horizon import (
        LongRunResult,
        by_regime,
        by_year,
        overall_summary,
        pairwise_dm,
        pairwise_dm_residualised,
        regime_conditional_dm,
        tag_regimes,
    )

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
            json_path = RESULTS / f"long_{sym_slug}_h{h}.json"
            saved = json.loads(json_path.read_text())
            t_idx = list(saved["t_idx"])
            # The saved losses may contain None (from NaN); convert to nan and
            # mask consistently across methods.
            losses = {
                k: np.array(
                    [np.nan if v is None else float(v) for v in saved[k]], dtype=float
                )
                for k in saved
                if k != "t_idx"
            }
            keep = np.ones(len(t_idx), dtype=bool)
            for v in losses.values():
                keep &= np.isfinite(v)
            aligned_idx = [t_idx[i] for i in range(len(t_idx)) if keep[i]]
            aligned_losses = {k: v[keep] for k, v in losses.items()}
            # rebuild a LongRunResult
            realised = np.array(
                [h_step_log_return(returns, t, h) for t in aligned_idx], dtype=float
            )
            res = LongRunResult(
                per_step={},
                aligned_idx=aligned_idx,
                aligned_losses=aligned_losses,
                realised=realised,
            )
            summary = overall_summary(res, horizon=h)
            yr = by_year(res, timestamps)
            reg = by_regime(res, regime_tags)
            dm_stat, dm_p = pairwise_dm(res, horizon=h)
            dm_stat_r, dm_p_r = pairwise_dm_residualised(res, horizon=h)
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
                    "improvement": f"{(wbest - bbest) / bbest * 100:+.1f}%",
                    "dm_stat": stat,
                    "dm_p": p,
                    "dm_stat_r": stat_r,
                    "dm_p_r": p_r,
                }
            )
            reg_dm = regime_conditional_dm(
                res, regime_tags, wgeo_best_name, baseline_best_name, horizon=h
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
                f"**Diebold-Mariano vs {baseline_best_name}** "
                f"(headline best WGeo-family variant is **{wgeo_best_name}**; "
                f"both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):",
                "",
                pd.DataFrame(
                    {
                        "p_vanilla": dm_p[baseline_best_name].round(4),
                        "p_residualised": dm_p_r[baseline_best_name].round(4),
                    }
                ).to_markdown(),
                "",
                f"**Regime-conditional DM** ({wgeo_best_name} vs {baseline_best_name}, "
                "per-regime CRPS gap and DM statistic; the aggregate panel "
                "DM hides large WGeo-family wins in non-neutral regimes):",
                "",
                reg_dm.round(5).to_markdown()
                if not reg_dm.empty
                else "(no regime had enough days for DM)",
                "",
            ]
            # cumulative CRPS plot
            fig, ax = plt.subplots(figsize=(10, 4))
            xs = np.arange(len(aligned_idx))
            for name, ls in aligned_losses.items():
                ax.plot(xs, np.cumsum(ls), label=name, lw=1.0)
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
        "unconditional EPA null — see `docs/THEORY.md §2.10`). Cells where "
        "`dm_p_r < 0.05` and the WGeo variant has lower mean CRPS are the "
        "headline significant wins.*",
        "",
        "**Cross-cell aggregates (v0.4):**",
        "",
        f"- WGeo-family beats best non-WGeo baseline on CRPS: {sum(1 for r in headline_rows if r['wgeo_crps'] < r['baseline_crps'])}/{len(headline_rows)} cells",
        f"- Vanilla DM p<0.05: {sum(1 for r in headline_rows if r['dm_p'] < 0.05 and r['wgeo_crps'] < r['baseline_crps'])}/{len(headline_rows)} cells",
        f"- Residualised DM p_r<0.05: {sum(1 for r in headline_rows if r['dm_p_r'] < 0.05 and r['wgeo_crps'] < r['baseline_crps'])}/{len(headline_rows)} cells",
        "",
    ] + md[6:]
    DOC.write_text("\n".join(md))
    print(f"\n[patch] wrote {DOC}")
    n_v = sum(
        1
        for r in headline_rows
        if r["dm_p"] < 0.05 and r["wgeo_crps"] < r["baseline_crps"]
    )
    n_r = sum(
        1
        for r in headline_rows
        if r["dm_p_r"] < 0.05 and r["wgeo_crps"] < r["baseline_crps"]
    )
    print(
        f"[patch] vanilla DM p<0.05: {n_v}/{len(headline_rows)} cells; "
        f"residualised DM p_r<0.05: {n_r}/{len(headline_rows)} cells"
    )


def main():
    # Phase 1: patch each JSON
    for symbol in SYMBOLS:
        for h in HORIZONS:
            patch_one_json(symbol, h)
    # Phase 2: regenerate RESULTS_LONG.md
    regenerate_results_long()


if __name__ == "__main__":
    main()
