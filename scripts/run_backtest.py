"""Run the full walk-forward backtest across all forecasters & horizons,
write results to results/ and docs/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wbtc.backtest import BacktestConfig, compare_methods, load_returns
from wbtc.forecasters import (
    GarchNormal,
    GarchStudentT,
    RandomWalkDrift,
    StaticEmpirical,
    WassersteinGeodesic,
    WassersteinGeodesicGated,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "btcusdt_1d.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DOC = ROOT / "docs" / "RESULTS.md"

# in-sample-tuned hyperparameters (chosen on first 730 days; see THEORY §3)
WGEO_WINDOW = 90
WGEO_LOOKBACK = 20
KAPPA_STAR = 0.6
TAU = 5

METHODS = {
    "Static-Empirical": StaticEmpirical,
    "RW-Drift": RandomWalkDrift,
    "GARCH-N": GarchNormal,
    "GARCH-t": GarchStudentT,
    "WGeo": lambda: WassersteinGeodesic(window=WGEO_WINDOW, lookback=WGEO_LOOKBACK),
    "WGeo-Gated": lambda: WassersteinGeodesicGated(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, kappa_star=KAPPA_STAR, tau=TAU
    ),
}


def fmt_pct(x):
    return f"{x * 100:+.4f}"


def main():
    df = load_returns(DATA)
    print(
        f"[run] loaded {len(df)} daily returns "
        f"from {df['ts'].min().date()} to {df['ts'].max().date()}"
    )
    returns = df["r"].to_numpy()

    horizons = [1, 5, 21]
    all_results = {}
    md_lines = [
        "# Results — Wasserstein-Geodesic BTC Forecasting",
        "",
        f"Backtest run on {len(df)} daily BTC/USDT log-returns "
        f"({df['ts'].min().date()} → {df['ts'].max().date()}).",
        "Walk-forward, train window = 730 days, strict 365-day test set at end.",
        "All numbers reported on the test set unless stated.",
        "Scoring rule: **CRPS** (lower is better, strictly proper).",
        "",
    ]

    for h in horizons:
        print(f"\n[run] === horizon = {h} day(s) ===")
        cfg = BacktestConfig(train_window=730, horizon=h, K=50, test_holdout=365)
        out = compare_methods(returns, METHODS, cfg)
        all_results[h] = out

        # save raw aligned losses for posterity
        np.savez(
            RESULTS / f"losses_h{h}.npz",
            **{k: v for k, v in out["aligned"].items()},
            test_mask=out["test_mask"],
        )

        summary = out["summary"].copy()
        summary["mean_crps"] = summary["mean_crps"].map(lambda x: f"{x:.6f}")
        summary["ci_lo"] = summary["ci_lo"].map(lambda x: f"{x:.6f}")
        summary["ci_hi"] = summary["ci_hi"].map(lambda x: f"{x:.6f}")
        md_lines += [
            f"## Horizon h = {h} day(s)",
            "",
            "### Mean CRPS on test set (with stationary-bootstrap 95% CI)",
            "",
            summary.to_markdown(),
            "",
        ]

        dm_p = out["dm_p"].copy().astype(float)
        dm_stat = out["dm_stat"].copy().astype(float)
        md_lines += [
            "### Diebold-Mariano statistic (row vs column, negative = row better)",
            "",
            dm_stat.round(3).to_markdown(),
            "",
            "### Diebold-Mariano p-value",
            "",
            dm_p.round(4).to_markdown(),
            "",
        ]

        # plot per-method cumulative CRPS
        fig, ax = plt.subplots(figsize=(9, 4))
        ts = np.where(out["test_mask"])[0]
        for name, losses in out["aligned"].items():
            ax.plot(ts, np.cumsum(losses[out["test_mask"]]), label=name, lw=1.2)
        ax.set_title(f"Cumulative CRPS on test set, h={h}d")
        ax.set_xlabel("test-set step")
        ax.set_ylabel("cumulative CRPS")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = RESULTS / f"cum_crps_h{h}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        md_lines += [f"![cumulative CRPS h={h}d](../results/cum_crps_h{h}.png)", ""]

    # judgement section
    md_lines += [
        "## Verdict",
        "",
        "_(populated automatically — see falsification criteria in `THEORY.md` §4.)_",
        "",
    ]
    verdicts = []
    for h in horizons:
        s = all_results[h]["summary"]
        dm_p = all_results[h]["dm_p"]
        dm_stat = all_results[h]["dm_stat"]
        wgeo_crps = s.loc["WGeo-Gated", "mean_crps"]
        static_crps = s.loc["Static-Empirical", "mean_crps"]
        beat_static = wgeo_crps < static_crps
        garch_p = float(dm_p.loc["WGeo-Gated", "GARCH-N"])
        garch_dm = float(dm_stat.loc["WGeo-Gated", "GARCH-N"])
        beats_garch_significantly = (garch_dm < 0) and (garch_p < 0.10)
        gating_helps = s.loc["WGeo-Gated", "mean_crps"] <= s.loc["WGeo", "mean_crps"]
        verdicts.append(
            {
                "h": h,
                "beats_static": bool(beat_static),
                "beats_garch_DM10": bool(beats_garch_significantly),
                "gating_helps": bool(gating_helps),
                "wgeo_gated_crps": float(wgeo_crps),
                "garch_n_crps": float(s.loc["GARCH-N", "mean_crps"]),
                "static_crps": float(static_crps),
            }
        )
    md_lines += ["```json", json.dumps(verdicts, indent=2), "```", ""]
    md_lines += [
        "_(The interpretive verdict + coverage table in `docs/RESULTS.md` is "
        "hand-written and must be re-edited after each run.)_"
    ]

    # write the raw, auto-generated report to a separate file so the
    # interpretive RESULTS.md is not clobbered.
    auto_path = DOC.parent / "RESULTS_AUTO.md"
    auto_path.write_text("\n".join(md_lines))
    print(f"\n[run] wrote {auto_path} (auto)")
    print(f"[run] verdicts: {json.dumps(verdicts, indent=2)}")


if __name__ == "__main__":
    main()
