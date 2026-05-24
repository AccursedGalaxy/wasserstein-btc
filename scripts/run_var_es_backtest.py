"""Multi-asset VaR / Expected-Shortfall backtest panel.

For each (symbol, horizon, method, tail-level α) cell we re-run the
walk-forward harness with quantile recording on, extract VaR and ES per
step, then evaluate the four tail-calibration tests from
:mod:`wbtc.var_es`:

    Kupiec POF (unconditional coverage)   χ²(1)
    Christoffersen independence            χ²(1)
    Christoffersen conditional coverage    χ²(2)
    Acerbi-Szekely Z1 (conditional ES)     MC p-value
    Acerbi-Szekely Z2 (unconditional ES)   MC p-value

Output: ``docs/RESULTS_VAR_ES.md`` plus a JSON dump per cell under
``results/var_es_<symbol>_h{h}_a{a}.json``.

Why these tests. CRPS measures the *whole* forecast distribution; tail
calibration is what regulators and risk desks care about. The
falsification claim "WGeo's tails are at least as well-calibrated as
GARCH-t's" is sharper than CRPS gestures at and falls naturally out of
the quantile-function representation already used by every forecaster
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wbtc.backtest import load_returns
from wbtc.forecasters import (
    GarchNormal,
    GarchStudentT,
    GJRGarchStudentT,
    HistoricalSimulationBootstrap,
    StaticEmpirical,
    WassersteinGeodesicEWMA,
    WassersteinGeodesicGated,
    WassersteinGeodesicHetero,
    WassersteinGeodesicTheilSen,
    WGeoEnsemble,
)
from wbtc.long_horizon import aligned_quantile_matrices, run_long_horizon
from wbtc.report import slug
from wbtc.var_es import var_es_panel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DOC = ROOT / "docs" / "RESULTS_VAR_ES.md"

# The method panel mirrors run_long_horizon.py but drops methods whose
# left-tail quantile is not the model's primary output (RW-Drift, Adaptive)
# to keep the panel readable. GARCH-t and GJR-GARCH-t are the headline
# baselines because they are the standard fat-tailed parametric VaR/ES
# models in the econometric canon — beating them on these tests is the
# economically meaningful claim.
WGEO_WINDOW = 90
WGEO_LOOKBACK = 20
KAPPA_STAR = 0.6
TAU = 5

METHODS = {
    "Static": StaticEmpirical,
    "HS-Bootstrap": lambda: HistoricalSimulationBootstrap(n_paths=3000, rng_seed=0),
    "GARCH-N": GarchNormal,
    "GARCH-t": GarchStudentT,
    "GJR-GARCH-t": GJRGarchStudentT,
    "WGeo-Gated": lambda: WassersteinGeodesicGated(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, kappa_star=KAPPA_STAR, tau=TAU
    ),
    "WGeo-TheilSen": lambda: WassersteinGeodesicTheilSen(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
    "WGeo-EWMA": lambda: WassersteinGeodesicEWMA(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK, decay=0.85
    ),
    "WGeo-Hetero": lambda: WassersteinGeodesicHetero(
        window=WGEO_WINDOW, lookback=WGEO_LOOKBACK
    ),
    "WGeo-Ensemble": lambda: WGeoEnsemble(),
}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
# Headline focuses on h=1 (non-overlapping, χ² critical values valid).
# h=5 is included for completeness with an explicit overlap caveat in the
# generated report.
HORIZONS = [1, 5]
ALPHAS = [0.01, 0.05]
BURN_IN = 730
K_GRID = 50  # finer grid than CRPS panel — tail quantile interpolation
N_MC = 1000


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_signed(x: float, fmt: str = "{:+.3f}") -> str:
    if not np.isfinite(x):
        return "n/a"
    return fmt.format(x)


def _build_table(rows: list[dict], alpha: float) -> pd.DataFrame:
    """Per-method table for one (symbol, horizon, alpha) cell."""
    df = pd.DataFrame(rows)
    df["α (expected)"] = f"{alpha:.2%}"
    df["empirical"] = df["empirical_rate"].map(lambda x: f"{x:.2%}")
    df["violations"] = (
        df["n_violations"].astype(int).astype(str)
        + "/"
        + df["n"].astype(int).astype(str)
    )
    df["Kupiec p"] = df["kupiec_p"].map(_fmt_p)
    df["Indep p"] = df["indep_p"].map(_fmt_p)
    df["CC p"] = df["cc_p"].map(_fmt_p)
    df["Z1"] = df["z1"].map(_fmt_signed)
    df["Z1 p"] = df["z1_p"].map(_fmt_p)
    df["Z2"] = df["z2"].map(_fmt_signed)
    df["Z2 p"] = df["z2_p"].map(_fmt_p)
    return df[
        [
            "method",
            "α (expected)",
            "empirical",
            "violations",
            "Kupiec p",
            "Indep p",
            "CC p",
            "Z1",
            "Z1 p",
            "Z2",
            "Z2 p",
        ]
    ].set_index("method")


def main():
    md: list[str] = [
        "# VaR / Expected-Shortfall Backtests — Multi-Asset Panel",
        "",
        "Tail-calibration tests on the same multi-year walk-forward harness "
        "used for the CRPS panel in [`RESULTS_LONG.md`](RESULTS_LONG.md). The "
        "five tests reported per cell are the standard regulator + risk-desk "
        "battery; passing them is a sharper claim than CRPS gestures at because "
        "CRPS averages over the whole distribution and tail miscalibration can "
        "be hidden by good body fit.",
        "",
        "**What the tests check.** Let α be the tail level (e.g., 1% or 5%).",
        "",
        "- **Kupiec POF (LR_uc):** does the empirical violation rate match α? χ²(1).",
        "- **Christoffersen Indep (LR_ind):** are violations iid, or do they "
        "  cluster? χ²(1) — 1st-order Markov test.",
        "- **Conditional Coverage (LR_cc):** joint test, LR_uc + LR_ind, χ²(2).",
        "- **Acerbi-Szekely Z1:** average ratio of realised loss to predicted "
        "  ES on exceedance days. Z1 > 0 ⇒ ES underestimated.",
        "- **Acerbi-Szekely Z2:** unconditional ES analogue. Same sign "
        "  convention. MC p-values from the model's own predictive quantile "
        "  function (1000 draws per cell).",
        "",
        "**A method passes a test at level 0.05 when p > 0.05** (we fail to "
        "reject the null of correct calibration).",
        "",
        "**Headline question.** Do WGeo's tails match GARCH-t's tails? CRPS "
        "says they're close; this panel asks the sharper question.",
        "",
    ]

    # Aggregator across (symbol, horizon, alpha) cells: count cells per method
    # where each test fails to reject the null. A higher count is better.
    aggregate_passes: dict[str, dict[str, int]] = {}
    aggregate_total: dict[str, int] = {}

    for symbol in SYMBOLS:
        sym_slug = slug(symbol)
        df = load_returns(DATA / f"{sym_slug}_1d.parquet")
        returns = df["r"].to_numpy()
        timestamps = df["ts"]
        md += [f"## {symbol}", ""]

        for h in HORIZONS:
            print(f"\n[var-es] {symbol} h={h}")
            res = run_long_horizon(
                returns,
                timestamps,
                METHODS,
                burn_in=BURN_IN,
                horizon=h,
                K=K_GRID,
                record_quantiles=True,
            )
            quantile_mats = aligned_quantile_matrices(res)
            u_grid = (np.arange(K_GRID) + 0.5) / K_GRID
            realised = res.realised

            for alpha in ALPHAS:
                print(f"  alpha={alpha}")
                rng = np.random.default_rng(int(1000 * alpha) + h)
                rows: list[dict] = []
                for name, Q in quantile_mats.items():
                    panel = var_es_panel(
                        realised,
                        Q,
                        u_grid,
                        alpha,
                        n_mc=N_MC,
                        rng=rng,
                    )
                    row = {
                        "method": name,
                        "n": panel.n,
                        "n_violations": panel.n_violations,
                        "empirical_rate": panel.empirical_rate,
                        "kupiec_stat": panel.kupiec_stat,
                        "kupiec_p": panel.kupiec_p,
                        "indep_stat": panel.indep_stat,
                        "indep_p": panel.indep_p,
                        "cc_stat": panel.cc_stat,
                        "cc_p": panel.cc_p,
                        "z1": panel.z1,
                        "z1_p": panel.z1_p,
                        "z2": panel.z2,
                        "z2_p": panel.z2_p,
                    }
                    rows.append(row)
                    # tally aggregate passes
                    a = aggregate_passes.setdefault(
                        name, {"Kupiec": 0, "Indep": 0, "CC": 0, "Z1": 0, "Z2": 0}
                    )
                    if panel.kupiec_p > 0.05:
                        a["Kupiec"] += 1
                    if panel.indep_p > 0.05:
                        a["Indep"] += 1
                    if panel.cc_p > 0.05:
                        a["CC"] += 1
                    if panel.z1_p > 0.05:
                        a["Z1"] += 1
                    if panel.z2_p > 0.05:
                        a["Z2"] += 1
                    aggregate_total[name] = aggregate_total.get(name, 0) + 1

                table = _build_table(rows, alpha)
                overlap_note = (
                    " — **caveat:** overlapping h-step forecasts inflate "
                    "LR_ind; treat clustering p-values cautiously."
                    if h > 1
                    else ""
                )
                md += [
                    f"### {symbol}, h = {h}, α = {alpha:.0%}{overlap_note}",
                    "",
                    table.to_markdown(),
                    "",
                ]

                # dump raw JSON for this cell
                payload = {
                    "symbol": symbol,
                    "horizon": h,
                    "alpha": alpha,
                    "n": int(rows[0]["n"]) if rows else 0,
                    "methods": rows,
                }
                out_json = (
                    RESULTS / f"var_es_{sym_slug}_h{h}_a{int(alpha * 100):02d}.json"
                )
                out_json.write_text(json.dumps(payload, indent=2))

    # Aggregate "passes" table
    agg_rows = []
    for name, counts in aggregate_passes.items():
        total = aggregate_total[name]
        agg_rows.append(
            {
                "method": name,
                "cells": total,
                "Kupiec ✓": f"{counts['Kupiec']}/{total}",
                "Indep ✓": f"{counts['Indep']}/{total}",
                "CC ✓": f"{counts['CC']}/{total}",
                "Z1 ✓": f"{counts['Z1']}/{total}",
                "Z2 ✓": f"{counts['Z2']}/{total}",
            }
        )
    agg_md = pd.DataFrame(agg_rows).set_index("method").to_markdown()

    md = (
        md[:25]  # keep the preamble
        + [
            "## Aggregate — cells passing each test at p > 0.05",
            "",
            "A method that calibrates the tail correctly should not reject "
            "in most cells. Counts are over the full panel (symbol × horizon × "
            "α grid). Higher is better; the columns are the five tests above.",
            "",
            agg_md,
            "",
            "---",
            "",
        ]
        + md[25:]
    )

    DOC.write_text("\n".join(md))
    print(f"\n[var-es] wrote {DOC}")


if __name__ == "__main__":
    main()
