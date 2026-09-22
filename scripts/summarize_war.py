"""Render docs/RESULTS_WAR.md from results/war_sensitivity.json.

The JSON is produced by parsing `scripts/score_new_method.py --verbose`
output for the WAR variants (see docs/THEORY.md §2.11). This script only
formats; it runs no models. Usage: `uv run python scripts/summarize_war.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "war_sensitivity.json"
OUT = ROOT / "docs" / "RESULTS_WAR.md"

ALL_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _table(df: pd.DataFrame, methods: list[str], symbols: list[str]) -> str:
    d = df[df["method"].isin(methods) & df["symbol"].isin(symbols)].copy()
    d = d.sort_values(["method", "symbol", "h"], key=lambda c: c.map(
        {m: i for i, m in enumerate(methods)}) if c.name == "method" else c.map(
        {s: i for i, s in enumerate(symbols)}) if c.name == "symbol" else c)
    out = pd.DataFrame(
        {
            "method": d["method"],
            "symbol": d["symbol"],
            "h": d["h"],
            "CRPS": d["mean_crps"].map(lambda x: f"{x:.6f}"),
            "best classical": d["best_non_wgeo"] + " (" + d["best_crps"].map(lambda x: f"{x:.6f}") + ")",
            "Δ vs best classical": d["delta_vs_best"],
            "dm_p_r vs best": d["p_r"].map(lambda x: f"{x:.4f}"),
            "WGeo-Ensemble − WAR": d["vs_ens_dmean"].map(lambda x: "" if pd.isna(x) else f"{-x:+.6f}"),
            "dm_p (vanilla)": d["vs_ens_p"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}"),
        }
    )
    return out.to_markdown(index=False)


def main() -> int:
    data = json.loads(SRC.read_text())
    df = pd.DataFrame(data["rows"])
    md = [
        "# Wasserstein Autoregression benchmark — scoring and sensitivity",
        "",
        f"_Generated {data['generated']} by `scripts/summarize_war.py` from "
        f"`results/war_sensitivity.json` ({data['source']})._",
        "",
        "The Wasserstein autoregressive model of Zhang, Kokoszka & Petersen (2022) "
        "applied to the same rolling-90-day quantile vectors the WGeo family uses "
        "(\"WAR on rolling-ECDF densities\", [`THEORY.md §2.11`](THEORY.md)). "
        "Every row is a walk-forward run at exactly the test indices of the "
        "long-horizon panel, scored with CRPS against the saved per-step losses "
        "of every panel method. `dm_p_r vs best` is the residualised "
        "Diebold-Mariano p-value of the WAR variant against the best classical "
        "baseline in that cell (Static / HS-Bootstrap / GARCH-N / GARCH-t / "
        "GJR-GARCH-t); the last two columns compare `WGeo-Ensemble` with the WAR "
        "variant (negative Δ = WGeo-Ensemble better; vanilla DM p). The "
        "regenerated `RESULTS_LONG.md` Headline 3 carries the residualised "
        "statistic for the three panel variants and the §4 falsification count.",
        "",
        "## Reading",
        "",
        "- **WAR ≈ Static at h=1, worse at h=21 under the `sum` rule.** With "
        "lag-1 Wasserstein autocorrelation ≈0.98 (mostly mechanical: consecutive "
        "densities share 89/90 observations), WAR shrinks today's density toward "
        "the 730-day barycentre. At h=1 that is nearly the identity; over 21 "
        "iterated steps the shrinkage lags the volatility path.",
        "- **The h-day location rule matters more than the AR order.** `WAR-1-last` "
        "(WGeo's rule: terminal daily median, √h-scaled shape) beats GARCH-N at "
        "h=21 by 2.4–3.5% with residualised p<0.05 on BTC and ETH, close to "
        "`WGeo-Ensemble`'s own margins; `sum` (h-fold daily median) and `conv` "
        "(independent convolution of the h daily laws) are 1–4% worse than GARCH-N. "
        "Multiplying a noisy 90-day daily median by 21 injects location noise that "
        "the √h shortcut avoids. This is a statement about the panel's h-day "
        "conversion, not about WAR.",
        "- **Order selection changes nothing; the window grid does.** `WAR-Select` "
        "picks p=1 and the widest admissible K almost everywhere and matches `WAR-1` "
        "to the fourth decimal. `WAR-Paper`, confined to the paper's intraday grid "
        "K ∈ {20, 62}, is worse at h=21 (a 62-density barycentre is noisier). The "
        "paper's unpenalised in-sample criterion cannot separate orders on this data.",
        "- **WAR's tails are not calibrated** (`RESULTS_VAR_ES.md`): `WAR-1` passes "
        "Kupiec in 6/20 cells and `WAR-1-last` in 8/20, versus 17/20 for Static and "
        "`WGeo-Gated`; both fail every Acerbi-Szekely Z1 cell. Shrinking toward a "
        "730-day barycentre gives a body that scores well under CRPS and tails "
        "that under-react, the same defect as `WGeo-TheilSen` / `WGeo-EWMA`.",
        "- **Less overlap helps slightly.** `stride=10` (every 10th density) lowers "
        "the mechanical persistence and is marginally better than `stride=1`; a "
        "30-day density window is worse everywhere.",
        "",
        "- **Headline 3 verdict (`RESULTS_LONG.md`): the §4 test fails, 6/15.** "
        "`WGeo-Ensemble` beats `WAR-1-last` in all five h=1 cells, ties at h=5, and "
        "loses four of five h=21 cells with residualised p ≤ 0.047. On rolling-ECDF "
        "densities, mean reversion toward the barycentre beats extrapolation at 21 "
        "days.",
        "",
        "## Panel variants, 5 assets × 3 horizons",
        "",
        _table(df, ["WAR-1", "WAR-Select", "WAR-Paper"], ALL_SYMBOLS),
        "",
        "## Sensitivity: location rule, density window, stride (BTC + ETH)",
        "",
        _table(df, ["WAR-1", "WAR-1-last", "WAR-1-conv", "WAR-1-w30", "WAR-1-stride10"], ["BTC/USDT", "ETH/USDT"]),
        "",
        "`WAR-1`: p=1, all 641 densities, `location=\"sum\"`. `WAR-1-last`: "
        "`location=\"last\"`. `WAR-1-conv`: `location=\"conv\"` (3 000 seeded paths). "
        "`WAR-1-w30`: 30-day density window. `WAR-1-stride10`: every 10th density. "
        "`WAR-Paper`: K ∈ {20, 62}, p ∈ {1..10}; `WAR-Select`: K ∈ {20, 62, 250, all}, "
        "p ∈ {1..5}; both by the paper's sequential in-sample one-step W₂ criterion "
        "over 60 origins.",
        "",
        "Reproduce: `uv run python scripts/score_new_method.py --method WAR-1 --verbose` "
        "(and the other names in `_BUILTIN_FACTORIES`), then rebuild the JSON and this file.",
        "",
    ]
    OUT.write_text("\n".join(md))
    print(f"[war] wrote {OUT.relative_to(ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
