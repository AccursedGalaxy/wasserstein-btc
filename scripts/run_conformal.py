"""Evaluate the split-conformal calibration layer on the long-horizon panel.

For every (asset, horizon) cell this re-runs the walk-forward harness with
quantile recording for two base forecasters — the horizon-specific default
(``wbtc.default_forecaster``) and GARCH-N as a parametric control — then
applies :func:`wbtc.conformal.conformalize_path` with the harness lag
``h + 1`` (no look-ahead) for several calibration-window lengths, and
reports

  * mean CRPS of base vs calibrated on the rows where the layer is active,
    with the lag-(h-1) HAC Diebold-Mariano test,
  * empirical coverage at nominal levels 1/5/25/50/75/95/99 % with Kupiec
    p-values on non-overlapping (every h-th) origins,
  * the same split by epoch: early (origins before 2023-01-01, used to pick
    the calibration window) and late (2023-01-01 → panel end).

Outputs ``results/conformal_<asset>_h<h>.json`` and
``docs/RESULTS_CONFORMAL.md``. Runs in a few minutes on 8+ cores.

Usage:
    uv run wbtc conformal
    uv run python scripts/run_conformal.py --symbols BTC/USDT --horizons 1 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wbtc import default_forecaster
from wbtc.backtest import PANEL_DATA_END, _walk_forward_one, load_returns
from wbtc.conformal import conformalize_path, empirical_coverage
from wbtc.forecasters import GarchNormal
from wbtc.quantiles import make_grid
from wbtc.scoring import crps_from_quantiles, diebold_mariano
from wbtc.var_es import kupiec_pof_test

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "RESULTS_CONFORMAL.md"

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
HORIZONS = [1, 5, 21]
BURN_IN = 730
K = 50
CALIB_WINDOWS = [125, 250, 500]
MIN_CALIB = 50
LEVELS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
EPOCH_SPLIT = pd.Timestamp("2023-01-01", tz="UTC")

BASES = {
    "default": default_forecaster,  # takes h
    "GARCH-N": lambda h: GarchNormal(),
}


def slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


# ----------------------------- one cell ------------------------------------


def _run_base(returns: np.ndarray, h: int, base_name: str) -> pd.DataFrame:
    u = make_grid(K)
    factory = lambda: BASES[base_name](h)  # noqa: E731
    return _walk_forward_one(
        returns,
        factory,
        train_window=BURN_IN,
        horizon=h,
        u=u,
        stride=1,
        show_progress=False,
        label=f"{base_name} h={h}",
        record_quantiles=True,
    )


def _crps_path(Q: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([crps_from_quantiles(Q[t], u, y[t]) for t in range(len(y))])


def _coverage_block(Q: np.ndarray, u: np.ndarray, y: np.ndarray, h: int) -> dict:
    """Coverage at LEVELS on non-overlapping origins + Kupiec p-values."""
    sub = slice(None, None, h)
    Qs, ys = Q[sub], y[sub]
    cov = empirical_coverage(Qs, u, ys, levels=np.array(LEVELS))
    out = {}
    for a in LEVELS:
        qa = np.array([np.interp(a, u, Qs[t]) for t in range(len(ys))])
        hits = (ys <= qa).astype(int)
        res = kupiec_pof_test(hits, a)
        out[str(a)] = {"empirical": cov[a], "kupiec_p": res.p_value, "n": int(len(ys))}
    return out


def _mad_coverage(block: dict) -> float:
    """Mean |empirical - nominal| over LEVELS — the headline calibration gap."""
    return float(np.mean([abs(v["empirical"] - float(a)) for a, v in block.items()]))


def evaluate_cell(symbol: str, h: int) -> dict:
    df = load_returns(DATA / f"{slug(symbol)}_1d.parquet", end=PANEL_DATA_END)
    returns = df["r"].to_numpy()
    ts = pd.to_datetime(df["ts"], utc=True)
    u = make_grid(K)
    cell: dict = {"symbol": symbol, "h": h, "K": K, "lag": h + 1, "bases": {}}
    for base_name in BASES:
        per = _run_base(returns, h, base_name)
        per = per.dropna(subset=["crps"]).reset_index(drop=True)
        Q = np.stack(per["q"].to_list())
        y = per["y"].to_numpy(dtype=float)
        t_idx = per["t"].to_numpy(dtype=int)
        origin_ts = ts.iloc[t_idx].reset_index(drop=True)
        early = (origin_ts < EPOCH_SPLIT).to_numpy()

        paths = {
            w: conformalize_path(
                Q, y, u, lag=h + 1, calib_window=w, min_calib=MIN_CALIB
            )
            for w in CALIB_WINDOWS
        }
        # like-for-like: score only rows where every window is active
        active = np.logical_and.reduce([a for _, a in paths.values()])
        crps_base = _crps_path(Q, u, y)
        entry: dict = {
            "n_active": int(active.sum()),
            "n_early": int((active & early).sum()),
            "n_late": int((active & ~early).sum()),
            "first_active_origin": str(origin_ts[active].iloc[0].date())
            if active.any()
            else None,
            "base": {},
            "windows": {},
        }
        for name, mask in (
            ("all", active),
            ("early", active & early),
            ("late", active & ~early),
        ):
            entry["base"][name] = {
                "crps": float(crps_base[mask].mean()) if mask.any() else None,
                "coverage": _coverage_block(Q[mask], u, y[mask], h)
                if mask.sum() > 2 * h
                else {},
            }
            cov_b = entry["base"][name]["coverage"]
            entry["base"][name]["coverage_mad"] = _mad_coverage(cov_b) if cov_b else None
        for w, (Qc, _) in paths.items():
            crps_cal = _crps_path(Qc, u, y)
            wrec: dict = {}
            for name, mask in (
                ("all", active),
                ("early", active & early),
                ("late", active & ~early),
            ):
                if not mask.any():
                    wrec[name] = None
                    continue
                stat, p = diebold_mariano(crps_cal[mask], crps_base[mask], h=h)
                cov = (
                    _coverage_block(Qc[mask], u, y[mask], h)
                    if mask.sum() > 2 * h
                    else {}
                )
                wrec[name] = {
                    "crps": float(crps_cal[mask].mean()),
                    "crps_base": float(crps_base[mask].mean()),
                    "dm_stat": float(stat),
                    "dm_p": float(p),
                    "coverage": cov,
                    "coverage_mad": _mad_coverage(cov) if cov else None,
                }
            entry["windows"][str(w)] = wrec
        cell["bases"][base_name] = entry
    return cell


# ----------------------------- report --------------------------------------


def _pct(a: float, b: float) -> str:
    return f"{(a - b) / b * 100:+.1f}%"


def _p_mark(p: float) -> str:
    return "**" if p < 0.05 else ""


def render_report(cells: list[dict], chosen_window: int, selection: dict) -> str:
    L: list[str] = []
    L.append("# Split-conformal calibration layer — panel results\n")
    L.append(
        f"Generated by `scripts/run_conformal.py` on {pd.Timestamp.now('UTC').date()} "
        f"(panel end `{PANEL_DATA_END}`, burn-in {BURN_IN}, K={K}, "
        f"lag = h+1, min_calib = {MIN_CALIB}).\n"
    )
    L.append(
        "The layer (`wbtc.conformal`, THEORY.md §2.12) adds per-level split-conformal "
        "offsets, estimated on a rolling window of past forecast errors, to the base "
        "quantile forecast and re-projects onto monotone quantile functions. "
        "It is evaluated *only* on origins where every candidate window was active, "
        "so base and calibrated numbers are like-for-like. Coverage is measured on "
        "non-overlapping (every h-th) origins; Kupiec p-values are the "
        "unconditional-coverage test at each nominal level.\n"
    )
    L.append("## Calibration-window selection (early epoch, origins < 2023-01-01)\n")
    L.append(
        "Mean CRPS change vs base, averaged over the 15 cells, default forecaster:\n"
    )
    L.append(
        "| window | mean ΔCRPS | mean coverage MAD (base → cal) | cells improved |"
    )
    L.append("|---:|---:|---:|---:|")
    for w in CALIB_WINDOWS:
        s = selection[str(w)]
        L.append(
            f"| {w} | {s['mean_dcrps_pct']:+.2f}% | {s['mad_base']:.3f} → {s['mad_cal']:.3f} | "
            f"{s['n_improved']}/{s['n_cells']} |"
        )
    L.append(
        f"\n**Chosen: window = {chosen_window}** (lowest early-epoch mean CRPS). "
        "Everything below is reported for that window on the *late* epoch "
        "(2023-01-01 → panel end) and on the full active sample.\n"
    )

    L.append("## Summary by horizon (late epoch, chosen window)\n")
    L.append(
        "| base | h | cells | mean ΔCRPS | CRPS worse (p<0.05) | CRPS better (p<0.05) | "
        "mean coverage MAD base → cal | MAD improved |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for base_name in BASES:
        for h in sorted({c["h"] for c in cells}):
            rows = [
                (
                    c["bases"][base_name]["windows"][str(chosen_window)]["late"],
                    c["bases"][base_name]["base"]["late"],
                )
                for c in cells
                if c["h"] == h
            ]
            rows = [
                (r, b) for r, b in rows if r is not None and b.get("coverage_mad") is not None
            ]
            if not rows:
                continue
            d = [(r["crps"] - r["crps_base"]) / r["crps_base"] * 100 for r, _ in rows]
            worse = sum(1 for r, _ in rows if r["dm_p"] < 0.05 and r["crps"] > r["crps_base"])
            better = sum(1 for r, _ in rows if r["dm_p"] < 0.05 and r["crps"] < r["crps_base"])
            mb = float(np.mean([b["coverage_mad"] for _, b in rows]))
            mc = float(np.mean([r["coverage_mad"] for r, _ in rows]))
            imp = sum(1 for r, b in rows if r["coverage_mad"] < b["coverage_mad"])
            L.append(
                f"| {base_name} | {h} | {len(rows)} | {np.mean(d):+.1f}% | {worse} | {better} | "
                f"{mb:.3f} → {mc:.3f} | {imp}/{len(rows)} |"
            )
    L.append("")
    for base_name in BASES:
        L.append(f"## Base = `{base_name}`\n")
        L.append("### CRPS (late epoch)\n")
        L.append("| asset | h | n | base | conformal | Δ | DM p |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for c in cells:
            e = c["bases"][base_name]
            r = e["windows"][str(chosen_window)]["late"]
            if r is None:
                continue
            L.append(
                f"| {c['symbol']} | {c['h']} | {e['n_late']} | {r['crps_base']:.5f} | "
                f"{r['crps']:.5f} | {_pct(r['crps'], r['crps_base'])} | "
                f"{_p_mark(r['dm_p'])}{r['dm_p']:.3f}{_p_mark(r['dm_p'])} |"
            )
        L.append(
            "\n### Coverage (late epoch, non-overlapping origins) — empirical hit-rate at nominal level, Kupiec p in brackets\n"
        )
        hdr = (
            "| asset | h | variant | "
            + " | ".join(f"{a:.0%}" for a in LEVELS)
            + " | MAD |"
        )
        L.append(hdr)
        L.append("|---|---:|---|" + "---:|" * (len(LEVELS) + 1))
        for c in cells:
            e = c["bases"][base_name]
            b = e["base"]["late"]
            r = e["windows"][str(chosen_window)]["late"]
            if r is None or not b["coverage"]:
                continue
            for label, blk, mad in (
                ("base", b["coverage"], b["coverage_mad"]),
                ("conformal", r["coverage"], r["coverage_mad"]),
            ):
                cellsx = []
                for a in LEVELS:
                    v = blk[str(a)]
                    cellsx.append(f"{v['empirical']:.3f} ({v['kupiec_p']:.2f})")
                L.append(
                    f"| {c['symbol']} | {c['h']} | {label} | "
                    + " | ".join(cellsx)
                    + f" | {mad:.3f} |"
                )
        L.append("")
    L.append("## Reading guide\n")
    L.append(
        "* **Coverage MAD** is the mean absolute gap between empirical and nominal "
        "coverage over the seven levels. The conformal guarantee says this should "
        "shrink toward the finite-sample noise floor; it does not promise CRPS gains.\n"
        "* **CRPS** can go either way: a well-calibrated but *wider* forecast can score "
        "worse than a sharp mis-calibrated one on the body. A CRPS loss with a "
        "coverage gain means the base was sharp but over-confident.\n"
        "* Bold DM p-values are < 0.05 (lag-(h-1) HAC, two-sided).\n"
    )
    return "\n".join(L) + "\n"


def _selection_table(cells: list[dict]) -> tuple[int, dict]:
    sel: dict = {}
    for w in CALIB_WINDOWS:
        d, mb, mc, imp, n = [], [], [], 0, 0
        for c in cells:
            r = c["bases"]["default"]["windows"][str(w)]["early"]
            b = c["bases"]["default"]["base"]["early"]
            if r is None or b["crps"] is None:
                continue
            n += 1
            d.append((r["crps"] - r["crps_base"]) / r["crps_base"] * 100)
            imp += int(r["crps"] < r["crps_base"])
            if b.get("coverage_mad") is not None and r.get("coverage_mad") is not None:
                mb.append(b["coverage_mad"])
                mc.append(r["coverage_mad"])
        sel[str(w)] = {
            "mean_dcrps_pct": float(np.mean(d)) if d else float("nan"),
            "mad_base": float(np.mean(mb)) if mb else float("nan"),
            "mad_cal": float(np.mean(mc)) if mc else float("nan"),
            "n_improved": imp,
            "n_cells": n,
        }
    chosen = min(CALIB_WINDOWS, key=lambda w: sel[str(w)]["mean_dcrps_pct"])
    return chosen, sel


# ----------------------------- main ----------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild the markdown from the saved results/conformal_*.json.",
    )
    args = ap.parse_args(argv)

    jobs = [(s, h) for s in args.symbols for h in args.horizons]
    RESULTS.mkdir(exist_ok=True)
    if args.render_only:
        cells = [
            json.loads((RESULTS / f"conformal_{slug(s)}_h{h}.json").read_text())
            for s, h in jobs
            if (RESULTS / f"conformal_{slug(s)}_h{h}.json").exists()
        ]
    else:
        from joblib import Parallel, delayed

        print(
            f"[conformal] {len(jobs)} cells × {len(BASES)} bases, n_jobs={args.n_jobs}",
            flush=True,
        )
        cells = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(evaluate_cell)(s, h) for s, h in jobs
        )
        for c in cells:
            out = RESULTS / f"conformal_{slug(c['symbol'])}_h{c['h']}.json"
            out.write_text(json.dumps(c, indent=1))
    chosen, sel = _selection_table(cells)
    summary = {
        "chosen_window": chosen,
        "selection": sel,
        "panel_data_end": PANEL_DATA_END,
    }
    (RESULTS / "conformal_summary.json").write_text(json.dumps(summary, indent=1))
    DOC.write_text(render_report(cells, chosen, sel))
    print(f"[conformal] chosen window = {chosen}; wrote {DOC}", flush=True)
    for w in CALIB_WINDOWS:
        s = sel[str(w)]
        print(
            f"  window {w:>3}: early ΔCRPS {s['mean_dcrps_pct']:+.2f}%  MAD {s['mad_base']:.3f}→{s['mad_cal']:.3f}  improved {s['n_improved']}/{s['n_cells']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
