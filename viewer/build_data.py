"""Bundle results/ + data/ into a single JSON the static viewer consumes.

Read-only: does not touch src/wbtc/. Run from the repo root with:

    uv run python viewer/build_data.py

Outputs: viewer/public/data.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "public" / "data.json"

# methods grouped so the frontend can present families consistently
BASELINE_METHODS = [
    "Static",
    "RW-Drift",
    "HS-Bootstrap",
    "GARCH-N",
    "GARCH-t",
    "GJR-GARCH-t",
]
WGEO_METHODS = [
    "WGeo",
    "WGeo-Gated",
    "WGeo-TheilSen",
    "WGeo-EWMA",
    "WGeo-Hetero",
    "WGeo-GARCH-Ens",
    # v0.4 additions — recency-weighted base quantile + W₂ barycentre ensemble
    "WGeo-Adaptive",
    "WGeo-Ensemble",
]
EXTENDED_METHODS = [
    "HAR-RV",
    "CAViaR-SAV",
    "MS-Normal-2",
    "FIGARCH(1,d,0)",
    "SV-AR1",
    "BVAR-GARCH(BTC,ETH)",
]

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
HORIZONS = [1, 5, 21]


def _slug(symbol: str) -> str:
    return symbol.lower().replace("/", "")


def _safe_load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _summarise_method_arrays(d: dict, methods: list[str]) -> dict:
    """For each method present, compute mean CRPS and cumulative CRPS series."""
    out: dict[str, dict] = {}
    for m in methods:
        if m not in d:
            continue
        arr = np.asarray(d[m], dtype=float)
        cumsum = np.cumsum(arr).tolist()
        out[m] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(arr.size),
            "cum": cumsum,
        }
    return out


def _diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Simple HAC-light DM test (Newey-West with bandwidth=floor(n**(1/3)))."""
    d = loss_a - loss_b
    n = d.size
    if n < 20:
        return float("nan"), float("nan")
    mean = d.mean()
    L = int(np.floor(n ** (1 / 3)))
    var = (d - mean).var(ddof=0)
    for k in range(1, L + 1):
        cov_k = np.mean((d[k:] - mean) * (d[:-k] - mean))
        var += 2 * (1 - k / (L + 1)) * cov_k
    if var <= 0:
        return float("nan"), float("nan")
    stat = mean / math.sqrt(var / n)
    # two-sided p-value via standard-normal approximation
    from math import erf, sqrt as msqrt

    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / msqrt(2))))
    return float(stat), float(p)


def _wbtc_dm(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Vanilla DM with the same HAC settings as wbtc.scoring.diebold_mariano."""
    from wbtc.scoring import diebold_mariano

    stat, p = diebold_mariano(loss_a, loss_b, h=h)
    return float(stat), float(p)


def _wbtc_dm_residualised(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    controls: list[np.ndarray],
    h: int,
) -> tuple[float, float]:
    """Residualised DM (Giacomini-White-style) using wbtc's scoring impl."""
    from wbtc.scoring import diebold_mariano_residualised

    stat, p = diebold_mariano_residualised(loss_a, loss_b, controls, h=h)
    return float(stat), float(p)


def _downsample(xs: list[float], target: int) -> list[float]:
    """Bin-average to ~target points for chart smoothness."""
    n = len(xs)
    if n <= target:
        return xs
    step = n / target
    out: list[float] = []
    for i in range(target):
        lo = int(i * step)
        hi = int((i + 1) * step)
        out.append(float(np.mean(xs[lo:hi])))
    return out


def _approx_dates_for_n(symbol: str, n_test: int) -> list[str] | None:
    """Approximate t-indices to ISO dates using the parquet."""
    pq = DATA / f"{_slug(symbol)}_1d.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
    # walk-forward uses an initial 730-day train; align test to df[-n_test:]
    if len(df) < n_test:
        return None
    sub = df["ts"].iloc[-n_test:]
    return [str(t.date()) for t in pd.to_datetime(sub)]


def build_symbol_horizon(symbol: str, h: int) -> dict | None:
    slug = _slug(symbol)
    path = RESULTS / f"long_{slug}_h{h}.json"
    raw = _safe_load_json(path)
    if raw is None:
        return None

    methods = BASELINE_METHODS + WGEO_METHODS
    summary = _summarise_method_arrays(raw, methods)

    # Diebold-Mariano: each WGeo vs Static
    if "Static" in raw:
        static_arr = np.asarray(raw["Static"], dtype=float)
        for m in WGEO_METHODS:
            if m in raw:
                stat, p = _diebold_mariano(np.asarray(raw[m], dtype=float), static_arr)
                summary[m]["dm_vs_static"] = {"stat": stat, "p": p}

    # downsample cumulative arrays for chart payload size
    for m, info in summary.items():
        info["cum_ds"] = _downsample(info["cum"], 800)
        del info["cum"]  # keep payload small; viewer uses cum_ds

    n_test = max((s["n"] for s in summary.values()), default=0)
    dates = _approx_dates_for_n(symbol, n_test)

    return {
        "symbol": symbol,
        "h": h,
        "n_test": n_test,
        "methods": summary,
        "dates_ds": _downsample_dates(dates, 800) if dates else None,
    }


def _downsample_dates(dates: list[str], target: int) -> list[str]:
    n = len(dates)
    if n <= target:
        return dates
    step = n / target
    return [dates[min(int(i * step), n - 1)] for i in range(target)]


def build_extended(h: int) -> dict | None:
    raw = _safe_load_json(RESULTS / f"extended_btc_h{h}.json")
    if raw is None:
        return None
    # extended files include WGeo-GARCH-Ens + GARCH-t plus the 6 named comparators
    methods = ["WGeo-GARCH-Ens", "GARCH-t"] + EXTENDED_METHODS
    summary = _summarise_method_arrays(raw, methods)
    if "WGeo-GARCH-Ens" in raw:
        wge = np.asarray(raw["WGeo-GARCH-Ens"], dtype=float)
        for m in methods:
            if m == "WGeo-GARCH-Ens" or m not in raw:
                continue
            stat, p = _diebold_mariano(wge, np.asarray(raw[m], dtype=float))
            summary[m]["dm_vs_wgeo_ens"] = {"stat": stat, "p": p}
    for info in summary.values():
        info["cum_ds"] = _downsample(info["cum"], 800)
        del info["cum"]
    n_test = max((s["n"] for s in summary.values()), default=0)
    dates = _approx_dates_for_n("BTC/USDT", n_test)
    return {
        "symbol": "BTC/USDT",
        "h": h,
        "n_test": n_test,
        "methods": summary,
        "dates_ds": _downsample_dates(dates, 800) if dates else None,
    }


def build_prices() -> dict[str, list]:
    """Full daily OHLC per asset — no downsampling, so the candlestick chart
    has no visible gaps. dataZoom handles in-browser navigation."""
    out: dict[str, list] = {}
    for sym in SYMBOLS + ["XRP/USDT"]:
        pq = DATA / f"{_slug(sym)}_1d.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
        out[sym] = [
            {
                "t": str(pd.to_datetime(r["ts"]).date()),
                "o": float(r["open"]),
                "h": float(r["high"]),
                "l": float(r["low"]),
                "c": float(r["close"]),
                "v": float(r["volume"]),
            }
            for _, r in df.iterrows()
        ]
    return out


def build_returns_overlay() -> dict[str, dict]:
    """Full daily log-returns — no downsampling for the same reason as build_prices."""
    out = {}
    for sym in SYMBOLS + ["XRP/USDT"]:
        pq = DATA / f"{_slug(sym)}_1d.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
        df["r"] = np.log(df["close"].astype(float)).diff()
        df = df.dropna(subset=["r"]).reset_index(drop=True)
        out[sym] = {
            "dates": [str(pd.to_datetime(t).date()) for t in df["ts"].tolist()],
            "r": df["r"].tolist(),
        }
    return out


def build_sweep() -> list[dict]:
    p = RESULTS / "hyperparam_sweep.csv"
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "window": int(r["window"]),
                    "lookback": int(r["lookback"]),
                    "crps_early": float(r["crps_early"]),
                    "crps_late": float(r["crps_late"]),
                }
            )
    return rows


def build_provenance() -> dict | None:
    raw = _safe_load_json(RESULTS / "MANIFEST.json")
    if not raw:
        return None
    # use the most-recent long-horizon run for headline; fall back to last entry
    last = raw[-1]
    headline = None
    for entry in reversed(raw):
        if entry.get("extra", {}).get("headline"):
            headline = entry
            break
    return {
        "latest_run": {
            "timestamp": last.get("timestamp"),
            "git_sha": last.get("git_sha"),
            "git_dirty": last.get("git_dirty"),
            "python": last.get("python"),
            "packages": last.get("packages"),
            "data_sha256": last.get("data_sha256"),
            "entry_point": last.get("entry_point"),
        },
        "headline": (headline or {}).get("extra", {}).get("headline", []),
        "headline_run": {
            "timestamp": (headline or {}).get("timestamp"),
            "git_sha": (headline or {}).get("git_sha"),
        },
    }


def main() -> None:
    out = {
        "version": "0.4",
        "schema": 1,
        "symbols": SYMBOLS,
        "horizons": HORIZONS,
        "baseline_methods": BASELINE_METHODS,
        "wgeo_methods": WGEO_METHODS,
        "extended_methods": EXTENDED_METHODS,
        "long": {},
        "extended": {},
        "prices": build_prices(),
        "returns": build_returns_overlay(),
        "sweep": build_sweep(),
        "provenance": build_provenance(),
    }

    for sym in SYMBOLS:
        out["long"][sym] = {}
        for h in HORIZONS:
            section = build_symbol_horizon(sym, h)
            if section:
                out["long"][sym][str(h)] = section

    for h in [1, 5]:
        section = build_extended(h)
        if section:
            out["extended"][str(h)] = section

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({size_kb:.1f} KB)")
    print(
        f"  symbols={len(SYMBOLS)} horizons={HORIZONS}  "
        f"sweep_rows={len(out['sweep'])}  "
        f"prices={len(out['prices'])} assets"
    )


if __name__ == "__main__":
    main()
