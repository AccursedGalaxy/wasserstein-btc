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

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
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


_RETURNS_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "returns.json"
_RETURNS_CACHE: dict[str, np.ndarray] | None = None


def _load_returns_cache() -> dict[str, np.ndarray]:
    """Lazy-load the committed daily-return cache (~160 KB).

    Used when the parquet files are unavailable (CI runners hit Binance's
    HTTP 451 on Azure IPs). Refresh with
    ``uv run python viewer/cache/build_returns_cache.py`` whenever the
    backtest data is updated.
    """
    global _RETURNS_CACHE
    if _RETURNS_CACHE is None:
        if not _RETURNS_CACHE_PATH.exists():
            _RETURNS_CACHE = {}
        else:
            raw = json.loads(_RETURNS_CACHE_PATH.read_text())
            _RETURNS_CACHE = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
    return _RETURNS_CACHE


def _log_returns_from_parquet(symbol: str) -> np.ndarray | None:
    """Return the same log-return series used by the backtest harness.

    Mirrors ``wbtc.backtest.load_returns``: difference of log-close, drop
    the first NaN. The returned array's indices match what was passed to
    ``run_long_horizon`` as ``returns`` — so ``t_idx`` values from
    ``long_*.json`` index into it directly.

    Falls back to ``viewer/cache/returns.json`` (committed) when the
    parquet is unavailable.
    """
    pq = DATA / f"{_slug(symbol)}_1d.parquet"
    if pq.exists():
        df = pd.read_parquet(pq).sort_values("ts").reset_index(drop=True)
        log_close = np.log(df["close"].astype(float).to_numpy())
        return np.diff(log_close)
    cache = _load_returns_cache()
    return cache.get(symbol)


def _realised_h_step(returns: np.ndarray, t_idx: list[int], h: int) -> np.ndarray:
    """Realised h-step log return for each step in t_idx (mirrors h_step_log_return)."""
    out = np.empty(len(t_idx), dtype=float)
    for i, t in enumerate(t_idx):
        out[i] = float(returns[t + 1 : t + h + 1].sum())
    return out


def build_headline(symbol: str, h: int) -> dict | None:
    """Compute the v0.4 headline row for (symbol, h) from raw loss arrays.

    Mirrors ``scripts/run_long_horizon.py``'s headline logic: best
    WGeo-family variant (mean CRPS) vs best non-WGeo baseline, with the
    classic Diebold-Mariano statistic *and* the residualised
    Giacomini-White-style augmented variant (peer-method losses +
    |y|, y² as controls). The MANIFEST.json headline can lag behind
    when v0.4 methods are added incrementally — computing it from the
    raw loss arrays keeps the viewer in lock-step with the backend.
    """
    slug = _slug(symbol)
    raw = _safe_load_json(RESULTS / f"long_{slug}_h{h}.json")
    if not raw or "t_idx" not in raw:
        return None
    t_idx = list(raw["t_idx"])
    losses = {k: np.asarray(v, dtype=float) for k, v in raw.items() if k != "t_idx"}
    wgeo = [m for m in WGEO_METHODS if m in losses]
    baseline = [m for m in BASELINE_METHODS if m in losses]
    if not wgeo or not baseline:
        return None
    means = {m: float(losses[m].mean()) for m in wgeo + baseline}
    best_w = min(wgeo, key=lambda m: means[m])
    best_b = min(baseline, key=lambda m: means[m])

    # vanilla DM with the same lag-(h-1) HAC the long-horizon panel uses
    dm_stat, dm_p = _wbtc_dm(losses[best_w], losses[best_b], h=h)

    # Residualised DM controls. Peer-method losses are the dominant signal
    # and always available. The |y|, y² controls require the daily-return
    # parquet, which is gitignored — on CI runners (Binance 451 on Azure)
    # the parquets are absent and we fall back to peer-only residualisation.
    # The local dashboard, where parquets exist, gets the full v0.4 controls.
    peer_ctrls = [losses[k] for k in losses if k != best_w and k != best_b][:4]
    ctrls = list(peer_ctrls)
    returns = _log_returns_from_parquet(symbol)
    if returns is not None and len(returns) > max(t_idx) + h:
        y = _realised_h_step(returns, t_idx, h)
        ctrls.extend([np.abs(y), y * y])
    dm_stat_r, dm_p_r = _wbtc_dm_residualised(
        losses[best_w], losses[best_b], ctrls, h=h
    )

    wbest = means[best_w]
    bbest = means[best_b]
    improvement = (wbest - bbest) / bbest if bbest else float("nan")
    return {
        "symbol": symbol,
        "h": h,
        "n_test": int(losses[best_w].size),
        "best_wgeo": best_w,
        "best_baseline": best_b,
        "wgeo_crps": wbest,
        "baseline_crps": bbest,
        "improvement": f"{improvement * 100:+.1f}%",
        "dm_stat": dm_stat,
        "dm_p": dm_p,
        "dm_stat_r": dm_stat_r,
        "dm_p_r": dm_p_r,
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
        "schema": 2,
        "symbols": SYMBOLS,
        "horizons": HORIZONS,
        "baseline_methods": BASELINE_METHODS,
        "wgeo_methods": WGEO_METHODS,
        "extended_methods": EXTENDED_METHODS,
        "long": {},
        "extended": {},
        "headline": [],
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
            row = build_headline(sym, h)
            if row:
                out["headline"].append(row)

    # extended panel covers BTC for the full {1, 5, 21} horizon set in v0.4
    for h in HORIZONS:
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
