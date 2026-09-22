"""Daily live forecasts with the conformal layer, for personal use.

This module produces *today's* distributional forecast for each cached
asset and horizon, calibrated by :class:`wbtc.conformal.ConformalCalibrator`
on the most recent ``calib_window`` forecast errors, and writes three local
artefacts under ``results/live/`` (gitignored — nothing here is published):

* ``latest.json``     — every (asset, horizon) forecast from the last run,
* ``log.jsonl``       — append-only sealed log, one line per (asof, asset,
  horizon); re-running on the same day is a no-op for the log, so the file
  is a faithful record of what the model said *before* the outcome was
  known (the ROADMAP "live paper-trading" item),
* ``forecasts.html``  — a self-contained page with a price-level table and
  fan charts, for reading in a browser.

Target convention: a forecast "as of" day ``t`` (last closed daily candle)
covers the log-return from ``close_t`` to ``close_{t+h}``. This is the
live-API convention (no skipped day); the research harness skips one day,
see ``THEORY.md §2.12``.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import default_forecaster
from .conformal import ConformalCalibrator, empirical_coverage
from .data import load_returns
from .forecasters import Forecaster
from .quantiles import make_grid

__all__ = ["LiveForecast", "live_forecast", "run_daily", "DEFAULT_CALIB_WINDOW"]

# Chosen on the early epoch of the research panel (docs/RESULTS_CONFORMAL.md).
DEFAULT_CALIB_WINDOW = 500
DEFAULT_MIN_CALIB = 50
DEFAULT_TRAIN_WINDOW = 730
DEFAULT_K = 50
DEFAULT_HORIZONS = (1, 5, 21)
DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT")
REPORT_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class LiveForecast:
    """One calibrated forecast plus the diagnostics a trader needs to trust it."""

    symbol: str
    asof: pd.Timestamp
    horizon: int
    method: str
    last_close: float
    quantile_levels: np.ndarray
    base_quantiles: np.ndarray
    calibrated_quantiles: np.ndarray
    offsets: np.ndarray | None
    calibration_active: bool
    n_calib: int
    calib_window: int
    train_window: int
    # empirical coverage of the *base* forecast at REPORT_LEVELS over the
    # calibration window — "how often was the base 5% quantile breached"
    base_coverage_recent: dict[float, float]
    # day-to-day stability: mean absolute change of the base median and of the
    # 5–95 % width over the last 30 origins, in log-return units
    median_daily_drift: float
    width_daily_drift: float
    regime: dict[str, Any] = field(default_factory=dict)
    # recent history for charts: origin date, base/calibrated q05/q50/q95,
    # realised h-step return (NaN when not yet known)
    history: pd.DataFrame | None = field(default=None, repr=False)

    # -- convenience --

    def quantile(self, u: float, calibrated: bool = True) -> float:
        q = self.calibrated_quantiles if calibrated else self.base_quantiles
        return float(np.interp(u, self.quantile_levels, q))

    def price(self, u: float, calibrated: bool = True) -> float:
        return self.last_close * float(np.exp(self.quantile(u, calibrated)))

    @property
    def asof_close(self) -> pd.Timestamp:
        """Instant the anchor bar closed (bar label + 1 day, UTC)."""
        return self.asof + pd.Timedelta(days=1)

    @property
    def target_date(self) -> pd.Timestamp:
        """Label of the bar whose close settles the forecast."""
        return self.asof + pd.Timedelta(days=self.horizon)

    @property
    def target_close(self) -> pd.Timestamp:
        """Instant the forecast settles: close of the bar labelled ``target_date``."""
        return self.target_date + pd.Timedelta(days=1)

    @property
    def stale(self) -> bool:
        """True once the next bar after the anchor has already closed, i.e. the
        h=1 outcome is already knowable and the forecast is not 'live'."""
        return pd.Timestamp.now("UTC") >= self.asof_close + pd.Timedelta(days=1)

    def to_dict(self) -> dict[str, Any]:
        lv = list(REPORT_LEVELS)
        return {
            "symbol": self.symbol,
            "asof": str(self.asof.date()),
            "asof_close_utc": self.asof_close.isoformat(),
            "target_date": str(self.target_date.date()),
            "target_close_utc": self.target_close.isoformat(),
            "stale": self.stale,
            "horizon_days": self.horizon,
            "method": self.method,
            "conformal": {
                "active": self.calibration_active,
                "n_calib": self.n_calib,
                "calib_window": self.calib_window,
            },
            "train_window_days": self.train_window,
            "last_close": self.last_close,
            "median_log_return": self.quantile(0.5),
            "median_pct": (np.exp(self.quantile(0.5)) - 1) * 100,
            "quantile_levels": self.quantile_levels.tolist(),
            "base_quantiles": self.base_quantiles.tolist(),
            "calibrated_quantiles": self.calibrated_quantiles.tolist(),
            "offsets": None if self.offsets is None else self.offsets.tolist(),
            "levels": {
                str(u): {
                    "log_return": self.quantile(u),
                    "pct": (np.exp(self.quantile(u)) - 1) * 100,
                    "price": self.price(u),
                    "base_price": self.price(u, calibrated=False),
                }
                for u in lv
            },
            "base_coverage_recent": {
                str(k): v for k, v in self.base_coverage_recent.items()
            },
            "median_daily_drift": self.median_daily_drift,
            "width_daily_drift": self.width_daily_drift,
            "regime": self.regime,
        }


def _regime_info(base: Forecaster) -> dict[str, Any]:
    out: dict[str, Any] = {}
    w = getattr(base, "_weight", None)
    if w is not None:
        out["garch_weight"] = float(w)  # WGeoGarchEnsemble: 0 = pure WGeo
    fb = getattr(base, "_garch_fallback", None)
    if fb is not None:
        out["garch_fallback"] = bool(fb)
    return out


def live_forecast(
    symbol: str,
    horizon: int,
    *,
    asof: pd.Timestamp | str | None = None,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    K: int = DEFAULT_K,
    calib_window: int = DEFAULT_CALIB_WINDOW,
    min_calib: int = DEFAULT_MIN_CALIB,
    forecaster_factory: Callable[[int], Forecaster] | None = None,
    history_days: int = 60,
) -> LiveForecast:
    """Calibrated forecast for ``symbol`` at horizon ``horizon``.

    Walks the base forecaster forward over the last ``calib_window + horizon
    + history_days`` origins (each refit on its own trailing
    ``train_window``), feeding :class:`ConformalCalibrator` online so that
    today's forecast is corrected by the empirical errors of the base
    forecaster's own recent past. Takes a few seconds per call.
    """
    df = load_returns(symbol)
    # ``ts`` is the bar OPEN time; the bar labelled D closes at D + 1 day.
    # Never anchor on a bar that has not closed yet (a mid-day snapshot
    # written by an earlier fetch would otherwise be treated as a close).
    now = pd.Timestamp.now("UTC")
    df = df[df["ts"] + pd.Timedelta(days=1) <= now].reset_index(drop=True)
    if asof is not None:
        cutoff = pd.Timestamp(asof, tz="UTC")
        df = df[df["ts"] <= cutoff].reset_index(drop=True)
    r = df["r"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    ts = pd.to_datetime(df["ts"], utc=True)
    N = len(r)
    n_origins = calib_window + horizon + history_days + 1
    if N < train_window + n_origins:
        raise ValueError(f"{symbol}: need {train_window + n_origins} returns, have {N}")
    u = make_grid(K)
    make = forecaster_factory or default_forecaster
    cal = ConformalCalibrator(
        make(horizon), calib_window=calib_window, min_calib=min_calib
    )

    origins = range(N - n_origins, N)
    base_hist: list[np.ndarray] = []
    cal_hist: list[np.ndarray] = []
    q_cal = q_base = None
    for t in origins:
        window = r[t - train_window + 1 : t + 1]
        cal.fit(window)
        q_cal = cal.predict(horizon, u)
        q_base = cal.last_base
        base_hist.append(np.asarray(q_base, dtype=float))
        cal_hist.append(np.asarray(q_cal, dtype=float))
    assert q_cal is not None and q_base is not None
    base = cal.base

    # realised h-step targets for the historical origins (NaN if still open)
    y = np.full(n_origins, np.nan)
    for i, t in enumerate(origins):
        if t + horizon <= N - 1:
            y[i] = float(r[t + 1 : t + 1 + horizon].sum())
    B = np.stack(base_hist)
    C = np.stack(cal_hist)
    known = ~np.isnan(y)
    # coverage of the base over the last calib_window resolved origins
    kb = np.where(known)[0][-calib_window:]
    cov = (
        empirical_coverage(B[kb], u, y[kb], levels=np.array(REPORT_LEVELS))
        if len(kb)
        else {}
    )

    # stability diagnostics over the last 30 origins
    med = np.array([np.interp(0.5, u, q) for q in B[-31:]])
    wid = np.array([np.interp(0.95, u, q) - np.interp(0.05, u, q) for q in B[-31:]])
    hist = (
        pd.DataFrame(
            {
                "origin": [ts.iloc[t].date().isoformat() for t in origins],
                "close": [closes[t] for t in origins],
                "base_q05": [np.interp(0.05, u, q) for q in B],
                "base_q50": [np.interp(0.50, u, q) for q in B],
                "base_q95": [np.interp(0.95, u, q) for q in B],
                "cal_q05": [np.interp(0.05, u, q) for q in C],
                "cal_q50": [np.interp(0.50, u, q) for q in C],
                "cal_q95": [np.interp(0.95, u, q) for q in C],
                "realised": y,
            }
        )
        .tail(history_days + horizon + 1)
        .reset_index(drop=True)
    )

    return LiveForecast(
        symbol=symbol,
        asof=ts.iloc[N - 1],
        horizon=horizon,
        method=type(base).__name__,
        last_close=float(closes[N - 1]),
        quantile_levels=u,
        base_quantiles=np.asarray(q_base, dtype=float),
        calibrated_quantiles=np.asarray(q_cal, dtype=float),
        offsets=None if cal.last_offsets is None else np.asarray(cal.last_offsets),
        calibration_active=bool(cal.last_active),
        n_calib=cal.n_scores(horizon),
        calib_window=calib_window,
        train_window=train_window,
        base_coverage_recent=cov,
        median_daily_drift=float(np.mean(np.abs(np.diff(med)))),
        width_daily_drift=float(np.mean(np.abs(np.diff(wid)))),
        regime=_regime_info(base),
        history=hist,
    )


# ----------------------------- daily run ------------------------------------


def _fan_chart_b64(fcs: list[LiveForecast]) -> str:
    """Recent closes + calibrated bands for every horizon, as base64 PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f0 = fcs[0]
    hist = f0.history
    assert hist is not None
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=110)
    dates = pd.to_datetime(hist["origin"])
    ax.plot(dates, hist["close"], color="#222", lw=1.3, label="close")
    last_ts = f0.asof.tz_localize(None)
    colors = {1: "#7c83f2", 5: "#3fc2a5", 21: "#f0a64a"}
    for fc in sorted(fcs, key=lambda f: f.horizon):
        c = colors.get(fc.horizon, "#888")
        tgt = last_ts + pd.Timedelta(days=fc.horizon)
        for lo, hi, a in ((0.05, 0.95, 0.15), (0.25, 0.75, 0.30)):
            ax.fill_between(
                [last_ts, tgt],
                [fc.last_close, fc.price(lo)],
                [fc.last_close, fc.price(hi)],
                color=c,
                alpha=a,
                lw=0,
            )
        ax.plot(
            [last_ts, tgt],
            [fc.last_close, fc.price(0.5)],
            color=c,
            lw=1.4,
            label=f"h={fc.horizon} median (5–95 %, 25–75 %)",
        )
        ax.plot(
            [last_ts, tgt],
            [fc.last_close, fc.price(0.5, calibrated=False)],
            color=c,
            lw=0.8,
            ls="--",
        )
    ax.axvline(last_ts, color="#999", lw=0.6)
    ax.set_title(
        f"{f0.symbol} · as of {f0.asof.date()} · dashed = base median, solid = conformal"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _history_chart_b64(fc: LiveForecast) -> str:
    """Past base/calibrated 5–95 % bands vs the realised h-step return."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = fc.history
    assert h is not None
    fig, ax = plt.subplots(figsize=(9.5, 2.8), dpi=110)
    d = pd.to_datetime(h["origin"])
    ax.fill_between(
        d,
        h["base_q05"] * 100,
        h["base_q95"] * 100,
        color="#7c83f2",
        alpha=0.18,
        lw=0,
        label="base 5–95 %",
    )
    ax.plot(d, h["cal_q05"] * 100, color="#3fc2a5", lw=1, label="conformal 5 % / 95 %")
    ax.plot(d, h["cal_q95"] * 100, color="#3fc2a5", lw=1)
    ax.plot(
        d,
        h["cal_q50"] * 100,
        color="#3fc2a5",
        lw=0.8,
        ls="--",
        label="conformal median",
    )
    ax.plot(
        d,
        h["realised"] * 100,
        color="#222",
        lw=1.2,
        marker=".",
        ms=3,
        label=f"realised {fc.horizon}-day return",
    )
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_ylabel("%")
    ax.set_title(
        f"{fc.symbol} h={fc.horizon}: past forecasts vs outcome (origins on x)",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_price(x: float) -> str:
    return f"{x:,.2f}" if x >= 10 else f"{x:,.4f}"


def render_html(fcs: list[LiveForecast], generated: pd.Timestamp) -> str:
    if not fcs:
        raise ValueError("no forecasts to render")
    by_sym: dict[str, list[LiveForecast]] = {}
    for fc in fcs:
        by_sym.setdefault(fc.symbol, []).append(fc)
    css = """
    body{font-family:system-ui,sans-serif;margin:0;background:#f7f7f9;color:#1b1b1f}
    main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
    h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 8px}
    .muted{color:#6b6b73;font-size:13px}
    table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px;background:#fff;border:1px solid #e3e3e8;border-radius:8px;overflow:hidden}
    th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eee}th:first-child,td:first-child{text-align:left}
    th{background:#f0f0f4;font-weight:600;font-size:12px}
    .neg{color:#c0392b}.pos{color:#1e8e5a}.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;background:#e8f7f1;color:#1e8e5a}
    .tag.off{background:#fdeeee;color:#c0392b}
    img{max-width:100%;border:1px solid #e3e3e8;border-radius:8px;background:#fff;margin-top:8px}
    .note{font-size:12px;color:#6b6b73;margin-top:6px}
    """
    parts = [f"<title>wbtc live forecasts</title><style>{css}</style><main>"]
    parts.append("<h1>wbtc · live distributional forecasts</h1>")
    parts.append(
        f"<div class='muted'>generated {generated.strftime('%Y-%m-%d %H:%M UTC')} · quantiles are conformally "
        f"calibrated on the base forecaster's last {fcs[0].calib_window} errors · price levels = last close × exp(quantile) · "
        "target = close-to-close over h days · private, local file</div>"
    )
    if any(fc.stale for fc in fcs):
        parts.append(
            "<div class='muted' style='color:#c0392b'>STALE: the data cache is behind; "
            "these anchors are not the latest closed bar. Run <code>wbtc forecast-all</code> "
            "with fetching enabled.</div>"
        )
    parts.append(
        "<div class='muted'>as-of = last CLOSED daily bar (label = bar open date, UTC). "
        "A bar labelled D closes at D+1 00:00 UTC; the h-day target settles at the close of bar asof+h.</div>"
    )
    for sym, group in by_sym.items():
        group = sorted(group, key=lambda f: f.horizon)
        f0 = group[0]
        parts.append(
            f"<h2>{sym} <span class='muted'>last close {_fmt_price(f0.last_close)} · bar {f0.asof.date()} "
            f"(closed {f0.asof_close.strftime('%Y-%m-%d %H:%M')} UTC)</span></h2>"
        )
        parts.append(
            "<table><tr><th>h</th><th>settles (close of bar)</th><th>method</th><th>median</th>"
            "<th>5 %</th><th>25 %</th><th>75 %</th><th>95 %</th><th>5–95 % width</th>"
            "<th>base 5 % / 95 % hit-rate*</th><th>regime</th><th>conformal</th></tr>"
        )
        for fc in group:
            med = fc.quantile(0.5)
            cls = "pos" if med >= 0 else "neg"
            cov = fc.base_coverage_recent
            reg = (
                ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in fc.regime.items()
                )
                or "—"
            )
            tag = (
                f"<span class='tag'>on · n={fc.n_calib}</span>"
                if fc.calibration_active
                else "<span class='tag off'>off</span>"
            )
            parts.append(
                f"<tr><td>{fc.horizon}d</td><td>{fc.target_date.date()}</td><td>{fc.method}</td>"
                f"<td class='{cls}'>{(np.exp(med) - 1) * 100:+.2f} %<br><span class='muted'>{_fmt_price(fc.price(0.5))}</span></td>"
                + "".join(
                    f"<td>{_fmt_price(fc.price(u))}<br><span class='muted'>{(np.exp(fc.quantile(u)) - 1) * 100:+.1f} %</span></td>"
                    for u in (0.05, 0.25, 0.75, 0.95)
                )
                + f"<td>{(fc.quantile(0.95) - fc.quantile(0.05)) * 100:.1f} pp</td>"
                f"<td>{cov.get(0.05, float('nan')):.3f} / {cov.get(0.95, float('nan')):.3f}</td>"
                f"<td class='muted'>{reg}</td><td>{tag}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            "<div class='note'>*share of the last calibration-window outcomes that fell below the base 5 % quantile / below the base 95 % quantile (nominal 0.05 / 0.95). "
            "Regime: garch_weight is the WGeo↔GARCH blend of the h=5 ensemble (0 = calm, pure WGeo); garch_fallback flags a failed GARCH fit at h=21.</div>"
        )
        parts.append(
            f"<img src='data:image/png;base64,{_fan_chart_b64(group)}' alt='fan chart {sym}'>"
        )
        for fc in group:
            parts.append(
                f"<img src='data:image/png;base64,{_history_chart_b64(fc)}' alt='history {sym} h={fc.horizon}'>"
            )
    parts.append(
        "<p class='note'>Forecasts are distributions, not signals. The median is the centre of a wide band; the 5–95 % band is what the model thinks is plausible. "
        "A breach rate far from nominal in the hit-rate column means the base model has been mis-calibrated recently and the conformal offsets are doing real work.</p></main>"
    )
    return "\n".join(parts)


def _print_table(fcs: list[LiveForecast]) -> None:
    print(
        f"{'symbol':<9} {'h':>3} {'asof':<11} {'close':>12} {'median':>8} {'q05':>12} {'q25':>12} {'q75':>12} {'q95':>12} {'cal':>6} {'base hit 5/95':>14}"
    )
    print("-" * 122)
    for fc in fcs:
        med = (np.exp(fc.quantile(0.5)) - 1) * 100
        cov = fc.base_coverage_recent
        print(
            f"{fc.symbol.replace('/USDT', ''):<9} {fc.horizon:>3} {str(fc.asof.date()):<11} {fc.last_close:>12,.2f} "
            f"{med:>+7.2f}% {fc.price(0.05):>12,.2f} {fc.price(0.25):>12,.2f} {fc.price(0.75):>12,.2f} {fc.price(0.95):>12,.2f} "
            f"{'on' if fc.calibration_active else 'off':>6} {cov.get(0.05, float('nan')):>6.3f}/{cov.get(0.95, float('nan')):.3f}"
        )


def _log_key(rec: dict) -> tuple:
    return (
        rec["asof"],
        rec["symbol"],
        rec["horizon_days"],
        rec["method"],
        rec["train_window_days"],
        rec["conformal"]["calib_window"],
    )


def run_daily(
    symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
    *,
    out_dir: Path,
    calib_window: int = DEFAULT_CALIB_WINDOW,
    n_jobs: int = -1,
    quiet: bool = False,
) -> list[LiveForecast]:
    """Forecast every (symbol, horizon), write latest.json / log.jsonl / forecasts.html."""
    from joblib import Parallel, delayed

    jobs = [(s, h) for s in symbols for h in horizons]
    fcs: list[LiveForecast] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(live_forecast)(s, h, calib_window=calib_window) for s, h in jobs
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = pd.Timestamp.now("UTC")
    records = [fc.to_dict() for fc in fcs]
    (out_dir / "latest.json").write_text(
        json.dumps({"generated": generated.isoformat(), "forecasts": records}, indent=1)
    )
    # sealed append-only log: one row per (asof, symbol, horizon, config);
    # stale forecasts (anchor bar's successor already closed) are never
    # logged, so every row was written before its h=1 outcome was knowable.
    log = out_dir / "log.jsonl"
    seen: set[tuple] = set()
    n_bad = 0
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                seen.add(_log_key(d))
            except (ValueError, KeyError):
                n_bad += 1  # truncated line from an interrupted run
    n_new = n_stale = 0
    with log.open("a") as f:
        for rec in records:
            if rec["stale"]:
                n_stale += 1
                continue
            if _log_key(rec) in seen:
                continue
            rec_logged = {"logged_at": generated.isoformat(), **rec}
            f.write(json.dumps(rec_logged, separators=(",", ":")) + "\n")
            n_new += 1
    (out_dir / "forecasts.html").write_text(render_html(fcs, generated))
    if not quiet:
        _print_table(fcs)
        extra = f", {n_stale} stale rows NOT logged" if n_stale else ""
        extra += f", {n_bad} unreadable existing lines skipped" if n_bad else ""
        print(
            f"\nwrote {out_dir / 'latest.json'}, {out_dir / 'forecasts.html'}; "
            f"sealed log +{n_new} rows{extra} -> {log}"
        )
    return fcs
