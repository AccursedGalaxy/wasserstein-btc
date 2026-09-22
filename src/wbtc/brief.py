"""Trading brief: everything the trader agent needs, in one markdown document.

Private, local only (``results/live/``). Combines

* market context per asset (recent returns, realised-vol percentile,
  distance from the 60-day high) from the parquet cache,
* the latest conformal forecasts (``latest.json``) turned into the numbers
  a trader acts on: tail probabilities, expected shortfall, price levels,
* historical analogues: what happened over the next h days after the same
  60-day-return quintile in this asset's own history,
* Bitget USDT-M perpetual context (funding rate, mark price, contract
  minimums) via ccxt public endpoints — no API key needed,
* position sizing for a given account size / risk budget, derived from the
  calibrated 5 % expected shortfall, not from a chart stop,
* a scorecard of the sealed log: every logged forecast whose target bar has
  closed, scored on hit-rate and PIT.

The brief states distributions and risk; it never emits a buy/sell signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_returns
from .live import DEFAULT_SYMBOLS

__all__ = ["TradingConfig", "build_brief", "score_log", "load_config"]

DEFAULT_CONFIG = {
    "venue": "bitget",
    "instrument": "usdt-m perpetual futures",
    "account_usd": 100.0,
    "risk_per_trade_pct": 2.0,  # % of account lost on a 5%-tail event
    "max_leverage": 5.0,
    "horizons": [1, 5, 21],
    "symbols": list(DEFAULT_SYMBOLS),
    "notes": "testing account; sizes are for learning the loop, not for P&L",
}


@dataclass
class TradingConfig:
    venue: str
    instrument: str
    account_usd: float
    risk_per_trade_pct: float
    max_leverage: float
    horizons: list[int]
    symbols: list[str]
    notes: str = ""


def load_config(out_dir: Path) -> TradingConfig:
    """Read ``trading.json`` from the live dir, creating it with defaults."""
    p = out_dir / "trading.json"
    if not p.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(DEFAULT_CONFIG, indent=1))
    d = {**DEFAULT_CONFIG, **json.loads(p.read_text())}
    return TradingConfig(**{k: d[k] for k in TradingConfig.__dataclass_fields__})


# ----------------------------- forecast numbers ------------------------------


def _q(rec: dict, u: float, cal: bool = True) -> float:
    key = "calibrated_quantiles" if cal else "base_quantiles"
    return float(np.interp(u, rec["quantile_levels"], rec[key]))


def _p_below(rec: dict, x: float, cal: bool = True) -> float:
    key = "calibrated_quantiles" if cal else "base_quantiles"
    return float(np.interp(x, rec[key], rec["quantile_levels"], left=0.0, right=1.0))


def _es(rec: dict, a: float) -> float:
    u = np.asarray(rec["quantile_levels"])
    q = np.asarray(rec["calibrated_quantiles"])
    return float(q[u <= a].mean())


def forecast_numbers(rec: dict) -> dict[str, Any]:
    c = rec["last_close"]
    pct = lambda x: (np.exp(x) - 1) * 100  # noqa: E731
    return {
        "median_pct": pct(_q(rec, 0.5)),
        "p_down": _p_below(rec, 0.0),
        "p_down_5": _p_below(rec, np.log(0.95)),
        "p_down_10": _p_below(rec, np.log(0.90)),
        "p_up_5": 1 - _p_below(rec, np.log(1.05)),
        "p_up_10": 1 - _p_below(rec, np.log(1.10)),
        "es5_pct": pct(_es(rec, 0.05)),
        "q05": c * np.exp(_q(rec, 0.05)),
        "q25": c * np.exp(_q(rec, 0.25)),
        "q75": c * np.exp(_q(rec, 0.75)),
        "q95": c * np.exp(_q(rec, 0.95)),
        "q05_base": c * np.exp(_q(rec, 0.05, cal=False)),
        "q95_base": c * np.exp(_q(rec, 0.95, cal=False)),
        "width_pp": (_q(rec, 0.95) - _q(rec, 0.05)) * 100,
    }


# ----------------------------- market context --------------------------------


def market_context(symbol: str, asof: str) -> dict[str, Any]:
    df = load_returns(symbol)
    df = df[df["ts"] <= pd.Timestamp(asof, tz="UTC")].reset_index(drop=True)
    r = df["r"].to_numpy()
    c = df["close"].to_numpy()
    sig = np.array([r[i - 20 : i].std() for i in range(len(r) - 252, len(r) + 1)])
    return {
        "last7_pct": [round(float(x) * 100, 2) for x in r[-7:]],
        "ret_5d": float(r[-5:].sum() * 100),
        "ret_21d": float(r[-21:].sum() * 100),
        "ret_60d": float(r[-60:].sum() * 100),
        "from_60d_high": float((c[-1] / c[-60:].max() - 1) * 100),
        "from_60d_low": float((c[-1] / c[-60:].min() - 1) * 100),
        "vol20_pct_day": float(sig[-1] * 100),
        "vol20_pctile_1y": float(np.mean(sig[:-1] < sig[-1])),
        "vol20_60d_ago": float(sig[-61] * 100),
    }


def analogues(symbol: str, asof: str, horizons: list[int]) -> dict[str, Any]:
    """Next-h returns after the same 60-day-return quintile, non-overlapping.

    Small samples, wide dispersion: the agent treats this as a prior on the
    sign, never as a forecast.
    """
    df = load_returns(symbol)
    df = df[df["ts"] <= pd.Timestamp(asof, tz="UTC")].reset_index(drop=True)
    r = df["r"].to_numpy()
    n = len(r)
    hmax = max(horizons)
    cum60 = np.array([r[t - 60 : t].sum() for t in range(60, n - hmax)])
    now60 = r[-60:].sum()
    edges = np.quantile(cum60, [0.2, 0.4, 0.6, 0.8])
    bucket = int(np.searchsorted(edges, now60))
    lo = -np.inf if bucket == 0 else edges[bucket - 1]
    hi = np.inf if bucket == 4 else edges[bucket]
    idx = [t for t, x in zip(range(60, n - hmax), cum60) if lo <= x < hi]
    ep: list[int] = []
    for t in idx:
        if not ep or t - ep[-1] >= hmax:
            ep.append(t)
    out: dict[str, Any] = {
        "quintile": bucket + 1,
        "range_pct": [
            None if np.isinf(lo) else lo * 100,
            None if np.isinf(hi) else hi * 100,
        ],
        "n_episodes": len(ep),
        "by_h": {},
    }
    for h in horizons:
        f = np.array([r[t : t + h].sum() for t in ep]) * 100
        out["by_h"][str(h)] = {
            "median": float(np.median(f)),
            "p_neg": float(np.mean(f < 0)),
            "p10": float(np.percentile(f, 10)),
            "p90": float(np.percentile(f, 90)),
        }
    return out


# ----------------------------- bitget context --------------------------------


def bitget_context(symbol: str) -> dict[str, Any]:
    """Funding + contract limits for the USDT-M perp; empty dict on failure."""
    try:
        import ccxt

        ex = ccxt.bitget({"enableRateLimit": True})
        swap = f"{symbol}:USDT"
        m = ex.load_markets()[swap]
        fr = ex.fetch_funding_rate(swap)
        tk = ex.fetch_ticker(swap)
        rate = fr.get("fundingRate")
        return {
            "symbol": swap,
            "funding_rate_8h": rate,
            "funding_annualised_pct": None if rate is None else rate * 3 * 365 * 100,
            "next_funding_utc": pd.Timestamp(
                fr["fundingTimestamp"], unit="ms", tz="UTC"
            ).isoformat()
            if fr.get("fundingTimestamp")
            else None,
            "mark_or_last": tk.get("last"),
            "min_amount": m["limits"]["amount"]["min"],
            "min_cost_usd": m["limits"]["cost"]["min"],
            "max_leverage": m["limits"]["leverage"]["max"],
        }
    except Exception as e:  # offline or venue hiccup: the brief still renders
        return {"error": str(e)[:120]}


# ----------------------------- sizing ---------------------------------------


def sizing(rec: dict, nums: dict, cfg: TradingConfig, bg: dict) -> dict[str, Any]:
    """Risk-budget sizing: lose ``risk_per_trade_pct`` of the account on a
    5 %-tail event over the horizon (calibrated ES). Leverage is what that
    notional implies, capped at ``max_leverage``."""
    risk_usd = cfg.account_usd * cfg.risk_per_trade_pct / 100
    es = abs(nums["es5_pct"]) / 100
    notional = risk_usd / es if es > 0 else 0.0
    lev = notional / cfg.account_usd
    capped = lev > cfg.max_leverage
    if capped:
        lev = cfg.max_leverage
        notional = lev * cfg.account_usd
    price = rec["last_close"]
    qty = notional / price
    min_amt = bg.get("min_amount") or 0
    min_cost = bg.get("min_cost_usd") or 0
    return {
        "risk_usd": risk_usd,
        "notional_usd": notional,
        "qty": qty,
        "leverage": lev,
        "capped": capped,
        "below_min_order": (qty < min_amt) or (notional < min_cost),
        "stop_q05": nums["q05"],
        "stop_loss_usd_at_q05": notional * (1 - nums["q05"] / price),
        "liq_buffer_note": (
            (
                f"at {lev:.1f}x a {100 / lev:.0f}% adverse move liquidates "
                f"(calibrated 5% tail {nums['es5_pct']:+.1f}%)"
            )
            if lev > 1
            else f"no leverage needed ({lev:.2f}x of account)"
        ),
    }


# ----------------------------- sealed-log scorecard ---------------------------


def score_log(out_dir: Path) -> dict[str, Any]:
    """Score every sealed row whose target bar has closed."""
    log = out_dir / "log.jsonl"
    if not log.exists():
        return {"n_resolved": 0, "n_open": 0, "rows": []}
    now = pd.Timestamp.now("UTC")
    rows: list[dict] = []
    n_open = 0
    cache: dict[str, pd.DataFrame] = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if pd.Timestamp(d["target_close_utc"]) > now:
            n_open += 1
            continue
        sym = d["symbol"]
        if sym not in cache:
            cache[sym] = load_returns(sym).set_index("ts")
        df = cache[sym]
        a = pd.Timestamp(d["asof"], tz="UTC")
        tgt = pd.Timestamp(d["target_date"], tz="UTC")
        if tgt not in df.index or a not in df.index:
            n_open += 1
            continue
        y = float(np.log(df.loc[tgt, "close"] / df.loc[a, "close"]))
        pit = float(
            np.interp(
                y, d["calibrated_quantiles"], d["quantile_levels"], left=0, right=1
            )
        )
        rows.append(
            {
                "asof": d["asof"],
                "symbol": sym,
                "h": d["horizon_days"],
                "realised_pct": (np.exp(y) - 1) * 100,
                "median_pct": d["median_pct"],
                "pit": pit,
                "below_q05": y < _q(d, 0.05),
                "above_q95": y > _q(d, 0.95),
            }
        )
    summary: dict[str, Any] = {"n_resolved": len(rows), "n_open": n_open, "rows": rows}
    if rows:
        pits = np.array([r["pit"] for r in rows])
        summary["hit_q05"] = float(np.mean([r["below_q05"] for r in rows]))
        summary["hit_q95"] = float(np.mean([r["above_q95"] for r in rows]))
        summary["pit_mean"] = float(pits.mean())
        summary["pit_quartiles"] = [
            float(np.mean((pits >= lo) & (pits < lo + 0.25)))
            for lo in (0, 0.25, 0.5, 0.75)
        ]
    return summary


# ----------------------------- render -----------------------------------------


def _f(x: float) -> str:
    return f"{x:,.2f}" if x >= 10 else f"{x:,.4f}"


def build_brief(out_dir: Path, cfg: TradingConfig) -> str:
    latest = json.loads((out_dir / "latest.json").read_text())
    recs = {(r["symbol"], r["horizon_days"]): r for r in latest["forecasts"]}
    generated = pd.Timestamp(latest["generated"])
    L: list[str] = []
    L.append("# Trading brief (private)\n")
    L.append(
        f"forecasts generated {generated.strftime('%Y-%m-%d %H:%M UTC')} · "
        f"brief built {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M UTC')} · "
        f"venue {cfg.venue} {cfg.instrument} · account ${cfg.account_usd:,.0f} · "
        f"risk/trade {cfg.risk_per_trade_pct}% (${cfg.account_usd * cfg.risk_per_trade_pct / 100:.2f}) · "
        f"max leverage {cfg.max_leverage}x\n"
    )
    stale = [r for r in latest["forecasts"] if r.get("stale")]
    if stale:
        L.append(
            "**WARNING: forecasts are STALE** (anchor is not the latest closed bar). Re-run `wbtc forecast-all`.\n"
        )

    # scorecard first: how has the model been doing on what it sealed
    sc = score_log(out_dir)
    L.append("## Sealed-log scorecard\n")
    if sc["n_resolved"] == 0:
        L.append(
            f"No resolved forecasts yet ({sc['n_open']} open). The log starts paying off after the first targets close.\n"
        )
    else:
        L.append(
            f"{sc['n_resolved']} resolved, {sc['n_open']} open. Breach rate below q05: {sc['hit_q05']:.3f} "
            f"(nominal 0.05); above q95: {sc['hit_q95']:.3f} (nominal 0.05). PIT quartile shares "
            f"{[round(x, 2) for x in sc['pit_quartiles']]} (uniform = 0.25 each).\n"
        )
        recent = sc["rows"][-10:]
        L.append("| asof | asset | h | realised | median | PIT |")
        L.append("|---|---|---:|---:|---:|---:|")
        for r in recent:
            L.append(
                f"| {r['asof']} | {r['symbol']} | {r['h']} | {r['realised_pct']:+.2f}% | {r['median_pct']:+.2f}% | {r['pit']:.2f} |"
            )
        L.append("")

    for sym in cfg.symbols:
        any_rec = next((recs[(sym, h)] for h in cfg.horizons if (sym, h) in recs), None)
        if any_rec is None:
            continue
        asof = any_rec["asof"]
        mc = market_context(sym, asof)
        bg = bitget_context(sym)
        an = analogues(sym, asof, cfg.horizons)
        L.append(
            f"## {sym} — close {_f(any_rec['last_close'])} (bar {asof}, closed {any_rec['asof_close_utc'][:16]} UTC)\n"
        )
        L.append(
            f"**Market:** 5d {mc['ret_5d']:+.1f}%, 21d {mc['ret_21d']:+.1f}%, 60d {mc['ret_60d']:+.1f}%; "
            f"{mc['from_60d_high']:+.1f}% from 60d high, {mc['from_60d_low']:+.1f}% from 60d low. "
            f"Realised vol {mc['vol20_pct_day']:.2f}%/day = {mc['vol20_pctile_1y']:.0%} percentile of the last year "
            f"(was {mc['vol20_60d_ago']:.2f}% 60d ago). Last 7 days: {mc['last7_pct']}.\n"
        )
        if "error" in bg:
            L.append(f"**Bitget:** unavailable ({bg['error']}).\n")
        else:
            fr = bg["funding_rate_8h"]
            L.append(
                f"**Bitget {bg['symbol']}:** last {bg['mark_or_last']}, funding {fr * 100:+.4f}%/8h "
                f"({bg['funding_annualised_pct']:+.1f}% annualised, next {bg['next_funding_utc'][:16]} UTC), "
                f"min order {bg['min_amount']} / ${bg['min_cost_usd']}, max leverage {bg['max_leverage']}x.\n"
            )
        L.append(
            f"**Analogues (60d-return quintile {an['quintile']}/5, n={an['n_episodes']} non-overlapping episodes):** "
            + "; ".join(
                f"next {h}d median {v['median']:+.1f}%, P(<0) {v['p_neg']:.2f}, 10–90% range {v['p10']:+.1f}…{v['p90']:+.1f}%"
                for h, v in an["by_h"].items()
            )
            + ".\n"
        )
        L.append(
            "| h | settles | median | P(<0) | P(<−5%) | P(<−10%) | P(>+5%) | P(>+10%) | ES 5% | q05 (base) | q25 | q75 | q95 (base) | base breach q05/q95 | regime |"
        )
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|")
        sizes = []
        for h in cfg.horizons:
            rec = recs.get((sym, h))
            if rec is None:
                continue
            nm = forecast_numbers(rec)
            cv = rec["base_coverage_recent"]
            br = (
                f"{float(cv.get('0.05', float('nan'))):.3f} / "
                f"{1 - float(cv.get('0.95', float('nan'))):.3f}"
            )
            reg = (
                ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in rec["regime"].items()
                )
                or "—"
            )
            L.append(
                f"| {h} | {rec['target_close_utc'][:10]} | {nm['median_pct']:+.2f}% | {nm['p_down']:.2f} | {nm['p_down_5']:.2f} | "
                f"{nm['p_down_10']:.2f} | {nm['p_up_5']:.2f} | {nm['p_up_10']:.2f} | {nm['es5_pct']:+.1f}% | "
                f"{_f(nm['q05'])} ({_f(nm['q05_base'])}) | {_f(nm['q25'])} | {_f(nm['q75'])} | {_f(nm['q95'])} ({_f(nm['q95_base'])}) | {br} | {reg} |"
            )
            sizes.append((h, nm, sizing(rec, nm, cfg, bg)))
        L.append(
            f"\nBase breach = share of the last {any_rec['conformal']['calib_window']} outcomes "
            "below the base q05 / above the base q95 (nominal 0.05 / 0.05). Far above nominal = "
            "the base was over-confident; trust the conformal (unbracketed) levels.\n"
        )
        L.append(
            f"**Sizing at {cfg.risk_per_trade_pct}% risk (${cfg.account_usd * cfg.risk_per_trade_pct / 100:.2f}) per 5%-tail event, long or short:**\n"
        )
        L.append(
            "| h | notional | qty | leverage | stop at q05 | loss at stop | note |"
        )
        L.append("|---:|---:|---:|---:|---:|---:|---|")
        for h, nm, sz in sizes:
            note = []
            if sz["capped"]:
                note.append(f"capped at {cfg.max_leverage}x")
            if sz["below_min_order"]:
                note.append("BELOW Bitget min order")
            note.append(sz["liq_buffer_note"])
            L.append(
                f"| {h} | ${sz['notional_usd']:.2f} | {sz['qty']:.5f} | {sz['leverage']:.2f}x | {_f(sz['stop_q05'])} | "
                f"${sz['stop_loss_usd_at_q05']:.2f} | {'; '.join(note)} |"
            )
        L.append("")
    L.append("## How to read this\n")
    L.append(
        "* The forecaster has no directional edge by design; medians near zero are the honest answer. "
        "Its value is the calibrated width and tails. Analogues give a weak prior on sign from the asset's own history.\n"
        "* Sizing turns the 5% expected shortfall into a notional so that one bad-tail outcome costs exactly the risk budget. "
        "A stop at q05 is hit one time in twenty by construction; a stop inside the 25–75% band is noise.\n"
        "* Funding is paid every 8h on the notional: at 5x it is a material drag on multi-week holds; check the sign.\n"
        "* Everything here is a distribution, not advice. The decision and its reasoning go into the journal.\n"
    )
    return "\n".join(L) + "\n"
