"""Gate 1 — Intraday-κ trajectory-structure test.

Frozen per docs/PREREG.md §Gate 1 at tag prereg-v1.0. Run after the BTC
5-min density build, BEFORE implementing any forecaster.

No CLI flags. Reads ``data/btcusdt_intraday_density.parquet``. Writes
``results/gate_1_intraday_kappa.json`` and
``results/gate_1_intraday_kappa.png``.

Exit code: 0 = PASS, 1 = FAIL, 2 = setup error (data missing or
malformed).

Decision rule: PASS iff 1A-shuffle AND 1A-block AND 1B-events all pass.

If FAIL, the intraday-density track is abandoned and the daily-rolling-
window track becomes the paper headline. See docs/PREREG.md §Gate 1
"If Gate 1 fails" for the followup contract.

Constants below match docs/PREREG.md verbatim. Editing them requires
bumping PREREG.md to v1.1 with diff and dated rationale.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "btcusdt_intraday_density.parquet"
OUT_JSON = ROOT / "results" / "gate_1_intraday_kappa.json"
OUT_PNG = ROOT / "results" / "gate_1_intraday_kappa.png"

# --- Frozen constants (PREREG.md §Gate 1; do not edit without bumping pre-reg) ---
EARLY_EPOCH_END = date(2023, 12, 31)
TAU = 2  # tangent spacing in trading days
N_SURROGATE = 200
BLOCK_MEAN_LENGTH = 10  # Politis-Romano stationary bootstrap, mean block in days
SURROGATE_RATIO_THRESHOLD = 1.25  # real κ p99 / surrogate-median κ p99
EVENT_DAYS = [date(2020, 3, 12), date(2021, 5, 19), date(2022, 11, 8)]
EVENT_WINDOW_DAYS = 3  # [event, event+1, event+2]
EVENT_PERCENTILE = 95  # real κ percentile a window must exceed to "hit"
EVENT_HITS_REQUIRED = 2  # of 3 events
SHUFFLE_SEED_BASE = 0
BLOCK_SEED_BASE = 1000
MIN_USABLE_DAYS = 730  # below this, the early-epoch sample is too thin to test


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_kappa(Q: np.ndarray, tau: int) -> np.ndarray:
    """κ_t = 1 - cos(v1, v2) for v1 = Q[t]-Q[t-τ], v2 = Q[t-τ]-Q[t-2τ].

    Q has shape (T, K). Returns length-(T - 2τ) array; index 0 corresponds
    to t = 2τ in the input.
    """
    v1 = Q[2 * tau :] - Q[tau:-tau]
    v2 = Q[tau:-tau] - Q[: -2 * tau]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    safe = (n1 > 1e-12) & (n2 > 1e-12)
    cos = np.zeros_like(n1)
    cos[safe] = np.einsum("ij,ij->i", v1[safe], v2[safe]) / (n1[safe] * n2[safe])
    return 1.0 - cos


def shuffle_surrogate_kappa(
    Q: np.ndarray, tau: int, rng: np.random.Generator
) -> np.ndarray:
    perm = rng.permutation(Q.shape[0])
    return compute_kappa(Q[perm], tau)


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap. Geometric block lengths with
    mean ``mean_block``, circular wrap. Returns length-n index array.
    """
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    t = 0
    while t < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        end = min(t + block_len, n)
        for k in range(end - t):
            idx[t + k] = (start + k) % n
        t = end
    return idx


def block_surrogate_kappa(
    Q: np.ndarray, tau: int, rng: np.random.Generator
) -> np.ndarray:
    idx = stationary_bootstrap_indices(Q.shape[0], BLOCK_MEAN_LENGTH, rng)
    return compute_kappa(Q[idx], tau)


def event_hits(kappa: np.ndarray, days: np.ndarray, threshold: float) -> dict:
    """``days`` is the day-index array parallel to ``kappa`` (already aligned)."""
    out: dict = {}
    total = 0
    for ev in EVENT_DAYS:
        ev64 = np.datetime64(ev)
        end64 = ev64 + np.timedelta64(EVENT_WINDOW_DAYS - 1, "D")
        in_window = (days >= ev64) & (days <= end64)
        if not in_window.any():
            out[str(ev)] = {
                "hit": False,
                "max_kappa": None,
                "reason": "no data in window",
            }
            continue
        max_k = float(kappa[in_window].max())
        hit = max_k >= threshold
        out[str(ev)] = {"hit": bool(hit), "max_kappa": max_k}
        total += int(hit)
    out["total_hits"] = total
    out["pass"] = total >= EVENT_HITS_REQUIRED
    return out


def main() -> int:
    if not DATA.exists():
        print(
            f"ERROR: {DATA} not found. Build the intraday density first "
            f"(see docs/PREREG.md for the protocol-(i) data pipeline).",
            file=sys.stderr,
        )
        return 2

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    required_cols = {"day", "quantile", "is_excluded"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: input parquet missing columns: {sorted(missing)}", file=sys.stderr
        )
        return 2

    df = df[~df["is_excluded"].astype(bool)].copy()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df = df.sort_values("day").reset_index(drop=True)
    df = df[df["day"] <= EARLY_EPOCH_END].reset_index(drop=True)
    if len(df) < MIN_USABLE_DAYS:
        print(
            f"ERROR: only {len(df)} usable early-epoch days "
            f"(need >= {MIN_USABLE_DAYS}).",
            file=sys.stderr,
        )
        return 2

    try:
        Q = np.stack([np.asarray(q, dtype=float) for q in df["quantile"].to_numpy()])
    except ValueError as e:
        print(
            f"ERROR: quantile column not stackable (varying length?): {e}",
            file=sys.stderr,
        )
        return 2

    days = df["day"].to_numpy().astype("datetime64[D]")

    # ---- Real κ ----
    kappa_real = compute_kappa(Q, TAU)
    kappa_days = days[2 * TAU :]
    real_p95 = float(np.percentile(kappa_real, EVENT_PERCENTILE))
    real_p99 = float(np.percentile(kappa_real, 99))

    # ---- 1A — Day-shuffle surrogates ----
    shuffle_p99s = np.empty(N_SURROGATE)
    for i in range(N_SURROGATE):
        rng = np.random.default_rng(SHUFFLE_SEED_BASE + i)
        shuffle_p99s[i] = np.percentile(shuffle_surrogate_kappa(Q, TAU, rng), 99)
    shuffle_median_p99 = float(np.median(shuffle_p99s))
    ratio_shuffle = (
        real_p99 / shuffle_median_p99 if shuffle_median_p99 > 0 else float("inf")
    )
    pass_1a_shuffle = ratio_shuffle >= SURROGATE_RATIO_THRESHOLD

    # ---- 1A — Block-bootstrap surrogates ----
    block_p99s = np.empty(N_SURROGATE)
    for i in range(N_SURROGATE):
        rng = np.random.default_rng(BLOCK_SEED_BASE + i)
        block_p99s[i] = np.percentile(block_surrogate_kappa(Q, TAU, rng), 99)
    block_median_p99 = float(np.median(block_p99s))
    ratio_block = real_p99 / block_median_p99 if block_median_p99 > 0 else float("inf")
    pass_1a_block = ratio_block >= SURROGATE_RATIO_THRESHOLD

    # ---- 1B — Event-day elevation ----
    events_result = event_hits(kappa_real, kappa_days, real_p95)
    pass_1b = events_result["pass"]

    gate_pass = bool(pass_1a_shuffle and pass_1a_block and pass_1b)

    # ---- JSON output ----
    result = {
        "gate_1_pass": gate_pass,
        "data_sha256": sha256_file(DATA),
        "n_days_used": int(len(df)),
        "early_epoch_end": str(EARLY_EPOCH_END),
        "constants": {
            "tau": TAU,
            "n_surrogate": N_SURROGATE,
            "block_mean_length": BLOCK_MEAN_LENGTH,
            "surrogate_ratio_threshold": SURROGATE_RATIO_THRESHOLD,
            "event_percentile": EVENT_PERCENTILE,
            "event_hits_required": EVENT_HITS_REQUIRED,
            "event_window_days": EVENT_WINDOW_DAYS,
        },
        "real_kappa": {f"p{EVENT_PERCENTILE}": real_p95, "p99": real_p99},
        "1A_shuffle": {
            "surrogate_median_p99": shuffle_median_p99,
            "ratio": ratio_shuffle,
            "threshold": SURROGATE_RATIO_THRESHOLD,
            "pass": bool(pass_1a_shuffle),
        },
        "1A_block": {
            "surrogate_median_p99": block_median_p99,
            "ratio": ratio_block,
            "threshold": SURROGATE_RATIO_THRESHOLD,
            "pass": bool(pass_1a_block),
        },
        "1B_events": events_result,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    # ---- One-page diagnostic plot ----
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax = axes[0]
    ax.plot(kappa_days, kappa_real, lw=0.5, color="steelblue", label="real κ")
    ax.axhline(
        real_p95,
        color="orange",
        ls="--",
        lw=1,
        label=f"p{EVENT_PERCENTILE} = {real_p95:.3f}",
    )
    ax.axhline(real_p99, color="red", ls="--", lw=1, label=f"p99 = {real_p99:.3f}")
    ylim_top = max(real_p99 * 1.5, float(kappa_real.max()) * 1.02)
    ax.set_ylim(0, ylim_top)
    for ev in EVENT_DAYS:
        ax.axvline(np.datetime64(ev), color="black", ls=":", lw=1, alpha=0.6)
        ax.text(
            np.datetime64(ev),
            ylim_top * 0.95,
            str(ev),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    ax.set_title(
        f"Gate 1 — intraday-κ on BTC (early epoch through {EARLY_EPOCH_END}, "
        f"N={len(df)} days)"
    )
    ax.set_ylabel("κ = 1 − cos(v₁, v₂)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    lo = min(shuffle_p99s.min(), block_p99s.min(), real_p99) * 0.9
    hi = max(shuffle_p99s.max(), block_p99s.max(), real_p99) * 1.05
    bins = np.linspace(lo, hi, 40)
    ax.hist(
        shuffle_p99s,
        bins=bins,
        alpha=0.5,
        color="steelblue",
        label=f"shuffle p99 (median {shuffle_median_p99:.3f})",
    )
    ax.hist(
        block_p99s,
        bins=bins,
        alpha=0.5,
        color="seagreen",
        label=f"block-bootstrap p99 (median {block_median_p99:.3f})",
    )
    ax.axvline(real_p99, color="red", lw=2, label=f"real p99 = {real_p99:.3f}")
    ax.axvline(
        SURROGATE_RATIO_THRESHOLD * shuffle_median_p99,
        color="steelblue",
        ls=":",
        lw=1,
        label=f"{SURROGATE_RATIO_THRESHOLD}× shuffle median",
    )
    ax.axvline(
        SURROGATE_RATIO_THRESHOLD * block_median_p99,
        color="seagreen",
        ls=":",
        lw=1,
        label=f"{SURROGATE_RATIO_THRESHOLD}× block median",
    )
    ax.set_xlabel("κ 99th percentile")
    ax.set_ylabel(f"count (N={N_SURROGATE} surrogates)")
    ax.set_title(
        f"1A-shuffle ratio={ratio_shuffle:.2f} [{'PASS' if pass_1a_shuffle else 'FAIL'}] | "
        f"1A-block ratio={ratio_block:.2f} [{'PASS' if pass_1a_block else 'FAIL'}] | "
        f"1B-events {events_result['total_hits']}/3 [{'PASS' if pass_1b else 'FAIL'}] | "
        f"GATE: {'PASS' if gate_pass else 'FAIL'}"
    )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)

    # ---- Banner + exit code ----
    print("=" * 70)
    print(f"GATE 1 RESULT: {'PASS' if gate_pass else 'FAIL'}")
    print(
        f"  1A-shuffle:  real p99={real_p99:.3f}  surrogate median={shuffle_median_p99:.3f}  "
        f"ratio={ratio_shuffle:.3f}  (threshold {SURROGATE_RATIO_THRESHOLD})  "
        f"{'PASS' if pass_1a_shuffle else 'FAIL'}"
    )
    print(
        f"  1A-block:    real p99={real_p99:.3f}  surrogate median={block_median_p99:.3f}  "
        f"ratio={ratio_block:.3f}  (threshold {SURROGATE_RATIO_THRESHOLD})  "
        f"{'PASS' if pass_1a_block else 'FAIL'}"
    )
    print(
        f"  1B-events:   {events_result['total_hits']}/3 hits at p{EVENT_PERCENTILE}="
        f"{real_p95:.3f}  (need >= {EVENT_HITS_REQUIRED})  "
        f"{'PASS' if pass_1b else 'FAIL'}"
    )
    for ev in EVENT_DAYS:
        e = events_result[str(ev)]
        if "max_kappa" in e and e["max_kappa"] is not None:
            print(f"      {ev}: max κ in window = {e['max_kappa']:.3f}  hit={e['hit']}")
        else:
            print(f"      {ev}: {e.get('reason', 'no data')}")
    print(f"JSON: {OUT_JSON}")
    print(f"PNG:  {OUT_PNG}")
    print("=" * 70)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
