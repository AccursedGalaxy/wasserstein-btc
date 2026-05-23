"""Build the social-preview / OG image for the GitHub repo.

Pure post-hoc: reads existing per-step CRPS arrays from `results/long_*.json`,
computes the cumulative-CRPS lines for the methods we want to feature, and
renders a 1280x640 PNG suitable for GitHub's social preview slot
(Settings -> General -> Social preview).

Does NOT run any forecaster. Safe to run while a backtest is in progress.

Usage:
    uv run python scripts/build_social_preview.py
    # writes assets/social_preview.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "assets" / "social_preview.png"

# The cell with the largest published edge (ETH h=21, -3.16% vs best
# baseline) — the cleanest visual story for a social-preview thumbnail.
SOURCE_JSON = RESULTS_DIR / "long_ethusdt_h21.json"

# Methods to plot — keep tight so the chart reads at thumbnail size.
# v0.4 headline winner is WGeo-Ensemble (W2 barycentre of the v0.3 trio).
FEATURED = [
    ("WGeo-Ensemble", "#0b7285", 2.8, "-"),
    ("GARCH-N", "#c92a2a", 1.8, "--"),
    ("Static", "#868e96", 1.2, ":"),
]


def cumulative_mean(x: np.ndarray) -> np.ndarray:
    """Cumulative mean — visually monotonic, shows long-run edge clearly."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def main() -> None:
    if not SOURCE_JSON.exists():
        raise SystemExit(
            f"Missing {SOURCE_JSON}. Run `uv run wbtc backtest-long` first, "
            "or point SOURCE_JSON at an existing per-step CRPS file in results/."
        )
    payload = json.loads(SOURCE_JSON.read_text())
    # Top level is flat: {method_name: [per-step CRPS, ...]}.
    losses = {k: v for k, v in payload.items() if isinstance(v, list)}
    available = set(losses.keys())

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("white")

    for name, color, lw, ls in FEATURED:
        if name not in available:
            continue
        arr = np.asarray(losses[name], dtype=float)
        ax.plot(
            cumulative_mean(arr),
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=name,
        )

    ax.set_xlabel("Walk-forward day", fontsize=13)
    ax.set_ylabel("Cumulative mean CRPS (lower is better)", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.95)

    # Big headline overlay — top-left corner, generous whitespace
    fig.text(
        0.06,
        0.92,
        "wasserstein-btc",
        fontsize=28,
        weight="bold",
        color="#212529",
    )
    fig.text(
        0.06,
        0.86,
        "Distributional crypto forecasting on the 2-Wasserstein manifold",
        fontsize=15,
        color="#495057",
    )
    fig.text(
        0.06,
        0.82,
        "12 / 12 cells beat best classical baseline · 8 / 12 cells DM-significant (residualised, p<0.05) · 6.75y walk-forward",
        fontsize=11,
        color="#868e96",
    )
    fig.text(
        0.99,
        0.015,
        "Shown: ETH/USDT, h = 21d  ·  WGeo-Ensemble vs GARCH-N: −3.2% CRPS, residualised DM p < 0.0001  ·  github.com/AccursedGalaxy/wasserstein-btc",
        fontsize=10,
        color="#adb5bd",
        style="italic",
        ha="right",
    )

    plt.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.16)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=100, facecolor="white")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
