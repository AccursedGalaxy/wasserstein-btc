"""Shared reporting primitives used by the long-horizon and extended-baseline
scripts (and the v0.3 summariser).

Kept deliberately small: each helper is a thin, composable utility, not a
monolithic "render the whole report" function. The three scripts that
generate `docs/RESULTS_LONG.md`, `docs/RESULTS_EXTENDED.md`, and the v0.3
summary diverge in section structure and headline-row construction, so the
right seam is the cell-formatting / per-figure level, not the document level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["slug", "fmt_pct_diff", "fmt_markdown", "plot_cumulative_crps"]


def slug(symbol: str) -> str:
    """Slugified symbol for filenames: ``"BTC/USDT"`` -> ``"btcusdt"``."""
    return symbol.lower().replace("/", "")


def fmt_pct_diff(a: float, b: float) -> str:
    """Signed percentage difference of ``a`` relative to ``b``.

    Returns ``"—"`` when ``b == 0`` (avoids div-by-zero in headline tables).
    """
    if b == 0:
        return "—"
    return f"{(a - b) / b * 100:+.1f}%"


def fmt_markdown(
    df: pd.DataFrame,
    *,
    float_fmt: str = "{:.6f}",
    index: bool = True,
) -> str:
    """DataFrame -> markdown table with consistent float formatting.

    Wraps the ``df.map(lambda x: f"{x:.6f}" if isinstance(x, float) else x)
    .to_markdown()`` pattern used in every backtest-report section. Integer
    cells (e.g. row counts) are preserved verbatim because ``isinstance(int,
    float)`` is False.
    """
    formatted = df.map(lambda x: float_fmt.format(x) if isinstance(x, float) else x)
    return formatted.to_markdown(index=index)


def plot_cumulative_crps(
    aligned_losses: dict[str, np.ndarray],
    png_path: Path,
    *,
    title: str,
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 110,
) -> None:
    """Save the canonical cumulative-CRPS plot for a long-horizon result.

    One line per method, oldest-step on the left, legend top-left, 2 columns.
    Identical visual style to the figures embedded in both ``RESULTS_LONG.md``
    and ``RESULTS_EXTENDED.md`` so a reader scanning across documents sees a
    consistent treatment.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps = len(next(iter(aligned_losses.values())))
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(n_steps)
    for name, losses in aligned_losses.items():
        ax.plot(xs, np.cumsum(losses), label=name, lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative CRPS")
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
