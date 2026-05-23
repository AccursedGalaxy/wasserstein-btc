# Docs index

A one-paragraph orientation to every document in `docs/`. Read these in the
order shown for a coherent first pass.

## [THEORY.md](THEORY.md) — start here

The mathematical core. Defines the 1D Wasserstein-2 manifold of probability
measures, the quantile-function isometry that makes geodesics straight, the
tangent-space regression that produces forecasts, the curvature gate (h=1
only), the Theil-Sen robust slope (h ≥ 5), and the v0.3 additions:
recency-weighted slope (`WGeo-EWMA`, §2.6), GARCH-conditioned dispersion
(`WGeo-Hetero`, §2.7), and the regime-aware mixture (`WGeo-GARCH-Ens`,
§2.8). Contains the **explicit falsification criteria** (§4) that the
backtests are written to test against. Current version: v0.3.

## [RESEARCH_REPORT.md](RESEARCH_REPORT.md) — v0.3 paper-style writeup

The research-paper-style report on the v0.3 extensions: motivation,
mathematical formulation, experimental setup, headline numbers,
falsification verdicts, regime decomposition, robustness checks, and
limitations. Read this once the THEORY.md sections are familiar.

## [RESULTS_LONG.md](RESULTS_LONG.md) — the headline evidence

The 6.75-year, multi-asset (BTC + ETH + SOL + BNB), multi-regime
out-of-sample validation. ~2400 walk-forward steps per (asset, h)
spanning 2019-08 to 2026-05. Includes the TL;DR, headline table,
per-year tables, per-regime breakdowns, Diebold-Mariano significance
tests, the hyperparameter robustness sweep, and a sober verdict against
the THEORY §4 falsification criteria. **This is the document to cite for
numbers.**

## [RESULTS.md](RESULTS.md) — superseded but kept

The original v0.1 365-day-holdout backtest. Shows -11.7% CRPS vs GARCH-N at
h=21 on a single test year (2025-2026), DM p=0.006. Kept for the Christoffersen
quantile-coverage table; the headline numbers should be read against the
larger and more honest `RESULTS_LONG.md` instead.

## How to cite

When pointing to a number, prefer `RESULTS_LONG.md` (long-horizon evidence
over the full multi-year span). When pointing to a *concept*, prefer
`THEORY.md`. When pointing to coverage / calibration evidence, the table in
`RESULTS.md` is still the canonical source.

## How to extend

Adding a new forecaster, asset, or evaluation regime: see [`../CLAUDE.md`](../CLAUDE.md)
("Common tasks" section) and the protocol comments at the top of
`src/wbtc/forecasters.py`. New evidence should be appended to
`RESULTS_LONG.md` (don't rewrite — add a section), and the falsification
table in the verdict section should be re-rendered.
