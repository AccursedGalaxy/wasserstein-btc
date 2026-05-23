# Docs index

A one-paragraph orientation to every project document, including
repo-root files like `ROADMAP.md` and `CONTRIBUTING.md`. Read these in
the order shown for a coherent first pass.

## [THEORY.md](THEORY.md) — start here

The mathematical core. Defines the 1D Wasserstein-2 manifold of probability
measures, the quantile-function isometry that makes geodesics straight, the
tangent-space regression that produces forecasts, the curvature gate (h=1
only), the Theil-Sen robust slope (h ≥ 5), and the v0.3 additions:
recency-weighted slope (`WGeo-EWMA`, §2.6), GARCH-conditioned dispersion
(`WGeo-Hetero`, §2.7), and the regime-aware mixture (`WGeo-GARCH-Ens`,
§2.8). The v0.4 additions are the quantile-space Wasserstein-2 barycentre
ensemble (`WGeo-Ensemble`, §2.9) and the variance-reduced residualised
Diebold-Mariano test (§2.10, a Giacomini-White-style augmented test of the
same unconditional EPA null). Contains the **explicit falsification
criteria** (§4) that the backtests are written to test against. Current
version: v0.4.

## [RESEARCH_REPORT.md](RESEARCH_REPORT.md) — paper-style writeup (v0.3 + v0.4)

The research-paper-style report on the v0.3 extensions and the v0.4
additions (the W₂-barycentre ensemble and the residualised DM test):
motivation, mathematical formulation, experimental setup, headline
numbers, falsification verdicts, regime decomposition, robustness checks,
and limitations. Read this once the THEORY.md sections are familiar.

## [RESULTS_LONG.md](RESULTS_LONG.md) — the headline evidence

The 6.75-year, multi-asset (BTC + ETH + SOL + BNB), multi-regime
out-of-sample validation. ~2400 walk-forward steps per (asset, h)
spanning 2019-08 to 2026-05. Includes the TL;DR, headline table,
per-year tables, per-regime breakdowns, Diebold-Mariano significance
tests, the hyperparameter robustness sweep, and a sober verdict against
the THEORY §4 falsification criteria. **This is the document to cite for
numbers.**

## [RESULTS_EXTENDED.md](RESULTS_EXTENDED.md) — named-econometric comparator panel

The v0.4 extended-baselines report. Compares `WGeo-GARCH-Ens` (the v0.3
ensemble) against named methods from adjacent econometric families on
BTC at h ∈ {1, 5, 21}: HAR-RV (Corsi 2009), CAViaR-SAV (Engle-Manganelli
2004), 2-state Markov-switching Normal (Hamilton 1989), FIGARCH(1,d,0)
(Baillie-Bollerslev-Mikkelsen 1996), AR(1) Stochastic Volatility
(Taylor 1982 / Harvey-Ruiz-Shephard 1994 via Kalman QML), and a
bivariate VAR+GARCH using BTC + ETH jointly. Restricted to BTC so the
heavier per-step fits (FIGARCH MLE, CAViaR per-quantile QR) complete in
tractable time. Same walk-forward / CRPS protocol as `RESULTS_LONG.md`.

## [RESULTS.md](RESULTS.md) — superseded but kept

The original v0.1 365-day-holdout backtest. Shows -11.7% CRPS vs GARCH-N at
h=21 on a single test year (2025-2026), DM p=0.006. Kept for the Christoffersen
quantile-coverage table; the headline numbers should be read against the
larger and more honest `RESULTS_LONG.md` instead.

## Repo-root files

- [`../README.md`](../README.md) — public-facing project description,
  install, quick start, headline result. Read this first if you came
  from GitHub.
- [`../ROADMAP.md`](../ROADMAP.md) — what's missing for the project to
  be production-competitive (HAR-RV / FIGARCH / multivariate /
  intraday). The v0.4 and v0.5 buckets are the prioritised work list.
- [`../CHANGELOG.md`](../CHANGELOG.md) — v0.1 → v0.2 → v0.3 → v0.4 release
  notes including the honest-negative-findings sections.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — the conventions PRs must
  follow (walk-forward, strictly-proper scoring, no DL, no
  sklearn-pipeline refactors).
- [`../CITATION.cff`](../CITATION.cff) — structured citation metadata
  for academic use.
- [`../CLAUDE.md`](../CLAUDE.md) — agent-facing repo guide (codebase
  map, common tasks, gotchas). Aimed at LLM agents working on the
  codebase, but humans will find it useful too.

## How to cite

When pointing to a number, prefer `RESULTS_LONG.md` (long-horizon evidence
over the full multi-year span). When pointing to a *concept*, prefer
`THEORY.md`. When pointing to coverage / calibration evidence, the table in
`RESULTS.md` is still the canonical source. For academic citation, see
[`../CITATION.cff`](../CITATION.cff).

## How to extend

Adding a new forecaster, asset, or evaluation regime: see
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) ("Adding a new forecaster")
and the protocol comments at the top of `src/wbtc/forecasters.py`. New
evidence should be appended to `RESULTS_LONG.md` (don't rewrite — add a
section), and the falsification table in the verdict section should be
re-rendered.
