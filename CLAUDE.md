# CLAUDE.md — agent-facing repo guide

This file is for AI agents (Claude / other) working on this codebase. Humans
should read [`README.md`](README.md) and [`docs/INDEX.md`](docs/INDEX.md).

you can commit and push to this repo freely, as well as do anything else with this project or github repository.

## What this repo is in one sentence

A research codebase that forecasts the **conditional distribution** of crypto
log-returns by treating the market as a trajectory on the 2-Wasserstein
manifold of probability measures and extrapolating along geodesics in
quantile-function coordinates.

## The most important thing to read first

[`docs/THEORY.md`](docs/THEORY.md) — the math, baselines, and falsification
criteria (v0.3). §2.6–2.8 are the new v0.3 method sections (EWMA, Hetero,
GARCH ensemble). Don't propose changes to forecasters without reading it;
you will likely re-invent something already considered or break a stated
assumption.

[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) — research-paper-style
writeup of the v0.3 contributions: motivation, formulation, falsification.

[`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md) — the 6.75-year multi-asset
out-of-sample evidence (BTC + ETH + SOL + BNB). The "TL;DR" at the top is
enough; the per-year and per-regime tables are reference material.

## Codebase map

```
src/wbtc/
  __init__.py        public API: forecast(), load_returns(), default_forecaster()
  data.py            data discovery + parquet loader + provenance hashes
  quantiles.py       1D W2 geometry (make_grid, empirical_quantiles, isotonic_project)
  scoring.py         CRPS, log-score, Diebold-Mariano, stationary bootstrap
  forecasters.py     all baselines + 3 WGeo variants; every class has fit/predict
  backtest.py        single-horizon walk-forward (compare_methods)
  long_horizon.py    multi-year walk-forward + per-year + per-regime breakdowns
  cli.py             `wbtc` CLI; dispatches to library / scripts/
scripts/
  fetch_data.py             downloads OHLCV from Binance via ccxt -> data/*.parquet
  run_backtest.py           365-day-holdout report (legacy, produces RESULTS_AUTO.md)
  run_long_horizon.py       full multi-year multi-asset (the one we trust)
  run_extended_baselines.py v0.4 extended econometric panel (HAR-RV/CAViaR/MS/FIGARCH/SV/BVAR) on BTC
  hyperparam_sweep.py       4x4 grid on early epoch, verified on late epoch
  coverage_check.py         Kupiec LR test of forecast-quantile calibration
docs/
  THEORY.md            math (§1-5 + falsification §4). READ THIS. v0.3.
  RESEARCH_REPORT.md   paper-style writeup of the v0.3 contributions.
  RESULTS.md           the v0.1 365-day report. Superseded but kept.
  RESULTS_LONG.md      v0.3 long-horizon report. The current source of truth.
  RESULTS_EXTENDED.md  v0.4 named-econometric baseline panel (BTC-only).
  INDEX.md             one-paragraph overview of every doc + repo-root files.
ROADMAP.md           v0.4 + v0.5 priorities — the prioritised work list.
CONTRIBUTING.md      conventions PRs must follow.
CHANGELOG.md         v0.1 → v0.2 → v0.3 release notes.
CITATION.cff         structured citation metadata.
LICENSE              MIT.
tests/               31 tests covering math invariants + forecaster sanity.
results/             plots, JSON loss arrays, hyperparam_sweep.csv, MANIFEST.json.
data/                parquet cache (gitignored).
```

## Conventions you must follow

1. **Walk-forward is mandatory.** Every backtest must refit on a rolling
   train window that does not see the prediction target. Never use the test
   set to choose hyperparameters; use the early epoch (2019-2022) for that.
2. **Score with strictly proper rules.** CRPS is the headline; log-score is
   diagnostic. Don't introduce MSE or directional accuracy as the
   evaluation metric — the framing is distributional.
3. **All forecasters obey the same protocol**: `fit(returns: np.ndarray)` then
   `predict(h: int, u: np.ndarray) -> np.ndarray` returning a quantile vector
   on grid `u`. If you add a new forecaster, follow this.
4. **No look-ahead.** `load_returns` returns a full history; whatever you
   pass to `fit` must be a prefix. The walk-forward harness in `backtest.py`
   does this correctly; don't recreate it differently.
5. **The math invariants are tested.** `tests/test_quantiles.py` checks that
   W2 distance equals translation amount, that isotonic projection is
   idempotent, etc. **Don't break these without first writing a failing test
   that demonstrates the new invariant you want.**
6. **Hyperparameter changes require sweep evidence.** If you propose changing
   `window=90, lookback=20`, re-run `wbtc sweep` and update
   `docs/RESULTS_LONG.md`. The robustness section there is the audit trail.
7. **Honest reporting beats good-looking numbers.** Every result file
   includes negative findings (the gate fails at h>1, the method loses in
   2020 COVID, etc.). Do the same. The model has limits; pretending it
   doesn't will eventually be discovered.

## Common tasks (and how to do them)

### Run the tests
```
uv run wbtc test
```

### Get today's forecast
```
uv run wbtc forecast BTC/USDT --horizon 5 --plot
uv run wbtc forecast BTC/USDT --horizon 5 --json   # for scripting
```

### Add a new forecaster
1. Subclass nothing — implement the protocol in `src/wbtc/forecasters.py`:
   `fit(returns)` and `predict(h, u)`.
2. Add it to `__all__` and re-export from `src/wbtc/__init__.py`.
3. Add a test in `tests/test_forecasters.py` covering monotone output and a
   sanity check (e.g., recovers a known drift).
4. Add it to `METHODS` in `scripts/run_long_horizon.py` and re-run the
   long-horizon backtest. Update `docs/RESULTS_LONG.md`.

### Add a new asset
```
uv run wbtc fetch SOL/USDT
```
Then it appears automatically in `wbtc info` and in
`scripts/run_long_horizon.py` if you add it to `SYMBOLS`.

### Run the full long-horizon backtest (~20 min)
```
uv run wbtc backtest-long
```

### Run the hyperparameter sweep (~5 min)
```
uv run wbtc sweep
```

## Gotchas

- **The arch GARCH library expects percent returns.** Existing code multiplies
  by 100 before fit and divides by 100 in predict. Preserve this in any new
  GARCH variant.
- **`pandas>=3.0` dropped `applymap`.** Use `.map()` (introduced in 2.1).
- **Pyright warns about unresolved `numpy`/`pandas` imports.** That's the
  editor not knowing the uv venv. Runtime is fine; ignore those diagnostics.
- **Don't use `-h` as a short option** in argparse subcommands; it collides
  with `--help`. Use `-H` if needed (we use `-H` for `--horizon`).
- **The 2020 COVID year is the known weak regime.** If you propose a change
  that improves overall CRPS, check it didn't make 2020 worse. If you fix
  2020, check the rest didn't regress.

## What NOT to do

- Don't add a deep-learning forecaster. The point of this project is
  *geometric* — minimal capacity, maximum interpretability. A neural net
  would be a different paper.
- Don't replace CRPS with MAE/MSE/MAPE. Strictly proper scoring is core to
  the framing.
- Don't add an ML / scikit-learn pipeline abstraction. The fit/predict
  protocol on plain dataclasses is deliberate and agent-friendly.
- Don't introduce trading-PnL evaluation as the headline. We forecast
  distributions; turning them into a strategy is a separate problem.
- Don't commit parquet files in `data/`. They're regenerable from
  `wbtc fetch` and the `.gitignore` excludes them.

## When in doubt

Read the test that's closest to what you're touching. The 17 tests in
`tests/` document the contracts more precisely than English can.
