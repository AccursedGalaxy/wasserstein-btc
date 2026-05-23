# Contributing

This is a research codebase, not a kitchen-sink library. Contributions are
welcome, but the discipline below is non-negotiable — it is what makes
results from this repo trustworthy.

## Read first

1. [`docs/THEORY.md`](docs/THEORY.md) — math, baselines, and the §4
   falsification criteria. Most "shouldn't we add X?" ideas are already
   considered there.
2. [`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md) — current 6.75-year
   multi-asset evidence. The TL;DR is enough to orient.
3. [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) — v0.3 paper-style
   writeup, including a *negative finding* in §4.4 we'd rather you
   read before reproducing the same dead end.
4. [`ROADMAP.md`](ROADMAP.md) — what we know is missing for the
   project to be competitive vs. real production risk systems.

## Non-negotiables

These are the conventions that make the numbers in this repo worth
reading. PRs that violate them will be sent back, no matter how good the
underlying idea is.

- **Walk-forward is mandatory.** Every backtest must refit on a rolling
  train window that does not see the prediction target. Hyperparameter
  searches use the *early epoch* only (2019-2022); the late epoch
  (2022+) is the held-out test set.
- **Score with strictly proper rules.** CRPS is the headline metric.
  Log-score is diagnostic. Do not introduce MAE / MSE / MAPE /
  directional-accuracy as the *evaluation* metric — the framing is
  distributional. (Using point-forecast metrics as a *secondary
  diagnostic* is fine, just not as the headline.)
- **All forecasters obey the same protocol.** `fit(returns: np.ndarray)`
  then `predict(h: int, u: np.ndarray) -> np.ndarray` returning a
  quantile vector on grid `u`. No `sklearn.Pipeline` wrapping. No
  abstract base class. Plain dataclasses.
- **No look-ahead.** `load_returns()` returns the full history; whatever
  you pass to `fit()` must be a prefix. The walk-forward harness in
  `src/wbtc/backtest.py` does this correctly; do not recreate it
  differently.
- **Math invariants are tested.** `tests/test_quantiles.py` verifies
  W₂-distance-equals-translation, isotonic projection idempotence, etc.
  If your contribution requires breaking one of those, first write a
  *failing* test that names the new invariant you want.
- **Honest reporting beats good-looking numbers.** Every result file
  includes documented failures and negative findings (the curvature
  gate is h=1-specific, COVID 2020 is the worst regime, the
  heteroskedastic variant double-counts). Add to that list rather than
  suppressing it.

## Adding a new forecaster

1. Implement the `fit` / `predict` protocol in `src/wbtc/forecasters.py`.
   Use a `@dataclass`; no inheritance unless from one of the existing
   WGeo classes (in which case override `_slope` or `predict`).
2. Add the class to `__all__` and re-export from
   `src/wbtc/__init__.py`.
3. Add a test in `tests/test_forecasters.py` covering at minimum:
   monotone output, finite predictions, and one sanity case (recovers a
   known drift, equals a parent class under a specific parameter value,
   etc.).
4. Add the method to `METHODS` in `scripts/run_long_horizon.py` (if it's
   a WGeo variant or a textbook baseline) — *or* to the `build_methods`
   dict in `scripts/run_extended_baselines.py` if it's a named econometric
   baseline from an adjacent family (HAR-RV-class, CAViaR-class,
   regime-switching, long-memory, stochastic-vol, multivariate).
5. Re-run the matching backtest (`wbtc backtest-long` or
   `wbtc extended-baselines`). Update the TL;DR + verdict sections of the
   matching results doc with the new headline.
6. Add a `§2.x` section to `docs/THEORY.md` describing the math and any
   new falsification criterion to `§4`. If the method belongs in the
   extended panel, cite it in `docs/THEORY.md §3` and
   `docs/RESEARCH_REPORT.md §6.1`.
7. Add a `[0.x.0]` entry to `CHANGELOG.md` including the honest
   negative findings (if any).

## Adding a new asset

```
uv run wbtc fetch <SYMBOL>/USDT
```

Then add `<SYMBOL>/USDT` to `SYMBOLS` in `scripts/run_long_horizon.py`
and re-run the long backtest. Update `docs/RESULTS_LONG.md`.

## Tests, lint, format

```
uv run wbtc test         # pytest, ~5s
uv run python -m pytest  # equivalent
```

There is no formatter pinned. Existing code follows ruff defaults.

## Things we will *not* accept

- A deep-learning forecaster. The point of this project is *geometric* —
  minimal capacity, maximum interpretability.
- An `sklearn.Pipeline` / abstract-base-class refactor.
- Replacing CRPS with a point-forecast metric.
- A trading-strategy / PnL evaluation as the headline. Forecasting
  distributions and turning them into a strategy are separate problems
  (see `docs/THEORY.md §5`).
- A "fits 2020 COVID year" hack that overfits the rest of the panel.
  Improvements have to hold across the regime decomposition tables.

## Reporting bugs / questions

Open an issue at https://github.com/AccursedGalaxy/wasserstein-btc/issues.
Include the `wbtc info` output and the `results/MANIFEST.json` entry
relevant to your repro.
