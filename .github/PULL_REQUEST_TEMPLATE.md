<!--
Thanks for contributing! A few things to mention so the review is fast:

- If you changed a forecaster, please re-run `uv run wbtc backtest-long`
  and link the resulting numbers (or paste the headline table).
- If you changed defaults (e.g. `window=90, lookback=20`), please re-run
  `uv run wbtc sweep` and update `docs/RESULTS_LONG.md`.
- New invariants need failing tests first (`tests/test_quantiles.py`).
- See `CONTRIBUTING.md` for the full conventions.
-->

## Summary

<!-- What does this PR do, in one or two sentences? -->

## Type of change

- [ ] Bug fix (does not change the public API)
- [ ] New forecaster (follows `fit(returns) → predict(h, u)` protocol)
- [ ] Methodological change (alters CRPS numbers — please link backtest output)
- [ ] Documentation / tests / tooling only
- [ ] Other:

## Falsification / evidence

<!--
If the change affects forecasting numbers, link the runs:
- backtest-long output:
- hyperparam-sweep output (if defaults changed):
- new tests / failing-then-passing demonstration:
-->

## Checklist

- [ ] Tests pass locally (`uv run wbtc test`)
- [ ] New behaviour has tests
- [ ] Walk-forward protocol preserved (no look-ahead)
- [ ] Docs updated if user-facing behaviour changed
- [ ] CHANGELOG entry added under `[Unreleased]`
