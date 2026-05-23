# Changelog

All notable changes to this project will be documented here. Dates ISO-8601.

## [0.2.0] — 2026-05-23

### Added

- **Multi-asset, multi-year long-horizon backtest** (`wbtc backtest-long` /
  `scripts/run_long_horizon.py`): 2470 walk-forward days, BTC + ETH, three
  horizons. Per-year + per-regime breakdowns in
  [`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md).
- **Hyperparameter robustness sweep** (`wbtc sweep` /
  `scripts/hyperparam_sweep.py`): 4×4 grid on early epoch (2019-2022),
  verified on held-out late epoch (2022-2026). CRPS surface flatness
  proves no overfitting.
- **`WassersteinGeodesicTheilSen`** — robust-slope variant of the proposed
  method. Replaces OLS slope with Theil-Sen median-of-pairwise-slopes
  (29.3% breakdown point). Matches or beats the curvature-gate variant at
  h ≥ 5 on long-horizon data, with one fewer hyperparameter.
- **`HistoricalSimulationBootstrap`** + **`GJRGarchStudentT`** baselines —
  tougher baseline panel including industry-standard HS-bootstrap and
  asymmetric GARCH.
- **`wbtc` CLI** (`src/wbtc/cli.py`): single command with subcommands
  `fetch`, `forecast`, `backtest`, `backtest-long`, `sweep`, `info`,
  `test`. Replaces the five separate script paths.
- **Public Python API** in `src/wbtc/__init__.py`: `forecast()`,
  `default_forecaster()`, `load_returns()`, `available_symbols()`,
  `data_info()`, `ForecastResult`.
- **Fan-chart visualisation** for `wbtc forecast --plot` (forecast quantile
  envelope vs recent realised price).
- **CLAUDE.md** — agent-facing codebase guide.
- **`docs/INDEX.md`** — one-paragraph index of every document.
- **GitHub Actions CI** running pytest on push/PR.

### Changed

- `THEORY.md` bumped to v0.2 with §2.5 documenting the Theil-Sen variant and
  noting that the curvature gate is now h=1-specific.
- `fetch_data.py` accepts multiple symbols on the CLI; defaults to
  BTC/ETH/SOL.
- `run_backtest.py` no longer clobbers the hand-edited interpretive
  `RESULTS.md`; writes to `RESULTS_AUTO.md` instead.

### Honest negative findings

- The novel **regime-curvature gate** only earns its keep at h=1; at h ≥ 5
  it is matched/beaten by the simpler Theil-Sen variant. The robust slope
  is now the recommended default for h ≥ 5.
- The proposed method **loses to naive at h=21 in 2020 (COVID crash)** by
  ~1%. Constant-velocity geodesic extrapolation cannot survive a
  once-in-a-decade discontinuity. Documented openly in
  `docs/RESULTS_LONG.md`.
- **GARCH wins in the high-vol regime** (~3% of days); WGeo wins in
  neutral/low-vol (62% of days). The methods are complementary, not
  strictly competing.

## [0.1.0] — 2026-05-23

### Added

- Initial release: tangent-space W2-geodesic forecaster (`WassersteinGeodesic`)
  with optional curvature-gate variant (`WassersteinGeodesicGated`).
- Baselines: `StaticEmpirical`, `RandomWalkDrift`, `GarchNormal`,
  `GarchStudentT`.
- Single-horizon walk-forward backtest with stationary-bootstrap CIs and
  Diebold-Mariano significance tests.
- Christoffersen quantile-coverage check (`scripts/coverage_check.py`).
- Initial 365-day-holdout backtest report in [`docs/RESULTS.md`](docs/RESULTS.md):
  -11.7% CRPS vs GARCH-N at h=21d (DM p=0.006), well-calibrated coverage
  across 20 quantile levels (Kupiec LR p > 0.12 everywhere).
- Mathematical theory + falsification criteria in [`docs/THEORY.md`](docs/THEORY.md).
- 14 unit tests (CRPS closed-form check, W2 distance == translation,
  isotonic projection idempotence, drift recovery).
