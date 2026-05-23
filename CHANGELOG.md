# Changelog

All notable changes to this project will be documented here. Dates ISO-8601.

## [Unreleased — v0.4 extended baselines]

### Added

- **Six named-econometric baselines** for the comparator panel, each
  implemented faithfully from scratch and tested on synthetic data:
  - `HARRV` — Heterogeneous Autoregressive of Realised Variance (Corsi
    2009) on daily-r² as RV proxy, daily/weekly/monthly aggregates,
    NNLS-constrained coefficients, Student-t innovations.
  - `CAViaRSAV` — Symmetric Absolute Value CAViaR (Engle-Manganelli
    2004) with an anchor-grid fit + interpolation for tractable
    per-step cost.
  - `MarkovSwitching2` — 2-state Markov-switching Normal (Hamilton
    1989, simplified) fit by full Hamilton EM; mixture-of-Gaussians
    h-step quantile inversion.
  - `FIGARCH` — FIGARCH(1, d, 0) (Baillie-Bollerslev-Mikkelsen 1996)
    via truncated ARCH-∞ representation, Gaussian QML, Student-t
    innovations.
  - `StochasticVolatilityAR1` — discrete-time SV with AR(1) log-
    variance (Taylor 1982 / Harvey-Ruiz-Shephard 1994) fit by Kalman
    quasi-likelihood on `log r²`.
  - `BivariateVARGarch` — bivariate VAR(1) on (BTC, ETH) for the mean
    + univariate GARCH(1,1) on BTC residuals. The exogenous ETH series
    is supplied at construction and aligned to walk-forward windows by
    tail-suffix match.
- **`scripts/run_extended_baselines.py`** + **`wbtc extended-baselines`**
  CLI command — runs the six new baselines plus `WGeo-GARCH-Ens` and
  `GARCH-t` (anchors) on BTC at h ∈ {1, 5, 21} with the same
  walk-forward harness, writes `docs/RESULTS_EXTENDED.md`.
- 6 new tests in `tests/test_forecasters.py` covering each forecaster
  (monotone quantiles, horizon-widening, regime recovery for MS,
  cross-coefficient learning for BVAR-GARCH). Suite: 37/37.

### Changed

- `default_forecaster(horizon)` is unchanged — the v0.4 baselines exist
  for comparison, not for replacement of the WGeo defaults.
- README "Honest limitations" updated to reference the new
  `RESULTS_EXTENDED.md` instead of stating the named-method gap.
- `docs/INDEX.md` and `ROADMAP.md` updated to reflect that the v0.4 #1
  baseline-coverage item is now landed.

## [0.3.0] — 2026-05-23

### Added

- **`WassersteinGeodesicEWMA`** — recency-weighted slope variant (weighted
  least squares with exponential decay `λ ∈ (0, 1]`). At `λ = 1` it
  collapses exactly to OLS WGeo (proved in
  `tests/test_forecasters.py::test_ewma_decay_one_matches_ols`). Wins
  ETH h=1, SOL h=21, BNB h=5 outright.
- **`WassersteinGeodesicHetero`** — replaces the `√h` dispersion factor
  with a GARCH(1,1)-implied conditional/unconditional vol ratio
  `s_h(t)`. Direction still by Theil-Sen; fallback `s_h = 1` on GARCH-fit
  failure (so weakly ≥ `WGeo-TheilSen` by construction).
- **`WGeoGarchEnsemble`** — continuous smoothstep mixture between
  `WGeo-TheilSen` and `GARCH-N`, weighted by trailing-year realised-vol
  percentile rank. Mixture in quantile-function coordinates is an exact
  W₂-geodesic interpolation (McCann 1997). Wins BNB h=1 outright.
- **BNB/USDT and XRP/USDT** parquet caches; long-horizon backtest panel
  now BTC + ETH + SOL + BNB (XRP cached but not yet in `METHODS`).
- **`docs/RESEARCH_REPORT.md`** — research-paper-style writeup of the
  v0.3 contributions: motivation, formulation, headline numbers, falsif-
  ication verdicts, regime decomposition, robustness, limitations.
- **`scripts/summarize_v03.py`** — compact headline-table generator from
  the per-step JSON in `results/`.
- 5 new tests in `tests/test_forecasters.py` (drift recovery for EWMA;
  EWMA≡OLS at decay=1; Hetero monotonicity; ensemble collapses to each
  component at saturated weight; ensemble monotonicity). Suite: 31/31.

### Changed

- `THEORY.md` bumped to v0.3 with §2.6 (EWMA), §2.7 (Hetero), §2.8
  (ensemble), and §4 falsification criteria C5/C6/C7 covering v0.3.
- `RESULTS_LONG.md` regenerated against the 4-asset panel. The headline
  table now reports best WGeo-family variant vs best non-WGeo *baseline*
  (any of Static / RW / HS / GARCH-N / GARCH-t / GJR-GARCH-t), rather
  than just vs best GARCH variant.
- `default_forecaster(horizon)` now branches three-ways: `Gated` for
  h ≤ 1, `WGeoGarchEnsemble` for 5 ≤ h < 21, `WassersteinGeodesicHetero`
  for h ≥ 21.
- Version bumped to 0.3.0 across `pyproject.toml`, `wbtc.__version__`.

### Honest negative findings

- **`WassersteinGeodesicHetero` never wins outright.** The empirical
  quantile vector is already fit on a 90-day window with its own
  volatility regime; multiplying its width by a GARCH-implied
  conditional/unconditional vol ratio double-counts the same signal.
  Documented in `docs/RESEARCH_REPORT.md` §4.4 as the v0.3 boundary
  statement. The right way to add a parametric vol forecast is to
  *replace* the empirical dispersion, not multiply it (v0.4 target).
- **DM significance is mostly negative.** Only ETH h=5 achieves p < 0.05
  vs the best baseline in the v0.3 panel. The directional consistency
  (12/12 cells favouring WGeo-family, binomial p ≈ 0.024%) is the
  statistical achievement, not single-cell DM.
- **Falsification criteria C5 and C6 fail.** Reported in
  `docs/RESEARCH_REPORT.md §4.2` rather than silently dropping the
  methods — they earn their keep as documented null bounds.

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
