# wasserstein-btc

Tangent-space **Wasserstein-geodesic distributional forecasting** for
crypto log-returns, with a novel **regime-curvature gate** (for h=1) and a
**Theil-Sen robust-slope** variant (recommended default for h ≥ 5).

This repository forecasts the *whole conditional distribution* of future
returns, not a single number. The market is treated as a trajectory on the
2-Wasserstein manifold of probability measures; we estimate the local
tangent velocity by per-quantile robust regression against time and
propagate it forward.

## Headline result — 6.75-year out-of-sample, multi-asset

Walk-forward over 2470 days (2019-08 → 2026-05), including the 2020 COVID
crash, 2021 ATH cycle, 2022 LUNA + FTX collapses, and the 2025 bull run.
Scoring: CRPS (strictly proper, lower is better).

| Asset | h     | Best WGeo CRPS | Best GARCH CRPS | Improvement | Diebold-Mariano p |
|------:|------:|---------------:|----------------:|------------:|------------------:|
| BTC   | 1 d   | 0.01620 | 0.01646 | -1.6% | **1.2 × 10⁻⁶** |
| BTC   | 5 d   | 0.03714 | 0.03781 | -1.8% | **0.011** |
| BTC   | 21 d  | 0.08330 | 0.08485 | -1.8% | 0.500 |
| ETH   | 1 d   | 0.02179 | 0.02195 | -0.7% | 0.053 |
| ETH   | 5 d   | 0.04930 | 0.05037 | -2.1% | **0.0007** |
| ETH   | 21 d  | 0.10940 | 0.11297 | -3.2% | 0.156 |

Full per-year and per-regime breakdowns, plus the hyperparameter robustness
sweep, in **[`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md)**. Mathematics in
**[`docs/THEORY.md`](docs/THEORY.md)**.

## Why this is new

- We forecast the entire 1D return *distribution*, not the mean or variance.
- We work in quantile-function coordinates on $\mathcal{P}_2(\mathbb{R})$,
  where W₂ geodesics are *straight lines* (McCann 1997).  Forecasting
  reduces to robust per-quantile regression in time.
- A cosine-curvature gate (h=1) and a Theil-Sen robust slope (h≥5) detect
  and downweight regime-change perturbations without explicit regime
  modelling.
- Evaluated with strictly proper scoring rules (CRPS, Christoffersen
  coverage) and Diebold-Mariano significance tests against GARCH(1,1),
  GARCH-t, GJR-GARCH-t, Historical-Simulation Bootstrap, and naive baselines.

The closest published work (Saluzzi & Soize 2025,
[arXiv:2507.07570](https://arxiv.org/abs/2507.07570)) applies
Koopman/EDMD-on-W₂ to housing prices with no regime adaptation. This
project applies a different mechanism (tangent-space regression + robust
slope/curvature gate) to crypto returns, with rigorous distributional
scoring.

## Reproducing

```bash
uv sync
uv run python scripts/fetch_data.py BTC/USDT ETH/USDT   # cache OHLCV → data/
uv run python scripts/run_long_horizon.py               # multi-year multi-asset
uv run python scripts/hyperparam_sweep.py               # robustness check
uv run pytest                                           # 17 unit + property tests
```

## Honest limitations

- Improvement vs GARCH at h=21 is real (~2-3%) but not statistically
  significant given the n=2450 sample size.
- The proposed method **loses to naive at h=21 in 2020** (COVID crash).
  Constant-velocity geodesic extrapolation cannot survive a one-in-a-decade
  discontinuity.
- **GARCH wins in the high-vol regime** (~3% of days); WGeo wins in
  neutral/low-vol regimes (62% of days). The methods are complementary.
- No trading strategy is built on top — lower CRPS is necessary but not
  sufficient for P&L.

## Disclaimer

This is research code. **Not financial advice.** Falsification criteria are
documented in `THEORY.md §4` and tested against the full long-horizon
backtest in `RESULTS_LONG.md`.
