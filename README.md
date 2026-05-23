# wasserstein-btc

Tangent-space **Wasserstein-geodesic distributional forecasting** for
Bitcoin log-returns, with a novel **regime-curvature gate**.

This repository forecasts the *whole conditional distribution* of future
Bitcoin returns, not a single number. The method treats the market as a
trajectory on the 2-Wasserstein manifold of probability measures, estimates
its local tangent velocity by linear regression of empirical quantile
functions against time, and propagates it forward — with a continuous fall-
back to a static-empirical forecast whenever the geodesic curvature spikes
(regime change).

The math, baselines, evaluation protocol and falsification criteria are in
**[`docs/THEORY.md`](docs/THEORY.md)**. Results live in
**[`docs/RESULTS.md`](docs/RESULTS.md)** once backtests complete.

## Why this might be new

- We forecast the entire 1D return *distribution*, not the mean or variance.
- We work directly in quantile-function coordinates on
  $\mathcal{P}_2(\mathbb{R})$, where W2 geodesics are *straight lines*.
  Forecasting reduces to per-quantile linear regression in time.
- A cosine-curvature gate detects when the constant-velocity geodesic
  assumption fails and degrades gracefully to a static forecast.
- Evaluated with strictly proper scoring rules (CRPS, log-score, Christoffersen
  coverage) and Diebold-Mariano significance tests vs GARCH baselines.

The closest published work (Saluzzi & Soize 2025,
[arXiv:2507.07570](https://arxiv.org/abs/2507.07570)) does
Koopman/EDMD-on-W2 on housing prices with no regime gating. This project
applies a different mechanism (tangent-space regression + curvature gate) to
crypto returns, with proper distributional scoring.

## Reproducing

```bash
uv sync
uv run python scripts/fetch_data.py        # BTC/USDT daily via ccxt -> data/
uv run python scripts/run_backtest.py      # writes results/ and docs/RESULTS.md
uv run pytest                              # unit + property tests
```

## Disclaimer

This is research code. **Not financial advice.** The README will be
updated honestly with whatever the backtest shows — including
failure. See §4 of [`docs/THEORY.md`](docs/THEORY.md) for the explicit
falsification criteria.
