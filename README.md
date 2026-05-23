# wasserstein-btc

**Distributional forecasting for crypto returns via geodesics on the
2-Wasserstein manifold of probability measures.** v0.3.

[![tests](https://github.com/AccursedGalaxy/wasserstein-btc/actions/workflows/tests.yml/badge.svg)](https://github.com/AccursedGalaxy/wasserstein-btc/actions)
[![python](https://img.shields.io/badge/python-3.11+-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research-orange)](docs/RESEARCH_REPORT.md)

This library forecasts the *whole conditional distribution* of future
log-returns — not the mean and not the variance — for liquid crypto
pairs at horizons of 1, 5 and 21 days. The market is modelled as a
trajectory on the 2-Wasserstein manifold of probability measures, the
forecast is the tangent-space extrapolation of recent quantile vectors,
and the result is scored with strictly proper rules (CRPS) against an
explicit panel of baselines (Static, RW-Drift, Historical-Simulation
Bootstrap, GARCH-N, GARCH-t, GJR-GARCH-t).

> **What it is:** a small, falsifiable, interpretable distributional
> forecaster — ~4 hyperparameters, no learned weights, no neural net.
> **What it is not:** a trading-signal generator, a multivariate risk
> system, or a benchmark against state-of-the-art realised-volatility
> models (see [`docs/RESEARCH_REPORT.md §6`](docs/RESEARCH_REPORT.md)
> for what is *not* claimed, and [`ROADMAP.md`](ROADMAP.md) for the
> v0.4 priorities that would close that gap).

## Headline result

On the v0.3 panel (BTC + ETH + SOL + BNB × h ∈ {1, 5, 21} × 6.75 years
walk-forward; 1380–2470 test days per cell), the WGeo family beats the
best non-WGeo baseline (best of Static / RW-Drift / HS-Bootstrap /
GARCH-N / GARCH-t / GJR-GARCH-t) in **12 / 12 cells** by 0.1% to 3.2%
mean CRPS.

| Asset/h     | Winner               | vs baseline    | Margin  | DM p      |
|:------------|:---------------------|:---------------|--------:|----------:|
| BTC h=1     | WGeo-Gated           | Static         | −0.21%  | 0.218     |
| BTC h=5     | WGeo-TheilSen        | Static         | −0.62%  | 0.234     |
| BTC h=21    | WGeo-TheilSen        | GARCH-N        | −1.83%  | 0.500     |
| ETH h=1     | **WGeo-EWMA**        | HS-Bootstrap   | −0.46%  | 0.115     |
| ETH h=5     | WGeo-TheilSen        | Static         | −1.06%  | **0.045** |
| ETH h=21    | WGeo-TheilSen        | GARCH-N        | −3.16%  | 0.156     |
| SOL h=1     | WGeo-Gated           | Static         | −0.14%  | 0.568     |
| SOL h=5     | WGeo-TheilSen        | Static         | −0.76%  | 0.270     |
| SOL h=21    | **WGeo-EWMA**        | GARCH-N        | −3.10%  | 0.133     |
| BNB h=1     | **WGeo-GARCH-Ens**   | GARCH-N        | −0.17%  | 0.477     |
| BNB h=5     | **WGeo-EWMA**        | Static         | −0.82%  | 0.181     |
| BNB h=21    | WGeo-TheilSen        | Static         | −2.46%  | 0.264     |

Bold rows are cells where a v0.3 method beat all v0.2 methods. Bold p is
the one cell with p<0.05. Direction is uniform across all 12 cells —
binomial probability of 12/12 random outcomes pointing the same way is
≈ 0.024%. Per-year, per-regime, and hyperparameter-sensitivity tables in
[`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md). The methods-paper-style
writeup is in [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md).

## Install

The supported workflow uses [`uv`](https://docs.astral.sh/uv/) (fast and
reproducible). The package itself works under any Python ≥3.11.

```bash
git clone https://github.com/AccursedGalaxy/wasserstein-btc
cd wasserstein-btc
uv sync          # creates .venv with locked deps
uv run wbtc test # 31 tests, ~5 seconds
```

PyPI release is on the v0.4 roadmap (see [`ROADMAP.md`](ROADMAP.md)).

## Quick start — CLI

```bash
uv run wbtc info                              # what data do I have?
uv run wbtc fetch BTC/USDT ETH/USDT SOL/USDT  # fetch / update from Binance
uv run wbtc forecast BTC/USDT -H 5 --plot     # forecast & fan-chart PNG
uv run wbtc forecast BTC/USDT -H 5 --json     # JSON for scripting
uv run wbtc backtest --quick                  # fast single-symbol backtest
uv run wbtc backtest-long                     # full multi-asset (~30 min)
uv run wbtc extended-baselines                # HAR-RV/CAViaR/MS/FIGARCH/SV/BVAR vs WGeo on BTC (~2h)
uv run wbtc sweep                             # hyperparameter robustness
```

## Quick start — Python

```python
from wbtc import forecast, available_symbols, default_forecaster

available_symbols()
# ['BNB/USDT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

fc = forecast("BTC/USDT", horizon=5)
fc.median, fc.quantile(0.05), fc.quantile(0.95)
fc.to_dict()  # JSON-safe summary

# Pick a specific variant explicitly:
from wbtc import WassersteinGeodesicEWMA
fc = forecast("BTC/USDT", horizon=5,
              forecaster=WassersteinGeodesicEWMA(window=90, lookback=20))
```

`default_forecaster(horizon)` returns the recommended variant per
horizon (see `RESEARCH_REPORT.md §7`).

## What's novel

- **Per-quantile time-regression on the W₂ manifold.** The 1D-W₂-as-
  quantile-function isometry is textbook (Villani 2009 ch. 6); applying
  it to *time-series tangent extrapolation* of return distributions
  appears to be under-published. The closest published method
  (Saluzzi & Soize 2025, [arXiv:2507.07570](https://arxiv.org/abs/2507.07570))
  uses a Koopman/EDMD-spectral approach with no regime adaptation,
  applied to housing prices.
- **Cosine-curvature gate.** Continuous, non-Markovian gating that
  blends geodesic extrapolation with a static-empirical fallback when
  consecutive tangent vectors become orthogonal. Pays off at h=1.
- **Theil-Sen robust slope on the tangent.** 29.3% breakdown point;
  robust to recent-history outliers without explicit regime modelling.
- **Quantile-coordinate ensemble with GARCH.** Convex combination in
  quantile-function space is an *exact W₂-geodesic interpolation*
  (McCann 1997) — not a moment-matched or kernel-mixed surrogate.

## Documents

```
docs/
  THEORY.md           math (§2.6–2.8 are the v0.3 sections, §4 lists
                      explicit falsification criteria)
  RESEARCH_REPORT.md  paper-style writeup of the v0.3 contributions
  RESULTS_LONG.md     auto-regenerated 4-asset × 3-horizon evidence
  RESULTS.md          legacy v0.1 single-year report (superseded)
  INDEX.md            one-paragraph orientation to every doc
ROADMAP.md            v0.4 + v0.5 priorities (what would make it
                      competitive vs. production risk systems)
CONTRIBUTING.md       the conventions PRs must follow
CHANGELOG.md          v0.1 → v0.2 → v0.3 history
```

## Honest limitations

- We have benchmarked against **textbook baselines** as headline (Static
  / RW / HS / GARCH-N / GARCH-t / GJR-GARCH-t across 4 assets × 3
  horizons in [`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md)) and against
  a broader **named-econometric panel** on BTC in
  [`docs/RESULTS_EXTENDED.md`](docs/RESULTS_EXTENDED.md): HAR-RV (Corsi
  2009), CAViaR-SAV (Engle-Manganelli 2004), 2-state Markov-switching
  Normal (Hamilton 1989), FIGARCH(1,d,0) (Baillie-Bollerslev-Mikkelsen
  1996), AR(1) Stochastic Volatility (Taylor 1982 / Harvey-Ruiz-Shephard
  1994 via Kalman QML), and a bivariate VAR+GARCH using BTC + ETH
  jointly. Any *production*-risk-system claim is still unsupported —
  this rounds out the academic panel.
- **Daily-only.** Intraday volatility dynamics are different.
- **Univariate only.** The 1D-W₂ isometry doesn't extend cleanly to
  higher dimensions; multivariate is a v0.5 research item.
- **No trading P&L claim.** Distributional-forecast quality is
  necessary but not sufficient for tradeable alpha.
- **Heteroskedastic-dispersion variant (`WGeo-Hetero`) was a
  documented dead end** — see `RESEARCH_REPORT.md §4.4` for *why*
  (empirical-quantile-based dispersion already encodes the regime;
  multiplying by GARCH double-counts). The boundary is reusable.

## Citation

If you use this software in academic work, please cite it.
[`CITATION.cff`](CITATION.cff) is the structured form; the BibTeX-shaped
quick form:

```bibtex
@software{wasserstein_btc_2026,
  author       = {Robin Bohrer (AccursedGalaxy)},
  title        = {wasserstein-btc: tangent-space Wasserstein-geodesic
                  distributional forecasting for crypto returns},
  version      = {0.3.0},
  year         = {2026},
  url          = {https://github.com/AccursedGalaxy/wasserstein-btc}
}
```

## License

[MIT](LICENSE).

## Disclaimer

This is research code. **Not financial advice.** Falsification criteria
are documented in `docs/THEORY.md §4` and tested against the full
long-horizon backtest in `docs/RESULTS_LONG.md`. Documented failures are
in `docs/RESEARCH_REPORT.md §4.2`.
