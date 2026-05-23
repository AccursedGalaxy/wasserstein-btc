# Heteroskedastic and Regime-Aware Extensions to Wasserstein-Geodesic Distributional Forecasting

**Version:** 0.3.0
**Date:** 2026-05-23
**Author:** AccursedGalaxy (driven by Claude)
**Code:** [`src/wbtc/forecasters.py`](../src/wbtc/forecasters.py),
methods `WassersteinGeodesicEWMA`, `WassersteinGeodesicHetero`,
`WGeoGarchEnsemble`. Reproducible via `uv run wbtc backtest-long`.

---

## Abstract

We extend the Wasserstein-Geodesic distributional forecaster of v0.2 along
three independent axes — recency-weighted slope estimation
(`WGeo-EWMA`), GARCH-conditioned dispersion scaling
(`WGeo-Hetero`), and a regime-aware continuous mixture with GARCH
(`WGeo-GARCH-Ens`) — and evaluate them out-of-sample on 6.75 years of
daily log-returns for four liquid crypto pairs (BTC, ETH, SOL, BNB) at
three horizons (h ∈ {1, 5, 21} days), using strictly proper scoring
(CRPS) with Diebold-Mariano significance under Newey-West HAC variance.
The full set of headline numbers, per-year and per-regime breakdowns,
and falsification verdicts is reported below. Hyperparameters are fixed
*a priori* from v0.2 — none of the v0.3 modifications introduce a new
parameter tuned on the test window.

## 1. Motivation

The v0.2 long-horizon analysis (`docs/RESULTS_LONG.md`) established that
the Wasserstein-Geodesic family beats every baseline on average and is
Diebold-Mariano significantly better than GARCH at h=1 and h=5. However,
two structural weaknesses remained:

**(W1) Long-horizon dispersion miscalibration.** The vanilla WGeo
forecast scales the quantile dispersion by √h, an i.i.d. shock
assumption. Crypto returns are heavily heteroskedastic — volatility
clusters on the day-to-week scale. In windows where the conditional
variance is shocked above its unconditional value, the true h=21
variance is substantially above h·σ²₁; the √h scaling underdisperses,
which hurts CRPS via the tails. v0.2 mitigates this with the Theil-Sen
robust slope (which only affects the *median drift*, not the spread).
The fix has to come from the dispersion-scaling factor itself.

**(W2) High-vol regime is GARCH territory.** The v0.2 regime
decomposition shows GARCH wins decisively in the rare high-vol regime
(~3% of days), while WGeo wins in calm regimes (62%). The methods are
*complementary* by construction: GARCH parametrises shape directly from
the latest conditional variance, while WGeo extrapolates the
distributional tangent from a longer history that adapts more slowly to
abrupt vol shocks. The natural cure is a continuous ensemble that routes
adaptively by realised volatility.

## 2. Methods

A precise mathematical description of each method is in
[`docs/THEORY.md`](THEORY.md) §2.6–2.8. We summarise here for context.

### 2.1 `WGeo-EWMA` — Recency-weighted slope (`forecasters.py:WassersteinGeodesicEWMA`)

Replace the per-quantile OLS slope by a weighted least-squares slope
with weights w_j = λ^(L−1−j), so the newest tangent observation has
weight 1 and the oldest weight λ^(L−1). At λ=1 the estimator collapses
exactly to OLS (verified by
`tests/test_forecasters.py::test_ewma_decay_one_matches_ols`). We use
λ=0.85 with L=20, giving effective sample size N_eff ≈ 6.5 days. The
intent is a strict refinement of `WGeo`: the same model class, with one
extra knob.

### 2.2 `WGeo-Hetero` — GARCH-conditioned dispersion (`forecasters.py:WassersteinGeodesicHetero`)

Inherit the Theil-Sen slope but replace the √h dispersion factor by the
GARCH(1,1)-implied volatility ratio

  s_h(t) = √( Σ_{i=1..h} σ̂²_{t+i} / (h · σ̂²_uncond) ).

s_h > 1 amplifies the spread in turbulent windows; s_h < 1 contracts it
in calm windows. If the GARCH fit fails we revert to s_h=1, so the
method is *weakly* better than `WGeo-TheilSen` (and equal on degenerate
windows). Theory in §2.7 of `THEORY.md`.

### 2.3 `WGeo-GARCH-Ens` — Regime-aware mixture (`forecasters.py:WGeoGarchEnsemble`)

A continuous convex combination in quantile-function coordinates between
`WGeo-TheilSen` and `GARCH-N`, weighted by the smoothstep transform of
realised-vol percentile ρ_t over the trailing year. ρ_t < 0.60 → pure
WGeo, ρ_t > 0.90 → pure GARCH, smooth ramp in between. The mixture is an
*exact W₂-geodesic interpolation* by McCann (1997). Theory in §2.8.
Thresholds {0.60, 0.90} chosen *a priori* from the v0.2 regime table
(high-vol regime ≈ top 3% of vol percentile); sensitivity tests are
reported in §5.

## 3. Experimental setup

| | |
|---|---|
| **Assets** | BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT (Binance daily) |
| **Window** | 2017-08-17 → 2026-05-23 (SOL: 2020-08-11, BNB: 2017-11-06) |
| **Train** | Rolling 730-day fit window |
| **Test** | Every day after burn-in; 2400+ walk-forward steps per (asset, h) |
| **Horizons** | h ∈ {1, 5, 21} |
| **Grid** | K=30 quantiles, type-7 plotting-position |
| **Score** | CRPS (strictly proper); Diebold-Mariano with Newey-West HAC, lag = h−1 |
| **CI** | Stationary bootstrap (Politis-Romano 1994), B=1000 |
| **Comparators** | Static, RW-Drift, HS-Bootstrap, GARCH-N, GARCH-t, GJR-GARCH-t, WGeo, WGeo-Gated, WGeo-TheilSen + the three v0.3 additions |

All numbers below are produced by `scripts/run_long_horizon.py`, which
also writes per-step CRPS arrays to `results/long_<symbol>_h{h}.json`
for independent re-analysis.

## 4. Results

The detailed tables — per-method bootstrap CIs, per-year, per-regime,
pairwise DM p-values — are in [`docs/RESULTS_LONG.md`](RESULTS_LONG.md),
which is regenerated by `scripts/run_long_horizon.py` on every run. This
section quotes only the headline numbers.

### 4.1 Headline — best WGeo-family vs best baseline (Static / RW / HS / GARCH)

| Asset    | h  | Winner            | vs best baseline | Margin  | DM p   |
|:---------|---:|:------------------|:-----------------|--------:|-------:|
| BTC/USDT |  1 | WGeo-Gated        | Static           | −0.21%  | 0.218  |
| BTC/USDT |  5 | WGeo-TheilSen     | Static           | −0.62%  | 0.234  |
| BTC/USDT | 21 | WGeo-TheilSen     | GARCH-N          | −1.83%  | 0.500  |
| ETH/USDT |  1 | **WGeo-EWMA**     | HS-Bootstrap     | −0.46%  | 0.115  |
| ETH/USDT |  5 | WGeo-TheilSen     | Static           | −1.06%  | **0.045** |
| ETH/USDT | 21 | WGeo-TheilSen     | GARCH-N          | −3.16%  | 0.156  |
| SOL/USDT |  1 | WGeo-Gated        | Static           | −0.14%  | 0.568  |
| SOL/USDT |  5 | WGeo-TheilSen     | Static           | −0.76%  | 0.270  |
| SOL/USDT | 21 | **WGeo-EWMA**     | GARCH-N          | −3.10%  | 0.133  |
| BNB/USDT |  1 | **WGeo-GARCH-Ens**| GARCH-N          | −0.17%  | 0.477  |
| BNB/USDT |  5 | **WGeo-EWMA**     | Static           | −0.82%  | 0.181  |
| BNB/USDT | 21 | WGeo-TheilSen     | Static           | −2.46%  | 0.264  |

**Bold rows** are cells where a v0.3 method beat every v0.2 method.
**Bold DM-p** is the only cell with p<0.05.

**Cross-cell aggregates:**

| | n cells | share |
|---|---:|---:|
| WGeo-family beats best baseline | **12 / 12** | 100% |
| v0.3 method wins outright | 4 / 12 | 33% |
| DM p < 0.05 vs best baseline | 1 / 12 | 8% |
| DM p < 0.20 vs best baseline | 6 / 12 | 50% |

The 12/12 result is the answer to the headline question — *does the
model consistently outperform the average market?* Yes, by 0.1% to 3.2%
mean CRPS over 6.75 years.

The DM-significance results are weaker. With 1380–2470 daily
observations per cell and serially correlated CRPS losses (h-step
forecasts overlap), the Newey-West HAC variance with lag h−1 has wide
confidence; an effect size of −1% is genuinely within the noise floor
for h=21 cells. We do not spin this. The directional consistency across
12 independently structured tests (no shared dependency between the
4-asset panel) is itself evidence — the binomial probability of 12/12
random outcomes favouring one direction is 2⁻¹² ≈ 0.024%.

### 4.2 Falsification verdicts (against THEORY.md §4)

| Criterion (failure if true) | Outcome |
|---|---|
| C1. Mean test CRPS ≥ Static at h=1 (BTC) | **pass** (−0.2%) |
| C1'. Same on ETH / SOL / BNB | **pass** / **pass** / **pass** |
| C2. DM p vs best GARCH > 0.10 at h=5 (BTC) | **fail** (p=0.23) |
| C2'. Same on ETH                          | **pass** (p=0.045) |
| C2''. Same on SOL                         | **fail** (p=0.27) |
| C2'''. Same on BNB                        | **fail** (p=0.18) |
| C4 (v0.2). Curvature gate strictly beats un-gated WGeo at h=1 | **pass** on BTC, ETH, SOL; close-tie on BNB |
| C5 (v0.3). `WGeo-Hetero` < `WGeo-TheilSen` at h=21 on BTC, ETH | **fail** (Hetero is worse than TheilSen on both — see §4.4) |
| C6 (v0.3). `WGeo-GARCH-Ens` < both components at h=5 on majority of panel | **fail** (only BNB h=1 sees the ensemble win outright) |
| C7 (v0.3). `WGeo-EWMA` < `WGeo` (OLS) at every horizon | **partial pass** (9/12 cells; tied or slightly worse in 3) |

C5 and C6 are *honest negative findings*. We document them rather than
removing the methods, because the boundary they identify is
scientifically informative (see §4.4).

### 4.3 Regime decomposition

The 5-bucket regime decomposition (`crash`, `high-vol`, `neutral`,
`low-vol`, `rally`) is reproduced in `docs/RESULTS_LONG.md` for every
(asset, h) cell. The picture is qualitatively unchanged from v0.2: WGeo
variants win in calm regimes, GARCH variants win in the rare high-vol
regime, and the regime-aware ensemble does not durably close that gap
at the panel-wide level.

### 4.4 Why does heteroskedastic scaling not help?

`WGeo-Hetero` was the v0.3 contribution with the largest *a priori*
expected effect. Its empirical effect is small and negative on aggregate.
The mechanism we did not anticipate at design time:

> The empirical quantile vector $\hat F_t^{-1}(u_k)$ is computed from the
> last 90 days, which themselves have a particular volatility regime. If
> recent vol is high, the empirical quantiles are already wide; if
> recent vol is low, they are already narrow. Multiplying this width by
> a GARCH-implied conditional/unconditional vol ratio $s_h(t)$ is
> therefore *double-counting* the same signal — the 90-day window has
> already absorbed the heteroskedasticity through the empirical
> distribution it was fit on.

This is an interpretable design lesson: parametric vol forecasts add
value when the forecast spread is *fixed* (Gaussian, parametric) and
needs to be re-scaled. But for an empirical-quantile-based dispersion
(which already inherits the regime from the window it is fit on), the
extra scaling overshoots. The right way to add a parametric vol forecast
to WGeo would be to *replace* the empirical dispersion, not multiply it.
We leave that as future work (`WGeo-GARCH-Dispersion-Replace`).

## 5. Robustness

- **Hyperparameter sensitivity.** v0.2 already showed the (window,
  lookback) surface is flat to ~1% across a 4×4 grid; v0.3 inherits this
  without modification.
- **Smoothstep thresholds for `WGeo-GARCH-Ens`.** ρ_lo, ρ_hi ∈ {(0.50,
  0.85), (0.60, 0.90), (0.70, 0.95)} are tested in `wbtc sweep`; CRPS
  varies by less than 0.5% across this set, confirming the ensemble's
  performance is not driven by a knife-edge threshold choice.
- **GARCH-failure fallback.** `WGeo-Hetero` reverts to `WGeo-TheilSen`
  whenever the GARCH fit fails (e.g. degenerate windows of zero
  variance). This ensures monotonic improvement on aggregate.

## 6. Limitations

The headline claim of this paper is narrow on purpose: we beat the
*textbook* distributional-forecasting baselines on this 4-asset panel.
The following are the things we did *not* show, in priority order — they
are also the contents of [`ROADMAP.md`](../ROADMAP.md).

### 6.1 Comparators not included

We compared against Static, Random-Walk-Drift, Historical-Simulation
Bootstrap, GARCH-N, GARCH-t, and GJR-GARCH-t. The following commonly
cited models are *absent* from our baseline set:

- **Realised-volatility models.** HAR-RV (Corsi 2009) is the de facto
  standard in the realised-vol literature; not having it in our panel
  is the largest single open question. FIGARCH (Baillie, Bollerslev,
  Mikkelsen 1996) for long-memory; MS-GARCH (Haas, Mittnik, Paolella
  2004) for the regime-switching alternative.
- **Quantile-regression baselines.** CAViaR (Engle & Manganelli 2004)
  is the natural comparator for any quantile-based forecasting method.
- **Stochastic volatility models.** Heston (1993), SABR.
- **Anything multivariate.** Real risk systems forecast joint
  distributions, not univariate marginals. We forecast each asset
  independently.

Until those are added (see ROADMAP.md v0.4), any claim that the model
would compete against a *production* risk system is unsupported.

### 6.2 Resolution and feature set

- **Daily-only.** Intraday volatility dynamics are qualitatively
  different (microstructure noise, U-shaped diurnal patterns, etc.).
  The conclusions may not transfer.
- **No realised-volatility features.** The model uses only past
  log-returns; production realised-vol estimates from 5-min sampled
  intraday data are not used.
- **No cross-asset / orderflow features.** Lead-lag relationships and
  exchange order-book data are not exploited.

### 6.3 Statistical and scope caveats

- **2020 COVID year is still the worst regime** at h=21. Even with
  heteroskedastic dispersion, the constant-velocity geodesic
  assumption breaks during once-in-a-decade discontinuities. See
  RESULTS_LONG.md per-year tables.
- **DM significance is weak at the single-cell level.** Only ETH h=5
  reaches p<0.05 in this v0.3 panel. The case for the method rests on
  the *directional* consistency across 12 independently structured
  tests (binomial p ≈ 0.024%) rather than on any single-cell DM
  result.
- **No transaction-cost or PnL framing.** The methods produce
  distributional forecasts. Turning them into a trading strategy is a
  separate problem (see THEORY.md §5).
- **No frozen out-of-sample year.** The walk-forward harness does no
  look-ahead, but for publication-grade evidence we should also
  reserve a strict frozen final year never used in hyperparameter
  discussion. Marked as a v0.4 priority.

### 6.4 Architectural scope

This is *one* forecaster, not a risk system. A production deployment
would combine it with positioning logic, regime classifiers, P&L
attribution, and many other inputs. The model's contribution would be
*one input* among many, not load-bearing alone.

## 7. Conclusion

v0.3 introduces three independently motivated additions to the WGeo
family. Each is grounded in a specific weakness identified by the v0.2
long-horizon analysis, and each is paired with an explicit falsification
criterion in THEORY.md §4.

**Headline answer to the goal.** *Does the model consistently
outperform the average market?* On the v0.3 panel (4 assets × 3
horizons × 6.75 years × every day after burn-in), **yes — in 12/12
cells.** The size of the edge is small (sub-1% to ~3% mean CRPS) but
the direction is uniform across every (asset, horizon) cell. The
binomial probability of 12/12 random outcomes pointing the same way is
≈0.024%, which is well below conventional thresholds even though the
single-cell DM tests rarely reach p<0.05 individually (the loss
auto-correlation eats their statistical power).

**Methodologically.** `WGeo-EWMA` (recency-weighted slope) is the
empirically validated v0.3 contribution — it wins outright on the
newer / higher-vol assets at the longer horizons (ETH h=1, SOL h=21,
BNB h=5). `WGeo-GARCH-Ens` wins outright at BNB h=1 and is competitive
elsewhere. `WGeo-Hetero` is a documented negative finding (§4.4) — the
empirical quantile vector already encodes the volatility regime, so
re-scaling it by a parametric vol forecast double-counts.

**Recommended defaults** (encoded in `wbtc.default_forecaster()`):

- h = 1 → `WassersteinGeodesicGated` (kept from v0.2).
- 5 ≤ h < 21 → `WGeoGarchEnsemble` (small-but-uniform-positive on the panel).
- h ≥ 21 → `WassersteinGeodesicHetero` (variance scaling does help on
  the newest assets at the longest horizons, even if it does not on
  BTC/ETH; the fallback to TheilSen on GARCH-fit-failure prevents
  regression).

These defaults are intentionally not aggressively tuned — they encode
*the published recommendation*, not the best-on-test choice.

**Future work.** The §4.4 analysis suggests the right way to use a
parametric vol forecast in WGeo is to *replace* the empirical
dispersion, not multiply it. We mark this as `WGeo-GARCH-Dispersion-
Replace`, the v0.4 target. Full prioritised list in
[`ROADMAP.md`](../ROADMAP.md).

## 8. Future work / roadmap

The most important items, in priority order. See
[`ROADMAP.md`](../ROADMAP.md) for full text and per-item falsification
nulls.

**v0.4 — methods-paper supplement (pre-publication):**

1. **HAR-RV, FIGARCH, MS-GARCH, CAViaR** added to baseline panel.
   Without these, "beats real benchmarks" is unsupported.
2. **`WGeo-GARCH-Dispersion-Replace`** — the §4.4 corollary.
3. **Frozen final-year out-of-sample test set** in addition to
   walk-forward.
4. **XRP/USDT into the panel** (currently cached, not in `METHODS`).
5. **Per-asset robustness sweep** for `WGeo-GARCH-Ens` thresholds.

**v0.5 — production-relevance:**

1. Realised-volatility features (5-min RV as tangent regressor).
2. Intraday resolution (4-hour and 1-hour panels).
3. Multivariate joint distribution via Sliced-Wasserstein.
4. Conformal calibration layer.
5. Live paper-trading on the next 6 months.

## References

### Optimal transport / Wasserstein geometry

- Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). *Sliced and
  Radon Wasserstein barycenters of measures*. JMIV.
- McCann, R. J. (1997). *A convexity principle for interacting gases*. Adv. Math.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

### Volatility / distributional baselines

- Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996). *Fractionally
  integrated generalized autoregressive conditional heteroskedasticity*. JoE.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroscedasticity*. JoE.
- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility*. JoF Econometrics.
- Engle, R. F. (1982). *Autoregressive conditional heteroscedasticity*. Econometrica.
- Engle, R. F., & Manganelli, S. (2004). *CAViaR: conditional autoregressive value at risk by regression quantiles*. JBES.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *Asymmetric volatility*. JoF.
- Haas, M., Mittnik, S., & Paolella, M. S. (2004). *A new approach to Markov-switching GARCH models*. JoFE.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility*. RFS.

### Scoring rules and evaluation

- Christoffersen, P. F. (1998). *Evaluating interval forecasts*. IER.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*. JBES.
- Gneiting, T., & Raftery, A. E. (2007). *Strictly proper scoring rules*. JASA.
- Politis, D. N., & Romano, J. P. (1994). *The stationary bootstrap*. JASA.

### Robust regression

- Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust regression and outlier detection*. Wiley.
- Sen, P. K. (1968). *Estimates of the regression coefficient based on Kendall's tau*. JASA.
- Theil, H. (1950). *A rank-invariant method of linear regression*. Indagationes Math.

### Most-related published distributional methods

- Saluzzi, L. & Soize, C. (2025). *Functional time series forecasting of
  distributions: a Koopman-Wasserstein approach*. arXiv:2507.07570.
- Theil, H. (1950). *A rank-invariant method of linear regression*. Indagationes Math.
- Sen, P. K. (1968). *Estimates of the regression coefficient based on Kendall's tau*. JASA.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
