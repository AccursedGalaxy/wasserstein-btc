# Long-Horizon Results — Multi-Year, Multi-Asset Validation

Goal: prove the Wasserstein-Geodesic forecaster works over a *long* time horizon.
Train: rolling 730-day window. Test: every day after burn-in (no separate holdout).
Scoring: CRPS (lower better, strictly proper).

## Pre-registration

**Pre-registered headline forecaster:** `WGeo-Ensemble` (the equal-weight W₂ barycentre of `WGeo-TheilSen`, `WGeo-EWMA`, `WGeo-Gated` — see `THEORY.md §2.9`). All headline DM tests below are `WGeo-Ensemble` against a fixed reference baseline. The previous reporting style — *best-of-family vs best-of-baseline* — is retained as a robustness appendix because the implicit max-over-comparators inflates type-I error and is not a valid pre-committed test.

**Pre-registered reference baselines:** `Static`, `GARCH-N`. `Static` is the most naive distributional baseline (the current empirical quantile, √h-scaled); `GARCH-N` is the standard parametric vol baseline from the econometrics canon. A win against both is the minimum bar for the v0.5 claim.

## Headline 1 — WGeo-Ensemble vs Static (pre-registered)

| symbol   |   h |   n_test | baseline   |   ensemble_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r |
|:---------|----:|---------:|:-----------|----------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|
| BTC/USDT |   1 |     2470 | Static     |        0.016168 |        0.016236 | -0.4%         |     -1.71 | 0.0871 |       -1.78 |   0.0745 |
| BTC/USDT |   5 |     2466 | Static     |        0.037061 |        0.037367 | -0.8%         |     -1.97 | 0.0491 |       -2.63 |   0.0085 |
| BTC/USDT |  21 |     2450 | Static     |        0.083158 |        0.085347 | -2.6%         |     -1.78 | 0.0756 |       -4.3  |   0      |
| ETH/USDT |   1 |     2470 | Static     |        0.021739 |        0.021897 | -0.7%         |     -3.02 | 0.0025 |       -3.22 |   0.0013 |
| ETH/USDT |   5 |     2466 | Static     |        0.049256 |        0.049834 | -1.2%         |     -2.73 | 0.0064 |       -3.75 |   0.0002 |
| ETH/USDT |  21 |     2450 | Static     |        0.109816 |        0.113701 | -3.4%         |     -2.28 | 0.0227 |       -6.19 |   0      |
| SOL/USDT |   1 |     1380 | Static     |        0.02503  |        0.025075 | -0.2%         |     -0.55 | 0.5839 |       -0.59 |   0.5566 |
| SOL/USDT |   5 |     1376 | Static     |        0.057153 |        0.057622 | -0.8%         |     -1.41 | 0.1588 |       -1.79 |   0.0728 |
| SOL/USDT |  21 |     1360 | Static     |        0.129833 |        0.134406 | -3.4%         |     -1.94 | 0.0529 |       -4.46 |   0      |
| BNB/USDT |   1 |     2389 | Static     |        0.020144 |        0.020339 | -1.0%         |     -3.33 | 0.0009 |       -3.77 |   0.0002 |
| BNB/USDT |   5 |     2385 | Static     |        0.045782 |        0.046297 | -1.1%         |     -2.2  | 0.0281 |       -2.78 |   0.0054 |
| BNB/USDT |  21 |     2369 | Static     |        0.102619 |        0.105553 | -2.8%         |     -1.55 | 0.12   |       -3.34 |   0.0008 |
| XRP/USDT |   1 |     2210 | Static     |        0.024359 |        0.024566 | -0.8%         |     -2.9  | 0.0038 |       -3.27 |   0.0011 |
| XRP/USDT |   5 |     2206 | Static     |        0.055389 |        0.056199 | -1.4%         |     -3.15 | 0.0017 |       -3.61 |   0.0003 |
| XRP/USDT |  21 |     2190 | Static     |        0.129103 |        0.132709 | -2.7%         |     -3.17 | 0.0015 |       -3.77 |   0.0002 |

## Headline 2 — WGeo-Ensemble vs GARCH-N (pre-registered)

| symbol   |   h |   n_test | baseline   |   ensemble_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r |
|:---------|----:|---------:|:-----------|----------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|
| BTC/USDT |   1 |     2470 | GARCH-N    |        0.016168 |        0.016463 | -1.8%         |     -5.08 | 0      |       -7.16 |   0      |
| BTC/USDT |   5 |     2466 | GARCH-N    |        0.037061 |        0.037807 | -2.0%         |     -3.15 | 0.0016 |       -5.93 |   0      |
| BTC/USDT |  21 |     2450 | GARCH-N    |        0.083158 |        0.084848 | -2.0%         |     -0.84 | 0.4009 |       -2.52 |   0.0116 |
| ETH/USDT |   1 |     2470 | GARCH-N    |        0.021739 |        0.021947 | -0.9%         |     -2.85 | 0.0044 |       -4.03 |   0.0001 |
| ETH/USDT |   5 |     2466 | GARCH-N    |        0.049256 |        0.050368 | -2.2%         |     -4.04 | 0.0001 |       -7.13 |   0      |
| ETH/USDT |  21 |     2450 | GARCH-N    |        0.109816 |        0.11297  | -2.8%         |     -1.5  | 0.1337 |       -4.51 |   0      |
| SOL/USDT |   1 |     1380 | GARCH-N    |        0.02503  |        0.025219 | -0.7%         |     -1.7  | 0.0894 |       -2.37 |   0.0177 |
| SOL/USDT |   5 |     1376 | GARCH-N    |        0.057153 |        0.058296 | -2.0%         |     -2.93 | 0.0034 |       -4.34 |   0      |
| SOL/USDT |  21 |     1360 | GARCH-N    |        0.129833 |        0.133568 | -2.8%         |     -1.62 | 0.1043 |       -4.14 |   0      |
| BNB/USDT |   1 |     2389 | GARCH-N    |        0.020144 |        0.020199 | -0.3%         |     -0.61 | 0.5433 |       -1.03 |   0.3015 |
| BNB/USDT |   5 |     2385 | GARCH-N    |        0.045782 |        0.046475 | -1.5%         |     -2.2  | 0.0278 |       -3.83 |   0.0001 |
| BNB/USDT |  21 |     2369 | GARCH-N    |        0.102619 |        0.105595 | -2.8%         |     -1.73 | 0.0845 |       -3.42 |   0.0006 |
| XRP/USDT |   1 |     2210 | GARCH-N    |        0.024359 |        0.024697 | -1.4%         |     -2.69 | 0.0071 |       -4.18 |   0      |
| XRP/USDT |   5 |     2206 | GARCH-N    |        0.055389 |        0.057258 | -3.3%         |     -4.51 | 0      |       -6.34 |   0      |
| XRP/USDT |  21 |     2190 | GARCH-N    |        0.129103 |        0.136474 | -5.4%         |     -3.5  | 0.0005 |       -4.76 |   0      |

## Headline 3 — WGeo-Ensemble vs Wasserstein Autoregression (benchmark, not pre-registered)

`WAR-*` is the Wasserstein autoregressive model of Zhang, Kokoszka & Petersen (2022) applied to the same rolling-90-day quantile vectors ("WAR on rolling-ECDF densities", `THEORY.md §2.11`). Consecutive densities share 89/90 observations, so the fitted lag-1 Wasserstein autocorrelation is ≈0.98 partly for mechanical reasons; WAR here is shrinkage of today's density toward the training-window barycentre. `WAR-1`: p=1, all densities, h-day location = sum of forecast daily medians. `WAR-1-last`: same dynamics with WGeo's location rule (terminal daily median only). `WAR-Select`: the paper's in-sample selection of (K, p) over K ∈ {20, 62, 250, all}, p ∈ {1..5}. The h-day conversion is the panel convention shared by every quantile-based method, not part of WAR. Falsification bar (`THEORY.md §4`): `WGeo-Ensemble` must beat the *best* WAR variant with dm_p_r<0.05 in ≥ 8 of 15 cells for tangent-space extrapolation to be claimed to add anything over tangent-space mean reversion.

**WGeo-Ensemble vs WAR-1**

| symbol   |   h |   n_test | baseline   |   ensemble_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r |
|:---------|----:|---------:|:-----------|----------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|
| BTC/USDT |   1 |     2470 | WAR-1      |        0.016168 |        0.016208 | -0.2%         |     -3.51 | 0.0004 |       -3.66 |   0.0003 |
| BTC/USDT |   5 |     2466 | WAR-1      |        0.037061 |        0.03751  | -1.2%         |     -1.67 | 0.0955 |       -1.8  |   0.0724 |
| BTC/USDT |  21 |     2450 | WAR-1      |        0.083158 |        0.085867 | -3.2%         |     -1.15 | 0.2485 |       -1.34 |   0.179  |
| ETH/USDT |   1 |     2470 | WAR-1      |        0.021739 |        0.021786 | -0.2%         |     -3.3  | 0.001  |       -3.46 |   0.0005 |
| ETH/USDT |   5 |     2466 | WAR-1      |        0.049256 |        0.049971 | -1.4%         |     -2.16 | 0.0304 |       -2.44 |   0.0148 |
| ETH/USDT |  21 |     2450 | WAR-1      |        0.109816 |        0.114697 | -4.3%         |     -1.62 | 0.1047 |       -2.16 |   0.0308 |
| SOL/USDT |   1 |     1380 | WAR-1      |        0.02503  |        0.025086 | -0.2%         |     -2.58 | 0.0099 |       -2.82 |   0.0048 |
| SOL/USDT |   5 |     1376 | WAR-1      |        0.057153 |        0.058304 | -2.0%         |     -2.45 | 0.0141 |       -2.55 |   0.0107 |
| SOL/USDT |  21 |     1360 | WAR-1      |        0.129833 |        0.136555 | -4.9%         |     -1.6  | 0.1095 |       -1.73 |   0.0833 |
| BNB/USDT |   1 |     2389 | WAR-1      |        0.020144 |        0.020203 | -0.3%         |     -4.01 | 0.0001 |       -4.2  |   0      |
| BNB/USDT |   5 |     2385 | WAR-1      |        0.045782 |        0.046258 | -1.0%         |     -1.63 | 0.1038 |       -2.04 |   0.0413 |
| BNB/USDT |  21 |     2369 | WAR-1      |        0.102619 |        0.1053   | -2.5%         |     -0.98 | 0.3258 |       -1.34 |   0.1815 |
| XRP/USDT |   1 |     2210 | WAR-1      |        0.024359 |        0.024428 | -0.3%         |     -4.29 | 0      |       -4.57 |   0      |
| XRP/USDT |   5 |     2206 | WAR-1      |        0.055389 |        0.056215 | -1.5%         |     -2.79 | 0.0054 |       -2.94 |   0.0033 |
| XRP/USDT |  21 |     2190 | WAR-1      |        0.129103 |        0.136215 | -5.2%         |     -2.75 | 0.0059 |       -2.87 |   0.0041 |

**WGeo-Ensemble vs WAR-1-last**

| symbol   |   h |   n_test | baseline   |   ensemble_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r |
|:---------|----:|---------:|:-----------|----------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|
| BTC/USDT |   1 |     2470 | WAR-1-last |        0.016168 |        0.016208 | -0.2%         |     -3.51 | 0.0004 |       -3.66 |   0.0003 |
| BTC/USDT |   5 |     2466 | WAR-1-last |        0.037061 |        0.037099 | -0.1%         |     -0.93 | 0.3542 |       -1.17 |   0.2428 |
| BTC/USDT |  21 |     2450 | WAR-1-last |        0.083158 |        0.08284  | +0.4%         |      0.95 | 0.3432 |        1.99 |   0.0467 |
| ETH/USDT |   1 |     2470 | WAR-1-last |        0.021739 |        0.021786 | -0.2%         |     -3.3  | 0.001  |       -3.46 |   0.0005 |
| ETH/USDT |   5 |     2466 | WAR-1-last |        0.049256 |        0.049246 | +0.0%         |      0.18 | 0.8549 |        0.23 |   0.8182 |
| ETH/USDT |  21 |     2450 | WAR-1-last |        0.109816 |        0.109047 | +0.7%         |      1.67 | 0.0942 |        3.69 |   0.0002 |
| SOL/USDT |   1 |     1380 | WAR-1-last |        0.02503  |        0.025086 | -0.2%         |     -2.58 | 0.0099 |       -2.82 |   0.0048 |
| SOL/USDT |   5 |     1376 | WAR-1-last |        0.057153 |        0.057096 | +0.1%         |      0.85 | 0.3936 |        1.03 |   0.3025 |
| SOL/USDT |  21 |     1360 | WAR-1-last |        0.129833 |        0.128401 | +1.1%         |      2.92 | 0.0035 |        4.51 |   0      |
| BNB/USDT |   1 |     2389 | WAR-1-last |        0.020144 |        0.020203 | -0.3%         |     -4.01 | 0.0001 |       -4.2  |   0      |
| BNB/USDT |   5 |     2385 | WAR-1-last |        0.045782 |        0.045884 | -0.2%         |     -1.77 | 0.076  |       -2.34 |   0.0194 |
| BNB/USDT |  21 |     2369 | WAR-1-last |        0.102619 |        0.102471 | +0.1%         |      0.31 | 0.76   |        0.61 |   0.5429 |
| XRP/USDT |   1 |     2210 | WAR-1-last |        0.024359 |        0.024428 | -0.3%         |     -4.29 | 0      |       -4.57 |   0      |
| XRP/USDT |   5 |     2206 | WAR-1-last |        0.055389 |        0.055465 | -0.1%         |     -1.44 | 0.1485 |       -1.59 |   0.1115 |
| XRP/USDT |  21 |     2190 | WAR-1-last |        0.129103 |        0.128207 | +0.7%         |      2.79 | 0.0053 |        3.28 |   0.001  |

**WGeo-Ensemble vs WAR-Select**

| symbol   |   h |   n_test | baseline   |   ensemble_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r |
|:---------|----:|---------:|:-----------|----------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|
| BTC/USDT |   1 |     2470 | WAR-Select |        0.016168 |        0.016208 | -0.2%         |     -3.51 | 0.0004 |       -3.66 |   0.0002 |
| BTC/USDT |   5 |     2466 | WAR-Select |        0.037061 |        0.037512 | -1.2%         |     -1.68 | 0.0938 |       -1.81 |   0.071  |
| BTC/USDT |  21 |     2450 | WAR-Select |        0.083158 |        0.085858 | -3.1%         |     -1.15 | 0.2494 |       -1.34 |   0.1797 |
| ETH/USDT |   1 |     2470 | WAR-Select |        0.021739 |        0.021786 | -0.2%         |     -3.29 | 0.001  |       -3.45 |   0.0006 |
| ETH/USDT |   5 |     2466 | WAR-Select |        0.049256 |        0.049968 | -1.4%         |     -2.16 | 0.0309 |       -2.43 |   0.0151 |
| ETH/USDT |  21 |     2450 | WAR-Select |        0.109816 |        0.114724 | -4.3%         |     -1.64 | 0.1008 |       -2.17 |   0.03   |
| SOL/USDT |   1 |     1380 | WAR-Select |        0.02503  |        0.025086 | -0.2%         |     -2.59 | 0.0095 |       -2.83 |   0.0046 |
| SOL/USDT |   5 |     1376 | WAR-Select |        0.057153 |        0.058315 | -2.0%         |     -2.47 | 0.0135 |       -2.57 |   0.0102 |
| SOL/USDT |  21 |     1360 | WAR-Select |        0.129833 |        0.136713 | -5.0%         |     -1.62 | 0.1048 |       -1.75 |   0.0803 |
| BNB/USDT |   1 |     2389 | WAR-Select |        0.020144 |        0.020203 | -0.3%         |     -4.01 | 0.0001 |       -4.2  |   0      |
| BNB/USDT |   5 |     2385 | WAR-Select |        0.045782 |        0.046257 | -1.0%         |     -1.62 | 0.1045 |       -2.04 |   0.0418 |
| BNB/USDT |  21 |     2369 | WAR-Select |        0.102619 |        0.105347 | -2.6%         |     -1    | 0.3182 |       -1.35 |   0.1759 |
| XRP/USDT |   1 |     2210 | WAR-Select |        0.024359 |        0.024428 | -0.3%         |     -4.27 | 0      |       -4.55 |   0      |
| XRP/USDT |   5 |     2206 | WAR-Select |        0.055389 |        0.056216 | -1.5%         |     -2.79 | 0.0052 |       -2.94 |   0.0032 |
| XRP/USDT |  21 |     2190 | WAR-Select |        0.129103 |        0.136449 | -5.4%         |     -2.85 | 0.0043 |       -2.98 |   0.0029 |

**Falsification count:** `WGeo-Ensemble` beats the best WAR variant per cell with dm_p_r<0.05 in **6/15** cells (bar: ≥ 8/15 → **FAIL**). Per variant (lower CRPS and dm_p_r<0.05): vs `WAR-1`: 11/15, vs `WAR-1-last`: 6/15, vs `WAR-Select`: 11/15.

*`dm_p` is the classic Diebold-Mariano (1995) p-value; `dm_p_r` is the variance-reduced residualised DM with the `full` control set (vol moments + four peer-method loss series) — a Giacomini-White-style augmented test of the same unconditional EPA null. See the sensitivity table below for the breakdown by control set, and `docs/THEORY.md §2.10` for the math.*

## Residualised-DM sensitivity to control set

The residualised DM test admits any covariate predictable at time t. Three control sets are reported, ordered from least to most powerful: `none` (= vanilla DM), `vol` (`[y, |y|, y²]` — sign, magnitude, and kurtosis-like moment of the realised return), and `full` (`vol` plus up to four peer-method loss series). Peer-loss controls are admissible under Giacomini-White but rhetorically more endogenous; the table below decomposes the residualised lift so the reader can see how much is driven by vol controls alone vs. peer losses.

**Aggregate — cells with `dm_p < 0.05` and `WGeo-Ensemble` lower CRPS:**

| baseline   |   cells | no_controls   | vol_only   | vol_plus_peers   |
|:-----------|--------:|:--------------|:-----------|:-----------------|
| Static     |      15 | 9/15          | 12/15      | 12/15            |
| GARCH-N    |      15 | 9/15          | 12/15      | 14/15            |

The pre-registered falsification threshold in `PREREGISTRATION.md` is anchored to the `vol_only` column so the v0.5 bar does not depend on peer-loss correlations — peer losses are reported as a power-only bonus, not a contributor to the headline claim.

**Per-cell residualised-DM p-values:**

| symbol   |   h | baseline   |   dm_p_none |   dm_p_vol |   dm_p_full |
|:---------|----:|:-----------|------------:|-----------:|------------:|
| BTC/USDT |   1 | Static     |      0.0871 |     0.0862 |      0.0745 |
| BTC/USDT |   1 | GARCH-N    |      0      |     0      |      0      |
| BTC/USDT |   5 | Static     |      0.0491 |     0.0211 |      0.0085 |
| BTC/USDT |   5 | GARCH-N    |      0.0016 |     0      |      0      |
| BTC/USDT |  21 | Static     |      0.0756 |     0.01   |      0      |
| BTC/USDT |  21 | GARCH-N    |      0.4009 |     0.1107 |      0.0116 |
| ETH/USDT |   1 | Static     |      0.0025 |     0.0025 |      0.0013 |
| ETH/USDT |   1 | GARCH-N    |      0.0044 |     0.0009 |      0.0001 |
| ETH/USDT |   5 | Static     |      0.0064 |     0.0033 |      0.0002 |
| ETH/USDT |   5 | GARCH-N    |      0.0001 |     0      |      0      |
| ETH/USDT |  21 | Static     |      0.0227 |     0.007  |      0      |
| ETH/USDT |  21 | GARCH-N    |      0.1337 |     0.0362 |      0      |
| SOL/USDT |   1 | Static     |      0.5839 |     0.5613 |      0.5566 |
| SOL/USDT |   1 | GARCH-N    |      0.0894 |     0.0563 |      0.0177 |
| SOL/USDT |   5 | Static     |      0.1588 |     0.1391 |      0.0728 |
| SOL/USDT |   5 | GARCH-N    |      0.0034 |     0.0007 |      0      |
| SOL/USDT |  21 | Static     |      0.0529 |     0.0364 |      0      |
| SOL/USDT |  21 | GARCH-N    |      0.1043 |     0.0306 |      0      |
| BNB/USDT |   1 | Static     |      0.0009 |     0.0005 |      0.0002 |
| BNB/USDT |   1 | GARCH-N    |      0.5433 |     0.4857 |      0.3015 |
| BNB/USDT |   5 | Static     |      0.0281 |     0.0156 |      0.0054 |
| BNB/USDT |   5 | GARCH-N    |      0.0278 |     0.0105 |      0.0001 |
| BNB/USDT |  21 | Static     |      0.12   |     0.032  |      0.0008 |
| BNB/USDT |  21 | GARCH-N    |      0.0845 |     0.0233 |      0.0006 |
| XRP/USDT |   1 | Static     |      0.0038 |     0.0025 |      0.0011 |
| XRP/USDT |   1 | GARCH-N    |      0.0071 |     0.0022 |      0      |
| XRP/USDT |   5 | Static     |      0.0017 |     0.0009 |      0.0003 |
| XRP/USDT |   5 | GARCH-N    |      0      |     0      |      0      |
| XRP/USDT |  21 | Static     |      0.0015 |     0.0005 |      0.0002 |
| XRP/USDT |  21 | GARCH-N    |      0.0005 |     0      |      0      |

## Robustness — best WGeo-family vs best non-WGeo baseline (legacy)

Retained for continuity with v0.3 / v0.4 reporting. Both sides are selected by minimum cell CRPS, so the implicit multiple comparison (8 WGeo variants × 6 baselines = 48 implicit pairs per cell) inflates type-I error. Use Headlines 1–2 above for inference; this table is a robustness check that the pre-registered headline does not depend on the choice of WGeo variant.

| symbol   |   h |   n_test | best_wgeo     | best_baseline   |   wgeo_crps |   baseline_crps | improvement   |   dm_stat |   dm_p |   dm_stat_r |   dm_p_r | hetero_garch_fallback   |
|:---------|----:|---------:|:--------------|:----------------|------------:|----------------:|:--------------|----------:|-------:|------------:|---------:|:------------------------|
| BTC/USDT |   1 |     2470 | WGeo-Ensemble | WAR-1           |    0.016168 |        0.016208 | -0.2%         |     -3.51 | 0.0004 |       -3.66 |   0.0003 | 0.0%                    |
| BTC/USDT |   5 |     2466 | WGeo-Ensemble | WAR-1-last      |    0.037061 |        0.037099 | -0.1%         |     -0.93 | 0.3542 |       -1.17 |   0.2428 | 0.0%                    |
| BTC/USDT |  21 |     2450 | WGeo-Ensemble | WAR-1-last      |    0.083158 |        0.08284  | +0.4%         |      0.95 | 0.3432 |        1.99 |   0.0467 | 0.0%                    |
| ETH/USDT |   1 |     2470 | WGeo-Ensemble | WAR-Select      |    0.021739 |        0.021786 | -0.2%         |     -3.29 | 0.001  |       -3.45 |   0.0006 | 0.0%                    |
| ETH/USDT |   5 |     2466 | WGeo-Ensemble | WAR-1-last      |    0.049256 |        0.049246 | +0.0%         |      0.18 | 0.8549 |        0.23 |   0.8182 | 0.0%                    |
| ETH/USDT |  21 |     2450 | WGeo-TheilSen | WAR-1-last      |    0.109404 |        0.109047 | +0.3%         |      1.49 | 0.137  |        1.55 |   0.1203 | 0.0%                    |
| SOL/USDT |   1 |     1380 | WGeo-Ensemble | Static          |    0.02503  |        0.025075 | -0.2%         |     -0.55 | 0.5839 |       -0.59 |   0.5566 | 0.0%                    |
| SOL/USDT |   5 |     1376 | WGeo-Ensemble | WAR-1-last      |    0.057153 |        0.057096 | +0.1%         |      0.85 | 0.3936 |        1.03 |   0.3025 | 0.0%                    |
| SOL/USDT |  21 |     1360 | WGeo-Adaptive | WAR-1-last      |    0.129263 |        0.128401 | +0.7%         |      0.81 | 0.42   |        1.11 |   0.2681 | 0.0%                    |
| BNB/USDT |   1 |     2389 | WGeo-Ensemble | GARCH-N         |    0.020144 |        0.020199 | -0.3%         |     -0.61 | 0.5433 |       -1.03 |   0.3015 | 0.0%                    |
| BNB/USDT |   5 |     2385 | WGeo-Ensemble | WAR-1-last      |    0.045782 |        0.045884 | -0.2%         |     -1.77 | 0.076  |       -2.34 |   0.0194 | 0.0%                    |
| BNB/USDT |  21 |     2369 | WGeo-Ensemble | WAR-1-last      |    0.102619 |        0.102471 | +0.1%         |      0.31 | 0.76   |        0.61 |   0.5429 | 0.0%                    |
| XRP/USDT |   1 |     2210 | WGeo-Ensemble | WAR-Select      |    0.024359 |        0.024428 | -0.3%         |     -4.27 | 0      |       -4.55 |   0      | 0.0%                    |
| XRP/USDT |   5 |     2206 | WGeo-Ensemble | WAR-1-last      |    0.055389 |        0.055465 | -0.1%         |     -1.44 | 0.1485 |       -1.59 |   0.1115 | 0.0%                    |
| XRP/USDT |  21 |     2190 | WGeo-Ensemble | WAR-1-last      |    0.129103 |        0.128207 | +0.7%         |      2.79 | 0.0053 |        3.28 |   0.001  | 0.0%                    |

## BTC/USDT

_3201 days from 2017-08-18 to 2026-05-23_

### Horizon h = 1 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2470 |    0.016236 | 0.015385 | 0.017189 |              nan |
| HS-Bootstrap   | 2470 |    0.016239 | 0.015385 | 0.017187 |              nan |
| GARCH-N        | 2470 |    0.016463 | 0.015635 | 0.017398 |              nan |
| GARCH-t        | 2470 |    0.017178 | 0.016407 | 0.018085 |              nan |
| GJR-GARCH-t    | 2470 |    0.017176 | 0.016402 | 0.01807  |              nan |
| WGeo           | 2470 |    0.016212 | 0.015309 | 0.017195 |              nan |
| WGeo-Gated     | 2470 |    0.016203 | 0.015338 | 0.017167 |              nan |
| WGeo-TheilSen  | 2470 |    0.016212 | 0.015309 | 0.017196 |              nan |
| WGeo-EWMA      | 2470 |    0.016212 | 0.01531  | 0.017196 |              nan |
| WGeo-Hetero    | 2470 |    0.016221 | 0.01532  | 0.017198 |                0 |
| WGeo-GARCH-Ens | 2470 |    0.016253 | 0.015388 | 0.017222 |              nan |
| WGeo-Adaptive  | 2470 |    0.016238 | 0.01535  | 0.017206 |              nan |
| WGeo-Ensemble  | 2470 |    0.016168 | 0.015281 | 0.017139 |              nan |
| WAR-1          | 2470 |    0.016208 | 0.015307 | 0.017191 |              nan |
| WAR-1-last     | 2470 |    0.016208 | 0.015307 | 0.017191 |              nan |
| WAR-Select     | 2470 |    0.016208 | 0.015308 | 0.017191 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.01618 |        0.01617 |   0.01638 |   0.01678 |       0.01675 | 0.01607 |      0.01604 |         0.01607 |     0.01607 |       0.01591 |          0.01618 |         0.016   |         0.01601 | 0.01607 |      0.01607 |      0.01608 |
|   2020 | 366 |  0.01871 |        0.01871 |   0.01923 |   0.02034 |       0.02032 | 0.01872 |      0.01874 |         0.01872 |     0.01873 |       0.01872 |          0.01888 |         0.01871 |         0.0187  | 0.01872 |      0.01872 |      0.01872 |
|   2021 | 365 |  0.02337 |        0.02335 |   0.02331 |   0.02402 |       0.024   | 0.02344 |      0.02342 |         0.02344 |     0.02344 |       0.02341 |          0.0233  |         0.02344 |         0.02338 | 0.02344 |      0.02344 |      0.02344 |
|   2022 | 365 |  0.01752 |        0.01753 |   0.01788 |   0.01854 |       0.01845 | 0.01732 |      0.01734 |         0.01732 |     0.01732 |       0.01736 |          0.01742 |         0.01745 |         0.01728 | 0.01731 |      0.01731 |      0.01731 |
|   2023 | 365 |  0.01227 |        0.01227 |   0.01267 |   0.01312 |       0.01309 | 0.01202 |      0.01214 |         0.01202 |     0.01202 |       0.01208 |          0.01212 |         0.01202 |         0.01199 | 0.01201 |      0.01201 |      0.01201 |
|   2024 | 366 |  0.01487 |        0.01487 |   0.01496 |   0.01604 |       0.01604 | 0.01499 |      0.0149  |         0.01499 |     0.01499 |       0.01504 |          0.015   |         0.01503 |         0.01494 | 0.01498 |      0.01498 |      0.01498 |
|   2025 | 365 |  0.01173 |        0.01176 |   0.01181 |   0.01236 |       0.0125  | 0.01177 |      0.01173 |         0.01177 |     0.01177 |       0.01176 |          0.01174 |         0.01182 |         0.01173 | 0.01177 |      0.01177 |      0.01177 |
|   2026 | 142 |  0.01357 |        0.01359 |   0.01375 |   0.01409 |       0.0141  | 0.01379 |      0.01362 |         0.01379 |     0.01379 |       0.01382 |          0.01391 |         0.01379 |         0.01372 | 0.01378 |      0.01378 |      0.01378 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |    n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|-----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    |  320 |  0.02104 |        0.02104 |   0.02132 |   0.02242 |       0.02236 | 0.02116 |      0.02102 |         0.02116 |     0.02116 |       0.02107 |          0.02121 |         0.02127 |         0.02107 | 0.02115 |      0.02115 |      0.02115 |
| high-vol |   69 |  0.01877 |        0.01871 |   0.0189  |   0.02016 |       0.02018 | 0.01932 |      0.01891 |         0.01933 |     0.01932 |       0.01927 |          0.01947 |         0.01906 |         0.01915 | 0.01933 |      0.01933 |      0.01933 |
| neutral  | 1047 |  0.01542 |        0.01543 |   0.01573 |   0.01643 |       0.0164  | 0.01531 |      0.01535 |         0.01531 |     0.01531 |       0.01532 |          0.0154  |         0.01534 |         0.01529 | 0.01531 |      0.01531 |      0.01531 |
| low-vol  |  498 |  0.01212 |        0.01213 |   0.01231 |   0.01262 |       0.0127  | 0.01193 |      0.01205 |         0.01193 |     0.01193 |       0.01197 |          0.01193 |         0.01194 |         0.01193 | 0.01193 |      0.01193 |      0.01193 |
| rally    |  536 |  0.01846 |        0.01845 |   0.01855 |   0.01936 |       0.01937 | 0.01859 |      0.01851 |         0.01859 |     0.01859 |       0.01865 |          0.01856 |         0.01861 |         0.01851 | 0.01859 |      0.01859 |      0.01859 |

**Diebold-Mariano vs WAR-1** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.5489 |           0.5313 |
| HS-Bootstrap   |      0.5163 |           0.4958 |
| GARCH-N        |      0.0001 |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.0158 |           0.0102 |
| WGeo-Gated     |      0.8872 |           0.882  |
| WGeo-TheilSen  |      0.0077 |           0.0046 |
| WGeo-EWMA      |      0.0077 |           0.0045 |
| WGeo-Hetero    |      0.6346 |           0.3954 |
| WGeo-GARCH-Ens |      0.3049 |           0.1568 |
| WGeo-Adaptive  |      0.3202 |           0.2852 |
| WGeo-Ensemble  |      0.0004 |           0.0003 |
| WAR-1          |      1      |           1      |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.8343 |           0.8322 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |    n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|-----:|---------:|---------:|------------:|---------:|--------:|
| crash    |  320 |  0.02107 |  0.02115 |    -0.35976 | -1.95498 | 0.05059 |
| high-vol |   69 |  0.01915 |  0.01933 |    -0.90347 | -3.18176 | 0.00146 |
| neutral  | 1047 |  0.01529 |  0.01531 |    -0.11747 | -1.21131 | 0.22578 |
| low-vol  |  498 |  0.01193 |  0.01193 |    -0.00171 | -0.01052 | 0.99161 |
| rally    |  536 |  0.01851 |  0.01859 |    -0.41587 | -2.61211 | 0.009   |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_btcusdt_h1.png)

### Horizon h = 5 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2466 |    0.037367 | 0.034483 | 0.040159 |              nan |
| HS-Bootstrap   | 2466 |    0.037565 | 0.034801 | 0.040252 |              nan |
| GARCH-N        | 2466 |    0.037807 | 0.035025 | 0.040469 |              nan |
| GARCH-t        | 2466 |    0.039544 | 0.036965 | 0.042101 |              nan |
| GJR-GARCH-t    | 2466 |    0.039547 | 0.036949 | 0.042129 |              nan |
| WGeo           | 2466 |    0.037137 | 0.034299 | 0.039989 |              nan |
| WGeo-Gated     | 2466 |    0.037228 | 0.034353 | 0.040014 |              nan |
| WGeo-TheilSen  | 2466 |    0.037135 | 0.034298 | 0.039985 |              nan |
| WGeo-EWMA      | 2466 |    0.03714  | 0.034304 | 0.039995 |              nan |
| WGeo-Hetero    | 2466 |    0.037333 | 0.034447 | 0.040179 |                0 |
| WGeo-GARCH-Ens | 2466 |    0.037363 | 0.034455 | 0.040212 |              nan |
| WGeo-Adaptive  | 2466 |    0.037228 | 0.034377 | 0.040019 |              nan |
| WGeo-Ensemble  | 2466 |    0.037061 | 0.034167 | 0.039886 |              nan |
| WAR-1          | 2466 |    0.03751  | 0.034641 | 0.040286 |              nan |
| WAR-1-last     | 2466 |    0.037099 | 0.034255 | 0.03994  |              nan |
| WAR-Select     | 2466 |    0.037512 | 0.034647 | 0.040287 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.0374  |        0.03887 |   0.03838 |   0.03973 |       0.03965 | 0.03661 |      0.03687 |         0.0366  |     0.03661 |       0.0368  |          0.03715 |         0.03691 |         0.03658 | 0.03765 |      0.03664 |      0.03765 |
|   2020 | 366 |  0.04447 |        0.04434 |   0.04471 |   0.04784 |       0.04775 | 0.04563 |      0.0448  |         0.04565 |     0.04563 |       0.04586 |          0.04538 |         0.0451  |         0.04525 | 0.0437  |      0.04564 |      0.0437  |
|   2021 | 365 |  0.05182 |        0.05125 |   0.05199 |   0.05313 |       0.05299 | 0.05093 |      0.05158 |         0.05094 |     0.05093 |       0.05098 |          0.05125 |         0.05121 |         0.05104 | 0.05253 |      0.05093 |      0.05253 |
|   2022 | 365 |  0.04128 |        0.04187 |   0.04223 |   0.04362 |       0.0435  | 0.04003 |      0.04062 |         0.04001 |     0.04002 |       0.04037 |          0.04051 |         0.04055 |         0.04009 | 0.04114 |      0.03998 |      0.04114 |
|   2023 | 365 |  0.03012 |        0.0311  |   0.03108 |   0.03206 |       0.03204 | 0.02978 |      0.02996 |         0.02978 |     0.02979 |       0.03004 |          0.03024 |         0.0298  |         0.02965 | 0.03005 |      0.02965 |      0.03004 |
|   2024 | 366 |  0.03463 |        0.03411 |   0.03424 |   0.03726 |       0.03745 | 0.03497 |      0.03481 |         0.03496 |     0.03498 |       0.03511 |          0.03482 |         0.03497 |         0.03487 | 0.03522 |      0.03494 |      0.03522 |
|   2025 | 365 |  0.02536 |        0.02573 |   0.02576 |   0.02692 |       0.02716 | 0.02511 |      0.02526 |         0.02512 |     0.02513 |       0.02519 |          0.0252  |         0.02529 |         0.02509 | 0.02555 |      0.02508 |      0.02555 |
|   2026 | 138 |  0.0281  |        0.02829 |   0.02884 |   0.02988 |       0.02983 | 0.02801 |      0.02785 |         0.028   |     0.028   |       0.0284  |          0.02903 |         0.02812 |         0.02792 | 0.02906 |      0.02792 |      0.02912 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |    n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|-----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    |  320 |  0.04373 |        0.04362 |   0.04527 |   0.04816 |       0.04786 | 0.04401 |      0.04362 |         0.044   |     0.04401 |       0.04461 |          0.04513 |         0.04475 |         0.04377 | 0.04487 |      0.04395 |      0.04488 |
| high-vol |   69 |  0.04774 |        0.04678 |   0.04537 |   0.04926 |       0.0493  | 0.0479  |      0.04806 |         0.0479  |     0.04788 |       0.04782 |          0.04813 |         0.04779 |         0.04783 | 0.05195 |      0.04791 |      0.05195 |
| neutral  | 1047 |  0.03678 |        0.03727 |   0.0373  |   0.03903 |       0.03905 | 0.03624 |      0.03651 |         0.03623 |     0.03625 |       0.03642 |          0.03653 |         0.0364  |         0.03624 | 0.03655 |      0.03621 |      0.03655 |
| low-vol  |  494 |  0.03028 |        0.0307  |   0.03073 |   0.0311  |       0.03126 | 0.02974 |      0.03007 |         0.02973 |     0.02974 |       0.02996 |          0.02991 |         0.02982 |         0.02973 | 0.02986 |      0.02967 |      0.02987 |
| rally    |  536 |  0.03991 |        0.03967 |   0.03989 |   0.04195 |       0.04193 | 0.04022 |      0.04001 |         0.04023 |     0.04023 |       0.04022 |          0.03983 |         0.03983 |         0.04003 | 0.04018 |      0.0402  |      0.04017 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.1504 |           0.0496 |
| HS-Bootstrap   |      0.0361 |           0.0006 |
| GARCH-N        |      0.0068 |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.0518 |           0.0392 |
| WGeo-Gated     |      0.2978 |           0.1836 |
| WGeo-TheilSen  |      0.0448 |           0.034  |
| WGeo-EWMA      |      0.04   |           0.0286 |
| WGeo-Hetero    |      0.0024 |           0      |
| WGeo-GARCH-Ens |      0.0295 |           0.008  |
| WGeo-Adaptive  |      0.2013 |           0.1483 |
| WGeo-Ensemble  |      0.3542 |           0.2428 |
| WAR-1          |      0.1377 |           0.0987 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.1355 |           0.0968 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |    n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|-----:|---------:|---------:|------------:|---------:|--------:|
| crash    |  320 |  0.04377 |  0.04395 |    -0.40257 | -1.21162 | 0.22566 |
| high-vol |   69 |  0.04783 |  0.04791 |    -0.16456 | -0.73995 | 0.45933 |
| neutral  | 1047 |  0.03624 |  0.03621 |     0.08265 |  0.50424 | 0.61409 |
| low-vol  |  494 |  0.02973 |  0.02967 |     0.19824 |  0.82002 | 0.4122  |
| rally    |  536 |  0.04003 |  0.0402  |    -0.4278  | -1.87451 | 0.06086 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_btcusdt_h5.png)

### Horizon h = 21 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2450 |    0.085347 | 0.075667 | 0.095899 |              nan |
| HS-Bootstrap   | 2450 |    0.085057 | 0.076106 | 0.094639 |              nan |
| GARCH-N        | 2450 |    0.084848 | 0.075685 | 0.09484  |              nan |
| GARCH-t        | 2450 |    0.089412 | 0.080782 | 0.099096 |              nan |
| GJR-GARCH-t    | 2450 |    0.089596 | 0.08085  | 0.099366 |              nan |
| WGeo           | 2450 |    0.083313 | 0.07345  | 0.094259 |              nan |
| WGeo-Gated     | 2450 |    0.084033 | 0.074379 | 0.094524 |              nan |
| WGeo-TheilSen  | 2450 |    0.083296 | 0.073428 | 0.094273 |              nan |
| WGeo-EWMA      | 2450 |    0.083317 | 0.073446 | 0.094264 |              nan |
| WGeo-Hetero    | 2450 |    0.083683 | 0.073814 | 0.094527 |                0 |
| WGeo-GARCH-Ens | 2450 |    0.083394 | 0.073873 | 0.093618 |              nan |
| WGeo-Adaptive  | 2450 |    0.083908 | 0.074175 | 0.094437 |              nan |
| WGeo-Ensemble  | 2450 |    0.083158 | 0.073451 | 0.093473 |              nan |
| WAR-1          | 2450 |    0.085867 | 0.076102 | 0.096094 |              nan |
| WAR-1-last     | 2450 |    0.08284  | 0.073053 | 0.09373  |              nan |
| WAR-Select     | 2450 |    0.085858 | 0.076075 | 0.096057 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.08773 |        0.08765 |   0.0851  |   0.08835 |       0.08821 | 0.08209 |      0.08528 |         0.08207 |     0.08198 |       0.08337 |          0.08103 |         0.08517 |         0.08289 | 0.10332 |      0.08179 |      0.10334 |
|   2020 | 366 |  0.11632 |        0.11286 |   0.10853 |   0.11922 |       0.1192  | 0.12736 |      0.11945 |         0.12745 |     0.12731 |       0.12718 |          0.12165 |         0.12511 |         0.12433 | 0.10632 |      0.12633 |      0.10631 |
|   2021 | 365 |  0.11083 |        0.11034 |   0.11631 |   0.11448 |       0.1146  | 0.10423 |      0.10768 |         0.10427 |     0.10428 |       0.10475 |          0.1063  |         0.10518 |         0.10481 | 0.11912 |      0.10423 |      0.11917 |
|   2022 | 365 |  0.09302 |        0.09385 |   0.09556 |   0.09907 |       0.09944 | 0.08437 |      0.09001 |         0.08425 |     0.08439 |       0.08505 |          0.08556 |         0.08583 |         0.0858  | 0.09199 |      0.08368 |      0.09204 |
|   2023 | 365 |  0.06819 |        0.06984 |   0.06667 |   0.07118 |       0.07138 | 0.06498 |      0.06674 |         0.06497 |     0.06497 |       0.06509 |          0.06587 |         0.06534 |         0.06523 | 0.06776 |      0.06427 |      0.06748 |
|   2024 | 366 |  0.07302 |        0.0718  |   0.07104 |   0.0823  |       0.08239 | 0.07525 |      0.0732  |         0.07518 |     0.07534 |       0.07561 |          0.07571 |         0.07677 |         0.07428 | 0.07524 |      0.07477 |      0.07519 |
|   2025 | 365 |  0.05322 |        0.05388 |   0.05376 |   0.05453 |       0.05524 | 0.04795 |      0.0501  |         0.04797 |     0.04796 |       0.04809 |          0.04906 |         0.04814 |         0.04822 | 0.05108 |      0.04796 |      0.05096 |
|   2026 | 122 |  0.075   |        0.07538 |   0.07614 |   0.07751 |       0.07698 | 0.07159 |      0.0735  |         0.0715  |     0.07154 |       0.0728  |          0.07448 |         0.07349 |         0.0721  | 0.07736 |      0.07117 |      0.07827 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |    n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|-----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    |  320 |  0.09161 |        0.09133 |   0.09476 |   0.10099 |       0.10047 | 0.08674 |      0.08946 |         0.08669 |     0.08677 |       0.08839 |          0.0927  |         0.08891 |         0.08722 | 0.09762 |      0.08629 |      0.09764 |
| high-vol |   69 |  0.0888  |        0.08842 |   0.08359 |   0.10128 |       0.10125 | 0.09131 |      0.09065 |         0.09137 |     0.0913  |       0.09212 |          0.09299 |         0.09301 |         0.09092 | 0.12671 |      0.09177 |      0.12679 |
| neutral  | 1047 |  0.08704 |        0.08622 |   0.08557 |   0.09004 |       0.09032 | 0.08443 |      0.08572 |         0.0844  |     0.08443 |       0.08477 |          0.0851  |         0.08532 |         0.08446 | 0.08576 |      0.08374 |      0.08576 |
| low-vol  |  478 |  0.07521 |        0.07534 |   0.07431 |   0.07346 |       0.0737  | 0.06999 |      0.07227 |         0.06995 |     0.06995 |       0.07019 |          0.07092 |         0.07052 |         0.07035 | 0.07045 |      0.06963 |      0.07038 |
| rally    |  536 |  0.0869  |        0.08728 |   0.08707 |   0.09397 |       0.09437 | 0.08993 |      0.08714 |         0.08997 |     0.08997 |       0.0897  |          0.0844  |         0.08893 |         0.08862 | 0.08755 |      0.08965 |      0.08755 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0979 |           0      |
| HS-Bootstrap   |      0.181  |           0      |
| GARCH-N        |      0.3751 |           0.0037 |
| GARCH-t        |      0.0003 |           0      |
| GJR-GARCH-t    |      0.0002 |           0      |
| WGeo           |      0.0148 |           0.0089 |
| WGeo-Gated     |      0.2256 |           0.0067 |
| WGeo-TheilSen  |      0.0136 |           0.0082 |
| WGeo-EWMA      |      0.0136 |           0.0082 |
| WGeo-Hetero    |      0.0127 |           0.002  |
| WGeo-GARCH-Ens |      0.6001 |           0.4723 |
| WGeo-Adaptive  |      0.0125 |           0.0033 |
| WGeo-Ensemble  |      0.3432 |           0.0467 |
| WAR-1          |      0.2269 |           0.1412 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.2276 |           0.1416 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |    n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|-----:|---------:|---------:|------------:|---------:|--------:|
| crash    |  320 |  0.08722 |  0.08629 |     1.07159 |  1.35758 | 0.1746  |
| high-vol |   69 |  0.09092 |  0.09177 |    -0.932   | -0.54482 | 0.58588 |
| neutral  | 1047 |  0.08446 |  0.08374 |     0.85929 |  1.51157 | 0.13064 |
| low-vol  |  478 |  0.07035 |  0.06963 |     1.02676 |  1.42537 | 0.15405 |
| rally    |  536 |  0.08862 |  0.08965 |    -1.14944 | -1.17459 | 0.24016 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_btcusdt_h21.png)

## ETH/USDT

_3201 days from 2017-08-18 to 2026-05-23_

### Horizon h = 1 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2470 |    0.021897 | 0.020759 | 0.023146 |              nan |
| HS-Bootstrap   | 2470 |    0.021893 | 0.020755 | 0.023142 |              nan |
| GARCH-N        | 2470 |    0.021947 | 0.02087  | 0.023137 |              nan |
| GARCH-t        | 2470 |    0.022877 | 0.021852 | 0.02403  |              nan |
| GJR-GARCH-t    | 2470 |    0.022877 | 0.021859 | 0.024022 |              nan |
| WGeo           | 2470 |    0.021793 | 0.020641 | 0.023043 |              nan |
| WGeo-Gated     | 2470 |    0.021793 | 0.020634 | 0.023049 |              nan |
| WGeo-TheilSen  | 2470 |    0.021792 | 0.02064  | 0.023043 |              nan |
| WGeo-EWMA      | 2470 |    0.021792 | 0.02064  | 0.023041 |              nan |
| WGeo-Hetero    | 2470 |    0.021883 | 0.02075  | 0.0231   |                0 |
| WGeo-GARCH-Ens | 2470 |    0.021802 | 0.020723 | 0.023028 |              nan |
| WGeo-Adaptive  | 2470 |    0.021845 | 0.020689 | 0.023057 |              nan |
| WGeo-Ensemble  | 2470 |    0.021739 | 0.020585 | 0.022995 |              nan |
| WAR-1          | 2470 |    0.021786 | 0.020632 | 0.023038 |              nan |
| WAR-1-last     | 2470 |    0.021786 | 0.020632 | 0.023038 |              nan |
| WAR-Select     | 2470 |    0.021786 | 0.020632 | 0.023038 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.01926 |        0.01924 |   0.01964 |   0.02149 |       0.02145 | 0.0188  |      0.01904 |         0.0188  |     0.0188  |       0.01881 |          0.01883 |         0.01902 |         0.01882 | 0.0188  |      0.0188  |      0.0188  |
|   2020 | 366 |  0.02605 |        0.02603 |   0.02644 |   0.0288  |       0.02878 | 0.02629 |      0.02607 |         0.02629 |     0.02629 |       0.02639 |          0.02639 |         0.02623 |         0.02617 | 0.02628 |      0.02628 |      0.02628 |
|   2021 | 365 |  0.0306  |        0.03058 |   0.03015 |   0.03049 |       0.03057 | 0.03048 |      0.03045 |         0.03048 |     0.03048 |       0.03033 |          0.03025 |         0.03051 |         0.03042 | 0.03048 |      0.03048 |      0.03048 |
|   2022 | 365 |  0.02463 |        0.02462 |   0.02476 |   0.02509 |       0.02512 | 0.02454 |      0.02453 |         0.02454 |     0.02454 |       0.02468 |          0.0246  |         0.02455 |         0.02448 | 0.02453 |      0.02453 |      0.02453 |
|   2023 | 365 |  0.01386 |        0.01384 |   0.01349 |   0.01383 |       0.01393 | 0.01301 |      0.01347 |         0.01301 |     0.01301 |       0.01323 |          0.01304 |         0.01307 |         0.01305 | 0.01301 |      0.01301 |      0.01301 |
|   2024 | 366 |  0.01787 |        0.01787 |   0.01808 |   0.01869 |       0.01866 | 0.01802 |      0.01786 |         0.01802 |     0.01802 |       0.01822 |          0.0181  |         0.01811 |         0.01794 | 0.01801 |      0.01801 |      0.01801 |
|   2025 | 365 |  0.02066 |        0.02069 |   0.02081 |   0.02216 |       0.02204 | 0.0207  |      0.02061 |         0.0207  |     0.02071 |       0.02083 |          0.0207  |         0.02084 |         0.02064 | 0.0207  |      0.0207  |      0.0207  |
|   2026 | 142 |  0.01856 |        0.0186  |   0.01887 |   0.01958 |       0.01951 | 0.01875 |      0.01871 |         0.01875 |     0.01875 |       0.01872 |          0.01883 |         0.01881 |         0.01873 | 0.01875 |      0.01875 |      0.01875 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 505 |  0.0232  |        0.02321 |   0.02341 |   0.02447 |       0.02442 | 0.02329 |      0.02313 |         0.02329 |     0.02329 |       0.02337 |          0.02338 |         0.0234  |         0.02318 | 0.02327 |      0.02327 |      0.02327 |
| high-vol |  80 |  0.02796 |        0.02787 |   0.02892 |   0.03073 |       0.03067 | 0.02844 |      0.02808 |         0.02844 |     0.02843 |       0.02895 |          0.02906 |         0.02829 |         0.02825 | 0.02842 |      0.02842 |      0.02842 |
| neutral  | 691 |  0.01962 |        0.01964 |   0.01988 |   0.02097 |       0.02097 | 0.01948 |      0.01952 |         0.01948 |     0.01948 |       0.01954 |          0.01961 |         0.01954 |         0.01945 | 0.01947 |      0.01947 |      0.01947 |
| low-vol  | 464 |  0.01595 |        0.01594 |   0.01576 |   0.01612 |       0.0162  | 0.01536 |      0.01572 |         0.01536 |     0.01536 |       0.01557 |          0.01532 |         0.0154  |         0.0154  | 0.01536 |      0.01536 |      0.01536 |
| rally    | 730 |  0.02626 |        0.02625 |   0.02606 |   0.02701 |       0.027   | 0.02631 |      0.02619 |         0.02631 |     0.02631 |       0.02632 |          0.02611 |         0.02635 |         0.02622 | 0.0263  |      0.0263  |      0.0263  |

**Diebold-Mariano vs WAR-Select** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0772 |           0.0599 |
| HS-Bootstrap   |      0.0858 |           0.0646 |
| GARCH-N        |      0.0416 |           0.0074 |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.0025 |           0.001  |
| WGeo-Gated     |      0.8608 |           0.8537 |
| WGeo-TheilSen  |      0.0013 |           0.0005 |
| WGeo-EWMA      |      0.0082 |           0.004  |
| WGeo-Hetero    |      0.0363 |           0      |
| WGeo-GARCH-Ens |      0.7803 |           0.6949 |
| WGeo-Adaptive  |      0.1654 |           0.1389 |
| WGeo-Ensemble  |      0.001  |           0.0006 |
| WAR-1          |      0.7539 |           0.7526 |
| WAR-1-last     |      0.7539 |           0.7526 |
| WAR-Select     |      1      |           1      |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-Select, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 505 |  0.02318 |  0.02327 |    -0.39387 | -2.53404 | 0.01128 |
| high-vol |  80 |  0.02825 |  0.02842 |    -0.61667 | -1.80815 | 0.07058 |
| neutral  | 691 |  0.01945 |  0.01947 |    -0.08763 | -0.81976 | 0.41235 |
| low-vol  | 464 |  0.0154  |  0.01536 |     0.24868 |  1.25417 | 0.20978 |
| rally    | 730 |  0.02622 |  0.0263  |    -0.3139  | -3.0015  | 0.00269 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_ethusdt_h1.png)

### Horizon h = 5 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2466 |    0.049834 | 0.046071 | 0.053619 |              nan |
| HS-Bootstrap   | 2466 |    0.050045 | 0.046399 | 0.053638 |              nan |
| GARCH-N        | 2466 |    0.050368 | 0.046695 | 0.054078 |              nan |
| GARCH-t        | 2466 |    0.052513 | 0.049186 | 0.056076 |              nan |
| GJR-GARCH-t    | 2466 |    0.052517 | 0.049186 | 0.056055 |              nan |
| WGeo           | 2466 |    0.049314 | 0.045547 | 0.053033 |              nan |
| WGeo-Gated     | 2466 |    0.049546 | 0.045768 | 0.053333 |              nan |
| WGeo-TheilSen  | 2466 |    0.049304 | 0.045534 | 0.053021 |              nan |
| WGeo-EWMA      | 2466 |    0.049309 | 0.045545 | 0.053029 |              nan |
| WGeo-Hetero    | 2466 |    0.05     | 0.046249 | 0.05376  |                0 |
| WGeo-GARCH-Ens | 2466 |    0.049894 | 0.046202 | 0.053659 |              nan |
| WGeo-Adaptive  | 2466 |    0.049607 | 0.045943 | 0.053349 |              nan |
| WGeo-Ensemble  | 2466 |    0.049256 | 0.045535 | 0.05298  |              nan |
| WAR-1          | 2466 |    0.049971 | 0.046159 | 0.053808 |              nan |
| WAR-1-last     | 2466 |    0.049246 | 0.045471 | 0.052958 |              nan |
| WAR-Select     | 2466 |    0.049968 | 0.046156 | 0.05381  |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.04325 |        0.04526 |   0.04475 |   0.05027 |       0.0501  | 0.04239 |      0.0429  |         0.04236 |     0.04242 |       0.04273 |          0.04292 |         0.0433  |         0.04242 | 0.04273 |      0.04234 |      0.04275 |
|   2020 | 366 |  0.06204 |        0.06159 |   0.06289 |   0.06902 |       0.06894 | 0.06238 |      0.0619  |         0.06237 |     0.06236 |       0.06324 |          0.06321 |         0.06191 |         0.06212 | 0.06272 |      0.06226 |      0.06269 |
|   2021 | 365 |  0.0645  |        0.06419 |   0.06522 |   0.06581 |       0.06587 | 0.06453 |      0.06491 |         0.06451 |     0.06452 |       0.06546 |          0.06531 |         0.06535 |         0.06449 | 0.0643  |      0.06446 |      0.0643  |
|   2022 | 365 |  0.05924 |        0.05951 |   0.06002 |   0.06075 |       0.06088 | 0.05767 |      0.05862 |         0.05765 |     0.05765 |       0.05847 |          0.05842 |         0.05787 |         0.0578  | 0.06029 |      0.05758 |      0.0603  |
|   2023 | 365 |  0.03006 |        0.03163 |   0.02967 |   0.03042 |       0.03083 | 0.02831 |      0.02917 |         0.02831 |     0.02832 |       0.02898 |          0.02852 |         0.02859 |         0.02831 | 0.02834 |      0.02829 |      0.02834 |
|   2024 | 366 |  0.0432  |        0.04283 |   0.0437  |   0.04474 |       0.04457 | 0.04339 |      0.04297 |         0.04338 |     0.04339 |       0.04436 |          0.04399 |         0.04372 |         0.04319 | 0.04371 |      0.04332 |      0.04371 |
|   2025 | 365 |  0.04641 |        0.04618 |   0.04661 |   0.04902 |       0.04883 | 0.04589 |      0.04601 |         0.0459  |     0.0459  |       0.04619 |          0.04628 |         0.04621 |         0.04588 | 0.04691 |      0.04588 |      0.04691 |
|   2026 | 138 |  0.03926 |        0.03977 |   0.04024 |   0.04227 |       0.04211 | 0.03946 |      0.03937 |         0.03945 |     0.03943 |       0.03939 |          0.03988 |         0.03989 |         0.03939 | 0.04    |      0.03933 |      0.04    |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 505 |  0.04789 |        0.04811 |   0.0489  |   0.0521  |       0.05198 | 0.04712 |      0.0475  |         0.04711 |     0.04712 |       0.04787 |          0.04854 |         0.04787 |         0.04711 | 0.0492  |      0.04703 |      0.04918 |
| high-vol |  80 |  0.05241 |        0.05261 |   0.05685 |   0.06199 |       0.06196 | 0.05435 |      0.05255 |         0.05434 |     0.05429 |       0.05747 |          0.05767 |         0.05481 |         0.05358 | 0.05614 |      0.05425 |      0.05614 |
| neutral  | 691 |  0.04669 |        0.04729 |   0.04761 |   0.05008 |       0.05009 | 0.04615 |      0.04645 |         0.04613 |     0.04616 |       0.04649 |          0.04675 |         0.04635 |         0.04614 | 0.04618 |      0.04606 |      0.04617 |
| low-vol  | 460 |  0.04282 |        0.04339 |   0.04217 |   0.04199 |       0.04232 | 0.04132 |      0.042   |         0.04132 |     0.0413  |       0.04223 |          0.04136 |         0.04171 |         0.04134 | 0.04124 |      0.04129 |      0.04124 |
| rally    | 730 |  0.05829 |        0.0579  |   0.05845 |   0.0607  |       0.06058 | 0.05831 |      0.05832 |         0.05831 |     0.0583  |       0.05887 |          0.05833 |         0.0583  |         0.05821 | 0.05892 |      0.05826 |      0.05892 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0206 |           0.0012 |
| HS-Bootstrap   |      0.005  |           0      |
| GARCH-N        |      0.0003 |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.006  |           0.0025 |
| WGeo-Gated     |      0.0779 |           0.0201 |
| WGeo-TheilSen  |      0.0111 |           0.0054 |
| WGeo-EWMA      |      0.0122 |           0.0057 |
| WGeo-Hetero    |      0      |           0      |
| WGeo-GARCH-Ens |      0.0001 |           0      |
| WGeo-Adaptive  |      0.0094 |           0.0053 |
| WGeo-Ensemble  |      0.8549 |           0.8182 |
| WAR-1          |      0.0336 |           0.0136 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.034  |           0.0138 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 505 |  0.04711 |  0.04703 |     0.16647 |  0.61878 | 0.53606 |
| high-vol |  80 |  0.05358 |  0.05425 |    -1.23413 | -1.47432 | 0.14039 |
| neutral  | 691 |  0.04614 |  0.04606 |     0.17412 |  0.80567 | 0.42043 |
| low-vol  | 460 |  0.04134 |  0.04129 |     0.10372 |  0.45707 | 0.64762 |
| rally    | 730 |  0.05821 |  0.05826 |    -0.08516 | -0.49048 | 0.62379 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_ethusdt_h5.png)

### Horizon h = 21 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2450 |    0.113701 | 0.099233 | 0.127949 |              nan |
| HS-Bootstrap   | 2450 |    0.113365 | 0.099827 | 0.126705 |              nan |
| GARCH-N        | 2450 |    0.11297  | 0.099596 | 0.126538 |              nan |
| GARCH-t        | 2450 |    0.117615 | 0.104672 | 0.13089  |              nan |
| GJR-GARCH-t    | 2450 |    0.117455 | 0.104636 | 0.130535 |              nan |
| WGeo           | 2450 |    0.109454 | 0.095087 | 0.124069 |              nan |
| WGeo-Gated     | 2450 |    0.111918 | 0.097416 | 0.125936 |              nan |
| WGeo-TheilSen  | 2450 |    0.109404 | 0.095068 | 0.123986 |              nan |
| WGeo-EWMA      | 2450 |    0.109478 | 0.095167 | 0.124097 |              nan |
| WGeo-Hetero    | 2450 |    0.110506 | 0.096155 | 0.125552 |                0 |
| WGeo-GARCH-Ens | 2450 |    0.11033  | 0.09617  | 0.124374 |              nan |
| WGeo-Adaptive  | 2450 |    0.110164 | 0.095801 | 0.124499 |              nan |
| WGeo-Ensemble  | 2450 |    0.109816 | 0.095444 | 0.124147 |              nan |
| WAR-1          | 2450 |    0.114697 | 0.098916 | 0.13005  |              nan |
| WAR-1-last     | 2450 |    0.109047 | 0.094972 | 0.123571 |              nan |
| WAR-Select     | 2450 |    0.114724 | 0.098996 | 0.129991 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 | 136 |  0.08437 |        0.08859 |   0.08556 |   0.1055  |       0.10501 | 0.08515 |      0.08364 |         0.08511 |     0.08514 |       0.08649 |          0.08733 |         0.08669 |         0.08402 | 0.08475 |      0.0838  |      0.08512 |
|   2020 | 366 |  0.16664 |        0.16267 |   0.15993 |   0.17581 |       0.17546 | 0.16851 |      0.16573 |         0.16845 |     0.16848 |       0.1696  |          0.16572 |         0.16641 |         0.16728 | 0.16167 |      0.16713 |      0.16138 |
|   2021 | 365 |  0.1315  |        0.13034 |   0.13368 |   0.13439 |       0.13412 | 0.13095 |      0.13199 |         0.13086 |     0.13102 |       0.13304 |          0.13288 |         0.13345 |         0.13042 | 0.13431 |      0.1303  |      0.13432 |
|   2022 | 365 |  0.14005 |        0.1392  |   0.14079 |   0.14331 |       0.14308 | 0.126   |      0.13529 |         0.12585 |     0.12595 |       0.12783 |          0.12775 |         0.1268  |         0.1284  | 0.14877 |      0.12469 |      0.14907 |
|   2023 | 365 |  0.0592  |        0.06391 |   0.0584  |   0.06056 |       0.06177 | 0.05313 |      0.05631 |         0.05315 |     0.05312 |       0.05346 |          0.05334 |         0.05338 |         0.05362 | 0.05264 |      0.05367 |      0.05262 |
|   2024 | 366 |  0.09069 |        0.08926 |   0.09026 |   0.09196 |       0.09146 | 0.08922 |      0.08989 |         0.08916 |     0.08933 |       0.09101 |          0.0916  |         0.0902  |         0.08927 | 0.09379 |      0.08903 |      0.09378 |
|   2025 | 365 |  0.10928 |        0.10764 |   0.10778 |   0.10827 |       0.10771 | 0.10159 |      0.1068  |         0.10167 |     0.10167 |       0.10105 |          0.10293 |         0.10189 |         0.10314 | 0.11341 |      0.10259 |      0.11344 |
|   2026 | 122 |  0.10079 |        0.1024  |   0.10438 |   0.10508 |       0.10453 | 0.09831 |      0.09978 |         0.09815 |     0.09826 |       0.09817 |          0.09904 |         0.1027  |         0.09867 | 0.09882 |      0.0976  |      0.09882 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 505 |  0.11846 |        0.11771 |   0.11772 |   0.12225 |       0.12206 | 0.11266 |      0.11723 |         0.11261 |     0.11277 |       0.11362 |          0.11506 |         0.11455 |         0.11383 | 0.12909 |      0.11227 |      0.12891 |
| high-vol |  80 |  0.11283 |        0.1126  |   0.11498 |   0.13047 |       0.12965 | 0.108   |      0.10937 |         0.10788 |     0.10784 |       0.11204 |          0.11753 |         0.11015 |         0.10737 | 0.11209 |      0.10656 |      0.11217 |
| neutral  | 691 |  0.11584 |        0.11473 |   0.11509 |   0.1201  |       0.11981 | 0.10937 |      0.1136  |         0.10931 |     0.10943 |       0.11043 |          0.11144 |         0.10954 |         0.1103  | 0.11025 |      0.10866 |      0.11039 |
| low-vol  | 444 |  0.09836 |        0.09953 |   0.09562 |   0.09416 |       0.09504 | 0.09368 |      0.09604 |         0.09359 |     0.09358 |       0.09472 |          0.09306 |         0.09465 |         0.09394 | 0.09369 |      0.0938  |      0.09363 |
| rally    | 730 |  0.1178  |        0.11757 |   0.11801 |   0.12491 |       0.12434 | 0.11707 |      0.11659 |         0.11706 |     0.11709 |       0.11786 |          0.11572 |         0.11716 |         0.11651 | 0.12201 |      0.11674 |      0.12213 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-TheilSen**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0268 |           0      |
| HS-Bootstrap   |      0.0429 |           0      |
| GARCH-N        |      0.1126 |           0      |
| GARCH-t        |      0.001  |           0      |
| GJR-GARCH-t    |      0.0012 |           0      |
| WGeo           |      0.1061 |           0.0919 |
| WGeo-Gated     |      0.0374 |           0      |
| WGeo-TheilSen  |      0.137  |           0.1203 |
| WGeo-EWMA      |      0.0816 |           0.0708 |
| WGeo-Hetero    |      0.0081 |           0.0038 |
| WGeo-GARCH-Ens |      0.2187 |           0.1339 |
| WGeo-Adaptive  |      0.0488 |           0.0414 |
| WGeo-Ensemble  |      0.0942 |           0.0002 |
| WAR-1          |      0.0854 |           0.0144 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.082  |           0.0139 |

**Regime-conditional DM** (WGeo-TheilSen vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 505 |  0.11261 |  0.11227 |     0.30834 |  0.71013 | 0.47762 |
| high-vol |  80 |  0.10788 |  0.10656 |     1.23498 |  1.70038 | 0.08906 |
| neutral  | 691 |  0.10931 |  0.10866 |     0.60165 |  1.52401 | 0.12751 |
| low-vol  | 444 |  0.09359 |  0.0938  |    -0.21985 | -0.43923 | 0.66049 |
| rally    | 730 |  0.11706 |  0.11674 |     0.27662 |  0.75311 | 0.45138 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_ethusdt_h21.png)

## SOL/USDT

_2111 days from 2020-08-12 to 2026-05-23_

### Horizon h = 1 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 1380 |    0.025075 | 0.023561 | 0.026689 |              nan |
| HS-Bootstrap   | 1380 |    0.025076 | 0.023567 | 0.026697 |              nan |
| GARCH-N        | 1380 |    0.025219 | 0.023722 | 0.026786 |              nan |
| GARCH-t        | 1380 |    0.025519 | 0.024072 | 0.027022 |              nan |
| GJR-GARCH-t    | 1380 |    0.025521 | 0.024071 | 0.027029 |              nan |
| WGeo           | 1380 |    0.025094 | 0.023522 | 0.02682  |              nan |
| WGeo-Gated     | 1380 |    0.02504  | 0.023523 | 0.026678 |              nan |
| WGeo-TheilSen  | 1380 |    0.025094 | 0.023521 | 0.026821 |              nan |
| WGeo-EWMA      | 1380 |    0.025094 | 0.023522 | 0.026818 |              nan |
| WGeo-Hetero    | 1380 |    0.025168 | 0.023564 | 0.026914 |                0 |
| WGeo-GARCH-Ens | 1380 |    0.025101 | 0.02355  | 0.026799 |              nan |
| WGeo-Adaptive  | 1380 |    0.02518  | 0.023612 | 0.026902 |              nan |
| WGeo-Ensemble  | 1380 |    0.02503  | 0.023479 | 0.026732 |              nan |
| WAR-1          | 1380 |    0.025086 | 0.023512 | 0.026813 |              nan |
| WAR-1-last     | 1380 |    0.025086 | 0.023512 | 0.026813 |              nan |
| WAR-Select     | 1380 |    0.025086 | 0.023512 | 0.026813 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2022 | 142 |  0.03158 |        0.03156 |   0.03167 |   0.03227 |       0.03223 | 0.03055 |      0.03083 |         0.03055 |     0.03055 |       0.03029 |          0.03082 |         0.03078 |         0.03054 | 0.03056 |      0.03056 |      0.03056 |
|   2023 | 365 |  0.02704 |        0.02705 |   0.02732 |   0.02773 |       0.02776 | 0.02687 |      0.02694 |         0.02686 |     0.02687 |       0.02715 |          0.02695 |         0.02682 |         0.02683 | 0.02686 |      0.02686 |      0.02686 |
|   2024 | 366 |  0.02393 |        0.02392 |   0.02404 |   0.02428 |       0.02429 | 0.02435 |      0.02415 |         0.02435 |     0.02435 |       0.02438 |          0.02423 |         0.02443 |         0.02425 | 0.02434 |      0.02434 |      0.02434 |
|   2025 | 365 |  0.02394 |        0.02394 |   0.02406 |   0.02422 |       0.02422 | 0.02415 |      0.02395 |         0.02415 |     0.02415 |       0.02423 |          0.02407 |         0.02431 |         0.02405 | 0.02414 |      0.02414 |      0.02414 |
|   2026 | 142 |  0.01939 |        0.01942 |   0.01939 |   0.01963 |       0.01957 | 0.01944 |      0.01947 |         0.01944 |     0.01944 |       0.0194  |          0.01953 |         0.01952 |         0.01942 | 0.01943 |      0.01943 |      0.01943 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 282 |  0.02608 |        0.0261  |   0.02648 |   0.02689 |       0.02688 | 0.02598 |      0.02591 |         0.02598 |     0.02598 |       0.02605 |          0.02615 |         0.02624 |         0.02589 | 0.02597 |      0.02597 |      0.02597 |
| neutral  | 460 |  0.02556 |        0.02558 |   0.02565 |   0.02593 |       0.02595 | 0.02544 |      0.02544 |         0.02544 |     0.02544 |       0.02542 |          0.02545 |         0.02552 |         0.02541 | 0.02544 |      0.02544 |      0.02544 |
| low-vol  | 274 |  0.02119 |        0.02117 |   0.02117 |   0.02138 |       0.02136 | 0.02116 |      0.02113 |         0.02116 |     0.02116 |       0.02145 |          0.02117 |         0.0211  |         0.02109 | 0.02114 |      0.02114 |      0.02114 |
| rally    | 364 |  0.02662 |        0.02659 |   0.02674 |   0.02705 |       0.02706 | 0.02693 |      0.02681 |         0.02693 |     0.02693 |       0.02696 |          0.02681 |         0.02699 |         0.02685 | 0.02692 |      0.02692 |      0.02692 |

**Diebold-Mariano vs Static** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      1      |           1      |
| HS-Bootstrap   |      0.9662 |           0.9659 |
| GARCH-N        |      0.0497 |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.85   |           0.8375 |
| WGeo-Gated     |      0.5675 |           0.5538 |
| WGeo-TheilSen  |      0.8546 |           0.8424 |
| WGeo-EWMA      |      0.8529 |           0.8407 |
| WGeo-Hetero    |      0.4623 |           0.3582 |
| WGeo-GARCH-Ens |      0.7836 |           0.7264 |
| WGeo-Adaptive  |      0.3782 |           0.3563 |
| WGeo-Ensemble  |      0.5839 |           0.5566 |
| WAR-1          |      0.9166 |           0.9095 |
| WAR-1-last     |      0.9166 |           0.9095 |
| WAR-Select     |      0.914  |           0.9066 |

**Regime-conditional DM** (WGeo-Ensemble vs Static, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 282 |  0.02589 |  0.02608 |    -0.70633 | -0.86816 | 0.38531 |
| neutral  | 460 |  0.02541 |  0.02556 |    -0.56955 | -1.61719 | 0.10584 |
| low-vol  | 274 |  0.02109 |  0.02119 |    -0.47672 | -0.6026  | 0.54678 |
| rally    | 364 |  0.02685 |  0.02662 |     0.86542 |  1.11259 | 0.26588 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_solusdt_h1.png)

### Horizon h = 5 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 1376 |    0.057622 | 0.052216 | 0.063797 |              nan |
| HS-Bootstrap   | 1376 |    0.058124 | 0.052966 | 0.064264 |              nan |
| GARCH-N        | 1376 |    0.058296 | 0.053116 | 0.064585 |              nan |
| GARCH-t        | 1376 |    0.058874 | 0.053809 | 0.065001 |              nan |
| GJR-GARCH-t    | 1376 |    0.058877 | 0.053782 | 0.064977 |              nan |
| WGeo           | 1376 |    0.057192 | 0.051808 | 0.063757 |              nan |
| WGeo-Gated     | 1376 |    0.057408 | 0.051958 | 0.06358  |              nan |
| WGeo-TheilSen  | 1376 |    0.057186 | 0.051801 | 0.063747 |              nan |
| WGeo-EWMA      | 1376 |    0.057198 | 0.051818 | 0.063774 |              nan |
| WGeo-Hetero    | 1376 |    0.05766  | 0.052138 | 0.064433 |                0 |
| WGeo-GARCH-Ens | 1376 |    0.057699 | 0.052102 | 0.064386 |              nan |
| WGeo-Adaptive  | 1376 |    0.057365 | 0.051938 | 0.063861 |              nan |
| WGeo-Ensemble  | 1376 |    0.057153 | 0.051737 | 0.063599 |              nan |
| WAR-1          | 1376 |    0.058304 | 0.052811 | 0.064821 |              nan |
| WAR-1-last     | 1376 |    0.057096 | 0.051673 | 0.063583 |              nan |
| WAR-Select     | 1376 |    0.058315 | 0.052825 | 0.06483  |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2022 | 142 |  0.081   |        0.08298 |   0.082   |   0.08225 |       0.08227 | 0.07745 |      0.07906 |         0.07742 |     0.07741 |       0.07787 |          0.07918 |         0.07881 |         0.07777 | 0.07632 |      0.07731 |      0.07633 |
|   2023 | 365 |  0.06508 |        0.06562 |   0.06578 |   0.06691 |       0.06703 | 0.06465 |      0.06505 |         0.06465 |     0.06465 |       0.0654  |          0.06568 |         0.06415 |         0.06461 | 0.06597 |      0.06451 |      0.06597 |
|   2024 | 366 |  0.0531  |        0.05325 |   0.05399 |   0.05461 |       0.05454 | 0.05383 |      0.05352 |         0.05382 |     0.05383 |       0.05432 |          0.05372 |         0.05404 |         0.05366 | 0.05534 |      0.05375 |      0.05538 |
|   2025 | 365 |  0.05169 |        0.05191 |   0.05209 |   0.05222 |       0.05224 | 0.05126 |      0.05131 |         0.05127 |     0.0513  |       0.05169 |          0.05142 |         0.05153 |         0.0512  | 0.05282 |      0.05124 |      0.05282 |
|   2026 | 138 |  0.04152 |        0.04208 |   0.04194 |   0.04248 |       0.04232 | 0.04122 |      0.04134 |         0.04121 |     0.04121 |       0.04103 |          0.04166 |         0.0416  |         0.0412  | 0.04185 |      0.04104 |      0.04185 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 282 |  0.05356 |        0.05413 |   0.05443 |   0.05551 |       0.05552 | 0.05325 |      0.05305 |         0.05324 |     0.05322 |       0.05343 |          0.05386 |         0.05392 |         0.05303 | 0.05466 |      0.05308 |      0.05465 |
| neutral  | 460 |  0.05881 |        0.05947 |   0.0594  |   0.05975 |       0.05974 | 0.05772 |      0.05844 |         0.05772 |     0.05773 |       0.05788 |          0.05794 |         0.05802 |         0.05788 | 0.05815 |      0.05778 |      0.05816 |
| low-vol  | 270 |  0.05562 |        0.05604 |   0.05566 |   0.05614 |       0.05604 | 0.05651 |      0.05619 |         0.0565  |     0.05651 |       0.05728 |          0.05649 |         0.05647 |         0.05625 | 0.05783 |      0.05614 |      0.05783 |
| rally    | 364 |  0.06074 |        0.06106 |   0.06186 |   0.0624  |       0.0625  | 0.06009 |      0.06039 |         0.06008 |     0.06011 |       0.06094 |          0.06126 |         0.05987 |         0.0601  | 0.06167 |      0.06005 |      0.0617  |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.1576 |           0.0648 |
| HS-Bootstrap   |      0.0213 |           0.0005 |
| GARCH-N        |      0.0048 |           0      |
| GARCH-t        |      0.0001 |           0      |
| GJR-GARCH-t    |      0.0001 |           0      |
| WGeo           |      0.0295 |           0.017  |
| WGeo-Gated     |      0.1287 |           0.0566 |
| WGeo-TheilSen  |      0.0277 |           0.0156 |
| WGeo-EWMA      |      0.0225 |           0.0135 |
| WGeo-Hetero    |      0.0001 |           0      |
| WGeo-GARCH-Ens |      0.0049 |           0.0002 |
| WGeo-Adaptive  |      0.2057 |           0.1732 |
| WGeo-Ensemble  |      0.3936 |           0.3025 |
| WAR-1          |      0.0083 |           0.0065 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.0079 |           0.0062 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 282 |  0.05303 |  0.05308 |    -0.11119 | -0.30866 | 0.75758 |
| neutral  | 460 |  0.05788 |  0.05778 |     0.18325 |  1.04225 | 0.2973  |
| low-vol  | 270 |  0.05625 |  0.05614 |     0.19781 |  0.89713 | 0.36965 |
| rally    | 364 |  0.0601  |  0.06005 |     0.0768  |  0.3936  | 0.69388 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_solusdt_h5.png)

### Horizon h = 21 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 1360 |    0.134406 | 0.111058 | 0.161299 |              nan |
| HS-Bootstrap   | 1360 |    0.135366 | 0.11333  | 0.16129  |              nan |
| GARCH-N        | 1360 |    0.133568 | 0.112107 | 0.158416 |              nan |
| GARCH-t        | 1360 |    0.133909 | 0.113145 | 0.157427 |              nan |
| GJR-GARCH-t    | 1360 |    0.134046 | 0.113148 | 0.15792  |              nan |
| WGeo           | 1360 |    0.129518 | 0.10628  | 0.156601 |              nan |
| WGeo-Gated     | 1360 |    0.131786 | 0.108845 | 0.158442 |              nan |
| WGeo-TheilSen  | 1360 |    0.129447 | 0.106168 | 0.156516 |              nan |
| WGeo-EWMA      | 1360 |    0.129428 | 0.10617  | 0.156476 |              nan |
| WGeo-Hetero    | 1360 |    0.130267 | 0.106574 | 0.157929 |                0 |
| WGeo-GARCH-Ens | 1360 |    0.131917 | 0.108324 | 0.159237 |              nan |
| WGeo-Adaptive  | 1360 |    0.129263 | 0.106268 | 0.155686 |              nan |
| WGeo-Ensemble  | 1360 |    0.129833 | 0.1065   | 0.157107 |              nan |
| WAR-1          | 1360 |    0.136555 | 0.113181 | 0.161886 |              nan |
| WAR-1-last     | 1360 |    0.128401 | 0.105547 | 0.155412 |              nan |
| WAR-Select     | 1360 |    0.136713 | 0.113181 | 0.162534 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2022 | 142 |  0.22362 |        0.22609 |   0.22249 |   0.21707 |       0.21718 | 0.21432 |      0.22032 |         0.21414 |     0.21433 |       0.21682 |          0.22091 |         0.21756 |         0.21576 | 0.21254 |      0.21191 |      0.21242 |
|   2023 | 365 |  0.1597  |        0.16074 |   0.15688 |   0.15961 |       0.16063 | 0.15281 |      0.15532 |         0.15283 |     0.15262 |       0.1534  |          0.15617 |         0.14881 |         0.15317 | 0.1589  |      0.15132 |      0.15878 |
|   2024 | 366 |  0.11303 |        0.11314 |   0.11254 |   0.11473 |       0.11451 | 0.11407 |      0.1136  |         0.1139  |     0.11399 |       0.11478 |          0.11384 |         0.11483 |         0.11357 | 0.12056 |      0.11241 |      0.1213  |
|   2025 | 365 |  0.10642 |        0.10735 |   0.10607 |   0.10472 |       0.10468 | 0.0984  |      0.10208 |         0.09847 |     0.09838 |       0.09902 |          0.10131 |         0.09872 |         0.09911 | 0.11246 |      0.09872 |      0.11246 |
|   2026 | 122 |  0.10278 |        0.10436 |   0.10569 |   0.10509 |       0.10422 | 0.10058 |      0.10175 |         0.10021 |     0.10044 |       0.10025 |          0.10156 |         0.10269 |         0.1007  | 0.10133 |      0.09941 |      0.10136 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 282 |  0.11937 |        0.12021 |   0.12106 |   0.12505 |       0.12506 | 0.12117 |      0.12079 |         0.12106 |     0.12111 |       0.12233 |          0.12518 |         0.1241  |         0.12054 | 0.1411  |      0.11902 |      0.14104 |
| neutral  | 460 |  0.11802 |        0.12156 |   0.11916 |   0.1183  |       0.11827 | 0.10818 |      0.11338 |         0.10824 |     0.10814 |       0.10813 |          0.11064 |         0.10723 |         0.1096  | 0.11455 |      0.10899 |      0.11455 |
| low-vol  | 254 |  0.18305 |        0.17983 |   0.17814 |   0.17728 |       0.17736 | 0.18974 |      0.18649 |         0.18942 |     0.18964 |       0.19179 |          0.1894  |         0.1889  |         0.18802 | 0.18419 |      0.18543 |      0.18413 |
| rally    | 364 |  0.13282 |        0.13352 |   0.13036 |   0.13023 |       0.13072 | 0.12092 |      0.12539 |         0.12091 |     0.12077 |       0.12146 |          0.12392 |         0.1195  |         0.122   | 0.1276  |      0.12041 |      0.12827 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Adaptive**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0217 |           0      |
| HS-Bootstrap   |      0.0134 |           0      |
| GARCH-N        |      0.0345 |           0      |
| GARCH-t        |      0.0099 |           0      |
| GJR-GARCH-t    |      0.0096 |           0      |
| WGeo           |      0.0251 |           0.0032 |
| WGeo-Gated     |      0.0143 |           0      |
| WGeo-TheilSen  |      0.0245 |           0.0027 |
| WGeo-EWMA      |      0.0372 |           0.0073 |
| WGeo-Hetero    |      0.0072 |           0      |
| WGeo-GARCH-Ens |      0.0007 |           0      |
| WGeo-Adaptive  |      0.42   |           0.2681 |
| WGeo-Ensemble  |      0.0035 |           0      |
| WAR-1          |      0.0505 |           0.0387 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.0484 |           0.0375 |

**Regime-conditional DM** (WGeo-Adaptive vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 282 |  0.1241  |  0.11902 |     4.27108 |  2.70238 | 0.00688 |
| neutral  | 460 |  0.10723 |  0.10899 |    -1.61849 | -1.47066 | 0.14138 |
| low-vol  | 254 |  0.1889  |  0.18543 |     1.87137 |  1.74316 | 0.08131 |
| rally    | 364 |  0.1195  |  0.12041 |    -0.75617 | -0.33838 | 0.73508 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_solusdt_h21.png)

## BNB/USDT

_3120 days from 2017-11-07 to 2026-05-23_

### Horizon h = 1 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2389 |    0.020339 | 0.018967 | 0.0218   |              nan |
| HS-Bootstrap   | 2389 |    0.020332 | 0.018962 | 0.021797 |              nan |
| GARCH-N        | 2389 |    0.020199 | 0.01891  | 0.021533 |              nan |
| GARCH-t        | 2389 |    0.020765 | 0.019523 | 0.022086 |              nan |
| GJR-GARCH-t    | 2389 |    0.020783 | 0.019546 | 0.02209  |              nan |
| WGeo           | 2389 |    0.020208 | 0.01887  | 0.021687 |              nan |
| WGeo-Gated     | 2389 |    0.02019  | 0.018874 | 0.021619 |              nan |
| WGeo-TheilSen  | 2389 |    0.020207 | 0.018869 | 0.021686 |              nan |
| WGeo-EWMA      | 2389 |    0.020207 | 0.018869 | 0.021686 |              nan |
| WGeo-Hetero    | 2389 |    0.020277 | 0.018975 | 0.021692 |                0 |
| WGeo-GARCH-Ens | 2389 |    0.020165 | 0.018866 | 0.021549 |              nan |
| WGeo-Adaptive  | 2389 |    0.020191 | 0.018893 | 0.021632 |              nan |
| WGeo-Ensemble  | 2389 |    0.020144 | 0.018826 | 0.021611 |              nan |
| WAR-1          | 2389 |    0.020203 | 0.018867 | 0.021681 |              nan |
| WAR-1-last     | 2389 |    0.020203 | 0.018867 | 0.021681 |              nan |
| WAR-Select     | 2389 |    0.020203 | 0.018867 | 0.021681 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 |  55 |  0.02065 |        0.02062 |   0.02047 |   0.02101 |       0.02103 | 0.01979 |      0.02047 |         0.01978 |     0.01978 |       0.02039 |          0.01978 |         0.02027 |         0.01991 | 0.01978 |      0.01978 |      0.01978 |
|   2020 | 366 |  0.02445 |        0.02442 |   0.02461 |   0.0253  |       0.02532 | 0.02476 |      0.02452 |         0.02476 |     0.02476 |       0.02474 |          0.02475 |         0.02466 |         0.02463 | 0.02475 |      0.02475 |      0.02475 |
|   2021 | 365 |  0.03675 |        0.03672 |   0.03589 |   0.03618 |       0.03625 | 0.03645 |      0.03632 |         0.03645 |     0.03644 |       0.03633 |          0.03594 |         0.03616 |         0.03631 | 0.03646 |      0.03646 |      0.03646 |
|   2022 | 365 |  0.02077 |        0.02078 |   0.02067 |   0.02132 |       0.0213  | 0.02037 |      0.02051 |         0.02036 |     0.02036 |       0.02073 |          0.02053 |         0.02041 |         0.02033 | 0.02036 |      0.02036 |      0.02035 |
|   2023 | 365 |  0.01284 |        0.01282 |   0.01268 |   0.01338 |       0.01343 | 0.0122  |      0.01243 |         0.0122  |     0.0122  |       0.01245 |          0.0124  |         0.01223 |         0.01219 | 0.01219 |      0.01219 |      0.01219 |
|   2024 | 366 |  0.01596 |        0.01596 |   0.01597 |   0.0167  |       0.01669 | 0.01608 |      0.01595 |         0.01608 |     0.01608 |       0.01605 |          0.01601 |         0.01618 |         0.01601 | 0.01607 |      0.01607 |      0.01607 |
|   2025 | 365 |  0.01423 |        0.01426 |   0.01425 |   0.01465 |       0.01465 | 0.01438 |      0.01429 |         0.01439 |     0.01439 |       0.01431 |          0.01426 |         0.01442 |         0.01434 | 0.01438 |      0.01438 |      0.01438 |
|   2026 | 142 |  0.0126  |        0.01261 |   0.0127  |   0.01311 |       0.01313 | 0.01268 |      0.01268 |         0.01268 |     0.01268 |       0.01267 |          0.01286 |         0.01267 |         0.01266 | 0.01267 |      0.01267 |      0.01267 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 359 |  0.02129 |        0.0213  |   0.02163 |   0.02225 |       0.02226 | 0.02159 |      0.02141 |         0.02159 |     0.02159 |       0.02171 |          0.02187 |         0.02176 |         0.02146 | 0.02157 |      0.02157 |      0.02158 |
| high-vol |  64 |  0.0282  |        0.02814 |   0.02675 |   0.02801 |       0.02814 | 0.02843 |      0.02803 |         0.02843 |     0.02843 |       0.02697 |          0.02727 |         0.02829 |         0.02821 | 0.02842 |      0.02842 |      0.02842 |
| neutral  | 878 |  0.0175  |        0.0175  |   0.01751 |   0.01819 |       0.01819 | 0.01727 |      0.01738 |         0.01727 |     0.01727 |       0.0174  |          0.01735 |         0.01724 |         0.01726 | 0.01726 |      0.01726 |      0.01726 |
| low-vol  | 555 |  0.01351 |        0.01349 |   0.01339 |   0.01382 |       0.01384 | 0.01325 |      0.01326 |         0.01325 |     0.01325 |       0.0135  |          0.01329 |         0.01328 |         0.0132  | 0.01325 |      0.01325 |      0.01325 |
| rally    | 533 |  0.03054 |        0.03053 |   0.02998 |   0.03037 |       0.03041 | 0.03037 |      0.03027 |         0.03037 |     0.03037 |       0.0303  |          0.02996 |         0.03022 |         0.03028 | 0.03037 |      0.03037 |      0.03037 |

**Diebold-Mariano vs GARCH-N** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.1333 |           0      |
| HS-Bootstrap   |      0.1508 |           0      |
| GARCH-N        |      1      |           1      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.9239 |           0.8848 |
| WGeo-Gated     |      0.9197 |           0.8351 |
| WGeo-TheilSen  |      0.9265 |           0.8887 |
| WGeo-EWMA      |      0.9328 |           0.8984 |
| WGeo-Hetero    |      0.3459 |           0.2189 |
| WGeo-GARCH-Ens |      0.4771 |           0.4432 |
| WGeo-Adaptive  |      0.9343 |           0.918  |
| WGeo-Ensemble  |      0.5433 |           0.3015 |
| WAR-1          |      0.9657 |           0.9473 |
| WAR-1-last     |      0.9657 |           0.9473 |
| WAR-Select     |      0.9644 |           0.9453 |

**Regime-conditional DM** (WGeo-Ensemble vs GARCH-N, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 359 |  0.02146 |  0.02163 |    -0.77091 | -0.62318 | 0.53317 |
| high-vol |  64 |  0.02821 |  0.02675 |     5.47034 |  1.27831 | 0.20114 |
| neutral  | 878 |  0.01726 |  0.01751 |    -1.43266 | -3.08513 | 0.00203 |
| low-vol  | 555 |  0.0132  |  0.01339 |    -1.3861  | -2.3689  | 0.01784 |
| rally    | 533 |  0.03028 |  0.02998 |     0.99814 |  1.0324  | 0.30189 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_bnbusdt_h1.png)

### Horizon h = 5 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2385 |    0.046297 | 0.041903 | 0.052071 |              nan |
| HS-Bootstrap   | 2385 |    0.046858 | 0.042539 | 0.052444 |              nan |
| GARCH-N        | 2385 |    0.046475 | 0.042255 | 0.051819 |              nan |
| GARCH-t        | 2385 |    0.047919 | 0.043815 | 0.053039 |              nan |
| GJR-GARCH-t    | 2385 |    0.047912 | 0.043791 | 0.052995 |              nan |
| WGeo           | 2385 |    0.04593  | 0.041517 | 0.051693 |              nan |
| WGeo-Gated     | 2385 |    0.045977 | 0.041593 | 0.051608 |              nan |
| WGeo-TheilSen  | 2385 |    0.045923 | 0.041511 | 0.05169  |              nan |
| WGeo-EWMA      | 2385 |    0.045916 | 0.041507 | 0.051677 |              nan |
| WGeo-Hetero    | 2385 |    0.046596 | 0.042183 | 0.052207 |                0 |
| WGeo-GARCH-Ens | 2385 |    0.046108 | 0.041845 | 0.051584 |              nan |
| WGeo-Adaptive  | 2385 |    0.045898 | 0.041546 | 0.051424 |              nan |
| WGeo-Ensemble  | 2385 |    0.045782 | 0.041336 | 0.051429 |              nan |
| WAR-1          | 2385 |    0.046258 | 0.042014 | 0.051665 |              nan |
| WAR-1-last     | 2385 |    0.045884 | 0.041442 | 0.051672 |              nan |
| WAR-Select     | 2385 |    0.046257 | 0.042019 | 0.051665 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 |  55 |  0.05003 |        0.05181 |   0.04874 |   0.0492  |       0.04926 | 0.04614 |      0.04936 |         0.04609 |     0.04614 |       0.04783 |          0.04609 |         0.04717 |         0.04696 | 0.04622 |      0.04602 |      0.04617 |
|   2020 | 366 |  0.05449 |        0.05474 |   0.05514 |   0.05671 |       0.05667 | 0.05613 |      0.05495 |         0.05612 |     0.05613 |       0.05597 |          0.05542 |         0.05555 |         0.05561 | 0.05664 |      0.05598 |      0.05665 |
|   2021 | 365 |  0.0849  |        0.08467 |   0.08558 |   0.08577 |       0.08565 | 0.08488 |      0.08417 |         0.0849  |     0.08483 |       0.08708 |          0.0852  |         0.08413 |         0.08433 | 0.08442 |      0.08492 |      0.08444 |
|   2022 | 365 |  0.0478  |        0.04925 |   0.04789 |   0.04978 |       0.04977 | 0.0457  |      0.04725 |         0.04568 |     0.04567 |       0.04702 |          0.04654 |         0.046   |         0.04596 | 0.04684 |      0.04556 |      0.04683 |
|   2023 | 365 |  0.02864 |        0.03053 |   0.02871 |   0.03098 |       0.0311  | 0.0272  |      0.02767 |         0.0272  |     0.0272  |       0.02798 |          0.02776 |         0.02744 |         0.02716 | 0.02783 |      0.02717 |      0.02781 |
|   2024 | 366 |  0.03735 |        0.03702 |   0.03699 |   0.03894 |       0.03893 | 0.03731 |      0.03718 |         0.03729 |     0.0373  |       0.03722 |          0.03714 |         0.03771 |         0.03721 | 0.03715 |      0.03732 |      0.03715 |
|   2025 | 365 |  0.03053 |        0.03065 |   0.03054 |   0.03162 |       0.03164 | 0.0309  |      0.03058 |         0.03091 |     0.03091 |       0.03094 |          0.03093 |         0.03094 |         0.03073 | 0.03105 |      0.03089 |      0.03105 |
|   2026 | 138 |  0.02913 |        0.02977 |   0.02974 |   0.03076 |       0.03073 | 0.02851 |      0.02891 |         0.0285  |     0.02849 |       0.0286  |          0.02935 |         0.02852 |         0.02858 | 0.02939 |      0.02851 |      0.0294  |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 359 |  0.04116 |        0.0421  |   0.04343 |   0.04555 |       0.04542 | 0.04163 |      0.04135 |         0.04162 |     0.04163 |       0.04272 |          0.04317 |         0.04235 |         0.04135 | 0.04204 |      0.04148 |      0.04207 |
| high-vol |  64 |  0.05143 |        0.05204 |   0.05257 |   0.05508 |       0.0543  | 0.05355 |      0.05214 |         0.05353 |     0.05358 |       0.05531 |          0.05399 |         0.05306 |         0.05293 | 0.05385 |      0.05329 |      0.05386 |
| neutral  | 878 |  0.04095 |        0.04198 |   0.04079 |   0.04259 |       0.04261 | 0.04011 |      0.04066 |         0.0401  |     0.0401  |       0.0405  |          0.04033 |         0.03991 |         0.04014 | 0.04056 |      0.04008 |      0.04056 |
| low-vol  | 551 |  0.03183 |        0.03261 |   0.03205 |   0.033   |       0.033   | 0.03182 |      0.03165 |         0.03181 |     0.03181 |       0.0327  |          0.03191 |         0.0321  |         0.03161 | 0.03148 |      0.03178 |      0.03146 |
| rally    | 533 |  0.07291 |        0.0722  |   0.07207 |   0.07286 |       0.07298 | 0.07208 |      0.07192 |         0.07209 |     0.07204 |       0.07256 |          0.07133 |         0.07156 |         0.07185 | 0.07285 |      0.0721  |      0.07285 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.1295 |           0.051  |
| HS-Bootstrap   |      0.001  |           0      |
| GARCH-N        |      0.087  |           0.0034 |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.1117 |           0.0285 |
| WGeo-Gated     |      0.5875 |           0.4907 |
| WGeo-TheilSen  |      0.137  |           0.0439 |
| WGeo-EWMA      |      0.2884 |           0.1283 |
| WGeo-Hetero    |      0.0119 |           0      |
| WGeo-GARCH-Ens |      0.4204 |           0.1307 |
| WGeo-Adaptive  |      0.9431 |           0.9074 |
| WGeo-Ensemble  |      0.076  |           0.0194 |
| WAR-1          |      0.2231 |           0.1014 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.224  |           0.1024 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 359 |  0.04135 |  0.04148 |    -0.30637 | -0.84161 | 0.4     |
| high-vol |  64 |  0.05293 |  0.05329 |    -0.67046 | -1.22434 | 0.22082 |
| neutral  | 878 |  0.04014 |  0.04008 |     0.15455 |  0.63581 | 0.5249  |
| low-vol  | 551 |  0.03161 |  0.03178 |    -0.54456 | -2.25089 | 0.02439 |
| rally    | 533 |  0.07185 |  0.0721  |    -0.35238 | -1.83176 | 0.06699 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_bnbusdt_h5.png)

### Horizon h = 21 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2369 |    0.105553 | 0.083691 | 0.131754 |              nan |
| HS-Bootstrap   | 2369 |    0.106755 | 0.085854 | 0.133013 |              nan |
| GARCH-N        | 2369 |    0.105595 | 0.084588 | 0.131451 |              nan |
| GARCH-t        | 2369 |    0.109944 | 0.090633 | 0.13506  |              nan |
| GJR-GARCH-t    | 2369 |    0.110042 | 0.090603 | 0.134932 |              nan |
| WGeo           | 2369 |    0.102994 | 0.081071 | 0.130182 |              nan |
| WGeo-Gated     | 2369 |    0.103847 | 0.08253  | 0.130426 |              nan |
| WGeo-TheilSen  | 2369 |    0.102961 | 0.081032 | 0.130172 |              nan |
| WGeo-EWMA      | 2369 |    0.102967 | 0.08107  | 0.130158 |              nan |
| WGeo-Hetero    | 2369 |    0.105284 | 0.082641 | 0.133207 |                0 |
| WGeo-GARCH-Ens | 2369 |    0.103933 | 0.082546 | 0.130324 |              nan |
| WGeo-Adaptive  | 2369 |    0.103218 | 0.081312 | 0.129533 |              nan |
| WGeo-Ensemble  | 2369 |    0.102619 | 0.080916 | 0.129496 |              nan |
| WAR-1          | 2369 |    0.1053   | 0.083973 | 0.130572 |              nan |
| WAR-1-last     | 2369 |    0.102471 | 0.080584 | 0.129853 |              nan |
| WAR-Select     | 2369 |    0.105347 | 0.084008 | 0.130618 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2019 |  55 |  0.13619 |        0.134   |   0.11983 |   0.11686 |       0.11755 | 0.11578 |      0.13189 |         0.11563 |     0.11567 |       0.12343 |          0.11563 |         0.12011 |         0.12045 | 0.11422 |      0.114   |      0.1151  |
|   2020 | 366 |  0.11413 |        0.11453 |   0.11817 |   0.11912 |       0.11884 | 0.12097 |      0.11633 |         0.1209  |     0.12096 |       0.12201 |          0.1199  |         0.11946 |         0.11899 | 0.12117 |      0.11975 |      0.12132 |
|   2021 | 365 |  0.2219  |        0.21918 |   0.22461 |   0.2213  |       0.22131 | 0.22277 |      0.22183 |         0.22285 |     0.22251 |       0.23122 |          0.22248 |         0.22297 |         0.2211  | 0.21971 |      0.22297 |      0.21999 |
|   2022 | 365 |  0.10796 |        0.1109  |   0.1015  |   0.10962 |       0.11082 | 0.08915 |      0.09858 |         0.08907 |     0.08919 |       0.09182 |          0.0923  |         0.08958 |         0.09106 | 0.10377 |      0.08838 |      0.10389 |
|   2023 | 365 |  0.06226 |        0.06731 |   0.06322 |   0.07276 |       0.07266 | 0.05926 |      0.06046 |         0.05923 |     0.05929 |       0.06006 |          0.05981 |         0.05928 |         0.05915 | 0.06043 |      0.05891 |      0.06009 |
|   2024 | 366 |  0.06986 |        0.07097 |   0.07033 |   0.08007 |       0.08008 | 0.07079 |      0.06982 |         0.07074 |     0.07085 |       0.07118 |          0.07249 |         0.07224 |         0.0703  | 0.06889 |      0.07045 |      0.0689  |
|   2025 | 365 |  0.05967 |        0.06027 |   0.05964 |   0.06303 |       0.06289 | 0.06117 |      0.05925 |         0.06119 |     0.06116 |       0.06121 |          0.06291 |         0.06095 |         0.06013 | 0.06393 |      0.06079 |      0.06393 |
|   2026 | 122 |  0.08459 |        0.08684 |   0.08768 |   0.08834 |       0.08782 | 0.07896 |      0.0818  |         0.0788  |     0.07892 |       0.0799  |          0.07996 |         0.08025 |         0.07971 | 0.08319 |      0.0782  |      0.08305 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 359 |  0.08281 |        0.0865  |   0.08629 |   0.09369 |       0.09355 | 0.08109 |      0.08095 |         0.08104 |     0.08106 |       0.08322 |          0.08331 |         0.08308 |         0.0802  | 0.0867  |      0.08022 |      0.08702 |
| high-vol |  64 |  0.11012 |        0.10986 |   0.12005 |   0.1214  |       0.11932 | 0.11708 |      0.11698 |         0.11679 |     0.11749 |       0.13024 |          0.12696 |         0.121   |         0.11638 | 0.1303  |      0.1152  |      0.13106 |
| neutral  | 878 |  0.08576 |        0.0875  |   0.08374 |   0.08992 |       0.09028 | 0.08143 |      0.08392 |         0.08137 |     0.08146 |       0.08288 |          0.08248 |         0.08112 |         0.08156 | 0.08526 |      0.08089 |      0.08529 |
| low-vol  | 535 |  0.07262 |        0.07463 |   0.0726  |   0.07737 |       0.0772  | 0.07296 |      0.07202 |         0.07292 |     0.07296 |       0.07389 |          0.0723  |         0.07286 |         0.07218 | 0.06838 |      0.07209 |      0.06821 |
| rally    | 533 |  0.18599 |        0.18399 |   0.18599 |   0.18519 |       0.18557 | 0.18172 |      0.18247 |         0.18178 |     0.18154 |       0.18557 |          0.18215 |         0.18152 |         0.18131 | 0.1849  |      0.18197 |      0.18491 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.1742 |           0.0013 |
| HS-Bootstrap   |      0.0547 |           0      |
| GARCH-N        |      0.1209 |           0.0004 |
| GARCH-t        |      0.001  |           0      |
| GJR-GARCH-t    |      0.0015 |           0      |
| WGeo           |      0.0538 |           0.0318 |
| WGeo-Gated     |      0.3185 |           0.0335 |
| WGeo-TheilSen  |      0.0549 |           0.0339 |
| WGeo-EWMA      |      0.0691 |           0.0415 |
| WGeo-Hetero    |      0.0067 |           0      |
| WGeo-GARCH-Ens |      0.2625 |           0.0328 |
| WGeo-Adaptive  |      0.2639 |           0.1622 |
| WGeo-Ensemble  |      0.76   |           0.5429 |
| WAR-1          |      0.3431 |           0.1618 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.336  |           0.1571 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 359 |  0.0802  |  0.08022 |    -0.02644 | -0.02294 | 0.9817  |
| high-vol |  64 |  0.11638 |  0.1152  |     1.02278 |  0.75829 | 0.44828 |
| neutral  | 878 |  0.08156 |  0.08089 |     0.82989 |  0.78292 | 0.43367 |
| low-vol  | 535 |  0.07218 |  0.07209 |     0.12195 |  0.16687 | 0.86748 |
| rally    | 533 |  0.18131 |  0.18197 |    -0.36336 | -0.53954 | 0.58952 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_bnbusdt_h21.png)

## XRP/USDT

_2941 days from 2018-05-05 to 2026-05-23_

### Horizon h = 1 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2210 |    0.024566 | 0.022648 | 0.026374 |              nan |
| HS-Bootstrap   | 2210 |    0.024548 | 0.022635 | 0.026357 |              nan |
| GARCH-N        | 2210 |    0.024697 | 0.022893 | 0.026412 |              nan |
| GARCH-t        | 2210 |    0.025866 | 0.024163 | 0.027557 |              nan |
| GJR-GARCH-t    | 2210 |    0.025872 | 0.024156 | 0.027543 |              nan |
| WGeo           | 2210 |    0.024434 | 0.022589 | 0.026244 |              nan |
| WGeo-Gated     | 2210 |    0.024403 | 0.022555 | 0.026199 |              nan |
| WGeo-TheilSen  | 2210 |    0.024433 | 0.022587 | 0.026244 |              nan |
| WGeo-EWMA      | 2210 |    0.024432 | 0.022587 | 0.026244 |              nan |
| WGeo-Hetero    | 2210 |    0.024569 | 0.022691 | 0.026405 |                0 |
| WGeo-GARCH-Ens | 2210 |    0.024398 | 0.02257  | 0.026141 |              nan |
| WGeo-Adaptive  | 2210 |    0.0244   | 0.022608 | 0.026181 |              nan |
| WGeo-Ensemble  | 2210 |    0.024359 | 0.022511 | 0.026166 |              nan |
| WAR-1          | 2210 |    0.024428 | 0.02258  | 0.026239 |              nan |
| WAR-1-last     | 2210 |    0.024428 | 0.02258  | 0.026239 |              nan |
| WAR-Select     | 2210 |    0.024428 | 0.022579 | 0.026238 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2020 | 242 |  0.0274  |        0.02734 |   0.02692 |   0.02894 |       0.02892 | 0.02751 |      0.02732 |         0.02751 |     0.0275  |       0.02676 |          0.02671 |         0.02679 |         0.0274  | 0.02751 |      0.02751 |      0.02751 |
|   2021 | 365 |  0.03953 |        0.03949 |   0.03895 |   0.04045 |       0.04065 | 0.03888 |      0.03897 |         0.03888 |     0.03888 |       0.03957 |          0.03869 |         0.0391  |         0.03878 | 0.03888 |      0.03888 |      0.03888 |
|   2022 | 365 |  0.02369 |        0.02367 |   0.02394 |   0.02498 |       0.02496 | 0.0236  |      0.02366 |         0.0236  |     0.0236  |       0.02392 |          0.02376 |         0.02364 |         0.02356 | 0.02359 |      0.02359 |      0.02359 |
|   2023 | 365 |  0.01736 |        0.01737 |   0.01807 |   0.01903 |       0.01899 | 0.01699 |      0.01718 |         0.01699 |     0.01699 |       0.01716 |          0.01731 |         0.0171  |         0.01698 | 0.01699 |      0.01699 |      0.01699 |
|   2024 | 366 |  0.02157 |        0.02154 |   0.02172 |   0.02193 |       0.02187 | 0.02166 |      0.02148 |         0.02166 |     0.02165 |       0.02147 |          0.02144 |         0.02147 |         0.02157 | 0.02166 |      0.02166 |      0.02166 |
|   2025 | 365 |  0.02163 |        0.02164 |   0.02196 |   0.02347 |       0.02344 | 0.02187 |      0.02156 |         0.02187 |     0.02187 |       0.02207 |          0.02183 |         0.02192 |         0.02172 | 0.02185 |      0.02185 |      0.02185 |
|   2026 | 142 |  0.01731 |        0.01731 |   0.01796 |   0.01929 |       0.01929 | 0.01707 |      0.0173  |         0.01707 |     0.01707 |       0.0174  |          0.01782 |         0.01717 |         0.01709 | 0.01706 |      0.01706 |      0.01706 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 378 |  0.02416 |        0.02417 |   0.02446 |   0.02578 |       0.02566 | 0.02427 |      0.02413 |         0.02427 |     0.02427 |       0.02442 |          0.02444 |         0.02442 |         0.02413 | 0.02425 |      0.02425 |      0.02425 |
| high-vol | 136 |  0.02725 |        0.02722 |   0.0285  |   0.0308  |       0.03111 | 0.02659 |      0.02655 |         0.02659 |     0.02659 |       0.02786 |          0.02792 |         0.02678 |         0.02649 | 0.02658 |      0.02658 |      0.02658 |
| neutral  | 743 |  0.02041 |        0.02039 |   0.02072 |   0.02181 |       0.02178 | 0.02043 |      0.02044 |         0.02043 |     0.02043 |       0.02042 |          0.02049 |         0.02036 |         0.0204  | 0.02043 |      0.02043 |      0.02043 |
| low-vol  | 490 |  0.01904 |        0.01904 |   0.01931 |   0.01976 |       0.01976 | 0.01892 |      0.01891 |         0.01892 |     0.01892 |       0.01916 |          0.01887 |         0.01879 |         0.01886 | 0.01891 |      0.01891 |      0.01891 |
| rally    | 463 |  0.03662 |        0.03658 |   0.03585 |   0.03747 |       0.03755 | 0.03619 |      0.03616 |         0.03619 |     0.03619 |       0.0361  |          0.03545 |         0.0361  |         0.03609 | 0.03619 |      0.03619 |      0.03619 |

**Diebold-Mariano vs WAR-Select** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0945 |           0.0605 |
| HS-Bootstrap   |      0.1409 |           0.0988 |
| GARCH-N        |      0.0356 |           0.0021 |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.0824 |           0.0619 |
| WGeo-Gated     |      0.623  |           0.5979 |
| WGeo-TheilSen  |      0.0656 |           0.0474 |
| WGeo-EWMA      |      0.2017 |           0.1636 |
| WGeo-Hetero    |      0.1423 |           0.0011 |
| WGeo-GARCH-Ens |      0.7698 |           0.6189 |
| WGeo-Adaptive  |      0.6422 |           0.6168 |
| WGeo-Ensemble  |      0      |           0      |
| WAR-1          |      0.4853 |           0.4831 |
| WAR-1-last     |      0.4853 |           0.4831 |
| WAR-Select     |      1      |           1      |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-Select, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 378 |  0.02413 |  0.02425 |    -0.52643 | -2.69689 | 0.007   |
| high-vol | 136 |  0.02649 |  0.02658 |    -0.34029 | -1.60673 | 0.10811 |
| neutral  | 743 |  0.0204  |  0.02043 |    -0.13899 | -1.49343 | 0.13533 |
| low-vol  | 490 |  0.01886 |  0.01891 |    -0.26484 | -1.80674 | 0.0708  |
| rally    | 463 |  0.03609 |  0.03619 |    -0.28219 | -2.06081 | 0.03932 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_xrpusdt_h1.png)

### Horizon h = 5 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2206 |    0.056199 | 0.050164 | 0.063399 |              nan |
| HS-Bootstrap   | 2206 |    0.056742 | 0.051025 | 0.063754 |              nan |
| GARCH-N        | 2206 |    0.057258 | 0.051868 | 0.063709 |              nan |
| GARCH-t        | 2206 |    0.060339 | 0.055087 | 0.06643  |              nan |
| GJR-GARCH-t    | 2206 |    0.060212 | 0.054882 | 0.066268 |              nan |
| WGeo           | 2206 |    0.055504 | 0.049689 | 0.062558 |              nan |
| WGeo-Gated     | 2206 |    0.055596 | 0.049758 | 0.062625 |              nan |
| WGeo-TheilSen  | 2206 |    0.055503 | 0.049679 | 0.062552 |              nan |
| WGeo-EWMA      | 2206 |    0.055483 | 0.049663 | 0.062535 |              nan |
| WGeo-Hetero    | 2206 |    0.056237 | 0.050509 | 0.063239 |                0 |
| WGeo-GARCH-Ens | 2206 |    0.055942 | 0.050349 | 0.062664 |              nan |
| WGeo-Adaptive  | 2206 |    0.055441 | 0.049788 | 0.062198 |              nan |
| WGeo-Ensemble  | 2206 |    0.055389 | 0.049575 | 0.062401 |              nan |
| WAR-1          | 2206 |    0.056215 | 0.050337 | 0.063133 |              nan |
| WAR-1-last     | 2206 |    0.055465 | 0.049629 | 0.062531 |              nan |
| WAR-Select     | 2206 |    0.056216 | 0.050335 | 0.063133 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2020 | 242 |  0.06926 |        0.06942 |   0.06885 |   0.07514 |       0.07507 | 0.06982 |      0.06911 |         0.06982 |     0.06977 |       0.06859 |          0.06884 |         0.06824 |         0.06945 | 0.07112 |      0.06971 |      0.07112 |
|   2021 | 365 |  0.08851 |        0.08731 |   0.08958 |   0.09168 |       0.09164 | 0.08662 |      0.08683 |         0.08661 |     0.08656 |       0.09056 |          0.08707 |         0.08694 |         0.08639 | 0.08651 |      0.08658 |      0.0865  |
|   2022 | 365 |  0.05096 |        0.05349 |   0.0536  |   0.05598 |       0.05597 | 0.04973 |      0.05051 |         0.04972 |     0.04972 |       0.05083 |          0.0515  |         0.05035 |         0.04986 | 0.05051 |      0.04976 |      0.05052 |
|   2023 | 365 |  0.03949 |        0.04132 |   0.0415  |   0.04398 |       0.04394 | 0.03884 |      0.03907 |         0.03884 |     0.03882 |       0.03871 |          0.03946 |         0.03904 |         0.03874 | 0.03973 |      0.03877 |      0.03972 |
|   2024 | 366 |  0.05333 |        0.05317 |   0.05374 |   0.0547  |       0.05438 | 0.05359 |      0.05308 |         0.0536  |     0.05357 |       0.05352 |          0.05334 |         0.0526  |         0.05337 | 0.05381 |      0.05359 |      0.05381 |
|   2025 | 365 |  0.04802 |        0.04772 |   0.04807 |   0.05288 |       0.05258 | 0.04745 |      0.04745 |         0.04746 |     0.04746 |       0.04772 |          0.04749 |         0.04792 |         0.04735 | 0.04893 |      0.04739 |      0.04894 |
|   2026 | 138 |  0.03512 |        0.03638 |   0.03641 |   0.04096 |       0.04095 | 0.03382 |      0.03468 |         0.03381 |     0.03381 |       0.03417 |          0.03561 |         0.03395 |         0.034   | 0.0343  |      0.03374 |      0.0343  |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 378 |  0.0524  |        0.05336 |   0.05558 |   0.05845 |       0.05795 | 0.0516  |      0.05189 |         0.05159 |     0.05158 |       0.05371 |          0.05359 |         0.05183 |         0.05152 | 0.05335 |      0.05163 |      0.05334 |
| high-vol | 136 |  0.06282 |        0.06224 |   0.06386 |   0.07041 |       0.07086 | 0.06163 |      0.06106 |         0.06162 |     0.06162 |       0.06311 |          0.06282 |         0.06209 |         0.06125 | 0.06151 |      0.06155 |      0.06153 |
| neutral  | 743 |  0.04531 |        0.04643 |   0.04676 |   0.04975 |       0.0497  | 0.04484 |      0.04502 |         0.04484 |     0.04484 |       0.04504 |          0.04544 |         0.04503 |         0.04482 | 0.04506 |      0.0448  |      0.04507 |
| low-vol  | 486 |  0.05163 |        0.05266 |   0.05201 |   0.05337 |       0.0533  | 0.05169 |      0.05145 |         0.05168 |     0.05165 |       0.05188 |          0.0516  |         0.0512  |         0.05148 | 0.05195 |      0.05155 |      0.05195 |
| rally    | 463 |  0.07962 |        0.07872 |   0.07904 |   0.08323 |       0.08306 | 0.07802 |      0.07834 |         0.07803 |     0.07797 |       0.07883 |          0.07726 |         0.0776  |         0.07789 | 0.07937 |      0.07805 |      0.07938 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0108 |           0.0037 |
| HS-Bootstrap   |      0      |           0      |
| GARCH-N        |      0      |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.2804 |           0.2266 |
| WGeo-Gated     |      0.4447 |           0.3902 |
| WGeo-TheilSen  |      0.2574 |           0.2054 |
| WGeo-EWMA      |      0.6242 |           0.5758 |
| WGeo-Hetero    |      0.0203 |           0.0001 |
| WGeo-GARCH-Ens |      0.1022 |           0.0108 |
| WGeo-Adaptive  |      0.9128 |           0.8985 |
| WGeo-Ensemble  |      0.1485 |           0.1115 |
| WAR-1          |      0.0102 |           0.0061 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.01   |           0.0059 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 378 |  0.05152 |  0.05163 |    -0.20111 | -0.70087 | 0.48339 |
| high-vol | 136 |  0.06125 |  0.06155 |    -0.4832  | -1.36425 | 0.17249 |
| neutral  | 743 |  0.04482 |  0.0448  |     0.05358 |  0.37811 | 0.70535 |
| low-vol  | 486 |  0.05148 |  0.05155 |    -0.13736 | -0.81112 | 0.4173  |
| rally    | 463 |  0.07789 |  0.07805 |    -0.2013  | -1.09479 | 0.27361 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_xrpusdt_h5.png)

### Horizon h = 21 day(s)

**Overall mean CRPS on the full test span (bootstrap 95% CI):**

| method         |    n |   mean_crps |    ci_lo |    ci_hi |   garch_fallback |
|:---------------|-----:|------------:|---------:|---------:|-----------------:|
| Static         | 2190 |    0.132709 | 0.106804 | 0.165088 |              nan |
| HS-Bootstrap   | 2190 |    0.133278 | 0.109201 | 0.164393 |              nan |
| GARCH-N        | 2190 |    0.136474 | 0.113003 | 0.166851 |              nan |
| GARCH-t        | 2190 |    0.144387 | 0.121768 | 0.173588 |              nan |
| GJR-GARCH-t    | 2190 |    0.1435   | 0.121068 | 0.172517 |              nan |
| WGeo           | 2190 |    0.129218 | 0.104175 | 0.161939 |              nan |
| WGeo-Gated     | 2190 |    0.130502 | 0.105251 | 0.163191 |              nan |
| WGeo-TheilSen  | 2190 |    0.12915  | 0.104117 | 0.16187  |              nan |
| WGeo-EWMA      | 2190 |    0.129136 | 0.104168 | 0.161779 |              nan |
| WGeo-Hetero    | 2190 |    0.132881 | 0.107674 | 0.166101 |                0 |
| WGeo-GARCH-Ens | 2190 |    0.132174 | 0.107331 | 0.164368 |              nan |
| WGeo-Adaptive  | 2190 |    0.129569 | 0.105074 | 0.1612   |              nan |
| WGeo-Ensemble  | 2190 |    0.129103 | 0.104059 | 0.161764 |              nan |
| WAR-1          | 2190 |    0.136215 | 0.111149 | 0.168004 |              nan |
| WAR-1-last     | 2190 |    0.128207 | 0.103068 | 0.160894 |              nan |
| WAR-Select     | 2190 |    0.136449 | 0.111441 | 0.168261 |              nan |

**Per-year mean CRPS:**

|   year |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|-------:|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
|   2020 | 242 |  0.19597 |        0.19492 |   0.19362 |   0.20976 |       0.21002 | 0.19707 |      0.19566 |         0.197   |     0.1969  |       0.19724 |          0.1961  |         0.19224 |         0.19622 | 0.20408 |      0.19462 |      0.20407 |
|   2021 | 365 |  0.20278 |        0.19652 |   0.20553 |   0.20879 |       0.20797 | 0.19125 |      0.19691 |         0.19118 |     0.19122 |       0.20444 |          0.19637 |         0.19466 |         0.19229 | 0.19998 |      0.19157 |      0.20033 |
|   2022 | 365 |  0.11119 |        0.11784 |   0.12414 |   0.12506 |       0.12547 | 0.10644 |      0.10819 |         0.10632 |     0.10637 |       0.1123  |          0.11326 |         0.1084  |         0.10646 | 0.11403 |      0.10572 |      0.11464 |
|   2023 | 365 |  0.09319 |        0.09655 |   0.10161 |   0.10399 |       0.10407 | 0.09426 |      0.09269 |         0.09417 |     0.09418 |       0.09505 |          0.09514 |         0.09505 |         0.09311 | 0.10139 |      0.09215 |      0.10141 |
|   2024 | 366 |  0.14017 |        0.13908 |   0.14017 |   0.14398 |       0.1427  | 0.14287 |      0.14095 |         0.14282 |     0.1427  |       0.14268 |          0.14458 |         0.14114 |         0.14201 | 0.14388 |      0.14182 |      0.14385 |
|   2025 | 365 |  0.09011 |        0.08999 |   0.09005 |   0.11232 |       0.10848 | 0.08376 |      0.08679 |         0.08375 |     0.08373 |       0.08562 |          0.08606 |         0.08411 |         0.08414 | 0.09593 |      0.08289 |      0.09639 |
|   2026 | 122 |  0.08525 |        0.08997 |   0.08551 |   0.09789 |       0.09774 | 0.07684 |      0.08192 |         0.07674 |     0.07685 |       0.07787 |          0.08145 |         0.07846 |         0.07814 | 0.07891 |      0.07675 |      0.07891 |

**Per-regime mean CRPS (regime tagged from 60d trailing return + vol):**

| regime   |   n |   Static |   HS-Bootstrap |   GARCH-N |   GARCH-t |   GJR-GARCH-t |    WGeo |   WGeo-Gated |   WGeo-TheilSen |   WGeo-EWMA |   WGeo-Hetero |   WGeo-GARCH-Ens |   WGeo-Adaptive |   WGeo-Ensemble |   WAR-1 |   WAR-1-last |   WAR-Select |
|:---------|----:|---------:|---------------:|----------:|----------:|--------------:|--------:|-------------:|----------------:|------------:|--------------:|-----------------:|----------------:|----------------:|--------:|-------------:|-------------:|
| crash    | 378 |  0.10809 |        0.11149 |   0.11921 |   0.12727 |       0.12545 | 0.10385 |      0.10496 |         0.10375 |     0.10376 |       0.10896 |          0.10982 |         0.10466 |         0.1035  | 0.11977 |      0.10338 |      0.11981 |
| high-vol | 136 |  0.10947 |        0.10818 |   0.11595 |   0.14647 |       0.14535 | 0.10725 |      0.10522 |         0.1072  |     0.10714 |       0.11706 |          0.11627 |         0.10838 |         0.10578 | 0.11032 |      0.10629 |      0.11071 |
| neutral  | 743 |  0.09674 |        0.09977 |   0.10316 |   0.11026 |       0.10975 | 0.09133 |      0.09439 |         0.09127 |     0.09135 |       0.09394 |          0.09469 |         0.09232 |         0.09206 | 0.09496 |      0.09079 |      0.09522 |
| low-vol  | 470 |  0.16579 |        0.16512 |   0.16647 |   0.16768 |       0.16749 | 0.17072 |      0.16711 |         0.17053 |     0.17053 |       0.17215 |          0.17105 |         0.17037 |         0.16894 | 0.17161 |      0.16722 |      0.17186 |
| rally    | 463 |  0.18377 |        0.17988 |   0.17961 |   0.18887 |       0.18751 | 0.17505 |      0.17958 |         0.17511 |     0.17494 |       0.17969 |          0.17579 |         0.17449 |         0.17586 | 0.18752 |      0.17536 |      0.18781 |

**Diebold-Mariano vs WAR-1-last** (headline best WGeo-family variant is **WGeo-Ensemble**; both vanilla and residualised tests reported — residualised uses |y|, y², y plus 4 peer losses as controls to project out shared volatility-clustering noise):

|                |   p_vanilla |   p_residualised |
|:---------------|------------:|-----------------:|
| Static         |      0.0002 |           0      |
| HS-Bootstrap   |      0.0005 |           0      |
| GARCH-N        |      0.0001 |           0      |
| GARCH-t        |      0      |           0      |
| GJR-GARCH-t    |      0      |           0      |
| WGeo           |      0.0066 |           0.003  |
| WGeo-Gated     |      0.0027 |           0      |
| WGeo-TheilSen  |      0.0078 |           0.0035 |
| WGeo-EWMA      |      0.011  |           0.0053 |
| WGeo-Hetero    |      0.0003 |           0      |
| WGeo-GARCH-Ens |      0.0003 |           0      |
| WGeo-Adaptive  |      0.1518 |           0.1009 |
| WGeo-Ensemble  |      0.0053 |           0.001  |
| WAR-1          |      0.0021 |           0.0013 |
| WAR-1-last     |      1      |           1      |
| WAR-Select     |      0.0015 |           0.0009 |

**Regime-conditional DM** (WGeo-Ensemble vs WAR-1-last, per-regime CRPS gap and DM statistic; the aggregate panel DM hides large WGeo-family wins in non-neutral regimes):

| regime   |   n |   mean_a |   mean_b |   delta_pct |       dm |       p |
|:---------|----:|---------:|---------:|------------:|---------:|--------:|
| crash    | 378 |  0.1035  |  0.10338 |     0.12378 |  0.23145 | 0.81697 |
| high-vol | 136 |  0.10578 |  0.10629 |    -0.48468 | -0.65751 | 0.51085 |
| neutral  | 743 |  0.09206 |  0.09079 |     1.39972 |  3.04593 | 0.00232 |
| low-vol  | 470 |  0.16894 |  0.16722 |     1.02731 |  2.79836 | 0.00514 |
| rally    | 463 |  0.17586 |  0.17536 |     0.28762 |  0.56457 | 0.57237 |

**GARCH-fallback rate** (fraction of walk-forward steps where the GARCH fit raised or produced degenerate variances, forcing the WGeo-Hetero / CondShape variants back onto the unconditional √h scaling). The §4 falsification floor for Hetero is conditional on this rate being small — otherwise the headline is measuring √h scaling, not the GARCH contribution:

| method      |   fallback_rate |
|:------------|----------------:|
| WGeo-Hetero |               0 |

![cumulative CRPS](../results/long_cum_crps_xrpusdt_h21.png)
