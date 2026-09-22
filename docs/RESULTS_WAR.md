# Wasserstein Autoregression benchmark — scoring and sensitivity

_Generated 2026-09-22 by `scripts/summarize_war.py` from `results/war_sensitivity.json` (scripts/score_new_method.py --verbose against the results/long_*.json saved by the 2026-05-24 panel; raw logs in results/war_scores/)._

The Wasserstein autoregressive model of Zhang, Kokoszka & Petersen (2022) applied to the same rolling-90-day quantile vectors the WGeo family uses ("WAR on rolling-ECDF densities", [`THEORY.md §2.11`](THEORY.md)). Every row is a walk-forward run at exactly the test indices of the long-horizon panel, scored with CRPS against the saved per-step losses of every panel method. `dm_p_r vs best` is the residualised Diebold-Mariano p-value of the WAR variant against the best classical baseline in that cell (Static / HS-Bootstrap / GARCH-N / GARCH-t / GJR-GARCH-t); the last two columns compare `WGeo-Ensemble` with the WAR variant (negative Δ = WGeo-Ensemble better; vanilla DM p). The regenerated `RESULTS_LONG.md` Headline 3 carries the residualised statistic for the three panel variants and the §4 falsification count.

## Reading

- **WAR ≈ Static at h=1, worse at h=21 under the `sum` rule.** With lag-1 Wasserstein autocorrelation ≈0.98 (mostly mechanical: consecutive densities share 89/90 observations), WAR shrinks today's density toward the 730-day barycentre. At h=1 that is nearly the identity; over 21 iterated steps the shrinkage lags the volatility path.
- **The h-day location rule matters more than the AR order.** `WAR-1-last` (WGeo's rule: terminal daily median, √h-scaled shape) beats GARCH-N at h=21 by 2.4–3.5% with residualised p<0.05 on BTC and ETH, close to `WGeo-Ensemble`'s own margins; `sum` (h-fold daily median) and `conv` (independent convolution of the h daily laws) are 1–4% worse than GARCH-N. Multiplying a noisy 90-day daily median by 21 injects location noise that the √h shortcut avoids. This is a statement about the panel's h-day conversion, not about WAR.
- **Order selection changes nothing; the window grid does.** `WAR-Select` picks p=1 and the widest admissible K almost everywhere and matches `WAR-1` to the fourth decimal. `WAR-Paper`, confined to the paper's intraday grid K ∈ {20, 62}, is worse at h=21 (a 62-density barycentre is noisier). The paper's unpenalised in-sample criterion cannot separate orders on this data.
- **WAR's tails are not calibrated** (`RESULTS_VAR_ES.md`): `WAR-1` passes Kupiec in 6/20 cells and `WAR-1-last` in 8/20, versus 17/20 for Static and `WGeo-Gated`; both fail every Acerbi-Szekely Z1 cell. Shrinking toward a 730-day barycentre gives a body that scores well under CRPS and tails that under-react, the same defect as `WGeo-TheilSen` / `WGeo-EWMA`.
- **Less overlap helps slightly.** `stride=10` (every 10th density) lowers the mechanical persistence and is marginally better than `stride=1`; a 30-day density window is worse everywhere.

- **Headline 3 verdict (`RESULTS_LONG.md`): the §4 test fails, 6/15.** `WGeo-Ensemble` beats `WAR-1-last` in all five h=1 cells, ties at h=5, and loses four of five h=21 cells with residualised p ≤ 0.047. On rolling-ECDF densities, mean reversion toward the barycentre beats extrapolation at 21 days.

## Panel variants, 5 assets × 3 horizons

| method     | symbol   |   h |     CRPS | best classical          | Δ vs best classical   |   dm_p_r vs best |   WGeo-Ensemble − WAR |   dm_p (vanilla) |
|:-----------|:---------|----:|---------:|:------------------------|:----------------------|-----------------:|----------------------:|-----------------:|
| WAR-1      | BTC/USDT |   1 | 0.016208 | Static (0.016236)       | -0.18%                |           0.5326 |             -3.9e-05  |           0.0004 |
| WAR-1      | BTC/USDT |   5 | 0.03751  | Static (0.037367)       | +0.38%                |           0.6356 |             -0.000449 |           0.0955 |
| WAR-1      | BTC/USDT |  21 | 0.085867 | GARCH-N (0.084848)      | +1.20%                |           0.6109 |             -0.002709 |           0.2485 |
| WAR-1      | ETH/USDT |   1 | 0.021786 | HS-Bootstrap (0.021893) | -0.49%                |           0.065  |             -4.6e-05  |           0.001  |
| WAR-1      | ETH/USDT |   5 | 0.049971 | Static (0.049834)       | +0.27%                |           0.7106 |             -0.000715 |           0.0304 |
| WAR-1      | ETH/USDT |  21 | 0.114697 | GARCH-N (0.112970)      | +1.53%                |           0.4639 |             -0.004881 |           0.1047 |
| WAR-1      | SOL/USDT |   1 | 0.025086 | Static (0.025075)       | +0.04%                |           0.9097 |             -5.6e-05  |           0.0099 |
| WAR-1      | SOL/USDT |   5 | 0.058304 | Static (0.057622)       | +1.18%                |           0.2656 |             -0.001151 |           0.0141 |
| WAR-1      | SOL/USDT |  21 | 0.136555 | GARCH-N (0.133568)      | +2.24%                |           0.4474 |             -0.006723 |           0.1095 |
| WAR-1      | BNB/USDT |   1 | 0.020203 | GARCH-N (0.020199)      | +0.02%                |           0.9481 |             -5.8e-05  |           0.0001 |
| WAR-1      | BNB/USDT |   5 | 0.046258 | Static (0.046297)       | -0.08%                |           0.9065 |             -0.000476 |           0.1038 |
| WAR-1      | BNB/USDT |  21 | 0.1053   | Static (0.105553)       | -0.24%                |           0.9083 |             -0.002681 |           0.3258 |
| WAR-1      | XRP/USDT |   1 | 0.024428 | HS-Bootstrap (0.024548) | -0.49%                |           0.1011 |             -7e-05    |           0      |
| WAR-1      | XRP/USDT |   5 | 0.056215 | Static (0.056199)       | +0.03%                |           0.9707 |             -0.000826 |           0.0054 |
| WAR-1      | XRP/USDT |  21 | 0.136215 | Static (0.132709)       | +2.64%                |           0.1936 |             -0.007112 |           0.0059 |
| WAR-Select | BTC/USDT |   1 | 0.016208 | Static (0.016236)       | -0.18%                |           0.533  |             -3.9e-05  |           0.0004 |
| WAR-Select | BTC/USDT |   5 | 0.037512 | Static (0.037367)       | +0.39%                |           0.6309 |             -0.000451 |           0.0938 |
| WAR-Select | BTC/USDT |  21 | 0.085858 | GARCH-N (0.084848)      | +1.19%                |           0.6137 |             -0.002699 |           0.2494 |
| WAR-Select | ETH/USDT |   1 | 0.021786 | HS-Bootstrap (0.021893) | -0.49%                |           0.0648 |             -4.6e-05  |           0.001  |
| WAR-Select | ETH/USDT |   5 | 0.049968 | Static (0.049834)       | +0.27%                |           0.716  |             -0.000712 |           0.0309 |
| WAR-Select | ETH/USDT |  21 | 0.114724 | GARCH-N (0.112970)      | +1.55%                |           0.4576 |             -0.004908 |           0.1008 |
| WAR-Select | SOL/USDT |   1 | 0.025086 | Static (0.025075)       | +0.04%                |           0.9069 |             -5.6e-05  |           0.0095 |
| WAR-Select | SOL/USDT |   5 | 0.058315 | Static (0.057622)       | +1.20%                |           0.2587 |             -0.001162 |           0.0135 |
| WAR-Select | SOL/USDT |  21 | 0.136713 | GARCH-N (0.133568)      | +2.35%                |           0.4283 |             -0.00688  |           0.1048 |
| WAR-Select | BNB/USDT |   1 | 0.020203 | GARCH-N (0.020199)      | +0.02%                |           0.9461 |             -5.9e-05  |           0.0001 |
| WAR-Select | BNB/USDT |   5 | 0.046257 | Static (0.046297)       | -0.08%                |           0.9053 |             -0.000476 |           0.1045 |
| WAR-Select | BNB/USDT |  21 | 0.105347 | Static (0.105553)       | -0.19%                |           0.9255 |             -0.002728 |           0.3182 |
| WAR-Select | XRP/USDT |   1 | 0.024428 | HS-Bootstrap (0.024548) | -0.49%                |           0.1003 |             -6.9e-05  |           0      |
| WAR-Select | XRP/USDT |   5 | 0.056216 | Static (0.056199)       | +0.03%                |           0.969  |             -0.000827 |           0.0052 |
| WAR-Select | XRP/USDT |  21 | 0.136449 | Static (0.132709)       | +2.82%                |           0.1636 |             -0.007346 |           0.0043 |
| WAR-Paper  | BTC/USDT |   1 | 0.016209 | Static (0.016236)       | -0.17%                |           0.5514 |             -4.1e-05  |           0.0003 |
| WAR-Paper  | BTC/USDT |   5 | 0.037561 | Static (0.037367)       | +0.52%                |           0.5155 |             -0.0005   |           0.0611 |
| WAR-Paper  | BTC/USDT |  21 | 0.088117 | GARCH-N (0.084848)      | +3.85%                |           0.1343 |             -0.004958 |           0.0479 |
| WAR-Paper  | ETH/USDT |   1 | 0.021783 | HS-Bootstrap (0.021893) | -0.50%                |           0.0574 |             -4.4e-05  |           0.0022 |
| WAR-Paper  | ETH/USDT |   5 | 0.04988  | Static (0.049834)       | +0.09%                |           0.9    |             -0.000625 |           0.0605 |
| WAR-Paper  | ETH/USDT |  21 | 0.114697 | GARCH-N (0.112970)      | +1.53%                |           0.483  |             -0.004881 |           0.1296 |
| WAR-Paper  | SOL/USDT |   1 | 0.02508  | Static (0.025075)       | +0.02%                |           0.9579 |             -5e-05    |           0.0223 |
| WAR-Paper  | SOL/USDT |   5 | 0.058179 | Static (0.057622)       | +0.97%                |           0.3619 |             -0.001026 |           0.0298 |
| WAR-Paper  | SOL/USDT |  21 | 0.136314 | GARCH-N (0.133568)      | +2.06%                |           0.5153 |             -0.006482 |           0.1675 |
| WAR-Paper  | BNB/USDT |   1 | 0.020207 | GARCH-N (0.020199)      | +0.04%                |           0.8899 |             -6.3e-05  |           0      |
| WAR-Paper  | BNB/USDT |   5 | 0.046284 | Static (0.046297)       | -0.03%                |           0.9693 |             -0.000502 |           0.0703 |
| WAR-Paper  | BNB/USDT |  21 | 0.105162 | Static (0.105553)       | -0.37%                |           0.8692 |             -0.002543 |           0.3543 |
| WAR-Paper  | XRP/USDT |   1 | 0.024424 | HS-Bootstrap (0.024548) | -0.51%                |           0.0853 |             -6.5e-05  |           0.0001 |
| WAR-Paper  | XRP/USDT |   5 | 0.056134 | Static (0.056199)       | -0.12%                |           0.8729 |             -0.000745 |           0.0087 |
| WAR-Paper  | XRP/USDT |  21 | 0.135479 | Static (0.132709)       | +2.09%                |           0.2988 |             -0.006376 |           0.0161 |

## Sensitivity: location rule, density window, stride (BTC + ETH)

| method         | symbol   |   h |     CRPS | best classical          | Δ vs best classical   |   dm_p_r vs best |   WGeo-Ensemble − WAR |   dm_p (vanilla) |
|:---------------|:---------|----:|---------:|:------------------------|:----------------------|-----------------:|----------------------:|-----------------:|
| WAR-1          | BTC/USDT |   1 | 0.016208 | Static (0.016236)       | -0.18%                |           0.5326 |             -3.9e-05  |           0.0004 |
| WAR-1          | BTC/USDT |   5 | 0.03751  | Static (0.037367)       | +0.38%                |           0.6356 |             -0.000449 |           0.0955 |
| WAR-1          | BTC/USDT |  21 | 0.085867 | GARCH-N (0.084848)      | +1.20%                |           0.6109 |             -0.002709 |           0.2485 |
| WAR-1          | ETH/USDT |   1 | 0.021786 | HS-Bootstrap (0.021893) | -0.49%                |           0.065  |             -4.6e-05  |           0.001  |
| WAR-1          | ETH/USDT |   5 | 0.049971 | Static (0.049834)       | +0.27%                |           0.7106 |             -0.000715 |           0.0304 |
| WAR-1          | ETH/USDT |  21 | 0.114697 | GARCH-N (0.112970)      | +1.53%                |           0.4639 |             -0.004881 |           0.1047 |
| WAR-1-last     | BTC/USDT |   1 | 0.016208 | Static (0.016236)       | -0.18%                |           0.5326 |             -3.9e-05  |           0.0004 |
| WAR-1-last     | BTC/USDT |   5 | 0.037099 | Static (0.037367)       | -0.72%                |           0.0723 |             -3.8e-05  |           0.3542 |
| WAR-1-last     | BTC/USDT |  21 | 0.08284  | GARCH-N (0.084848)      | -2.37%                |           0.0132 |              0.000318 |           0.3432 |
| WAR-1-last     | ETH/USDT |   1 | 0.021786 | HS-Bootstrap (0.021893) | -0.49%                |           0.065  |             -4.6e-05  |           0.001  |
| WAR-1-last     | ETH/USDT |   5 | 0.049246 | Static (0.049834)       | -1.18%                |           0.0015 |              1e-05    |           0.8549 |
| WAR-1-last     | ETH/USDT |  21 | 0.109047 | GARCH-N (0.112970)      | -3.47%                |           0      |              0.00077  |           0.0942 |
| WAR-1-conv     | BTC/USDT |   1 | 0.016207 | Static (0.016236)       | -0.18%                |           0.5275 |             -3.9e-05  |           0.0009 |
| WAR-1-conv     | BTC/USDT |   5 | 0.037803 | Static (0.037367)       | +1.17%                |           0.2058 |             -0.000742 |           0.021  |
| WAR-1-conv     | BTC/USDT |  21 | 0.088276 | GARCH-N (0.084848)      | +4.04%                |           0.1229 |             -0.005117 |           0.048  |
| WAR-1-conv     | ETH/USDT |   1 | 0.021788 | HS-Bootstrap (0.021893) | -0.48%                |           0.0689 |             -4.8e-05  |           0.0008 |
| WAR-1-conv     | ETH/USDT |   5 | 0.050363 | Static (0.049834)       | +1.06%                |           0.2277 |             -0.001107 |           0.0032 |
| WAR-1-conv     | ETH/USDT |  21 | 0.116954 | GARCH-N (0.112970)      | +3.53%                |           0.2214 |             -0.007137 |           0.0355 |
| WAR-1-w30      | BTC/USDT |   1 | 0.016359 | Static (0.016236)       | +0.75%                |           0.0645 |             -0.00019  |           0.0002 |
| WAR-1-w30      | BTC/USDT |   5 | 0.038324 | Static (0.037367)       | +2.56%                |           0.0247 |             -0.001263 |           0.0021 |
| WAR-1-w30      | BTC/USDT |  21 | 0.087472 | GARCH-N (0.084848)      | +3.09%                |           0.2322 |             -0.004313 |           0.0814 |
| WAR-1-w30      | ETH/USDT |   1 | 0.021998 | HS-Bootstrap (0.021893) | +0.48%                |           0.2334 |             -0.000259 |           0.0003 |
| WAR-1-w30      | ETH/USDT |   5 | 0.051183 | Static (0.049834)       | +2.71%                |           0.0262 |             -0.001927 |           0.001  |
| WAR-1-w30      | ETH/USDT |  21 | 0.118592 | GARCH-N (0.112970)      | +4.98%                |           0.1444 |             -0.008775 |           0.0285 |
| WAR-1-stride10 | BTC/USDT |   1 | 0.016178 | Static (0.016236)       | -0.36%                |           0.1534 |             -1e-05    |           0.1796 |
| WAR-1-stride10 | BTC/USDT |   5 | 0.037178 | Static (0.037367)       | -0.51%                |           0.3687 |             -0.000117 |           0.5651 |
| WAR-1-stride10 | BTC/USDT |  21 | 0.083936 | GARCH-N (0.084848)      | -1.07%                |           0.3775 |             -0.000778 |           0.6144 |
| WAR-1-stride10 | ETH/USDT |   1 | 0.021751 | HS-Bootstrap (0.021893) | -0.65%                |           0.0054 |             -1.2e-05  |           0.1718 |
| WAR-1-stride10 | ETH/USDT |   5 | 0.049595 | Static (0.049834)       | -0.48%                |           0.3206 |             -0.000339 |           0.1703 |
| WAR-1-stride10 | ETH/USDT |  21 | 0.113031 | GARCH-N (0.112970)      | +0.05%                |           0.9455 |             -0.003215 |           0.1123 |

`WAR-1`: p=1, all 641 densities, `location="sum"`. `WAR-1-last`: `location="last"`. `WAR-1-conv`: `location="conv"` (3 000 seeded paths). `WAR-1-w30`: 30-day density window. `WAR-1-stride10`: every 10th density. `WAR-Paper`: K ∈ {20, 62}, p ∈ {1..10}; `WAR-Select`: K ∈ {20, 62, 250, all}, p ∈ {1..5}; both by the paper's sequential in-sample one-step W₂ criterion over 60 origins.

Reproduce: `uv run python scripts/score_new_method.py --method WAR-1 --verbose` (and the other names in `_BUILTIN_FACTORIES`), then rebuild the JSON and this file.
