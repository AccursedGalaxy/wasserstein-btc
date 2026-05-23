# Pre-registration — v0.5 Out-of-Sample Evaluation

| Field | Value |
|---|---|
| **Lock date** | 2026-05-24 |
| **Evaluation window** | 2026-06-01 → 2027-05-31 (12 months) |
| **Evaluation date** | 2027-06-01 (after the window closes) |
| **Code state at lock** | `master` at commit ahead of `1a3ab99` (this PR) — git-tag `preregistration-v0.5` to be applied to the merge commit |

This document is the contract. Every choice that could be tuned to favour
the headline number after seeing data is locked here, before the
evaluation window opens. Any deviation between this document and what is
actually run on 2027-06-01 must be reported as a deviation in
`CHANGELOG.md` with an ISO-8601 date and a one-line reason. A deviation
does not invalidate the result; an unreported deviation does.

## Why pre-register

The v0.4 / v0.5 reports use ≈ 6.75 years of data with no held-out test
set; hyperparameters were tuned on the early epoch and the headline
numbers were chosen after seeing the rest. That is the right
methodology for a methods paper, but it does not address the
researcher-degrees-of-freedom concern: someone with access to the same
data could plausibly find *some* family of forecasters that wins on
*some* subset of comparators and *some* DM control set. Pre-registering
the forecaster, the comparators, the control set, and the falsification
threshold *before* the evaluation window opens is the only way to
distinguish "the method works" from "the method was chosen because it
won here." The 12-month evaluation window from 2026-06-01 onward is
data the model has never been tuned against, by construction.

## What is locked

### Headline forecaster

**Name:** `WGeo-Ensemble`

**Specification:** The equal-weight Wasserstein-2 barycentre, in
quantile-function coordinates, of three component forecasters:

```python
from wbtc.forecasters import (
    WassersteinGeodesicTheilSen,
    WassersteinGeodesicEWMA,
    WassersteinGeodesicGated,
    WGeoEnsemble,
)

def headline() -> WGeoEnsemble:
    return WGeoEnsemble(
        components=[
            lambda: WassersteinGeodesicTheilSen(window=90, lookback=20),
            lambda: WassersteinGeodesicEWMA(window=90, lookback=20, decay=0.85),
            lambda: WassersteinGeodesicGated(
                window=90, lookback=20, kappa_star=0.6, tau=5
            ),
        ],
        weights=None,  # equal — the W₂ barycentre
    )
```

The barycentre is computed as the equal-weight quantile-function average
of the three components, then isotonically projected back onto the
quantile cone. See [`docs/THEORY.md`](docs/THEORY.md) §2.9.

### Reference baselines

Exactly two — both factories from `wbtc.forecasters`, no arguments:

1. `StaticEmpirical` (reported as `Static` in tables)
2. `GarchNormal` (reported as `GARCH-N` in tables)

`Static` is the naive distributional baseline (current empirical
quantile, √h-scaled); `GARCH-N` is the standard parametric
volatility baseline. A win against both is the minimum bar for the v0.5
claim. Other baselines from `wbtc.forecasters` (RW-Drift, HS-Bootstrap,
GARCH-t, GJR-GARCH-t, etc.) may be reported as additional context but
do not contribute to the falsification decision.

### Assets

Four symbols, in this exact order:

1. `BTC/USDT`
2. `ETH/USDT`
3. `SOL/USDT`
4. `BNB/USDT`

Data source: Binance via `ccxt`, daily close, fetched with
`uv run wbtc fetch <symbol>`. Cached parquet provenance hashes are
recorded in `results/MANIFEST.json` at the time of evaluation.

Adding a fifth symbol (e.g. `XRP/USDT`, as noted in `ROADMAP.md`) is
permitted as a *supplementary* table; the headline is the
pre-registered four-asset panel.

### Horizons

Three horizons, in this exact order:

1. `h = 1` day
2. `h = 5` days
3. `h = 21` days

4 assets × 3 horizons = **12 cells**.

### Scoring rule

CRPS computed by `wbtc.scoring.crps_from_quantiles` on a 30-point
quantile grid (`K = 30`, made by `wbtc.quantiles.make_grid`). Per-step
CRPS aggregated by arithmetic mean.

### Diebold-Mariano control set

Residualised Diebold-Mariano with the **`vol`** control set:

```python
controls = [y, np.abs(y), y * y]
```

where `y` is the realised h-step log-return. This is the
mid-tier of the three sensitivity buckets reported in
[`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md). The `full` set (vol +
four peer-method loss series) is admissible under Giacomini-White but
rhetorically more endogenous; anchoring the falsification floor to
`vol` makes the bar robust to peer-loss correlations.

Test computed by `wbtc.scoring.diebold_mariano_residualised` with
Newey-West HAC lag `h - 1`. Two-sided p-values, threshold `p < 0.05`.

### Falsification threshold

The v0.5 claim is falsified if **either** of the following holds:

1. Fewer than **6 of 12** cells have residualised DM p_r < 0.05 (with
   `WGeo-Ensemble` having lower mean CRPS) under the `vol` control set,
   against `Static`.
2. Fewer than **6 of 12** cells have residualised DM p_r < 0.05 (with
   `WGeo-Ensemble` having lower mean CRPS) under the `vol` control set,
   against `GARCH-N`.

In v0.5 the `vol`-only count is 9/12 against each baseline; the floor
of 6/12 is set 3 cells below that to allow finite-sample drift in the
2026-06-01 → 2027-05-31 window without auto-falsifying. A clear pass
would be ≥ 9/12 against both, a clear fail would be ≤ 4/12 against
either, and 5–6/12 against either is an ambiguous outcome that triggers
a full re-examination of the framing (not a quiet republish).

### Harness configuration

These knobs are locked and may not be tuned during the evaluation window:

| Knob | Value | Source |
|---|---|---|
| Rolling train window | 730 days | `BURN_IN` in `scripts/run_long_horizon.py` |
| Quantile grid points | 30 | `K` argument to `run_long_horizon` |
| Walk-forward stride | 1 (per-step refit) | `_walk_forward_one` default |
| Burn-in start | After first 730 days of each asset's series | `BURN_IN` |
| Newey-West HAC lag | `h - 1` | `diebold_mariano` |
| Bootstrap block mean for CI | `max(2, h)` | `stationary_bootstrap_ci` |

### Component hyperparameters

| Component | Parameter | Value |
|---|---|---|
| `WGeo-TheilSen` | `window` | 90 |
| `WGeo-TheilSen` | `lookback` | 20 |
| `WGeo-EWMA` | `window` | 90 |
| `WGeo-EWMA` | `lookback` | 20 |
| `WGeo-EWMA` | `decay` | 0.85 |
| `WGeo-Gated` | `window` | 90 |
| `WGeo-Gated` | `lookback` | 20 |
| `WGeo-Gated` | `kappa_star` | 0.6 |
| `WGeo-Gated` | `tau` | 5 |
| `WGeo-Ensemble` | `weights` | equal (None) |

These match the values in `scripts/run_long_horizon.py` at the lock
date. They were chosen by hyperparameter sweep on the 2019-2022 early
epoch (`scripts/hyperparam_sweep.py` / `docs/RESULTS_LONG.md`
robustness section) and may not be re-swept against post-2026-06-01
data.

## What is NOT locked

The following are explicitly out of scope for the pre-registration —
they may evolve without invalidating the v0.5 claim:

- **Implementation refactors** that do not change forecast output. Any
  change must keep `tests/` green and pass a numerical regression
  check against `results/long_*.json`.
- **Additional supplementary baselines or assets** beyond the
  pre-registered four. Reportable as context, not as the headline.
- **Plot styling, documentation prose, downstream report generation.**
- **Per-asset sensitivity sweeps for `WGeo-Hetero` thresholds** (a
  `ROADMAP.md` item) and other variants outside the locked headline
  specification.
- **Data fetching mechanics** — the cached parquet under `data/` is
  re-fetchable from `wbtc fetch`; what is locked is the data *source*
  (Binance via ccxt, daily close), not the cache.

## Procedure on 2027-06-01

1. Pull `master`. Verify the git tag `preregistration-v0.5` matches
   this document.
2. Refresh the parquet cache: `uv run wbtc fetch BTC/USDT`, repeated
   for the other three symbols. Verify timestamps cover 2026-06-01
   through 2027-05-31.
3. Run `uv run wbtc backtest-long` (which dispatches to
   `scripts/run_long_horizon.py`). This regenerates per-step CRPS arrays
   for all assets, all horizons, all methods including the headline.
4. Slice the per-step output to the pre-registered evaluation window
   (`ts ∈ [2026-06-01, 2027-05-31]`).
5. Compute the residualised DM panel under the `vol` control set, for
   `WGeo-Ensemble` vs `Static` and `WGeo-Ensemble` vs `GARCH-N`. Count
   cells with `p_r < 0.05` and lower-CRPS direction.
6. Apply the falsification check above. Write the outcome (pass /
   ambiguous / fail) to a new section in `docs/RESULTS_LONG.md` with
   the date, the cell counts, and the full per-cell DM table under all
   three control sets for transparency.
7. **No re-fit, no re-tune, no re-spec, no peeking.** If the result is
   surprising, the rule is to publish the surprise, not to fix it.

## Reporting

The evaluation outcome is reported as a new top section in
`docs/RESULTS_LONG.md` titled
`## v0.5 Pre-registered Out-of-Sample Result — 2027-06-01`,
containing:

- The cell counts (vol-only) against each baseline.
- The verdict: pass / ambiguous / fail.
- The per-cell DM table under all three control sets for transparency.
- The `garch_fallback` rate for `WGeo-Hetero` over the new window (if
  Hetero is reported as supplementary context).
- Any noted deviation from this document, with reason.

## Signatures

| Role | Name | Date |
|---|---|---|
| Author | Robin Bohrer | 2026-05-24 |
| Lock witness | (git tag `preregistration-v0.5` on merge commit) | TBD |
