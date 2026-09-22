# Credibility failures of the daily rolling-window track — acknowledgement

_Written 2026-09-22, as required by `docs/PREREG.md` v1.0 §"If Gate 1
fails". Gate 1 failed on that date; `docs/RESULTS_LONG.md` is therefore
the paper headline, and this note records what a reader of that document
must know before trusting it._

`docs/PREREG.md` listed four credibility failures of the v0.3–v0.5 results
when it was locked on 2026-05-24. Status of each:

## 1. Duplicate baselines — fixed

`StaticEmpirical` and `RandomWalkDrift` were byte-identical, so every
"beats Static *and* RW-Drift" cell was one comparison counted twice.
Fixed on 2026-09-22: `RW-Drift` removed from every panel and baseline
list; the class survives only as a `DeprecationWarning` alias with a
regression test asserting identical output. `RESULTS_LONG.md` carries an
erratum until its tables are regenerated. The pre-registered v0.5
headline (vs `Static`, vs `GARCH-N`) never used RW-Drift and is unaffected.

## 2. Missing real benchmark — open

Wasserstein Autoregression (Zhang, Kokoszka, Petersen 2022, *J. Time
Series Analysis*, arXiv:2006.12640) is the closest published method and is
neither implemented, benchmarked, nor cited. The WGeo family is a less
elaborated cousin of WAR. Any claim that WGeo is competitive with the
literature is unsupported until WAR is in the panel on the same
walk-forward harness. This is now the first item a paper submission
would need.

## 3. Manifold framing is computationally inert in 1D — acknowledged

McCann (1997) makes 1D W₂ geodesics linear interpolations of quantile
functions, so the geometric description is correct but does no work: the
code is per-quantile linear extrapolation with √h spread scaling. The
framing should be presented as a *coordinate choice* that makes the
extrapolation well-posed (monotone output, isotonic projection), not as a
source of modelling power. Gate 1's failure reinforces this: the one
statistic that tried to use the manifold structure (κ curvature) turned
out uninformative on the data object where the WAR benchmark would have
been valid.

## 4. WGeo-Hetero double-counts volatility — acknowledged, not headline

The recent-window quantile width and the GARCH conditional/unconditional
ratio share a window, so the "regime-aware spread" multiplier inflates an
already-elevated shape. `WassersteinGeodesicCondShape` decouples the two
windows but is not in the headline. `WGeo-Hetero` is a component of
nothing pre-registered: the v0.5 headline `WGeo-Ensemble` is the
barycentre of TheilSen, EWMA and Gated, none of which use the GARCH
multiplier. The VaR/ES panel (`RESULTS_VAR_ES.md`) shows `WGeo-Hetero`
failing Kupiec in 20/20 cells, which is the empirical face of this defect.

## What the headline can still claim

- A small (1–5%), consistent CRPS improvement over `Static` and `GARCH-N`
  on five assets and three horizons, with pre-registered tests and a
  12-month frozen holdout (2026-06-01 → 2027-05-31) still to be scored.
- Nothing about competitiveness with WAR or with production risk systems.
- Nothing about tail calibration for the ensemble: `WGeo-Ensemble` passes
  Kupiec in 14/20 cells but Acerbi-Szekely Z1 in 0/20.
