# Roadmap

Concrete next steps for the project, in priority order. This is *not* a
"nice to have" backlog — it's the list of items that, if completed,
would move the project from "beats textbook baselines on daily crypto"
to "competitive against production risk systems."

The single most important thing this roadmap acknowledges: **we have
only benchmarked against textbook baselines (Static, RW-Drift,
HS-Bootstrap, GARCH-N, GARCH-t, GJR-GARCH-t).** That is enough to
publish a clean methods paper. It is not enough to claim the model
would compete with state-of-the-art realised-volatility or
multivariate models used in production at banks / serious quant
funds.

## Near-term (v0.4 — methods paper supplement)

These are pre-publication priorities. Each is small enough to be a
single PR and each strengthens the paper.

- [x] **Stronger baselines.** HAR-RV (Corsi 2009), FIGARCH(1,d,0)
  (Baillie-Bollerslev-Mikkelsen 1996), and a 2-state Markov-switching
  Normal (Hamilton 1989) — *plus* AR(1) Stochastic Volatility (Taylor
  1982 / Harvey-Ruiz-Shephard 1994) and a bivariate VAR+GARCH
  (BTC+ETH) — added to the comparator panel. See
  [`docs/RESULTS_EXTENDED.md`](docs/RESULTS_EXTENDED.md). Run via
  `uv run wbtc extended-baselines`.
- [x] **CAViaR (Engle-Manganelli 2004) as a quantile-regression
  baseline.** Implemented as `CAViaRSAV` (Symmetric Absolute Value)
  with an anchor-grid + interpolation fit for tractable per-step
  cost. Included in the extended baselines panel.
- [x] **`WGeo-Ensemble` — Wasserstein-2 barycentre of the v0.3 trio in
  quantile-function coordinates.** Theory says it must dominate the
  component average (Jensen on convex CRPS-in-CDF); empirically lifts
  vanilla DM from 1/12 to 4/12 cells with $p < 0.05$. Implemented as
  `WGeoEnsemble`, documented in `THEORY.md §2.9`.
- [x] **Residualised Diebold-Mariano (Giacomini-White augmented).**
  Variance-reducing covariate regression on the loss differential —
  preserves the unconditional EPA null, projects out shared volatility
  noise. Lifts the panel to 8/12 cells with $p_r < 0.05$. Implemented
  as `diebold_mariano_residualised`, documented in `THEORY.md §2.10`.
- [x] **`WGeo-CondShape` — long-window unconditional shape × GARCH-
  conditioned scale.** The right way to use a parametric vol forecast
  identified in §4.4: decouple shape and scale windows so the GARCH
  multiplier no longer double-counts. Implemented as
  `WassersteinGeodesicCondShape`; empirically not the headline winner
  but provides the v0.4 boundary statement.
- [ ] **Frozen final-year test set.** The walk-forward harness already
  does no look-ahead, but for publication we additionally reserve
  2026-01-01 → 2026-12-31 as a strict frozen holdout never touched by
  hyperparameter discussion. Re-run the v0.3 panel + new comparators on
  that frozen set.
- [ ] **XRP/USDT into the panel.** Currently cached but not in
  `METHODS`. Add to `SYMBOLS` in `scripts/run_long_horizon.py`. 5-asset
  panel is more convincing than 4 for cross-asset generalisation.
- [ ] **Per-asset sensitivity sweep for the `WGeoGarchEnsemble`
  thresholds** (ρ_lo, ρ_hi). Reported in the paper as a robustness
  appendix.

## Mid-term (v0.5 — production-relevance)

These move the project from "publishable" toward "an input a serious
risk team would actually use."

- [ ] **Realised-volatility features.** Add the (5-min) intraday RV as
  a regressor on the tangent slope. Expected meaningful CRPS gain,
  especially at h=21 where the v0.3 edge is weakest.
- [ ] **Intraday resolution.** Daily is too slow to be decision-relevant
  for most production use. Repeat the v0.3 panel at 4-hour and 1-hour
  resolutions with HAR-RV-style aggregation.
- [ ] **Multivariate extension.** BTC and ETH joint distribution. The
  1D-W₂-as-quantile isometry does *not* extend cleanly to higher
  dimensions (Brenier maps in 2D+ aren't quantile functions), so this
  is a real research problem — likely via Sliced-Wasserstein
  (Bonneel et al. 2015) or low-rank optimal-transport.
- [ ] **Conformal calibration layer.** Adding split-conformal on top of
  the WGeo forecast quantiles to guarantee marginal coverage under
  arbitrary deviations from i.i.d. The (Lei, Wasserman, etc.) literature
  has the machinery.
- [ ] **Live paper-trading on the next 6 months.** Forecasts written to
  a sealed log on day t, evaluated against the realised return on day
  t+h. Beating CRPS on past data is necessary but not sufficient.

## Long-term (v1.0 — ecosystem)

- [ ] **PyPI release.** Once v0.4 lands, publish to PyPI with a tagged
  release. `pip install wbtc`.
- [ ] **Read-the-docs / mkdocs site** for the public API.
- [ ] **Companion paper on arXiv.** The current `docs/RESEARCH_REPORT.md`
  is the seed; needs an introduction, a related-work section beyond the
  five references currently cited, and a multivariate extension to
  justify a journal submission.
- [ ] **External replication.** Find one external researcher with a
  different crypto-data source (e.g. Kraken instead of Binance) and
  have them reproduce the headline numbers. The MANIFEST.json
  provenance + the small surface area should make this tractable.

## What this roadmap will NOT pursue

- Deep-learning forecasters. Different paper, different project.
- A trading P&L headline. The framing is distributional forecasting.
- Becoming a "risk-management platform". This is a single forecaster,
  not a system. It is *an input* into risk management.

## How to propose adding to this roadmap

Open an issue tagged `roadmap`. Include:

1. Which v0.x bucket you think it belongs in.
2. The falsification criterion you'd add to `THEORY.md §4` if the
   feature were implemented.
3. The smallest experiment that would tell you the feature pays off
   (or doesn't).
