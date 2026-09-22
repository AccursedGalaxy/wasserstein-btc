# Pre-registration v1.0 — Intraday-density track kill conditions

**Status.** Locked. Tag: `prereg-v1.0`. Commit: see `git rev-parse prereg-v1.0`.
Amended v1.1 on 2026-09-22 (density-construction lock; see the amendment section before the signatories).

**Outcome.** **Gate 1 FAILED on 2026-09-22.** The intraday-density track is
abandoned per §"If Gate 1 fails". See the *Gate 1 outcome* section at the
end of this document. No Gate 2 or Gate 3 will be run.

**Lock date.** 2026-05-24

**Author.** AccursedGalaxy (Robin Bohrer)

**Scope.** This document freezes three numerical kill conditions that govern
the go/no-go decisions for a new experimental track on intraday-density
forecasting. The track itself (`src/wbtc/intraday/`) does not yet exist and
will only be built if the gates below pass in sequence. The gates are
ordered cheap-first so that the project can be killed in days rather than
months if the headline thesis does not survive the new data object. Edits
to this file after the lock require a v1.1 with diff and dated rationale.

## Why this exists

The previous headline track (`docs/RESULTS_LONG.md`, 90-day rolling-window
densities scored under CRPS) has four documented credibility failures:

1. **Duplicate baselines.** `StaticEmpirical` and `RandomWalkDrift` are
   byte-identical in `src/wbtc/forecasters.py` lines 113-121 and 133-137.
   Every "beats Static AND RW-Drift" cell in `RESULTS_LONG.md` is one
   comparison double-counted.
2. **Missing real benchmark.** The closest published competitor —
   Wasserstein Autoregression (Zhang, Kokoszka, Petersen 2022, *Journal of
   Time Series Analysis*, arXiv:2006.12640) — is not implemented, not
   benchmarked against, and not cited. The codebase's "WGeo" is a less
   elaborated cousin of WAR.
3. **Manifold framing is inert in 1D.** The geometric description is
   accurate (McCann 1997 makes 1D W_2 geodesics linear interpolations of
   quantile functions) but does no computational work; the actual code is
   per-quantile OLS extrapolation with √h spread scaling.
4. **WGeo-Hetero double-counts vol.** Recent-window quantile width and a
   GARCH conditional/unconditional ratio share the same window, so the
   "regime-aware spread" multiplier inflates an already-elevated shape.
   The fix exists as `WassersteinGeodesicCondShape` but is not in the
   headline panel.

The reformulated thesis is sharper and falsifiable:

> WAR's IID-innovation assumption (their assumption A2, §2) breaks at
> distributional regime changes. A cosine-curvature statistic κ on
> consecutive tangent vectors detects the breakdown out-of-sample; gating
> to a trend-extrapolation alternative recovers loss at break days where
> WAR fails. The WAR paper's own §6 names "stationarity testing and
> change-point detection for density time series" as open work.

The decision to test this thesis on intraday-density time series (one
density per day, built from ~288 5-min returns) — protocol (i) — is
correct for benchmark validity: successive intraday densities are
approximately independent draws, which is the data object WAR's theorems
require. But the same property that rescues the benchmark may dissolve
the headline statistic: κ assumes a smooth trajectory whose curvature
can be measured, and independent draws have no trajectory. The three
gates below are the pre-commitment that this risk is being tested
honestly, before any code is written.

## Gate 1 — Intraday-κ has trajectory structure

**Run when.** Immediately after the BTC 5-min density build. Before any
forecaster is implemented.

**Script.** `scripts/gate_1_intraday_kappa.py`, frozen at this commit. No
CLI flags. Reads `data/btcusdt_intraday_density.parquet`. Writes
`results/gate_1_intraday_kappa.json` and
`results/gate_1_intraday_kappa.png`. Process exit code: 0 = PASS, 1 =
FAIL, 2 = setup error (data missing or malformed).

**Decision rule.** PASS iff 1A-shuffle AND 1A-block AND 1B-events all pass.

### 1A — Phase-shuffle null (both surrogate types must pass)

Compute κ_t = 1 − cos(v₁, v₂) where v₁ = Q[t] − Q[t-τ], v₂ = Q[t-τ] −
Q[t-2τ], with τ=2, on the real ordered sequence of BTC intraday
density-quantile vectors over the early epoch (2017-08-17 through
2023-12-31, inclusive).

Generate N=200 surrogate sequences under each of two nulls:

- **Day-shuffle:** uniform random permutation of day order (preserves the
  marginal distribution of densities, destroys the trajectory). Seeds:
  0..199.
- **Stationary block-bootstrap** (Politis-Romano 1994): geometric block
  lengths with mean L=10 days, circular wrap. Preserves week-scale local
  structure, destroys month-scale trajectory. This is the *harder* null
  because it tolerates some short-range autocorrelation. Seeds: 1000..1199.

For each surrogate, compute the 99th percentile of its κ series.

**Kill condition.** The 99th percentile of real-κ must exceed the median
(across N=200 surrogates) of surrogate-κ 99th percentiles by factor
≥1.25. Tested independently against shuffle and block-bootstrap. Both
must pass.

### 1B — Event-day elevation

For each event in {2020-03-12, 2021-05-19, 2022-11-08} (COVID crash,
China-ban / Tether stress, FTX collapse start), examine κ_t for t in
{event_day, event_day+1, event_day+2}.

A "hit" = at least one κ value in the 3-day window exceeds the empirical
95th percentile of κ over the full early epoch.

**Kill condition.** ≥2 of 3 events must register a hit.

### If Gate 1 fails

The intraday-density track is abandoned. `docs/RESULTS_LONG.md` (the
daily-rolling-window track) becomes the paper headline, with the four
credibility failures listed above acknowledged in writing in
`docs/archive/`. No further intraday work.

## Gate 2 — KL has discriminating headroom

**Run when.** Gate 1 PASS, plus WAR(1) implementation only (the cheapest
forecaster). Before any other forecaster is built.

**Procedure.** Over the early epoch, compute mean KL divergence of
WAR(1) one-step-ahead forecast against realised intraday density.
Compute the irreducible-noise floor by resampling: at each day t, draw a
synthetic resample of that day's 5-min returns of equal size, build its
KDE, score the original against the resample. Mean over the early epoch
is the floor.

**Kill condition.** If `mean(KL_WAR) / noise_floor < 1.5`, KL cannot
discriminate methods (WAR is within 50% of the noise the metric cannot
see past). Switch primary metric to W₂ before any other forecaster is
built; rerun the headroom test on W₂ with the same threshold. If
`mean(W₂_WAR) / W₂_noise_floor < 1.5` also fails, abandon protocol (i)
and fall back to the daily-rolling-window track.

## Gate 3 — Diagnostic test has power

**Run when.** Gate 1 PASS + Gate 2 PASS + WAR(1) + WGeo-trend
implementations only. Before pre-reg v2.0 (analysis lock) is written.

**Procedure.** Set κ\* = 80th percentile of κ over the early epoch.
Count regime-B days (κ_t > κ\*) in the candidate holdout window per
symbol. Simulate per-symbol Diebold-Mariano test under H₁: Cohen's d=0.3
for (WAR_loss − WGeo-trend_loss) on regime-B days. (d=0.3 is the cautious
"small-to-medium" effect size; if the true effect is larger the test
sees it more easily, but planning for d=0.3 forces honest confrontation
with under-power.) Compute probability that the headline rule "≥3/5
symbols reach FDR-BH p<0.05" fires.

**Kill condition.** If power < 0.60 under d=0.3, revise *exactly one* of:

- holdout boundary (candidate: move from 2024-01-01 to 2022-06-30,
  putting FTX/Terra/SVB in the no-touch window);
- panel size (candidate: add MATIC/AVAX/LINK to push to 8 symbols);
- decision rule (candidate: drop FDR-BH from per-symbol to pooled).

Pick one. Document why in writing in this file (as a v1.1 amendment).
Rerun. If still <0.60 after one revision, the diagnostic test is
structurally underpowered: the paper's headline becomes Gate 1's
*detection* claim only (κ predicts breakdown), and the *diagnostic* claim
(κ discriminates which model wins) is reported as inconclusive rather
than retro-revised to fit available power.

## Missing-forecast policy

Methods in the eventual panel (FFWAR FPCA, LQDT inversion, CoDa) will
occasionally fail on numerically pathological days. The loss-assignment
policy is frozen here to remove a researcher degree of freedom:

- Headline DM is run **twice** — once excluding failed days from all
  methods (intersect; the safe default), once scoring failures at
  panel-worst loss for that day (union; the "no free lunch for breaking"
  rule). Both go in the table.
- If the two diverge in their PASS/FAIL verdict on any cell, the
  divergence is reported as the headline finding for that cell, not
  resolved by picking the more favourable.
- Per-method per-regime failure rates are reported as a results table,
  not a footnote. If a competitor fails disproportionately at regime-B
  days (κ > κ\*), that is itself a finding ("the competitor cannot handle
  regime change"), not a nuisance to paper over.

## Things this pre-reg does NOT freeze

The following remain open and will be locked in pre-reg v1.1, *only
after* Gate 1 passes and the intraday-density-construction code exists:

- KDE bandwidth (current candidate: Silverman)
- Common-support grid bounds (current candidate: [-0.20, 0.20])
- Quantile grid K (current candidate: K=100, matching WAR §5.2)
- Data-cleaning policy specifics (exclusion thresholds, depeg-day
  calendar)
- Holdout boundary (current candidate: 2024-01-01; Gate 3 may revise to
  2022-06-30)

The full analysis lock — every metric, control, threshold, stratification,
and the byte-deterministic analysis script that produces the final
tables — is **pre-reg v2.0**, written *immediately before* the holdout
is touched. v1.0 (this file) locks only the kill conditions and the
methodology by which v2.0 is reached.

## Deprecations triggered by this lock

- `docs/RESULTS_EXTENDED.md` and `docs/RESULTS.md` (the v0.4 extended
  econometric panel and the v0.1 365-day report) are moved to
  `docs/archive/` with one-line headers acknowledging the duplicate-
  baseline bug and noting they are retained for provenance, not as
  load-bearing claims. Move executed in a follow-up commit so this lock
  commit stays minimal.
- `docs/RESULTS_LONG.md` and `docs/RESULTS_VAR_ES.md` remain operational
  as the appendix-track headline if Gate 1 fails, or as supporting
  evidence on the daily-rolling-window object if Gate 1 passes.

## Amendment v1.1 — 2026-09-22 — intraday-density construction lock

**Scope.** Freezes the "not frozen in v1.0" construction choices for the
BTC intraday-density data object, *before* Gate 1 is run. Kill conditions
and gate constants from v1.0 are untouched. Values are the v1.0 "current
candidate" values wherever one was named; the two v1.0 left open entirely
(cleaning thresholds, depeg calendar) are set here for the first time.

| Choice | v1.0 status | v1.1 value | Where |
|---|---|---|---|
| Kernel / bandwidth | candidate: Silverman | Gaussian kernel, Silverman `0.9·min(sd, IQR/1.34)·n^(-1/5)`; floor 1e-6 | `wbtc.density.silverman_bandwidth` |
| Common support | candidate: [-0.20, 0.20] | [-0.20, 0.20]; KDE CDF renormalised to the support | `wbtc.density.SUPPORT_LO/HI` |
| Quantile grid K | candidate: 100 | K = 100 at `u_k = (k-0.5)/K` | `wbtc.density.K_QUANTILES` |
| CDF inversion grid | open | 8001 points (step 5e-5), linear interpolation | `wbtc.density.GRID_POINTS` |
| Return definition | open | within-UTC-day log-close differences of consecutive 5-min candles; no overnight gap | `wbtc.density.intraday_log_returns` |
| Short-day exclusion | open | `is_excluded` if fewer than 230 returns (80% of 287) | `wbtc.density.MIN_OBS_PER_DAY` |
| Depeg / calendar exclusion | open | empty for BTC/USDT | `scripts/build_intraday_density.py::EXCLUDE_DAYS` |
| Partial current day | open | dropped at build time | `scripts/build_intraday_density.py` |
| Source data | open | Binance BTC/USDT 5m via ccxt (`wbtc fetch --timeframe 5m BTC/USDT`) | `scripts/fetch_data.py` |

**Rationale for the two new choices.** 80% coverage keeps exchange-outage
days (Binance had several multi-hour halts in 2018–2021) out of the
trajectory without discarding days that lost a handful of candles; a
stricter rule would punch more holes into the ordered sequence κ is
computed on, which is the quantity Gate 1 measures. The empty calendar
records that no BTC-specific event exclusion was applied; the three Gate 1
event days are *in* the sample by construction.

**Provenance.** `wbtc build-density` writes
`results/intraday_density_manifest.json` (source SHA-256, output SHA-256,
day count, excluded days with reasons, spec dict). Gate 1's JSON output
records the input SHA-256; the two must match for a reported Gate 1 result
to count.

Holdout boundary remains open (candidate 2024-01-01; Gate 3 may revise).

## Gate 1 outcome — 2026-09-22 — FAIL

Run with `wbtc gate-1` on the v1.1 data object (`results/intraday_density_manifest.json`,
output SHA-256 `28b1db9c…cbdc3`, matched by the gate JSON). 2318 usable
early-epoch days (10 short days excluded, all 2017–2020 exchange halts).

| Sub-test | Real | Surrogate median | Ratio / hits | Threshold | Verdict |
|---|---|---|---|---|---|
| 1A day-shuffle | κ p99 = 1.994 | 1.999 | 0.997 | ≥ 1.25 | FAIL |
| 1A block-bootstrap (L=10) | κ p99 = 1.994 | 1.995 | 0.999 | ≥ 1.25 | FAIL |
| 1B event elevation (p95 = 1.979) | 2020-03-12: 1.967 · 2021-05-19: 1.994 · 2022-11-08: 1.861 | — | 1 / 3 | ≥ 2 / 3 | FAIL |

**Reading.** All three sub-tests fail by a wide margin, and they fail in the
way the v1.0 text anticipated ("independent draws have no trajectory").
For a sequence of exchangeable quantile vectors, the two tangent vectors
`v₁ = Q[t] − Q[t−τ]` and `v₂ = Q[t−τ] − Q[t−2τ]` share the `−Q[t−τ]` term,
so their cosine is centred near −0.5 and κ near 1.5 with a hard ceiling
at 2. The real series has median κ = 1.58 and a p99 of 1.994; every
surrogate lands at the same ceiling. κ does not measure curvature of a
trajectory here because there is no trajectory to measure: consecutive
intraday densities are close to independent draws, which is exactly the
property that made them the right data object for benchmarking against
Wasserstein Autoregression.

The data object itself is sound: the three event days sit at the 99.5th,
99.8th and 95.2nd percentiles of intraday 5–95 quantile width. The
densities saw the crises; the curvature statistic could not.

**Consequence (per v1.0 §"If Gate 1 fails").** The intraday-density track
is abandoned. `docs/RESULTS_LONG.md` (daily rolling-window track) is the
paper headline. The four credibility failures listed in *Why this exists*
are acknowledged in `docs/archive/CREDIBILITY_FAILURES.md`. No further
intraday work. The density module (`wbtc.density`) and the 5-min fetch
stay in the tree because they are the reproducibility path for this
result, not because the track continues.

**Not pursued, deliberately.** Rescuing κ by redefining it (e.g. using
non-overlapping tangents, smoothing Q over a window first, or replacing
cosine with a W₂ second difference) would be a post-hoc revision of a
pre-registered kill condition after seeing the data. Any such statistic
belongs in a new pre-registration with its own nulls, not in an amendment
to this one.

## Lock signatories

- Author: AccursedGalaxy (Robin Bohrer)
- Tag: `prereg-v1.0`
- Commit hash: resolvable via `git rev-parse prereg-v1.0`
- Locked: 2026-05-24
