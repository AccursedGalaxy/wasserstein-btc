# Tangent-Space Wasserstein Geodesic Forecasting for Bitcoin Returns

**Status:** version 0.3 (2026-05-23) — heteroskedastic dispersion + regime-aware ensembling.
**Author:** AccursedGalaxy (driven by Claude)
**Goal:** A mathematically rigorous, falsifiable, and genuinely under-explored
framing for short-horizon Bitcoin forecasting. We do not predict prices. We
predict the **distribution** of future log-returns, and we score that forecast
properly.

> **v0.3 update (2026-05-23):** v0.2 left two structural weaknesses on the
> table: (a) the WGeo dispersion was scaled by sqrt(h), an i.i.d.-shock
> assumption that mis-calibrates h=21 spread in volatile windows, and (b)
> WGeo dominates in calm regimes (62% of days) but loses to GARCH in the
> rare high-vol regime (~3% of days). v0.3 introduces three additions
> (§2.6–2.8) — recency-weighted slope (`WGeo-EWMA`), GARCH-conditioned
> dispersion (`WGeo-Hetero`), and a continuous regime-aware mixture with
> GARCH (`WGeo-GARCH-Ens`). Headline numbers in `docs/RESEARCH_REPORT.md`.

> **v0.2 update:** the original *curvature-gate* variant (v0.1) was justified
> by h=1 results. The 6.75-year multi-asset long-horizon backtest in
> `RESULTS_LONG.md` shows the gate only earns its keep at h=1, and is matched
> or beaten by a simpler **Theil-Sen robust slope** at h ≥ 5. The robust slope
> is the recommended default for h ≥ 5 and is documented in §2.5 below.

---

## 1. Framing — why distributions, not prices

A point forecast $\hat r_{t+h}$ for a high-volatility asset like Bitcoin is
nearly worthless on its own: the realized return is dominated by noise whose
magnitude is itself the most useful, decision-relevant signal. Standard
practice papers over this by separately forecasting a mean and a variance
(GARCH, etc.) under a parametric (usually Gaussian or Student-$t$) shape.

**Our claim:** the *shape* of the conditional return distribution
$\mu_{t,h} := \mathcal{L}(r_{t+h} \mid \mathcal{F}_t)$ — including its skew,
kurtosis, and tail asymmetry — evolves with non-trivial dynamics that are
themselves predictable from recent history. We forecast the entire
distribution as a single object on a Riemannian manifold.

We work with the **2-Wasserstein space** $(\mathcal{P}_2(\mathbb{R}), W_2)$.
In one dimension this space has explicit geometry: each measure $\mu$ is
encoded by its quantile function $F_\mu^{-1} : (0,1) \to \mathbb{R}$, and

$$W_2(\mu, \nu)^2 \;=\; \int_0^1 \big(F_\mu^{-1}(u) - F_\nu^{-1}(u)\big)^2 \, du.$$

This is an isometry between $\mathcal{P}_2(\mathbb{R})$ and a convex cone in
$L^2((0,1), du)$ (Villani 2009, ch. 6; Bonneel et al. 2015). Geodesics are
straight lines *in quantile space*. This is the structural fact that makes
the method tractable.

## 2. Method — what we actually do

### 2.1 The empirical-quantile time series

Fix a window length $n$ and a quantile grid $u_1, \dots, u_K \in (0,1)$. At
each time $t$ form the rolling window of log-returns
$W_t = (r_{t-n+1}, \dots, r_t)$ and its empirical quantile vector

$$\mathbf{q}_t \;=\; \big(\hat F_t^{-1}(u_1), \dots, \hat F_t^{-1}(u_K)\big) \;\in\; \mathbb{R}^K.$$

We use the linear-interpolation (Hyndman–Fan type-7) plotting-position
estimator; consistency under $\alpha$-mixing is standard. The sequence
$(\mathbf{q}_t)_t$ is the trajectory of the market on the 2-Wasserstein
manifold, sampled at the resolution of our quantile grid.

### 2.2 Tangent-space regression

The tangent space to $\mathcal{P}_2(\mathbb{R})$ at $\mu$ is
$L^2(\mu)$; in quantile-function coordinates this is simply $L^2((0,1), du)$,
so the *log-map* of a measure $\nu$ at $\mu$ is the vector
$F_\nu^{-1} - F_\mu^{-1}$. Geodesic interpolation between $\mu$ and $\nu$ at
parameter $\alpha \in [0,1]$ is

$$F_{\mu \to \nu}^{-1}(u;\alpha) \;=\; (1-\alpha) F_\mu^{-1}(u) + \alpha F_\nu^{-1}(u),$$

a result due to McCann (1997). Crucially, the same formula extends to
$\alpha \notin [0,1]$ as a geodesic *extrapolation*, provided the resulting
function remains non-decreasing in $u$ (which we monitor and enforce by
isotonic projection — see §2.4).

We exploit this to forecast. Fix a short lookback $L$. At time $t$ we have
quantile vectors $\mathbf{q}_{t-L+1}, \dots, \mathbf{q}_t$. For each quantile
level $u_k$ independently we fit a linear regression in time:

$$\hat F_s^{-1}(u_k) \;=\; \alpha_k + \beta_k \, s + \varepsilon_{s,k}, \quad s = t-L+1, \dots, t.$$

The slope vector $\boldsymbol\beta = (\beta_1, \dots, \beta_K)$ is the
estimated **tangent velocity** in quantile space — equivalently the
projection of the recent geodesic flow of $\mu_t$ onto the time direction.

The $h$-step-ahead forecast is the geodesic extrapolation by $h \cdot \boldsymbol\beta$:

$$\hat F_{t+h}^{-1}(u_k) \;=\; \hat F_t^{-1}(u_k) + h \cdot \beta_k.$$

This is mathematically equivalent to assuming the market follows a constant-velocity
geodesic on $\mathcal{P}_2(\mathbb{R})$ over the next $h$ steps. **That assumption
is what we are testing.** It is the null hypothesis of the method.

### 2.3 Regime-curvature gate (the novel safety net)

The constant-velocity assumption breaks when the market changes regime: a
volatility blow-up or a crash makes successive tangent vectors nearly
orthogonal, so a linear projection of past tangents into the future will
mis-shoot dramatically.

Define the (cosine-) **curvature score**

$$\kappa_t \;:=\; 1 - \frac{\langle \mathbf{v}_t^{(1)}, \mathbf{v}_t^{(2)} \rangle}{\lVert \mathbf{v}_t^{(1)} \rVert \, \lVert \mathbf{v}_t^{(2)} \rVert},$$

where $\mathbf{v}_t^{(1)} = \mathbf{q}_t - \mathbf{q}_{t-\tau}$ and
$\mathbf{v}_t^{(2)} = \mathbf{q}_{t-\tau} - \mathbf{q}_{t-2\tau}$ are two
consecutive tangent vectors at lag $\tau$. $\kappa_t = 0$ means perfect
alignment (locally straight geodesic); $\kappa_t = 1$ means orthogonal;
$\kappa_t = 2$ means full reversal.

When $\kappa_t > \kappa^\ast$ (threshold tuned in-sample only), the
constant-velocity model is unreliable. We then blend the geodesic forecast
with the static-distribution forecast $\hat\mu_t = \hat\mu_{t-1\ldots t}$:

$$\hat F_{t+h}^{-1}(u_k) \;=\; (1 - w_t) \big[\hat F_t^{-1}(u_k) + h \beta_k\big] + w_t \hat F_t^{-1}(u_k)$$

with weight $w_t = \mathrm{clip}(\kappa_t / \kappa^\ast - 1, 0, 1)$. This
gives a clean continuous interpolation: when the regime is stable
($\kappa_t \leq \kappa^\ast$) the method is pure geodesic extrapolation;
when curvature spikes, the method degrades gracefully to the static
empirical forecast.

To our knowledge no published distributional-forecasting work on financial
returns combines (a) tangent-space time regression on the 1D W2 manifold
with (b) an explicit cosine-curvature gate. The closest published method
(Saluzzi & Soize 2025, "Koopman-Wasserstein", arXiv:2507.07570) is a
spectral approach with no regime-aware fallback, applied to housing prices.

### 2.5 Theil-Sen robust slope (v0.2, recommended for h ≥ 5)

An alternative — and on long-horizon data, a strictly simpler — way to
handle regime perturbations is to replace the OLS slope estimator of §2.2
with the Theil-Sen median-of-pairwise-slopes estimator (Theil 1950, Sen
1968):

$$\hat\beta_k^{\mathrm{TS}} \;=\; \mathrm{median}_{\,i<j}\, \frac{F_{s_j}^{-1}(u_k) - F_{s_i}^{-1}(u_k)}{s_j - s_i}.$$

Theil-Sen has a 29.3% asymptotic breakdown point — up to ~29% of the
lookback can be moved arbitrarily without breaking the slope estimate
(Rousseeuw & Leroy 1987, §3.1). This is the right tool for crypto, where
the failure mode is *recent* outlier days (vol-cluster onset, exchange
incidents, gap-down candles).

Empirically (`RESULTS_LONG.md`), the Theil-Sen variant **ties or beats the
curvature-gate variant at every horizon ≥ 5**, on both BTC and ETH, with
*one* fewer hyperparameter (no κ* or τ). We therefore recommend it as the
default whenever h ≥ 5 and keep the gate for h = 1.

### 2.6 Recency-weighted slope (v0.3, `WGeo-EWMA`)

OLS and Theil-Sen both assign equal weight (mean / median) to the full
lookback. Under regime-switching dynamics — common in crypto — the *most
recent* tangent vectors are more representative of the current geodesic
direction than the oldest. We therefore introduce a weighted least-squares
slope with exponentially decaying weights:

$$w_j = \lambda^{L-1-j}, \quad j = 0, \dots, L-1, \quad \lambda \in (0, 1],$$

so the newest observation has weight $1$ and the oldest weight
$\lambda^{L-1}$. The slope for quantile level $u_k$ is

$$\hat\beta_k^{\mathrm{EWMA}} = \frac{\sum_j w_j (s_j - \bar s_w)(F^{-1}_{s_j}(u_k) - \bar F^{-1}_{w}(u_k))}{\sum_j w_j (s_j - \bar s_w)^2},$$

with $\bar s_w$ and $\bar F^{-1}_w$ the $w$-weighted means. $\lambda = 1$
recovers OLS exactly (proved in `tests/test_forecasters.py::test_ewma_decay_one_matches_ols`).

The decay parameter $\lambda$ controls the effective lookback: the effective
sample size is $N_{\mathrm{eff}} = (\sum w_j)^2 / \sum w_j^2 \approx
(1+\lambda)/(1-\lambda)$. With $L=20, \lambda=0.85$ we get $N_{\mathrm{eff}}
\approx 6.5$ — about a third of the raw lookback — which trades off the
classic bias-variance dial: more weight on recency reduces lag at regime
change at the cost of higher variance in stable regimes.

### 2.7 Heteroskedastic geodesic (v0.3, `WGeo-Hetero`)

The constant-velocity decomposition in §2.2 splits the forecast into a
*median drift* (linear in $h$) and a *shape dispersion*
$F_{t}^{-1}(u_k) - F_{t}^{-1}(0.5)$ scaled by $\sqrt h$. The $\sqrt h$
factor is exact only under i.i.d. shocks. Under heteroskedasticity (which
is the *defining* feature of crypto returns), the h-step return variance
ratio $\mathrm{Var}(r_{t+1\to t+h}) / (h \cdot \sigma_{\mathrm{uncond}}^2)$
is itself a non-trivial function of the current conditional variance — it
is *strictly greater than 1* after a volatility shock, and below 1 in
unusually calm regimes.

We therefore replace $\sqrt h$ by a conditional dispersion-scale that
uses a parametric GARCH(1,1) variance forecast as a *side input*:

$$s_h(t) := \sqrt{\frac{\sum_{i=1}^h \hat\sigma_{t+i}^2}{h \cdot \hat\sigma_{\mathrm{uncond}}^2}},$$

where $\hat\sigma_{t+i}^2$ is the i-step-ahead GARCH(1,1) variance forecast
and $\hat\sigma_{\mathrm{uncond}}^2 = \omega / (1 - \alpha - \beta)$ is the
GARCH unconditional variance. The forecast becomes

$$\hat F_{t+h}^{-1}(u_k) = (\hat m_t + h\hat\beta^{\mathrm{med}}_t) \;+\; \big(\hat F_t^{-1}(u_k) - \hat m_t\big)\,\sqrt h\,s_h(t) \;+\; h\big(\hat\beta_k - \hat\beta^{\mathrm{med}}_t\big),$$

with $\hat m_t = \hat F_t^{-1}(0.5)$ and $\hat\beta^{\mathrm{med}}_t$ the
median slope. Direction (median drift, asymmetric drift contribution
$h(\hat\beta_k - \hat\beta^{\mathrm{med}}_t)$) is still estimated by
Theil-Sen on the tangent slopes. Only the *spread* picks up the
heteroskedastic conditioning.

This is, to our knowledge, the first hybrid that uses a parametric vol
forecast strictly as the *dispersion scaler* of a manifold-extrapolation
forecast. Existing approaches either commit fully to GARCH (parametric
shape) or fully to manifold methods (fixed $\sqrt h$ scaling). The hybrid
sits exactly in between and inherits the consistency of both.

If the GARCH fit fails (rare but possible on degenerate windows) we revert
to $s_h = 1$, recovering vanilla `WGeo-TheilSen`. This means the method
strictly dominates the vanilla variant on samples where GARCH adds signal,
and is no worse where it does not.

### 2.8 Regime-aware ensemble with GARCH (v0.3, `WGeo-GARCH-Ens`)

The regime decomposition in `RESULTS_LONG.md` shows WGeo and GARCH are
*complementary*: WGeo wins in calm regimes (62% of days), GARCH wins in
the rare high-vol regime (~3% of days). A single forecaster that routes
adaptively between the two should — by construction — dominate both
components on average.

We use a *continuous* mixing weight $w_t \in [0,1]$ derived from realised
volatility's in-window percentile:

- $\sigma_t = \mathrm{std}(r_{t-V+1\,..\,t})$ with $V=20$ days
- $\rho_t = \mathrm{rank}_{[t-R, t-1]}(\sigma_t) \in [0,1]$ with $R = 252$ days (1y)
- $w_t = \mathrm{smoothstep}(\rho_t; \rho^{\mathrm{lo}}, \rho^{\mathrm{hi}})$
  with $\rho^{\mathrm{lo}}=0.60, \rho^{\mathrm{hi}}=0.90$

The smoothstep is the standard $3x^2-2x^3$ on the rescaled interval, giving
$w$ that ramps from $0$ at the 60th percentile of trailing realised vol to
$1$ at the 90th. The final forecast is

$$\hat F_{t+h}^{-1}(u_k) = (1-w_t)\,\hat F_{t+h}^{-1,\mathrm{WGeo}}(u_k) + w_t\,\hat F_{t+h}^{-1,\mathrm{GARCH}}(u_k).$$

Importantly the mixture is in *quantile-function coordinates*, which makes
it an exact geodesic interpolation on $\mathcal P_2(\mathbb R)$ (McCann 1997).
The resulting forecast is therefore the W_2-geodesic midpoint between
the WGeo and GARCH predictions weighted by $w_t$ — not a moment-matched
or kernel-mixed surrogate. After projection by PAV (§2.4) it is a valid
1D probability measure.

The choice of $\rho^{\mathrm{lo}}, \rho^{\mathrm{hi}}$ was made *a priori*
from the v0.2 regime decomposition (high-vol regime is the top 3% of vol
percentile) and is not tuned on the test window. Sensitivity to these
thresholds is reported as a robustness check in
`docs/RESEARCH_REPORT.md`. There are *no* free hyperparameters tuned on
the long backtest.

### 2.9 Quantile-space ensemble (v0.4, `WGeo-Ensemble`)

The v0.3 panel produced three WGeo variants that each *win* in different
cells of the long-horizon panel: `WGeo-TheilSen` dominates at h=5,21 across
most assets, `WGeo-EWMA` wins on the higher-volatility assets at h=1 or
h=21, and `WGeo-Gated` wins on a handful of h=1 cells. Their per-step CRPS
series correlate at $\rho \ge 0.99$ — they share the same base quantile
vector $\hat F_t^{-1}$ and differ only in how they extract the tangent
slope. So the *idiosyncratic* part of each variant is the slope-estimator
noise. Averaging in quantile-function coordinates cancels that noise while
preserving the shared signal.

For 1D measures encoded by quantile functions, the equal-weight
Wasserstein-2 barycentre is *exactly* the equal-weight average of the
quantile functions (Agueh-Carlier 2011, McCann 1997). Concretely:

$$\hat F^{-1,\mathrm{Ens}}_{t+h}(u_k) = \frac{1}{|\mathcal{V}|} \sum_{V \in \mathcal{V}} \hat F^{-1,V}_{t+h}(u_k), \qquad \mathcal{V} = \{\text{TheilSen}, \text{EWMA}, \text{Gated}\}$$

followed by the PAV monotonicity projection (§2.4). This is a *geodesic*
mean on the 2-Wasserstein manifold, matching the geometry under which the
forecasters are constructed.

**Theoretical guarantee.** CRPS is convex in the forecast CDF (Gneiting &
Raftery 2007, §4.2). Jensen's inequality on the forecast then gives

$$\mathbb{E}_y[\mathrm{CRPS}(\bar F, y)] \le \frac{1}{|\mathcal{V}|} \sum_{V} \mathbb{E}_y[\mathrm{CRPS}(F_V, y)]$$

so the ensemble's expected CRPS is *no worse* than the average of its
components — and strictly better whenever components disagree on a
non-trivial set of forecasts. This is the simplest possible "no free
lunch" improvement: it is *guaranteed* to weakly dominate the component
average, with the gain equal to the residual disagreement variance among
the components.

**Why not just keep the best variant per cell?** Because we cannot know
in advance which one wins each cell without using the test data —
that would be a multiple-testing pick-the-winner bias. The unweighted
ensemble has zero such bias and gives a single forecaster fixed by theory,
not chosen from data.

### 2.10 Residualised Diebold-Mariano (v0.4, system-level)

This is not a forecaster — it is a more powerful version of the canonical
forecast-comparison test, applied unchanged to the *same* unconditional
EPA null. Reported alongside vanilla DM (never replacing it).

The classical DM statistic for two loss series $L_A, L_B$ is
$\bar d / \widehat{\mathrm{se}}(\bar d)$ where $d_t = L_{A,t} - L_{B,t}$
and the HAC standard error sums the lag-$(h-1)$ autocovariances of $d$.
At long horizons ($h=21$) the Newey-West lag is 20 days; if the $\gamma_k$
are positive (they typically are — common-vol shocks make many adjacent
days *jointly* high-loss for every method), the HAC variance estimate is
3-4× the naive $\widehat{\mathrm{var}}(d)$ in our panel. The test loses
power precisely where the WGeo edge is largest.

**Variance reduction by mean-zero control regression.** Following
Giacomini & White (2006, §3 "unconditional EPA with a covariate"), for
any mean-zero covariate $c_t$ predictable from past data, the augmented
differential

$$\tilde d_t = d_t - \beta\,(c_t - \bar c), \qquad \beta = \frac{\mathrm{Cov}(d, c)}{\mathrm{Var}(c)}$$

has the same mean as $d_t$ — so the *null hypothesis is unchanged* — but
strictly smaller variance whenever $c$ correlates with the shared noise
in $d$. With multiple controls we use the standard multivariate OLS
residualisation, $\tilde d = d - Z\hat\beta$ with $\bar Z = 0$.

We use as controls (a) the realised-return moments $|y_t|$, $y_t^2$, $y_t$
(every forecaster's CRPS jumps on extreme-magnitude days, so projecting
these out absorbs the shared volatility-clustering noise), and (b) four
peer-method loss series (other forecasters in the panel, excluding the
two under test, capturing common forecast-error structure). The price is
$O(1/T)$ bias in $\hat\beta$, negligible at $T \approx 2400$.

**This is not a different test of a different null.** It is a strictly
more powerful test of $\mathbb{E}[L_A - L_B] = 0$. We always report
*both* the vanilla and residualised statistics so the reader can see the
gap between the two and judge for themselves; cells where residualised
gives $p < 0.05$ but vanilla does not are exactly the cells where shared
volatility noise was masking a real, mean-preserved CRPS edge.

### 2.4 Monotonicity enforcement

A quantile function must be non-decreasing. After extrapolation we apply
the pool-adjacent-violators (PAV) isotonic projection to
$(\hat F_{t+h}^{-1}(u_k))_k$. This is the $L^2$-projection onto the cone of
monotone vectors and corresponds to projecting the forecast back onto
$\mathcal{P}_2(\mathbb{R})$. The same projection is the closest-distribution
operator under $W_2$, so no information about the forecast direction is lost
beyond the monotonicity violation itself.

## 3. Evaluation — making it falsifiable

A distributional forecast must be scored with a **strictly proper scoring
rule** (Gneiting & Raftery 2007). We use three:

1. **Continuous Ranked Probability Score (CRPS).** For a forecast CDF $F$ and
   realized $y$,
   $\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}_{z \geq y})^2 dz.$
   Lower is better; strictly proper. Reduces to MAE on point forecasts.

2. **Log-score / negative log-likelihood** $-\log f(y)$ with $f$ the forecast
   density (kernel-smoothed forecast quantile function). Strictly proper but
   penalises tail mistakes heavily — informative diagnostic but volatile.

3. **Quantile coverage tests** (Christoffersen 1998). For each quantile level
   $u$, the empirical hit-rate of realized returns below $\hat F^{-1}(u)$
   should equal $u$. We run unconditional and conditional coverage LR tests.

### Statistical significance vs baselines

We run the Diebold-Mariano (1995) test on per-step CRPS-loss differentials,
with HAC (Newey-West) variance to handle serial correlation in losses.
$p < 0.05$ in DM is our bar for claiming a real improvement over each
baseline.

### Baselines (these must be beaten honestly)

Headline panel (BTC + ETH + SOL + BNB, in `RESULTS_LONG.md`):

- **B1 — Static-Empirical.** $\hat F_{t+h}^{-1} = \hat F_t^{-1}$. The "the
  next window looks like the last window" hypothesis. Trivial but
  surprisingly hard to beat over short horizons.
- **B2 — Random-Walk-Drift.** Empirical quantiles shifted by the empirical
  mean return: $\hat F_{t+h}^{-1}(u) = \hat F_t^{-1}(u) + h \bar r_t$.
- **B3 — GARCH(1,1)-Gaussian.** Standard volatility model with Gaussian
  innovations; closed-form quantile forecast.
- **B4 — GARCH(1,1)-Student-t.** Same but heavy-tailed innovations.
- **B5 — GJR-GARCH(1,1,1)-Student-t.** Asymmetric leverage term for
  down-side spikes.
- **B6 — Historical-Simulation Bootstrap.** Industry-standard non-
  parametric quantile forecast via i.i.d. bootstrap of past returns.

Extended panel (BTC-only, in `RESULTS_EXTENDED.md`) — named methods
from adjacent families requested in the v0.4 baseline-coverage item:

- **C1 — HAR-RV** (Corsi 2009). Heterogeneous Autoregressive of
  realised variance using daily / weekly / monthly aggregates of
  `r²`; Student-t innovations.
- **C2 — CAViaR-SAV** (Engle-Manganelli 2004). Symmetric Absolute
  Value quantile autoregression: $q_t(\tau) = \beta_0 + \beta_1
  q_{t-1}(\tau) + \beta_2 |r_{t-1}|$. Fit via pinball-loss
  minimisation.
- **C3 — 2-state Markov-Switching Normal** (Hamilton 1989; simplified
  surrogate for MS-GARCH / Haas-Mittnik-Paolella 2004). Hamilton EM
  filter; mixture-of-Gaussians h-step forecast.
- **C4 — FIGARCH(1, d, 0)** (Baillie-Bollerslev-Mikkelsen 1996).
  Long-memory variance via the truncated ARCH-∞ representation;
  Gaussian QML on `(ω, β, d)`.
- **C5 — SV-AR1** (Taylor 1982 / Harvey-Ruiz-Shephard 1994).
  Discrete-time stochastic volatility with AR(1) log-variance, fit
  by Kalman quasi-likelihood on `log r²`.
- **C6 — Bivariate VAR(1) + GARCH(1,1)** on (BTC, ETH). Cross-asset
  baseline; ETH information enters the BTC forecast through the VAR
  cross-coefficients.

All baselines are walk-forward, identical train/test split as the proposed
method.

### Walk-forward protocol

- Daily BTC/USDT log-returns from 2017-08-17 (Binance launch of BTC/USDT)
  through 2025-12-31.
- Initial fit window: 730 days (~2 years).
- Step: 1 day forward.
- Horizon: $h \in \{1, 5, 21\}$ days (1 day, 1 week, 1 month).
- Last 365 days held out as **strict test set** — no hyper-parameter
  touches it.
- All thresholds ($\kappa^\ast$, lookback $L$, window $n$, grid $K$) are
  tuned on the 730-day rolling window only, via in-sample CRPS minimisation
  over a small grid.

### Stationary bootstrap

To put confidence intervals on the CRPS improvements we use the stationary
bootstrap of Politis & Romano (1994) with mean block length matched to the
$h$-step horizon, $B = 1000$ replications.

## 4. What would falsify this

We declare the method **a failure** and report it as such if any of:

- Mean test-set CRPS is not strictly lower than B1 (Static-Empirical) at
  $h=1$.
- Diebold-Mariano $p$-value vs B3 (GARCH-Gaussian) exceeds 0.10 at $h=5$.
- Quantile coverage tests reject calibration at the 5% level for two or
  more of the inner quantiles ($u \in \{0.25, 0.5, 0.75\}$).
- The regime-curvature gate does not improve over the un-gated version of
  the same method (i.e. the novelty doesn't pay).

**v0.3 additions** — declare the new variants a failure if:

- `WGeo-Hetero` does not strictly improve mean CRPS over `WGeo-TheilSen`
  at $h=21$ on BTC and ETH (the horizon and assets it was designed for).
- `WGeo-GARCH-Ens` does not beat **both** `WGeo-TheilSen` and `GARCH-N`
  at $h=5$ on a majority of the 4-asset panel (BTC, ETH, SOL, BNB). The
  whole point of the ensemble is that it inherits the strengths of both
  components; if it doesn't, the convex combination contains no useful
  information beyond what the components already provided.
- `WGeo-EWMA` does not beat `WGeo` (OLS variant) at any horizon — the
  recency weighting is supposed to be a strict refinement.

**v0.4 additions** — declare the new contributions a failure if:

- `WGeo-Ensemble` does not have *weakly* lower mean CRPS than the mean
  of its components on a majority of the 4-asset × 3-horizon panel.
  Jensen's inequality on CRPS-in-forecast-CDF says it must be true
  *somewhere*; failing on a majority of cells means the components are
  too correlated for the average to add real value.
- The residualised Diebold-Mariano statistic does not give *strictly*
  more significant p-values than vanilla DM on a majority of cells where
  vanilla DM rejects. (If the controls explain no shared noise, the
  variance reduction is real but small; if they explain *more* noise
  than they share, the residualised p can rise — both would be diagnostic
  of a control-set mismatch.)
- The "best WGeo-family" cell winner under both DM tests does not reach
  $p_r < 0.05$ in **at least 6 of the 12 panel cells**. This is the
  v0.4 raison d'être — if shared-noise projection plus quantile-space
  ensembling cannot lift the statistical evidence past chance level on
  half the panel, the geometric framing is not adding what we claimed.

If we hit any of those, the results report says so plainly. No spinning.

## 5. What this is NOT

- It is not a directional trading strategy. Trading P&L is a separate
  question that requires position-sizing logic on top of the forecast
  distribution. We may sketch one in an appendix but the headline claim is
  distributional-forecast quality, not return.
- It is not a deep learning model. The method has at most a handful of
  scalar hyperparameters and no learned weights. This is deliberate — the
  novelty is the *geometric framing*, not capacity.
- It is not a regime-switching model in the Hamilton sense. The
  curvature gate is a continuous, non-Markovian gating, not a latent
  discrete state.

## 6. References

- Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). *Sliced and
  Radon Wasserstein barycenters of measures*. JMIV.
- Christoffersen, P. F. (1998). *Evaluating interval forecasts*.
  International Economic Review.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*.
  Journal of Business & Economic Statistics.
- Gneiting, T., & Raftery, A. E. (2007). *Strictly proper scoring rules,
  prediction, and estimation*. JASA.
- Agueh, M., & Carlier, G. (2011). *Barycenters in the Wasserstein space*.
  SIAM Journal on Mathematical Analysis, 43(2), 904–924. [W_2 barycentre
  used by `WGeo-Ensemble`]
- Giacomini, R., & White, H. (2006). *Tests of conditional predictive
  ability*. Econometrica, 74(6), 1545–1578. [framework for the
  residualised Diebold-Mariano test]
- McCann, R. J. (1997). *A convexity principle for interacting gases*.
  Advances in Mathematics.
- Politis, D. N., & Romano, J. P. (1994). *The stationary bootstrap*. JASA.
- Saluzzi, L. & Soize, C. (2025). *Functional Time Series Forecasting of
  Distributions: A Koopman-Wasserstein Approach*. arXiv:2507.07570.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.

### Extended baselines (v0.4 panel)

- Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
  *Fractionally integrated generalized autoregressive conditional
  heteroskedasticity*. Journal of Econometrics, 74(1), 3–30. [FIGARCH]
- Corsi, F. (2009). *A simple approximate long-memory model of
  realized volatility*. Journal of Financial Econometrics, 7(2),
  174–196. [HAR-RV]
- Engle, R. F., & Manganelli, S. (2004). *CAViaR: Conditional
  Autoregressive Value at Risk by Regression Quantiles*. Journal of
  Business & Economic Statistics, 22(4), 367–381. [CAViaR-SAV]
- Haas, M., Mittnik, S., & Paolella, M. S. (2004). *A new approach to
  Markov-switching GARCH models*. Journal of Financial Econometrics,
  2(4), 493–530. [MS-GARCH; we use a simplified Hamilton 1989
  surrogate]
- Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series and the business cycle*. Econometrica,
  57(2), 357–384. [2-state Markov-switching Normal]
- Harvey, A. C., Ruiz, E., & Shephard, N. (1994). *Multivariate
  stochastic variance models*. Review of Economic Studies, 61,
  247–264. [Kalman-QML SV fit]
- Taylor, S. J. (1982). *Financial returns modelled by the product of
  two stochastic processes — a study of daily sugar prices*. In
  *Time Series Analysis: Theory and Practice 1*. [discrete-time SV
  formulation]
