# Tangent-Space Wasserstein Geodesic Forecasting for Bitcoin Returns

**Status:** version 0.2 (2026-05-23) — revised after long-horizon evidence.
**Author:** AccursedGalaxy (driven by Claude)
**Goal:** A mathematically rigorous, falsifiable, and genuinely under-explored
framing for short-horizon Bitcoin forecasting. We do not predict prices. We
predict the **distribution** of future log-returns, and we score that forecast
properly.

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

- **B1 — Static-Empirical.** $\hat F_{t+h}^{-1} = \hat F_t^{-1}$. The "the
  next window looks like the last window" hypothesis. Trivial but
  surprisingly hard to beat over short horizons.
- **B2 — Random-Walk-Drift.** Empirical quantiles shifted by the empirical
  mean return: $\hat F_{t+h}^{-1}(u) = \hat F_t^{-1}(u) + h \bar r_t$.
- **B3 — GARCH(1,1)-Gaussian.** Standard volatility model with Gaussian
  innovations; closed-form quantile forecast.
- **B4 — GARCH(1,1)-Student-t.** Same but heavy-tailed innovations.

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
- McCann, R. J. (1997). *A convexity principle for interacting gases*.
  Advances in Mathematics.
- Politis, D. N., & Romano, J. P. (1994). *The stationary bootstrap*. JASA.
- Saluzzi, L. & Soize, C. (2025). *Functional Time Series Forecasting of
  Distributions: A Koopman-Wasserstein Approach*. arXiv:2507.07570.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
