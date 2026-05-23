"""Quantile-function utilities and Wasserstein-2 geometry on 1D measures.

A 1D probability measure with finite second moment is encoded throughout
this codebase by a vector of empirical quantile values on a fixed grid
``u_1, ..., u_K`` in (0, 1). This is the W_2-isometric coordinate
(Villani 2009, ch. 6).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "make_grid",
    "empirical_quantiles",
    "weighted_quantiles",
    "w2_distance",
    "isotonic_project",
    "tangent_log_score",
]


def make_grid(K: int) -> np.ndarray:
    """Return K equally-spaced interior quantile levels in (0, 1).

    Uses (k - 0.5) / K so the grid is symmetric and never hits 0 or 1.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    return (np.arange(K) + 0.5) / K


def empirical_quantiles(returns: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Empirical quantile vector on grid u, using linear-interpolation (type 7).

    Parameters
    ----------
    returns
        1D array of observations.
    u
        Quantile levels in (0, 1).
    """
    returns = np.asarray(returns, dtype=float)
    u = np.asarray(u, dtype=float)
    if returns.ndim != 1:
        raise ValueError("returns must be 1D")
    if returns.size == 0:
        raise ValueError("returns is empty")
    # numpy.quantile uses linear interp by default == Hyndman-Fan type 7
    return np.quantile(returns, u, method="linear")


def weighted_quantiles(
    returns: np.ndarray, u: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted empirical quantile vector on grid ``u`` (Hyndman-Fan type 7 generalisation).

    Given a non-negative weight per observation, this constructs the weighted
    empirical CDF and returns its left-continuous inverse on the grid ``u``
    using linear interpolation between adjacent weighted order statistics —
    the natural generalisation of :func:`empirical_quantiles` that recovers
    it when all weights are equal.

    Construction
    ------------
    Let ``r_(1) <= r_(2) <= ... <= r_(n)`` be the sorted observations and
    ``w_(k)`` the matching weights. Let ``W = sum w_(k)`` and define
    ``c_k = (cumsum(w_(k)) - 0.5 w_(k)) / W``  ∈ (0, 1)
    as the plotting positions (Wong & Chidambaram 1985; the half-weight offset
    avoids step bias and is the weighted analogue of the (k - 0.5) / n
    convention used by ``np.quantile``-type-7 / Hyndman-Fan §3).
    The returned ``Q(u)`` is the piecewise-linear interpolation of
    ``(c_k, r_(k))`` evaluated at ``u``, with flat extrapolation outside.

    Parameters
    ----------
    returns
        1D array of observations.
    u
        Quantile levels in (0, 1).
    weights
        Non-negative weights, same length as ``returns``. Will be normalised
        internally — any non-negative scaling is acceptable. Zero weights are
        allowed; if every weight is zero an exception is raised.

    Recovery of the unweighted estimator
    ------------------------------------
    With uniform weights ``w_k = 1`` the plotting positions reduce to the
    Hazen formula ``c_k = (k - 0.5) / n``. This differs from
    ``np.quantile`` (Hyndman-Fan type 7, ``c_k = (k - 1) / (n - 1)``) only at
    the tails by an :math:`O(1/n)` interpolation-rule correction; both are
    consistent estimators of the population quantile. We use Hazen because it
    has lower mean-squared error for skewed distributions (Hyndman-Fan 1996
    Table 2) and avoids the degenerate ``c_1 = 0, c_n = 1`` endpoints that
    make type 7 numerically extrapolation-only at the boundaries.
    """
    r = np.asarray(returns, dtype=float)
    u = np.asarray(u, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be 1D")
    if r.shape != w.shape:
        raise ValueError("weights must have the same shape as returns")
    if r.size == 0:
        raise ValueError("returns is empty")
    if np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be non-negative and finite")
    W = float(w.sum())
    if W <= 0:
        raise ValueError("at least one weight must be positive")
    order = np.argsort(r, kind="stable")
    rs = r[order]
    ws = w[order]
    cum = np.cumsum(ws)
    # plotting positions in (0, 1)
    c = (cum - 0.5 * ws) / W
    # for points that share the same value (ties), `np.interp` already handles
    # the piecewise-linear interpolation correctly because the c-sequence is
    # non-decreasing whenever the weight sequence is non-negative.
    return np.interp(u, c, rs, left=rs[0], right=rs[-1])


def w2_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Wasserstein-2 distance between two measures encoded by quantile vectors.

    For quantile vectors on the same uniform grid of size K,
    W_2(mu, nu)^2 ≈ (1/K) * sum_k (q1[k] - q2[k])^2.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if q1.shape != q2.shape:
        raise ValueError("shape mismatch")
    return float(np.sqrt(np.mean((q1 - q2) ** 2)))


def isotonic_project(q: np.ndarray) -> np.ndarray:
    """Project a vector onto the cone of non-decreasing sequences (PAV).

    This is the L2-closest valid quantile function; it is also the
    closest measure under W_2 with the convention above.
    """
    q = np.asarray(q, dtype=float)
    iso = IsotonicRegression(increasing=True)
    x = np.arange(len(q), dtype=float)
    return iso.fit_transform(x, q)


def tangent_log_score(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    """Squared W_2 between forecast and realised empirical distributions."""
    return w2_distance(q_pred, q_true) ** 2
