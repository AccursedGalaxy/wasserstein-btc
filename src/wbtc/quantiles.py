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
