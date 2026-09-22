"""Split-conformal calibration layer on top of any quantile forecaster.

The forecasters in :mod:`wbtc.forecasters` produce a quantile vector
``Q_t(u)`` on a grid ``u``. Nothing in the Wasserstein-geodesic construction
forces ``P(y_t <= Q_t(u)) = u`` — the h-day spread comes from a
``sqrt(h)`` rule and the location from a tangent-space slope, and both can
be mis-calibrated in a regime the training window has not seen. This module
adds the distribution-free correction from the conformal-prediction
literature (Vovk et al. 2005; Lei et al. 2018; the per-level form is the
Romano-Patterson-Candès 2019 "conformalized quantile regression" applied at
every grid level instead of one interval).

Method (per level, rolling split-conformal)
-------------------------------------------
For each grid level ``u_k`` treat the past forecast errors

    e_{s,k} = y_s - Q_s(u_k),        s in the calibration window

as conformity scores and set the offset

    d_k = e_{(ceil((n + 1) u_k)),k}   (order statistic, n = window length)

The calibrated forecast is the isotonic projection of ``Q_t(u) + d``. If
the scores are exchangeable, ``P(y_t <= Q_t(u_k) + d_k) >= u_k`` exactly in
finite samples (split-conformal guarantee). Under drift the rolling window
turns this into an approximate guarantee whose error is governed by how
non-stationary the scores are inside the window — see ``THEORY.md §2.12``
for the statement and the falsification criterion.

Two entry points share one core:

* :func:`conformalize_path` — offline, vectorised over a walk-forward path
  of base forecasts. Used by ``scripts/run_conformal.py`` to evaluate the
  layer without re-running any forecaster, and honours the harness
  no-look-ahead rule through the ``lag`` argument.
* :class:`ConformalCalibrator` — online wrapper that satisfies the
  ``fit``/``predict`` protocol. It keeps a rolling score buffer across
  successive ``fit`` calls when the training window slides forward, which
  is exactly what the daily live run in :mod:`wbtc.live` needs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .forecasters import Forecaster
from .quantiles import isotonic_project

__all__ = [
    "conformal_offsets",
    "conformalize",
    "conformalize_path",
    "pit_values",
    "empirical_coverage",
    "ConformalCalibrator",
]


# ----------------------------- core primitives -------------------------------


def conformal_offsets(scores: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Per-level split-conformal offsets from a matrix of conformity scores.

    Parameters
    ----------
    scores
        ``(n, K)`` array with ``scores[s, k] = y_s - Q_s(u_k)``.
    u
        ``(K,)`` quantile levels in (0, 1), matching the columns.

    Returns
    -------
    ``(K,)`` array ``d`` with ``d_k`` the ``ceil((n + 1) u_k)``-th smallest
    score in column ``k`` (clipped to the sample when ``(n + 1) u_k > n``,
    which is the usual finite-sample conservative choice).
    """
    E = np.asarray(scores, dtype=float)
    u = np.asarray(u, dtype=float)
    if E.ndim != 2:
        raise ValueError("scores must be 2D (n, K)")
    n, K = E.shape
    if u.shape != (K,):
        raise ValueError("u must have one entry per score column")
    if n == 0:
        raise ValueError("need at least one calibration score")
    srt = np.sort(E, axis=0)
    idx = np.ceil((n + 1) * u).astype(int) - 1
    idx = np.clip(idx, 0, n - 1)
    return srt[idx, np.arange(K)]


def conformalize(q: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Apply offsets to a base quantile vector and restore monotonicity."""
    q = np.asarray(q, dtype=float)
    d = np.asarray(offsets, dtype=float)
    if q.shape != d.shape:
        raise ValueError("q and offsets must have the same shape")
    return isotonic_project(q + d)


def pit_values(Q: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability-integral-transform of each ``y_t`` under its forecast.

    The forecast CDF is the piecewise-linear interpolant of ``(Q_t, u)``,
    with mass ``u_0`` below ``Q_t(u_0)`` and ``1 - u_{K-1}`` above
    ``Q_t(u_{K-1})`` treated as point masses at the grid ends (so PITs are
    clipped to ``[u_0, u_{K-1}]``). Good enough for calibration histograms.
    """
    Q = np.asarray(Q, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.empty(len(y), dtype=float)
    for t in range(len(y)):
        out[t] = float(np.interp(y[t], Q[t], u))
    return out


def empirical_coverage(
    Q: np.ndarray, u: np.ndarray, y: np.ndarray, levels: np.ndarray | None = None
) -> dict[float, float]:
    """``{level: mean(y_t <= Q_t(level))}`` over the path.

    ``levels`` defaults to the grid ``u`` itself; other levels are obtained
    by linear interpolation of each forecast quantile function.
    """
    Q = np.asarray(Q, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    lv = u if levels is None else np.asarray(levels, dtype=float)
    out: dict[float, float] = {}
    for a in lv:
        qa = np.array([np.interp(a, u, Q[t]) for t in range(len(y))])
        out[float(a)] = float(np.mean(y <= qa))
    return out


# ----------------------------- offline path ---------------------------------


def conformalize_path(
    Q: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    *,
    lag: int,
    calib_window: int = 250,
    min_calib: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling split-conformal correction along a walk-forward path.

    Parameters
    ----------
    Q, y, u
        Base forecasts ``(T, K)`` at consecutive origins, realised targets
        ``(T,)`` and the grid ``(K,)``.
    lag
        Number of origins that must elapse before the score at origin ``s``
        may be used: origin ``t`` uses scores from ``s <= t - lag``. For the
        walk-forward harness in :mod:`wbtc.backtest` (training window ends
        at ``t - 1``, target covers ``t + 1 .. t + h``) the realised value
        of origin ``s`` is inside the window of origin ``t`` iff
        ``t - 1 >= s + h``, i.e. ``lag = h + 1``. For the live API (window
        ends at the origin itself) ``lag = h``.
    calib_window, min_calib
        Rolling window length in origins, and the minimum number of scores
        before the correction is switched on. Below ``min_calib`` the base
        forecast is returned unchanged.

    Returns
    -------
    ``(Q_cal, active)`` where ``active[t]`` is True on rows where the
    correction was applied.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if calib_window < min_calib or min_calib < 1:
        raise ValueError("need 1 <= min_calib <= calib_window")
    T = Q.shape[0]
    E = y[:, None] - Q
    Q_cal = Q.copy()
    active = np.zeros(T, dtype=bool)
    for t in range(T):
        hi = t - lag + 1  # exclusive: scores s <= t - lag
        lo = max(0, hi - calib_window)
        if hi - lo < min_calib:
            continue
        d = conformal_offsets(E[lo:hi], u)
        Q_cal[t] = conformalize(Q[t], d)
        active[t] = True
    return Q_cal, active


# ----------------------------- online wrapper -------------------------------


@dataclass
class ConformalCalibrator:
    """Online split-conformal wrapper satisfying the ``fit``/``predict`` protocol.

    Wraps a base forecaster. On every ``fit(returns)`` it checks whether the
    new training window is the previous one shifted forward by ``k`` steps
    (``returns[:-k] == prev[k:]``); if so the newly revealed returns settle
    pending forecasts and their scores enter a rolling buffer. Any other
    window (first call, a jump, a different asset) resets the buffer. Until
    ``min_calib`` scores have accumulated for a horizon, ``predict`` returns
    the base forecast unchanged (``last_active`` tells you which).

    Target convention: the ``h``-step target of a forecast made after a
    window ending at return ``r_c`` is ``r_{c+1} + ... + r_{c+h}`` — the
    live-API convention (no skipped day). The harness convention differs by
    one step; use :func:`conformalize_path` with ``lag = h + 1`` to
    evaluate the layer on harness output.
    """

    base: Forecaster
    calib_window: int = 250
    min_calib: int = 50
    max_shift: int = 30

    _prev: np.ndarray | None = field(default=None, repr=False)
    _counter: int = field(default=0, repr=False)
    _buf_start: int = field(default=0, repr=False)  # counter of _buf[0]
    _buf: list[float] = field(default_factory=list, repr=False)
    _pending: dict[int, deque] = field(default_factory=dict, repr=False)
    _scores: dict[int, deque] = field(default_factory=dict, repr=False)
    _u: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    last_active: bool = field(default=False, repr=False)
    last_offsets: np.ndarray | None = field(default=None, repr=False)
    last_base: np.ndarray | None = field(default=None, repr=False)

    # -- window bookkeeping --

    def _shift_of(self, r: np.ndarray) -> int | None:
        prev = self._prev
        if prev is None or len(prev) != len(r):
            return None
        for k in range(0, min(self.max_shift, len(r) - 1) + 1):
            if k == 0:
                same = np.array_equal(prev, r)
            else:
                same = np.array_equal(prev[k:], r[:-k])
            if same:
                return k
        return None

    def reset(self) -> None:
        self._prev = None
        self._counter = 0
        self._buf_start = 0
        self._buf = []
        self._pending = {}
        self._scores = {}
        self._u = {}
        self.last_active = False
        self.last_offsets = None
        self.last_base = None

    def n_scores(self, h: int) -> int:
        return len(self._scores.get(h, ()))

    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        k = self._shift_of(r)
        if k is None:
            self.reset()
            self._buf = r.tolist()
            self._buf_start = 0
            self._counter = len(r)  # counter = index of the *next* return
        elif k > 0:
            self._buf.extend(r[-k:].tolist())
            self._counter += k
        self._prev = r.copy()
        self._settle()
        self._trim_buffer()
        self.base.fit(r)

    def _settle(self) -> None:
        """Move pending forecasts whose target is fully observed into scores."""
        for h, pend in self._pending.items():
            scores = self._scores.setdefault(h, deque(maxlen=self.calib_window))
            while pend and pend[0][0] + h <= self._counter - 1:
                c, q = pend.popleft()
                lo = c + 1 - self._buf_start
                y = float(np.sum(self._buf[lo : lo + h]))
                scores.append(y - q)

    def _trim_buffer(self) -> None:
        # keep enough history for the oldest pending origin
        oldest = self._counter
        for pend in self._pending.values():
            if pend:
                oldest = min(oldest, pend[0][0])
        drop = oldest - self._buf_start
        if drop > 0:
            del self._buf[:drop]
            self._buf_start += drop

    # -- forecasting --

    def predict(self, h: int, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        q = np.asarray(self.base.predict(h, u), dtype=float)
        self.last_base = q.copy()
        if h in self._u and not np.array_equal(self._u[h], u):
            raise ValueError("quantile grid changed between calls for the same h")
        self._u[h] = u
        pend = self._pending.setdefault(h, deque())
        # record the base forecast at the current origin (counter - 1 is the
        # last observed return; the forecast targets counter .. counter+h-1)
        origin = self._counter - 1
        if not pend or pend[-1][0] != origin:
            pend.append((origin, q.copy()))
        scores = self._scores.get(h)
        if scores is None or len(scores) < self.min_calib:
            self.last_active = False
            self.last_offsets = None
            return q
        E = np.stack(list(scores))
        d = conformal_offsets(E, u)
        self.last_active = True
        self.last_offsets = d
        return conformalize(q, d)
