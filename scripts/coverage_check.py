"""Christoffersen-style quantile coverage check for the proposed forecaster.

For each quantile level u_k we count the empirical hit-rate
    pi_k = mean( y_t < q_pred_t(u_k) )
and report the unconditional likelihood-ratio statistic
    LR_uc = -2 [ N_0 log(1-u_k) + N_1 log(u_k) - N_0 log(1-pi_k) - N_1 log(pi_k) ]
which is chi^2(1) under the null pi_k = u_k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from wbtc.backtest import BacktestConfig, h_step_log_return, load_returns
from wbtc.forecasters import WassersteinGeodesicGated
from wbtc.quantiles import make_grid
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "btcusdt_1d.parquet"


def kupiec_lr(hits: np.ndarray, target: float) -> tuple[float, float]:
    n1 = int(hits.sum())
    n0 = len(hits) - n1
    pi = n1 / max(1, len(hits))
    if pi <= 0 or pi >= 1:
        return 0.0, 1.0
    ll_null = n0 * np.log(1 - target) + n1 * np.log(target)
    ll_alt = n0 * np.log(1 - pi) + n1 * np.log(pi)
    lr = -2.0 * (ll_null - ll_alt)
    p = 1.0 - chi2.cdf(lr, df=1)
    return float(lr), float(p)


def main():
    df = load_returns(DATA)
    returns = df["r"].to_numpy()
    cfg = BacktestConfig(train_window=730, horizon=1, K=20, test_holdout=365)
    u = make_grid(cfg.K)

    rows = []
    hits = {float(uk): [] for uk in u}
    n = cfg.train_window
    for t in tqdm(range(n, len(returns) - cfg.horizon), desc="coverage"):
        window = returns[t - n : t]
        f = WassersteinGeodesicGated(window=90, lookback=20)
        f.fit(window)
        q = f.predict(cfg.horizon, u)
        y = h_step_log_return(returns, t, cfg.horizon)
        if y is None:
            continue
        for uk, qk in zip(u, q):
            hits[float(uk)].append(int(y < qk))
        rows.append({"t": t, "y": y})

    # restrict to test set: last 365
    test_start = n + (len(returns) - n - cfg.horizon) - cfg.test_holdout
    keep_idx = [i for i, r in enumerate(rows) if r["t"] >= test_start]

    print(f"\n{'u':>6}  {'emp_hit':>10}  {'expected':>10}  {'LR':>8}  {'p':>8}")
    for uk in u:
        h_all = np.array(hits[float(uk)])
        h_test = h_all[keep_idx]
        emp = h_test.mean()
        lr, p = kupiec_lr(h_test, float(uk))
        print(f"{float(uk):6.3f}  {emp:10.4f}  {float(uk):10.4f}  {lr:8.3f}  {p:8.4f}")


if __name__ == "__main__":
    main()
