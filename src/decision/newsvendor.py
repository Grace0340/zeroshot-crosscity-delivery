"""Newsvendor-style capacity provisioning replay.

For each (region, hour): replay a provisioned capacity c against realized
demand y with cost = co * max(c - y, 0) + cu * max(y - c, 0).
- point strategy:    c = point forecast (median)
- interval strategy: c = calibrated upper bound, or the calibrated quantile at
  the critical level cu / (co + cu)
"""
import numpy as np


def replay_cost(capacity: np.ndarray, demand: np.ndarray, co: float, cu: float) -> float:
    over = np.maximum(capacity - demand, 0.0)
    under = np.maximum(demand - capacity, 0.0)
    return float((co * over + cu * under).sum())


def evaluate_policies(y_true, y_point, y_upper, cost_ratios):
    """y_true/y_point/y_upper: [T, N]; cost_ratios: [[co, cu], ...]"""
    rows = []
    for co, cu in cost_ratios:
        base = replay_cost(y_point, y_true, co, cu)
        intv = replay_cost(y_upper, y_true, co, cu)
        rows.append({
            "co": co, "cu": cu,
            "cost_point": base,
            "cost_interval": intv,
            "saving_pct": 100.0 * (base - intv) / base if base > 0 else 0.0,
        })
    return rows
