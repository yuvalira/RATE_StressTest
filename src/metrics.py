"""Aggregation utilities shared by both experiments: turn repeated point
estimates into bias/variance/RMSE, and run a full (grid point x seed x
estimator) sweep into one tidy long DataFrame.
"""

from typing import Callable

import numpy as np
import pandas as pd


def compute_errors(estimates: pd.Series, true_value: float) -> dict:
    """Bias/variance/RMSE of a set of point estimates against a known
    ground-truth value."""
    estimates = np.asarray(estimates, dtype=float)
    mean_estimate = estimates.mean()
    bias = mean_estimate - true_value
    variance = estimates.var(ddof=1) if len(estimates) > 1 else 0.0
    rmse = np.sqrt(np.mean((estimates - true_value) ** 2))
    return {"mean_estimate": mean_estimate, "bias": bias, "variance": variance, "rmse": rmse}


def run_grid(
    param_grid: list[dict],
    simulate_fn: Callable[..., pd.DataFrame],
    estimator_fns: dict[str, Callable[[pd.DataFrame], dict]],
    true_ate_value: float,
    n_seeds: int,
    estimand: str = "ATE",
) -> pd.DataFrame:
    """For each grid point x seed x estimator, simulate data and compute a
    point estimate; then aggregate across seeds into bias/variance/RMSE.

    Returns a tidy long DataFrame with columns:
        [*grid_params, estimator, mean_estimate, bias, variance, rmse, n_seeds]
    """
    raw_rows = []
    for params in param_grid:
        for seed in range(n_seeds):
            df = simulate_fn(seed=seed, **params)
            for name, fn in estimator_fns.items():
                estimate = fn(df)[estimand]
                raw_rows.append({**params, "seed": seed, "estimator": name, "estimate": estimate})

    raw = pd.DataFrame(raw_rows)
    grid_cols = list(param_grid[0].keys())

    results = []
    for keys, group in raw.groupby(grid_cols + ["estimator"], dropna=False):
        row = dict(zip(grid_cols + ["estimator"], keys))
        row.update(compute_errors(group["estimate"], true_ate_value))
        row["n_seeds"] = len(group)
        results.append(row)

    return pd.DataFrame(results)
