from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def trial_by_trial_update(targets, observed):
    """Estimate how trial t error predicts correction on trial t+1."""
    targets = np.asarray(targets, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if targets.shape != observed.shape:
        raise ValueError("targets and observed must have the same shape")
    if len(targets) < 3:
        raise ValueError("At least three trials are required")

    error_t = targets[:-1] - observed[:-1]
    correction_next = observed[1:] - observed[:-1]

    model = LinearRegression()
    model.fit(error_t.reshape(-1, 1), correction_next)
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(error_t.reshape(-1, 1), correction_next)),
        "n_pairs": int(len(error_t)),
    }


def compare_conditions(condition_to_data):
    """Fit the same trial-by-trial model to multiple conditions."""
    results = {}
    for label, data in condition_to_data.items():
        results[label] = trial_by_trial_update(data["targets"], data["observed"])
    return results
