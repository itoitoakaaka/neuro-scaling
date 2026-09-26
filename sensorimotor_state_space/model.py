from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def simulate_state_space(
    targets,
    retention=0.90,
    error_sensitivity=0.25,
    process_noise_sd=0.0,
    observation_noise_sd=0.0,
    initial_state=0.0,
    random_state=None,
):
    """Simulate a one-state error-based sensorimotor adaptation model."""
    targets = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(random_state)

    latent = np.zeros(len(targets), dtype=float)
    observed = np.zeros(len(targets), dtype=float)
    errors = np.zeros(len(targets), dtype=float)

    state = float(initial_state)
    for t, target in enumerate(targets):
        latent[t] = state
        observed[t] = state + rng.normal(0.0, observation_noise_sd)
        errors[t] = target - observed[t]
        state = (
            retention * state
            + error_sensitivity * errors[t]
            + rng.normal(0.0, process_noise_sd)
        )

    return {
        "latent_state": latent,
        "observed": observed,
        "error": errors,
    }


def predict_observed(targets, retention, error_sensitivity, initial_state=0.0):
    """Deterministic model prediction used for fitting."""
    return simulate_state_space(
        targets,
        retention=retention,
        error_sensitivity=error_sensitivity,
        process_noise_sd=0.0,
        observation_noise_sd=0.0,
        initial_state=initial_state,
        random_state=0,
    )["observed"]


def fit_state_space(targets, observed, initial=(0.9, 0.2)):
    """Fit retention A and error sensitivity B by least squares."""
    targets = np.asarray(targets, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if targets.shape != observed.shape:
        raise ValueError("targets and observed must have the same shape")

    def objective(params):
        retention, error_sensitivity = params
        prediction = predict_observed(
            targets,
            retention=retention,
            error_sensitivity=error_sensitivity,
            initial_state=observed[0],
        )
        return float(np.mean((observed - prediction) ** 2))

    result = minimize(
        objective,
        x0=np.asarray(initial, dtype=float),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        method="L-BFGS-B",
    )

    return {
        "retention": float(result.x[0]),
        "error_sensitivity": float(result.x[1]),
        "mse": float(result.fun),
        "success": bool(result.success),
    }
