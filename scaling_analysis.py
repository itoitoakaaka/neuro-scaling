"""Leakage-safe scaling-curve utilities for EEG decoding."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _stratified_subset_indices(y, n_subset, rng):
    """Draw an approximately stratified subset without replacement."""
    classes, counts = np.unique(y, return_counts=True)
    selected = []

    allocations = np.floor(n_subset * counts / counts.sum()).astype(int)
    allocations = np.maximum(allocations, 2)

    while allocations.sum() > n_subset:
        candidates = np.where(allocations > 2)[0]
        if len(candidates) == 0:
            break
        allocations[candidates[np.argmax(allocations[candidates])]] -= 1

    while allocations.sum() < n_subset:
        remaining = counts - allocations
        candidates = np.where(remaining > 0)[0]
        if len(candidates) == 0:
            break
        allocations[candidates[np.argmax(remaining[candidates])]] += 1

    for cls, n_take in zip(classes, allocations):
        cls_idx = np.flatnonzero(y == cls)
        selected.extend(rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False))

    return np.asarray(selected, dtype=int)


def compute_scaling_curve(
    X,
    y,
    data_fractions=None,
    n_cv=5,
    n_repeats=5,
    random_state=42,
):
    """Estimate decoding accuracy as a function of available data."""
    if data_fractions is None:
        data_fractions = np.arange(0.2, 1.01, 0.1)

    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    score_means = []
    score_stds = []
    minimum_subset = min(len(y), max(2 * n_cv, 2 * len(np.unique(y))))

    for frac in data_fractions:
        n_subset = max(int(round(len(y) * float(frac))), minimum_subset)
        n_subset = min(n_subset, len(y))
        repeated_scores = []

        for repeat in range(n_repeats):
            idx = _stratified_subset_indices(y, n_subset, rng)
            X_sub, y_sub = X[idx], y[idx]

            _, counts = np.unique(y_sub, return_counts=True)
            n_splits = min(n_cv, int(counts.min()))
            if n_splits < 2:
                continue

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + repeat,
            )
            repeated_scores.extend(
                cross_val_score(model, X_sub, y_sub, cv=cv, scoring="accuracy")
            )

        score_means.append(np.mean(repeated_scores) if repeated_scores else np.nan)
        score_stds.append(np.std(repeated_scores) if repeated_scores else np.nan)

    return (
        np.asarray(data_fractions, dtype=float),
        np.asarray(score_means, dtype=float),
        np.asarray(score_stds, dtype=float),
    )


def fit_power_law(fractions, scores):
    """Fit y = a*x^b + c to finite scaling-curve points."""
    from scipy.optimize import curve_fit

    fractions = np.asarray(fractions, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(fractions) & np.isfinite(scores)
    x = fractions[valid]
    y = scores[valid]

    if len(x) < 4:
        return None

    def power_law(x_value, a, b, c):
        return a * np.power(x_value, b) + c

    try:
        params, _ = curve_fit(
            power_law,
            x,
            y,
            p0=[-0.3, -0.5, 0.8],
            maxfev=10000,
        )
        return params
    except (RuntimeError, ValueError):
        return None


def plot_scaling_curve(
    fractions,
    scores_mean,
    scores_std,
    params=None,
    output_path="scaling_curve.png",
):
    """Plot decoding accuracy versus available data fraction."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        fractions,
        scores_mean,
        yerr=scores_std,
        fmt="o-",
        capsize=4,
        label="Cross-validated accuracy",
    )

    if params is not None:
        a, b, c = params
        x_fit = np.linspace(float(fractions[0]), float(fractions[-1]), 100)
        y_fit = a * np.power(x_fit, b) + c
        plt.plot(x_fit, y_fit, "--", label="Power-law fit")

    plt.xlabel("Data fraction")
    plt.ylabel("Accuracy")
    plt.title("EEG decoding performance vs. data scale")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")
