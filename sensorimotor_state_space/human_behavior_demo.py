from __future__ import annotations

import numpy as np

from model import simulate_state_space
from trial_by_trial import compare_conditions


def build_synthetic_human_behavior(random_state=42):
    """Create two synthetic adaptation conditions for a public demo."""
    targets = np.concatenate([np.zeros(10), np.ones(50), np.zeros(20)])
    land = simulate_state_space(
        targets, retention=0.93, error_sensitivity=0.18,
        observation_noise_sd=0.03, random_state=random_state,
    )
    water = simulate_state_space(
        targets, retention=0.88, error_sensitivity=0.28,
        observation_noise_sd=0.03, random_state=random_state + 1,
    )
    return {
        "Land": {"targets": targets, "observed": land["observed"]},
        "Water": {"targets": targets, "observed": water["observed"]},
    }


def main():
    results = compare_conditions(build_synthetic_human_behavior())
    for condition, stats in results.items():
        print(condition)
        print(f"  trial-by-trial gain: {stats['slope']:.3f}")
        print(f"  R^2: {stats['r2']:.3f}")
        print(f"  trial pairs: {stats['n_pairs']}")


if __name__ == "__main__":
    main()
