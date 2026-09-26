import numpy as np

from model import fit_state_space, simulate_state_space


def main():
    # Step perturbation followed by washout.
    targets = np.concatenate(
        [
            np.zeros(10),
            np.ones(60),
            np.zeros(30),
        ]
    )

    true_a = 0.92
    true_b = 0.22

    simulated = simulate_state_space(
        targets,
        retention=true_a,
        error_sensitivity=true_b,
        process_noise_sd=0.015,
        observation_noise_sd=0.025,
        random_state=42,
    )

    fitted = fit_state_space(targets, simulated["observed"])

    print("True parameters")
    print(f"  retention A: {true_a:.3f}")
    print(f"  error sensitivity B: {true_b:.3f}")

    print("\nRecovered parameters")
    print(f"  retention A: {fitted['retention']:.3f}")
    print(f"  error sensitivity B: {fitted['error_sensitivity']:.3f}")
    print(f"  MSE: {fitted['mse']:.6f}")
    print(f"  optimizer success: {fitted['success']}")


if __name__ == "__main__":
    main()
