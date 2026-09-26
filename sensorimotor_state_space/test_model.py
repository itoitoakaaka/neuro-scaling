import unittest

import numpy as np

from model import fit_state_space, simulate_state_space


class StateSpaceTests(unittest.TestCase):
    def test_parameter_recovery_without_noise(self):
        targets = np.concatenate([np.zeros(5), np.ones(80), np.zeros(20)])
        true_a = 0.9
        true_b = 0.25
        observed = simulate_state_space(
            targets,
            retention=true_a,
            error_sensitivity=true_b,
            random_state=1,
        )["observed"]

        fitted = fit_state_space(targets, observed)

        self.assertAlmostEqual(fitted["retention"], true_a, places=2)
        self.assertAlmostEqual(fitted["error_sensitivity"], true_b, places=2)


if __name__ == "__main__":
    unittest.main()
