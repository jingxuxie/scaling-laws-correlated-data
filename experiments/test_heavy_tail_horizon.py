import unittest

import numpy as np

from heavy_tail_horizon import (
    fit_exponent,
    sample_stationary_touched_modes,
    stationary_residual_tail,
)


class HeavyTailHorizonTests(unittest.TestCase):
    def test_stationary_path_always_touches_initial_mode(self) -> None:
        rng = np.random.default_rng(4)
        modes, residual = sample_stationary_touched_modes(
            rng, raw_horizon=128, a=2.0, r=2.0
        )
        self.assertGreaterEqual(modes.size, 1)
        self.assertGreaterEqual(residual, 1)

    def test_residual_tail_decreases(self) -> None:
        small = stationary_residual_tail(64, 2.0, 2.0, max_mode=100_000)
        large = stationary_residual_tail(1024, 2.0, 2.0, max_mode=100_000)
        self.assertGreater(small, large)
        self.assertGreater(large, 0.0)

    def test_power_fit(self) -> None:
        horizons = np.asarray([16, 32, 64, 128], dtype=np.int64)
        values = horizons.astype(np.float64) ** -0.4
        exponent, start, end = fit_exponent(horizons, values, 4)
        self.assertAlmostEqual(exponent, 0.4, places=12)
        self.assertEqual(start, 16)
        self.assertEqual(end, 128)


if __name__ == "__main__":
    unittest.main()
