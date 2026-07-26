import unittest

import numpy as np

from raw_horizon import (
    conditional_noiseless_risk,
    integer_duration,
    sample_innovation_modes,
    sample_touched_modes,
)


class RawHorizonTests(unittest.TestCase):
    def test_integer_duration(self) -> None:
        modes = np.asarray([1, 2, 3, 16], dtype=np.int64)
        np.testing.assert_array_equal(integer_duration(modes, 0.0), np.ones(4, dtype=np.int64))
        np.testing.assert_array_equal(integer_duration(modes, 0.5), np.asarray([1, 2, 2, 4]))

    def test_touched_trajectory_crosses_horizon(self) -> None:
        rng = np.random.default_rng(3)
        modes, elapsed = sample_touched_modes(rng, raw_horizon=250, a=2.0, r=0.6)
        durations = integer_duration(modes, 0.6)
        self.assertGreaterEqual(elapsed, 250)
        self.assertEqual(elapsed, int(np.sum(durations)))
        self.assertLess(int(np.sum(durations[:-1])), 250)

    def test_sampling_and_risk_are_reproducible(self) -> None:
        rng_a = np.random.default_rng(9)
        rng_b = np.random.default_rng(9)
        sample_a = sample_innovation_modes(rng_a, 500, a=2.0, r=0.5)
        sample_b = sample_innovation_modes(rng_b, 500, a=2.0, r=0.5)
        np.testing.assert_array_equal(sample_a, sample_b)
        risk = conditional_noiseless_risk(sample_a, b=1.8, model_size=1024)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)


if __name__ == "__main__":
    unittest.main()
