import unittest

import numpy as np

from real_sequential import (
    fit_proxy,
    integrated_autocorrelation_times,
)


class RealSequentialTests(unittest.TestCase):
    def test_iat_detects_persistence(self) -> None:
        rng = np.random.default_rng(9)
        n = 4096
        scores = np.empty((n, 2), dtype=np.float64)
        scores[0] = rng.normal(size=2)
        rho = np.asarray([0.1, 0.9])
        for index in range(1, n):
            scores[index] = rho * scores[index - 1] + np.sqrt(1 - rho**2) * rng.normal(size=2)
        tau = integrated_autocorrelation_times(scores, max_lag=128)
        self.assertGreater(tau[1], tau[0] * 3)
        self.assertGreaterEqual(tau[0], 1.0)

    def test_proxy_recovers_generating_family(self) -> None:
        sizes = np.asarray([16, 32, 64, 128, 256, 512], dtype=np.int64)
        rate = np.asarray([1.0, 0.2, 0.05])
        energy = np.asarray([0.4, 0.35, 0.25])
        excess = 2.3 * np.asarray(
            [np.dot(energy, np.exp(-0.07 * n * rate)) for n in sizes]
        )
        fit, prediction = fit_proxy(
            name="test",
            sizes=sizes,
            excess=excess,
            rate=rate,
            energy=energy,
            fit_count=3,
        )
        self.assertLess(fit.extrapolation_log_rmse, 0.05)
        np.testing.assert_allclose(prediction, excess, rtol=0.06)


if __name__ == "__main__":
    unittest.main()
