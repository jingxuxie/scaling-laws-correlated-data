from __future__ import annotations

import unittest

import numpy as np

from exact_risk import exact_noiseless_risk
from monte_carlo import simulate_risk


class MonteCarloTests(unittest.TestCase):
    def test_monte_carlo_matches_exact_noiseless_risk(self) -> None:
        parameters = dict(a=2.0, b=1.8, r=0.4, model_size=2048)
        risks = simulate_risk(
            blocks=512,
            trials=2500,
            sigma=0.0,
            seed=123,
            **parameters,
        )
        exact = exact_noiseless_risk(512, **parameters)[2]
        standard_error = float(np.std(risks, ddof=1) / np.sqrt(len(risks)))
        self.assertLess(abs(float(np.mean(risks)) - exact), 4.5 * standard_error)

    def test_noise_increases_risk(self) -> None:
        parameters = dict(
            blocks=256,
            trials=300,
            a=2.0,
            b=1.8,
            r=0.4,
            model_size=1024,
            seed=11,
        )
        noiseless = simulate_risk(sigma=0.0, **parameters)
        noisy = simulate_risk(sigma=0.2, **parameters)
        self.assertGreater(float(np.mean(noisy)), float(np.mean(noiseless)))


if __name__ == "__main__":
    unittest.main()
