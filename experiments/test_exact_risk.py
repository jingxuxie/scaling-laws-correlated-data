from __future__ import annotations

import math
import unittest

import numpy as np

from exact_risk import evaluate_curve, exact_noiseless_risk, power_law_sequences


class ExactRiskTests(unittest.TestCase):
    def test_mode_distribution_normalizes(self) -> None:
        _, q, _, _ = power_law_sequences(a=2.0, b=1.8, r=0.5, model_size=200_000)
        # The omitted power-law tail is small and positive.
        self.assertLess(abs(float(np.sum(q)) - 1.0), 0.003)

    def test_risk_decreases_with_blocks(self) -> None:
        risks = [
            exact_noiseless_risk(B, 2.0, 1.8, 0.4, 100_000)[2]
            for B in (64, 256, 1024, 4096)
        ]
        self.assertTrue(all(x > y for x, y in zip(risks, risks[1:])))

    def test_risk_decreases_with_model_size(self) -> None:
        risks = [
            exact_noiseless_risk(4096, 2.0, 1.8, 0.4, M)[2]
            for M in (32, 128, 512, 2048)
        ]
        self.assertTrue(all(x > y for x, y in zip(risks, risks[1:])))

    def test_fitted_slope_matches_theory(self) -> None:
        rows, summary = evaluate_curve(
            a=2.0,
            b=1.8,
            r=0.4,
            model_size=300_000,
            block_powers=list(range(9, 26)),
            fit_points=7,
        )
        self.assertTrue(rows)
        self.assertLess(abs(summary.fitted_slope - summary.predicted_slope), 0.018)

    def test_zero_blocks_recovers_prior_energy(self) -> None:
        _, _, total = exact_noiseless_risk(0, 2.0, 1.8, 0.4, 100_000)
        self.assertTrue(math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-5))


if __name__ == "__main__":
    unittest.main()
