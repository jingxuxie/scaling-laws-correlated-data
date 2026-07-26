import unittest

import numpy as np

from noise_regimes import (
    binomial_reciprocal_expectation,
    direct_h,
    exact_risk_over_models,
)


class NoiseRegimeTests(unittest.TestCase):
    def test_quadrature_matches_direct_sum(self) -> None:
        blocks = 17
        q = np.asarray([0.01, 0.1, 0.4, 0.8])
        estimated = binomial_reciprocal_expectation(
            blocks, q, quadrature_order=120
        )
        reference = np.asarray([direct_h(blocks, value) for value in q])
        np.testing.assert_allclose(estimated, reference, rtol=2e-11, atol=2e-13)

    def test_exact_risk_is_finite(self) -> None:
        models, risks, pieces = exact_risk_over_models(
            blocks=128,
            a=2.0,
            b=1.8,
            r=0.5,
            sigma=0.2,
            max_model_size=128,
            quadrature_order=60,
        )
        self.assertEqual(models.size, 128)
        self.assertTrue(np.all(np.isfinite(risks)))
        self.assertTrue(np.all(risks > 0))
        self.assertEqual(set(pieces), {"approximation", "coverage", "variance"})


if __name__ == "__main__":
    unittest.main()
