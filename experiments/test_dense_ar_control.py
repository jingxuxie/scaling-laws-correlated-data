import unittest

import numpy as np

from dense_ar_control import minimum_norm_risk, sample_dense_ar, spectral_problem


class DenseARControlTests(unittest.TestCase):
    def test_stationary_marginal_scale(self) -> None:
        lam, _, _ = spectral_problem(8, 2.0, 1.8, 3)
        rng = np.random.default_rng(5)
        x = sample_dense_ar(rng, 8000, lam, 0.7)
        empirical = np.mean(x**2, axis=0)
        np.testing.assert_allclose(empirical[:4], lam[:4], rtol=0.18, atol=0.0)

    def test_full_rank_noiseless_fit_has_small_risk(self) -> None:
        lam, theta, tail = spectral_problem(10, 2.0, 1.8, 7)
        rng = np.random.default_rng(9)
        x = sample_dense_ar(rng, 14, lam, None)
        risk = minimum_norm_risk(x, theta, lam, tail, 1e-10)
        self.assertLess(risk, tail + 1e-5)


if __name__ == "__main__":
    unittest.main()
