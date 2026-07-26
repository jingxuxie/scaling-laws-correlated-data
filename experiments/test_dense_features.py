import unittest

import numpy as np

from dense_features import (
    decoded_dense_risk,
    exact_coverage_risk,
    make_dictionary,
    max_row_coherence,
)


class DenseFeatureTests(unittest.TestCase):
    def test_decoded_risk_is_representation_invariant(self) -> None:
        dictionary = make_dictionary(12, 0.35, 7)
        theta = np.linspace(-1.0, 1.0, 12)
        lam = np.arange(1, 13, dtype=float) ** -2
        lam /= lam.sum()
        seen = np.array([0, 2, 5, 9], dtype=np.int64)
        tail = 0.017
        risk = decoded_dense_risk(dictionary, theta, lam, tail, seen)
        unseen = np.setdiff1d(np.arange(12), seen)
        expected = tail + float(np.dot(lam[unseen], theta[unseen] ** 2))
        self.assertAlmostEqual(risk, expected, places=10)

    def test_orthogonal_dictionary_is_well_conditioned(self) -> None:
        dictionary = make_dictionary(24, 0.0, 11)
        self.assertLess(np.linalg.cond(dictionary), 1.0000001)
        self.assertLess(max_row_coherence(dictionary), 1e-12)

    def test_exact_coverage_at_zero_like_scale(self) -> None:
        q = np.array([0.2, 0.1])
        target = np.array([0.3, 0.4])
        expected = 0.05 + np.dot(target, (1 - q) ** 3)
        self.assertAlmostEqual(exact_coverage_risk(3, q, target, 0.05), expected)


if __name__ == "__main__":
    unittest.main()
