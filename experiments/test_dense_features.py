import unittest

import numpy as np

from dense_features import (
    conditional_dense_risk,
    exact_coverage_risk,
    make_dictionary,
    spectral_sequences,
)


class DenseFeatureTests(unittest.TestCase):
    def test_rotation_is_orthogonal(self) -> None:
        q = make_dictionary(32, 0.0, seed=3)
        np.testing.assert_allclose(q @ q.T, np.eye(32), atol=2e-12)

    def test_orthogonal_conditional_risk_is_missing_energy(self) -> None:
        m = 24
        lam, q, target, tail = spectral_sequences(2.0, 1.8, 0.4, m)
        dictionary = make_dictionary(m, 0.0, seed=5)
        theta = np.sqrt(target / lam)
        teacher = dictionary.T @ theta
        seen = np.asarray([0, 2, 7, 11], dtype=np.int64)
        risk = conditional_dense_risk(
            dictionary, theta, teacher, lam, tail, seen
        )
        expected = tail + float(np.sum(np.delete(target, seen)))
        self.assertAlmostEqual(risk, expected, places=9)
        self.assertGreater(exact_coverage_risk(10, q, target, tail), tail)


if __name__ == "__main__":
    unittest.main()
