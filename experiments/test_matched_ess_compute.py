import unittest

import numpy as np

from matched_ess_compute import (
    exact_noiseless_risk,
    matched_profiles,
    optimize_compute_budget,
    run_compute_experiment,
    simulate_matched_ess,
)


class MatchedESSComputeTests(unittest.TestCase):
    def test_trace_iat_matches_exactly(self) -> None:
        profile = matched_profiles(a=2.0, r=0.8, trace_iat_target=15.0)
        self.assertAlmostEqual(profile["uniform_trace_iat"], 15.0, places=12)
        self.assertAlmostEqual(profile["aligned_trace_iat"], 15.0, places=12)
        self.assertGreater(profile["aligned_scale"], 1.0)

    def test_exact_risk_decreases(self) -> None:
        small = exact_noiseless_risk(2.0, 1.8, 0.8, 256, 128)
        large = exact_noiseless_risk(2.0, 1.8, 0.8, 256, 4096)
        self.assertLess(large, small)

    def test_compute_optimizer_respects_budget(self) -> None:
        model, blocks, risk = optimize_compute_budget(
            a=2.0,
            b=1.8,
            r=0.8,
            budget=2**14,
            max_model_size=512,
            grid_size=40,
        )
        self.assertLessEqual(model * blocks, 2**14)
        self.assertGreater(model, 1)
        self.assertGreater(blocks, 1)
        self.assertGreater(risk, 0.0)

    def test_small_compute_exponent_sanity(self) -> None:
        rows, summaries = run_compute_experiment(
            a=2.0,
            b=1.8,
            r_values=[0.0, 0.8],
            budgets=[2**k for k in range(10, 18)],
            max_model_size=1024,
            grid_size=100,
            fit_points=4,
        )
        self.assertTrue(rows)
        self.assertAlmostEqual(
            summaries[0.0].fitted_risk_exponent,
            summaries[0.0].predicted_risk_exponent,
            delta=0.03,
        )
        self.assertAlmostEqual(
            summaries[0.8].fitted_risk_exponent,
            summaries[0.8].predicted_risk_exponent,
            delta=0.03,
        )

    def test_small_matched_iat_simulation(self) -> None:
        _, summary = simulate_matched_ess(
            a=2.0,
            b=1.8,
            r=0.8,
            trace_iat_target=15.0,
            model_size=4096,
            horizons=[2**k for k in range(8, 14)],
            trials=30,
            fit_points=3,
            seed=7,
        )
        self.assertAlmostEqual(summary.uniform_trace_iat, 15.0, places=10)
        self.assertAlmostEqual(summary.aligned_trace_iat, 15.0, places=10)
        self.assertGreater(summary.uniform_fitted_exponent, summary.aligned_fitted_exponent)
        self.assertTrue(np.isfinite(summary.uniform_fitted_exponent))
        self.assertTrue(np.isfinite(summary.aligned_fitted_exponent))


if __name__ == "__main__":
    unittest.main()
