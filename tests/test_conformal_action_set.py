from __future__ import annotations

import math
import unittest

from sbcc.conformal_action_set import (
    EpisodeConformalActionCalibrator,
    action_nonconformity,
    episode_nonconformity,
    split_conformal_quantile,
)


class ConformalActionSetTests(unittest.TestCase):
    def test_nonconformity_credits_any_truth_optimal_action(self) -> None:
        self.assertEqual(
            action_nonconformity(
                {"left": 1.0, "right": 3.0, "wait": 2.0},
                ["left", "right"],
            ),
            0.0,
        )

    def test_episode_score_is_worst_point_not_independent_points(self) -> None:
        score = episode_nonconformity(
            [
                {"safe": 2.0, "risky": 1.0},
                {"safe": 0.0, "risky": 3.0},
            ],
            [["safe"], ["safe"]],
        )
        self.assertEqual(score, 3.0)

    def test_finite_sample_quantile_uses_conservative_rank(self) -> None:
        self.assertEqual(
            split_conformal_quantile(range(9), alpha=0.1),
            8.0,
        )
        self.assertTrue(
            math.isinf(split_conformal_quantile(range(10), alpha=0.05))
        )

    def test_fitted_threshold_produces_set_or_singleton(self) -> None:
        calibration_scores = [
            [{"a": 2.0, "b": 1.0}]
            for _ in range(9)
        ]
        calibration_truth = [[["a"]] for _ in range(9)]
        calibrator = EpisodeConformalActionCalibrator.fit(
            calibration_scores, calibration_truth, alpha=0.1
        )
        self.assertEqual(calibrator.threshold, 0.0)
        self.assertEqual(calibrator.predict({"a": 3.0, "b": 2.0}).actions, ("a",))
        self.assertEqual(
            calibrator.predict({"a": 3.0, "b": 3.0}).actions,
            ("a", "b"),
        )

    def test_infinite_threshold_abstains_over_all_actions(self) -> None:
        calibrator = EpisodeConformalActionCalibrator.fit(
            [[{"a": 1.0, "b": 0.0}]] * 10,
            [[["a"]]] * 10,
            alpha=0.05,
        )
        prediction = calibrator.predict({"a": 10.0, "b": -10.0})
        self.assertTrue(math.isinf(prediction.threshold))
        self.assertEqual(prediction.actions, ("a", "b"))
        self.assertIsNone(prediction.singleton_action)

    def test_evaluation_reports_risk_coverage_tradeoff(self) -> None:
        calibrator = EpisodeConformalActionCalibrator(
            alpha=0.1, threshold=0.5, calibration_episode_count=20
        )
        report = calibrator.evaluate(
            [
                [
                    {"left": 2.0, "right": 1.0},
                    {"left": 1.0, "right": 1.0},
                ],
                [{"left": 0.0, "right": 2.0}],
            ],
            [
                [["left"], ["right"]],
                [["left"]],
            ],
        )
        self.assertAlmostEqual(report["pointwise_optimal_action_coverage"], 2 / 3)
        self.assertAlmostEqual(report["episode_simultaneous_coverage"], 0.5)
        self.assertAlmostEqual(report["singleton_rate"], 2 / 3)
        self.assertAlmostEqual(report["wrong_singleton_rate"], 0.5)
        self.assertAlmostEqual(report["unconditional_wrong_singleton_rate"], 1 / 3)

    def test_invalid_alignment_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            EpisodeConformalActionCalibrator.fit([], [], alpha=0.1)
        with self.assertRaises(ValueError):
            action_nonconformity({"a": 1.0}, ["b"])




if __name__ == "__main__":
    unittest.main()
