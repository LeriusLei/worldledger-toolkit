from __future__ import annotations

import math
import unittest

from sbcc.statistical_analysis import (
    clopper_pearson_interval,
    exact_mcnemar_power,
    exact_mcnemar_test,
    paired_episode_bootstrap_difference,
    required_mcnemar_episodes,
)


class StatisticalAnalysisTests(unittest.TestCase):
    def test_zero_events_in_sixty_has_nonzero_exact_upper_bound(self) -> None:
        lower, upper = clopper_pearson_interval(
            0, 60, confidence_level=0.95, alternative="upper"
        )
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 1.0 - 0.05 ** (1.0 / 60.0), places=12)

    def test_two_sided_interval_contains_observed_rate(self) -> None:
        lower, upper = clopper_pearson_interval(3, 60)
        self.assertLess(lower, 3 / 60)
        self.assertGreater(upper, 3 / 60)

    def test_mcnemar_uses_only_discordant_episode_pairs(self) -> None:
        report = exact_mcnemar_test(
            [True] * 7 + [False] * 5,
            [False] * 7 + [True] + [False] * 4,
        )
        self.assertEqual(report["improvement_discordant_episodes"], 7)
        self.assertEqual(report["harm_discordant_episodes"], 1)
        self.assertEqual(report["discordant_episode_count"], 8)
        self.assertAlmostEqual(report["exact_p_value"], 9 / 256)

    def test_power_increases_for_a_stronger_paired_effect(self) -> None:
        weak = exact_mcnemar_power(
            100, improvement_probability=0.08, harm_probability=0.04
        )
        strong = exact_mcnemar_power(
            100, improvement_probability=0.12, harm_probability=0.01
        )
        self.assertGreater(strong, weak)

    def test_required_episode_count_reaches_target(self) -> None:
        required = required_mcnemar_episodes(
            improvement_probability=0.10,
            harm_probability=0.02,
            target_power=0.8,
        )
        self.assertIsNotNone(required)
        assert required is not None
        self.assertGreaterEqual(
            exact_mcnemar_power(
                required,
                improvement_probability=0.10,
                harm_probability=0.02,
            ),
            0.8,
        )

    def test_paired_bootstrap_keeps_episode_pairs_together(self) -> None:
        report = paired_episode_bootstrap_difference(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            bootstrap_samples=200,
            seed=7,
        )
        self.assertEqual(report["mean_method_minus_baseline"], 1.0)
        self.assertEqual(report["confidence_interval"], [1.0, 1.0])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            clopper_pearson_interval(2, 1)
        with self.assertRaises(ValueError):
            exact_mcnemar_test([], [])
        with self.assertRaises(ValueError):
            exact_mcnemar_power(
                10, improvement_probability=0.8, harm_probability=0.3
            )


if __name__ == "__main__":
    unittest.main()
