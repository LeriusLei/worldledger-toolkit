from __future__ import annotations

import random
import unittest

from sbcc.agents import (
    run_falsifywm,
    run_global_information_gain,
    run_map,
    run_robust,
)
from sbcc.sprinkler import Hypothesis, Posterior, SprinklerWorld, make_hypotheses


class SprinklerBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hypotheses = make_hypotheses(16)
        self.prior = Posterior.uniform(self.hypotheses)

    def test_all_candidates_have_identical_factual_likelihood(self) -> None:
        history = SprinklerWorld.factual_history()
        scores = {
            hypothesis.factual_log_likelihood(history)
            for hypothesis in self.hypotheses
        }
        self.assertEqual(scores, {0.0})

    def test_counterfactual_action_rankings_disagree(self) -> None:
        optimal_actions = {hypothesis.optimal_action for hypothesis in self.hypotheses}
        self.assertEqual(optimal_actions, {"no_water", "water"})
        self.assertEqual(len(self.prior.decision_classes()), 2)

    def test_global_information_gain_chases_nuisance(self) -> None:
        world = SprinklerWorld(Hypothesis(response_type=1, phenotype=7))
        trace = run_global_information_gain(world, self.prior, random.Random(0))
        self.assertEqual(trace.probe, "phenotype")
        self.assertEqual(trace.decision_classes_after, 2)
        self.assertAlmostEqual(trace.entropy_after, 1.0)

    def test_falsifywm_resolves_decision_but_not_full_model(self) -> None:
        world = SprinklerWorld(Hypothesis(response_type=1, phenotype=7))
        trace = run_falsifywm(world, self.prior, random.Random(0))
        self.assertEqual(trace.probe, "root_zone")
        self.assertEqual(trace.action, "water")
        self.assertEqual(trace.decision_classes_after, 1)
        self.assertAlmostEqual(trace.entropy_after, 4.0)
        self.assertAlmostEqual(trace.total_utility, 0.95)

    def test_falsifywm_uses_noisy_but_valuable_probe(self) -> None:
        world = SprinklerWorld(
            Hypothesis(response_type=1, phenotype=7),
            root_zone_accuracy=0.8,
            rng=random.Random(1),
        )
        trace = run_falsifywm(world, self.prior, random.Random(0))
        self.assertEqual(trace.probe, "root_zone")
        self.assertEqual(trace.action, "water")
        self.assertAlmostEqual(trace.prediction_confidence, 0.8)

    def test_falsifywm_declines_probe_below_break_even_value(self) -> None:
        world = SprinklerWorld(
            Hypothesis(response_type=1, phenotype=7),
            root_zone_accuracy=0.55,
            probe_costs={"root_zone": 0.10, "phenotype": 0.05},
            rng=random.Random(0),
        )
        trace = run_falsifywm(world, self.prior, random.Random(0))
        self.assertIsNone(trace.probe)

    def test_map_can_be_confident_and_wrong(self) -> None:
        world = SprinklerWorld(Hypothesis(response_type=1, phenotype=7))
        trace = run_map(world, self.prior, random.Random(0))
        self.assertEqual(trace.action, "no_water")
        self.assertEqual(trace.prediction_confidence, 1.0)
        self.assertEqual(trace.action_reward, 0.0)

    def test_robust_policy_uses_safe_action(self) -> None:
        world = SprinklerWorld(Hypothesis(response_type=1, phenotype=7))
        trace = run_robust(world, self.prior, random.Random(0))
        self.assertEqual(trace.action, "safe")
        self.assertAlmostEqual(trace.total_utility, 0.45)

    def test_only_one_probe_is_allowed(self) -> None:
        world = SprinklerWorld(Hypothesis(response_type=0, phenotype=3))
        world.probe("root_zone")
        with self.assertRaises(RuntimeError):
            world.probe("phenotype")

    def test_factual_scorer_rejects_counterfactual_leakage(self) -> None:
        hypothesis = self.hypotheses[0]
        with self.assertRaises(ValueError):
            hypothesis.factual_log_likelihood((("water", "normal", 1.0),))


if __name__ == "__main__":
    unittest.main()
