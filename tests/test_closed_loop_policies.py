from __future__ import annotations

import unittest

from sbcc.closed_loop_policies import CandidateCommitteePolicy
from sbcc.decision_ensemble import CandidateEvaluation


def evaluations(_observations, _actions):
    return (
        CandidateEvaluation("a", 0.8, {"left": 1.0, "right": 0.0}),
        CandidateEvaluation("b", 0.8, {"left": 0.0, "right": 1.0}),
    )


class ClosedLoopPolicyTests(unittest.TestCase):
    def make(self, mode, **kwargs):
        return CandidateCommitteePolicy(
            evaluation_provider=evaluations,
            action_decoder={"left": 0, "right": 1},
            mode=mode,
            near_best_std_multiplier=1.0,
            **kwargs,
        )

    def test_worldledger_abstains_and_executes_frozen_fallback(self) -> None:
        policy = self.make("worldledger", fallback_mode="minimax")
        policy.reset(1)
        decision = policy.decide({"public": True}, step=0)
        self.assertIsNone(decision.committed_action)
        self.assertEqual(decision.executed_action, 0)
        self.assertEqual(decision.diagnostics["reason"], "abstain_fallback_minimax")

    def test_worldledger_commits_only_when_candidate_rankings_agree(self) -> None:
        def agreeing(_observations, _actions):
            return (
                CandidateEvaluation("a", 0.5, {"left": 2.0, "right": 0.0}),
                CandidateEvaluation("b", 0.5, {"left": 1.0, "right": 0.5}),
            )

        policy = CandidateCommitteePolicy(
            evaluation_provider=agreeing,
            action_decoder={"left": 0, "right": 1},
            mode="worldledger",
        )
        policy.reset(1)
        decision = policy.decide(None, step=0)
        self.assertEqual(decision.committed_action, 0)
        self.assertIn(decision.diagnostics["reason"], {"ranking_consensus", "interval_dominance"})

    def test_map_bma_and_minimax_are_deployment_eligible_policies(self) -> None:
        expected = {"map": 0, "bma": 0, "minimax": 0}
        for mode, action in expected.items():
            with self.subTest(mode=mode):
                policy = self.make(mode)
                policy.reset(2)
                decision = policy.decide(None, step=0)
                self.assertEqual(decision.executed_action, action)
                self.assertEqual(decision.committed_action, action)

    def test_coherent_thompson_keeps_one_complete_candidate_per_episode(self) -> None:
        policy = self.make("coherent_thompson")
        policy.reset(3)
        first = policy.decide({"t": 0}, step=0)
        second = policy.decide({"t": 1}, step=1)
        self.assertEqual(
            first.diagnostics["selected_candidate_id"],
            second.diagnostics["selected_candidate_id"],
        )

    def test_provider_receives_only_observation_and_action_histories(self) -> None:
        calls = []

        def provider(observations, actions):
            calls.append((observations, actions))
            return (CandidateEvaluation("a", 1.0, {"left": 1.0}),)

        policy = CandidateCommitteePolicy(
            evaluation_provider=provider,
            action_decoder={"left": 0},
            mode="map",
        )
        policy.reset(4)
        policy.decide({"visible": 1}, step=0)
        self.assertEqual(calls, [(({"visible": 1},), ())])

    def test_all_zero_values_are_exposed_without_changing_frozen_actions(self):
        for mode in ("map", "bma", "minimax", "coherent_thompson", "worldledger"):
            with self.subTest(mode=mode):
                policy = CandidateCommitteePolicy(
                    evaluation_provider=lambda o, a: (
                        CandidateEvaluation("a", 1.0, {"left": 0.0, "right": 0.0}),
                        CandidateEvaluation("b", 1.0, {"left": 0.0, "right": 0.0}),
                    ),
                    action_decoder={"left": 0, "right": 1}, mode=mode,
                )
                policy.reset(9)
                decision = policy.decide(None, step=0)
                self.assertEqual(decision.executed_action, 0)
                self.assertTrue(decision.diagnostics["all_candidates_action_values_tied"])
                self.assertEqual(decision.diagnostics["candidate_action_value_ranges"], {"a": 0.0, "b": 0.0})
                if mode == "worldledger":
                    self.assertIsNone(decision.committed_action)
                    self.assertEqual(decision.diagnostics["reason"], "abstain_fallback_minimax")

    def test_seeded_uniform_tie_break_is_reproducible_and_mode_independent(self):
        def tied(_observations, _actions):
            return (
                CandidateEvaluation("a", 1.0, {"0": 0.0, "1": 0.0, "2": 0.0}),
                CandidateEvaluation("b", 1.0, {"0": 0.0, "1": 0.0, "2": 0.0}),
            )
        actions = []
        for mode in ("map", "bma", "minimax", "coherent_thompson", "worldledger"):
            policy = CandidateCommitteePolicy(
                evaluation_provider=tied, action_decoder={"0": 0, "1": 1, "2": 2},
                mode=mode, tie_breaking="seeded-uniform-v1",
            )
            policy.reset(1717)
            actions.append(policy.decide(None, step=0).executed_action)
        self.assertEqual(len(set(actions)), 1)
        self.assertIn(actions[0], (0, 1, 2))

    def test_seeded_uniform_tie_break_does_not_depend_on_action_order(self):
        def provider(order):
            return lambda o, a: (CandidateEvaluation("a", 1.0, {key: 0.0 for key in order}),)
        selected = []
        for order in (("0", "1", "2"), ("2", "0", "1")):
            policy = CandidateCommitteePolicy(
                evaluation_provider=provider(order), action_decoder={"0": 0, "1": 1, "2": 2},
                mode="map", tie_breaking="seeded-uniform-v1",
            )
            policy.reset(88)
            selected.append(policy.decide(None, step=4).executed_action)
        self.assertEqual(selected[0], selected[1])


if __name__ == "__main__":
    unittest.main()
