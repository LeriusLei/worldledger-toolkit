from __future__ import annotations

import unittest

from sbcc.decision_ensemble import (
    CandidateEvaluation,
    DecisionCommittee,
    build_candidate_evaluations,
)


class DecisionCommitteeTests(unittest.TestCase):
    def test_all_tied_candidates_do_not_create_false_consensus(self) -> None:
        committee = DecisionCommittee(
            (
                CandidateEvaluation("a", 1.0, {"left": 0.0, "right": 0.0}),
                CandidateEvaluation("b", 1.0, {"left": 0.0, "right": 0.0}),
            )
        )
        self.assertIsNone(committee.consensus_action())
        certificate = committee.interval_certificate()
        self.assertIsNone(certificate.action)
        self.assertEqual(certificate.reason, "decision_ambiguous")

    def test_equal_factual_scores_can_have_two_decision_classes(self) -> None:
        committee = DecisionCommittee(
            [
                CandidateEvaluation("dry", 0.0, {"wait": 1.0, "water": 0.0}),
                CandidateEvaluation("wet", 0.0, {"wait": 0.0, "water": 1.0}),
            ]
        )
        self.assertEqual(len(committee.decision_classes()), 2)
        self.assertIsNone(committee.interval_certificate().action)
        self.assertAlmostEqual(committee.model_entropy_bits(), 1.0)

    def test_different_predictions_can_still_have_ranking_consensus(self) -> None:
        committee = DecisionCommittee(
            [
                CandidateEvaluation("m1", 0.0, {"left": 2.0, "right": 1.5}),
                CandidateEvaluation("m2", 0.0, {"left": 3.0, "right": 2.5}),
            ]
        )
        certificate = committee.interval_certificate()
        self.assertEqual(certificate.action, "left")
        self.assertEqual(certificate.reason, "ranking_consensus")
        self.assertEqual(len(committee.decision_classes()), 1)

    def test_interval_dominance_is_stronger_than_ranking_consensus(self) -> None:
        committee = DecisionCommittee(
            [
                CandidateEvaluation("m1", 0.0, {"left": 3.0, "right": 1.0}),
                CandidateEvaluation("m2", 0.0, {"left": 4.0, "right": 2.0}),
            ]
        )
        certificate = committee.interval_certificate()
        self.assertEqual(certificate.action, "left")
        self.assertEqual(certificate.reason, "interval_dominance")
        self.assertEqual(certificate.lower_bound, 3.0)
        self.assertEqual(certificate.competing_upper_bound, 2.0)

    def test_near_best_filter_matches_one_standard_deviation_rule(self) -> None:
        committee = DecisionCommittee(
            [
                CandidateEvaluation("best", 0.0, {"a": 1.0, "b": 0.0}),
                CandidateEvaluation("near", -0.1, {"a": 0.9, "b": 0.1}),
                CandidateEvaluation("bad", -10.0, {"a": 0.0, "b": 1.0}),
            ],
            temperature=0.1,
        )
        self.assertNotIn("bad", committee.weights)
        self.assertIn("best", committee.weights)

    def test_minimax_regret_uses_complete_candidate_rows(self) -> None:
        committee = DecisionCommittee(
            [
                CandidateEvaluation("m1", 0.0, {"left": 1.0, "right": 0.0, "safe": 0.45}),
                CandidateEvaluation("m2", 0.0, {"left": 0.0, "right": 1.0, "safe": 0.45}),
            ]
        )
        self.assertEqual(committee.minimax_regret_action(), "safe")
        self.assertAlmostEqual(committee.maximum_regret("safe"), 0.55)

    def test_adapter_rejects_candidate_misalignment(self) -> None:
        with self.assertRaises(ValueError):
            build_candidate_evaluations(
                {"m1": 0.0},
                {"m2": {"a": 1.0}},
            )


if __name__ == "__main__":
    unittest.main()
