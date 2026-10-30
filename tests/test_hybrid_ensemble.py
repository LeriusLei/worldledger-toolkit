from __future__ import annotations

import unittest

from sbcc.decision_ensemble import DecisionCommittee
from sbcc.hybrid_ensemble import (
    audit_componentwise_mixing,
    index_incoherent_probability,
    make_crossed_reward_counterexample,
)


class HybridEnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = make_crossed_reward_counterexample()
        self.audit = audit_componentwise_mixing(self.candidates)

    def test_both_complete_worlds_prefer_safe(self) -> None:
        for candidate in self.candidates:
            self.assertEqual(candidate.evaluation().action_values["safe"], 1.0)
            self.assertEqual(candidate.evaluation().action_values["risky"], 0.0)

    def test_coherent_committee_certifies_safe(self) -> None:
        committee = DecisionCommittee(
            [candidate.evaluation() for candidate in self.candidates]
        )
        certificate = committee.interval_certificate()
        self.assertEqual(certificate.action, "safe")
        self.assertEqual(certificate.reason, "interval_dominance")

    def test_component_pair_count_grows_quadratically(self) -> None:
        self.assertEqual(self.audit.candidate_count, 2)
        self.assertEqual(self.audit.combination_count, 4)
        self.assertEqual(self.audit.complete_candidate_count, 2)
        self.assertAlmostEqual(self.audit.hybrid_fraction, 0.5)

    def test_crossed_components_prefer_unsupported_risky_action(self) -> None:
        hybrids = [
            row for row in self.audit.combinations if not row.is_complete_candidate
        ]
        self.assertEqual(len(hybrids), 2)
        for row in hybrids:
            self.assertEqual(row.ranking[0], "risky")
            self.assertFalse(row.ranking_supported_by_complete_candidate)
            self.assertFalse(row.top_action_supported_by_complete_candidate)

    def test_exact_support_violation_rates_are_one_half(self) -> None:
        self.assertEqual(self.audit.unsupported_ranking_count, 2)
        self.assertEqual(self.audit.unsupported_top_action_count, 2)
        self.assertAlmostEqual(self.audit.ranking_support_violation_rate, 0.5)
        self.assertAlmostEqual(self.audit.unsupported_top_action_rate, 0.5)

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit_componentwise_mixing(
                (self.candidates[0], self.candidates[0])
            )

    def test_uniform_independent_component_indices_expand_support(self) -> None:
        self.assertAlmostEqual(
            index_incoherent_probability((0.5, 0.5), component_count=2),
            0.5,
        )
        self.assertAlmostEqual(
            index_incoherent_probability((0.5, 0.5), component_count=3),
            0.75,
        )

    def test_index_probability_rejects_invalid_distribution(self) -> None:
        with self.assertRaises(ValueError):
            index_incoherent_probability((0.6, 0.6), component_count=2)


if __name__ == "__main__":
    unittest.main()
