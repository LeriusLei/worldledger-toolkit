from __future__ import annotations

import unittest

from sbcc.candidate_ranking import (
    build_simple_baseline_predictions,
    episode_cluster_bootstrap_map_selection_regret,
    leave_one_generation_batch_out_selection_regret,
    pairwise_ranking_accuracy,
    score_action_predictions,
    summarize_committee_disagreement,
    summarize_score_accuracy_relation,
)


TRUTH = [
    {0: 1.0, 1: 0.5, 2: 0.0},
    {0: 0.0, 1: 1.0, 2: 0.2},
]
CORRECT = [dict(row) for row in TRUTH]
WRONG = [
    {0: 0.0, 1: 0.5, 2: 1.0},
    {0: 1.0, 1: 0.0, 2: 0.2},
]


class CandidateRankingTests(unittest.TestCase):
    def test_all_tied_prediction_gets_half_pairwise_credit_and_regret(self) -> None:
        tied = {0: 0.0, 1: 0.0, 2: 0.0}
        self.assertEqual(pairwise_ranking_accuracy(TRUTH[0], tied), 0.5)
        metrics = score_action_predictions([TRUTH[0]], [tied])
        self.assertEqual(metrics["uninformative_prediction_rate"], 1.0)
        self.assertGreater(metrics["mean_worst_tie_regret"], 0.0)

    def test_higher_factual_score_can_be_decisively_worse(self) -> None:
        correct_metrics = score_action_predictions(TRUTH, CORRECT)
        wrong_metrics = score_action_predictions(TRUTH, WRONG)
        relation = summarize_score_accuracy_relation(
            [
                {
                    "candidate_id": "high_score_wrong",
                    "factual_score": 0.9,
                    **wrong_metrics,
                },
                {
                    "candidate_id": "low_score_correct",
                    "factual_score": 0.7,
                    **correct_metrics,
                },
            ]
        )
        self.assertEqual(relation["score_accuracy_spearman"], -1.0)
        self.assertEqual(relation["decisive_inversion_count"], 1)
        self.assertEqual(relation["map_candidate_id"], "high_score_wrong")
        self.assertGreaterEqual(relation["map_selection_regret"], 0.05)
        self.assertTrue(relation["map_selection_failure_signal"])
        self.assertTrue(relation["screening_signal"])
        self.assertFalse(relation["publishable_signal"])

    def test_simple_baselines_and_disagreement_are_reported(self) -> None:
        baselines = build_simple_baseline_predictions(
            [WRONG, CORRECT], [0.9, 0.7]
        )
        self.assertEqual(
            set(baselines),
            {
                "map",
                "uniform_bma",
                "score_softmax_bma",
                "fractional_top_vote",
                "minimax",
            },
        )
        self.assertEqual(baselines["map"], WRONG)
        disagreement = summarize_committee_disagreement([WRONG, CORRECT])
        self.assertEqual(disagreement["top_set_disagreement_rate"], 1.0)
        self.assertEqual(disagreement["ranking_disagreement_rate"], 1.0)

    def test_episode_cluster_bootstrap_keeps_points_in_episode_together(self) -> None:
        truth = TRUTH + TRUTH
        wrong = WRONG + WRONG
        correct = CORRECT + CORRECT
        report = episode_cluster_bootstrap_map_selection_regret(
            truth,
            [wrong, correct],
            [0.9, 0.7],
            ["episode-a", "episode-a", "episode-b", "episode-b"],
            bootstrap_samples=200,
            seed=7,
        )
        self.assertGreaterEqual(report["observed_map_selection_regret"], 0.05)
        self.assertEqual(report["cluster_count"], 2)
        self.assertEqual(report["bootstrap_samples"], 200)
        self.assertEqual(report["bootstrap_probability_positive"], 1.0)

    def test_leave_one_batch_out_reports_generation_stability(self) -> None:
        summaries = [
            {
                "candidate_id": "wrong-a",
                "source_batch_index": 0,
                "factual_score": 0.9,
                "mean_pairwise_ranking_accuracy": 0.4,
            },
            {
                "candidate_id": "wrong-b",
                "source_batch_index": 1,
                "factual_score": 0.85,
                "mean_pairwise_ranking_accuracy": 0.5,
            },
            {
                "candidate_id": "good-a",
                "source_batch_index": 2,
                "factual_score": 0.7,
                "mean_pairwise_ranking_accuracy": 1.0,
            },
            {
                "candidate_id": "good-b",
                "source_batch_index": 2,
                "factual_score": 0.69,
                "mean_pairwise_ranking_accuracy": 0.95,
            },
        ]
        report = leave_one_generation_batch_out_selection_regret(summaries)
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(report["source_batch_count"], 3)
        self.assertGreater(report["maximum_leave_one_batch_out_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
