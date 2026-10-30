from __future__ import annotations

import unittest

from sbcc.refinement_audit import summarize_refinement_edges


def candidate(
    source: str,
    code_hash: str,
    score: float,
    *,
    parent: str | None = None,
) -> dict:
    return {
        "source_batch_index": 0,
        "source_path": source,
        "resolved_parent_source_path": parent,
        "code_sha256": code_hash,
        "score": {"train_coverage": score, "test_coverage": score},
    }


def worker(index: int, code_hash: str, accuracy: float, prediction_sha: str) -> dict:
    return {
        "candidate_index": index,
        "code_sha256": code_hash,
        "status": "success",
        "prediction_sha256": prediction_sha,
        "metrics": {"mean_pairwise_ranking_accuracy": accuracy},
    }


class RefinementAuditTests(unittest.TestCase):
    def test_score_improvement_can_be_decision_null(self) -> None:
        candidates = [
            candidate("parent.txt", "a" * 64, -2.0),
            candidate("child.txt", "b" * 64, -1.0, parent="parent.txt"),
        ]
        workers = [
            worker(0, "a" * 64, 0.8, "c" * 64),
            worker(1, "b" * 64, 0.8, "c" * 64),
        ]
        report = summarize_refinement_edges(candidates, workers)
        self.assertEqual(report["runtime_comparable_edge_count"], 1)
        self.assertEqual(report["score_improving_edge_count"], 1)
        self.assertEqual(report["decision_null_refinement_rate"], 1.0)
        self.assertEqual(report["score_improving_decision_null_rate"], 1.0)

    def test_score_improvement_can_regress_counterfactual_utility(self) -> None:
        candidates = [
            candidate("parent.txt", "a" * 64, 0.1),
            candidate("child.txt", "b" * 64, 0.2, parent="parent.txt"),
        ]
        workers = [
            worker(0, "a" * 64, 0.9, "c" * 64),
            worker(1, "b" * 64, 0.7, "d" * 64),
        ]
        report = summarize_refinement_edges(candidates, workers)
        self.assertEqual(report["score_improving_utility_regression_rate"], 1.0)
        self.assertFalse(report["edges"][0]["decision_null_refinement"])

    def test_worker_hash_mismatch_fails_closed(self) -> None:
        candidates = [
            candidate("parent.txt", "a" * 64, 0.1),
            candidate("child.txt", "b" * 64, 0.2, parent="parent.txt"),
        ]
        workers = [
            worker(0, "e" * 64, 0.9, "c" * 64),
            worker(1, "b" * 64, 0.7, "d" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "parent worker code hash"):
            summarize_refinement_edges(candidates, workers)


if __name__ == "__main__":
    unittest.main()
