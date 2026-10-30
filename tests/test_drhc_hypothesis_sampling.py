"""Oracle-blind complexity allocation and complete candidate accounting."""

import unittest

from worldledger.evaluator.baselines.hypothesis_coverage_sampling import (
    GeneratedCandidate, assigned_strata, select_budgeted_subset, source_stratum,
)


def source(branches: int) -> str:
    expression = "state"
    for index in range(branches):
        expression = f"state if action == 'a{index}' else ({expression})"
    return ("def transition(state, action):\n return " + expression + "\n"
            + "def observation(state, action, next_state):\n return ('x',)\n"
            + "def reward(state, action, next_state):\n return 0\n")


class CoverageSamplingTests(unittest.TestCase):
    def test_strata_and_balanced_selection_without_oracle(self):
        self.assertEqual(assigned_strata(3), ("0-1", "2-3", "4_or_more"))
        candidates = (GeneratedCandidate("c", source(4)), GeneratedCandidate("a", source(0)),
                      GeneratedCandidate("b", source(2)), GeneratedCandidate("extra", source(0)))
        result = select_budgeted_subset(3, candidates)
        self.assertEqual(result.selected_strata, ("0-1", "2-3", "4_or_more"))
        self.assertEqual(set(result.selected_candidate_ids), {"a", "b", "c"})
        self.assertEqual(result.unselected_candidate_ids, ("extra",))
        self.assertEqual(result.unfilled_slots, 0)

    def test_failures_and_duplicates_are_not_silently_replaced(self):
        result = select_budgeted_subset(3, (GeneratedCandidate("a", source(0)),
                                            GeneratedCandidate("failed", None),
                                            GeneratedCandidate("bad", "import socket")))
        self.assertEqual(result.selected_candidate_ids, ("a",))
        self.assertEqual(set(result.invalid_candidate_ids), {"failed", "bad"})
        self.assertEqual(result.unfilled_slots, 2)
        with self.assertRaises(ValueError):
            select_budgeted_subset(1, (GeneratedCandidate("same", source(0)),
                                       GeneratedCandidate("same", source(0))))

    def test_registered_three_slots_keep_off_stratum_valid_outputs(self):
        result = select_budgeted_subset(3, tuple(GeneratedCandidate(str(i), source(0)) for i in range(3)))
        self.assertEqual(set(result.selected_candidate_ids), {"0", "1", "2"})
        self.assertEqual(result.unselected_candidate_ids, ())
        self.assertEqual(result.unfilled_slots, 0)


if __name__ == "__main__":
    unittest.main()
