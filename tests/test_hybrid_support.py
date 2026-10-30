from __future__ import annotations

import unittest

from sbcc.hybrid_support import audit_hybrid_prediction_support


class HybridSupportTests(unittest.TestCase):
    def test_crossed_ranking_outside_complete_support_is_detected(self) -> None:
        complete = {
            "up": [{"safe": 1.0, "risky": 0.0}],
            "down": [{"safe": 1.0, "risky": 0.0}],
        }
        crossed = {
            ("up", "down"): [{"safe": 1.0, "risky": 2.0}],
            ("down", "up"): [{"safe": 1.0, "risky": 2.0}],
        }
        report = audit_hybrid_prediction_support(complete, crossed)
        self.assertEqual(report["unsupported_ranking_rate"], 1.0)
        self.assertEqual(report["unsupported_top_set_rate"], 1.0)
        self.assertEqual(report["disjoint_top_set_rate"], 1.0)
        self.assertEqual(report["introduced_top_action_rate"], 1.0)
        self.assertEqual(report["outside_complete_value_envelope_rate"], 1.0)

    def test_crossed_prediction_matching_a_complete_world_is_supported(self) -> None:
        complete = {
            "a": [{0: 1.0, 1: 0.0}],
            "b": [{0: 0.0, 1: 1.0}],
        }
        crossed = {("a", "b"): [{0: 0.0, 1: 1.0}]}
        report = audit_hybrid_prediction_support(complete, crossed)
        self.assertEqual(report["unsupported_ranking_rate"], 0.0)
        self.assertEqual(report["exact_complete_prediction_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
