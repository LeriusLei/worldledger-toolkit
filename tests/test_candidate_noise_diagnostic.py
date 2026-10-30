import unittest

from sbcc.candidate_noise_diagnostic import evaluate_candidate_noise


class CandidateNoiseDiagnosticTests(unittest.TestCase):
    def test_common_seed_removes_shared_pf_noise(self):
        def estimate(candidate, seed):
            return [seed * 0.001 + (0.5 if candidate == "b" else 0.0)]

        result = evaluate_candidate_noise(("a", "b"), estimate,
                                          pf_seeds=(11, 29))
        self.assertTrue(result["disagreement_resolved"])
        self.assertEqual(result["between_candidate_return_variance_by_plan"], [0.125])

    def test_noise_floor_blocks_candidate_label(self):
        def estimate(candidate, seed):
            return [float(seed) + (0.01 if candidate == "b" else 0.0)]

        result = evaluate_candidate_noise(("a", "b"), estimate,
                                          pf_seeds=(1, 3))
        self.assertFalse(result["disagreement_resolved"])
        self.assertEqual(result["label"], "unresolved_at_monte_carlo_noise_floor")

    def test_rejects_one_seed(self):
        with self.assertRaises(ValueError):
            evaluate_candidate_noise(("a", "b"), lambda _c, _s: [0.0], pf_seeds=(1,))


if __name__ == "__main__":
    unittest.main()
