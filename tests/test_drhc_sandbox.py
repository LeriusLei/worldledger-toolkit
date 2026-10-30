"""Critical sandbox paths: safe behavior, escapes, limits, and taxonomy."""

import unittest

from worldledger.sandbox.executor import CandidateExecutionError, CandidateTimeout, run_candidate
from worldledger.sandbox.policy import SourceRejected, validate_source
from worldledger.sandbox.validator import classify_candidate


SAFE = """def transition(state, action):
    return (state[0] + 1, state[1]) if action == 'east' else state
def observation(state, action, next_state):
    return ('position', next_state[0], next_state[1])
def reward(state, action, next_state):
    return 1 if action == 'east' else 0
"""


class DrhcSandboxTests(unittest.TestCase):
    def test_repeatable_candidate_and_statuses(self):
        expected = ((2, 1), 1, ("position", 2, 1))
        self.assertEqual(run_candidate(SAFE, (1, 1), "east"), expected)
        self.assertEqual(run_candidate(SAFE, (1, 1), "east"), expected)
        case = {"state": [1, 1], "action": "wait", "next_state": [1, 1],
                "reward": 0, "observation": ["position", 1, 1]}
        probes = [{"state": [1, 1], "action": "east"}]
        behaviors, decisions = set(), set()
        self.assertEqual(classify_candidate(SAFE, [case], probes, behaviors, decisions,
                                            decision_signature=lambda _: "same-decision")["status"],
                         "valid_new_decision_class")
        self.assertEqual(classify_candidate(SAFE, [case], probes, behaviors, decisions,
                                            decision_signature=lambda _: "same-decision")["status"], "valid_duplicate")
        other = SAFE.replace("return 1 if", "return 2 if")
        self.assertEqual(classify_candidate(other, [case], probes, behaviors, decisions,
                                            decision_signature=lambda _: "same-decision")["status"],
                         "valid_new_behavior_class")
        third = SAFE.replace("return 1 if", "return 3 if")
        self.assertEqual(classify_candidate(third, [case], probes, behaviors, decisions,
                                            decision_signature=lambda _: "new-decision")["status"],
                         "valid_new_decision_class")
        wrong = dict(case, reward=1)
        self.assertEqual(classify_candidate(SAFE, [wrong], probes, behaviors, decisions)["status"], "history_inconsistent")

    def test_network_filesystem_subprocess_and_loops_rejected(self):
        attacks = ("import socket", "open('x')", "__import__('os')", "while True: pass",
                   "(1).__class__", "[x for x in state]", "eval('1')")
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(SourceRejected):
                validate_source(SAFE + "\n" + attack)
        with self.assertRaises(SourceRejected):
            validate_source(SAFE.replace("return 1 if", "return open('x') if"))

    def test_memory_time_and_runtime_failures(self):
        with self.assertRaises(CandidateExecutionError):
            run_candidate(SAFE, (1,) * 20000, "east")
        with self.assertRaises(CandidateTimeout):
            run_candidate(SAFE, (1, 1), "east", timeout_seconds=1e-12)
        broken = SAFE.replace("state[0] + 1", "state[99] + 1")
        self.assertEqual(classify_candidate(broken, [], [{"state": [1, 1], "action": "east"}], set(), set())["status"], "execution_failed")
        self.assertEqual(classify_candidate("def bad(:", [], [], set(), set())["status"], "syntax_failed")
        self.assertEqual(classify_candidate(SAFE, [], [{"state": [1, 1], "action": "east"}], set(), set(),
                                            timeout_seconds=1e-12)["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
