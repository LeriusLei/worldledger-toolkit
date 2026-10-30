"""Frozen finite-family bridge checks, not LLM induction evidence."""

from dataclasses import replace
import unittest

from worldledger.benchmark.generators.inventory import generate_family_instance
from worldledger.benchmark.history_generator import generate_histories
from worldledger.sandbox.bridge import history_consistent, matching_models


HOLD_MODEL = """def transition(state, action):
    return (max(0, state[0] - 1), state[1])
def observation(state, action, next_state):
    return ('stock', next_state[0], next_state[1])
def reward(state, action, next_state):
    return -1
"""


class DrhcBridgeTests(unittest.TestCase):
    def test_history_and_exact_reachable_matching(self):
        instance = replace(generate_family_instance(0), allowed_actions=("hold",))
        history = generate_histories(generate_family_instance(0), instance.models[0])[0]
        self.assertTrue(history_consistent(instance, history, HOLD_MODEL))
        self.assertEqual(set(matching_models(instance, HOLD_MODEL, 5)),
                         {model.model_id for model in instance.models})

    def test_history_inconsistency_and_off_family_not_silently_dropped(self):
        instance = replace(generate_family_instance(0), allowed_actions=("hold",))
        history = generate_histories(generate_family_instance(0), instance.models[0])[0]
        wrong = HOLD_MODEL.replace("return -1", "return 0")
        self.assertFalse(history_consistent(instance, history, wrong))
        self.assertEqual(matching_models(instance, wrong, 5), ())


if __name__ == "__main__":
    unittest.main()
