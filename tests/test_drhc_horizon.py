"""Nonanticipative horizon controls, separate from historical v3.1 tests."""

import json
from pathlib import Path
import unittest

from worldledger.benchmark.model_family import Instance, four_models
from worldledger.evaluator.horizon import choose_policy, policy_risk


ROOT = Path(__file__).resolve().parents[1]


def hand_constructed_grid() -> Instance:
    config = {
        "width": 4, "height": 4, "start": [1, 1], "walls": [[3, 1], [3, 0]],
        "gate_a_action": "east", "gate_b_action": "west",
        "gate_a_kind": "portal", "gate_b_kind": "portal",
        "portal_target": [3, 3], "blocked_target": [0, 0],
        "reward_a": 12, "reward_b": 9, "blocked_reward": -4,
        "move_reward": -1, "wait_reward": -1, "probe_cost": -2,
        "sensor_mask_a": 0, "sensor_mask_b": 1,
        "reward_function": "procedural-gate-v2",
        "observation_function": "position-and-active-sensor-v1",
    }
    return Instance("hand-horizon", 0, "grid", four_models("hand-horizon"),
                    (((1, 1), 1),), config,
                    ("east", "north", "west", "south", "wait", "probe_a", "probe_b"),
                    ((1, 1), (0, 1), (1, 2)))


class DrhcHorizonTests(unittest.TestCase):
    def test_h1_and_h5_repeat_exactly(self):
        instance = hand_constructed_grid()
        support = {m.model_id for m in instance.models}
        for horizon in (1, 5):
            first = choose_policy(instance, support, horizon)
            second = choose_policy(instance, support, horizon)
            self.assertEqual(first, second)
            self.assertEqual(policy_risk(instance, support, first, horizon),
                             policy_risk(instance, support, second, horizon))

    def test_hand_constructed_short_and_long_actions_differ(self):
        instance = hand_constructed_grid()
        support = {m.model_id for m in instance.models}
        self.assertEqual(choose_policy(instance, support, 1).action, "east")
        self.assertEqual(choose_policy(instance, support, 5).action, "north")

    def test_horizon_contract_is_frozen_without_rewriting_v31(self):
        config = json.loads((ROOT / "worldledger/protocols/v32/horizon_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["primary_horizon"], config["secondary_horizon"], config["optional_horizon"]), (5, 1, 10))
        self.assertEqual(config["discount_factor"], 1)
        self.assertEqual(config["off_support_branch"], "first registered action at every remaining step")


if __name__ == "__main__":
    unittest.main()
