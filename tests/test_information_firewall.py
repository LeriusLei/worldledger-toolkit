from __future__ import annotations

import random
import unittest

import numpy as np

from sbcc.information_firewall import (
    EvaluatorOnlyChannel,
    paired_trace_firewall_check,
)


class InformationFirewallTests(unittest.TestCase):
    def test_shadow_result_is_not_returned_or_visible_before_seal(self) -> None:
        channel = EvaluatorOnlyChannel(lambda payload: payload["hidden"] * 2, enabled=True)
        returned = channel.evaluate({"hidden": 7})
        self.assertIsNone(returned)
        with self.assertRaisesRegex(RuntimeError, "before episode seal"):
            _ = channel.records
        self.assertEqual(channel.seal()[0].result, 14)

    def test_payload_mutation_and_rng_consumption_do_not_change_online_trace(self) -> None:
        def shadow_evaluator(payload):
            payload["position"] = 10_000
            random.random()
            np.random.random()
            return {"forked_truth": payload["position"]}

        def run_episode(channel):
            random.seed(17)
            np.random.seed(23)
            state = {"position": 0}
            trace = []
            for step in range(6):
                action = int(random.random() + np.random.random() > 1.0)
                self.assertIsNone(channel.evaluate(state))
                state["position"] += action
                reward_noise = round(random.random() + np.random.random(), 12)
                trace.append((step, action, state["position"], reward_noise))
            return trace

        report = paired_trace_firewall_check(run_episode, shadow_evaluator)
        self.assertTrue(report["trace_identical"])
        self.assertEqual(report["baseline_steps"], 6)
        self.assertEqual(report["shadow_evaluations"], 6)
        self.assertIsNone(report["first_mismatch_index"])

    def test_channel_rejects_evaluation_after_episode_seal(self) -> None:
        channel = EvaluatorOnlyChannel(lambda payload: payload, enabled=True)
        channel.seal()
        with self.assertRaisesRegex(RuntimeError, "after episode seal"):
            channel.evaluate({"hidden": 1})


if __name__ == "__main__":
    unittest.main()
