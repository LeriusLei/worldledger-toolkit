from __future__ import annotations

import hashlib
import random
import unittest

import numpy as np

from sbcc.closed_loop_artifact import summarize_sealed_episode
from sbcc.closed_loop_runner import (
    PolicyDecision,
    PinductorEnvironmentAdapter,
    PinductorObservationPolicyAdapter,
    ResetResult,
    StepResult,
    audit_seed_paired_firewall,
    run_closed_loop_episode,
    run_seed_paired_arm,
)


def digest(state) -> str:
    return hashlib.sha256(str(state).encode("utf-8")).hexdigest()


class ToyEnvironment:
    def __init__(self):
        self.hidden = 0
        self.step_index = 0

    def reset(self, seed: int) -> ResetResult:
        self.hidden = seed % 3
        self.step_index = 0
        return ResetResult({"visible": 0}, {"hidden": self.hidden, "t": 0})

    def step(self, action: int) -> StepResult:
        self.hidden += action + 1
        self.step_index += 1
        ended = self.step_index == 2
        return StepResult(
            observation={"visible": self.step_index},
            evaluator_state={"hidden": self.hidden, "t": self.step_index},
            reward=1.0 if ended and self.hidden % 2 == 0 else 0.0,
            terminated=ended,
            truncated=False,
        )


class RecordingPolicy:
    def __init__(self, action: int):
        self.action = action
        self.seen = []

    def reset(self, policy_seed: int) -> None:
        self.seen = []

    def decide(self, observation, *, step: int) -> PolicyDecision:
        self.seen.append(observation)
        self.assert_observation_only(observation)
        return PolicyDecision(self.action, self.action, {"step": step})

    @staticmethod
    def assert_observation_only(observation) -> None:
        if isinstance(observation, dict) and "hidden" in observation:
            raise AssertionError("hidden state leaked into policy input")


def shadow(payload):
    state = payload["evaluator_state"]
    best = [0] if state["hidden"] % 2 == 0 else [1]
    return {"best_actions": best}


class ClosedLoopRunnerTests(unittest.TestCase):
    def test_policy_receives_observations_only_and_truth_seals(self) -> None:
        policy = RecordingPolicy(0)
        episode = run_closed_loop_episode(
            episode_id=0,
            environment_seed=10,
            policy_seed=20,
            environment=ToyEnvironment(),
            policy=policy,
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=4,
        )
        self.assertEqual(policy.seen, [{"visible": 0}, {"visible": 1}])
        self.assertTrue(episode["truth_unlocked_only_after_episode_seal"])
        self.assertEqual(len(episode["shadow_truth"]), 2)
        self.assertEqual(summarize_sealed_episode(episode)["decision_count"], 2)

    def test_shadow_rng_and_payload_mutation_cannot_change_online_trace(self) -> None:
        def noisy_shadow(payload):
            random.random()
            np.random.random()
            payload["evaluator_state"]["hidden"] = 999
            return {"best_actions": [0]}

        kwargs = dict(
            episode_id=1,
            environment_seed=31,
            policy_seed=41,
            state_digest=digest,
            max_steps=4,
        )
        quiet = run_closed_loop_episode(
            **kwargs,
            environment=ToyEnvironment(),
            policy=RecordingPolicy(0),
            shadow_evaluator=lambda payload: {"best_actions": [0]},
        )
        noisy = run_closed_loop_episode(
            **kwargs,
            environment=ToyEnvironment(),
            policy=RecordingPolicy(0),
            shadow_evaluator=noisy_shadow,
        )
        self.assertEqual(quiet["online_steps"], noisy["online_steps"])

    def test_caller_rng_is_restored(self) -> None:
        random.seed(7)
        np.random.seed(7)
        expected_py = random.random()
        expected_np = float(np.random.random())
        random.seed(7)
        np.random.seed(7)
        run_closed_loop_episode(
            episode_id=2,
            environment_seed=1,
            policy_seed=2,
            environment=ToyEnvironment(),
            policy=RecordingPolicy(0),
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=4,
        )
        self.assertEqual(random.random(), expected_py)
        self.assertEqual(float(np.random.random()), expected_np)

    def test_same_seeds_pair_episodes_but_policies_visit_own_states(self) -> None:
        kwargs = dict(
            episode_id=3,
            environment_seed=5,
            policy_seed=6,
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=4,
        )
        left = run_closed_loop_episode(
            **kwargs, environment=ToyEnvironment(), policy=RecordingPolicy(0)
        )
        right = run_closed_loop_episode(
            **kwargs, environment=ToyEnvironment(), policy=RecordingPolicy(1)
        )
        self.assertEqual(left["environment_seed"], right["environment_seed"])
        self.assertNotEqual(
            left["online_steps"][1]["state_digest"],
            right["online_steps"][1]["state_digest"],
        )

    def test_seed_paired_arm_matches_builder_input_schema(self) -> None:
        arm = run_seed_paired_arm(
            identifier="score-bma",
            identifier_field="baseline_id",
            operating_point_id="coverage-1",
            seed_rows=[
                {"episode": 0, "environment_seed": 10, "action_seed": 20},
                {"episode": 1, "environment_seed": 11, "action_seed": 21},
            ],
            environment_factory=ToyEnvironment,
            policy_factory=lambda: RecordingPolicy(0),
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=4,
            deployment_eligible=True,
        )
        self.assertEqual(arm["baseline_id"], "score-bma")
        self.assertTrue(arm["deployment_eligible"])
        self.assertEqual([row["episode_id"] for row in arm["episodes"]], [0, 1])

    def test_runner_forces_final_step_truncation(self) -> None:
        class EndlessEnvironment(ToyEnvironment):
            def step(self, action: int) -> StepResult:
                result = super().step(action)
                return StepResult(
                    result.observation,
                    result.evaluator_state,
                    0.0,
                    False,
                    False,
                )

        episode = run_closed_loop_episode(
            episode_id=4,
            environment_seed=1,
            policy_seed=2,
            environment=EndlessEnvironment(),
            policy=RecordingPolicy(0),
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=2,
        )
        self.assertTrue(episode["online_steps"][-1]["truncated"])

    def test_structural_pinductor_adapters(self) -> None:
        class FakeEnv:
            def reset(self, seed=0):
                self.last = {"obs": seed}
                return {"state": seed}

            def get_last_obs(self):
                return self.last

            def step(self, action):
                return {"obs": 1}, {"state": 1}, 1, True, False, {"x": 1}

        class FakeAgent:
            fully_obs = False

            def reset(self):
                self.did_reset = True

            def get_next_action(self, observation):
                return 2

        environment = PinductorEnvironmentAdapter(
            FakeEnv(), expose_reset_observation=True
        )
        self.assertEqual(environment.reset(9).observation, {"obs": 9})
        self.assertTrue(environment.step(2).terminated)
        policy = PinductorObservationPolicyAdapter(FakeAgent())
        policy.reset(3)
        self.assertEqual(policy.decide(None, step=0).executed_action, 2)

    def test_real_runner_firewall_compares_disabled_and_enabled_shadow(self) -> None:
        report = audit_seed_paired_firewall(
            seed_rows=[
                {"episode": 0, "environment_seed": 10, "action_seed": 20},
                {"episode": 1, "environment_seed": 11, "action_seed": 21},
            ],
            environment_factory=ToyEnvironment,
            policy_factory=lambda: RecordingPolicy(0),
            state_digest=digest,
            shadow_evaluator=shadow,
            max_steps=4,
        )
        self.assertTrue(report["all_paired_traces_identical"])
        self.assertTrue(report["all_disabled_runs_have_zero_shadow_records"])
        self.assertTrue(
            report["all_enabled_runs_have_one_shadow_record_per_decision"]
        )


if __name__ == "__main__":
    unittest.main()
