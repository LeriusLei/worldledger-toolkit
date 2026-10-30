from __future__ import annotations

import hashlib
import unittest

from sbcc.closed_loop_artifact import (
    SealedEpisodeRecorder,
    build_closed_loop_pilot_artifact,
    summarize_sealed_episode,
)


SHA = "a" * 64
STATIC_SHA = "b" * 64


def attested_manifest(*, successful: int = 8, requested: int = 8) -> dict:
    return {
        "synthetic": False,
        "manifest_kind": "runtime_attested_candidate_pool",
        "runtime_audit_complete": True,
        "replication_ready": successful >= 8,
        "source_static_manifest_sha256": STATIC_SHA,
        "runtime_requested_candidate_count": requested,
        "runtime_executable_candidate_count": successful,
        "accepted_candidates": [{}] * successful,
    }


def passing_diagnostic(*, successful: int = 8, requested: int = 8) -> dict:
    return {
        "scientific_evidence": True,
        "synthetic": False,
        "status": "PASS_SCREENING_SIGNAL",
        "manifest_sha256": STATIC_SHA,
        "requested_candidate_count": requested,
        "successful_candidate_count": successful,
    }


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sealed_episode(
    episode_id: int,
    environment_seed: int,
    policy_seed: int,
    *,
    committed: bool,
    wrong: bool,
    reward: float,
) -> dict:
    state = digest(f"state-{episode_id}-{environment_seed}-{policy_seed}")
    executed = 1 if wrong or not committed else 0
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "evaluation_mode": "seed-paired-independent-trajectories",
        "truth_unlocked_only_after_episode_seal": True,
        "online_steps": [
            {
                "step": 0,
                "state_digest": state,
                "executed_action": executed,
                "committed_action": executed if committed else None,
                "reward": reward,
                "terminated": True,
                "truncated": False,
            }
        ],
        "shadow_truth": [
            {"step": 0, "state_digest": state, "best_actions": [0]}
        ],
    }


def heldout_summary() -> dict:
    return {
        "status": "PASS",
        "dataset_role": "pilot",
        "environment": "lava",
        "master_seed": 20_260_830,
        "episodes": 120,
        "calibration_episode_ids": list(range(60)),
        "test_episode_ids": list(range(60, 120)),
        "episode_seeds": [
            {
                "episode": index,
                "environment_seed": 1_000 + index,
                "action_seed": 2_000 + index,
            }
            for index in range(120)
        ],
    }


def arm(identifier: str, *, field: str, wrong_test: int) -> dict:
    episodes = []
    for index in range(120):
        within_split = index if index < 60 else index - 60
        committed = within_split < 48
        wrong = index >= 60 and within_split < wrong_test
        episodes.append(
            sealed_episode(
                index,
                1_000 + index,
                2_000 + index,
                committed=committed,
                wrong=wrong,
                reward=1.0 if within_split < 42 else 0.0,
            )
        )
    return {
        field: identifier,
        "operating_point_id": f"{identifier}-coverage-0.8",
        "episodes": episodes,
    }


def build() -> dict:
    baseline = arm("score-bma", field="baseline_id", wrong_test=12)
    baseline["deployment_eligible"] = True
    return build_closed_loop_pilot_artifact(
        method_run=arm("worldledger-v1", field="method_id", wrong_test=0),
        baseline_runs=[baseline],
        heldout_summary=heldout_summary(),
        candidate_manifest=attested_manifest(),
        candidate_manifest_sha256=SHA,
        diagnostic_report=passing_diagnostic(),
        information_firewall_passed=True,
    )


class ClosedLoopArtifactTests(unittest.TestCase):
    def test_recorder_unlocks_truth_only_after_episode_end_and_seal(self) -> None:
        state = digest("recorder-state")

        def evaluator(payload):
            return {
                "step": payload["step"],
                "state_digest": payload["state_digest"],
                "best_actions": [0],
            }

        recorder = SealedEpisodeRecorder(
            episode_id=0,
            environment_seed=1,
            policy_seed=2,
            evaluator=evaluator,
        )
        returned = recorder.record_decision(
            state_digest=state,
            evaluator_payload={"hidden_state": [1, 2, 3]},
            executed_action=0,
            committed_action=0,
            reward=1.0,
            terminated=True,
            truncated=False,
        )
        self.assertIsNone(returned)
        summary = summarize_sealed_episode(recorder.seal())
        self.assertFalse(summary["wrong_commit"])
        self.assertEqual(summary["commitment_coverage"], 1.0)

    def test_abstained_fallback_affects_return_but_not_wrong_commit(self) -> None:
        episode = sealed_episode(0, 1, 2, committed=False, wrong=True, reward=0.0)
        summary = summarize_sealed_episode(episode)
        self.assertFalse(summary["wrong_commit"])
        self.assertEqual(summary["commitment_coverage"], 0.0)
        self.assertEqual(summary["normalized_return"], 0.0)

    def test_wrong_commit_and_shadow_alignment_are_checked(self) -> None:
        episode = sealed_episode(0, 1, 2, committed=True, wrong=True, reward=0.0)
        self.assertTrue(summarize_sealed_episode(episode)["wrong_commit"])
        episode["shadow_truth"][0]["state_digest"] = digest("different")
        with self.assertRaises(ValueError):
            summarize_sealed_episode(episode)

    def test_fixed_state_replay_is_rejected_as_closed_loop_evidence(self) -> None:
        episode = sealed_episode(0, 1, 2, committed=True, wrong=False, reward=1.0)
        episode["evaluation_mode"] = "fixed-state-replay"
        with self.assertRaises(ValueError):
            summarize_sealed_episode(episode)

    def test_disabled_shadow_trace_is_firewall_only_not_evidence(self) -> None:
        episode = sealed_episode(0, 1, 2, committed=True, wrong=False, reward=1.0)
        episode["shadow_evaluator_enabled"] = False
        episode["shadow_truth"] = []
        with self.assertRaisesRegex(ValueError, "firewall diagnostics"):
            summarize_sealed_episode(episode)


    def test_builder_rejects_seed_drift_synthetic_diagnostic_and_failed_firewall(self) -> None:
        pilot_summary = heldout_summary()
        method = arm("worldledger-v1", field="method_id", wrong_test=0)
        baseline = arm("score-bma", field="baseline_id", wrong_test=12)
        baseline["deployment_eligible"] = True
        method["episodes"][0]["environment_seed"] += 1
        arguments = {
            "method_run": method,
            "baseline_runs": [baseline],
            "heldout_summary": pilot_summary,
            "candidate_manifest": attested_manifest(),
            "candidate_manifest_sha256": SHA,
            "diagnostic_report": {
                "scientific_evidence": True,
                "synthetic": False,
                "status": "PASS_SCREENING_SIGNAL",
                "manifest_sha256": STATIC_SHA,
                "requested_candidate_count": 8,
                "successful_candidate_count": 8,
            },
            "information_firewall_passed": True,
        }
        with self.assertRaises(ValueError):
            build_closed_loop_pilot_artifact(**arguments)
        arguments["method_run"] = arm(
            "worldledger-v1", field="method_id", wrong_test=0
        )
        arguments["diagnostic_report"] = {
            "scientific_evidence": False,
            "synthetic": True,
            "status": "SYNTHETIC_PASS",
            "manifest_sha256": STATIC_SHA,
            "requested_candidate_count": 8,
            "successful_candidate_count": 8,
        }
        with self.assertRaises(ValueError):
            build_closed_loop_pilot_artifact(**arguments)
        arguments["diagnostic_report"] = {
            "scientific_evidence": True,
            "synthetic": False,
            "status": "PASS_SCREENING_SIGNAL",
            "manifest_sha256": STATIC_SHA,
            "requested_candidate_count": 8,
            "successful_candidate_count": 8,
        }
        arguments["information_firewall_passed"] = False
        with self.assertRaises(ValueError):
            build_closed_loop_pilot_artifact(**arguments)

    def test_builder_accepts_eight_successes_from_larger_bound_manifest(self) -> None:
        baseline = arm("score-bma", field="baseline_id", wrong_test=12)
        baseline["deployment_eligible"] = True
        arguments = {
            "method_run": arm("worldledger-v1", field="method_id", wrong_test=0),
            "baseline_runs": [baseline],
            "heldout_summary": heldout_summary(),
            "candidate_manifest": attested_manifest(successful=8, requested=12),
            "candidate_manifest_sha256": SHA,
            "diagnostic_report": {
                "scientific_evidence": True,
                "synthetic": False,
                "status": "PASS_SCREENING_SIGNAL",
                "manifest_sha256": STATIC_SHA,
                "requested_candidate_count": 12,
                "successful_candidate_count": 8,
            },
            "information_firewall_passed": True,
        }
        report = build_closed_loop_pilot_artifact(**arguments)
        self.assertEqual(report["successful_candidate_count"], 8)
        arguments["diagnostic_report"]["manifest_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            build_closed_loop_pilot_artifact(**arguments)


if __name__ == "__main__":
    unittest.main()
