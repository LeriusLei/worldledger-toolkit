"""Sealed evaluator traces and pilot artifacts for paired closed-loop studies."""

from __future__ import annotations

import math
import re
import copy
from typing import Any, Callable, Mapping, Sequence

from .information_firewall import EvaluatorOnlyChannel


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class SealedEpisodeRecorder:
    """Record online actions while keeping fork truth inaccessible until seal."""

    def __init__(
        self,
        *,
        episode_id: int,
        environment_seed: int,
        policy_seed: int,
        evaluator: Callable[[Any], Mapping[str, Any]],
        evaluator_enabled: bool = True,
    ) -> None:
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (episode_id, environment_seed, policy_seed)
        ):
            raise ValueError("episode id and seeds must be integers")
        self._episode_id = episode_id
        self._environment_seed = environment_seed
        self._policy_seed = policy_seed
        self._channel = EvaluatorOnlyChannel(evaluator, enabled=evaluator_enabled)
        self._online_steps: list[dict[str, Any]] = []
        self._sealed = False
        self._ended = False

    def record_decision(
        self,
        *,
        state_digest: str,
        evaluator_payload: Any,
        executed_action: int,
        committed_action: int | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        policy_diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._sealed or self._ended:
            raise RuntimeError("cannot record after episode end or seal")
        if not _valid_digest(state_digest):
            raise ValueError("state_digest must be 64 lowercase hex chars")
        if not isinstance(executed_action, int) or isinstance(executed_action, bool):
            raise ValueError("executed_action must be an integer")
        if committed_action is not None and (
            not isinstance(committed_action, int)
            or isinstance(committed_action, bool)
            or committed_action != executed_action
        ):
            raise ValueError("a committed action must be the executed integer action")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise ValueError("termination flags must be booleans")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward):
            raise ValueError("reward must be finite")
        if policy_diagnostics is not None and not isinstance(
            policy_diagnostics, Mapping
        ):
            raise ValueError("policy_diagnostics must be a mapping or None")
        step = len(self._online_steps)
        returned = self._channel.evaluate(
            {
                "step": step,
                "state_digest": state_digest,
                "payload": evaluator_payload,
            }
        )
        if returned is not None:
            raise AssertionError("evaluator-only channel leaked an online value")
        self._online_steps.append(
            {
                "step": step,
                "state_digest": state_digest,
                "executed_action": executed_action,
                "committed_action": committed_action,
                "reward": numeric_reward,
                "terminated": terminated,
                "truncated": truncated,
                "policy_diagnostics": copy.deepcopy(
                    dict(policy_diagnostics or {})
                ),
            }
        )
        self._ended = terminated or truncated

    def seal(self) -> dict[str, Any]:
        if self._sealed:
            raise RuntimeError("episode recorder can only be sealed once")
        if not self._online_steps or not self._ended:
            raise RuntimeError("episode must end before evaluator truth is unlocked")
        records = self._channel.seal()
        self._sealed = True
        shadow_truth = []
        for record in records:
            result = record.result
            if not isinstance(result, Mapping):
                raise ValueError("shadow evaluator must return a mapping")
            shadow_truth.append(dict(result))
        return {
            "schema_version": 1,
            "episode_id": self._episode_id,
            "environment_seed": self._environment_seed,
            "policy_seed": self._policy_seed,
            "evaluation_mode": "seed-paired-independent-trajectories",
            "truth_unlocked_only_after_episode_seal": True,
            "shadow_evaluator_enabled": self._channel.enabled,
            "online_steps": self._online_steps,
            "shadow_truth": shadow_truth,
        }


def summarize_sealed_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Join sealed online/shadow records and compute one independent-unit row."""

    if episode.get("schema_version") != 1:
        raise ValueError("sealed episode schema_version must equal 1")
    if episode.get("evaluation_mode") != "seed-paired-independent-trajectories":
        raise ValueError("fixed-state replay cannot be used as closed-loop evidence")
    if episode.get("truth_unlocked_only_after_episode_seal") is not True:
        raise ValueError("truth must remain hidden until episode seal")
    if episode.get("shadow_evaluator_enabled", True) is not True:
        raise ValueError("disabled shadow traces are firewall diagnostics, not evidence")
    online = episode.get("online_steps")
    shadow = episode.get("shadow_truth")
    if not isinstance(online, list) or not online or not isinstance(shadow, list):
        raise ValueError("sealed episode requires online and shadow step lists")
    if len(online) != len(shadow):
        raise ValueError("online and shadow decision counts must align")
    wrong_commit = False
    committed = 0
    rewards = []
    for expected_step, (step, truth) in enumerate(zip(online, shadow)):
        if not isinstance(step, Mapping) or not isinstance(truth, Mapping):
            raise ValueError("online and shadow steps must be objects")
        if step.get("step") != expected_step or truth.get("step") != expected_step:
            raise ValueError("step indices must be consecutive and aligned")
        if step.get("state_digest") != truth.get("state_digest") or not _valid_digest(
            step.get("state_digest")
        ):
            raise ValueError("shadow truth must match the exact pre-action state digest")
        best_actions = truth.get("best_actions")
        if (
            not isinstance(best_actions, list)
            or not best_actions
            or any(not isinstance(action, int) or isinstance(action, bool) for action in best_actions)
        ):
            raise ValueError("truth best_actions must be a non-empty integer list")
        executed = step.get("executed_action")
        commitment = step.get("committed_action")
        if not isinstance(executed, int) or isinstance(executed, bool):
            raise ValueError("executed action must be an integer")
        if commitment is not None:
            if (
                not isinstance(commitment, int)
                or isinstance(commitment, bool)
                or commitment != executed
            ):
                raise ValueError("committed action must equal the executed action")
            committed += 1
            wrong_commit = wrong_commit or commitment not in set(best_actions)
        reward = float(step.get("reward"))
        if not math.isfinite(reward):
            raise ValueError("step reward must be finite")
        rewards.append(reward)
    if not bool(online[-1].get("terminated") or online[-1].get("truncated")):
        raise ValueError("last online step must terminate or truncate the episode")
    raw_return = sum(rewards)
    if not -1e-12 <= raw_return <= 1.0 + 1e-12:
        raise ValueError("binary-success normalized episode return must lie in [0,1]")
    return {
        "episode_id": episode.get("episode_id"),
        "environment_seed": episode.get("environment_seed"),
        "policy_seed": episode.get("policy_seed"),
        "wrong_commit": wrong_commit,
        "commitment_coverage": committed / len(online),
        "normalized_return": min(1.0, max(0.0, raw_return)),
        "decision_count": len(online),
        "committed_decision_count": committed,
    }


def _summarize_arm(
    arm: Mapping[str, Any],
    *,
    identifier_field: str,
    expected_episode_seeds: Mapping[int, tuple[int, int]],
    calibration_ids: Sequence[int],
    test_ids: Sequence[int],
) -> dict[str, Any]:
    identifier = arm.get(identifier_field)
    operating_point = arm.get("operating_point_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"arm requires {identifier_field}")
    if not isinstance(operating_point, str) or not operating_point:
        raise ValueError("arm requires a frozen operating_point_id")
    episodes_raw = arm.get("episodes")
    if not isinstance(episodes_raw, list) or len(episodes_raw) != len(
        expected_episode_seeds
    ):
        raise ValueError("each arm must contain every paired episode exactly once")
    rows = [summarize_sealed_episode(episode) for episode in episodes_raw]
    by_id = {row["episode_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected_episode_seeds):
        raise ValueError("arm episode ids must exactly match the frozen seed manifest")
    for episode_id, (environment_seed, policy_seed) in expected_episode_seeds.items():
        row = by_id[episode_id]
        if (
            row["environment_seed"] != environment_seed
            or row["policy_seed"] != policy_seed
        ):
            raise ValueError("arm seed does not match the frozen paired seed manifest")
    return {
        identifier_field: identifier,
        "operating_point_id": operating_point,
        "calibration_episodes": [by_id[index] for index in calibration_ids],
        "pilot_test_episodes": [by_id[index] for index in test_ids],
    }


def build_closed_loop_pilot_artifact(
    *,
    method_run: Mapping[str, Any],
    baseline_runs: Sequence[Mapping[str, Any]],
    heldout_summary: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_sha256: str,
    diagnostic_report: Mapping[str, Any],
    information_firewall_passed: bool,
) -> dict[str, Any]:
    """Build the exact pilot schema consumed by the preregistration freezer."""

    if heldout_summary.get("status") != "PASS" or heldout_summary.get(
        "dataset_role"
    ) != "pilot":
        raise ValueError("heldout seed manifest must be a passing pilot dataset")
    if heldout_summary.get("master_seed") != 20_260_830:
        raise ValueError("pilot seed manifest must use master seed 20260830")
    if heldout_summary.get("episodes") != 120:
        raise ValueError("main pilot protocol requires exactly 120 paired seeds")
    calibration_ids = [int(value) for value in heldout_summary.get("calibration_episode_ids", [])]
    test_ids = [int(value) for value in heldout_summary.get("test_episode_ids", [])]
    if len(calibration_ids) != 60 or len(test_ids) != 60 or set(calibration_ids) & set(test_ids):
        raise ValueError("pilot requires disjoint 60/60 calibration/test ids")
    seed_rows = heldout_summary.get("episode_seeds")
    if not isinstance(seed_rows, list) or len(seed_rows) != 120:
        raise ValueError("heldout summary requires 120 episode seed records")
    expected_seeds = {
        int(row["episode"]): (
            int(row["environment_seed"]),
            int(row["action_seed"]),
        )
        for row in seed_rows
    }
    if len(expected_seeds) != 120 or set(expected_seeds) != set(calibration_ids) | set(test_ids):
        raise ValueError("seed records must exactly cover the frozen split")
    if not _valid_digest(candidate_manifest_sha256):
        raise ValueError("candidate manifest requires a lowercase SHA-256")
    accepted = candidate_manifest.get("accepted_candidates")
    if (
        candidate_manifest.get("synthetic") is True
        or not isinstance(accepted, list)
        or candidate_manifest.get("manifest_kind")
        != "runtime_attested_candidate_pool"
        or candidate_manifest.get("runtime_audit_complete") is not True
        or candidate_manifest.get("replication_ready") is not True
    ):
        raise ValueError("candidate manifest must be a runtime-attested real pool")
    requested_count = candidate_manifest.get("runtime_requested_candidate_count")
    expected_diagnostic_sha = candidate_manifest.get(
        "source_static_manifest_sha256"
    )
    successful_count = diagnostic_report.get("successful_candidate_count")
    if (
        diagnostic_report.get("scientific_evidence") is not True
        or diagnostic_report.get("synthetic") is not False
        or diagnostic_report.get("status") != "PASS_SCREENING_SIGNAL"
        or diagnostic_report.get("manifest_sha256") != expected_diagnostic_sha
        or diagnostic_report.get("requested_candidate_count") != requested_count
        or not isinstance(successful_count, int)
        or isinstance(successful_count, bool)
        or not 8 <= successful_count <= requested_count
        or successful_count != len(accepted)
        or candidate_manifest.get("runtime_executable_candidate_count")
        != successful_count
    ):
        raise ValueError("real candidate diagnostic has not passed the surprise gate")
    if information_firewall_passed is not True:
        raise ValueError("closed-loop pilot requires a passing evaluator firewall")
    method = _summarize_arm(
        method_run,
        identifier_field="method_id",
        expected_episode_seeds=expected_seeds,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
    )
    baselines = []
    for run in baseline_runs:
        baseline = _summarize_arm(
            run,
            identifier_field="baseline_id",
            expected_episode_seeds=expected_seeds,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
        )
        if run.get("deployment_eligible") is not True:
            raise ValueError("all submitted baseline runs must be deployment eligible")
        baseline["deployment_eligible"] = True
        baselines.append(baseline)
    if not baselines:
        raise ValueError("at least one deployment-eligible baseline run is required")
    return {
        "schema_version": 1,
        "dataset_role": "pilot",
        "scientific_evidence": True,
        "synthetic": False,
        "environment": heldout_summary.get("environment"),
        "evaluation_mode": "seed-paired-independent-trajectories",
        "fixed_state_replay_used_for_closed_loop_claim": False,
        "pilot_master_seed": 20_260_830,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "source_static_manifest_sha256": expected_diagnostic_sha,
        "successful_candidate_count": successful_count,
        "surprise_gate_passed": True,
        "information_firewall_passed": True,
        "method": method,
        "baselines": baselines,
        "limitations": [
            "paired methods share episode seeds but may visit different states",
            "fork truth is joined only after each arm episode is sealed",
            "fixed random-policy replay remains diagnostic-only and is not closed-loop evidence",
        ],
    }
