"""Observation-only, seed-paired runner for sealed closed-loop evaluation.

The online policy receives a deep copy of the observation and nothing else.
Hidden environment state is carried in a separate evaluator lane and can only
be joined to the online trace after an episode ends and is sealed.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .closed_loop_artifact import SealedEpisodeRecorder


@dataclass(frozen=True)
class ResetResult:
    observation: Any
    evaluator_state: Any


@dataclass(frozen=True)
class StepResult:
    observation: Any
    evaluator_state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    executed_action: int
    committed_action: int | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.executed_action, int) or isinstance(
            self.executed_action, bool
        ):
            raise ValueError("executed_action must be an integer")
        if self.committed_action is not None and (
            not isinstance(self.committed_action, int)
            or isinstance(self.committed_action, bool)
            or self.committed_action != self.executed_action
        ):
            raise ValueError("committed_action must equal executed_action or be None")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")


class ClosedLoopEnvironment(Protocol):
    def reset(self, seed: int) -> ResetResult:
        ...

    def step(self, action: int) -> StepResult:
        ...


class ObservationOnlyPolicy(Protocol):
    def reset(self, policy_seed: int) -> None:
        ...

    def decide(self, observation: Any, *, step: int) -> PolicyDecision:
        ...


@dataclass
class _RngLane:
    python_state: object
    numpy_state: tuple[Any, ...]

    @classmethod
    def seeded(cls, seed: int) -> "_RngLane":
        random.seed(seed)
        np.random.seed(seed % (2**32))
        return cls(random.getstate(), np.random.get_state())

    def enter(self) -> None:
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)

    def leave(self) -> None:
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()


def _preserve_global_rng(function: Callable[..., Any]) -> Callable[..., Any]:
    """Restore caller Python/NumPy RNG even when an episode raises."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    return wrapped


def _validate_seed(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


@_preserve_global_rng
def run_closed_loop_episode(
    *,
    episode_id: int,
    environment_seed: int,
    policy_seed: int,
    environment: ClosedLoopEnvironment,
    policy: ObservationOnlyPolicy,
    state_digest: Callable[[Any], str],
    shadow_evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    max_steps: int,
    evaluator_enabled: bool = True,
) -> dict[str, Any]:
    """Run one online episode with isolated Python/NumPy RNG lanes.

    ``shadow_evaluator`` receives only an evaluator payload containing the
    pre-action hidden state, executed action, and remaining horizon.  Its
    result is inaccessible to the policy and must provide ``best_actions``.
    """

    episode_id = _validate_seed(episode_id, "episode_id")
    environment_seed = _validate_seed(environment_seed, "environment_seed")
    policy_seed = _validate_seed(policy_seed, "policy_seed")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    environment_rng = _RngLane.seeded(environment_seed)
    environment_rng.enter()
    reset = environment.reset(environment_seed)
    if not isinstance(reset, ResetResult):
        raise TypeError("environment.reset must return ResetResult")
    environment_rng.leave()

    policy_rng = _RngLane.seeded(policy_seed)
    policy_rng.enter()
    policy.reset(policy_seed)
    policy_rng.leave()

    def sealed_evaluator(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
        result = shadow_evaluator(envelope["payload"])
        if not isinstance(result, Mapping):
            raise ValueError("shadow_evaluator must return a mapping")
        if "step" in result or "state_digest" in result:
            raise ValueError("shadow_evaluator cannot override alignment fields")
        return {
            "step": envelope["step"],
            "state_digest": envelope["state_digest"],
            **dict(result),
        }

    recorder = SealedEpisodeRecorder(
        episode_id=episode_id,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        evaluator=sealed_evaluator,
        evaluator_enabled=evaluator_enabled,
    )
    observation = reset.observation
    evaluator_state = reset.evaluator_state
    for step in range(max_steps):
        policy_rng.enter()
        decision = policy.decide(copy.deepcopy(observation), step=step)
        policy_rng.leave()
        if not isinstance(decision, PolicyDecision):
            raise TypeError("policy.decide must return PolicyDecision")

        pre_action_state = copy.deepcopy(evaluator_state)
        pre_action_digest = state_digest(pre_action_state)
        environment_rng.enter()
        transition = environment.step(decision.executed_action)
        if not isinstance(transition, StepResult):
            raise TypeError("environment.step must return StepResult")
        environment_rng.leave()
        forced_truncation = step + 1 == max_steps and not transition.terminated
        truncated = bool(transition.truncated or forced_truncation)
        recorder.record_decision(
            state_digest=pre_action_digest,
            evaluator_payload={
                "evaluator_state": pre_action_state,
                "executed_action": decision.executed_action,
                "remaining_horizon": max_steps - step,
            },
            executed_action=decision.executed_action,
            committed_action=decision.committed_action,
            reward=transition.reward,
            terminated=bool(transition.terminated),
            truncated=truncated,
            policy_diagnostics=decision.diagnostics,
        )
        observation = transition.observation
        evaluator_state = transition.evaluator_state
        if transition.terminated or truncated:
            return recorder.seal()
    raise AssertionError("max_steps truncation should have ended the episode")


def audit_seed_paired_firewall(
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    environment_factory: Callable[[], ClosedLoopEnvironment],
    policy_factory: Callable[[], ObservationOnlyPolicy],
    state_digest: Callable[[Any], str],
    shadow_evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    """Compare evaluator-disabled/enabled online traces on frozen seeds."""

    rows = []
    for row in seed_rows:
        kwargs = {
            "episode_id": _validate_seed(row.get("episode"), "episode"),
            "environment_seed": _validate_seed(
                row.get("environment_seed"), "environment_seed"
            ),
            "policy_seed": _validate_seed(row.get("action_seed"), "action_seed"),
            "state_digest": state_digest,
            "shadow_evaluator": shadow_evaluator,
            "max_steps": max_steps,
        }
        disabled = run_closed_loop_episode(
            **kwargs,
            environment=environment_factory(),
            policy=policy_factory(),
            evaluator_enabled=False,
        )
        enabled = run_closed_loop_episode(
            **kwargs,
            environment=environment_factory(),
            policy=policy_factory(),
            evaluator_enabled=True,
        )
        identical = disabled["online_steps"] == enabled["online_steps"]
        rows.append(
            {
                "episode": kwargs["episode_id"],
                "environment_seed": kwargs["environment_seed"],
                "policy_seed": kwargs["policy_seed"],
                "trace_identical": identical,
                "disabled_shadow_records": len(disabled["shadow_truth"]),
                "enabled_shadow_records": len(enabled["shadow_truth"]),
                "decision_count": len(enabled["online_steps"]),
            }
        )
    return {
        "schema_version": 1,
        "firewall_seed_count": len(rows),
        "all_paired_traces_identical": bool(rows)
        and all(row["trace_identical"] for row in rows),
        "all_disabled_runs_have_zero_shadow_records": all(
            row["disabled_shadow_records"] == 0 for row in rows
        ),
        "all_enabled_runs_have_one_shadow_record_per_decision": all(
            row["enabled_shadow_records"] == row["decision_count"] for row in rows
        ),
        "evaluation_comparison": "same policy/environment seeds; shadow disabled versus enabled",
        "episodes": rows,
    }


def make_fork_oracle_shadow_evaluator(oracle: Any) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Convert :class:`FiniteHorizonForkOracle` into the sealed runner API."""

    def evaluate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        decision = oracle.evaluate(
            payload["evaluator_state"], horizon=int(payload["remaining_horizon"])
        )
        return {
            "best_actions": [int(action) for action in decision.best_actions],
            "action_values": {
                str(entry.action): float(entry.value)
                for entry in decision.action_values
            },
            "value_margin": float(decision.value_margin),
        }

    return evaluate


def pinductor_state_digest(state: Any) -> str:
    """Stable SHA-256 for the evaluator-side structural MiniGrid state."""

    digest = hashlib.sha256()
    digest.update(state.grid.tobytes())
    digest.update(
        json.dumps([int(value) for value in state.agent_pos]).encode("ascii")
    )
    digest.update(str(int(state.agent_dir)).encode("ascii"))
    digest.update(str(state.carrying).encode("ascii"))
    return digest.hexdigest()


def run_seed_paired_arm(
    *,
    identifier: str,
    identifier_field: str,
    operating_point_id: str,
    seed_rows: Sequence[Mapping[str, Any]],
    environment_factory: Callable[[], ClosedLoopEnvironment],
    policy_factory: Callable[[], ObservationOnlyPolicy],
    state_digest: Callable[[Any], str],
    shadow_evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    max_steps: int,
    deployment_eligible: bool | None = None,
) -> dict[str, Any]:
    """Run every frozen seed independently and emit the arm artifact schema."""

    if identifier_field not in {"method_id", "baseline_id"}:
        raise ValueError("identifier_field must be method_id or baseline_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be non-empty")
    if not isinstance(operating_point_id, str) or not operating_point_id:
        raise ValueError("operating_point_id must be non-empty")
    ids = [row.get("episode") for row in seed_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("seed_rows must contain unique episode ids")

    episodes = []
    for row in seed_rows:
        episodes.append(
            run_closed_loop_episode(
                episode_id=_validate_seed(row.get("episode"), "episode"),
                environment_seed=_validate_seed(
                    row.get("environment_seed"), "environment_seed"
                ),
                policy_seed=_validate_seed(row.get("action_seed"), "action_seed"),
                environment=environment_factory(),
                policy=policy_factory(),
                state_digest=state_digest,
                shadow_evaluator=shadow_evaluator,
                max_steps=max_steps,
            )
        )
    artifact: dict[str, Any] = {
        identifier_field: identifier,
        "operating_point_id": operating_point_id,
        "episodes": episodes,
    }
    if identifier_field == "baseline_id":
        if deployment_eligible is not True:
            raise ValueError("baseline arms must be explicitly deployment eligible")
        artifact["deployment_eligible"] = True
    elif deployment_eligible is not None:
        raise ValueError("deployment_eligible is only valid for baseline arms")
    return artifact


class PinductorEnvironmentAdapter:
    """Adapter for Pinductor's state-returning MiniGrid wrapper.

    Pinductor's current partial-observation policy expects ``None`` before the
    first action.  Set ``expose_reset_observation=True`` only for a policy that
    was explicitly written to consume the reset observation.
    """

    def __init__(self, environment: Any, *, expose_reset_observation: bool = False):
        self.environment = environment
        self.expose_reset_observation = bool(expose_reset_observation)

    def reset(self, seed: int) -> ResetResult:
        state = self.environment.reset(seed=seed)
        observation = (
            self.environment.get_last_obs() if self.expose_reset_observation else None
        )
        return ResetResult(observation=observation, evaluator_state=state)

    def step(self, action: int) -> StepResult:
        observation, state, reward, terminated, truncated, info = (
            self.environment.step(action)
        )
        return StepResult(
            observation=observation,
            evaluator_state=state,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )


class PinductorObservationPolicyAdapter:
    """Expose a frozen partially-observed Pinductor agent as an online policy."""

    def __init__(self, agent: Any, *, commit_every_action: bool = True):
        if bool(getattr(agent, "fully_obs", True)):
            raise ValueError("closed-loop evidence requires a partially observed agent")
        self.agent = agent
        self.commit_every_action = bool(commit_every_action)

    def reset(self, policy_seed: int) -> None:
        self.agent.reset()

    def decide(self, observation: Any, *, step: int) -> PolicyDecision:
        action = int(self.agent.get_next_action(observation))
        return PolicyDecision(
            executed_action=action,
            committed_action=action if self.commit_every_action else None,
            diagnostics={"adapter": "pinductor_observation_only", "step": step},
        )
