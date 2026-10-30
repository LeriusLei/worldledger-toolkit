"""Deployment-eligible decision policies over complete candidate worlds."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any, Callable, Mapping, Sequence

from .closed_loop_runner import PolicyDecision
from .decision_ensemble import CandidateEvaluation, DecisionCommittee


EvaluationProvider = Callable[
    [tuple[Any, ...], tuple[int, ...]], Sequence[CandidateEvaluation]
]


class CandidateCommitteePolicy:
    """MAP, BMA, minimax, coherent Thompson, or WorldLedger decisions.

    The provider sees only observation/action history.  It must evaluate every
    complete candidate under that same factual history; hidden evaluator state
    is intentionally absent from this interface.
    """

    MODES = {"map", "bma", "minimax", "coherent_thompson", "worldledger"}

    def __init__(
        self,
        *,
        evaluation_provider: EvaluationProvider,
        action_decoder: Mapping[str, int] | Callable[[str], int],
        mode: str,
        temperature: float = 0.1,
        near_best_std_multiplier: float = 1.0,
        certificate_margin: float = 0.0,
        fallback_mode: str = "minimax",
        tie_breaking: str = "first-v1",
        action_tolerance: float = 1e-9,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown committee policy mode: {mode}")
        if fallback_mode not in {"map", "bma", "minimax"}:
            raise ValueError("fallback_mode must be map, bma, or minimax")
        if certificate_margin < 0.0:
            raise ValueError("certificate_margin must be non-negative")
        if tie_breaking not in {"first-v1", "seeded-uniform-v1"}:
            raise ValueError("tie_breaking must be first-v1 or seeded-uniform-v1")
        if action_tolerance < 0.0:
            raise ValueError("action_tolerance must be non-negative")
        self.evaluation_provider = evaluation_provider
        self.action_decoder = action_decoder
        self.mode = mode
        self.temperature = float(temperature)
        self.near_best_std_multiplier = float(near_best_std_multiplier)
        self.certificate_margin = float(certificate_margin)
        self.fallback_mode = fallback_mode
        self.tie_breaking = tie_breaking
        self.action_tolerance = float(action_tolerance)
        self.observation_history: list[Any] = []
        self.action_history: list[int] = []
        self._rng = random.Random(0)
        self._thompson_candidate_id: str | None = None
        self._policy_seed = 0

    def reset(self, policy_seed: int) -> None:
        self.observation_history = []
        self.action_history = []
        self._rng = random.Random(policy_seed)
        self._thompson_candidate_id = None
        self._policy_seed = int(policy_seed)
        provider_reset = getattr(self.evaluation_provider, "reset", None)
        if callable(provider_reset):
            provider_reset(policy_seed)

    def _decode(self, action: str) -> int:
        value = (
            self.action_decoder(action)
            if callable(self.action_decoder)
            else self.action_decoder[action]
        )
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("action_decoder must produce integer actions")
        return value

    def _select_score_max(self, scores: Mapping[str, float], *, step: int) -> str:
        values = list(scores.items())
        if not values or any(not math.isfinite(float(value)) for _, value in values):
            raise ValueError("action scores must be non-empty and finite")
        best = max(float(value) for _, value in values)
        tied = [action for action, value in values if best - float(value) <= self.action_tolerance]
        if self.tie_breaking == "first-v1" or len(tied) == 1:
            return tied[0]
        stable = sorted(tied)
        material = f"{self._policy_seed}:{step}:{','.join(stable)}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(seed).choice(stable)

    def _candidate_action(self, candidate: CandidateEvaluation, *, step: int) -> str:
        return self._select_score_max(candidate.action_values, step=step)

    @staticmethod
    def _map_candidate(
        candidates: Sequence[CandidateEvaluation],
    ) -> CandidateEvaluation:
        return max(candidates, key=lambda candidate: float(candidate.factual_score))

    def _fallback_action(
        self,
        committee: DecisionCommittee,
        candidates: Sequence[CandidateEvaluation],
        *,
        step: int,
    ) -> str:
        if self.fallback_mode == "map":
            return self._candidate_action(self._map_candidate(candidates), step=step)
        if self.fallback_mode == "bma":
            return self._select_score_max(committee.weighted_action_values(), step=step)
        return self._select_score_max(
            {action: -regret for action, regret in committee.maximum_regrets().items()},
            step=step,
        )

    def decide(self, observation: Any, *, step: int) -> PolicyDecision:
        self.observation_history.append(copy.deepcopy(observation))
        candidates = tuple(
            self.evaluation_provider(
                tuple(copy.deepcopy(self.observation_history)),
                tuple(self.action_history),
            )
        )
        if not candidates:
            raise ValueError("evaluation_provider returned no candidates")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate ids must be unique at each decision")
        committee = DecisionCommittee(
            candidates,
            temperature=self.temperature,
            near_best_std_multiplier=self.near_best_std_multiplier,
            action_tolerance=self.action_tolerance,
        )
        committed = True
        reason = self.mode
        candidate_id = None
        certificate = None

        if self.mode == "map":
            selected = self._map_candidate(candidates)
            candidate_id = selected.candidate_id
            action_name = self._candidate_action(selected, step=step)
        elif self.mode == "bma":
            action_name = self._select_score_max(committee.weighted_action_values(), step=step)
        elif self.mode == "minimax":
            action_name = self._select_score_max(
                {action: -regret for action, regret in committee.maximum_regrets().items()},
                step=step,
            )
        elif self.mode == "coherent_thompson":
            if self._thompson_candidate_id is None:
                draw = self._rng.random()
                cumulative = 0.0
                for selected, weight in zip(committee.candidates, committee.weights.values()):
                    cumulative += weight
                    if draw <= cumulative:
                        self._thompson_candidate_id = selected.candidate_id
                        break
                if self._thompson_candidate_id is None:
                    self._thompson_candidate_id = committee.candidates[-1].candidate_id
            matches = [
                candidate
                for candidate in candidates
                if candidate.candidate_id == self._thompson_candidate_id
            ]
            if not matches:
                raise ValueError("episode-level Thompson candidate disappeared")
            candidate_id = self._thompson_candidate_id
            action_name = self._candidate_action(matches[0], step=step)
        else:
            certificate = committee.interval_certificate(self.certificate_margin)
            if certificate.action is None:
                action_name = self._fallback_action(committee, candidates, step=step)
                committed = False
                reason = f"abstain_fallback_{self.fallback_mode}"
            else:
                action_name = certificate.action
                reason = certificate.reason

        action = self._decode(action_name)
        self.action_history.append(action)
        diagnostics = {
            "mode": self.mode,
            "reason": reason,
            "action_name": action_name,
            "retained_candidate_ids": [
                candidate.candidate_id for candidate in committee.candidates
            ],
            "weights": dict(committee.weights),
            "model_entropy_bits": committee.model_entropy_bits(),
            "tie_breaking": self.tie_breaking,
            "action_tolerance": self.action_tolerance,
        }
        # A sparse-reward, short-horizon planner can return identical values
        # for every action. Expose this without silently changing tie-breaking
        # or treating it as evidence of confident, useful planning.
        action_ranges = {
            candidate.candidate_id: max(candidate.action_values.values())
            - min(candidate.action_values.values())
            for candidate in candidates
        }
        diagnostics["candidate_action_value_ranges"] = action_ranges
        diagnostics["all_candidates_action_values_tied"] = all(
            value <= self.action_tolerance for value in action_ranges.values()
        )
        diagnostics["action_value_tie_tolerance"] = self.action_tolerance
        if candidate_id is not None:
            diagnostics["selected_candidate_id"] = candidate_id
        if certificate is not None:
            diagnostics["certificate_lower_bound"] = certificate.lower_bound
            diagnostics["competing_upper_bound"] = certificate.competing_upper_bound
        return PolicyDecision(
            executed_action=action,
            committed_action=action if committed else None,
            diagnostics=diagnostics,
        )
