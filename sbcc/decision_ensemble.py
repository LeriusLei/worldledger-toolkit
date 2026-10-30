"""Decision-layer committee over coherent executable world hypotheses.

This module deliberately starts *after* model induction. An adapter evaluates
each complete candidate world model under the same belief/history and supplies
per-action values. The committee never samples transition, observation, and
reward components independently: one row is one coherent hypothesis.

The API is intentionally independent of Pinductor so it can be unit-tested
without its large dependency stack. A Pinductor adapter only needs to convert
top ``RexNode`` candidates into ``CandidateEvaluation`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log2
from statistics import pstdev
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CandidateEvaluation:
    """A complete world hypothesis evaluated at one decision point."""

    candidate_id: str
    factual_score: float
    action_values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.action_values:
            raise ValueError("each candidate needs at least one action value")


@dataclass(frozen=True)
class ActionCertificate:
    action: Optional[str]
    reason: str
    lower_bound: Optional[float]
    competing_upper_bound: Optional[float]


class DecisionCommittee:
    """Near-best candidate set retained through action selection."""

    def __init__(
        self,
        candidates: Sequence[CandidateEvaluation],
        *,
        temperature: float = 0.1,
        near_best_std_multiplier: float = 1.0,
        action_tolerance: float = 1e-9,
    ) -> None:
        if not candidates:
            raise ValueError("committee requires at least one candidate")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if action_tolerance < 0:
            raise ValueError("action_tolerance must be non-negative")
        action_sets = {tuple(candidate.action_values) for candidate in candidates}
        if len(action_sets) != 1:
            raise ValueError("all candidates must evaluate the same ordered action set")

        scores = [float(candidate.factual_score) for candidate in candidates]
        score_std = pstdev(scores) if len(scores) > 1 else 0.0
        floor = max(scores) - near_best_std_multiplier * score_std
        self._candidates = tuple(
            candidate for candidate in candidates if candidate.factual_score >= floor
        )
        stable = [candidate.factual_score - max(c.factual_score for c in self._candidates) for candidate in self._candidates]
        raw_weights = [exp(score / temperature) for score in stable]
        total = sum(raw_weights)
        self._weights = tuple(weight / total for weight in raw_weights)
        self._actions = tuple(self._candidates[0].action_values)
        self._floor = floor
        self._action_tolerance = float(action_tolerance)

    @property
    def candidates(self) -> Tuple[CandidateEvaluation, ...]:
        return self._candidates

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            candidate.candidate_id: weight
            for candidate, weight in zip(self._candidates, self._weights)
        }

    @property
    def near_best_floor(self) -> float:
        return self._floor

    def model_entropy_bits(self) -> float:
        return -sum(weight * log2(weight) for weight in self._weights if weight > 0)

    def ranking(self, candidate: CandidateEvaluation) -> Tuple[str, ...]:
        return tuple(
            sorted(
                self._actions,
                key=lambda action: (-candidate.action_values[action], self._actions.index(action)),
            )
        )

    def decision_classes(self) -> Mapping[Tuple[str, ...], Tuple[str, ...]]:
        grouped: Dict[Tuple[str, ...], list[str]] = {}
        for candidate in self._candidates:
            grouped.setdefault(self.ranking(candidate), []).append(candidate.candidate_id)
        return {ranking: tuple(ids) for ranking, ids in grouped.items()}

    def action_interval(self, action: str) -> Tuple[float, float]:
        if action not in self._actions:
            raise ValueError(f"unknown action: {action}")
        values = [candidate.action_values[action] for candidate in self._candidates]
        return min(values), max(values)

    def weighted_action_value(self, action: str) -> float:
        if action not in self._actions:
            raise ValueError(f"unknown action: {action}")
        return sum(
            weight * candidate.action_values[action]
            for candidate, weight in zip(self._candidates, self._weights)
        )

    def weighted_action_values(self) -> Mapping[str, float]:
        return {action: self.weighted_action_value(action) for action in self._actions}

    def bma_action(self) -> str:
        return max(
            self._actions,
            key=lambda action: (self.weighted_action_value(action), -self._actions.index(action)),
        )

    def consensus_action(self) -> Optional[str]:
        singleton_best_actions = []
        for candidate in self._candidates:
            best = max(float(value) for value in candidate.action_values.values())
            best_actions = tuple(
                action
                for action in self._actions
                if best - float(candidate.action_values[action]) <= self._action_tolerance
            )
            if len(best_actions) != 1:
                return None
            singleton_best_actions.append(best_actions[0])
        unique = set(singleton_best_actions)
        return next(iter(unique)) if len(unique) == 1 else None

    def interval_certificate(self, margin: float = 0.0) -> ActionCertificate:
        """Certify an action whose worst case beats every rival's best case."""

        for action in self._actions:
            lower, _ = self.action_interval(action)
            rival_upper = max(
                self.action_interval(rival)[1]
                for rival in self._actions
                if rival != action
            ) if len(self._actions) > 1 else float("-inf")
            if lower > rival_upper + margin:
                return ActionCertificate(action, "interval_dominance", lower, rival_upper)

        consensus = self.consensus_action()
        if consensus is not None:
            lower, _ = self.action_interval(consensus)
            rival_upper = max(
                self.action_interval(rival)[1]
                for rival in self._actions
                if rival != consensus
            ) if len(self._actions) > 1 else float("-inf")
            return ActionCertificate(consensus, "ranking_consensus", lower, rival_upper)
        return ActionCertificate(None, "decision_ambiguous", None, None)

    def maximum_regret(self, action: str) -> float:
        if action not in self._actions:
            raise ValueError(f"unknown action: {action}")
        return max(
            max(candidate.action_values.values()) - candidate.action_values[action]
            for candidate in self._candidates
        )

    def maximum_regrets(self) -> Mapping[str, float]:
        return {action: self.maximum_regret(action) for action in self._actions}

    def minimax_regret_action(self) -> str:
        return min(
            self._actions,
            key=lambda action: (self.maximum_regret(action), self._actions.index(action)),
        )

    def weighted_decision_risk(self) -> float:
        return min(
            sum(
                weight
                * (max(candidate.action_values.values()) - candidate.action_values[action])
                for candidate, weight in zip(self._candidates, self._weights)
            )
            for action in self._actions
        )


def build_candidate_evaluations(
    candidate_scores: Mapping[str, float],
    action_values_by_candidate: Mapping[str, Mapping[str, float]],
) -> Tuple[CandidateEvaluation, ...]:
    """Strict adapter boundary shared by Pinductor and other model inducers."""

    if set(candidate_scores) != set(action_values_by_candidate):
        raise ValueError("scores and action values must refer to the same candidates")
    return tuple(
        CandidateEvaluation(candidate_id, candidate_scores[candidate_id], values)
        for candidate_id, values in action_values_by_candidate.items()
    )
