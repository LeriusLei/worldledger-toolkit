"""Audit component-wise mixtures of complete executable world hypotheses.

The minimal counterexample in this module isolates a structural failure mode:
transition and reward components are each reasonable inside their originating
world, while a crossed combination recommends an action that no complete
candidate supports.  This is a set-closure diagnostic, not a claim that every
component ensemble will fail in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Hashable, Mapping, Sequence, Tuple

from .decision_ensemble import CandidateEvaluation


Transition = Callable[[str], Hashable]
Reward = Callable[[str, Hashable], float]


def index_incoherent_probability(
    candidate_probabilities: Sequence[float], *, component_count: int
) -> float:
    """Probability that independent component indices are not all identical.

    This is an index-level quantity. Shared or duplicate components can make an
    off-diagonal index tuple semantically equal to a supported complete world,
    so it is an upper envelope for semantic support violations, not their rate.
    """

    probabilities = [float(value) for value in candidate_probabilities]
    if component_count < 2:
        raise ValueError("component_count must be at least two")
    if not probabilities or any(value < 0.0 for value in probabilities):
        raise ValueError("candidate probabilities must be non-empty and non-negative")
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("candidate probabilities must sum to one")
    return 1.0 - sum(value**component_count for value in probabilities)


def rank_actions(
    action_values: Mapping[str, float], action_order: Sequence[str]
) -> Tuple[str, ...]:
    """Return a deterministic descending action ranking."""

    return tuple(
        sorted(
            action_order,
            key=lambda action: (-action_values[action], action_order.index(action)),
        )
    )


@dataclass(frozen=True)
class CompleteWorld:
    """A complete transition-reward hypothesis evaluated as one unit."""

    candidate_id: str
    actions: Tuple[str, ...]
    transition: Transition
    reward: Reward
    factual_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")

    def action_values(self) -> Mapping[str, float]:
        return {
            action: float(self.reward(action, self.transition(action)))
            for action in self.actions
        }

    def evaluation(self) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate_id=self.candidate_id,
            factual_score=self.factual_score,
            action_values=self.action_values(),
        )


@dataclass(frozen=True)
class ComponentCombination:
    """One transition/reward pairing considered by a component ensemble."""

    transition_candidate_id: str
    reward_candidate_id: str
    action_values: Mapping[str, float]
    ranking: Tuple[str, ...]
    is_complete_candidate: bool
    ranking_supported_by_complete_candidate: bool
    top_action_supported_by_complete_candidate: bool


@dataclass(frozen=True)
class HybridAudit:
    """Exact enumeration summary for independent component selection."""

    candidate_count: int
    combination_count: int
    complete_candidate_count: int
    unsupported_ranking_count: int
    unsupported_top_action_count: int
    combinations: Tuple[ComponentCombination, ...]

    @property
    def hybrid_fraction(self) -> float:
        return 1.0 - self.complete_candidate_count / self.combination_count

    @property
    def ranking_support_violation_rate(self) -> float:
        return self.unsupported_ranking_count / self.combination_count

    @property
    def unsupported_top_action_rate(self) -> float:
        return self.unsupported_top_action_count / self.combination_count


def audit_componentwise_mixing(candidates: Sequence[CompleteWorld]) -> HybridAudit:
    """Enumerate transition/reward pairs and compare them with candidate support."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be unique")
    action_orders = {candidate.actions for candidate in candidates}
    if len(action_orders) != 1:
        raise ValueError("all candidates must expose the same ordered actions")

    actions = candidates[0].actions
    candidate_rankings = {
        rank_actions(candidate.action_values(), actions) for candidate in candidates
    }
    candidate_top_actions = {ranking[0] for ranking in candidate_rankings}
    rows = []

    for transition_world, reward_world in product(candidates, repeat=2):
        values = {
            action: float(
                reward_world.reward(action, transition_world.transition(action))
            )
            for action in actions
        }
        ranking = rank_actions(values, actions)
        rows.append(
            ComponentCombination(
                transition_candidate_id=transition_world.candidate_id,
                reward_candidate_id=reward_world.candidate_id,
                action_values=values,
                ranking=ranking,
                is_complete_candidate=(
                    transition_world.candidate_id == reward_world.candidate_id
                ),
                ranking_supported_by_complete_candidate=ranking in candidate_rankings,
                top_action_supported_by_complete_candidate=(
                    ranking[0] in candidate_top_actions
                ),
            )
        )

    return HybridAudit(
        candidate_count=len(candidates),
        combination_count=len(rows),
        complete_candidate_count=sum(row.is_complete_candidate for row in rows),
        unsupported_ranking_count=sum(
            not row.ranking_supported_by_complete_candidate for row in rows
        ),
        unsupported_top_action_count=sum(
            not row.top_action_supported_by_complete_candidate for row in rows
        ),
        combinations=tuple(rows),
    )


def make_crossed_reward_counterexample() -> Tuple[CompleteWorld, CompleteWorld]:
    """Create two safe-preferring worlds whose crossed components prefer risk.

    In M_up, the risky action reaches ``high`` and the matching reward model
    treats ``high`` as bad.  In M_down it reaches ``low`` and the matching
    reward model treats ``low`` as bad.  Crossing either transition with the
    other reward model assigns value 2 to risk, although both complete worlds
    assign value 0 to risk and value 1 to safety.
    """

    actions = ("safe", "risky")

    def transition_up(action: str) -> str:
        return "safe_terminal" if action == "safe" else "high"

    def transition_down(action: str) -> str:
        return "safe_terminal" if action == "safe" else "low"

    def reward_up_world(action: str, next_state: Hashable) -> float:
        if action == "safe":
            return 1.0
        return 0.0 if next_state == "high" else 2.0

    def reward_down_world(action: str, next_state: Hashable) -> float:
        if action == "safe":
            return 1.0
        return 0.0 if next_state == "low" else 2.0

    return (
        CompleteWorld(
            "M_up",
            actions,
            transition_up,
            reward_up_world,
        ),
        CompleteWorld(
            "M_down",
            actions,
            transition_down,
            reward_down_world,
        ),
    )
