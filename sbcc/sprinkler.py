"""A minimal factually-indistinguishable irrigation decision problem.

The environment has two hidden factors:

``response_type``
    Whether watering or not watering is optimal. This defines the two
    decision-equivalence classes.

``phenotype``
    A high-cardinality nuisance variable. It is easy to measure and can remove
    many bits of model entropy, but it never changes the best irrigation action.

The logged factual history contains only a neutral safe action. Consequently,
every candidate hypothesis has exactly the same factual likelihood. The agent
may purchase one probe and then execute one irreversible action. A separate
``TwinEvaluator`` exposes counterfactual rewards for evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from random import Random
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ACTIONS: Tuple[str, ...] = ("no_water", "water", "safe")
PROBES: Tuple[str, ...] = ("root_zone", "phenotype")
PROBE_COST: Mapping[str, float] = {"root_zone": 0.05, "phenotype": 0.05}
SAFE_REWARD = 0.45


@dataclass(frozen=True, order=True)
class Hypothesis:
    """One executable world hypothesis."""

    response_type: int
    phenotype: int

    def __post_init__(self) -> None:
        if self.response_type not in (0, 1):
            raise ValueError("response_type must be 0 or 1")
        if self.phenotype < 0:
            raise ValueError("phenotype must be non-negative")

    @property
    def optimal_action(self) -> str:
        return "water" if self.response_type == 1 else "no_water"

    def reward(self, action: str) -> float:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if action == "safe":
            return SAFE_REWARD
        return 1.0 if action == self.optimal_action else 0.0

    def factual_log_likelihood(self, factual_history: Sequence[Tuple[str, str, float]]) -> float:
        """Score the neutral logged history.

        The benchmark deliberately logs only ``safe`` actions with the public
        observation ``normal`` and reward ``SAFE_REWARD``. These facts are
        generated identically by every hypothesis. Passing any other event is
        an error: it prevents accidental counterfactual leakage into fitting.
        """

        for action, observation, reward in factual_history:
            if action != "safe" or observation != "normal" or reward != SAFE_REWARD:
                raise ValueError(
                    "The candidate scorer accepts only the neutral factual history; "
                    "counterfactual or probe data must not be passed here."
                )
        return 0.0


def make_hypotheses(nuisance_cardinality: int) -> Tuple[Hypothesis, ...]:
    if nuisance_cardinality < 2:
        raise ValueError("nuisance_cardinality must be at least 2")
    return tuple(
        Hypothesis(response_type=response, phenotype=phenotype)
        for response in (0, 1)
        for phenotype in range(nuisance_cardinality)
    )


class Posterior:
    """Finite posterior over executable hypotheses."""

    def __init__(self, weights: Mapping[Hypothesis, float]):
        if not weights:
            raise ValueError("posterior needs at least one hypothesis")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("posterior weights must be non-negative")
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("posterior weights must have positive mass")
        self._weights: Dict[Hypothesis, float] = {
            hypothesis: float(weight) / total
            for hypothesis, weight in weights.items()
            if weight > 0
        }

    @classmethod
    def uniform(cls, hypotheses: Iterable[Hypothesis]) -> "Posterior":
        hypotheses = tuple(hypotheses)
        if not hypotheses:
            raise ValueError("cannot build a posterior over an empty hypothesis set")
        return cls({hypothesis: 1.0 for hypothesis in hypotheses})

    @property
    def weights(self) -> Mapping[Hypothesis, float]:
        return dict(self._weights)

    @property
    def support(self) -> Tuple[Hypothesis, ...]:
        return tuple(self._weights)

    def entropy_bits(self) -> float:
        return -sum(weight * log2(weight) for weight in self._weights.values())

    def decision_classes(self) -> Tuple[int, ...]:
        return tuple(sorted({hypothesis.response_type for hypothesis in self._weights}))

    def expected_reward(self, action: str) -> float:
        return sum(
            weight * hypothesis.reward(action)
            for hypothesis, weight in self._weights.items()
        )

    def worst_case_reward(self, action: str) -> float:
        return min(hypothesis.reward(action) for hypothesis in self._weights)

    def minimax_regret(self) -> float:
        return min(
            max(1.0 - hypothesis.reward(action) for hypothesis in self._weights)
            for action in ACTIONS
        )

    def _probe_likelihood(self, probe: str, observation: int, accuracy: float) -> Dict[Hypothesis, float]:
        if not 0.0 < accuracy <= 1.0:
            raise ValueError("probe accuracy must lie in (0, 1]")
        possible = self.possible_observations(probe)
        if observation not in possible:
            return {hypothesis: 0.0 for hypothesis in self._weights}
        wrong_probability = 0.0 if len(possible) == 1 else (1.0 - accuracy) / (len(possible) - 1)
        likelihoods: Dict[Hypothesis, float] = {}
        for hypothesis in self._weights:
            truth = hypothesis.response_type if probe == "root_zone" else hypothesis.phenotype
            likelihoods[hypothesis] = accuracy if truth == observation else wrong_probability
        return likelihoods

    def condition(self, probe: str, observation: int, accuracy: float = 1.0) -> "Posterior":
        likelihoods = self._probe_likelihood(probe, observation, accuracy)
        selected = {
            hypothesis: weight * likelihoods[hypothesis]
            for hypothesis, weight in self._weights.items()
            if weight * likelihoods[hypothesis] > 0
        }
        if not selected:
            raise ValueError(f"observation {observation} has zero posterior probability")
        return Posterior(selected)

    def possible_observations(self, probe: str) -> Tuple[int, ...]:
        if probe == "root_zone":
            return tuple(sorted({hypothesis.response_type for hypothesis in self._weights}))
        if probe == "phenotype":
            return tuple(sorted({hypothesis.phenotype for hypothesis in self._weights}))
        raise ValueError(f"unknown probe: {probe}")

    def observation_probability(self, probe: str, observation: int, accuracy: float = 1.0) -> float:
        likelihoods = self._probe_likelihood(probe, observation, accuracy)
        return sum(
            weight * likelihoods[hypothesis]
            for hypothesis, weight in self._weights.items()
        )

    def expected_entropy_after(self, probe: str, accuracy: float = 1.0) -> float:
        return sum(
            self.observation_probability(probe, observation, accuracy)
            * self.condition(probe, observation, accuracy).entropy_bits()
            for observation in self.possible_observations(probe)
        )

    def expected_minimax_regret_after(self, probe: str) -> float:
        return sum(
            self.observation_probability(probe, observation)
            * self.condition(probe, observation).minimax_regret()
            for observation in self.possible_observations(probe)
        )

    def decision_risk(self) -> float:
        """Posterior expected regret of the best available decision action."""

        return min(
            sum(
                weight * (1.0 - hypothesis.reward(action))
                for hypothesis, weight in self._weights.items()
            )
            for action in ACTIONS
        )

    def expected_decision_risk_after(self, probe: str, accuracy: float = 1.0) -> float:
        return sum(
            self.observation_probability(probe, observation, accuracy)
            * self.condition(probe, observation, accuracy).decision_risk()
            for observation in self.possible_observations(probe)
        )


class SprinklerWorld:
    """Execution interface visible to the agent.

    Hidden variables are intentionally private. Policies receive observations
    only through ``probe`` and rewards only through ``act``.
    """

    def __init__(
        self,
        true_model: Hypothesis,
        *,
        nuisance_cardinality: int = 16,
        root_zone_accuracy: float = 1.0,
        phenotype_accuracy: float = 1.0,
        probe_costs: Optional[Mapping[str, float]] = None,
        rng: Optional[Random] = None,
    ):
        if nuisance_cardinality < 2 or true_model.phenotype >= nuisance_cardinality:
            raise ValueError("nuisance cardinality is incompatible with the true phenotype")
        for accuracy in (root_zone_accuracy, phenotype_accuracy):
            if not 0.0 < accuracy <= 1.0:
                raise ValueError("probe accuracies must lie in (0, 1]")
        self.__true_model = true_model
        self._nuisance_cardinality = nuisance_cardinality
        self._probe_accuracy = {
            "root_zone": root_zone_accuracy,
            "phenotype": phenotype_accuracy,
        }
        self._probe_cost = dict(PROBE_COST if probe_costs is None else probe_costs)
        if set(self._probe_cost) != set(PROBES) or any(cost < 0 for cost in self._probe_cost.values()):
            raise ValueError("probe_costs must provide non-negative costs for both probes")
        self._rng = rng or Random(0)
        self._probe_used = False
        self._acted = False

    @staticmethod
    def factual_history() -> Tuple[Tuple[str, str, float], ...]:
        return (("safe", "normal", SAFE_REWARD),)

    def probe(self, probe: str) -> int:
        if self._acted:
            raise RuntimeError("cannot probe after the irreversible action")
        if self._probe_used:
            raise RuntimeError("the benchmark allows at most one probe")
        if probe not in PROBES:
            raise ValueError(f"unknown probe: {probe}")
        self._probe_used = True
        accuracy = self._probe_accuracy[probe]
        if probe == "root_zone":
            truth = self.__true_model.response_type
            if self._rng.random() <= accuracy:
                return truth
            return 1 - truth

        truth = self.__true_model.phenotype
        if self._rng.random() <= accuracy:
            return truth
        alternatives = [value for value in range(self._nuisance_cardinality) if value != truth]
        return self._rng.choice(alternatives)

    def act(self, action: str) -> float:
        if self._acted:
            raise RuntimeError("only one irreversible action is allowed")
        self._acted = True
        return self.__true_model.reward(action)

    @property
    def probe_used(self) -> bool:
        return self._probe_used

    def probe_accuracy(self, probe: str) -> float:
        if probe not in PROBES:
            raise ValueError(f"unknown probe: {probe}")
        return self._probe_accuracy[probe]

    def probe_cost(self, probe: str) -> float:
        if probe not in PROBES:
            raise ValueError(f"unknown probe: {probe}")
        return self._probe_cost[probe]


class TwinEvaluator:
    """Privileged counterfactual evaluator, constructed after an episode.

    This class is deliberately separate from ``SprinklerWorld``. The experiment
    runner creates it only after the policy has committed to an action.
    """

    def __init__(self, true_model: Hypothesis):
        self._true_model = true_model

    def action_rewards(self) -> Mapping[str, float]:
        return {action: self._true_model.reward(action) for action in ACTIONS}

    def optimal_action(self) -> str:
        return self._true_model.optimal_action
