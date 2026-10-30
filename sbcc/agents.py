"""Baselines and FalsifyWM for the minimal Sprinkler benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional, Tuple

from .sprinkler import ACTIONS, PROBES, Posterior, SprinklerWorld


@dataclass(frozen=True)
class DecisionTrace:
    method: str
    action: str
    probe: Optional[str]
    probe_observation: Optional[int]
    action_reward: float
    total_utility: float
    entropy_before: float
    entropy_after: float
    decision_classes_before: int
    decision_classes_after: int
    predicted_response: Optional[int]
    prediction_confidence: float


def _argmax_action(scores: dict[str, float]) -> str:
    """Stable action tie-breaking: no_water, then water, then safe."""

    return max(ACTIONS, key=lambda action: (scores[action], -ACTIONS.index(action)))


def _bayes_action(posterior: Posterior) -> str:
    return _argmax_action({action: posterior.expected_reward(action) for action in ACTIONS})


def _robust_action(posterior: Posterior) -> str:
    return _argmax_action({action: posterior.worst_case_reward(action) for action in ACTIONS})


def _response_prediction(posterior: Posterior) -> Tuple[Optional[int], float]:
    masses = {
        response: sum(
            weight
            for hypothesis, weight in posterior.weights.items()
            if hypothesis.response_type == response
        )
        for response in (0, 1)
    }
    if masses[0] == masses[1]:
        return None, masses[0]
    prediction = max(masses, key=masses.get)
    return prediction, masses[prediction]


def _trace(
    method: str,
    world: SprinklerWorld,
    prior: Posterior,
    posterior: Posterior,
    action: str,
    probe: Optional[str] = None,
    probe_observation: Optional[int] = None,
    force_confidence: Optional[float] = None,
) -> DecisionTrace:
    reward = world.act(action)
    predicted_response, confidence = _response_prediction(posterior)
    if force_confidence is not None:
        confidence = force_confidence
    return DecisionTrace(
        method=method,
        action=action,
        probe=probe,
        probe_observation=probe_observation,
        action_reward=reward,
        total_utility=reward - (world.probe_cost(probe) if probe else 0.0),
        entropy_before=prior.entropy_bits(),
        entropy_after=posterior.entropy_bits(),
        decision_classes_before=len(prior.decision_classes()),
        decision_classes_after=len(posterior.decision_classes()),
        predicted_response=predicted_response,
        prediction_confidence=confidence,
    )


def run_map(world: SprinklerWorld, prior: Posterior, rng: Random) -> DecisionTrace:
    del rng
    # A single-model learner commits to the first maximum-posterior model.
    model = max(prior.weights, key=prior.weights.get)
    posterior = Posterior({model: 1.0})
    return _trace(
        "MAP",
        world,
        prior,
        posterior,
        model.optimal_action,
        force_confidence=1.0,
    )


def run_bma(world: SprinklerWorld, prior: Posterior, rng: Random) -> DecisionTrace:
    del rng
    return _trace("BMA", world, prior, prior, _bayes_action(prior))


def run_robust(world: SprinklerWorld, prior: Posterior, rng: Random) -> DecisionTrace:
    del rng
    return _trace("Robust", world, prior, prior, _robust_action(prior))


def _probe_and_act(
    method: str,
    world: SprinklerWorld,
    prior: Posterior,
    probe: str,
) -> DecisionTrace:
    observation = world.probe(probe)
    posterior = prior.condition(probe, observation, world.probe_accuracy(probe))
    action = _bayes_action(posterior)
    return _trace(method, world, prior, posterior, action, probe, observation)


def run_global_information_gain(
    world: SprinklerWorld, prior: Posterior, rng: Random
) -> DecisionTrace:
    del rng
    entropy_before = prior.entropy_bits()
    scores = {
        probe: (
            entropy_before - prior.expected_entropy_after(probe, world.probe_accuracy(probe))
        )
        / max(world.probe_cost(probe), 1e-12)
        for probe in PROBES
    }
    probe = max(PROBES, key=lambda item: (scores[item], -PROBES.index(item)))
    return _probe_and_act("GlobalIG", world, prior, probe)


def run_random_probe(world: SprinklerWorld, prior: Posterior, rng: Random) -> DecisionTrace:
    return _probe_and_act("RandomProbe", world, prior, rng.choice(PROBES))


def run_falsifywm(world: SprinklerWorld, prior: Posterior, rng: Random) -> DecisionTrace:
    del rng
    if len(prior.decision_classes()) == 1:
        return _trace("FalsifyWM", world, prior, prior, _bayes_action(prior))

    risk_before = prior.decision_risk()
    benefits = {
        probe: risk_before
        - prior.expected_decision_risk_after(probe, world.probe_accuracy(probe))
        for probe in PROBES
    }
    scores = {
        probe: benefits[probe] / max(world.probe_cost(probe), 1e-12)
        for probe in PROBES
    }
    probe = max(PROBES, key=lambda item: (scores[item], -PROBES.index(item)))
    if benefits[probe] <= world.probe_cost(probe) + 1e-12:
        return _trace("FalsifyWM", world, prior, prior, _bayes_action(prior))
    return _probe_and_act("FalsifyWM", world, prior, probe)


def run_oracle(world: SprinklerWorld, prior: Posterior, rng: Random, response_type: int) -> DecisionTrace:
    del rng
    action = "water" if response_type == 1 else "no_water"
    posterior = Posterior(
        {
            hypothesis: weight
            for hypothesis, weight in prior.weights.items()
            if hypothesis.response_type == response_type
        }
    )
    return _trace("Oracle", world, prior, posterior, action, force_confidence=1.0)


POLICIES = {
    "MAP": run_map,
    "BMA": run_bma,
    "Robust": run_robust,
    "GlobalIG": run_global_information_gain,
    "RandomProbe": run_random_probe,
    "FalsifyWM": run_falsifywm,
}
