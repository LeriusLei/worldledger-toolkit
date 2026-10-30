"""Exact positive-likelihood filtering over the finite latent-state belief."""

from fractions import Fraction

from worldledger.benchmark.model_family import Instance, State, WorldModel
from worldledger.evaluator.simulation import observe, step


Belief = dict[State, Fraction]


def initial_belief(instance: Instance, model: WorldModel, initial_observation: list) -> Belief:
    total = sum(weight for _, weight in instance.initial_state_distribution)
    if total <= 0:
        raise ValueError("initial distribution must have positive total weight")
    belief: Belief = {}
    for state, weight in instance.initial_state_distribution:
        if weight < 0:
            raise ValueError("negative initial weight")
        if list(observe(instance, model, state)) == initial_observation:
            belief[state] = belief.get(state, Fraction()) + Fraction(weight, total)
    return belief


def posterior_weight(instance: Instance, model: WorldModel, history: dict) -> Fraction:
    belief = initial_belief(instance, model, history["initial_observation"])
    for event in history["events"]:
        updated: Belief = {}
        for state, mass in belief.items():
            next_state, reward, observation = step(instance, model, state, event["action"])
            if reward == event["reward"] and list(observation) == event["observation"]:
                updated[next_state] = updated.get(next_state, Fraction()) + mass
        belief = updated
    return sum(belief.values(), Fraction())


def consistent_models(instance: Instance, history: dict) -> tuple[WorldModel, ...]:
    return tuple(model for model in instance.models if posterior_weight(instance, model, history) > 0)
