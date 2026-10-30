"""Canonical observed behavior and decision-value quotients, never code hashes."""

from fractions import Fraction
from functools import lru_cache

from worldledger.benchmark.model_family import Instance, WorldModel
from worldledger.evaluator.decision import action_values
from worldledger.evaluator.probe_domain import probe_states as derived_probe_states
from worldledger.evaluator.simulation import observe, step


def _fraction(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _probe_q_vectors(instance: Instance, model: WorldModel, states: tuple,
                     actions: tuple[str, ...], horizon: int) -> tuple:
    """Exact model-known Q at each frozen state; benchmark transitions are deterministic."""
    @lru_cache(maxsize=None)
    def value(state, remaining):
        if remaining == 0:
            return Fraction()
        continuations = []
        for action in actions:
            next_state, reward, _ = step(instance, model, state, action)
            continuations.append(Fraction(reward) + value(next_state, remaining - 1))
        return max(continuations)

    return tuple(
        (state, tuple(_fraction(Fraction(reward) + value(next_state, horizon - 1))
                      for action in actions
                      for next_state, reward, _ in (step(instance, model, state, action),)))
        for state in states
    )


def behavioral_signature(instance: Instance, model: WorldModel, horizon: int,
                         frozen_states: tuple | None = None,
                         frozen_actions: tuple[str, ...] | None = None) -> tuple:
    states = instance.probe_states if frozen_states is None else frozen_states
    actions = instance.allowed_actions if frozen_actions is None else frozen_actions
    transitions = tuple(
        (state, action, *step(instance, model, state, action))
        for state in states
        for action in actions
    )
    initial_observations = tuple(observe(instance, model, state) for state in states)
    if frozen_states is not None:
        values = _probe_q_vectors(instance, model, states, actions, horizon)
    else:
        q_values = action_values(instance, model, horizon)
        values = tuple(_fraction(q_values[a]) for a in actions)
    return transitions, initial_observations, values


def decision_signature(instance: Instance, model: WorldModel, horizon: int,
                       frozen_states: tuple | None = None,
                       frozen_actions: tuple[str, ...] | None = None) -> tuple:
    actions = instance.allowed_actions if frozen_actions is None else frozen_actions
    if frozen_states is not None:
        return _probe_q_vectors(instance, model, frozen_states, actions, horizon)
    values = action_values(instance, model, horizon)
    return tuple(_fraction(values[a]) for a in actions)


def quotient(instance: Instance, horizon: int, *, frozen_probe: bool = False) -> dict[str, dict[str, str]]:
    """Legacy quotient by default; DRHC v0.3 uses frozen_probe=True at h=5."""
    result: dict[str, dict[str, str]] = {}
    states = derived_probe_states(instance, horizon) if frozen_probe else instance.probe_states
    actions = instance.allowed_actions
    for kind, signature_fn, prefix in (
        ("behavior", behavioral_signature, "b"),
        ("decision", decision_signature, "d"),
    ):
        signatures = {
            model.model_id: (
                signature_fn(instance, model, horizon,
                             states if frozen_probe else None, actions)
            )
            for model in instance.models
        }
        distinct = sorted(set(signatures.values()), key=repr)
        labels = {signature: f"{prefix}{index}" for index, signature in enumerate(distinct)}
        result[kind] = {model_id: labels[signature] for model_id, signature in signatures.items()}
    return result
