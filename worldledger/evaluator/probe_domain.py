"""Candidate-independent finite probe domain for DRHC v0.3."""

from worldledger.benchmark.model_family import Instance, State
from worldledger.evaluator.simulation import step


def probe_states(instance: Instance, horizon: int) -> tuple[State, ...]:
    """Registered states plus all oracle-family states reachable within horizon.

    The domain depends only on the frozen benchmark family, not on generated
    candidates, histories, observed outcomes, or their identities.
    """
    if horizon < 1:
        raise ValueError("positive probe horizon required")
    current = {state for state, _ in instance.initial_state_distribution}
    domain = set(instance.probe_states) | current
    for _ in range(horizon):
        current = {
            step(instance, model, state, action)[0]
            for state in current
            for model in instance.models
            for action in instance.allowed_actions
        }
        domain.update(current)
    return tuple(sorted(domain))
