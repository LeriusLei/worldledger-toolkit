"""Map executable candidates to finite oracle behavior on frozen reachable domain.

The oracle is used only by the evaluator. Nothing from this module is rendered
into generation prompts. A non-matching candidate is off-family, not discarded.
"""

from worldledger.benchmark.model_family import Instance
from worldledger.evaluator.simulation import step
from worldledger.evaluator.probe_domain import probe_states
from worldledger.sandbox.executor import run_candidate


def history_consistent(instance: Instance, history: dict, source: str) -> bool:
    if len(instance.initial_state_distribution) != 1:
        raise ValueError("v3.1 replay requires point-mass initial state")
    state = instance.initial_state_distribution[0][0]
    for event in history["events"]:
        state, reward, observation = run_candidate(source, state, event["action"])
        if reward != event["reward"] or observation != tuple(event["observation"]):
            return False
    return True


def reachable_domain(instance: Instance, horizon: int) -> tuple[tuple[int, int], ...]:
    if horizon < 1:
        raise ValueError("positive horizon required")
    reachable = {state for state, _ in instance.initial_state_distribution}
    domain = set(reachable)
    for _ in range(horizon):
        next_states = {
            step(instance, model, state, action)[0]
            for state in reachable for model in instance.models for action in instance.allowed_actions
        }
        domain.update(next_states)
        reachable = next_states
    return tuple(sorted(domain))


def matching_models(instance: Instance, source: str, horizon: int) -> tuple[str, ...]:
    """Exact agreement on the same candidate-independent v0.3 probe domain."""
    domain = probe_states(instance, horizon)
    candidate = {
        (state, action): run_candidate(source, state, action)
        for state in domain for action in instance.allowed_actions
    }
    return tuple(sorted(
        model.model_id for model in instance.models
        if all(candidate[(state, action)] == step(instance, model, state, action)
               for state in domain for action in instance.allowed_actions)
    ))
