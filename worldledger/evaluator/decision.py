"""Exact finite-horizon Q and prior-free minimax-regret first action."""

from fractions import Fraction

from worldledger.benchmark.model_family import Instance, State, WorldModel
from worldledger.evaluator.simulation import step


Belief = tuple[tuple[State, Fraction], ...]


def _normalized(rows: dict[State, Fraction]) -> Belief:
    total = sum(rows.values(), Fraction())
    if total <= 0:
        raise ValueError("empty decision belief")
    return tuple(sorted((state, mass / total) for state, mass in rows.items() if mass > 0))


def frozen_belief(instance: Instance) -> Belief:
    rows: dict[State, Fraction] = {}
    for state, weight in instance.initial_state_distribution:
        rows[state] = rows.get(state, Fraction()) + weight
    return _normalized(rows)


def _q(instance: Instance, model: WorldModel, belief: Belief, action: str, horizon: int) -> Fraction:
    immediate = Fraction()
    groups: dict[tuple, dict[State, Fraction]] = {}
    for state, mass in belief:
        next_state, reward, observation = step(instance, model, state, action)
        immediate += mass * reward
        group = groups.setdefault(observation, {})
        group[next_state] = group.get(next_state, Fraction()) + mass
    if horizon == 1:
        return immediate
    continuation = Fraction()
    for group in groups.values():
        group_mass = sum(group.values(), Fraction())
        conditional = _normalized(group)
        continuation += group_mass * max(
            _q(instance, model, conditional, next_action, horizon - 1)
            for next_action in instance.allowed_actions
        )
    return immediate + continuation


def action_values(instance: Instance, model: WorldModel, horizon: int) -> dict[str, Fraction]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    belief = frozen_belief(instance)
    return {action: _q(instance, model, belief, action, horizon) for action in instance.allowed_actions}


def model_regret(values: dict[str, Fraction], action: str) -> Fraction:
    return max(values.values()) - values[action]


def minimax_action(instance: Instance, model_ids: set[str], q_by_model: dict[str, dict[str, Fraction]]) -> str:
    if not model_ids:
        raise ValueError("minimax requires nonempty candidate support")
    return min(
        instance.allowed_actions,
        key=lambda action: (max(model_regret(q_by_model[model_id], action) for model_id in model_ids),
                            instance.allowed_actions.index(action)),
    )
