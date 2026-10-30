"""Exact finite-support, observation-contingent minimax-regret planning.

Policies share an action across models until observed (reward, observation)
separates them. This module never assumes the true model becomes known after
the first action. It uses exact rational arithmetic and no frontier truncation.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from worldledger.benchmark.model_family import Instance, State, WorldModel
from worldledger.evaluator.decision import action_values, frozen_belief
from worldledger.evaluator.simulation import step


Belief = tuple[tuple[State, Fraction], ...]
Node = tuple[tuple[str, Belief], ...]


@dataclass(frozen=True)
class Policy:
    action: str
    children: tuple[tuple[tuple, "Policy"], ...] = ()

    def child(self, event: tuple) -> "Policy | None":
        return next((policy for key, policy in self.children if key == event), None)


@dataclass(frozen=True)
class Option:
    values: tuple[Fraction, ...]
    policy: Policy


def _normalize(rows: dict[State, Fraction]) -> Belief:
    total = sum(rows.values(), Fraction())
    if total <= 0:
        raise ValueError("empty posterior")
    return tuple(sorted((state, mass / total) for state, mass in rows.items() if mass))


def _frontier(options: list[Option]) -> tuple[Option, ...]:
    kept: list[Option] = []
    for option in options:
        if any(all(a >= b for a, b in zip(other.values, option.values)) for other in kept):
            continue
        kept = [other for other in kept if not all(a >= b for a, b in zip(option.values, other.values))]
        kept.append(option)
    return tuple(kept)


def _events(instance: Instance, models: dict[str, WorldModel], node: Node, action: str):
    immediate = {}
    branch_rows: dict[tuple, dict[str, dict[State, Fraction]]] = {}
    branch_weight: dict[tuple, dict[str, Fraction]] = {}
    for model_id, belief in node:
        immediate[model_id] = Fraction()
        for state, mass in belief:
            next_state, reward, observation = step(instance, models[model_id], state, action)
            event = (reward, observation)
            immediate[model_id] += mass * reward
            rows = branch_rows.setdefault(event, {}).setdefault(model_id, {})
            rows[next_state] = rows.get(next_state, Fraction()) + mass
            weights = branch_weight.setdefault(event, {})
            weights[model_id] = weights.get(model_id, Fraction()) + mass
    return immediate, branch_rows, branch_weight


def choose_policy(instance: Instance, model_ids: set[str], horizon: int) -> Policy:
    """Choose one exact shared policy, minimizing worst-case model regret."""
    if horizon < 1 or not model_ids:
        raise ValueError("positive horizon and nonempty support required")
    models = {model.model_id: model for model in instance.models}
    if not model_ids <= models.keys():
        raise ValueError("unknown model")
    initial = frozen_belief(instance)
    root = tuple((model_id, initial) for model_id in sorted(model_ids))
    ideal = {model_id: max(action_values(instance, models[model_id], horizon).values())
             for model_id in model_ids}

    @lru_cache(maxsize=None)
    def solve(node: Node, remaining: int) -> tuple[Option, ...]:
        ids = tuple(model_id for model_id, _ in node)
        all_options: list[Option] = []
        for action in instance.allowed_actions:
            immediate, rows, weights = _events(instance, models, node, action)
            partial = [Option(tuple(immediate[x] for x in ids), Policy(action))]
            if remaining > 1:
                for event in sorted(rows, key=repr):
                    child_node = tuple((x, _normalize(state_rows)) for x, state_rows in sorted(rows[event].items()))
                    children = solve(child_node, remaining - 1)
                    next_partial = []
                    child_index = {model_id: index for index, (model_id, _) in enumerate(child_node)}
                    for old in partial:
                        for child in children:
                            values = tuple(
                                old.values[index] + weights[event].get(model_id, Fraction()) * child.values[child_index[model_id]]
                                if model_id in child_index else old.values[index]
                                for index, model_id in enumerate(ids)
                            )
                            policy = Policy(action, old.policy.children + ((event, child.policy),))
                            next_partial.append(Option(values, policy))
                    partial = list(_frontier(next_partial))
            all_options.extend(partial)
        return _frontier(all_options)

    options = solve(root, horizon)
    return min(options, key=lambda option: (
        max(ideal[model_id] - option.values[index] for index, model_id in enumerate(sorted(model_ids))),
        instance.allowed_actions.index(option.policy.action), repr(option.policy),
    )).policy


def policy_value(instance: Instance, model: WorldModel, policy: Policy, horizon: int) -> Fraction:
    """Evaluate a frozen policy in a model; off-support events use first action."""
    if horizon < 1:
        raise ValueError("positive horizon required")

    def value(belief: Belief, current: Policy, remaining: int) -> Fraction:
        immediate = Fraction()
        branches: dict[tuple, dict[State, Fraction]] = {}
        for state, mass in belief:
            next_state, reward, observation = step(instance, model, state, current.action)
            immediate += mass * reward
            rows = branches.setdefault((reward, observation), {})
            rows[next_state] = rows.get(next_state, Fraction()) + mass
        if remaining == 1:
            return immediate
        continuation = Fraction()
        for event, rows in branches.items():
            probability = sum(rows.values(), Fraction())
            child = current.child(event) or Policy(instance.allowed_actions[0])
            continuation += probability * value(_normalize(rows), child, remaining - 1)
        return immediate + continuation

    return value(frozen_belief(instance), policy, horizon)


def policy_risk(instance: Instance, true_ids: set[str], policy: Policy, horizon: int) -> Fraction:
    if not true_ids:
        raise ValueError("nonempty truth support required")
    return max(
        max(action_values(instance, instance.model(model_id), horizon).values())
        - policy_value(instance, instance.model(model_id), policy, horizon)
        for model_id in true_ids
    )
