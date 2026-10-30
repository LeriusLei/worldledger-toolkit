"""Exact finite-horizon shared-policy minimax regret for finite kernels.

Policies branch on public (reward, observation). Each model carries its own
unnormalized state mass; actions are shared before a public branch separates
models. The finite Pareto frontier retains every undominated return vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache


@dataclass(frozen=True)
class Family:
    name: str
    states: tuple
    actions: tuple[str, ...]
    initial_state: object
    models: tuple[tuple[str, object], ...]
    kernel: object


@dataclass(frozen=True)
class Policy:
    action: str | None
    branches: tuple[tuple[tuple, "Policy"], ...] = ()


STOP = Policy(None)


def _prune(entries: list[tuple[tuple[Fraction, ...], Policy]]) -> tuple:
    # Preserve a reproducible first policy for equal return vectors.
    unique = {}
    for vector, policy in entries:
        unique.setdefault(vector, policy)
    vectors = list(unique)
    frontier = [v for v in vectors if not any(
        w != v and all(x >= y for x, y in zip(w, v))
        for w in vectors)]
    return tuple((v, unique[v]) for v in frontier)


def _root(family: Family, ids: tuple[str, ...]) -> tuple:
    return tuple((identifier, family.initial_state, Fraction(1)) for identifier in ids)


def _frontier(family: Family, ids: tuple[str, ...], horizon: int) -> tuple:
    by_id = dict(family.models)
    zero = (Fraction(0),) * len(ids)

    @lru_cache(maxsize=None)
    def solve(masses: tuple, remaining: int) -> tuple:
        if remaining == 0:
            return ((zero, STOP),)
        entries = []
        for action in family.actions:
            immediate = [Fraction(0) for _ in ids]
            branches: dict[tuple, dict[tuple, Fraction]] = {}
            for identifier, state, mass in masses:
                index = ids.index(identifier)
                for probability, (next_state, reward, observation) in family.kernel(
                        by_id[identifier], state, action):
                    if probability <= 0:
                        raise ValueError("nonpositive probability")
                    weight = mass * probability
                    immediate[index] += weight * reward
                    key = (reward, tuple(observation))
                    branch = branches.setdefault(key, {})
                    particle = (identifier, next_state)
                    branch[particle] = branch.get(particle, Fraction()) + weight
            options = [((tuple(sorted((identifier, state, mass) for
                                      (identifier, state), mass in particles.items())), key))
                       for key, particles in sorted(branches.items(), key=lambda item: repr(item[0]))]
            candidates = [(tuple(immediate), ())]
            for child_masses, key in options:
                expanded = []
                for base_vector, children in candidates:
                    for vector, child_policy in solve(child_masses, remaining - 1):
                        expanded.append((tuple(a + b for a, b in zip(base_vector, vector)),
                                         children + ((key, child_policy),)))
                # Branch choices can be pruned before combining later branches.
                candidates = [(v, p.branches) for v, p in _prune(
                    [(v, Policy(action, children)) for v, children in expanded])]
            entries.extend((vector, Policy(action, children)) for vector, children in candidates)
        return _prune(entries)

    return solve(_root(family, ids), horizon)


def _single_model_optimum(family: Family, identifier: str, horizon: int) -> Fraction:
    model = dict(family.models)[identifier]

    @lru_cache(maxsize=None)
    def value(state: object, remaining: int) -> Fraction:
        if remaining == 0:
            return Fraction()
        return max(sum((p * (reward + value(next_state, remaining - 1))
                        for p, (next_state, reward, _) in family.kernel(model, state, action)),
                       Fraction()) for action in family.actions)

    return value(family.initial_state, horizon)


def choose_policy(family: Family, ids: set[str], horizon: int) -> tuple[Policy, tuple]:
    if type(horizon) is not int or horizon < 1 or not ids:
        raise ValueError("positive horizon and nonempty support required")
    registered = {key for key, _ in family.models}
    if not ids <= registered:
        raise ValueError("off-contract support")
    ordered = tuple(sorted(ids))
    optimal = tuple(_single_model_optimum(family, key, horizon) for key in ordered)
    vector, policy = min(_frontier(family, ordered, horizon),
                         key=lambda item: (max(o - v for o, v in zip(optimal, item[0])),
                                           family.actions.index(item[1].action), repr(item[1])))
    return policy, vector


def evaluate_policy(family: Family, identifier: str, policy: Policy,
                    horizon: int, *, fallback_action: str | None = None) -> Fraction:
    """Evaluate one shared tree under a truth model; unseen branches use fixed fallback."""
    model = dict(family.models)[identifier]
    fallback = fallback_action or family.actions[0]

    @lru_cache(maxsize=None)
    def run(state: object, node: Policy, remaining: int) -> Fraction:
        if remaining == 0:
            return Fraction()
        action = node.action if node.action is not None else fallback
        branches = dict(node.branches)
        return sum((p * (reward + run(next_state,
                                      branches.get((reward, tuple(observation)), STOP),
                                      remaining - 1))
                    for p, (next_state, reward, observation) in
                    family.kernel(model, state, action)), Fraction())

    return run(family.initial_state, policy, horizon)


def updated_support(family: Family, support: set[str], state: object,
                    action: str, reward: int, observation: tuple) -> set[str]:
    """Filter a public branch; an impossible observation yields empty support."""
    by_id = dict(family.models)
    if not support <= set(by_id) or action not in family.actions or state not in family.states:
        raise ValueError("invalid support, action or state")
    return {identifier for identifier in support if any(
        p > 0 and observed_reward == reward and tuple(observed) == tuple(observation)
        for p, (_, observed_reward, observed) in
        family.kernel(by_id[identifier], state, action))}


def omission_delta(family: Family, full: set[str], subset: set[str], horizon: int) -> dict:
    universe = {key for key, _ in family.models}
    if full != universe:
        raise ValueError("incomplete contract denominator")
    if not subset or not subset <= full:
        return {"coverage": None, "delta_omit": None,
                "reason": "empty_candidate_pool" if not subset else "off_contract_candidate"}
    full_policy, _ = choose_policy(family, full, horizon)
    subset_policy, _ = choose_policy(family, subset, horizon)
    optima = {key: _single_model_optimum(family, key, horizon) for key in full}
    def risk(policy: Policy) -> Fraction:
        return max(optima[key] - evaluate_policy(family, key, policy, horizon)
                   for key in full)
    full_risk, subset_risk = risk(full_policy), risk(subset_policy)
    delta = subset_risk - full_risk
    if delta < 0:
        raise AssertionError("full support risk exceeds subset risk")
    return {"coverage": [len(subset), len(full)], "delta_omit": [delta.numerator, delta.denominator],
            "full_risk": [full_risk.numerator, full_risk.denominator],
            "subset_risk": [subset_risk.numerator, subset_risk.denominator],
            "full_action": full_policy.action, "subset_action": subset_policy.action,
            "reason": None}


def machine_family(contract: dict) -> Family:
    from worldledger.world_model.machine import ACTIONS, STATES, enumerate_machine, machine_kernel
    enumeration = enumerate_machine(contract)
    models = tuple((key, members[0]) for key, members in enumeration["groups"].items())
    return Family("machine", STATES, ACTIONS, "worn", models,
                  lambda assignment, state, action: machine_kernel(contract, assignment, state, action))


def grid_family(contract: dict, base: dict, assignments: tuple | None = None) -> Family:
    from worldledger.world_model.enumerator import enumerate_world_contract
    from worldledger.world_model.validator import ACTIONS, legal_states, world_kernel
    enumeration = enumerate_world_contract(contract, base)
    selected = tuple(enumeration.unique_models if assignments is None else
                     (m for m in enumeration.unique_models if m.assignment in assignments))
    if not selected:
        raise ValueError("empty Grid model selection")
    return Family("grid", legal_states(base), ACTIONS, tuple(contract["state"]["initial_state"]),
                  tuple((m.canonical_model_id, m.assignment) for m in selected),
                  lambda assignment, state, action: world_kernel(contract, base, assignment, state, action))
