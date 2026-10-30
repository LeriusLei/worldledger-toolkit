"""Exact one-step minimax-regret control for the finite v0.6 DSL.

This intentionally does not reuse the older h=5 planner, whose WorldModel
interface cannot represent move_reward as a per-model DSL parameter.
"""

from __future__ import annotations

from fractions import Fraction

from worldledger.world_model.validator import ACTIONS, step


def one_step_values(spec: dict, assignment: tuple[int, int, int]) -> dict[str, int]:
    initial = tuple(spec["state_space"]["constraints"]["initial_position"])
    return {action: step(spec, assignment, initial, action)[1] for action in ACTIONS}


def minimax_action(spec: dict, support: set[tuple[int, int, int]]) -> str:
    if not support:
        raise ValueError("nonempty admissible support required")
    values = {assignment: one_step_values(spec, assignment) for assignment in support}
    return min(ACTIONS, key=lambda action: (
        max(max(q.values()) - q[action] for q in values.values()),
        ACTIONS.index(action)))


def risk(spec: dict, truth: set[tuple[int, int, int]], action: str) -> int:
    if not truth or action not in ACTIONS:
        raise ValueError("nonempty truth support and registered action required")
    return max(max(q.values()) - q[action]
               for q in (one_step_values(spec, assignment) for assignment in truth))


def omission_delta(spec: dict, full_support: set[tuple[int, int, int]],
                   generated_support: set[tuple[int, int, int]]) -> dict:
    """One-step empty-history control; reject an uncertified full denominator."""
    from worldledger.world_model.enumerator import enumerate_models

    if full_support != {model.assignment for model in enumerate_models(spec).models}:
        raise ValueError("full support must equal exhaustive empty-history DSL space")
    if not generated_support or not generated_support <= full_support:
        raise ValueError("generated support must be a nonempty admissible subset")
    full_action = minimax_action(spec, full_support)
    generated_action = minimax_action(spec, generated_support)
    full_risk = risk(spec, full_support, full_action)
    generated_risk = risk(spec, full_support, generated_action)
    delta = generated_risk - full_risk
    if delta < 0:
        raise AssertionError("full-support minimax risk exceeded subset risk")
    return {"horizon": 1, "full_action": full_action,
            "generated_action": generated_action,
            "full_risk": full_risk, "generated_risk": generated_risk,
            "delta_omit": delta}


def kernel_omission_delta(contract: dict, base: dict,
                          full_ids: set[str], generated_ids: set[str]) -> dict:
    """Certified empty-history one-step regret on canonical stochastic classes."""
    from worldledger.world_model.enumerator import enumerate_world_contract
    from worldledger.world_model.validator import world_kernel

    enumeration = enumerate_world_contract(contract, base)
    by_id = {model.canonical_model_id: model.assignment
             for model in enumeration.unique_models}
    if full_ids != set(by_id):
        raise ValueError("uncertified full hypothesis-space denominator")
    if not generated_ids or not generated_ids <= full_ids:
        raise ValueError("off-contract or empty generated support")
    start = tuple(contract["state"]["initial_state"])

    def q(identifier: str) -> dict[str, Fraction]:
        assignment = by_id[identifier]
        return {action: sum((p * outcome[1] for p, outcome in
                             world_kernel(contract, base, assignment, start, action)), Fraction())
                for action in ACTIONS}

    vectors = {identifier: q(identifier) for identifier in full_ids}

    def choose(ids: set[str]) -> str:
        return min(ACTIONS, key=lambda action: (
            max(max(vectors[key].values()) - vectors[key][action] for key in ids),
            ACTIONS.index(action)))

    def risk(action: str) -> Fraction:
        return max(max(vector.values()) - vector[action] for vector in vectors.values())

    full_action, generated_action = choose(full_ids), choose(generated_ids)
    full_risk, generated_risk = risk(full_action), risk(generated_action)
    delta = generated_risk - full_risk
    if delta < 0:
        raise AssertionError("full-support minimax was worse than subset policy")
    return {"horizon": 1, "full_action": full_action,
            "generated_action": generated_action,
            "full_risk": [full_risk.numerator, full_risk.denominator],
            "generated_risk": [generated_risk.numerator, generated_risk.denominator],
            "delta_omit": [delta.numerator, delta.denominator]}


def eligible_pool_metrics(contract: dict, base: dict, candidate_ids: set[str]) -> dict:
    """Report N/A rather than turn an off-contract candidate into an omission."""
    from worldledger.world_model.enumerator import enumerate_world_contract

    full = set(enumerate_world_contract(contract, base).canonical_hashes)
    if not candidate_ids:
        return {"coverage": None, "delta_omit": None, "reason": "empty_candidate_pool"}
    if not candidate_ids <= full:
        return {"coverage": None, "delta_omit": None, "reason": "off_contract_candidate"}
    result = kernel_omission_delta(contract, base, full, candidate_ids)
    return {"coverage": [len(candidate_ids), len(full)],
            "delta_omit": result["delta_omit"], "reason": None,
            "decision_control": result}
