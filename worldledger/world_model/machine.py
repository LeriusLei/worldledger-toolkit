"""Finite independent maintenance ontology for the v0.6.2 offline gate."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import hashlib
import json
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "protocols/v062/machine_contract.json"
STATES = ("healthy", "worn", "critical", "failed")
ACTIONS = ("operate", "inspect", "service", "replace")
REWARDS = {"healthy": 6, "worn": 4, "critical": -2, "failed": -12}


def load_machine_contract() -> dict:
    contract = json.loads(PATH.read_text(encoding="utf-8"))
    validate_machine_contract(contract)
    return contract


def validate_machine_contract(contract: dict) -> None:
    expected = {
        "contract_id": "drhc-v0.6.2-machine-maintenance-1",
        "status": "frozen_for_offline_validation_only",
        "no_api_authorization": True,
        "states": list(STATES), "initial_state": "worn", "actions": list(ACTIONS),
        "parameters": [
            {"name": "fast_deterioration", "domain": [0, 1]},
            {"name": "inspection_marker", "domain": [0, 1]}],
        "assignment_count": 4,
        "semantics": {
            "operate_reward_by_state": REWARDS,
            "operate_transition": "fast advances one state, slow stays; failed absorbs",
            "inspect_reward": -2, "inspect_transition": "stay",
            "inspect_observation": "next state and inspection_marker",
            "service_reward": -5, "service_transition": "healthy",
            "replace_reward": -6, "replace_transition": "healthy",
            "ordinary_observation": "next state",
            "probability": "deterministic rational 1"}}
    if contract != expected:
        raise ValueError("machine contract mutation or incomplete ontology")


def machine_kernel(contract: dict, assignment: tuple[int, int], state: str,
                   action: str) -> tuple:
    validate_machine_contract(contract)
    if (not isinstance(assignment, tuple) or len(assignment) != 2 or
            any(type(v) is not int or v not in (0, 1) for v in assignment)):
        raise ValueError("assignment outside contract")
    if state not in STATES or action not in ACTIONS:
        raise ValueError("illegal state or action")
    fast, marker = assignment
    if action == "operate":
        next_state = STATES[min(3, STATES.index(state) + fast)]
        reward = REWARDS[state]
    elif action == "inspect":
        next_state, reward = state, -2
    elif action == "service":
        next_state, reward = "healthy", -5
    else:
        next_state, reward = "healthy", -6
    observation = (("condition", next_state, "marker", marker) if action == "inspect"
                   else ("condition", next_state))
    return ((Fraction(1), (next_state, reward, observation)),)


def machine_signature(contract: dict, assignment: tuple[int, int]) -> tuple:
    return tuple((state, action, tuple(
        (p.numerator, p.denominator, next_state, reward, obs)
        for p, (next_state, reward, obs) in machine_kernel(contract, assignment, state, action)))
        for state in STATES for action in ACTIONS)


def machine_id(contract: dict, assignment: tuple[int, int]) -> str:
    payload = json.dumps(machine_signature(contract, assignment), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_machine_id(contract: dict, source: str) -> str:
    """Map restricted candidate JSON by its full interpreted behavior."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate candidate key")
            result[key] = value
        return result

    try:
        candidate = json.loads(source, object_pairs_hook=unique_pairs)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid candidate source") from exc
    if not isinstance(candidate, dict) or set(candidate) != {
            "fast_deterioration", "inspection_marker"}:
        raise ValueError("candidate outside machine DSL")
    assignment = (candidate["fast_deterioration"], candidate["inspection_marker"])
    # The total-kernel interpreter validates both values and all legal actions.
    return machine_id(contract, assignment)


def enumerate_machine(contract: dict) -> dict:
    validate_machine_contract(contract)
    domains = tuple(p["domain"] for p in contract["parameters"])
    assignments = tuple(product(*domains))
    independent = {(a, b) for a in (0, 1) for b in (0, 1)}
    if set(assignments) != independent or len(assignments) != contract["assignment_count"]:
        raise ValueError("enumeration incomplete")
    groups: dict[str, list[tuple[int, int]]] = {}
    checks = 0
    for assignment in assignments:
        for state in STATES:
            for action in ACTIONS:
                kernel = machine_kernel(contract, assignment, state, action)
                if sum((p for p, _ in kernel), Fraction()) != 1:
                    raise ValueError("unnormalized kernel")
                for p, (next_state, reward, obs) in kernel:
                    if (p <= 0 or next_state not in STATES or type(reward) is not int
                            or obs[:2] != ("condition", next_state)):
                        raise ValueError("state, reward or observation closure failure")
                checks += 1
        groups.setdefault(machine_id(contract, assignment), []).append(assignment)
    return {"assignments": assignments, "groups": groups,
            "assignment_count": len(assignments), "behavior_class_count": len(groups),
            "state_action_checks": checks, "independent_product_equal": True,
            "collision_groups": [members for members in groups.values() if len(members) > 1]}
