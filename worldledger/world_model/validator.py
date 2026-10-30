"""Fail-closed validation and interpretation of the v0.6 bounded Grid DSL."""

from __future__ import annotations

from itertools import product
from fractions import Fraction
import hashlib
import json
from pathlib import Path


SPEC = Path(__file__).resolve().parents[1] / "protocols/v06/world_dsl_spec.json"
WORLD_CONTRACT = SPEC.with_name("world_contract.json")
ACTIONS = ("east", "north", "west", "south", "wait", "probe_a", "probe_b")
PARAMETERS = ("mechanism_a", "mechanism_b", "move_reward")


class ContractError(ValueError):
    pass


def load_contract(path: Path = SPEC) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("dsl_id") != "drhc-v06-grid-seed4-bounded-dsl-1":
        raise ContractError("unsupported DSL identity")
    state = spec.get("state_space", {})
    variables = state.get("variables", [])
    if [(v.get("name"), v.get("type"), v.get("domain")) for v in variables] != [
        ("x", "integer", [0, 1, 2]), ("y", "integer", [0, 1, 2, 3])
    ]:
        raise ContractError("unsupported or unbounded state domain")
    if state.get("constraints") != {"excluded_positions": [[2, 0]],
                                    "initial_position": [1, 1],
                                    "all_nonwall_positions_are_legal": True}:
        raise ContractError("state constraints changed")
    if spec.get("action_space", {}).get("legal_actions") != list(ACTIONS):
        raise ContractError("action domain changed")
    if spec["action_space"].get("action_constraints") != (
            "Actions are available from every legal state; movement into a boundary or wall stays in place."):
        raise ContractError("action constraints changed")
    params = spec.get("parameters", [])
    if [(p.get("name"), p.get("domain")) for p in params] != [
        ("mechanism_a", [0, 1]), ("mechanism_b", [0, 1]), ("move_reward", [-1, 0])
    ]:
        raise ContractError("parameter domains changed")
    transition = spec.get("transition_generator", {})
    if transition.get("allowed_operators") != ["deterministic", "conditional", "bounded_modification"]:
        raise ContractError("unsupported transition operator")
    if transition.get("unsupported_operators") != ["stochastic", "arbitrary_python", "unbounded_state_update"]:
        raise ContractError("unsupported operator list changed")
    if transition.get("movement_deltas") != {"east": [1, 0], "north": [0, -1],
                                               "west": [-1, 0], "south": [0, 1]}:
        raise ContractError("movement semantics changed")
    if transition.get("gate_rules") != [
        {"at": [1, 1], "action": "east", "kind": "door", "bit": "mechanism_a", "closed_target": [1, 1]},
        {"at": [1, 1], "action": "north", "kind": "door", "bit": "mechanism_b", "closed_target": [1, 1]}
    ] or transition.get("wait_and_probe_preserve_state") is not True:
        raise ContractError("gate or probe semantics changed")
    observation = spec.get("observation_generator", {})
    if observation.get("sensor_masks") != {"a": 1, "b": 0}:
        raise ContractError("sensor mapping changed")
    if observation.get("mapping") != (
            "non-probe=('position',x,y); probe_a=('position',x,y,'sensor_a',mechanism_a XOR 1); "
            "probe_b=('position',x,y,'sensor_b',mechanism_b XOR 0)"):
        raise ContractError("observation mapping changed")
    reward = spec.get("reward_generator", {})
    if reward.get("operators") != ["integer_constant", "conditional_gate_bit", "bounded_parameter"]:
        raise ContractError("reward operators changed")
    if {k: reward.get(k) for k in ("blocked", "gate_a_open", "gate_b_open", "wait", "probe", "ordinary_move", "range")} != {
        "blocked": -4, "gate_a_open": 8, "gate_b_open": 11,
        "wait": -1, "probe": -2, "ordinary_move": "move_reward", "range": [-4, 11]
    }:
        raise ContractError("reward semantics changed")
    if spec.get("max_assignments") != 8:
        raise ContractError("enumeration bound changed")
    if spec.get("semantics") != (
            "A parameter assignment denotes the total deterministic T/R/O function on every legal state and action. "
            "Rules are applied in order: probe, wait, blocked boundary/wall, gate, ordinary move. "
            "Initial observation is ('position',1,1)."):
        raise ContractError("rule order changed")
    return spec


def legal_states(spec: dict) -> tuple[tuple[int, int], ...]:
    validate_contract(spec)
    xs, ys = (v["domain"] for v in spec["state_space"]["variables"])
    excluded = {tuple(p) for p in spec["state_space"]["constraints"]["excluded_positions"]}
    return tuple(p for p in product(xs, ys) if p not in excluded)


def validate_assignment(spec: dict, assignment: tuple[int, int, int]) -> None:
    validate_contract(spec)
    if (not isinstance(assignment, tuple) or len(assignment) != 3 or
            any(type(value) is not int or value not in domain["domain"]
                for value, domain in zip(assignment, spec["parameters"]))):
        raise ContractError("assignment outside finite DSL")


def step(spec: dict, assignment: tuple[int, int, int], state: tuple[int, int],
         action: str) -> tuple[tuple[int, int], int, tuple]:
    validate_assignment(spec, assignment)
    if state not in legal_states(spec) or action not in ACTIONS:
        raise ContractError("illegal state or action")
    a, b, move_reward = assignment
    if action in ("probe_a", "probe_b"):
        observation = (("position", *state, "sensor_a", a ^ 1) if action == "probe_a"
                       else ("position", *state, "sensor_b", b))
        return state, -2, observation
    if action == "wait":
        return state, -1, ("position", *state)
    dx, dy = spec["transition_generator"]["movement_deltas"][action]
    target = (state[0] + dx, state[1] + dy)
    if target not in legal_states(spec):
        return state, -4, ("position", *state)
    if state == (1, 1) and action in ("east", "north"):
        bit, opened = (a, 8) if action == "east" else (b, 11)
        next_state = target if bit else state
        return next_state, opened if bit else -4, ("position", *next_state)
    return target, move_reward, ("position", *target)


def validate_total_model(spec: dict, assignment: tuple[int, int, int]) -> int:
    checks = 0
    legal = set(legal_states(spec))
    for state in legal:
        for action in ACTIONS:
            next_state, reward, observation = step(spec, assignment, state, action)
            if next_state not in legal or type(reward) is not int or not -4 <= reward <= 11:
                raise ContractError("model state/reward closure failed")
            if observation[:3] != ("position", *next_state):
                raise ContractError("model observation alignment failed")
            checks += 1
    return checks


def load_world_contract(path: Path = WORLD_CONTRACT) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("base_dsl_ref") != "world_dsl_spec.json":
        raise ContractError("unregistered base DSL reference")
    base_path = path.with_name(contract["base_dsl_ref"])
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != contract["base_dsl_sha256"]:
        raise ContractError("base DSL hash mismatch")
    base = load_contract(base_path)
    validate_world_contract(contract, base)
    return contract, base


def validate_world_contract(contract: dict, base: dict) -> None:
    validate_contract(base)
    if contract.get("contract_id") != "drhc-v0.6.1-seed4-bounded-gate-noise-1":
        raise ContractError("unknown world contract")
    if contract.get("status") != "frozen_for_offline_validation_only" or contract.get("no_api_authorization") is not True:
        raise ContractError("unsafe execution status")
    state = contract.get("state", {})
    if state != {"variables": [{"name": "x", "domain": [0, 1, 2]},
                                {"name": "y", "domain": [0, 1, 2, 3]}],
                 "excluded_positions": [[2, 0]], "initial_state": [1, 1]}:
        raise ContractError("state domain differs from bounded profile")
    if contract.get("actions") != list(ACTIONS):
        raise ContractError("action domain changed")
    if contract.get("parameters") != [
        {"name": "mechanism_a", "domain": [0, 1]},
        {"name": "mechanism_b", "domain": [0, 1]},
        {"name": "move_reward", "domain": [-1, 0]},
        {"name": "gate_flip_probability_numerator", "domain": [0, 1], "denominator": 2},
    ]:
        raise ContractError("parameter domain changed")
    transition = contract.get("transition", {})
    if (transition.get("operators") != ["deterministic", "conditional", "bounded_bernoulli_gate_flip"]
            or transition.get("outcome_probability_denominator") != 2
            or transition.get("maximum_outcomes_per_state_action") != 2
            or transition.get("stochastic_trigger") != {"state": [1, 1],
                                                        "actions": {"east": "mechanism_a", "north": "mechanism_b"}}
            or transition.get("movement_deltas") != base["transition_generator"]["movement_deltas"]
            or transition.get("wall_and_boundary_action") != "stay"
            or transition.get("wait_and_probe_action") != "stay"
            or transition.get("stochastic_scope") != (
                "Only east/north gates from initial state; with probability p flip the corresponding mechanism bit for that step. All other transitions are deterministic.")):
        raise ContractError("transition operator or probability bound changed")
    observation = contract.get("observation", {})
    if (observation.get("finite_alphabet") != {
            "position_states": "all_legal_states",
            "tags": ["position", "sensor_a", "sensor_b"], "sensor_bits": [0, 1]}
            or observation.get("non_probe_mapping") != ["position", "next_x", "next_y"]
            or observation.get("probe_a_mapping") != ["position", "next_x", "next_y", "sensor_a", "mechanism_a XOR 1"]
            or observation.get("probe_b_mapping") != ["position", "next_x", "next_y", "sensor_b", "mechanism_b XOR 0"]
            or observation.get("observable_state") != ["x", "y"]
            or observation.get("hidden_state") != ["mechanism_a", "mechanism_b"]):
        raise ContractError("observation mapping changed")
    reward = contract.get("reward", {})
    if reward != {"finite_values": [-4, -2, -1, 0, 8, 11], "range": [-4, 11],
                  "operators": ["constant", "conditional_gate_bit", "bounded_move_reward"],
                  "blocked": -4, "probe": -2, "wait": -1,
                  "gate_a_open": 8, "gate_b_open": 11,
                  "ordinary_move_parameter": "move_reward"}:
        raise ContractError("reward mapping changed")
    if (contract.get("assignment_count") != 16 or contract.get("probability_semantics") !=
            "Exact rational probabilities; merge identical (next_state,reward,observation) outcomes before hashing; deterministic kernels use probability 1."):
        raise ContractError("count or rational semantics changed")


def world_kernel(contract: dict, base: dict, assignment: tuple[int, int, int, int],
                 state: tuple[int, int], action: str) -> tuple:
    validate_world_contract(contract, base)
    if (not isinstance(assignment, tuple) or len(assignment) != 4 or
            any(type(v) is not int or v not in contract["parameters"][i]["domain"]
                for i, v in enumerate(assignment))):
        raise ContractError("assignment outside world contract")
    a, b, move_reward, noise_num = assignment
    if state not in legal_states(base) or action not in ACTIONS:
        raise ContractError("state or action outside world contract")
    ordinary = step(base, (a, b, move_reward), state, action)
    if noise_num == 0 or state != (1, 1) or action not in ("east", "north"):
        return ((Fraction(1), ordinary),)
    flipped = ((1 - a, b, move_reward) if action == "east"
               else (a, 1 - b, move_reward))
    alternate = step(base, flipped, state, action)
    p = Fraction(noise_num, 2)
    combined: dict[tuple, Fraction] = {}
    for weight, outcome in ((1 - p, ordinary), (p, alternate)):
        combined[outcome] = combined.get(outcome, Fraction()) + weight
    return tuple((weight, outcome) for outcome, weight in
                 sorted(combined.items(), key=lambda pair: repr(pair[0])))


def validate_total_world(contract: dict, base: dict,
                         assignment: tuple[int, int, int, int]) -> int:
    count = 0
    legal = set(legal_states(base))
    for state in legal:
        for action in ACTIONS:
            kernel = world_kernel(contract, base, assignment, state, action)
            if not 1 <= len(kernel) <= 2 or sum((p for p, _ in kernel), Fraction()) != 1:
                raise ContractError("invalid probability kernel")
            for probability, (next_state, reward, observation) in kernel:
                if (probability <= 0 or next_state not in legal
                        or reward not in contract["reward"]["finite_values"]
                        or observation[:3] != ("position", *next_state)):
                    raise ContractError("invalid outcome")
            count += 1
    return count
