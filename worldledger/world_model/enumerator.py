"""Exhaustive Cartesian-product enumeration for a validated finite DSL."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from worldledger.world_model.canonical import canonical_model_id, model_hash
from worldledger.world_model.validator import (
    validate_contract, validate_total_model, validate_total_world,
    validate_world_contract,
)


@dataclass(frozen=True)
class Model:
    assignment: tuple[int, int, int]
    behavior_hash: str


@dataclass(frozen=True)
class Enumeration:
    models: tuple[Model, ...]
    unique_models: tuple[Model, ...]
    assignment_count: int
    unique_behavior_count: int
    collision_groups: tuple[tuple[str, tuple[tuple[int, int, int], ...]], ...]


def enumerate_models(spec: dict) -> Enumeration:
    validate_contract(spec)
    domains = [parameter["domain"] for parameter in spec["parameters"]]
    expected = 1
    for domain in domains:
        if not domain or len(set(domain)) != len(domain):
            raise ValueError("empty or duplicate parameter domain")
        expected *= len(domain)
    if expected != spec["max_assignments"]:
        raise ValueError("DSL count does not match declared cap")
    models = []
    groups: dict[str, list[tuple[int, int, int]]] = {}
    for values in product(*domains):
        assignment = tuple(values)
        validate_total_model(spec, assignment)
        signature = model_hash(spec, assignment)
        models.append(Model(assignment, signature))
        groups.setdefault(signature, []).append(assignment)
    if len(models) != expected or len({m.assignment for m in models}) != expected:
        raise AssertionError("incomplete or duplicate Cartesian-product traversal")
    representatives = tuple(next(model for model in models if model.behavior_hash == key)
                            for key in sorted(groups))
    return Enumeration(tuple(models), representatives, expected, len(groups),
                       tuple((key, tuple(value)) for key, value in sorted(groups.items())))


@dataclass(frozen=True)
class ContractWorld:
    assignment: tuple[int, int, int, int]
    canonical_model_id: str


@dataclass(frozen=True)
class WorldEnumeration:
    assignments: tuple[ContractWorld, ...]
    unique_models: tuple[ContractWorld, ...]
    assignment_count: int
    number_of_models: int
    canonical_hashes: tuple[str, ...]
    collision_groups: tuple[tuple[str, tuple[tuple[int, int, int, int], ...]], ...]


def enumerate_world_contract(contract: dict, base: dict) -> WorldEnumeration:
    """Enumerate every contract assignment and collapse equal stochastic kernels."""
    validate_world_contract(contract, base)
    domains = [parameter["domain"] for parameter in contract["parameters"]]
    expected = 1
    for domain in domains:
        if not domain or len(set(domain)) != len(domain):
            raise ValueError("nonempty duplicate-free finite domains required")
        expected *= len(domain)
    if expected != contract["assignment_count"]:
        raise ValueError("declared assignment count disagrees with product")
    models = []
    groups: dict[str, list[tuple[int, int, int, int]]] = {}
    for values in product(*domains):
        assignment = tuple(values)
        validate_total_world(contract, base, assignment)
        identifier = canonical_model_id(contract, base, assignment)
        models.append(ContractWorld(assignment, identifier))
        groups.setdefault(identifier, []).append(assignment)
    if len(models) != expected or len({m.assignment for m in models}) != expected:
        raise AssertionError("Cartesian-product traversal incomplete")
    representatives = tuple(next(model for model in models if model.canonical_model_id == key)
                            for key in sorted(groups))
    return WorldEnumeration(
        tuple(models), representatives, expected, len(groups), tuple(sorted(groups)),
        tuple((key, tuple(value)) for key, value in sorted(groups.items())))
