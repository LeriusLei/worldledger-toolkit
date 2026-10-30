"""Canonical behavioral and decision signatures; source text is irrelevant."""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
import hashlib
import json

from worldledger.world_model.validator import ACTIONS, legal_states, step, world_kernel


def behavior_signature(spec: dict, assignment: tuple[int, int, int]) -> tuple:
    return tuple((state, action, *step(spec, assignment, state, action))
                 for state in legal_states(spec) for action in ACTIONS)


def canonical_hash(signature: tuple) -> str:
    data = json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def model_hash(spec: dict, assignment: tuple[int, int, int]) -> str:
    return canonical_hash(behavior_signature(spec, assignment))


def source_behavior_signature(spec: dict, source: str) -> tuple:
    """Map executable source by behavior through the existing restricted interpreter."""
    from worldledger.v04.runtime import run_candidate_v04, validate_source_v04

    validate_source_v04(source)
    return tuple((state, action, *run_candidate_v04(source, state, action))
                 for state in legal_states(spec) for action in ACTIONS)


def source_match(spec: dict, source: str, models: tuple) -> tuple[tuple[int, int, int], ...]:
    signature = canonical_hash(source_behavior_signature(spec, source))
    return tuple(model.assignment for model in models if model.behavior_hash == signature)


def decision_signature(spec: dict, assignment: tuple[int, int, int], horizon: int) -> tuple:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("positive integer horizon required")

    @lru_cache(maxsize=None)
    def value(state: tuple[int, int], remaining: int) -> int:
        if remaining == 0:
            return 0
        return max(step(spec, assignment, state, action)[1] +
                   value(step(spec, assignment, state, action)[0], remaining - 1)
                   for action in ACTIONS)

    return tuple((state, tuple(step(spec, assignment, state, action)[1] +
                               value(step(spec, assignment, state, action)[0], horizon - 1)
                               for action in ACTIONS))
                 for state in legal_states(spec))


def kernel_signature(contract: dict, base: dict,
                     assignment: tuple[int, int, int, int]) -> tuple:
    """Exact ordered outcome distributions, independent of parameter/source names."""
    return tuple((state, action,
                  tuple((prob.numerator, prob.denominator, next_state, reward, observation)
                        for prob, (next_state, reward, observation)
                        in world_kernel(contract, base, assignment, state, action)))
                 for state in legal_states(base) for action in ACTIONS)


def canonical_model_id(contract: dict, base: dict,
                       assignment: tuple[int, int, int, int]) -> str:
    return canonical_hash(kernel_signature(contract, base, assignment))


def source_kernel_signature(base: dict, source: str) -> tuple:
    """Deterministic restricted source as a unit-probability world kernel."""
    from worldledger.v04.runtime import run_candidate_v04, validate_source_v04

    validate_source_v04(source)
    return tuple((state, action,
                  ((1, 1, *run_candidate_v04(source, state, action)),))
                 for state in legal_states(base) for action in ACTIONS)


def kernel_decision_signature(contract: dict, base: dict,
                              assignment: tuple[int, int, int, int], horizon: int) -> tuple:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("positive integer horizon required")

    @lru_cache(maxsize=None)
    def value(state: tuple[int, int], remaining: int) -> Fraction:
        if remaining == 0:
            return Fraction()
        return max(sum((prob * (reward + value(next_state, remaining - 1))
                        for prob, (next_state, reward, _) in
                        world_kernel(contract, base, assignment, state, action)), Fraction())
                   for action in ACTIONS)

    return tuple((state, tuple((q.numerator, q.denominator) for q in (
        sum((prob * (reward + value(next_state, horizon - 1))
             for prob, (next_state, reward, _) in
             world_kernel(contract, base, assignment, state, action)), Fraction())
        for action in ACTIONS))) for state in legal_states(base))
