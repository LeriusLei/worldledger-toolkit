"""HypoSpace-inspired, oracle-blind AST-stratified candidate budget adapter.

This is an inspired adaptation, not a reproduction. It never inspects oracle
support, true behavioral classes, hidden distribution or decision regret.
"""

import ast
from dataclasses import dataclass

from worldledger.sandbox.policy import SourceRejected, validate_source


STRATA = ("0-1", "2-3", "4_or_more")


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_id: str
    source: str | None


@dataclass(frozen=True)
class SubsetSelection:
    requested_budget: int
    selected_candidate_ids: tuple[str, ...]
    unselected_candidate_ids: tuple[str, ...]
    invalid_candidate_ids: tuple[str, ...]
    selected_strata: tuple[str, ...]
    unfilled_slots: int


def assigned_strata(candidate_budget: int) -> tuple[str, ...]:
    if type(candidate_budget) is not int or candidate_budget < 1:
        raise ValueError("positive integer candidate budget required")
    return tuple(STRATA[index % len(STRATA)] for index in range(candidate_budget))


def source_stratum(source: str) -> str:
    functions = validate_source(source)
    count = sum(isinstance(node, (ast.If, ast.IfExp))
                for function in functions.values() for node in ast.walk(function))
    return "0-1" if count <= 1 else "2-3" if count <= 3 else "4_or_more"


def select_budgeted_subset(candidate_budget: int,
                           generated_candidates: tuple[GeneratedCandidate, ...]) -> SubsetSelection:
    """Select fixed-stratum subset; retain every input ID in accounting.

    For the registered B3 arm, there are exactly three preassigned requested
    slots. If an output misses its requested stratum, it remains in the raw
    valid pool; this selector never hides it from denominator/cost accounting.
    """
    targets = assigned_strata(candidate_budget)
    identifiers = [item.candidate_id for item in generated_candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate candidate IDs")
    valid: list[tuple[str, str]] = []
    invalid: list[str] = []
    for item in generated_candidates:
        try:
            if item.source is None:
                raise SourceRejected("missing source")
            valid.append((item.candidate_id, source_stratum(item.source)))
        except SourceRejected:
            invalid.append(item.candidate_id)
    remaining = sorted(valid)
    selected: list[tuple[str, str]] = []
    for target in targets:
        if not remaining:
            break
        match_index = next((index for index, (_, stratum) in enumerate(remaining)
                            if stratum == target), 0)
        selected.append(remaining.pop(match_index))
    return SubsetSelection(candidate_budget, tuple(item[0] for item in selected),
                           tuple(item[0] for item in remaining), tuple(sorted(invalid)),
                           tuple(item[1] for item in selected), candidate_budget - len(selected))
