"""Focused offline tests kept within the v0.6.1 authorized code directory."""

from copy import deepcopy

import pytest

from worldledger.fixtures.mock_candidates import GRID4_M00
from worldledger.world_model.canonical import (
    canonical_hash, kernel_signature, source_kernel_signature,
)
from worldledger.world_model.enumerator import enumerate_world_contract
from worldledger.world_model.regret import eligible_pool_metrics, kernel_omission_delta
from worldledger.world_model.validator import (
    ContractError, load_world_contract, validate_world_contract, world_kernel,
)


def test_contract_rejects_probability_and_observation_mutation():
    contract, base = load_world_contract()
    changed = deepcopy(contract)
    changed["parameters"][3]["domain"] = [0, 1, 2]
    with pytest.raises(ContractError):
        validate_world_contract(changed, base)
    changed = deepcopy(contract)
    changed["observation"]["probe_a_mapping"][-1] = "mechanism_a"
    with pytest.raises(ContractError):
        validate_world_contract(changed, base)
    with pytest.raises(ContractError):
        world_kernel(contract, base, (0, 0, -1, 2), (1, 1), "east")


def test_enumeration_complete_and_rational_kernel_normalized():
    contract, base = load_world_contract()
    enumeration = enumerate_world_contract(contract, base)
    assert enumeration.assignment_count == enumeration.number_of_models == 16
    assert len(set(enumeration.canonical_hashes)) == 16
    kernel = world_kernel(contract, base, (0, 0, -1, 1), (1, 1), "east")
    assert len(kernel) == 2
    assert sorted(prob for prob, _ in kernel) == [1 / 2, 1 / 2]


def test_source_text_independence_and_behavior_membership():
    contract, base = load_world_contract()
    source_a = source_kernel_signature(base, GRID4_M00)
    source_b = source_kernel_signature(base, "# Formatting variant\n" + GRID4_M00)
    target = kernel_signature(contract, base, (0, 0, -1, 0))
    assert source_a == source_b == target
    assert canonical_hash(source_a) in enumerate_world_contract(contract, base).canonical_hashes


def test_regret_rejects_off_contract_pool_and_has_positive_control():
    contract, base = load_world_contract()
    models = enumerate_world_contract(contract, base).unique_models
    full = {model.canonical_model_id for model in models}
    subset = {model.canonical_model_id for model in models
              if model.assignment[:2] != (0, 1)}
    result = kernel_omission_delta(contract, base, full, subset)
    assert result["delta_omit"] == [3, 1]
    with pytest.raises(ValueError):
        kernel_omission_delta(contract, base, full, subset | {"off-contract"})
    with pytest.raises(ValueError):
        kernel_omission_delta(contract, base, subset, subset)
    assert eligible_pool_metrics(contract, base, subset | {"off-contract"}) == {
        "coverage": None, "delta_omit": None, "reason": "off_contract_candidate"}
    assert eligible_pool_metrics(contract, base, set()) == {
        "coverage": None, "delta_omit": None, "reason": "empty_candidate_pool"}
