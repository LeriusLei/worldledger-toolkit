"""Verify the public offline examples and cross-family finite denominators."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finite_family_examples_preserve_denominators():
    result = load_example("finite_worlds").main()
    for row in result["families"].values():
        assert row["support_by_history"] == [4, 2, 1]
        assert row["full_pool_coverage"] == 1
        assert row["full_pool_delta_omit"] == 0
        assert row["empty_pool_delta_omit"] is None
    assert result["machine"]["assignment_count"] == 4


def test_ledger_example_keeps_unknown_cost_and_detects_mutation():
    result = load_example("audit_ledger").main()
    assert result["file_mutation_detected"]
    assert result["total_conservative_cost"] == "2.40"
