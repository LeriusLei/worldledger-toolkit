# Offline examples and APIs

Install and run commands from the repository root as shown in the [README](../README.md). Installation downloads only ordinary Python dependencies; the examples themselves do not call model providers or fetch data.

## Finite-world evaluation

```python
from worldledger.benchmark.generators.grid import generate_family_instance
from worldledger.benchmark.history_generator import generate_histories
from worldledger.evaluator.consistency import consistent_models
from worldledger.evaluator.regret import evaluate_pool

world = generate_family_instance(4)
history = generate_histories(world, world.models[0])[0]
pool = [model.model_id for model in world.models]
result = evaluate_pool(world, history, pool, horizon=1)
assert result["hcr"] == 1.0
assert result["delta_omit"] == 0.0
```

`python examples/finite_worlds.py` does this for Grid, Inventory and Queue and prints support counts at three histories, full-pool coverage and an explicit empty-pool result. It also enumerates the independent four-model Machine Maintenance contract and computes a shared horizon-five policy.

## Audit ledger

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from research_audit import ResearchAuditHarness
from research_audit.harness import digest

manifest = {"task_id": "offline-demo"}
for field in ResearchAuditHarness.IDENTITY[1:]:
    manifest[field] = digest(field.encode())  # replace with actual file hashes

with TemporaryDirectory() as directory:
    run = ResearchAuditHarness(Path(directory), manifest)
    run.reserve("request-1", "1.00")
    run.persist_raw("sample-1", b'{"score":0.5}', "application/json")
    run.derived_metric("sample-1", "score", 0.5)
    run.settle("request-1", None)
    assert run.replay()["unknown_reserved_cost"] == "1.00"
```

Costs use one caller-selected currency/unit per run; this is synthetic bookkeeping, not a provider charge. `python examples/audit_ledger.py` uses actual synthetic input-file hashes and demonstrates mutation detection. It writes only to a temporary directory and prints the replay summary.

## Synthetic experiments

```bash
python experiments/run_sprinkler.py --seeds 400 --nuisance-cardinality 16 --output-dir artifacts/sprinkler
python experiments/run_hybrid_counterexample.py --output-dir artifacts/hybrid_counterexample
```

The first writes `summary.json`, `trials.csv`, `RESULTS.md` and optional PNG plots. The second writes `summary.json`, `combinations.csv` and `RESULTS.md`. Experiment `--output-dir` values may record local paths; inspect generated data before sharing your own runs. Published demo summaries use relative paths.

The Sprinkler report's legacy H1/H2/H3 labels refer only to its constructed toy checks. They are separate from similarly named hypotheses in later research protocols, whose historical stop rules are unchanged.

## Installed-wheel use

`python -m build` creates distributions under `dist/`. The packages and finite contract JSON are included. The repository examples/experiments are source-tree entry points, so keep a checkout for those commands. The library imports and contract loaders work from an installed wheel outside the checkout.
