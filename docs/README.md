# Documentation

Start with a runnable control, then follow the interfaces you want to reuse.

| Your goal | Start here | Next |
|---|---|---|
| Run the first exact evaluation | [Quick start](QUICKSTART.md) | [Example gallery](EXAMPLES.md) |
| Build a candidate/decision experiment | [System architecture](ARCHITECTURE.md) | [Finite tasks](../worldledger/benchmark/) · [Decision components](../sbcc/) |
| Keep an experiment auditable | [Ledger example](../examples/audit_ledger.py) | [Ledger implementation](../research_audit/harness.py) |
| Interpret a result | [Evidence boundaries](EVIDENCE.md) | [Demo summaries](demo_results/) · [Validation](VALIDATION.md) |
| Understand the release | [Project map](PROJECT_MAP.md) | [Source hashes](SOURCE_MANIFEST.json) · [Changelog](../CHANGELOG.md) |
| Contribute code or a control | [Contributing](../CONTRIBUTING.md) | [Security](../SECURITY.md) · [Visual design](DESIGN.md) |

## Reading the APIs

- **Finite worlds:** `generate_family_instance` → `generate_histories` → `consistent_models` → `evaluate_pool`.
- **Policies and uncertainty:** `DecisionCommittee`, `EpisodeConformalActionCalibrator`, and shared-policy finite-horizon evaluators.
- **Execution records:** `ResearchAuditHarness.inspect_files` → `persist_raw` → derived metrics → `replay`.

The package docs and tests are the implementation reference. Generic adapters require you to supply the relevant external environment interface; vendored third-party systems and live provider transports are not bundled.
