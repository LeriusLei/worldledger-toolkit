# Contributing

Open an issue describing the task, expected behavior, actual output and minimal offline reproduction. Avoid attaching credentials, private datasets or unredacted provider logs.

For changes:

1. Install `python -m pip install -e ".[dev,plots]"`.
2. Keep a focused change, with meaningful regression checks for changed behavior.
3. Run `python -m pytest -q`, `python scripts/privacy_check.py` and `python scripts/check_docs.py`.
4. Explain which contract or evidence claim changes, and include the complete denominator for any reported result.

Preserve deterministic tie breaking, failed/empty pools, evaluator-only truth boundaries and unknown-cost reservations. Put generated runs under `artifacts/`. Do not treat new seeds from one generation run as independent learned models. New empirical claims need their own protocol, independent evaluation and provenance.

The codebase is experimental. The legacy names `sbcc`, `v04` and `FalsifyWM` preserve source/API continuity; they are not separate maturity or novelty claims. Package releases should update version, changelog, citation and validation record together.
