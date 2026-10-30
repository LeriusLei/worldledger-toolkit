# Architecture and module guide

![Information flow](assets/architecture.svg)

## Finite models and exact evaluation

`worldledger.benchmark.generators` constructs Grid, Inventory and Queue instances with finite candidate families. `history_generator` creates public factual histories. `consistency` filters candidates using those histories; `behavioral_quotient` groups models by behavior and decision signatures under the selected probe/horizon convention.

`evaluate_pool` reports behavioral and decision coverage, history inconsistency, generation failures and regret. It keeps the generated pool distinct from its history-consistent subset: deployment regret and omission regret answer different questions. Empty pools and failures remain represented; undefined regret is `None`, not a favorable zero.

`worldledger.evaluator.horizon` chooses one shared policy that branches on public reward/observation. Model-wise full-information optima are comparator values, not information supplied to the policy. Exact enumeration grows rapidly with horizon, support and branching; the implementation targets small finite controls.

`worldledger.world_model` includes a bounded Grid DSL, rational stochastic kernels and an independent Machine Maintenance contract. `multistep` enumerates a finite Pareto frontier for nonanticipative minimax regret. The frozen contract JSON is package data, so these APIs also work from an installed wheel.

## Decision components (`sbcc`)

| Files | Role |
|---|---|
| `sprinkler`, `agents` | Finite pedagogical environment and seven policies including an evaluator oracle |
| `decision_ensemble`, `hybrid_ensemble`, `hybrid_support` | Coherent candidate aggregation and component-mixing diagnostics |
| `candidate_ranking`, `candidate_noise_diagnostic` | Ranking/selection summaries and noise controls |
| `conformal_action_set`, `statistical_analysis` | Episode-level action calibration and paired/binomial inference |
| `information_firewall`, `closed_loop_artifact`, `closed_loop_runner`, `closed_loop_policies` | Online traces, post-episode evaluator truth, paired execution and generic adapters |
| `fork_oracle`, `refinement_audit` | Trusted evaluator forking and lineage summaries |
| `imagined_rollouts` | Candidate-local imagination, lambda returns and categorical CEM |

The adapter names containing `Pinductor` preserve original APIs. Their released tests use synthetic stubs, and no third-party environment is automatically imported or downloaded. You must implement the environment/candidate interfaces for an external task. The pilot-artifact builder is a data-contract helper; historical preregistration, approval and paid-execution controllers are excluded.

Episode-level calibration requires an appropriate separate calibration set and exchangeability assumptions for its usual interpretation. It is not an online single-branch access rule: evaluator truth used in offline calibration cannot silently become available to a deployed agent.

## Information boundary

An online policy observes the executed branch only. A trusted evaluator may fork a simulator to score alternatives, but `EvaluatorOnlyChannel.evaluate()` returns no value. It deep-copies the payload and restores Python/NumPy random states. Results are accessible after `seal()`. The paired-trace check compares online behavior with evaluation enabled and disabled.

This is an in-process discipline. A closure retaining mutable agent/environment references can violate it. Other RNG libraries, process-global state and arbitrary external side effects are outside the channel's isolation contract. Use independent processes when that isolation is required.

## Bounded source interpreters

`worldledger.sandbox` accepts a tiny three-function source contract. The `v04` runtime additionally permits bounded local assignment. Both validate syntax and interpret AST nodes instead of using Python `exec` or `eval`. They impose logical size, depth, step and time limits.

These are research interpreters, not audited containment for hostile code. Host Python/runtime behavior and memory allocation still matter. Keep untrusted execution behind an appropriate external isolation boundary.

## Audit ledger

`ResearchAuditHarness` fixes six identity fields: task, protocol, metric, dataset, evaluator and threshold. `inspect_files()` hashes the actual files. Raw artifacts are persisted before derived metrics; every event links to its predecessor and manifest. `replay()` checks hashes and re-reads raw files. Reservations use decimal amounts; unanswered/unknown settlements keep their maximum reservation.

The ledger records execution facts. Scientific interpretation and stopping decisions belong outside it. It has no transport, provider authentication or automatic model call. It is single-writer, does not lock concurrent writers, and has no external trust anchor; someone able to rewrite every file can recompute the chain. A crash during an append can leave a partial line, which should be investigated rather than silently discarded. It also does not enforce a total spending limit or replace provider-side caps.
