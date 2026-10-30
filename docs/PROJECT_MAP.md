# Project map and release choices

The release extracts engineering value from a larger research archive into a small, independently installable repository. It uses an explicit file allowlist and fresh Git history.

| Original area | Public treatment | Reason |
|---|---|---|
| SBCC finite decision experiments | Sprinkler, coherent committees, mixing controls and reusable statistical/trace utilities included | Small, offline and reusable |
| WorldLedger/DRHC finite worlds | Task generators, exact evaluators, bounded interpreters and minimal frozen contracts included | Explicit finite interfaces with exact controls |
| Domain-neutral research audit | `research_stopping.harness` exposed as `research_audit` | Useful independently of a scientific stopping hypothesis |
| Imagination pilot | NumPy return/rollout/CEM utilities included with upstream notice | Component implementation; no full Dreamer training claim |
| Battery and Neuron simulations | Original domain-specific integrations excluded | Require separately versioned external simulators and evidence |
| Pinductor, PatchWorld and other external systems | Generic adapters only; vendored source and generated candidates excluded | Different source/runtime/license contracts; avoid redistributing an opaque vendor bundle |
| Paid/model-provider execution | All live transports, grants, credentials, raw responses and provider billing logs excluded | Public examples are offline and independently reproducible |
| DRHC/SBS proposals and later audit campaigns | Research lessons summarized in `EVIDENCE.md`; frozen private runs excluded | Preserve negative gates and avoid unsupported benchmark claims |
| Chat exports, manuscripts, local planning, caches and environment folders | Excluded | Private or machine-specific rather than reusable release material |
| Original repository history | Excluded | A clean tree does not remove secrets from old commits; this is a separate repository |

## Why one repository

Three packages share the same research interfaces: `worldledger` for finite evaluation, `sbcc` for decision controls and `research_audit` for execution records. A single initial release makes their integration reviewable. Later splitting is possible once each has independent users and versioning needs.

Calling the whole release a benchmark would require a stable task distribution, externally meaningful endpoints, independent instances and published comparison protocols. Calling it a small-model release would require trained weights, training/data provenance and model-specific evaluation. Those are not the contents of this release. The finite synthetic suites remain useful as transparent controls within the toolkit.

## Source provenance and adaptations

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) maps each extracted file to a nonpersonal relative source path and records source/release SHA-256. It contains no original commit history or private absolute paths. New documentation, examples, figures, packaging, privacy checks and audit tests are release additions.

Adaptations include normalizing text to LF, renaming the audit package, relocating hand-authored candidate fixtures, updating the Dreamer notice path, clarifying the toy runner's report wording and removing four archive-only test methods: one historical preregistration integration, two excluded Pinductor diagnostic-runner integrations and one direct comparison against the vendored Dreamer file. The corresponding released components retain their offline unit tests; unavailable archive integration is not reported as passing. Changes required by independent packaging, including line endings, are recorded as differing release hashes. Production algorithms were not retuned to improve the demo.
