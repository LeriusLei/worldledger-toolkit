# Release validation

Local validation for 0.1.0 on 2026-10-30 used Windows and Python 3.12.3. The Linux/Windows × Python 3.10/3.11/3.12 GitHub Actions matrix is configured separately; this local record does not claim those remote jobs have passed.

| Check | Local result |
|---|---|
| Released component, ledger and example tests | 106 passed |
| Sprinkler synthetic experiment | 400-seed run completed; configuration and summary published |
| Component-mixing counterexample | Four pairs enumerated from two complete candidates |
| Finite-world example | Grid/Inventory/Queue support counts 4 → 2 → 1; empty-pool regret remains null |
| Audit-ledger example | Actual input mutation detected; known cost 0.40 and unknown reservation 2.00 retained |
| Independent installation | Editable install in an isolated virtual environment succeeded |
| Wheel construction | Succeeded; finite contracts and both project/upstream license notices included |
| Wheel import outside the checkout | Grid contract enumerated 16 models; Machine Maintenance enumerated 4 |
| Relative Markdown links and extracted-source hashes | Passed |
| Public tree and staged Git blobs | Privacy pattern scan passed |
| Known local credentials | Exact-value comparison against three credential files found no matches in the public tree |
| Provider transport imports | None in the three released packages; only NumPy/SciPy are external runtime imports |
| Figures | Banner, architecture and synthetic-result plot visually inspected |

The public repository starts with new sanitized history. The original archive is separate and remains private. No old commit, raw provider response, model checkpoint or vendored source tree is included. Pattern scanning cannot prove that every possible future contribution is safe; inspect new data before publishing it.
