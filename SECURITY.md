# Security and privacy

For a suspected vulnerability or accidental credential exposure, use GitHub's private vulnerability reporting feature when it is available for this repository. Otherwise, open an issue requesting a private contact channel without including the vulnerability details or secret. Rotate any exposed credential at its issuer; deleting a file alone does not revoke it.

This release ships no provider transport, stored API key, private conversation, raw provider response, model checkpoint or private dataset. Examples are offline. Installation and CI use ordinary Python package dependencies and GitHub Actions.

The AST interpreters implement a deliberately small research language. They are not audited isolation for hostile code and should not be exposed as a multi-tenant execution service. Evaluator-only channels rely on trusted closures and Python process discipline. The ledger is single-writer and does not provide external signatures, encrypted storage, concurrency control or a hard provider-side spending cap. Persisted raw bytes may contain whatever the caller supplies; redact private material before recording or sharing it.

Run `python scripts/privacy_check.py` before committing and again on the final release tree. Its pattern checks are a defense in depth, not proof that arbitrary future contributions are free of private data.
