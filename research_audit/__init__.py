"""Domain-independent execution audit tools for research stopping studies."""

from .harness import ResearchAuditHarness, AuditError

__all__ = ["ResearchAuditHarness", "AuditError"]
