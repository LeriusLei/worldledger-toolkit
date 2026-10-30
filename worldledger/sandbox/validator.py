"""One terminal status per requested candidate slot; no silent deletion."""

from worldledger.sandbox.executor import CandidateExecutionError, CandidateTimeout, run_candidate
from worldledger.sandbox.policy import SourceRejected, validate_source


STATUSES = (
    "syntax_failed", "execution_failed", "timeout", "history_inconsistent",
    "valid_duplicate", "valid_new_behavior_class", "valid_new_decision_class",
)


def classify_candidate(source: str, history_cases: list[dict], probe_cases: list[dict],
                       seen_behaviors: set, seen_decisions: set, decision_signature=None,
                       timeout_seconds: float = 0.5) -> dict:
    """Classify one slot; behavior uses registered probes, not seen history."""
    try:
        validate_source(source)
    except SourceRejected as exc:
        return {"status": "syntax_failed", "detail": str(exc), "behavior": None}
    outcomes = []
    try:
        for case in history_cases:
            actual = run_candidate(source, tuple(case["state"]), case["action"], timeout_seconds)
            if actual != (tuple(case["next_state"]), case["reward"], tuple(case["observation"])):
                return {"status": "history_inconsistent", "detail": "observed transition mismatch", "behavior": None}
        for case in probe_cases:
            outcomes.append((tuple(case["state"]), case["action"],
                             run_candidate(source, tuple(case["state"]), case["action"], timeout_seconds)))
    except CandidateTimeout as exc:
        return {"status": "timeout", "detail": str(exc), "behavior": None}
    except CandidateExecutionError as exc:
        return {"status": "execution_failed", "detail": str(exc), "behavior": None}
    except Exception as exc:
        return {"status": "execution_failed", "detail": f"evaluator input error: {type(exc).__name__}", "behavior": None}
    behavior = tuple(outcomes)
    try:
        decision = decision_signature(behavior) if decision_signature else None
        if decision is not None:
            hash(decision)
    except Exception as exc:
        return {"status": "execution_failed", "detail": f"decision signature error: {type(exc).__name__}", "behavior": None}
    if behavior in seen_behaviors:
        status = "valid_duplicate"
    elif decision is not None and decision not in seen_decisions:
        status = "valid_new_decision_class"
    else:
        status = "valid_new_behavior_class"
    seen_behaviors.add(behavior)
    if decision is not None:
        seen_decisions.add(decision)
    return {"status": status, "detail": None, "behavior": behavior}
