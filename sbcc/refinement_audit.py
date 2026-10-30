"""Decision-utility audit for scored parent-to-child model refinements."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _combined_score(entry: Mapping[str, Any]) -> float:
    score = entry.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("refinement candidate requires a factual score")
    return (float(score["train_coverage"]) + float(score["test_coverage"])) / 2.0


def summarize_refinement_edges(
    candidates: Sequence[Mapping[str, Any]],
    workers: Sequence[Mapping[str, Any]],
    *,
    score_tolerance: float = 1e-12,
    utility_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compare factual score movement with counterfactual Q movement.

    Edges are resolved only within the same generation batch. A decision-null
    edge has byte-identical canonical Q predictions after independent runtime
    evaluation, even if source code or factual score changed.
    """

    if score_tolerance < 0 or utility_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    worker_by_index: Dict[int, Mapping[str, Any]] = {}
    for worker in workers:
        index = worker.get("candidate_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index in worker_by_index
        ):
            raise ValueError("workers require unique integer candidate indices")
        worker_by_index[index] = worker
    if any(index < 0 or index >= len(candidates) for index in worker_by_index):
        raise ValueError("worker candidate index is outside the pool")

    candidate_by_source = {}
    for index, candidate in enumerate(candidates):
        key = (
            candidate.get("source_batch_index", 0),
            candidate.get("source_path"),
        )
        if key[1] is not None:
            candidate_by_source[key] = (index, candidate)

    edges = []
    unresolved = 0
    for child_index, child in enumerate(candidates):
        parent_source = child.get("resolved_parent_source_path")
        if not parent_source:
            continue
        parent_record = candidate_by_source.get(
            (child.get("source_batch_index", 0), parent_source)
        )
        if parent_record is None:
            unresolved += 1
            continue
        parent_index, parent = parent_record
        parent_worker = worker_by_index.get(parent_index)
        child_worker = worker_by_index.get(child_index)
        runtime_comparable = (
            parent_worker is not None
            and child_worker is not None
            and parent_worker.get("status") == "success"
            and child_worker.get("status") == "success"
        )
        edge: Dict[str, Any] = {
            "source_batch_index": child.get("source_batch_index", 0),
            "parent_candidate_index": parent_index,
            "child_candidate_index": child_index,
            "parent_source_path": parent_source,
            "child_source_path": child.get("source_path"),
            "runtime_comparable": runtime_comparable,
        }
        if runtime_comparable:
            if parent_worker.get("code_sha256") != parent.get("code_sha256"):
                raise ValueError("parent worker code hash mismatch")
            if child_worker.get("code_sha256") != child.get("code_sha256"):
                raise ValueError("child worker code hash mismatch")
            parent_score = _combined_score(parent)
            child_score = _combined_score(child)
            parent_accuracy = float(
                parent_worker["metrics"]["mean_pairwise_ranking_accuracy"]
            )
            child_accuracy = float(
                child_worker["metrics"]["mean_pairwise_ranking_accuracy"]
            )
            parent_prediction_sha = parent_worker.get("prediction_sha256")
            child_prediction_sha = child_worker.get("prediction_sha256")
            if not parent_prediction_sha or not child_prediction_sha:
                raise ValueError("runtime-success workers require prediction SHA-256")
            score_delta = child_score - parent_score
            utility_delta = child_accuracy - parent_accuracy
            edge.update(
                {
                    "parent_factual_score": parent_score,
                    "child_factual_score": child_score,
                    "factual_score_delta": score_delta,
                    "parent_pairwise_ranking_accuracy": parent_accuracy,
                    "child_pairwise_ranking_accuracy": child_accuracy,
                    "counterfactual_utility_delta": utility_delta,
                    "factual_score_improved": score_delta > score_tolerance,
                    "counterfactual_utility_improved": utility_delta
                    > utility_tolerance,
                    "counterfactual_utility_regressed": utility_delta
                    < -utility_tolerance,
                    "decision_null_refinement": parent_prediction_sha
                    == child_prediction_sha,
                    "parent_prediction_sha256": parent_prediction_sha,
                    "child_prediction_sha256": child_prediction_sha,
                }
            )
        edges.append(edge)

    comparable = [edge for edge in edges if edge["runtime_comparable"]]
    score_improving = [edge for edge in comparable if edge["factual_score_improved"]]
    decision_null = [edge for edge in comparable if edge["decision_null_refinement"]]
    score_improving_null = [
        edge for edge in score_improving if edge["decision_null_refinement"]
    ]
    score_improving_regressed = [
        edge for edge in score_improving if edge["counterfactual_utility_regressed"]
    ]
    return {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "resolved_refinement_edge_count": len(edges),
        "unresolved_parent_edge_count": unresolved,
        "runtime_comparable_edge_count": len(comparable),
        "score_improving_edge_count": len(score_improving),
        "decision_null_refinement_count": len(decision_null),
        "decision_null_refinement_rate": (
            len(decision_null) / len(comparable) if comparable else None
        ),
        "score_improving_decision_null_count": len(score_improving_null),
        "score_improving_decision_null_rate": (
            len(score_improving_null) / len(score_improving)
            if score_improving
            else None
        ),
        "score_improving_utility_regression_count": len(score_improving_regressed),
        "score_improving_utility_regression_rate": (
            len(score_improving_regressed) / len(score_improving)
            if score_improving
            else None
        ),
        "edges": edges,
        "claim_boundary": (
            "edge counts are descriptive within generation trees; publishable claims "
            "require independent pools/environments and source-batch clustered uncertainty"
        ),
    }
