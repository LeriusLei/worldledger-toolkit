"""Tie-aware support audit for crossed executable-model components."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Mapping, Sequence, Tuple

from .candidate_ranking import optimal_action_set


def ranking_signature(
    values: Mapping[Any, float], *, tolerance: float = 1e-12
) -> Tuple[int, ...]:
    actions = list(values)
    signature = []
    for first, second in itertools.combinations(actions, 2):
        delta = float(values[first]) - float(values[second])
        signature.append(0 if abs(delta) <= tolerance else (1 if delta > 0 else -1))
    return tuple(signature)


def audit_hybrid_prediction_support(
    complete_predictions: Mapping[str, Sequence[Mapping[Any, float]]],
    crossed_predictions: Mapping[
        Tuple[str, str], Sequence[Mapping[Any, float]]
    ],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compare crossed transition/reward predictions with complete support."""

    if len(complete_predictions) < 2:
        raise ValueError("at least two complete candidates are required")
    point_counts = {len(rows) for rows in complete_predictions.values()}
    if len(point_counts) != 1 or not point_counts or next(iter(point_counts)) == 0:
        raise ValueError("complete candidate predictions must be aligned and non-empty")
    point_count = next(iter(point_counts))
    candidate_ids = set(complete_predictions)
    unsupported_ranking = 0
    unsupported_top_set = 0
    disjoint_top_set = 0
    introduced_top_action = 0
    outside_envelope = 0
    exact_complete_match = 0
    evaluated = 0
    pair_summaries = []

    for (transition_id, reward_id), rows in crossed_predictions.items():
        if transition_id == reward_id:
            raise ValueError("crossed predictions must use distinct candidate ids")
        if transition_id not in candidate_ids or reward_id not in candidate_ids:
            raise ValueError("crossed pair references an unknown complete candidate")
        if len(rows) != point_count:
            raise ValueError("crossed and complete predictions must align")
        pair_counts = {
            "unsupported_ranking": 0,
            "unsupported_top_set": 0,
            "disjoint_top_set": 0,
            "introduced_top_action": 0,
            "outside_envelope": 0,
            "exact_complete_match": 0,
        }
        for point, crossed in enumerate(rows):
            complete_rows = [
                complete_predictions[candidate_id][point]
                for candidate_id in complete_predictions
            ]
            action_sets = {tuple(row) for row in complete_rows}
            if len(action_sets) != 1 or tuple(crossed) not in action_sets:
                raise ValueError("all predictions must use the same ordered actions")
            complete_rankings = {
                ranking_signature(row, tolerance=tolerance) for row in complete_rows
            }
            complete_top_sets = {
                frozenset(optimal_action_set(row, tolerance=tolerance))
                for row in complete_rows
            }
            complete_top_union = set().union(*complete_top_sets)
            crossed_ranking = ranking_signature(crossed, tolerance=tolerance)
            crossed_top = frozenset(
                optimal_action_set(crossed, tolerance=tolerance)
            )
            ranking_bad = crossed_ranking not in complete_rankings
            top_set_bad = crossed_top not in complete_top_sets
            top_disjoint = not bool(set(crossed_top) & complete_top_union)
            top_introduced = bool(set(crossed_top) - complete_top_union)
            envelope_bad = any(
                float(crossed[action])
                < min(float(row[action]) for row in complete_rows) - tolerance
                or float(crossed[action])
                > max(float(row[action]) for row in complete_rows) + tolerance
                for action in crossed
            )
            exact_match = any(
                all(
                    abs(float(crossed[action]) - float(row[action])) <= tolerance
                    for action in crossed
                )
                for row in complete_rows
            )
            flags = {
                "unsupported_ranking": ranking_bad,
                "unsupported_top_set": top_set_bad,
                "disjoint_top_set": top_disjoint,
                "introduced_top_action": top_introduced,
                "outside_envelope": envelope_bad,
                "exact_complete_match": exact_match,
            }
            for name, flag in flags.items():
                pair_counts[name] += bool(flag)
            evaluated += 1
        unsupported_ranking += pair_counts["unsupported_ranking"]
        unsupported_top_set += pair_counts["unsupported_top_set"]
        disjoint_top_set += pair_counts["disjoint_top_set"]
        introduced_top_action += pair_counts["introduced_top_action"]
        outside_envelope += pair_counts["outside_envelope"]
        exact_complete_match += pair_counts["exact_complete_match"]
        pair_summaries.append(
            {
                "transition_candidate_id": transition_id,
                "reward_candidate_id": reward_id,
                **{
                    f"{name}_rate": value / point_count
                    for name, value in pair_counts.items()
                },
            }
        )
    return {
        "complete_candidate_count": len(complete_predictions),
        "crossed_pair_count": len(crossed_predictions),
        "decision_points": point_count,
        "crossed_pair_point_count": evaluated,
        "unsupported_ranking_rate": (
            unsupported_ranking / evaluated if evaluated else None
        ),
        "unsupported_top_set_rate": (
            unsupported_top_set / evaluated if evaluated else None
        ),
        "disjoint_top_set_rate": disjoint_top_set / evaluated if evaluated else None,
        "introduced_top_action_rate": (
            introduced_top_action / evaluated if evaluated else None
        ),
        "outside_complete_value_envelope_rate": (
            outside_envelope / evaluated if evaluated else None
        ),
        "exact_complete_prediction_match_rate": (
            exact_complete_match / evaluated if evaluated else None
        ),
        "pair_summaries": pair_summaries,
    }
