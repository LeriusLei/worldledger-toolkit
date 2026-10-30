"""Tie-aware diagnostics for factual scores versus counterfactual rankings."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _check_values(values: Mapping[Any, float]) -> Dict[Any, float]:
    if not values:
        raise ValueError("action-value mapping cannot be empty")
    checked = {action: float(value) for action, value in values.items()}
    if not all(math.isfinite(value) for value in checked.values()):
        raise ValueError("action values must be finite")
    return checked


def optimal_action_set(
    values: Mapping[Any, float], *, tolerance: float = 1e-12
) -> Tuple[Any, ...]:
    checked = _check_values(values)
    best = max(checked.values())
    return tuple(
        action for action, value in checked.items() if best - value <= tolerance
    )


def pairwise_ranking_accuracy(
    truth: Mapping[Any, float],
    prediction: Mapping[Any, float],
    *,
    tolerance: float = 1e-12,
) -> Optional[float]:
    """Score ordered action pairs; a predicted tie receives half credit."""

    truth_values = _check_values(truth)
    predicted_values = _check_values(prediction)
    if set(truth_values) != set(predicted_values):
        raise ValueError("truth and prediction must contain identical action sets")
    earned = 0.0
    comparable = 0
    actions = list(truth_values)
    for first, second in itertools.combinations(actions, 2):
        truth_delta = truth_values[first] - truth_values[second]
        if abs(truth_delta) <= tolerance:
            continue
        predicted_delta = predicted_values[first] - predicted_values[second]
        comparable += 1
        if abs(predicted_delta) <= tolerance:
            earned += 0.5
        elif (truth_delta > 0) == (predicted_delta > 0):
            earned += 1.0
    return earned / comparable if comparable else None


def score_action_predictions(
    truth_points: Sequence[Mapping[Any, float]],
    predicted_points: Sequence[Mapping[Any, float]],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    if len(truth_points) != len(predicted_points) or not truth_points:
        raise ValueError("truth and prediction must have equal non-zero length")
    pairwise_scores: List[float] = []
    top_overlap = 0
    exact_top_set = 0
    top_precisions = []
    top_recalls = []
    uniform_regrets = []
    worst_regrets = []
    uninformative = 0
    for truth, prediction in zip(truth_points, predicted_points):
        truth_values = _check_values(truth)
        predicted_values = _check_values(prediction)
        if set(truth_values) != set(predicted_values):
            raise ValueError("all aligned points must use identical action sets")
        pairwise = pairwise_ranking_accuracy(
            truth_values, predicted_values, tolerance=tolerance
        )
        if pairwise is not None:
            pairwise_scores.append(pairwise)
        truth_best = set(optimal_action_set(truth_values, tolerance=tolerance))
        predicted_best = set(
            optimal_action_set(predicted_values, tolerance=tolerance)
        )
        overlap = truth_best & predicted_best
        top_overlap += bool(overlap)
        exact_top_set += truth_best == predicted_best
        top_precisions.append(len(overlap) / len(predicted_best))
        top_recalls.append(len(overlap) / len(truth_best))
        best_truth_value = max(truth_values.values())
        selected_truth = [truth_values[action] for action in predicted_best]
        uniform_regrets.append(
            best_truth_value - sum(selected_truth) / len(selected_truth)
        )
        worst_regrets.append(best_truth_value - min(selected_truth))
        if max(predicted_values.values()) - min(predicted_values.values()) <= tolerance:
            uninformative += 1
    count = len(truth_points)
    return {
        "decision_points": count,
        "mean_pairwise_ranking_accuracy": (
            sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else None
        ),
        "pairwise_scorable_points": len(pairwise_scores),
        "top_set_overlap_rate": top_overlap / count,
        "exact_top_set_rate": exact_top_set / count,
        "mean_top_set_precision": sum(top_precisions) / count,
        "mean_top_set_recall": sum(top_recalls) / count,
        "mean_uniform_tie_regret": sum(uniform_regrets) / count,
        "mean_worst_tie_regret": sum(worst_regrets) / count,
        "uninformative_prediction_rate": uninformative / count,
    }


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = average
        position = end
    return ranks


def spearman_correlation(
    first: Sequence[float], second: Sequence[float]
) -> Optional[float]:
    if len(first) != len(second) or len(first) < 2:
        return None
    rank_first = _average_ranks([float(value) for value in first])
    rank_second = _average_ranks([float(value) for value in second])
    mean_first = sum(rank_first) / len(rank_first)
    mean_second = sum(rank_second) / len(rank_second)
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(rank_first, rank_second)
    )
    denominator_left = sum((value - mean_first) ** 2 for value in rank_first)
    denominator_right = sum((value - mean_second) ** 2 for value in rank_second)
    denominator = math.sqrt(denominator_left * denominator_right)
    return numerator / denominator if denominator > 0 else None


def summarize_score_accuracy_relation(
    candidate_summaries: Sequence[Mapping[str, Any]],
    *,
    minimum_score_gap: float = 0.02,
    minimum_accuracy_gap: float = 0.05,
    minimum_selection_regret: float = 0.05,
) -> Dict[str, Any]:
    usable = [
        summary
        for summary in candidate_summaries
        if summary.get("factual_score") is not None
        and summary.get("mean_pairwise_ranking_accuracy") is not None
    ]
    scores = [float(summary["factual_score"]) for summary in usable]
    accuracies = [
        float(summary["mean_pairwise_ranking_accuracy"]) for summary in usable
    ]
    inversions = []
    comparable_pairs = 0
    for first, second in itertools.combinations(usable, 2):
        score_gap = float(first["factual_score"]) - float(second["factual_score"])
        accuracy_gap = float(first["mean_pairwise_ranking_accuracy"]) - float(
            second["mean_pairwise_ranking_accuracy"]
        )
        if abs(score_gap) <= 1e-12:
            continue
        comparable_pairs += 1
        if score_gap * accuracy_gap < 0:
            inversions.append(
                {
                    "candidate_a": first["candidate_id"],
                    "candidate_b": second["candidate_id"],
                    "score_gap_a_minus_b": score_gap,
                    "accuracy_gap_a_minus_b": accuracy_gap,
                    "decisive": abs(score_gap) >= minimum_score_gap
                    and abs(accuracy_gap) >= minimum_accuracy_gap,
                }
            )
    rho = spearman_correlation(scores, accuracies)
    decisive_count = sum(bool(item["decisive"]) for item in inversions)
    map_index = max(range(len(usable)), key=lambda index: scores[index]) if usable else None
    best_accuracy = max(accuracies) if accuracies else None
    map_accuracy = accuracies[map_index] if map_index is not None else None
    selection_regret = (
        best_accuracy - map_accuracy
        if best_accuracy is not None and map_accuracy is not None
        else None
    )
    selection_failure = (
        selection_regret is not None
        and selection_regret + 1e-12 >= minimum_selection_regret
    )
    screening_signal = (
        (rho is not None and rho <= 0.3)
        or decisive_count > 0
        or selection_failure
    )
    return {
        "usable_candidate_count": len(usable),
        "score_accuracy_spearman": rho,
        "comparable_candidate_pairs": comparable_pairs,
        "score_accuracy_inversion_count": len(inversions),
        "score_accuracy_inversion_rate": (
            len(inversions) / comparable_pairs if comparable_pairs else None
        ),
        "decisive_inversion_count": decisive_count,
        "decisive_thresholds": {
            "minimum_factual_score_gap": minimum_score_gap,
            "minimum_ranking_accuracy_gap": minimum_accuracy_gap,
        },
        "map_candidate_id": (
            usable[map_index]["candidate_id"] if map_index is not None else None
        ),
        "map_ranking_accuracy": map_accuracy,
        "best_candidate_ranking_accuracy": best_accuracy,
        "map_selection_regret": selection_regret,
        "minimum_meaningful_selection_regret": minimum_selection_regret,
        "map_selection_failure_signal": selection_failure,
        "inversions": inversions,
        "screening_signal": screening_signal,
        "replication_ready_signal": screening_signal and len(usable) >= 8,
        "publishable_signal": False,
        "publishable_signal_reason": (
            "one candidate pool cannot establish publishable replication"
        ),
        "interpretation": (
            "screening_signal only authorizes replication; publishable_signal "
            "still requires at least two independently generated pools"
        ),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile needs non-empty values and probability in [0,1]")
    location = (len(sorted_values) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(sorted_values[lower])
    weight = location - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def episode_cluster_bootstrap_map_selection_regret(
    truth_points: Sequence[Mapping[Any, float]],
    candidate_predictions: Sequence[Sequence[Mapping[Any, float]]],
    factual_scores: Sequence[float],
    episode_ids: Sequence[Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_830,
    meaningful_threshold: float = 0.05,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Bootstrap MAP selection regret by resampling whole offline episodes."""

    point_count = len(truth_points)
    if point_count == 0 or len(episode_ids) != point_count:
        raise ValueError("truth points and episode ids must align and be non-empty")
    if not candidate_predictions or len(candidate_predictions) != len(factual_scores):
        raise ValueError("one factual score is required per candidate")
    if any(len(rows) != point_count for rows in candidate_predictions):
        raise ValueError("candidate predictions must align with truth points")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    clusters = list(dict.fromkeys(episode_ids))
    if len(clusters) < 2:
        raise ValueError("at least two episode clusters are required")
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    sums = [[0.0 for _ in clusters] for _ in candidate_predictions]
    counts = [[0 for _ in clusters] for _ in candidate_predictions]
    for point, (truth, episode_id) in enumerate(zip(truth_points, episode_ids)):
        cluster = cluster_index[episode_id]
        for candidate, rows in enumerate(candidate_predictions):
            value = pairwise_ranking_accuracy(
                truth, rows[point], tolerance=tolerance
            )
            if value is not None:
                sums[candidate][cluster] += value
                counts[candidate][cluster] += 1

    map_index = max(
        range(len(factual_scores)), key=lambda index: float(factual_scores[index])
    )

    def regret_for(sampled_clusters: Sequence[int]) -> float:
        accuracies = []
        for candidate in range(len(candidate_predictions)):
            total = sum(sums[candidate][cluster] for cluster in sampled_clusters)
            count = sum(counts[candidate][cluster] for cluster in sampled_clusters)
            if count == 0:
                raise ValueError("bootstrap sample has no pairwise-scorable points")
            accuracies.append(total / count)
        return max(accuracies) - accuracies[map_index]

    observed = regret_for(list(range(len(clusters))))
    generator = random.Random(seed)
    bootstrap = sorted(
        regret_for(
            [generator.randrange(len(clusters)) for _ in range(len(clusters))]
        )
        for _ in range(bootstrap_samples)
    )
    return {
        "method": "episode-cluster percentile bootstrap",
        "cluster_count": len(clusters),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "observed_map_selection_regret": observed,
        "confidence_level": 0.95,
        "confidence_interval": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "bootstrap_probability_positive": sum(value > 0 for value in bootstrap)
        / bootstrap_samples,
        "meaningful_threshold": meaningful_threshold,
        "bootstrap_probability_at_least_meaningful": sum(
            value + 1e-12 >= meaningful_threshold for value in bootstrap
        )
        / bootstrap_samples,
        "independence_warning": (
            "resamples whole offline episodes; candidate-generation batch "
            "stability must be reported separately"
        ),
    }


def leave_one_generation_batch_out_selection_regret(
    candidate_summaries: Sequence[Mapping[str, Any]],
    *,
    meaningful_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Check whether MAP selection failure survives each source-batch deletion."""

    usable = [
        summary
        for summary in candidate_summaries
        if summary.get("factual_score") is not None
        and summary.get("mean_pairwise_ranking_accuracy") is not None
        and summary.get("source_batch_index") is not None
    ]
    batches = sorted({int(summary["source_batch_index"]) for summary in usable})
    if len(batches) < 2:
        return {
            "status": "insufficient_generation_batches",
            "source_batch_count": len(batches),
            "deletions": [],
        }
    deletions = []
    for excluded in batches:
        remaining = [
            summary
            for summary in usable
            if int(summary["source_batch_index"]) != excluded
        ]
        if len(remaining) < 2:
            continue
        scores = [float(summary["factual_score"]) for summary in remaining]
        accuracies = [
            float(summary["mean_pairwise_ranking_accuracy"])
            for summary in remaining
        ]
        map_index = max(range(len(remaining)), key=lambda index: scores[index])
        regret = max(accuracies) - accuracies[map_index]
        deletions.append(
            {
                "excluded_source_batch_index": excluded,
                "remaining_candidate_count": len(remaining),
                "map_candidate_id": remaining[map_index]["candidate_id"],
                "map_selection_regret": regret,
                "at_least_meaningful": regret + 1e-12 >= meaningful_threshold,
            }
        )
    regrets = [row["map_selection_regret"] for row in deletions]
    return {
        "status": "evaluated" if deletions else "insufficient_candidates_after_deletion",
        "source_batch_count": len(batches),
        "meaningful_threshold": meaningful_threshold,
        "deletions": deletions,
        "minimum_leave_one_batch_out_regret": min(regrets) if regrets else None,
        "maximum_leave_one_batch_out_regret": max(regrets) if regrets else None,
        "all_deletions_at_least_meaningful": (
            all(row["at_least_meaningful"] for row in deletions)
            if deletions
            else None
        ),
    }


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = [(value - max(values)) / temperature for value in values]
    exponents = [math.exp(value) for value in shifted]
    total = sum(exponents)
    return [value / total for value in exponents]


def build_simple_baseline_predictions(
    candidate_predictions: Sequence[Sequence[Mapping[Any, float]]],
    factual_scores: Sequence[float],
    *,
    softmax_temperature: float = 0.1,
    tolerance: float = 1e-12,
) -> Dict[str, List[Dict[Any, float]]]:
    """Construct deliberately simple MAP/BMA/vote/minimax comparators."""

    if not candidate_predictions or len(candidate_predictions) != len(factual_scores):
        raise ValueError("one factual score is required per candidate")
    point_count = len(candidate_predictions[0])
    if point_count == 0 or any(len(rows) != point_count for rows in candidate_predictions):
        raise ValueError("candidate predictions must be non-empty and aligned")
    map_index = max(range(len(factual_scores)), key=lambda index: factual_scores[index])
    weights = _softmax([float(score) for score in factual_scores], softmax_temperature)
    result: Dict[str, List[Dict[Any, float]]] = {
        "map": [],
        "uniform_bma": [],
        "score_softmax_bma": [],
        "fractional_top_vote": [],
        "minimax": [],
    }
    for point in range(point_count):
        values = [_check_values(rows[point]) for rows in candidate_predictions]
        actions = list(values[0])
        if any(set(row) != set(actions) for row in values):
            raise ValueError("candidate action sets must align at every point")
        result["map"].append(dict(values[map_index]))
        result["uniform_bma"].append(
            {
                action: sum(row[action] for row in values) / len(values)
                for action in actions
            }
        )
        result["score_softmax_bma"].append(
            {
                action: sum(
                    weight * row[action] for weight, row in zip(weights, values)
                )
                for action in actions
            }
        )
        votes = {action: 0.0 for action in actions}
        for row in values:
            best = optimal_action_set(row, tolerance=tolerance)
            for action in best:
                votes[action] += 1.0 / len(best)
        result["fractional_top_vote"].append(votes)
        result["minimax"].append(
            {action: min(row[action] for row in values) for action in actions}
        )
    return result


def summarize_committee_disagreement(
    candidate_predictions: Sequence[Sequence[Mapping[Any, float]]],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    if len(candidate_predictions) < 2:
        return {
            "candidate_count": len(candidate_predictions),
            "decision_points": 0,
            "top_set_disagreement_rate": None,
            "ranking_disagreement_rate": None,
        }
    point_count = len(candidate_predictions[0])
    top_disagreement = 0
    ranking_disagreement = 0
    for point in range(point_count):
        rows = [_check_values(candidate[point]) for candidate in candidate_predictions]
        actions = list(rows[0])
        top_signatures = {
            tuple(optimal_action_set(row, tolerance=tolerance)) for row in rows
        }
        ranking_signatures = set()
        for row in rows:
            signature = []
            for first, second in itertools.combinations(actions, 2):
                delta = row[first] - row[second]
                signature.append(0 if abs(delta) <= tolerance else (1 if delta > 0 else -1))
            ranking_signatures.add(tuple(signature))
        top_disagreement += len(top_signatures) > 1
        ranking_disagreement += len(ranking_signatures) > 1
    return {
        "candidate_count": len(candidate_predictions),
        "decision_points": point_count,
        "top_set_disagreement_rate": top_disagreement / point_count,
        "ranking_disagreement_rate": ranking_disagreement / point_count,
    }
