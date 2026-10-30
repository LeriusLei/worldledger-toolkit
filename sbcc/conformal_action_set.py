"""Episode-level split-conformal action sets for a strong simple baseline.

The calibrator treats a whole episode as one exchangeable calibration unit.
Its nonconformity score is the worst action-ranking shortfall within that
episode.  This avoids pretending that many correlated decision points from
the same episode are independent observations.

The resulting guarantee is deliberately modest: under exchangeable episodes,
and when the score-producing model was fixed independently of calibration,
the returned sets cover at least one truth-optimal action simultaneously over
all decision points in a fresh episode with marginal probability at least
``1 - alpha``.  It is not an action-conditional or open-world guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


def _checked_scores(scores: Mapping[Any, float]) -> dict[Any, float]:
    if not scores:
        raise ValueError("action scores cannot be empty")
    checked = {action: float(value) for action, value in scores.items()}
    if not all(math.isfinite(value) for value in checked.values()):
        raise ValueError("action scores must be finite")
    return checked


def action_nonconformity(
    scores: Mapping[Any, float], truth_optimal_actions: Sequence[Any]
) -> float:
    """Return how far the best truth-optimal action trails the predicted best."""

    checked = _checked_scores(scores)
    truth = tuple(dict.fromkeys(truth_optimal_actions))
    if not truth:
        raise ValueError("truth optimal-action set cannot be empty")
    missing = [action for action in truth if action not in checked]
    if missing:
        raise ValueError(f"truth actions missing from score row: {missing!r}")
    return max(checked.values()) - max(checked[action] for action in truth)


def episode_nonconformity(
    score_rows: Sequence[Mapping[Any, float]],
    truth_optimal_sets: Sequence[Sequence[Any]],
) -> float:
    """Use the worst decision point so calibration respects episode clusters."""

    if not score_rows or len(score_rows) != len(truth_optimal_sets):
        raise ValueError("episode scores and truth sets must align and be non-empty")
    return max(
        action_nonconformity(scores, truth)
        for scores, truth in zip(score_rows, truth_optimal_sets)
    )


def split_conformal_quantile(
    calibration_nonconformities: Sequence[float], *, alpha: float
) -> float:
    """Finite-sample split-conformal quantile with a conservative infinity case."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not calibration_nonconformities:
        raise ValueError("at least one calibration episode is required")
    values = sorted(float(value) for value in calibration_nonconformities)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("nonconformities must be finite and non-negative")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    return values[rank - 1] if rank <= len(values) else math.inf


@dataclass(frozen=True)
class ConformalActionPrediction:
    actions: Tuple[Any, ...]
    threshold: float

    @property
    def singleton_action(self) -> Optional[Any]:
        return self.actions[0] if len(self.actions) == 1 else None


@dataclass(frozen=True)
class EpisodeConformalActionCalibrator:
    """A fitted episode-clustered split-conformal action-set predictor."""

    alpha: float
    threshold: float
    calibration_episode_count: int

    @classmethod
    def fit(
        cls,
        calibration_score_episodes: Sequence[Sequence[Mapping[Any, float]]],
        calibration_truth_episodes: Sequence[Sequence[Sequence[Any]]],
        *,
        alpha: float = 0.1,
    ) -> "EpisodeConformalActionCalibrator":
        if not calibration_score_episodes or len(calibration_score_episodes) != len(
            calibration_truth_episodes
        ):
            raise ValueError(
                "calibration score and truth episodes must align and be non-empty"
            )
        nonconformities = [
            episode_nonconformity(score_episode, truth_episode)
            for score_episode, truth_episode in zip(
                calibration_score_episodes, calibration_truth_episodes
            )
        ]
        return cls(
            alpha=alpha,
            threshold=split_conformal_quantile(nonconformities, alpha=alpha),
            calibration_episode_count=len(nonconformities),
        )

    def predict(self, scores: Mapping[Any, float]) -> ConformalActionPrediction:
        checked = _checked_scores(scores)
        best = max(checked.values())
        actions = tuple(
            action
            for action, value in checked.items()
            if best - value <= self.threshold
        )
        return ConformalActionPrediction(actions=actions, threshold=self.threshold)

    def evaluate(
        self,
        score_episodes: Sequence[Sequence[Mapping[Any, float]]],
        truth_episodes: Sequence[Sequence[Sequence[Any]]],
    ) -> dict[str, Any]:
        if not score_episodes or len(score_episodes) != len(truth_episodes):
            raise ValueError("evaluation episodes must align and be non-empty")
        point_count = 0
        covered_points = 0
        singleton_points = 0
        correct_singletons = 0
        simultaneously_covered_episodes = 0
        for score_episode, truth_episode in zip(score_episodes, truth_episodes):
            if not score_episode or len(score_episode) != len(truth_episode):
                raise ValueError(
                    "each evaluation episode must align and be non-empty"
                )
            episode_covered = True
            for scores, truth_actions in zip(score_episode, truth_episode):
                prediction = self.predict(scores)
                truth = set(truth_actions)
                if not truth:
                    raise ValueError("truth optimal-action set cannot be empty")
                if not truth.issubset(scores):
                    raise ValueError("truth action missing from score row")
                covered = bool(set(prediction.actions) & truth)
                point_count += 1
                covered_points += covered
                episode_covered = episode_covered and covered
                if prediction.singleton_action is not None:
                    singleton_points += 1
                    correct_singletons += prediction.singleton_action in truth
            simultaneously_covered_episodes += episode_covered
        episode_count = len(score_episodes)
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "calibration_episode_count": self.calibration_episode_count,
            "evaluation_episode_count": episode_count,
            "decision_point_count": point_count,
            "pointwise_optimal_action_coverage": covered_points / point_count,
            "episode_simultaneous_coverage": (
                simultaneously_covered_episodes / episode_count
            ),
            "singleton_rate": singleton_points / point_count,
            "singleton_accuracy": (
                correct_singletons / singleton_points if singleton_points else None
            ),
            "wrong_singleton_rate": (
                (singleton_points - correct_singletons) / singleton_points
                if singleton_points
                else None
            ),
            "unconditional_wrong_singleton_rate": (
                (singleton_points - correct_singletons) / point_count
            ),
            "guarantee_scope": (
                "marginal simultaneous optimal-action coverage over exchangeable "
                "episodes; not action-conditional and not valid under model/data reuse"
            ),
        }
