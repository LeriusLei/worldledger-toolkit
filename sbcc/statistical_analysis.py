"""Episode-level inference utilities for the SBCC preregistration.

Decision points inside one trajectory are dependent.  Every function here
therefore accepts one value per episode (or paired values per episode) and
refuses mismatched inputs.  SciPy is used for exact binomial calculations;
the research environment already depends on SciPy through its simulator stack.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Any, Sequence


def _require_scipy() -> tuple[Any, Any]:
    try:
        from scipy.stats import beta, binom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("exact statistical audit requires scipy") from exc
    return beta, binom


def clopper_pearson_interval(
    events: int,
    episodes: int,
    *,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Exact binomial interval for an episode-level event probability."""

    if episodes <= 0 or events < 0 or events > episodes:
        raise ValueError("require 0 <= events <= episodes and episodes > 0")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly in (0,1)")
    if alternative not in ("two-sided", "upper"):
        raise ValueError("alternative must be 'two-sided' or 'upper'")
    beta, _binom = _require_scipy()
    alpha = 1.0 - confidence_level
    if alternative == "upper":
        upper = (
            1.0
            if events == episodes
            else float(beta.ppf(confidence_level, events + 1, episodes - events))
        )
        return 0.0, upper
    lower = (
        0.0
        if events == 0
        else float(beta.ppf(alpha / 2.0, events, episodes - events + 1))
    )
    upper = (
        1.0
        if events == episodes
        else float(beta.ppf(1.0 - alpha / 2.0, events + 1, episodes - events))
    )
    return lower, upper


def exact_mcnemar_test(
    baseline_wrong: Sequence[bool],
    method_wrong: Sequence[bool],
    *,
    alternative: str = "method_better",
) -> dict[str, Any]:
    """Exact paired test over episode-level wrong-commit indicators."""

    if not baseline_wrong or len(baseline_wrong) != len(method_wrong):
        raise ValueError("paired episode indicators must align and be non-empty")
    if alternative not in ("method_better", "two-sided"):
        raise ValueError("unsupported alternative")
    _beta, binom = _require_scipy()
    improvements = sum(
        bool(baseline) and not bool(method)
        for baseline, method in zip(baseline_wrong, method_wrong)
    )
    harms = sum(
        not bool(baseline) and bool(method)
        for baseline, method in zip(baseline_wrong, method_wrong)
    )
    discordant = improvements + harms
    if discordant == 0:
        p_value = 1.0
    elif alternative == "method_better":
        p_value = float(binom.sf(improvements - 1, discordant, 0.5))
    else:
        smaller = min(improvements, harms)
        p_value = min(1.0, 2.0 * float(binom.cdf(smaller, discordant, 0.5)))
    baseline_rate = sum(bool(value) for value in baseline_wrong) / len(baseline_wrong)
    method_rate = sum(bool(value) for value in method_wrong) / len(method_wrong)
    return {
        "episode_count": len(baseline_wrong),
        "baseline_wrong_episode_rate": baseline_rate,
        "method_wrong_episode_rate": method_rate,
        "absolute_reduction": baseline_rate - method_rate,
        "relative_reduction": (
            (baseline_rate - method_rate) / baseline_rate
            if baseline_rate > 0.0
            else None
        ),
        "improvement_discordant_episodes": improvements,
        "harm_discordant_episodes": harms,
        "discordant_episode_count": discordant,
        "alternative": alternative,
        "exact_p_value": p_value,
    }


def exact_mcnemar_power(
    episodes: int,
    *,
    improvement_probability: float,
    harm_probability: float,
    alpha: float = 0.05,
) -> float:
    """Exact power for the one-sided McNemar test under paired probabilities."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0,1)")
    if (
        improvement_probability < 0.0
        or harm_probability < 0.0
        or improvement_probability + harm_probability > 1.0
    ):
        raise ValueError("discordant probabilities must be non-negative and sum <= 1")
    discordance_probability = improvement_probability + harm_probability
    if discordance_probability == 0.0:
        return 0.0
    _beta, binom = _require_scipy()
    conditional_improvement = improvement_probability / discordance_probability
    discordant_counts = list(range(episodes + 1))
    power = 0.0
    for discordant in discordant_counts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            weight = float(
                binom.pmf(discordant, episodes, discordance_probability)
            )
        if weight == 0.0 or discordant == 0:
            continue
        # Reject when P_null(X >= improvements) <= alpha.
        threshold = int(binom.isf(alpha, discordant, 0.5)) + 1
        conditional_rejection = (
            0.0
            if threshold > discordant
            else float(
                binom.sf(
                    threshold - 1,
                    discordant,
                    conditional_improvement,
                )
            )
        )
        power += weight * conditional_rejection
    return min(1.0, max(0.0, power))


def required_mcnemar_episodes(
    *,
    improvement_probability: float,
    harm_probability: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    maximum_episodes: int = 5_000,
) -> int | None:
    """Find the smallest episode count attaining the requested exact power."""

    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie strictly in (0,1)")
    if maximum_episodes <= 0:
        raise ValueError("maximum_episodes must be positive")
    previous = 0
    upper = 1
    while upper <= maximum_episodes:
        if exact_mcnemar_power(
            upper,
            improvement_probability=improvement_probability,
            harm_probability=harm_probability,
            alpha=alpha,
        ) >= target_power:
            break
        previous = upper
        upper *= 2
    if upper > maximum_episodes:
        upper = maximum_episodes
        if exact_mcnemar_power(
            upper,
            improvement_probability=improvement_probability,
            harm_probability=harm_probability,
            alpha=alpha,
        ) < target_power:
            return None
    low = max(1, previous + 1)
    high = upper
    while low < high:
        middle = (low + high) // 2
        power = exact_mcnemar_power(
            middle,
            improvement_probability=improvement_probability,
            harm_probability=harm_probability,
            alpha=alpha,
        )
        if power >= target_power:
            high = middle
        else:
            low = middle + 1
    # Exact tests are discrete; scan a small neighborhood to catch local dips.
    start = max(1, low - 20)
    for episodes in range(start, low + 1):
        if exact_mcnemar_power(
            episodes,
            improvement_probability=improvement_probability,
            harm_probability=harm_probability,
            alpha=alpha,
        ) >= target_power:
            return episodes
    return low


def paired_episode_bootstrap_difference(
    method_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_831,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile cluster bootstrap of mean(method - baseline) by episode."""

    if not method_values or len(method_values) != len(baseline_values):
        raise ValueError("paired episode values must align and be non-empty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly in (0,1)")
    differences = [
        float(method) - float(baseline)
        for method, baseline in zip(method_values, baseline_values)
    ]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("episode values must be finite")
    generator = random.Random(seed)
    count = len(differences)
    draws = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    )

    def quantile(probability: float) -> float:
        location = (len(draws) - 1) * probability
        lower = int(math.floor(location))
        upper = int(math.ceil(location))
        if lower == upper:
            return draws[lower]
        weight = location - lower
        return draws[lower] * (1.0 - weight) + draws[upper] * weight

    alpha = 1.0 - confidence_level
    observed = sum(differences) / count
    return {
        "method": "paired episode-cluster percentile bootstrap",
        "episode_count": count,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "mean_method_minus_baseline": observed,
        "confidence_level": confidence_level,
        "confidence_interval": [
            quantile(alpha / 2.0),
            quantile(1.0 - alpha / 2.0),
        ],
    }
