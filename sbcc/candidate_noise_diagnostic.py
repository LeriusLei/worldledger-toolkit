"""Separate candidate dispersion from particle-filter Monte Carlo variation.

The caller supplies an observation-history-only return estimator. The same
independent seed is sent to every candidate. No truth state or test label is
accepted by this API, and this diagnostic does not perform calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import random

import numpy as np


def evaluate_candidate_noise(
    candidate_ids: Sequence[str],
    estimate_returns: Callable[[str, int], Sequence[float]],
    *,
    pf_seeds: Sequence[int],
    noise_multiplier: float = 1.0,
) -> dict:
    """Return seed-paired variance components for identical action plans.

    ``estimate_returns(candidate_id, seed)`` must reset both Python and NumPy
    PF streams to ``seed`` before sampling and return one value per shared plan.
    A positive label requires between-candidate variance to exceed the
    within-candidate Monte Carlo variance, times ``noise_multiplier``.
    """
    ids = tuple(candidate_ids)
    seeds = tuple(int(x) for x in pf_seeds)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("at least two distinct candidates required")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two independent PF seeds required")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 1:
        raise ValueError("noise_multiplier must be finite and >= 1")
    rows = []
    plan_count = None
    for seed in seeds:
        values = []
        for candidate_id in ids:
            # Reset both streams identically for candidate-aligned sampling.
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                random.seed(seed)
                np.random.seed(seed)
                v = np.asarray(estimate_returns(candidate_id, seed), dtype=float)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
                raise ValueError("estimator must return finite 1D plan returns")
            if plan_count is None:
                plan_count = len(v)
            if len(v) != plan_count:
                raise ValueError("candidate plan counts differ")
            values.append(v)
        rows.append(np.stack(values))
    cube = np.stack(rows)  # seed x candidate x plan
    within = np.var(cube, axis=0, ddof=1).mean(axis=0)
    between = np.var(cube.mean(axis=0), axis=0, ddof=1)
    resolved = between > noise_multiplier * within
    return {
        "candidate_ids": list(ids),
        "pf_seeds": list(seeds),
        "common_random_numbers": True,
        "plan_count": int(plan_count),
        "within_candidate_return_variance_by_plan": within.tolist(),
        "between_candidate_return_variance_by_plan": between.tolist(),
        "disagreement_resolved_by_plan": resolved.tolist(),
        "disagreement_resolved": bool(np.any(resolved)),
        "noise_multiplier": float(noise_multiplier),
        "label": "candidate_disagreement" if np.any(resolved) else "unresolved_at_monte_carlo_noise_floor",
        "claim_boundary": "noise diagnostic only; not epistemic calibration or method effect",
    }
