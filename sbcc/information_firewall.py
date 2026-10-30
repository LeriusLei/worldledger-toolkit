"""Evaluator-only shadow channel for counterfactual metrics without agent leakage."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ShadowRecord:
    index: int
    result: Any


class EvaluatorOnlyChannel:
    """Run a shadow evaluator while returning no value to the online loop.

    The payload is deep-copied and Python/NumPy RNG states are restored after
    evaluation. Results remain inaccessible until ``seal`` is called after the
    online episode. This is a narrow information-flow primitive, not a process
    sandbox: an evaluator closure must still not retain mutable environment or
    agent references.
    """

    def __init__(self, evaluator: Callable[[Any], Any], *, enabled: bool) -> None:
        self._evaluator = evaluator
        self._enabled = bool(enabled)
        self._sealed = False
        self._records = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def records(self) -> Tuple[ShadowRecord, ...]:
        if not self._sealed:
            raise RuntimeError("shadow results are inaccessible before episode seal")
        return tuple(self._records)

    def evaluate(self, payload: Any) -> None:
        if self._sealed:
            raise RuntimeError("cannot evaluate after episode seal")
        if not self._enabled:
            return None
        isolated_payload = copy.deepcopy(payload)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            result = self._evaluator(isolated_payload)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self._records.append(ShadowRecord(index=len(self._records), result=result))
        return None

    def seal(self) -> Tuple[ShadowRecord, ...]:
        self._sealed = True
        return tuple(self._records)


def paired_trace_firewall_check(
    run_episode: Callable[[EvaluatorOnlyChannel], Sequence[Any]],
    evaluator: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Run identical episodes with shadow evaluation off/on and compare traces."""

    disabled = EvaluatorOnlyChannel(evaluator, enabled=False)
    baseline = tuple(run_episode(disabled))
    disabled.seal()
    enabled = EvaluatorOnlyChannel(evaluator, enabled=True)
    shadow = tuple(run_episode(enabled))
    records = enabled.seal()
    first_mismatch: Optional[int] = None
    for index, pair in enumerate(zip(baseline, shadow)):
        if pair[0] != pair[1]:
            first_mismatch = index
            break
    if first_mismatch is None and len(baseline) != len(shadow):
        first_mismatch = min(len(baseline), len(shadow))
    return {
        "trace_identical": baseline == shadow,
        "baseline_steps": len(baseline),
        "shadow_steps": len(shadow),
        "shadow_evaluations": len(records),
        "first_mismatch_index": first_mismatch,
        "baseline_trace": baseline,
        "shadow_trace": shadow,
        "information_boundary": (
            "shadow evaluate returns None; results unlock only after seal; payload "
            "is deep-copied; Python/NumPy RNG restored"
        ),
    }

