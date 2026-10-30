"""Evaluator-only finite-horizon action forks with RNG isolation.

The agent must never receive values produced here.  This module contains no
Pinductor dependency and works with pure transition/reward callbacks, making
the training-information boundary explicit and unit-testable.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, Mapping, Tuple


Transition = Callable[[Any, Any], Any]
Reward = Callable[[Any, Any, Any], Tuple[float, bool]]
StateKey = Callable[[Any], Hashable]


@contextmanager
def preserved_global_rng() -> Iterator[None]:
    """Restore Python/NumPy RNG state after an evaluator-only computation."""

    python_state = random.getstate()
    numpy_module = None
    numpy_state = None
    try:
        import numpy as np

        numpy_module = np
        numpy_state = np.random.get_state()
    except ImportError:
        pass
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_module is not None and numpy_state is not None:
            numpy_module.random.set_state(numpy_state)


@dataclass(frozen=True)
class ForkedActionValue:
    action: Any
    value: float
    immediate_reward: float
    immediately_terminated: bool


@dataclass(frozen=True)
class ForkedDecision:
    horizon: int
    gamma: float
    action_values: Tuple[ForkedActionValue, ...]
    best_actions: Tuple[Any, ...]
    value_margin: float

    def values(self) -> Mapping[Any, float]:
        return {entry.action: entry.value for entry in self.action_values}


class FiniteHorizonForkOracle:
    """Compute truth-model Q values without mutating state or global RNG."""

    def __init__(
        self,
        *,
        actions: Iterable[Any],
        transition: Transition,
        reward: Reward,
        gamma: float = 0.98,
        state_key: StateKey = lambda state: state,
        tolerance: float = 1e-12,
    ) -> None:
        self.actions = tuple(dict.fromkeys(actions))
        if not self.actions:
            raise ValueError("at least one action is required")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.transition = transition
        self.reward = reward
        self.gamma = float(gamma)
        self.state_key = state_key
        self.tolerance = tolerance
        self._value_cache: Dict[Tuple[Hashable, int], float] = {}

    def _step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        previous = copy.deepcopy(state)
        next_state = self.transition(previous, action)
        reward, terminated = self.reward(previous, action, next_state)
        return next_state, float(reward), bool(terminated)

    def _optimal_value(self, state: Any, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        cache_key = (self.state_key(state), horizon)
        if cache_key in self._value_cache:
            return self._value_cache[cache_key]
        values = []
        for action in self.actions:
            next_state, immediate, terminated = self._step(state, action)
            tail = 0.0 if terminated else self._optimal_value(next_state, horizon - 1)
            values.append(immediate + self.gamma * tail)
        result = max(values)
        self._value_cache[cache_key] = result
        return result

    def evaluate(self, state: Any, *, horizon: int) -> ForkedDecision:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        with preserved_global_rng():
            entries = []
            for action in self.actions:
                next_state, immediate, terminated = self._step(state, action)
                tail = (
                    0.0
                    if terminated or horizon == 1
                    else self._optimal_value(next_state, horizon - 1)
                )
                entries.append(
                    ForkedActionValue(
                        action=action,
                        value=immediate + self.gamma * tail,
                        immediate_reward=immediate,
                        immediately_terminated=terminated,
                    )
                )
        best = max(entry.value for entry in entries)
        best_actions = tuple(
            entry.action
            for entry in entries
            if best - entry.value <= self.tolerance
        )
        distinct = sorted({entry.value for entry in entries}, reverse=True)
        margin = distinct[0] - distinct[1] if len(distinct) > 1 else 0.0
        return ForkedDecision(
            horizon=horizon,
            gamma=self.gamma,
            action_values=tuple(entries),
            best_actions=best_actions,
            value_margin=margin,
        )


class StochasticFiniteHorizonForkOracle:
    """Sampled Bellman oracle for stochastic component-model wrappers.

    This is candidate-side planning support for Pinductor's component-wise
    Thompson baseline. It is never used as the ground-truth evaluator.
    """

    def __init__(
        self,
        *,
        actions: Iterable[Any],
        transition: Transition,
        reward: Reward,
        num_rollouts: int = 10,
        gamma: float = 0.98,
        state_key: StateKey = lambda state: state,
        tolerance: float = 1e-12,
    ) -> None:
        self.actions = tuple(dict.fromkeys(actions))
        if not self.actions:
            raise ValueError("at least one action is required")
        if (
            not isinstance(num_rollouts, int)
            or isinstance(num_rollouts, bool)
            or num_rollouts <= 0
        ):
            raise ValueError("num_rollouts must be a positive integer")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.transition = transition
        self.reward = reward
        self.num_rollouts = num_rollouts
        self.gamma = float(gamma)
        self.state_key = state_key
        self.tolerance = float(tolerance)
        self._value_cache: Dict[Tuple[Hashable, int], float] = {}

    def _action_value(
        self, state: Any, action: Any, horizon: int
    ) -> Tuple[float, float, bool]:
        values = []
        immediate_rewards = []
        terminations = []
        for _ in range(self.num_rollouts):
            previous = copy.deepcopy(state)
            next_state = self.transition(previous, action)
            immediate, terminated = self.reward(previous, action, next_state)
            immediate = float(immediate)
            terminated = bool(terminated)
            tail = (
                0.0
                if terminated or horizon <= 1
                else self._optimal_value(next_state, horizon - 1)
            )
            values.append(immediate + self.gamma * tail)
            immediate_rewards.append(immediate)
            terminations.append(terminated)
        return (
            sum(values) / len(values),
            sum(immediate_rewards) / len(immediate_rewards),
            all(terminations),
        )

    def _optimal_value(self, state: Any, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        key = (self.state_key(state), horizon)
        if key not in self._value_cache:
            self._value_cache[key] = max(
                self._action_value(state, action, horizon)[0]
                for action in self.actions
            )
        return self._value_cache[key]

    def evaluate(self, state: Any, *, horizon: int) -> ForkedDecision:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        with preserved_global_rng():
            entries = []
            for action in self.actions:
                value, immediate, terminated = self._action_value(
                    state, action, horizon
                )
                entries.append(
                    ForkedActionValue(
                        action=action,
                        value=value,
                        immediate_reward=immediate,
                        immediately_terminated=terminated,
                    )
                )
        best = max(entry.value for entry in entries)
        best_actions = tuple(
            entry.action for entry in entries if best - entry.value <= self.tolerance
        )
        distinct = sorted({entry.value for entry in entries}, reverse=True)
        margin = distinct[0] - distinct[1] if len(distinct) > 1 else 0.0
        return ForkedDecision(
            horizon=horizon,
            gamma=self.gamma,
            action_values=tuple(entries),
            best_actions=best_actions,
            value_margin=margin,
        )


def factual_transition_check(
    *,
    previous_state: Any,
    action: Any,
    recorded_next_state: Any,
    recorded_reward: float,
    recorded_terminated: bool,
    transition: Transition,
    reward: Reward,
) -> Dict[str, Any]:
    """Verify that evaluator truth reproduces one recorded factual step."""

    with preserved_global_rng():
        previous = copy.deepcopy(previous_state)
        predicted_next = transition(previous, action)
        predicted_reward, predicted_terminated = reward(
            previous, action, predicted_next
        )
    return {
        "next_state_match": predicted_next == recorded_next_state,
        "reward_match": abs(float(predicted_reward) - float(recorded_reward))
        <= 1e-12,
        "terminated_match": bool(predicted_terminated) == bool(recorded_terminated),
        "predicted_reward": float(predicted_reward),
        "predicted_terminated": bool(predicted_terminated),
    }
