"""Oracle-blind MAP comparator over unique executable candidate behavior classes."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from worldledger.sandbox.executor import run_candidate


@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    behavior_signature: tuple
    action_values: tuple[Fraction, ...]
    history_consistent: bool


@dataclass(frozen=True)
class MapSelection:
    selected_candidate_id: str
    action: str
    posterior_class_count: int


def evaluate_candidate(candidate_id: str, source: str, initial_state: tuple[int, int],
                       actions: tuple[str, ...], history_events: tuple[dict, ...],
                       probe_states: tuple[tuple[int, int], ...], horizon: int = 5) -> CandidateView:
    """Construct a view solely by running one candidate in the restricted interpreter."""
    if horizon < 1 or not actions or not probe_states:
        raise ValueError("positive horizon, actions and frozen probes required")
    state = initial_state
    consistent = True
    for event in history_events:
        next_state, reward, observation = run_candidate(source, state, event["action"])
        if reward != event["reward"] or list(observation) != event["observation"]:
            consistent = False
        state = next_state
    signature = tuple((probe, action, run_candidate(source, probe, action))
                      for probe in probe_states for action in actions)

    @lru_cache(maxsize=None)
    def value(current_state: tuple[int, int], remaining: int) -> Fraction:
        if remaining == 0:
            return Fraction()
        return max(Fraction(reward) + value(next_state, remaining - 1)
                   for action in actions
                   for next_state, reward, _ in (run_candidate(source, current_state, action),))

    q_values = tuple(Fraction(reward) + value(next_state, horizon - 1)
                     for action in actions
                     for next_state, reward, _ in (run_candidate(source, initial_state, action),))
    return CandidateView(candidate_id, signature, q_values, consistent)


def choose_map(candidates: tuple[CandidateView, ...], actions: tuple[str, ...]) -> MapSelection:
    """Class-uniform prior, 0/1 history likelihood, lexical class/candidate tie rule."""
    if not actions:
        raise ValueError("nonempty action set required")
    classes: dict[tuple, list[CandidateView]] = {}
    for candidate in candidates:
        if not candidate.history_consistent:
            continue
        if len(candidate.action_values) != len(actions):
            raise ValueError("candidate action vector does not match frozen actions")
        classes.setdefault(candidate.behavior_signature, []).append(candidate)
    if not classes:
        raise ValueError("empty admissible candidate pool; report as generation failure")
    for group in classes.values():
        if len({candidate.action_values for candidate in group}) != 1:
            raise ValueError("one behavior class has conflicting decision values")
    winning_class = min(classes, key=repr)
    selected = min(classes[winning_class], key=lambda candidate: candidate.candidate_id)
    action_index = min(range(len(actions)), key=lambda index: (-selected.action_values[index], index))
    return MapSelection(selected.candidate_id, actions[action_index], len(classes))
