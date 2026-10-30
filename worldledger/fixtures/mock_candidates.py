"""Fixed, mixed-quality JSON responses for one frozen manifest cell.

No prompt is read or sent. The positive source is a hand-authored executable
control for Grid seed 4 / model m00, not a sampled LLM output or study result.
"""

import json


CELL_PREFIX = "grid-00000004:H0:"
OUTPUT_KEYS = frozenset((
    "candidate_id", "transition_model", "observation_model",
    "reward_model", "executable_artifact",
))

GRID4_M00 = """def transition(state, action):
    if action == 'east':
        return state if state == (1, 1) or state[0] + 1 >= 3 or (state[0] + 1 == 2 and state[1] == 0) else (state[0] + 1, state[1])
    else:
        if action == 'north':
            return state if state == (1, 1) or state[1] - 1 < 0 or (state[0] == 2 and state[1] - 1 == 0) else (state[0], state[1] - 1)
        else:
            if action == 'west':
                return state if state[0] - 1 < 0 else (state[0] - 1, state[1])
            else:
                if action == 'south':
                    return state if state[1] + 1 >= 4 else (state[0], state[1] + 1)
                else:
                    return state
def observation(state, action, next_state):
    if action == 'probe_a':
        return ('position', next_state[0], next_state[1], 'sensor_a', 1)
    else:
        if action == 'probe_b':
            return ('position', next_state[0], next_state[1], 'sensor_b', 0)
        else:
            return ('position', next_state[0], next_state[1])
def reward(state, action, next_state):
    if action == 'probe_a' or action == 'probe_b':
        return -2
    else:
        if action == 'wait':
            return -1
        else:
            return -4 if next_state == state else -1
"""

OFF_FAMILY = GRID4_M00.replace("return -4 if next_state == state else -1",
                                "return -4 if next_state == state else -2")
RUNTIME_FAILURE = GRID4_M00.replace("state[0] + 1, state[1]",
                                    "state[99] + 1, state[1]")
HISTORY_INCONSISTENT = GRID4_M00.replace("if action == 'wait':\n            return -1",
                                         "if action == 'wait':\n            return 0")
SYNTAX_FAILURE = "def transition(:\n    pass"

# Keys are frozen generation-arm IDs and slot indices. A None response models
# a transport timeout: it is not a malformed extra-field candidate JSON.
FIXTURES = {
    ("single_generation", 0): SYNTAX_FAILURE,
    ("independent_sampling", 0): GRID4_M00,
    ("independent_sampling", 1): GRID4_M00,
    ("independent_sampling", 2): OFF_FAMILY,
    ("temperature_diversity", 0): RUNTIME_FAILURE,
    ("temperature_diversity", 1): HISTORY_INCONSISTENT,
    ("temperature_diversity", 2): None,
    ("hypospace_inspired_budgeted", 0): GRID4_M00,
    ("hypospace_inspired_budgeted", 1): OFF_FAMILY,
    ("hypospace_inspired_budgeted", 2): GRID4_M00,
}


def mock_response(slot_id: str) -> str | None:
    """Return only the frozen five-field JSON candidate contract, or no reply."""
    if not slot_id.startswith(CELL_PREFIX):
        raise ValueError("Pilot-0 fixtures are restricted to one frozen manifest cell")
    rest = slot_id[len(CELL_PREFIX):]
    arm, index_text = rest.rsplit(":", 1)
    source = FIXTURES[(arm, int(index_text))]
    if source is None:
        return None
    candidate = {
        "candidate_id": slot_id,
        "transition_model": "Fixed dry-run transition fixture",
        "observation_model": "Fixed dry-run observation fixture",
        "reward_model": "Fixed dry-run reward fixture",
        "executable_artifact": source,
    }
    assert frozenset(candidate) == OUTPUT_KEYS
    return json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
