"""Pre-registered action sequence: neutral, probe A, probe B."""

from worldledger.benchmark.model_family import Instance, WorldModel
from worldledger.evaluator.simulation import observe, step


def generate_histories(instance: Instance, true_model: WorldModel) -> tuple[dict, dict, dict]:
    if len(instance.initial_state_distribution) != 1:
        raise ValueError("history generator requires the frozen point-mass initial belief")
    state = instance.initial_state_distribution[0][0]
    neutral = "wait" if instance.family == "grid" else "hold"
    actions = (neutral, "probe_a", "probe_b")
    initial_observation = observe(instance, true_model, state)
    events: list[dict] = []
    histories: list[dict] = []
    for index, action in enumerate(actions):
        state, reward, observation = step(instance, true_model, state, action)
        events.append({"action": action, "reward": reward, "observation": list(observation)})
        histories.append({"history_id": f"H{index}", "initial_observation": list(initial_observation),
                          "events": [dict(event) for event in events]})
    return tuple(histories)  # type: ignore[return-value]
