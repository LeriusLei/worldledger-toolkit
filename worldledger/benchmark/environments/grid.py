"""Small grid with two hidden, independently gated routes."""

from worldledger.benchmark.model_family import Instance, State, WorldModel


def observe(instance: Instance, model: WorldModel, state: State, action: str | None = None) -> tuple:
    if action == "probe_a":
        return ("position", *state, "sensor_a", model.mechanism_a ^ instance.config["sensor_mask_a"])
    if action == "probe_b":
        return ("position", *state, "sensor_b", model.mechanism_b ^ instance.config["sensor_mask_b"])
    return ("position", *state)


def step(instance: Instance, model: WorldModel, state: State, action: str) -> tuple[State, int, tuple]:
    if action not in instance.allowed_actions:
        raise ValueError(f"unregistered action: {action}")
    cfg = instance.config
    if action in ("probe_a", "probe_b"):
        return state, cfg["probe_cost"], observe(instance, model, state, action)
    if action == "wait":
        return state, cfg["wait_reward"], observe(instance, model, state)
    delta = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[action]
    target = state[0] + delta[0], state[1] + delta[1]
    gate_a = state == tuple(cfg["start"]) and action == cfg["gate_a_action"]
    gate_b = state == tuple(cfg["start"]) and action == cfg["gate_b_action"]
    blocked = (
        not 0 <= target[0] < cfg["width"]
        or not 0 <= target[1] < cfg["height"]
        or list(target) in cfg["walls"]
    )
    if blocked:
        return state, cfg["blocked_reward"], observe(instance, model, state)
    if gate_a or gate_b:
        suffix = "a" if gate_a else "b"
        mechanism = model.mechanism_a if gate_a else model.mechanism_b
        kind = cfg[f"gate_{suffix}_kind"]
        if kind == "door" and not mechanism:
            next_state = tuple(cfg["blocked_target"])
        elif kind == "portal" and mechanism:
            next_state = tuple(cfg["portal_target"])
        else:
            next_state = target
        reward = cfg[f"reward_{suffix}"] if mechanism else cfg["blocked_reward"]
        return next_state, reward, observe(instance, model, next_state)
    return target, cfg["move_reward"], observe(instance, model, target)
