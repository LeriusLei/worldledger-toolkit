"""Exact finite queue/service dynamics for the independent synthetic transfer split."""

from worldledger.benchmark.model_family import Instance, State, WorldModel


def observe(instance: Instance, model: WorldModel, state: State, action: str | None = None) -> tuple:
    if action == "probe_a":
        return ("queue", *state, "service_a", model.mechanism_a ^ instance.config["sensor_mask_a"])
    if action == "probe_b":
        return ("queue", *state, "service_b", model.mechanism_b ^ instance.config["sensor_mask_b"])
    return ("queue", *state)


def step(instance: Instance, model: WorldModel, state: State, action: str) -> tuple[State, int, tuple]:
    if action not in instance.allowed_actions:
        raise ValueError(f"unregistered action: {action}")
    cfg = instance.config
    jobs, wear = state
    if action in ("probe_a", "probe_b"):
        return state, -cfg["probe_cost"], observe(instance, model, state, action)
    if action == "hold":
        next_state = (min(cfg["capacity"], jobs + cfg["arrival"]), wear)
        return next_state, -cfg["hold_penalty"] * next_state[0], observe(instance, model, next_state)
    if action == "maintain":
        next_state = (min(cfg["capacity"], jobs + cfg["arrival"]), 0)
        return next_state, -cfg["maintenance_cost"], observe(instance, model, next_state)
    suffix = "a" if action == "serve_a" else "b"
    active = bool(model.mechanism_a if suffix == "a" else model.mechanism_b)
    if active and wear < cfg["max_wear"]:
        done = min(jobs, cfg[f"service_{suffix}_rate"])
        next_state = (min(cfg["capacity"], jobs - done + cfg["arrival"]), wear + 1)
        reward = done * cfg["job_reward"] - next_state[0]
    else:
        next_state = (min(cfg["capacity"], jobs + cfg["arrival"]), min(cfg["max_wear"], wear + 1))
        reward = -cfg["failure_penalty"] - next_state[0]
    return next_state, reward, observe(instance, model, next_state)
