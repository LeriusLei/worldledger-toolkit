"""Exact inventory simulator with hidden suppliers and observable stock."""

from worldledger.benchmark.model_family import Instance, State, WorldModel


def observe(instance: Instance, model: WorldModel, state: State, action: str | None = None) -> tuple:
    if action == "probe_a":
        return ("stock", *state, "supplier_a", model.mechanism_a ^ instance.config["sensor_mask_a"])
    if action == "probe_b":
        return ("stock", *state, "supplier_b", model.mechanism_b ^ instance.config["sensor_mask_b"])
    return ("stock", *state)


def step(instance: Instance, model: WorldModel, state: State, action: str) -> tuple[State, int, tuple]:
    if action not in instance.allowed_actions:
        raise ValueError(f"unregistered action: {action}")
    cfg = instance.config
    stock, backlog = state
    if action in ("probe_a", "probe_b"):
        return state, cfg["probe_cost"], observe(instance, model, state, action)
    if action == "hold":
        next_state = (max(0, stock - 1), backlog)
        return next_state, cfg["hold_reward"], observe(instance, model, next_state)
    suffix = "a" if action == "order_a" else "b"
    available = bool(model.mechanism_a if suffix == "a" else model.mechanism_b)
    kind = cfg[f"supplier_{suffix}_kind"]
    shipment = cfg[f"shipment_{suffix}"]
    if kind == "shipment":
        next_state = ((min(cfg["capacity"], stock + shipment), backlog) if available
                      else (stock, backlog + cfg["failure_backlog_increment"]))
    elif kind == "backlog":
        next_state = ((stock, max(0, backlog - shipment)) if available
                      else (stock, backlog + cfg["failure_backlog_increment"]))
    else:
        next_state = (min(cfg["capacity"], stock + shipment), backlog)
    reward = cfg[f"reward_{suffix}"] if available else cfg["failed_order_reward"]
    return next_state, reward, observe(instance, model, next_state)
