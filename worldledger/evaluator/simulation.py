"""Dispatch without placing simulator semantics in the evaluator."""

from worldledger.benchmark.environments import grid, inventory, queue_transfer
from worldledger.benchmark.model_family import Instance, State, WorldModel


def environment(instance: Instance):
    return {"grid": grid, "inventory": inventory, "queue": queue_transfer}[instance.family]


def observe(instance: Instance, model: WorldModel, state: State, action: str | None = None) -> tuple:
    return environment(instance).observe(instance, model, state, action)


def step(instance: Instance, model: WorldModel, state: State, action: str) -> tuple[State, int, tuple]:
    return environment(instance).step(instance, model, state, action)
