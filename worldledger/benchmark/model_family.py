"""Finite hidden-mechanism family shared by both benchmark simulators."""

from dataclasses import dataclass
from typing import Any


State = tuple[int, int]


@dataclass(frozen=True)
class WorldModel:
    model_id: str
    mechanism_a: int
    mechanism_b: int


@dataclass(frozen=True)
class Instance:
    instance_id: str
    seed: int
    family: str
    models: tuple[WorldModel, ...]
    initial_state_distribution: tuple[tuple[State, int], ...]
    config: dict[str, Any]
    allowed_actions: tuple[str, ...]
    probe_states: tuple[State, ...]

    def model(self, model_id: str) -> WorldModel:
        return next(model for model in self.models if model.model_id == model_id)

    def as_record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "family": self.family,
            "hidden_models": [vars(model) for model in self.models],
            "initial_state_distribution": [
                {"state": list(state), "weight": weight}
                for state, weight in self.initial_state_distribution
            ],
            "reward_function": self.config["reward_function"],
            "observation_function": self.config["observation_function"],
            "allowed_actions": list(self.allowed_actions),
            "probe_states": [list(state) for state in self.probe_states],
            "environment_config": self.config,
        }


def four_models(instance_id: str) -> tuple[WorldModel, ...]:
    return tuple(
        WorldModel(f"{instance_id}:m{a}{b}", a, b)
        for a in (0, 1)
        for b in (0, 1)
    )
