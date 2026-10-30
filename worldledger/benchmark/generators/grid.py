"""Seeded grid instances; filtering never observes candidate-model output."""

import random

from worldledger.benchmark.model_family import Instance, four_models


def generate_family_instance(seed: int) -> Instance:
    rng = random.Random(seed)
    width = rng.choice((3, 4, 5))
    height = rng.choice((3, 4, 5))
    start = (1, 1)
    gate_a, gate_b = rng.sample(("east", "west", "north", "south"), 2)
    # All four neighboring cells are in-bounds. Layout and gate placement vary.
    possible_walls = [
        (x, y) for x in range(width) for y in range(height)
        if (x, y) not in (start, (2, 1), (1, 0), (0, 1), (1, 2), (0, 0), (width - 1, height - 1))
    ]
    walls = [list(cell) for cell in rng.sample(possible_walls, min(len(possible_walls), rng.randrange(0, 3)))]
    cfg = {
        "width": width, "height": height, "start": list(start), "walls": walls,
        "gate_a_action": gate_a, "gate_b_action": gate_b,
        "gate_a_kind": rng.choice(("door", "portal", "toll")),
        "gate_b_kind": rng.choice(("door", "portal", "toll")),
        "portal_target": [width - 1, height - 1],
        "blocked_target": [0, 0] if rng.randrange(2) else list(start),
        "reward_a": rng.randrange(8, 13), "reward_b": rng.randrange(8, 13),
        "blocked_reward": -rng.randrange(2, 5), "move_reward": -1,
        "wait_reward": -1, "probe_cost": -2,
        "sensor_mask_a": rng.randrange(2), "sensor_mask_b": rng.randrange(2),
        "reward_function": "procedural-gate-v2", "observation_function": "position-and-active-sensor-v1",
    }
    instance_id = f"grid-{seed:08d}"
    return Instance(instance_id, seed, "grid", four_models(instance_id), ((start, 1),), cfg,
                    ("east", "north", "west", "south", "wait", "probe_a", "probe_b"),
                    (start, (0, 1), (1, 2)))
