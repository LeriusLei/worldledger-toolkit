"""Deterministic held-out queue/service transfer family, never used for tuning."""

import random

from worldledger.benchmark.model_family import Instance, four_models


TRANSFER_SEEDS = (100, 101, 102, 103)


def generate_family_instance(seed: int) -> Instance:
    if seed not in TRANSFER_SEEDS:
        raise ValueError("unregistered transfer seed")
    rng = random.Random(seed)
    capacity = rng.choice((5, 6, 7))
    start = (rng.choice((3, 4)), rng.choice((0, 1)))
    config = {
        "capacity": capacity,
        "arrival": rng.choice((1, 2)),
        "service_a_rate": rng.choice((2, 3)),
        "service_b_rate": rng.choice((2, 3)),
        "max_wear": 3,
        "job_reward": rng.choice((4, 5)),
        "failure_penalty": rng.choice((4, 5)),
        "maintenance_cost": 2,
        "hold_penalty": 1,
        "probe_cost": 1,
        "sensor_mask_a": rng.randrange(2),
        "sensor_mask_b": rng.randrange(2),
        "reward_function": "queue-service-v1",
        "observation_function": "queue-wear-active-sensor-v1",
    }
    instance_id = f"queue-{seed:08d}"
    return Instance(instance_id, seed, "queue", four_models(instance_id), ((start, 1),),
                    config, ("serve_a", "serve_b", "hold", "maintain", "probe_a", "probe_b"),
                    (start, (min(capacity, start[0] + 1), start[1]), (max(0, start[0] - 1), 0)))
