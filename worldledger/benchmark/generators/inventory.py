"""Seeded non-grid inventory instances."""

import random

from worldledger.benchmark.model_family import Instance, four_models


def generate_family_instance(seed: int) -> Instance:
    rng = random.Random(seed)
    start = (rng.choice((1, 2, 3)), rng.choice((0, 1)))
    cfg = {
        "capacity": rng.choice((5, 6, 7)),
        "shipment_a": rng.choice((1, 2)), "shipment_b": rng.choice((1, 2, 3)),
        "supplier_a_kind": rng.choice(("shipment", "backlog", "quality")),
        "supplier_b_kind": rng.choice(("shipment", "backlog", "quality")),
        "reward_a": rng.randrange(8, 13), "reward_b": rng.randrange(8, 13),
        "failed_order_reward": -rng.randrange(2, 5), "hold_reward": -1,
        "failure_backlog_increment": rng.randrange(2),
        "probe_cost": -2,
        "sensor_mask_a": rng.randrange(2), "sensor_mask_b": rng.randrange(2),
        "reward_function": "procedural-supplier-v2", "observation_function": "stock-and-active-sensor-v1",
    }
    instance_id = f"inventory-{seed:08d}"
    return Instance(instance_id, seed, "inventory", four_models(instance_id), ((start, 1),), cfg,
                    ("order_a", "order_b", "hold", "probe_a", "probe_b"),
                    (start, (max(0, start[0] - 1), start[1]), (min(cfg["capacity"], start[0] + 1), start[1])))
