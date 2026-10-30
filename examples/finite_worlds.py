"""Small exact controls; no model/provider calls."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldledger.benchmark.generators import grid, inventory, queue_transfer
from worldledger.benchmark.history_generator import generate_histories
from worldledger.evaluator.consistency import consistent_models
from worldledger.evaluator.regret import evaluate_pool
from worldledger.world_model.machine import load_machine_contract, enumerate_machine
from worldledger.world_model.multistep import machine_family, choose_policy


def main():
    result = {"synthetic_control": True, "families": {}}
    for name, generator in (("grid", grid), ("inventory", inventory), ("queue", queue_transfer)):
        seed = queue_transfer.TRANSFER_SEEDS[0] if name == "queue" else 4
        instance = generator.generate_family_instance(seed)
        histories = generate_histories(instance, instance.models[0])
        pool = [model.model_id for model in instance.models]
        full = evaluate_pool(instance, histories[0], pool, horizon=1)
        empty = evaluate_pool(instance, histories[0], [], horizon=1)
        result["families"][name] = {
            "support_by_history": [len(consistent_models(instance, h)) for h in histories],
            "full_pool_coverage": full["hcr"], "full_pool_delta_omit": full["delta_omit"],
            "empty_pool_delta_omit": empty["delta_omit"],
        }
    contract = load_machine_contract()
    family = machine_family(contract)
    policy, _ = choose_policy(family, set(dict(family.models)), 5)
    result["machine"] = {"assignment_count": enumerate_machine(contract)["assignment_count"],
                         "horizon": 5, "shared_policy_first_action": policy.action}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
