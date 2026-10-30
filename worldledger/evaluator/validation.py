"""Executable positive/negative controls for Phase 1, independent of LLMs."""

from dataclasses import replace

from worldledger.benchmark.generators.grid import generate_family_instance
from worldledger.benchmark.history_generator import generate_histories
from worldledger.benchmark.model_family import WorldModel
from worldledger.evaluator.behavioral_quotient import quotient
from worldledger.evaluator.regret import evaluate_pool


def validate_controls(horizon: int) -> dict[str, bool]:
    instance = generate_family_instance(0)
    histories = generate_histories(instance, instance.models[0])
    h0, _, h2 = histories
    oracle = evaluate_pool(instance, h0, [m.model_id for m in instance.models], horizon)
    singleton = evaluate_pool(instance, h2, [instance.models[0].model_id], horizon)
    spurious = evaluate_pool(instance, h2, [m.model_id for m in instance.models], horizon)
    empty = evaluate_pool(instance, h0, [], horizon)
    failed = evaluate_pool(instance, h0, [None], horizon)

    # Separate source identity from executable behavior with a duplicate model.
    alias = WorldModel(instance.instance_id + ":alias", 0, 0)
    duplicate = replace(instance, models=instance.models + (alias,))
    labels = quotient(duplicate, horizon)
    false_code_diversity = (
        labels["behavior"][instance.models[0].model_id] == labels["behavior"][alias.model_id]
        and labels["decision"][instance.models[0].model_id] == labels["decision"][alias.model_id]
    )

    # Different transition and sensor outputs can still have identical one-step Q.
    neutral_payoff = dict(instance.config)
    neutral_payoff["reward_a"] = neutral_payoff["blocked_reward"]
    decision_irrelevant = replace(instance, config=neutral_payoff)
    irrelevant_labels = quotient(decision_irrelevant, horizon)
    a0, a1 = decision_irrelevant.models[0], decision_irrelevant.models[2]
    irrelevant_control = (
        irrelevant_labels["behavior"][a0.model_id] != irrelevant_labels["behavior"][a1.model_id]
        and irrelevant_labels["decision"][a0.model_id] == irrelevant_labels["decision"][a1.model_id]
    )
    return {
        "oracle_full_support": oracle["hcr"] == 1 and oracle["scr"] == 0 and oracle["delta_omit"] == 0,
        "k1_no_ambiguity": singleton["true_support_size"] == 1 and singleton["delta_omit"] == 0,
        "false_code_diversity_collapses": false_code_diversity,
        "decision_irrelevant_ambiguity_collapses": irrelevant_control,
        "spurious_history_inconsistent_rejected": spurious["history_inconsistent"] == 3 and spurious["scr"] > 0,
        "empty_pool_accounted": empty["delta_omit"] is None and empty["deployment_regret"] is None,
        "generation_failure_accounted": failed["generation_failures"] == 1 and failed["valid_candidates"] == 0,
    }
