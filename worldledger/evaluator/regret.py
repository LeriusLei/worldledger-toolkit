"""Coverage and worst-case risk with omission/deployment kept separate."""

from fractions import Fraction

from worldledger.benchmark.model_family import Instance, WorldModel
from worldledger.evaluator.behavioral_quotient import quotient
from worldledger.evaluator.consistency import consistent_models
from worldledger.evaluator.decision import action_values, minimax_action, model_regret
from worldledger.evaluator.horizon import choose_policy, policy_risk


def _risk(true_ids: set[str], action: str, q_by_model: dict[str, dict[str, Fraction]]) -> Fraction:
    return max(model_regret(q_by_model[model_id], action) for model_id in true_ids)


def evaluate_pool(instance: Instance, history: dict, candidates: list[str | None], horizon: int) -> dict:
    models = {model.model_id: model for model in instance.models}
    support = {model.model_id for model in consistent_models(instance, history)}
    if not support:
        raise ValueError("history has no admissible world")
    classes = quotient(instance, horizon)
    q_by_model = {model_id: action_values(instance, model, horizon) for model_id, model in models.items()}
    executable = [candidate for candidate in candidates if candidate in models]
    failures = len(candidates) - len(executable)
    inconsistent = sum(candidate not in support for candidate in executable)
    generated_ids = set(executable)
    retained_ids = generated_ids & support
    true_classes = {classes["behavior"][model_id] for model_id in support}
    generated_classes = {classes["behavior"][model_id] for model_id in generated_ids}
    retained_classes = {classes["behavior"][model_id] for model_id in retained_ids}
    true_decision_classes = {classes["decision"][model_id] for model_id in support}
    retained_decision_classes = {classes["decision"][model_id] for model_id in retained_ids}
    oracle_action = minimax_action(instance, support, q_by_model)
    oracle_risk = _risk(support, oracle_action, q_by_model)
    omission_action = minimax_action(instance, retained_ids, q_by_model) if retained_ids else None
    deployed_action = minimax_action(instance, generated_ids, q_by_model) if generated_ids else None
    omission_risk = _risk(support, omission_action, q_by_model) if omission_action else None
    deployment_risk = _risk(support, deployed_action, q_by_model) if deployed_action else None
    delta = omission_risk - oracle_risk if omission_risk is not None else None
    if delta is not None and delta < 0:
        raise AssertionError("subset support improved oracle minimax risk")
    return {
        "instance_id": instance.instance_id,
        "seed": instance.seed,
        "family": instance.family,
        "history_id": history["history_id"],
        "true_support_size": len(true_classes),
        "generated_candidates": len(candidates),
        "valid_candidates": len(executable),
        "behavior_classes": len(generated_classes),
        "decision_classes": len({classes["decision"][model_id] for model_id in generated_ids}),
        "generation_failures": failures,
        "history_inconsistent": inconsistent,
        "hcr": len(retained_classes) / len(true_classes),
        "decision_coverage": len(retained_decision_classes) / len(true_decision_classes),
        "scr": len(generated_classes - true_classes) / max(1, len(generated_classes)),
        "delta_omit": float(delta) if delta is not None else None,
        "deployment_regret": float(deployment_risk) if deployment_risk is not None else None,
        "oracle_risk": float(oracle_risk),
        "omission_risk": float(omission_risk) if omission_risk is not None else None,
        "oracle_action": oracle_action,
        "omission_action": omission_action,
        "deployment_action": deployed_action,
        "true_model_ids": sorted(support),
        "generated_model_ids": sorted(generated_ids),
        "retained_model_ids": sorted(retained_ids),
        "true_behavior_classes": sorted(true_classes),
        "generated_behavior_classes": sorted(generated_classes),
        "true_decision_classes": sorted(true_decision_classes),
        "retained_decision_classes": sorted(retained_decision_classes),
    }


def evaluate_pool_horizon(instance: Instance, history: dict, candidates: list[str | None], horizon: int) -> dict:
    """Shared-policy regret, with omission and raw-pool deployment separated.

    Candidate IDs here are finite-family controls. LLM source requires the
    separate sandbox and behavior-mapping bridge before this can score it.
    """
    models = {model.model_id: model for model in instance.models}
    support = {model.model_id for model in consistent_models(instance, history)}
    if not support:
        raise ValueError("history has no admissible world")
    generated = {candidate for candidate in candidates if candidate in models}
    retained = generated & support
    oracle = choose_policy(instance, support, horizon)
    oracle_risk = policy_risk(instance, support, oracle, horizon)
    omission = choose_policy(instance, retained, horizon) if retained else None
    deployment = choose_policy(instance, generated, horizon) if generated else None
    omission_risk = policy_risk(instance, support, omission, horizon) if omission else None
    deployment_risk = policy_risk(instance, support, deployment, horizon) if deployment else None
    delta = omission_risk - oracle_risk if omission_risk is not None else None
    if delta is not None and delta < 0:
        raise AssertionError("exact full-support policy was worse than subset policy")
    classes = quotient(instance, horizon, frozen_probe=True)
    true_classes = {classes["behavior"][x] for x in support}
    generated_classes = {classes["behavior"][x] for x in generated}
    retained_classes = {classes["behavior"][x] for x in retained}
    true_decision_classes = {classes["decision"][x] for x in support}
    retained_decision_classes = {classes["decision"][x] for x in retained}
    return {
        "instance_id": instance.instance_id, "history_id": history["history_id"], "horizon": horizon,
        "true_support_size": len(true_classes), "requested_candidates": len(candidates),
        "valid_candidates": len(candidates) - sum(candidate not in models for candidate in candidates),
        "generation_failures": sum(candidate not in models for candidate in candidates),
        "history_inconsistent": sum(candidate in models and candidate not in support for candidate in candidates),
        "hcr": len(retained_classes) / len(true_classes),
        "scr": len(generated_classes - true_classes) / max(1, len(generated_classes)),
        "decision_coverage": len(retained_decision_classes) / len(true_decision_classes),
        "oracle_action": oracle.action,
        "omission_action": omission.action if omission else None,
        "deployment_action": deployment.action if deployment else None,
        "oracle_risk": float(oracle_risk),
        "omission_risk": float(omission_risk) if omission_risk is not None else None,
        "delta_omit": float(delta) if delta is not None else None,
        "deployment_regret": float(deployment_risk) if deployment_risk is not None else None,
    }
