"""Run the synthetic Sprinkler decision-ambiguity control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sbcc.agents import POLICIES, run_oracle  # noqa: E402
from sbcc.sprinkler import (  # noqa: E402
    Posterior,
    SprinklerWorld,
    TwinEvaluator,
    make_hypotheses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--nuisance-cardinality", type=int, default=16)
    parser.add_argument("--root-zone-accuracy", type=float, default=1.0)
    parser.add_argument("--phenotype-accuracy", type=float, default=1.0)
    parser.add_argument("--root-zone-cost", type=float, default=0.05)
    parser.add_argument("--phenotype-cost", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "sprinkler",
    )
    return parser.parse_args()


def factual_nll_range(prior: Posterior) -> tuple[float, float]:
    history = SprinklerWorld.factual_history()
    nlls = [-hypothesis.factual_log_likelihood(history) for hypothesis in prior.support]
    return min(nlls), max(nlls)


def run_trials(
    seeds: int,
    nuisance_cardinality: int,
    *,
    root_zone_accuracy: float = 1.0,
    phenotype_accuracy: float = 1.0,
    root_zone_cost: float = 0.05,
    phenotype_cost: float = 0.05,
) -> List[dict]:
    hypotheses = make_hypotheses(nuisance_cardinality)
    prior = Posterior.uniform(hypotheses)
    records: List[dict] = []

    for seed in range(seeds):
        # Balanced factorial assignment prevents prior/action tie-breaking from
        # receiving an accidental advantage from finite random sampling.
        true_response = seed % 2
        true_phenotype = (seed // 2) % nuisance_cardinality
        true_model = next(
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.response_type == true_response
            and hypothesis.phenotype == true_phenotype
        )

        for method, policy in POLICIES.items():
            world = SprinklerWorld(
                true_model,
                nuisance_cardinality=nuisance_cardinality,
                root_zone_accuracy=root_zone_accuracy,
                phenotype_accuracy=phenotype_accuracy,
                probe_costs={"root_zone": root_zone_cost, "phenotype": phenotype_cost},
                rng=random.Random(seed * 7919 + len(method)),
            )
            trace = policy(world, prior, random.Random(seed * 1009 + len(method)))
            evaluator = TwinEvaluator(true_model)
            twins = evaluator.action_rewards()
            optimal_action = evaluator.optimal_action()
            record = asdict(trace)
            record.update(
                {
                    "seed": seed,
                    "true_response": true_model.response_type,
                    "true_phenotype": true_model.phenotype,
                    "optimal_action": optimal_action,
                    "correct_action": float(trace.action == optimal_action),
                    "high_confidence_wrong": float(
                        trace.prediction_confidence >= 0.9 and trace.action != optimal_action
                    ),
                    "counterfactual_rank_correct": float(trace.action == optimal_action),
                    "twin_no_water": twins["no_water"],
                    "twin_water": twins["water"],
                    "twin_safe": twins["safe"],
                }
            )
            records.append(record)

        oracle_world = SprinklerWorld(
            true_model,
            nuisance_cardinality=nuisance_cardinality,
            root_zone_accuracy=root_zone_accuracy,
            phenotype_accuracy=phenotype_accuracy,
            probe_costs={"root_zone": root_zone_cost, "phenotype": phenotype_cost},
            rng=random.Random(seed * 7919 + 999),
        )
        oracle_trace = run_oracle(
            oracle_world, prior, random.Random(seed), true_model.response_type
        )
        evaluator = TwinEvaluator(true_model)
        record = asdict(oracle_trace)
        record.update(
            {
                "seed": seed,
                "true_response": true_model.response_type,
                "true_phenotype": true_model.phenotype,
                "optimal_action": evaluator.optimal_action(),
                "correct_action": 1.0,
                "high_confidence_wrong": 0.0,
                "counterfactual_rank_correct": 1.0,
                "twin_no_water": evaluator.action_rewards()["no_water"],
                "twin_water": evaluator.action_rewards()["water"],
                "twin_safe": evaluator.action_rewards()["safe"],
            }
        )
        records.append(record)

    return records


def mean_and_se(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def summarize(records: Iterable[Mapping[str, object]]) -> Dict[str, dict]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    summary: Dict[str, dict] = {}
    for method, rows in grouped.items():
        result: Dict[str, object] = {"n": len(rows)}
        for metric in (
            "total_utility",
            "correct_action",
            "high_confidence_wrong",
            "counterfactual_rank_correct",
            "entropy_after",
            "decision_classes_after",
        ):
            mean, se = mean_and_se(float(row[metric]) for row in rows)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_se"] = se
        probes: Dict[str, int] = defaultdict(int)
        for row in rows:
            probes[str(row["probe"] or "none")] += 1
        result["probe_counts"] = dict(probes)
        summary[method] = result
    return summary


def write_csv(path: Path, records: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_markdown(
    path: Path,
    summary: Mapping[str, Mapping[str, object]],
    nll_range: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    order = ["MAP", "BMA", "Robust", "GlobalIG", "RandomProbe", "FalsifyWM", "Oracle"]
    lines = [
        "# Synthetic Sprinkler control",
        "",
        f"- Seeds: {args.seeds}",
        f"- Nuisance cardinality: {args.nuisance_cardinality}",
        f"- Candidate models: {2 * args.nuisance_cardinality}",
        f"- Root-zone probe: accuracy={args.root_zone_accuracy:.3f}, cost={args.root_zone_cost:.3f}",
        f"- Phenotype probe: accuracy={args.phenotype_accuracy:.3f}, cost={args.phenotype_cost:.3f}",
        f"- Factual NLL range across candidates: [{nll_range[0]:.6f}, {nll_range[1]:.6f}]",
        "- Training feedback: realized branch only; twin branches are evaluator-only.",
        "",
        "| Method | Utility | Correct action | High-confidence wrong | Entropy after (bits) | Decision classes after | Probes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in order:
        row = summary[method]
        probes = ", ".join(f"{key}:{value}" for key, value in row["probe_counts"].items())
        lines.append(
            "| {method} | {utility:.3f} ± {utility_se:.3f} | {correct:.3f} | "
            "{wrong:.3f} | {entropy:.3f} | {classes:.3f} | {probes} |".format(
                method=method,
                utility=row["total_utility_mean"],
                utility_se=row["total_utility_se"],
                correct=row["correct_action_mean"],
                wrong=row["high_confidence_wrong_mean"],
                entropy=row["entropy_after_mean"],
                classes=row["decision_classes_after_mean"],
                probes=probes,
            )
        )

    falsify = summary["FalsifyWM"]
    global_ig = summary["GlobalIG"]
    map_row = summary["MAP"]
    h1 = nll_range[0] == nll_range[1] and map_row["correct_action_mean"] < 0.75
    h2 = (
        falsify["entropy_after_mean"] > global_ig["entropy_after_mean"]
        and falsify["decision_classes_after_mean"] < global_ig["decision_classes_after_mean"]
    )
    h3 = falsify["total_utility_mean"] > global_ig["total_utility_mean"] + 0.25
    lines.extend(
        [
            "",
            "## Toy mechanism checks",
            "",
            f"- H1 — identical factual fit can leave counterfactual action unresolved: **{'PASS' if h1 else 'FAIL'}**",
            f"- H2 — model entropy can remain high after decision ambiguity vanishes: **{'PASS' if h2 else 'FAIL'}**",
            f"- H3 — decision-directed probing beats global information gain: **{'PASS' if h3 else 'FAIL'}**",
            "",
            "These are existence results in a deliberately constructed finite world. They do not establish real-task generality or change historical research stop rules.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(output_dir: Path, summary: Mapping[str, Mapping[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = ["MAP", "BMA", "Robust", "GlobalIG", "RandomProbe", "FalsifyWM", "Oracle"]
    utilities = [summary[method]["total_utility_mean"] for method in methods]
    errors = [1.96 * summary[method]["total_utility_se"] for method in methods]
    colors = ["#777777", "#999999", "#999999", "#d97706", "#60a5fa", "#0f766e", "#111827"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(methods, utilities, yerr=errors, capsize=3, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Net utility (reward - probe cost)")
    ax.set_title("Single-branch decision quality")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_by_method.png", dpi=180)
    plt.close(fig)

    points = ["GlobalIG", "FalsifyWM"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for method, color in zip(points, ["#d97706", "#0f766e"]):
        ax.scatter(
            summary[method]["entropy_after_mean"],
            summary[method]["decision_classes_after_mean"],
            s=120,
            label=method,
            color=color,
        )
        ax.annotate(method, (
            summary[method]["entropy_after_mean"],
            summary[method]["decision_classes_after_mean"],
        ), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Remaining model entropy (bits)")
    ax.set_ylabel("Remaining decision classes")
    ax.set_title("Knowing the model is not the same as knowing what to do")
    ax.set_yticks([1, 2])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_vs_decision_ambiguity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior = Posterior.uniform(make_hypotheses(args.nuisance_cardinality))
    nll_range = factual_nll_range(prior)
    records = run_trials(
        args.seeds,
        args.nuisance_cardinality,
        root_zone_accuracy=args.root_zone_accuracy,
        phenotype_accuracy=args.phenotype_accuracy,
        root_zone_cost=args.root_zone_cost,
        phenotype_cost=args.phenotype_cost,
    )
    summary = summarize(records)

    write_csv(args.output_dir / "trials.csv", records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": vars(args) | {"output_dir": str(args.output_dir)},
                "factual_nll_range": nll_range,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "RESULTS.md", summary, nll_range, args)
    write_plots(args.output_dir, summary)
    print((args.output_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
