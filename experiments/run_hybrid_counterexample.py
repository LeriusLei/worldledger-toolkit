"""Generate the exact component-mixing counterexample audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sbcc.hybrid_ensemble import (  # noqa: E402
    audit_componentwise_mixing,
    make_crossed_reward_counterexample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "hybrid_counterexample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_componentwise_mixing(make_crossed_reward_counterexample())

    rows = []
    for combination in audit.combinations:
        rows.append(
            {
                "transition_candidate": combination.transition_candidate_id,
                "reward_candidate": combination.reward_candidate_id,
                "is_complete_candidate": combination.is_complete_candidate,
                "safe_value": combination.action_values["safe"],
                "risky_value": combination.action_values["risky"],
                "top_action": combination.ranking[0],
                "ranking": ">".join(combination.ranking),
                "ranking_supported": (
                    combination.ranking_supported_by_complete_candidate
                ),
                "top_action_supported": (
                    combination.top_action_supported_by_complete_candidate
                ),
            }
        )

    with (args.output_dir / "combinations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "candidate_count": audit.candidate_count,
        "component_combinations": audit.combination_count,
        "complete_candidates": audit.complete_candidate_count,
        "hybrid_fraction": audit.hybrid_fraction,
        "unsupported_ranking_count": audit.unsupported_ranking_count,
        "ranking_support_violation_rate": audit.ranking_support_violation_rate,
        "unsupported_top_action_count": audit.unsupported_top_action_count,
        "unsupported_top_action_rate": audit.unsupported_top_action_rate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    markdown = f"""# Exact hybrid-world counterexample

Two complete candidate worlds both value `safe=1` and `risky=0`. Independent
transition/reward selection creates {audit.combination_count} component pairs,
of which {audit.complete_candidate_count} are complete candidates. Both crossed
pairs value `risky=2`, producing an action that no complete candidate supports.

| Metric | Exact value |
|---|---:|
| Candidate worlds | {audit.candidate_count} |
| Component combinations | {audit.combination_count} |
| Hybrid fraction | {audit.hybrid_fraction:.3f} |
| Unsupported-ranking rate | {audit.ranking_support_violation_rate:.3f} |
| Unsupported-top-action rate | {audit.unsupported_top_action_rate:.3f} |

This is an existence proof of non-closure, not empirical evidence that the
Pinductor ensemble fails at this rate. Go/No-Go 2 must measure the same metric
on real `RexNode` candidates.
"""
    (args.output_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
