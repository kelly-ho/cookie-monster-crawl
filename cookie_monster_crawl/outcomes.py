"""
Outcome tracking for strategy proposals.

Records whether each strategy's proposals improved harvest efficiency,
so the Critic agent can reference historical results during debate.
"""

import json
import os
from datetime import datetime


OUTCOMES_FILE = "results/outcomes.jsonl"


def record_outcome(
    strategy_file: str,
    strategy: dict,
    harvest_before: dict,
    harvest_after: dict,
    output_dir: str = "results",
):
    """Append one outcome record to outcomes.jsonl."""
    filepath = os.path.join(output_dir, "outcomes.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    feature_names = [f.get("name", "?") for f in strategy.get("feature_proposals", [])]
    policy_summaries = [p[:80] for p in strategy.get("policy_proposals", [])]
    config_summaries = [f"{c.get('parameter', '?')}: {c.get('current_value', '?')} → {c.get('proposed_value', '?')}" for c in strategy.get("config_proposals", [])]

    record = {
        "timestamp": datetime.now().isoformat(),
        "strategy_file": strategy_file,
        "feature_proposals": feature_names,
        "policy_proposals": policy_summaries,
        "config_proposals": config_summaries,
        "harvest_before": harvest_before,
        "harvest_after": harvest_after,
        "delta": {
            "harvest_pct": round(harvest_after.get("harvest_pct", 0) - harvest_before.get("harvest_pct", 0), 2),
            "recipes": harvest_after.get("recipes", 0) - harvest_before.get("recipes", 0),
        },
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_outcomes(output_dir: str = "results") -> list[dict]:
    """Read all outcome records. Returns empty list if file doesn't exist."""
    filepath = os.path.join(output_dir, "outcomes.jsonl")
    if not os.path.exists(filepath):
        return []

    outcomes = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outcomes.append(json.loads(line))
    return outcomes


def format_outcomes_for_prompt(outcomes: list[dict], max_entries: int = 5) -> str:
    """Format recent outcomes as readable text for the Critic prompt."""
    if not outcomes:
        return "No outcome history available yet. Focus your critique on logical consistency and evidence from the current crawl data."

    recent = outcomes[-max_entries:]
    lines = [f"## Outcome History ({len(recent)} most recent runs)\n"]

    for i, o in enumerate(recent, 1):
        delta = o.get("delta", {})
        delta_pct = delta.get("harvest_pct", 0)
        direction = "improved" if delta_pct > 0 else "regressed" if delta_pct < 0 else "unchanged"

        before = o.get("harvest_before", {})
        after = o.get("harvest_after", {})

        lines.append(f"### Run {i}: {direction} ({delta_pct:+.1f}%)")
        lines.append(f"Harvest: {before.get('harvest_pct', '?')}% → {after.get('harvest_pct', '?')}%")
        lines.append(f"Features proposed: {', '.join(o.get('feature_proposals', [])) or '(none)'}")
        lines.append(f"Config proposed: {', '.join(o.get('config_proposals', [])) or '(none)'}")
        lines.append(f"Policies proposed: {', '.join(o.get('policy_proposals', [])) or '(none)'}")
        lines.append("")

    return "\n".join(lines)
