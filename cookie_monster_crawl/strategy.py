"""
Strategy generator for cookie-monster-crawl.

Sends a replay analysis to Claude Opus and produces a JSON strategy doc
for the next crawl run.

Usage:
    python -m cookie_monster_crawl.strategy <replay_json> [options]
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import subprocess

MAX_FILTER_SAMPLES = 20
DOMAIN_COLLAPSE_THRESHOLD = 30  # collapse zero-visit domains into a summary above this count

STRATEGY_SCHEMA = {
    "timestamp": "<ISO timestamp>",
    "based_on_log": "<logfile path>",
    "previous_strategy": "<strategy file path or null>",
    "feature_proposals": [
        {
            "name": "<snake_case feature name>",
            "description": "<what this feature captures and why it would help>",
            "computation": "<how to compute from URL structure, anchor text, or crawl state>",
        }
    ],
    "policy_proposals": [
        "<description of a new crawling strategy, architectural change, or approach to try>"
    ],
    "seeds": {
        "keep": ["<homepage URL>"],
        "remove": ["<homepage URL>"],
        "add": ["<homepage URL>"],
    },
    "segment_additions": {
        "infrastructure": ["<segment string>"],
        "navigational": ["<segment string>"],
        "recipe_related": ["<segment string>"],
    },
    "reasoning": "<plain text explanation tying analysis to proposals>",
    "summary": "<2-3 sentence human-readable summary>",
}

SYSTEM_PROMPT = """You are a crawl strategy advisor for a recipe web crawler called cookie-monster-crawl.

## Architecture

The crawler uses a learned scoring model to prioritize URLs:

1. **Feature extraction**: Each discovered URL is converted to raw features based on its structure. Current features:
   - domain_share: fraction of crawled pages from this domain
   - lsh_count: number of near-duplicate URLs already seen (MinHash LSH)
   - dead_branch: 1 if the URL's path root has <20% recipe rate, else 0
   - anchor_word_count: number of words in the link's anchor text
   - path_depth: number of URL path segments
   - leaf_word_count: number of hyphen-separated words in the last path segment
   - leaf_is_infrastructure, leaf_is_navigational, leaf_is_recipe_related: 1 if leaf segment matches keyword lists
   - mid_infrastructure, mid_nav, mid_recipe: counts of mid-path segments matching keyword lists

2. **Learned model**: Logistic regression on these features predicts P(non-recipe). This probability is used as the priority score (lower = more likely recipe = crawled first). The model is retrained after each crawl run.

3. **Segment keyword lists** (infrastructure_segments.txt, navigational_segments.txt, recipe_related_segments.txt) control which URL path segments trigger the leaf_is_* and mid_* features.

## Your job

Your primary role is to propose **structural improvements** to the crawler — not to tune numbers. Specifically:

1. **Feature proposals** (most important): Propose new raw features that would help the model distinguish recipe URLs from non-recipe URLs. Good features are computable from URL structure, anchor text, or crawl state without fetching the page. Each proposal should include a name, what it captures, and how to compute it.

2. **Policy proposals**: Propose new crawling strategies or architectural changes. Examples: new deduplication approaches, multi-phase crawling, link graph analysis, domain-specific handling. Think beyond parameter tuning.

3. **Seeds and segments**: Curate the seed list and segment keyword lists based on what the data shows. This is secondary to feature and policy proposals.

Guidelines:
- Look at which features have zero or near-zero coefficients — they may need richer signal or may be poorly defined
- Look at the gap between the strongest and weakest features — what information is the model missing?
- Consider what signals a human would use to guess if a URL leads to a recipe page, then propose features that capture those signals
- For policy proposals, think about what the crawler's architecture fundamentally can't do right now
- Be specific and concrete in proposals — not "improve scoring" but "add has_date_in_path feature because recipe blogs use /YYYY/MM/slug patterns"

Return only valid JSON matching the schema. No markdown, no code blocks, no explanation outside the JSON."""


def load_model_info(model_path: str) -> dict | None:
    """Load model metadata and coefficients for strategy context."""
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        feature_names = data["feature_names"]
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        coefs = clf.coef_[0] / scaler.scale_
        return {
            "feature_names": feature_names,
            "coefficients": {
                name: round(float(coef), 4)
                for name, coef in zip(feature_names, coefs)
            },
            "trained_at": data.get("trained_at"),
            "n_samples": data.get("n_samples"),
            "logfiles": data.get("logfiles"),
        }
    except Exception as e:
        print(f"Warning: could not load model info from {model_path}: {e}", file=sys.stderr)
        return None


def load_json(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def condense_replay(replay: dict) -> dict:
    """Trim replay for token efficiency at scale."""
    condensed = dict(replay)

    # Cap filter samples
    if "filter_audit" in condensed:
        for reason, data in condensed["filter_audit"].items():
            samples = data.get("sample_urls", [])
            if len(samples) > MAX_FILTER_SAMPLES:
                data["sample_urls"] = samples[:MAX_FILTER_SAMPLES]

    # Collapse zero-visit domains if there are many
    if "domain_breakdown" in condensed:
        breakdown = condensed["domain_breakdown"]
        zero_visit = {d: v for d, v in breakdown.items() if v["pages_visited"] == 0}
        if len(zero_visit) > DOMAIN_COLLAPSE_THRESHOLD:
            active = {d: v for d, v in breakdown.items() if v["pages_visited"] > 0}
            active["_zero_visit_domains"] = {
                "count": len(zero_visit),
                "domains": list(zero_visit.keys()),
                "note": "These domains were seeded but received 0 page visits — starved by concentrated domains",
            }
            condensed["domain_breakdown"] = active

    return condensed


def build_messages(replay: dict, previous_strategy: dict | None, model_info: dict | None = None) -> list[dict]:
    previous_str = json.dumps(previous_strategy, indent=2) if previous_strategy else "null — this is the first run"

    model_section = ""
    if model_info:
        model_section = f"""
## Current Model
Trained on {model_info['n_samples']} samples.
Feature coefficients (positive = predicts recipe, negative = predicts non-recipe):
{json.dumps(model_info['coefficients'], indent=2)}
"""

    user_content = f"""## Output Schema
{json.dumps(STRATEGY_SCHEMA, indent=2)}

## Previous Strategy
{previous_str}
{model_section}
## Crawl Replay Analysis
{json.dumps(replay, indent=2)}

## Instructions
Analyze the replay data and model performance, then produce a strategy.
- Propose at least 2-3 new features that would improve the model
- Propose at least 1 architectural or policy change
- Curate seeds and segments based on what the data shows
- Explain your reasoning, tying proposals to specific patterns in the data
- Return only valid JSON matching the schema above"""

    return [{"role": "user", "content": user_content}]


def call_claude(messages: list[dict]) -> str:
    """Call Claude via the Claude Code CLI non-interactively."""
    # Combine system prompt + user message into a single prompt
    user_content = messages[0]["content"]
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"

    result = subprocess.run(
        ["claude", "-p", full_prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI exited with code {result.returncode}:\n{result.stderr}")

    response = json.loads(result.stdout)
    return response["result"]


def parse_strategy(response: str) -> dict:
    """Extract and validate JSON from Claude's response."""
    text = response.strip()
    # Strip markdown code fences if present despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)


def save_strategy(strategy: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(output_dir, f"strategy_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(strategy, f, indent=2)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate a crawl strategy from a replay analysis")
    parser.add_argument("replay_json", help="Path to replay --mode json output")
    parser.add_argument("--previous-strategy", default=None, help="Path to previous strategy JSON for iterative improvement")
    parser.add_argument("--model", default=None, help="Path to trained model pkl (provides feature coefficients to LLM)")
    parser.add_argument("--output-dir", default="results", help="Directory to write strategy JSON (default: results)")
    args = parser.parse_args()

    replay = load_json(args.replay_json)
    previous = load_json(args.previous_strategy) if args.previous_strategy else None
    model_info = load_model_info(args.model) if args.model else None

    condensed = condense_replay(replay)
    messages = build_messages(condensed, previous, model_info)

    print("Calling Claude Opus...", file=sys.stderr)
    response = call_claude(messages)

    try:
        strategy = parse_strategy(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Claude response as JSON: {e}", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(response, file=sys.stderr)
        sys.exit(1)

    # Stamp metadata
    strategy["timestamp"] = datetime.now().isoformat()
    strategy["based_on_log"] = replay.get("meta", {}).get("logfile", args.replay_json)
    strategy["previous_strategy"] = args.previous_strategy

    filepath = save_strategy(strategy, args.output_dir)
    print(f"Strategy saved to: {filepath}", file=sys.stderr)
    print()
    print(strategy.get("summary", "(no summary)"))


if __name__ == "__main__":
    main()
