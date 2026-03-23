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
    "seeds": {
        "keep": ["<homepage URL>"],
        "remove": ["<homepage URL>"],
        "add": ["<homepage URL>"],
    },
    "crawler_params": {
        "max_pages": "<int>",
        "concurrency": "<int>",
        "delay_secs": "<float>",
    },
    "scoring_config": {
        "max_score_threshold": "<float between 0.5 and 0.95>",
        "rescore_sensitivity": "<float>",
        "lock_penalty": "<float>",
        "lsh_threshold": "<float>",
        "num_perm": "<int>",
        "components": {
            "base": "<float>",
            "dead_branch_penalty": "<float>",
            "dead_branch_threshold": "<float between 0.0 and 1.0>",
            "domain_multiplier": "<float>",
            "lsh_multiplier": "<float>",
        },
        "anchor": {
            "empty": "<float>",
            "one_word": "<float>",
            "two_three_words": "<float>",
            "four_plus_words": "<float>",
        },
        "leaf": {
            "infrastructure": "<float>",
            "navigational": "<float>",
            "recipe_single_word": "<float>",
            "single_word_default": "<float>",
            "two_words": "<float>",
            "three_words": "<float>",
            "four_plus_words": "<float>",
        },
        "mid": {
            "infrastructure": "<float>",
            "navigational": "<float>",
            "recipe_related": "<float>",
        },
    },
    "segment_additions": {
        "infrastructure": ["<segment string>"],
        "navigational": ["<segment string>"],
        "recipe_related": ["<segment string>"],
    },
    "budget_notes": "<plain text explanation of budget allocation reasoning>",
    "reasoning": "<plain text explanation of all decisions>",
    "summary": "<2-3 sentence human-readable summary of key changes and why>",
}

SYSTEM_PROMPT = """You are a crawl strategy advisor for a recipe web crawler called cookie-monster-crawl.

The crawler works as follows:
- It starts from seed URLs (recipe website homepages) and follows internal links
- URLs are scored and prioritized using a min-heap priority queue (lower score = higher priority = crawled sooner)
- Score components: base (constant 0.4), dead_branch (penalizes path roots with <20% recipe rate), segments (rewards/penalizes based on URL path keywords), domain_share (penalizes over-represented domains), anchor (rewards descriptive anchor text), lsh (penalizes near-duplicate URLs)
- Final score = sigmoid(sum of components); URLs above max_score_threshold are filtered out entirely

Segment lists control the scoring:
- infrastructure_segments: path segments that strongly indicate non-recipe pages (e.g. "author", "login") — penalized heavily
- navigational_segments: mid-level navigational pages — penalized moderately
- recipe_related_segments: path segments that indicate recipe index pages — rewarded moderately

Scoring config controls all numeric weights. Key fields:
- components.base: constant added to every URL's raw score
- components.dead_branch_penalty: added when a path root has <dead_branch_threshold recipe rate
- components.domain_multiplier: scales the domain share penalty (higher = more aggressive concentration penalty)
- components.lsh_multiplier: scales the near-duplicate penalty per LSH match
- anchor.*: penalties/rewards based on anchor text word count (lower = higher priority)
- leaf.*: penalties/rewards for the last path segment by type and word count
- mid.*: penalties/rewards for mid-path segments by type
- max_score_threshold: URLs scoring above this are filtered entirely (0.5–0.95)
- rescore_sensitivity: how much a score must worsen before a URL is requeued

Your job: analyze a replay of a past crawl run and produce a JSON strategy document for the next run.

Guidelines:
- Domain concentration (one domain consuming >20% of budget) is a key problem — consider raising domain_multiplier
- Seeds with 0 pages visited were likely starved by concentrated domains — consider whether to keep them
- Filter samples showing legitimate recipe pages being filtered suggest the threshold is too aggressive or leaf/mid weights need adjustment
- Component analysis shows which scoring components actually differentiated recipes from non-recipes — tune weights accordingly
- dead_branch and lsh both at 0.0 means the model hasn't learned from failures yet — this is expected early on
- Only include scoring_config fields you want to change — omitted fields keep their current values

Return only valid JSON matching the schema. No markdown, no code blocks, no explanation outside the JSON."""


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


def build_messages(replay: dict, previous_strategy: dict | None) -> list[dict]:
    previous_str = json.dumps(previous_strategy, indent=2) if previous_strategy else "null — this is the first run"

    user_content = f"""## Output Schema
{json.dumps(STRATEGY_SCHEMA, indent=2)}

## Previous Strategy
{previous_str}

## Crawl Replay Analysis
{json.dumps(replay, indent=2)}

## Instructions
Analyze the replay data and produce a strategy document for the next crawl run.
- Identify the top 2-3 problems visible in the data
- Suggest concrete changes to seeds, crawler params, and segment lists
- For seeds.add, provide full homepage URLs (e.g. "https://www.example.com")
- Explain your reasoning in the reasoning field
- Write a 2-3 sentence human-readable summary in the summary field
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
    parser.add_argument("--output-dir", default="results", help="Directory to write strategy JSON (default: results)")
    args = parser.parse_args()

    replay = load_json(args.replay_json)
    previous = load_json(args.previous_strategy) if args.previous_strategy else None

    condensed = condense_replay(replay)
    messages = build_messages(condensed, previous)

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
