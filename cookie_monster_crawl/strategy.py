"""
Strategy generator for cookie-monster-crawl.

Three-phase strategy generation:
  1. Analyze — LLM identifies problems and requests investigations
  2. Investigate — tools execute those requests (fetch URLs, query logs, read files)
  3. Propose — LLM produces a strategy informed by investigation findings

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

from cookie_monster_crawl.investigation import TOOL_DESCRIPTIONS, execute as run_tools

PROJECT_ROOT = Path(__file__).parent.parent
MAX_FILTER_SAMPLES = 20
DOMAIN_COLLAPSE_THRESHOLD = 30

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

ANALYZE_SCHEMA = {
    "problems_identified": ["<problem description>"],
    "investigations": [
        {
            "id": "<unique_id>",
            "question": "<what you want to learn>",
            "tool": "<tool_name>",
            "args": {},
        }
    ],
}

# --- Shared architecture description ---

ARCHITECTURE_CONTEXT = """## Architecture

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

3. **Segment keyword lists** (infrastructure_segments.txt, navigational_segments.txt, recipe_related_segments.txt) control which URL path segments trigger the leaf_is_* and mid_* features."""

# --- Phase 1: Analyze ---

ANALYZE_PROMPT = f"""You are a crawl strategy advisor analyzing a recipe web crawler run.

{ARCHITECTURE_CONTEXT}

## Your job

Review the replay data and model info. Identify problems, anomalies, and opportunities. Then request investigations to gather evidence before making proposals.

## Available Tools

{json.dumps(TOOL_DESCRIPTIONS, indent=2)}

## Output Schema

Return JSON matching this structure:
{json.dumps(ANALYZE_SCHEMA, indent=2)}

Request 3-8 investigations. Focus on the most impactful questions — things that would change your recommendations if the answer surprised you.

CRITICAL: Your entire response must be a single JSON object. No explanation, no markdown, no code blocks."""

# --- Phase 3: Propose ---

PROPOSE_PROMPT = f"""You are a crawl strategy advisor for a recipe web crawler called cookie-monster-crawl.

{ARCHITECTURE_CONTEXT}

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
- Use the investigation findings to inform your proposals — they contain evidence you requested

Return only valid JSON matching the schema. No markdown, no code blocks, no explanation outside the JSON."""


# --- Helpers ---

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

    if "filter_audit" in condensed:
        for reason, data in condensed["filter_audit"].items():
            samples = data.get("sample_urls", [])
            if len(samples) > MAX_FILTER_SAMPLES:
                data["sample_urls"] = samples[:MAX_FILTER_SAMPLES]

    if "domain_breakdown" in condensed:
        breakdown = condensed["domain_breakdown"]
        zero_visit = {d: v for d, v in breakdown.items() if v["pages_visited"] == 0}
        if len(zero_visit) > DOMAIN_COLLAPSE_THRESHOLD:
            active = {d: v for d, v in breakdown.items() if v["pages_visited"] > 0}
            active["_zero_visit_domains"] = {
                "count": len(zero_visit),
                "domains": list(zero_visit.keys()),
                "note": "These domains were seeded but received 0 page visits",
            }
            condensed["domain_breakdown"] = active

    return condensed


def call_claude(system_prompt: str, user_content: str) -> str:
    """Call Claude via the Claude Code CLI non-interactively."""
    full_prompt = f"{system_prompt}\n\n{user_content}"

    result = subprocess.run(
        ["claude", "-p", full_prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI exited with code {result.returncode}:\n{result.stderr}")

    response = json.loads(result.stdout)
    return response["result"]


def parse_json_response(response: str) -> dict:
    """Extract and validate JSON from Claude's response."""
    text = response.strip()
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


# --- Message builders ---

def _model_section(model_info: dict | None) -> str:
    if not model_info:
        return ""
    return f"""
## Current Model
Trained on {model_info['n_samples']} samples.
Feature coefficients (positive = predicts recipe, negative = predicts non-recipe):
{json.dumps(model_info['coefficients'], indent=2)}
"""


def _replay_section(replay: dict, previous_strategy: dict | None) -> str:
    previous_str = json.dumps(previous_strategy, indent=2) if previous_strategy else "null — this is the first run"
    return f"""## Previous Strategy
{previous_str}

## Crawl Replay Analysis
{json.dumps(replay, indent=2)}"""


def build_analyze_content(replay: dict, previous_strategy: dict | None, model_info: dict | None) -> str:
    return f"""{_model_section(model_info)}
{_replay_section(replay, previous_strategy)}

## Instructions
Identify problems in the data and request investigations to gather evidence.
Return only valid JSON matching the schema above."""


def build_propose_content(replay: dict, previous_strategy: dict | None, model_info: dict | None, findings: dict[str, str]) -> str:
    findings_section = "## Investigation Findings\n\n"
    for inv_id, result in findings.items():
        findings_section += f"### {inv_id}\n{result}\n\n"

    return f"""## Output Schema
{json.dumps(STRATEGY_SCHEMA, indent=2)}

{_model_section(model_info)}
{_replay_section(replay, previous_strategy)}

{findings_section}
## Instructions
Analyze the replay data, model performance, and investigation findings, then produce a strategy.
- Propose at least 2-3 new features that would improve the model
- Propose at least 1 architectural or policy change
- Curate seeds and segments based on what the data shows
- Use investigation findings as evidence — they were gathered to answer your questions
- Explain your reasoning, tying proposals to specific patterns in the data
- Return only valid JSON matching the schema above"""


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate a crawl strategy from a replay analysis")
    parser.add_argument("replay_json", help="Path to replay --mode json output")
    parser.add_argument("--previous-strategy", default=None, help="Path to previous strategy JSON")
    parser.add_argument("--model", default=None, help="Path to trained model pkl")
    parser.add_argument("--output-dir", default="results", help="Directory to write strategy JSON")
    parser.add_argument("--skip-investigation", action="store_true", help="Skip investigation phase (single-call mode)")
    parser.add_argument("--max-fetches", type=int, default=5, help="Max URL fetches during investigation (default: 5)")
    args = parser.parse_args()

    replay = load_json(args.replay_json)
    previous = load_json(args.previous_strategy) if args.previous_strategy else None
    model_info = load_model_info(args.model) if args.model else None
    condensed = condense_replay(replay)

    logfile = replay.get("meta", {}).get("logfile", "")

    if args.skip_investigation:
        # Single-call mode (original behavior)
        print("Calling Claude (single-phase)...", file=sys.stderr)
        content = build_propose_content(condensed, previous, model_info, {})
        response = call_claude(PROPOSE_PROMPT, content)
    else:
        # Phase 1: Analyze
        print("Phase 1: Analyzing crawl data...", file=sys.stderr)
        analyze_content = build_analyze_content(condensed, previous, model_info)
        analyze_response = call_claude(ANALYZE_PROMPT, analyze_content)

        try:
            analysis = parse_json_response(analyze_response)
        except json.JSONDecodeError:
            print("Warning: could not parse analysis response, falling back to single-call mode", file=sys.stderr)
            content = build_propose_content(condensed, previous, model_info, {})
            response = call_claude(PROPOSE_PROMPT, content)
            analysis = None

        if analysis is not None:
            problems = analysis.get("problems_identified", [])
            investigations = analysis.get("investigations", [])

            print(f"\nProblems identified: {len(problems)}", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)

            print(f"\nInvestigations requested: {len(investigations)}", file=sys.stderr)
            for inv in investigations:
                print(f"  [{inv.get('tool')}] {inv.get('question', '')}", file=sys.stderr)

            # Phase 2: Investigate
            print("\nPhase 2: Running investigations...", file=sys.stderr)
            findings = run_tools(
                investigations,
                logfile=logfile,
                project_root=PROJECT_ROOT,
                max_fetches=args.max_fetches,
            )

            for inv_id, result in findings.items():
                preview = result[:100].replace("\n", " ")
                print(f"  {inv_id}: {preview}...", file=sys.stderr)

            # Phase 3: Propose
            print("\nPhase 3: Generating strategy...", file=sys.stderr)
            content = build_propose_content(condensed, previous, model_info, findings)
            response = call_claude(PROPOSE_PROMPT, content)

    try:
        strategy = parse_json_response(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse strategy response as JSON: {e}", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(response, file=sys.stderr)
        sys.exit(1)

    # Stamp metadata
    strategy["timestamp"] = datetime.now().isoformat()
    strategy["based_on_log"] = replay.get("meta", {}).get("logfile", args.replay_json)
    strategy["previous_strategy"] = args.previous_strategy

    filepath = save_strategy(strategy, args.output_dir)
    print(f"\nStrategy saved to: {filepath}", file=sys.stderr)
    print()
    print(strategy.get("summary", "(no summary)"))


if __name__ == "__main__":
    main()
