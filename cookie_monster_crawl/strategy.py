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
from cookie_monster_crawl.outcomes import load_outcomes, format_outcomes_for_prompt

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
    "config_proposals": [
        {
            "parameter": "<parameter name from crawl_config.json>",
            "current_value": "<current value>",
            "proposed_value": "<proposed value>",
            "rationale": "<why this change would improve harvest rate, with evidence>",
        }
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

2. **Learned model**: A trained classifier (random forest, gradient boosting, or logistic regression) on these features predicts P(non-recipe). This probability is used as the priority score (lower = more likely recipe = crawled first). The model is retrained after each crawl run.

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

Propose improvements to the crawler across four categories:

1. **Feature proposals**: Propose new raw features that would help the model distinguish recipe URLs from non-recipe URLs. Good features are computable from URL structure, anchor text, or crawl state without fetching the page. Each proposal should include a name, what it captures, and how to compute it.

2. **Config proposals**: Propose changes to crawl hyperparameters in crawl_config.json. Current values are shown in the config section below. Key tunable parameters:
   - `max_score_threshold`: URLs scoring above this are filtered. Higher = more permissive, lower = more aggressive filtering.
   - `lsh_threshold`: MinHash LSH similarity threshold for detecting near-duplicate junk URLs. Lower = catches more similar URLs, higher = requires closer matches.
   - `rescore_sensitivity`: How aggressively the batch rescore after seed pages shifts scores based on real domain statistics.
   - `lock_penalty`: Score penalty added when a domain's lock is held (another URL from that domain is being fetched).
   Each config proposal must include the current value, proposed value, and evidence-based rationale.

3. **Policy proposals**: Propose new crawling strategies or architectural changes. Examples: new deduplication approaches, multi-phase crawling, link graph analysis, domain-specific handling.

4. **Seeds and segments**: Curate the seed list and segment keyword lists based on what the data shows.

Guidelines:
- Review the previous strategy's proposals against the current model importances. If a feature was implemented but has near-zero importance, it failed — do not propose similar features. Explain what you learned from proposals that didn't work.
- Look at which features have zero or near-zero importance — they may need richer signal or may be poorly defined
- Look at the gap between the strongest and weakest features — what information is the model missing?
- Consider what signals a human would use to guess if a URL leads to a recipe page, then propose features that capture those signals
- For config proposals, use evidence from the crawl data (e.g. filter rates, domain harvest distributions, LSH match counts) to justify changes
- Be specific and concrete in proposals — not "improve scoring" but "add has_date_in_path feature because recipe blogs use /YYYY/MM/slug patterns"
- Use the investigation findings to inform your proposals — they contain evidence you requested

Return only valid JSON matching the schema. No markdown, no code blocks, no explanation outside the JSON."""

# --- Phase 4: Critique ---

CRITIQUE_SCHEMA = {
    "overall_assessment": "<accept|revise|reject>",
    "objections": [
        {
            "target": "<which proposal — e.g. 'feature:has_date_in_path' or 'segment:tips'>",
            "severity": "<critical|major|minor>",
            "claim": "<the specific claim being challenged>",
            "evidence": "<concrete evidence — URLs, log data, numbers from replay>",
            "suggestion": "<what should change>",
        }
    ],
    "investigations": [
        {
            "id": "<unique_id>",
            "question": "<what you want to verify about the proposal>",
            "tool": "<tool_name>",
            "args": {},
        }
    ],
    "endorsements": ["<proposals that are well-supported and should be kept as-is>"],
    "summary": "<2-3 sentence assessment>",
}

CRITIQUE_PROMPT = f"""You are a skeptical technical reviewer for a recipe web crawler called cookie-monster-crawl.

{ARCHITECTURE_CONTEXT}

## Your role

You are NOT the strategy proposer. Your goal is to stress-test the Proposer's strategy by finding specific, concrete evidence that proposals would fail, underperform, or cause unintended side effects.

You have a fundamentally different objective from the Proposer. The Proposer is rewarded for creativity and impact. You are rewarded for correctness and evidence. A good critique prevents wasted crawl cycles on ideas that sound plausible but don't work.

## Rules

1. **Evidence required**: Every objection must cite specific evidence — URL patterns from the crawl data, log query results, numbers from the replay analysis, or historical outcome data. An objection without evidence is speculation, and you must not raise it.

2. **Be specific**: Do not say "this might not work." Say "I checked 5 URLs matching this pattern and 3 would be misclassified because..." or "The replay data shows this feature has a correlation of X with recipe status, which is too weak to be useful."

3. **Use outcome history**: If similar proposals were tried before and had negligible or negative impact, this is strong evidence the current proposal needs more justification. Conversely, if similar approaches succeeded, endorse them.

4. **Severity calibration**:
   - critical: this would make the crawler worse than doing nothing
   - major: this would not achieve its stated goal but wouldn't cause harm
   - minor: this would work but could be improved

5. **Endorse good ideas**: You are skeptical, not adversarial. If a proposal is well-supported by the data, say so explicitly in your endorsements.

6. **Request investigations**: You may request up to 5 investigations to verify the Proposer's claims before finalizing your critique. Good investigations test specific assertions — if the Proposer claims a URL pattern only appears in recipe pages, query the log for counter-examples.

## Available Tools

{json.dumps(TOOL_DESCRIPTIONS, indent=2)}

## Output Schema

Return JSON matching this structure:
{json.dumps(CRITIQUE_SCHEMA, indent=2)}

CRITICAL: Your entire response must be a single JSON object. No explanation, no markdown, no code blocks."""

# --- Phase 5: Revise ---

REVISE_PROMPT = f"""You are a crawl strategy advisor for a recipe web crawler called cookie-monster-crawl.

{ARCHITECTURE_CONTEXT}

## Your role

You produced a strategy proposal and a technical reviewer has examined it, raising specific objections backed by evidence. Your job is to produce a revised strategy that addresses each objection.

## Rules

1. **Respond to every objection**: For each objection the Critic raised:
   - If the evidence is valid, modify your proposal and explain the change in your reasoning.
   - If the evidence is invalid or misinterpreted, rebut it with specific counter-reasoning.
   - Do NOT silently remove a challenged proposal. Either improve it or defend it.

2. **Keep endorsed elements**: Proposals the Critic endorsed should remain in your revision unless you have a reason to change them.

3. **Stay concrete**: Your revised proposals must be at least as specific as the originals. Do not weaken proposals into vague suggestions to avoid critique.

Return only valid JSON matching the schema. No markdown, no code blocks, no explanation outside the JSON."""


# --- Helpers ---

def load_model_info(model_path: str) -> dict | None:
    """Load model metadata and feature importances for strategy context."""
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        feature_names = data["feature_names"]
        model_type = data.get("model_type", "unknown")
        clf = model.named_steps["clf"]

        importances = {}
        if hasattr(clf, "feature_importances_"):
            importances = {name: round(float(imp), 4) for name, imp in zip(feature_names, clf.feature_importances_)}
        elif hasattr(clf, "coef_"):
            scaler = model.named_steps.get("scaler")
            coefs = clf.coef_[0] / scaler.scale_ if scaler else clf.coef_[0]
            importances = {name: round(float(coef), 4) for name, coef in zip(feature_names, coefs)}

        return {
            "model_type": model_type,
            "feature_names": feature_names,
            "feature_importances": importances,
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

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    import re
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    # Find first { to last } as fallback
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise json.JSONDecodeError("No JSON found in response", text, 0)


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
## Current Model ({model_info.get('model_type', 'unknown')})
Trained on {model_info['n_samples']} samples.
Feature importances (higher = more influential):
{json.dumps(model_info['feature_importances'], indent=2)}
"""


def _config_section() -> str:
    config_path = PROJECT_ROOT / "data" / "crawl_config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        scoring = config.get("scoring", {})
        return f"""
## Current Config (data/crawl_config.json)
{json.dumps(scoring, indent=2)}
"""
    except FileNotFoundError:
        return ""


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
{_config_section()}
{_replay_section(replay, previous_strategy)}

{findings_section}
## Instructions
Analyze the replay data, model performance, and investigation findings, then produce a strategy.
- Propose at least 2-3 new features that would improve the model
- Propose config parameter changes if the data suggests current values are suboptimal
- Propose at least 1 architectural or policy change
- Curate seeds and segments based on what the data shows
- Use investigation findings as evidence — they were gathered to answer your questions
- Explain your reasoning, tying proposals to specific patterns in the data
- Return only valid JSON matching the schema above"""


def build_critique_content(
    proposal: dict,
    condensed: dict,
    model_info: dict | None,
    findings: dict[str, str],
    outcomes: list[dict],
    critique_findings: dict[str, str] | None = None,
) -> str:
    outcomes_section = format_outcomes_for_prompt(outcomes)

    findings_section = ""
    if critique_findings:
        findings_section = "\n## Your Investigation Findings\n\n"
        for inv_id, result in critique_findings.items():
            findings_section += f"### {inv_id}\n{result}\n\n"

    original_findings_section = "## Proposer's Investigation Findings\n\n"
    for inv_id, result in findings.items():
        original_findings_section += f"### {inv_id}\n{result}\n\n"

    meta = condensed.get("meta", {})
    meta_section = f"""## Crawl Summary
Pages visited: {meta.get('pages_visited', '?')}
Recipes found: {meta.get('recipes_found', '?')}
URLs filtered: {meta.get('urls_filtered', '?')}"""

    return f"""## Output Schema
{json.dumps(CRITIQUE_SCHEMA, indent=2)}

{_model_section(model_info)}
{_config_section()}

{meta_section}

## Score Calibration
{json.dumps(condensed.get('score_calibration', []), indent=2)}

{outcomes_section}

{original_findings_section}

## Strategy Proposal to Review
{json.dumps(proposal, indent=2)}

{findings_section}
## Instructions
Review the Proposer's strategy. Find specific evidence that proposals would fail or underperform. Request investigations to verify claims you're uncertain about. Endorse proposals that are well-supported. For config proposals, verify that the evidence supports the proposed direction and magnitude of change.
Return only valid JSON matching the schema above."""


def build_revise_content(
    proposal: dict,
    critique: dict,
    condensed: dict,
    model_info: dict | None,
    findings: dict[str, str],
    critique_findings: dict[str, str] | None = None,
) -> str:
    critique_section = f"## Critic's Assessment\n{json.dumps(critique, indent=2)}"

    critique_findings_section = ""
    if critique_findings:
        critique_findings_section = "\n## Evidence Gathered by Critic\n\n"
        for inv_id, result in critique_findings.items():
            critique_findings_section += f"### {inv_id}\n{result}\n\n"

    return f"""## Output Schema
{json.dumps(STRATEGY_SCHEMA, indent=2)}

{_model_section(model_info)}
{_replay_section(condensed, None)}

## Your Original Proposal
{json.dumps(proposal, indent=2)}

{critique_section}

{critique_findings_section}
## Instructions
Address each objection from the Critic. Modify proposals where the evidence is valid, rebut where it is not. Keep endorsed elements. Produce a complete revised strategy.
Return only valid JSON matching the schema above."""


def run_debate(
    proposal: dict,
    condensed: dict,
    previous: dict | None,
    model_info: dict | None,
    findings: dict[str, str],
    logfile: str,
    max_fetches: int,
    rounds: int,
) -> tuple[dict, list[dict]]:
    """Run the Proposer-Critic debate. Returns (final_proposal, critique_log)."""
    outcomes = load_outcomes()
    critique_log = []

    for round_num in range(rounds):
        print(f"\nDebate round {round_num + 1}/{rounds}", file=sys.stderr)

        # Phase 4: Initial critique (may include investigation requests)
        print("  Critic: reviewing proposal...", file=sys.stderr)
        critique_content = build_critique_content(proposal, condensed, model_info, findings, outcomes)
        critique_response = call_claude(CRITIQUE_PROMPT, critique_content)

        try:
            critique = parse_json_response(critique_response)
        except json.JSONDecodeError:
            print("  Warning: could not parse critique, skipping debate round", file=sys.stderr)
            break

        # Phase 4b: Run critic's investigations if requested
        critic_investigations = critique.get("investigations", [])
        critique_findings = {}
        if critic_investigations:
            print(f"  Critic: running {len(critic_investigations)} investigations...", file=sys.stderr)
            critique_findings = run_tools(
                critic_investigations,
                logfile=logfile,
                project_root=PROJECT_ROOT,
                max_fetches=max_fetches,
            )
            for inv_id, result in critique_findings.items():
                preview = result[:80].replace("\n", " ")
                print(f"    {inv_id}: {preview}...", file=sys.stderr)

            # Re-critique with investigation findings
            print("  Critic: finalizing critique with evidence...", file=sys.stderr)
            critique_content = build_critique_content(
                proposal, condensed, model_info, findings, outcomes, critique_findings
            )
            critique_response = call_claude(CRITIQUE_PROMPT, critique_content)
            try:
                critique = parse_json_response(critique_response)
            except json.JSONDecodeError:
                print("  Warning: could not parse final critique", file=sys.stderr)

        # Log the critique
        objections = critique.get("objections", [])
        assessment = critique.get("overall_assessment", "unknown")
        print(f"  Critic assessment: {assessment} ({len(objections)} objections)", file=sys.stderr)
        for obj in objections:
            print(f"    [{obj.get('severity', '?')}] {obj.get('target', '?')}: {obj.get('claim', '')[:80]}", file=sys.stderr)

        critique_log.append({
            "round": round_num + 1,
            "assessment": assessment,
            "objections": len(objections),
            "endorsements": len(critique.get("endorsements", [])),
            "summary": critique.get("summary", ""),
        })

        # Early exit if critic accepts
        if assessment == "accept":
            print("  Critic accepted the proposal.", file=sys.stderr)
            break

        # Phase 5: Proposer revises
        print("  Proposer: revising based on critique...", file=sys.stderr)
        revise_content = build_revise_content(
            proposal, critique, condensed, model_info, findings, critique_findings
        )
        revise_response = call_claude(REVISE_PROMPT, revise_content)

        try:
            proposal = parse_json_response(revise_response)
        except json.JSONDecodeError:
            print("  Warning: could not parse revision, keeping previous proposal", file=sys.stderr)
            break

        print("  Revision complete.", file=sys.stderr)

    return proposal, critique_log


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate a crawl strategy from a replay analysis")
    parser.add_argument("replay_json", help="Path to replay --mode json output")
    parser.add_argument("--previous-strategy", default=None, help="Path to previous strategy JSON")
    parser.add_argument("--model", default=None, help="Path to trained model pkl")
    parser.add_argument("--output-dir", default="results", help="Directory to write strategy JSON")
    parser.add_argument("--skip-investigation", action="store_true", help="Skip investigation phase (single-call mode)")
    parser.add_argument("--max-fetches", type=int, default=5, help="Max URL fetches during investigation (default: 5)")
    parser.add_argument("--rounds", type=int, default=1, help="Number of critique-revise debate rounds (default: 1)")
    parser.add_argument("--no-critique", action="store_true", help="Skip the Critic debate phase")
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

    # Phase 4-5: Critique-Revise debate
    critique_log = []
    if not args.no_critique:
        strategy, critique_log = run_debate(
            strategy, condensed, previous, model_info, findings,
            logfile=logfile, max_fetches=args.max_fetches, rounds=args.rounds,
        )

    # Stamp metadata
    strategy["timestamp"] = datetime.now().isoformat()
    strategy["based_on_log"] = replay.get("meta", {}).get("logfile", args.replay_json)
    strategy["previous_strategy"] = args.previous_strategy
    strategy["debate_rounds"] = len(critique_log)
    strategy["critique_log"] = critique_log

    filepath = save_strategy(strategy, args.output_dir)
    print(f"\nStrategy saved to: {filepath}", file=sys.stderr)
    print()
    print(strategy.get("summary", "(no summary)"))


if __name__ == "__main__":
    main()
