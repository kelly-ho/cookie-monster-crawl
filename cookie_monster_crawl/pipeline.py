"""
Pipeline: chain crawl → replay → strategy → apply → train into a single run.

Usage:
    python -m cookie_monster_crawl.pipeline --model models/model_v22.pkl --seeds data/static-target.json
    python -m cookie_monster_crawl.pipeline --model models/model_v22.pkl --seeds data/static-target.json --skip-strategy
    python -m cookie_monster_crawl.pipeline --model models/model_v22.pkl --seeds data/static-target.json --domain-stats data/domain_stats.json
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

from cookie_monster_crawl.outcomes import record_outcome


def run(cmd, description):
    """Run a command, streaming output. Exit on failure."""
    print(f"\n{'='*60}")
    print(f"PIPELINE: {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nPIPELINE ERROR: '{description}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def find_latest_crawl_log():
    logs = sorted(glob.glob("crawl_logs/crawl_*.jsonl"), key=lambda f: Path(f).stat().st_mtime)
    return logs[-1] if logs else None


def find_latest_strategy():
    strategies = sorted(glob.glob("results/strategy_*.json"), key=lambda f: Path(f).stat().st_mtime)
    return strategies[-1] if strategies else None


def find_latest_run_result():
    results = sorted(glob.glob("results/run_*.json"), key=lambda f: Path(f).stat().st_mtime)
    return results[-1] if results else None


def load_harvest_info(result_path: str) -> dict:
    """Extract harvest info from a run result JSON."""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "harvest_pct": data.get("harvest_efficiency_percent", 0),
        "recipes": data.get("recipes_found", 0),
        "total_fetched": data.get("total_fetched", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Cookie Monster Crawl Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to scoring model pkl")
    parser.add_argument("--seeds", type=str, default="data/static-target.json", help="Path to seed URLs JSON")
    parser.add_argument("--max-pages", type=int, default=1000, help="Max pages per crawl (default: 1000)")
    parser.add_argument("--domain-stats", type=str, default=None, help="Path to domain_stats.json from previous crawl")
    parser.add_argument("--skip-strategy", action="store_true", help="Skip strategy/apply phase, just crawl and train")
    parser.add_argument("--train-model", type=str, default="logistic_regression", help="Model type for training (default: logistic_regression)")
    parser.add_argument("--train-logs", type=str, default=None, help="Glob pattern for training logs (default: all crawl logs)")
    parser.add_argument("--debate-rounds", type=int, default=1, help="Number of critique-revise rounds (default: 1)")
    parser.add_argument("--no-critique", action="store_true", help="Skip the Critic debate phase")
    args = parser.parse_args()

    # Record harvest before this run (for outcome tracking)
    pre_run_result = find_latest_run_result()
    harvest_before = load_harvest_info(pre_run_result) if pre_run_result else None

    # Step 1: Crawl
    crawl_cmd = f"python -m cookie_monster_crawl.crawler --max-pages {args.max_pages} --model {args.model} --seeds {args.seeds}"
    if args.domain_stats:
        crawl_cmd += f" --domain-stats {args.domain_stats}"
    run(crawl_cmd, "Crawl")

    crawl_log = find_latest_crawl_log()
    if not crawl_log:
        print("PIPELINE ERROR: No crawl log found")
        sys.exit(1)
    print(f"\nUsing crawl log: {crawl_log}")

    # Step 2: Replay (JSON mode for strategy input)
    replay_json = "replay_output.json"
    run(f"python -m cookie_monster_crawl.replay {crawl_log} --mode json > {replay_json}", "Replay analysis")

    if not args.skip_strategy:
        # Step 3: Strategy
        prev_strategy = find_latest_strategy()
        strategy_cmd = f"python -m cookie_monster_crawl.strategy {replay_json} --model {args.model} --rounds {args.debate_rounds}"
        if prev_strategy:
            strategy_cmd += f" --previous-strategy {prev_strategy}"
        if args.no_critique:
            strategy_cmd += " --no-critique"
        run(strategy_cmd, "Strategy generation")

        # Step 4: Apply
        latest_strategy = find_latest_strategy()
        if latest_strategy:
            run(f"python -m cookie_monster_crawl.apply {latest_strategy} --seeds {args.seeds}", "Apply strategy")

    # Step 5: Train
    train_logs = args.train_logs or "crawl_logs/crawl_*.jsonl"
    run(f"python -m cookie_monster_crawl.train {train_logs} --model {args.train_model}", "Train model")

    # Step 6: Record outcome (compare harvest before/after strategy application)
    post_run_result = find_latest_run_result()
    if harvest_before and post_run_result and post_run_result != pre_run_result:
        harvest_after = load_harvest_info(post_run_result)
        applied_strategy = find_latest_strategy()
        if applied_strategy:
            try:
                with open(applied_strategy, encoding="utf-8") as f:
                    strategy_data = json.load(f)
                record_outcome(applied_strategy, strategy_data, harvest_before, harvest_after)
                print(f"\nOutcome recorded: {harvest_before['harvest_pct']}% → {harvest_after['harvest_pct']}%")
            except Exception as e:
                print(f"\nWarning: could not record outcome: {e}")

    print(f"\nDomain stats saved to data/domain_stats.json (auto-saved by crawler)")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
