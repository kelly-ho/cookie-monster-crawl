"""
Apply a strategy document to the crawler's seed and segment files.

Reads a strategy JSON produced by strategy.py and applies:
  - Seed changes to data/static-target.json
  - Segment additions to data/*.txt files

Crawler params are printed as a reminder but not auto-applied.

Usage:
    python -m cookie_monster_crawl.apply <strategy_json> [options]
"""

import argparse
import json
import os
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
SEEDS_FILE = DATA_DIR / "static-target.json"
CONFIG_FILE = DATA_DIR / "crawl_config.json"
SEGMENT_FILES = {
    "infrastructure": DATA_DIR / "infrastructure_segments.txt",
    "navigational": DATA_DIR / "navigational_segments.txt",
    "recipe_related": DATA_DIR / "recipe_related_segments.txt",
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_strategy(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_seeds(filepath: Path) -> list[str]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f).get("seeds", [])


def load_segments(filepath: Path) -> set[str]:
    with open(filepath, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_config(filepath: Path) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Compute diffs
# ---------------------------------------------------------------------------

def compute_seed_diff(strategy: dict, current_seeds: list[str]) -> dict:
    current_set = set(current_seeds)
    to_keep = set(strategy["seeds"].get("keep", []))
    to_remove = set(strategy["seeds"].get("remove", []))
    to_add = set(strategy["seeds"].get("add", []))

    warnings = []
    for url in to_remove:
        if url not in current_set:
            warnings.append(f"  WARNING: '{url}' in seeds.remove but not in current seed file")
    for url in to_keep:
        if url not in current_set:
            warnings.append(f"  WARNING: '{url}' in seeds.keep but not in current seed file")

    new_seeds = [s for s in current_seeds if s not in to_remove]
    for url in to_add:
        if url not in set(new_seeds):
            new_seeds.append(url)

    return {
        "current": current_seeds,
        "new": new_seeds,
        "added": sorted(to_add - current_set),
        "removed": sorted(to_remove & current_set),
        "unchanged": len(new_seeds) - len(to_add - current_set),
        "warnings": warnings,
    }


def compute_config_diff(strategy: dict, current_config: dict) -> dict:
    """Compute which scoring config values would change."""
    proposed = strategy.get("scoring_config", {})
    if not proposed:
        return {}

    current_scoring = current_config.get("scoring", {})
    changes = {}

    def _diff(current: dict, proposed: dict, path: str = ""):
        for key, val in proposed.items():
            full_key = f"{path}.{key}" if path else key
            current_val = current.get(key)
            if isinstance(val, dict) and isinstance(current_val, dict):
                _diff(current_val, val, full_key)
            elif val != current_val:
                changes[full_key] = {"from": current_val, "to": val}

    _diff(current_scoring, proposed)
    return changes


def compute_segment_diffs(strategy: dict) -> dict[str, dict]:
    additions = strategy.get("segment_additions", {})
    diffs = {}

    for key, filepath in SEGMENT_FILES.items():
        existing = load_segments(filepath)
        proposed = {s.strip() for s in additions.get(key, []) if s.strip()}
        new_entries = sorted(proposed - existing)
        already_present = sorted(proposed & existing)
        diffs[key] = {
            "filepath": filepath,
            "new_entries": new_entries,
            "already_present": already_present,
        }

    return diffs


# ---------------------------------------------------------------------------
# Print diff
# ---------------------------------------------------------------------------

def print_diff(strategy: dict, seed_diff: dict, segment_diffs: dict[str, dict], config_diff: dict = None):
    print(f"\nSTRATEGY: {strategy.get('based_on_log', '(unknown log)')}")
    print(f"File:     {strategy.get('timestamp', '')}")

    print(f"\n{'─'*60}")
    print(f"SEEDS  ({SEEDS_FILE})")
    print(f"{'─'*60}")
    if seed_diff["added"]:
        for url in seed_diff["added"]:
            print(f"  + {url}")
    if seed_diff["removed"]:
        for url in seed_diff["removed"]:
            print(f"  - {url}")
    if not seed_diff["added"] and not seed_diff["removed"]:
        print(f"  (no changes — {seed_diff['unchanged']} seeds kept)")
    else:
        print(f"  {seed_diff['unchanged']} seeds unchanged")
    for w in seed_diff["warnings"]:
        print(w)

    print(f"\n{'─'*60}")
    print("SEGMENT ADDITIONS")
    print(f"{'─'*60}")
    any_changes = False
    for key, diff in segment_diffs.items():
        filename = diff["filepath"].name
        new = diff["new_entries"]
        dupes = diff["already_present"]
        if new:
            any_changes = True
            print(f"  {filename}  +{len(new)}  ({', '.join(new)})")
        if dupes:
            print(f"  {filename}  skipped {len(dupes)} already present  ({', '.join(dupes)})")
    if not any_changes:
        print("  (no new segment additions)")

    if config_diff:
        print(f"\n{'─'*60}")
        print(f"SCORING CONFIG  ({CONFIG_FILE.name})")
        print(f"{'─'*60}")
        for key, change in config_diff.items():
            print(f"  {key}: {change['from']} → {change['to']}")
    elif strategy.get("scoring_config"):
        print(f"\n{'─'*60}")
        print(f"SCORING CONFIG  ({CONFIG_FILE.name})")
        print(f"{'─'*60}")
        print("  (no changes from current config)")

    params = strategy.get("crawler_params", {})
    scoring = strategy.get("scoring_config", {})
    cli_parts = []
    if params.get("max_pages"):
        cli_parts.append(f"--max-pages {params['max_pages']}")
    if params.get("concurrency"):
        cli_parts.append(f"--concurrency {params['concurrency']}")
    if params.get("delay_secs"):
        cli_parts.append(f"--delay {params['delay_secs']}")
    if scoring.get("max_score_threshold"):
        cli_parts.append(f"--max-score {scoring['max_score_threshold']}")
    if cli_parts:
        print(f"\n{'─'*60}")
        print("CRAWLER PARAMS  (apply manually)")
        print(f"{'─'*60}")
        print(f"  {' '.join(cli_parts)}")

    print()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_seeds(new_seeds: list[str], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"seeds": new_seeds}, f, indent=2)


def write_segments(segment_diffs: dict[str, dict]):
    for diff in segment_diffs.values():
        if diff["new_entries"]:
            with open(diff["filepath"], "a", encoding="utf-8") as f:
                for entry in diff["new_entries"]:
                    f.write(entry + "\n")


def write_config(strategy: dict, current_config: dict, filepath: Path):
    """Deep-merge strategy's scoring_config into current config and write."""
    proposed = strategy.get("scoring_config", {})
    scoring = current_config.get("scoring", {})

    def _merge(base: dict, updates: dict):
        for key, val in updates.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                _merge(base[key], val)
            else:
                base[key] = val

    _merge(scoring, proposed)
    current_config["scoring"] = scoring
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2)


def confirm() -> bool:
    try:
        answer = input("Apply these changes? [y/N] ").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply a crawl strategy to seed and segment files")
    parser.add_argument("strategy_json", help="Path to strategy JSON produced by strategy.py")
    parser.add_argument("--dry-run", action="store_true", help="Print diff and exit without writing")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy_json)
    current_seeds = load_seeds(SEEDS_FILE)
    current_config = load_config(CONFIG_FILE)

    seed_diff = compute_seed_diff(strategy, current_seeds)
    segment_diffs = compute_segment_diffs(strategy)
    config_diff = compute_config_diff(strategy, current_config)

    print_diff(strategy, seed_diff, segment_diffs, config_diff)

    has_changes = (
        seed_diff["added"]
        or seed_diff["removed"]
        or any(d["new_entries"] for d in segment_diffs.values())
        or bool(config_diff)
    )

    if not has_changes:
        print("Nothing to apply.")
        return

    if args.dry_run:
        print("Dry run — no files modified.")
        return

    if not args.yes and not confirm():
        print("Aborted.")
        return

    write_seeds(seed_diff["new"], SEEDS_FILE)
    write_segments(segment_diffs)
    if config_diff:
        write_config(strategy, current_config, CONFIG_FILE)
    print("Applied.")


if __name__ == "__main__":
    main()
