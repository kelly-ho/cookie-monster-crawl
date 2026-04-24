"""
Apply a strategy document to the crawler's seed and segment files.

Reads a strategy JSON produced by strategy.py and applies:
  - Seed changes to data/static-target.json
  - Segment additions to data/*.txt files

Policy changes are printed as suggested CLI flags but not auto-applied.

Usage:
    python -m cookie_monster_crawl.apply <strategy_json> [options]
"""

import argparse
import json
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

def print_diff(strategy: dict, seed_diff: dict, segment_diffs: dict[str, dict]):
    print(f"\nSTRATEGY: {strategy.get('based_on_log', '(unknown log)')}")
    print(f"File:     {strategy.get('timestamp', '')}")

    features = strategy.get("feature_proposals", [])
    if features:
        print(f"\n{'─'*60}")
        print("FEATURE PROPOSALS  (requires code changes)")
        print(f"{'─'*60}")
        for feat in features:
            print(f"  {feat.get('name', '?')}")
            print(f"    {feat.get('description', '')}")
            print(f"    Computation: {feat.get('computation', '')}")

    configs = strategy.get("config_proposals", [])
    if configs:
        print(f"\n{'─'*60}")
        print(f"CONFIG PROPOSALS  ({CONFIG_FILE})")
        print(f"{'─'*60}")
        for c in configs:
            print(f"  {c.get('parameter', '?')}: {c.get('current_value', '?')} → {c.get('proposed_value', '?')}")
            print(f"    {c.get('rationale', '')}")

    policies = strategy.get("policy_proposals", [])
    if policies:
        print(f"\n{'─'*60}")
        print("POLICY PROPOSALS  (requires design decisions)")
        print(f"{'─'*60}")
        for p in policies:
            print(f"  - {p}")

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


def write_config(config_proposals: list[dict], filepath: Path):
    """Apply config proposals to crawl_config.json."""
    with open(filepath, encoding="utf-8") as f:
        config = json.load(f)

    scoring = config.get("scoring", {})
    for proposal in config_proposals:
        param = proposal["parameter"]
        value = proposal["proposed_value"]
        # Convert string values to appropriate types
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        # Check top-level scoring keys first, then nested
        if param in scoring:
            scoring[param] = value
        else:
            # Check nested dicts (components, anchor, leaf, mid)
            for section in ("components", "anchor", "leaf", "mid"):
                if section in scoring and param in scoring[section]:
                    scoring[section][param] = value
                    break

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


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

    seed_diff = compute_seed_diff(strategy, current_seeds)
    segment_diffs = compute_segment_diffs(strategy)

    print_diff(strategy, seed_diff, segment_diffs)

    config_proposals = strategy.get("config_proposals", [])

    has_changes = (
        seed_diff["added"]
        or seed_diff["removed"]
        or any(d["new_entries"] for d in segment_diffs.values())
        or config_proposals
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
    if config_proposals:
        write_config(config_proposals, CONFIG_FILE)
    print("Applied.")


if __name__ == "__main__":
    main()
