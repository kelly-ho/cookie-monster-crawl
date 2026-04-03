"""
Auto-implement feature proposals from a strategy document.

Reads feature_proposals from a strategy JSON, sends them along with
the current source code to Claude, applies the returned edits,
runs tests, retrains the model, and evaluates accuracy.

Usage:
    python -m cookie_monster_crawl.implement <strategy_json> [options]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from cookie_monster_crawl.train import next_model_path

PROJECT_ROOT = Path(__file__).parent.parent
UTILS_PATH = PROJECT_ROOT / "cookie_monster_crawl" / "utils.py"
TRAIN_PATH = PROJECT_ROOT / "cookie_monster_crawl" / "train.py"

IMPLEMENT_PROMPT = """You are a Python developer implementing new features for a URL scoring model.

You will be given:
1. The current source code for utils.py and train.py
2. A list of feature proposals, each with a name, description, and computation

Your job: produce JSON edits that add each proposed feature to the codebase.

## Rules

- Add each feature to BOTH `extract_features()` AND the `raw_features` dict inside `calculate_score()` in utils.py
- Add each feature name to `FEATURE_NAMES` in train.py
- Use the `url` variable and `urlparse` (already imported) for URL parsing
- Use `re` (already imported) for regex
- Use `segments` (list of path segments) and other local variables already available in those methods
- Keep feature values as simple numeric types (int or float)
- Do not modify any existing features or logic — only add new entries
- Do not add new imports unless absolutely necessary

## Output format

Return ONLY valid JSON matching this schema:

{
  "utils_extract_features": [
    {"name": "feature_name", "code": "python expression or statement"}
  ],
  "utils_calculate_score": [
    {"name": "feature_name", "code": "python expression or statement"}
  ],
  "train_feature_names": ["feature_name"],
  "new_imports": []
}

Each "code" value should be a Python expression that evaluates to the feature value.
It can reference: url, parsed (urlparse result), segments, leaf, mid_path, leaf_words, anchor_text, domain_share, similar_junk, stats, self.

For utils_extract_features, available locals are: url, domain, segments, root, stats, similar_junk, domain_share, leaf, mid_path, leaf_words, is_dead_branch, anchor_text.
For utils_calculate_score, available locals are: url, domain, segments, root, stats, similar_junk, domain_share, m, c, components, anchor_text.

CRITICAL: Your entire response must be a single JSON object. No explanation, no markdown, no code blocks, no text before or after the JSON. If you cannot produce the edits, return {"error": "reason"}."""


def load_strategy(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def call_claude(prompt: str) -> str:
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI exited with code {result.returncode}:\n{result.stderr}")
    response = json.loads(result.stdout)
    return response["result"]


def parse_response(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)


def apply_extract_features(utils_code: str, features: list) -> str:
    """Insert new features into extract_features() return dict."""
    # Find the closing of the return dict in extract_features
    marker = '"mid_recipe":             sum(1 for s in mid_path if s in self.recipe_related_segments),'
    new_lines = []
    for feat in features:
        new_lines.append(f'            "{feat["name"]}": {feat["code"]},')
    insert = "\n".join(new_lines)
    return utils_code.replace(marker, marker + "\n" + insert, 1)


def apply_calculate_score(utils_code: str, features: list) -> str:
    """Insert new features into calculate_score() raw_features dict."""
    marker = '"mid_recipe":             sum(1 for s in segments[:-1] if s in self.recipe_related_segments),'
    new_lines = []
    for feat in features:
        new_lines.append(f'            "{feat["name"]}": {feat["code"]},')
    insert = "\n".join(new_lines)
    return utils_code.replace(marker, marker + "\n" + insert, 1)


def apply_train_features(train_code: str, feature_names: list) -> str:
    """Add new feature names to FEATURE_NAMES list in train.py."""
    marker = '    "mid_recipe",'
    new_lines = []
    for name in feature_names:
        new_lines.append(f'    "{name}",')
    insert = "\n".join(new_lines)
    return train_code.replace(marker, marker + "\n" + insert, 1)


def apply_imports(utils_code: str, imports: list) -> str:
    """Add new imports to utils.py if needed."""
    for imp in imports:
        if imp not in utils_code:
            utils_code = imp + "\n" + utils_code
    return utils_code


def run_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def retrain(logfiles: list[str], output: str) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "cookie_monster_crawl.train"] + logfiles + ["--output", output]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0, result.stdout


def main():
    parser = argparse.ArgumentParser(description="Auto-implement feature proposals from a strategy")
    parser.add_argument("strategy_json", help="Path to strategy JSON with feature_proposals")
    parser.add_argument("--logfiles", nargs="+", required=True, help="Crawl log files for retraining")
    parser.add_argument("--features", nargs="+", default=None, help="Only implement these features by name (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show edits without applying")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy_json)
    proposals = strategy.get("feature_proposals", [])

    if args.features:
        selected = set(args.features)
        proposals = [p for p in proposals if p["name"] in selected]
        missing = selected - {p["name"] for p in proposals}
        if missing:
            print(f"Warning: features not found in strategy: {', '.join(missing)}", file=sys.stderr)

    if not proposals:
        print("No feature proposals to implement.")
        return

    print(f"Implementing {len(proposals)} feature proposals:")
    for p in proposals:
        print(f"  - {p['name']}: {p['description'][:80]}...")

    # Read current source
    utils_code = read_file(UTILS_PATH)
    train_code = read_file(TRAIN_PATH)

    # Build prompt
    proposals_str = json.dumps(proposals, indent=2)
    prompt = f"""{IMPLEMENT_PROMPT}

## Current utils.py
```python
{utils_code}
```

## Current train.py
```python
{train_code}
```

## Feature Proposals
{proposals_str}"""

    print("\nCalling Claude to generate implementations...")
    response = call_claude(prompt)

    try:
        edits = parse_response(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON: {e}")
        print("Raw response:")
        print(response)
        sys.exit(1)

    print("\nProposed edits:")
    for feat in edits.get("utils_extract_features", []):
        print(f"  extract_features += {feat['name']}: {feat['code']}")
    for feat in edits.get("utils_calculate_score", []):
        print(f"  calculate_score  += {feat['name']}: {feat['code']}")
    for name in edits.get("train_feature_names", []):
        print(f"  FEATURE_NAMES    += {name}")
    for imp in edits.get("new_imports", []):
        print(f"  import           += {imp}")

    if args.dry_run:
        print("\nDry run — no files modified.")
        return

    # Back up originals
    utils_backup = UTILS_PATH.with_suffix(".py.bak")
    train_backup = TRAIN_PATH.with_suffix(".py.bak")
    shutil.copy2(UTILS_PATH, utils_backup)
    shutil.copy2(TRAIN_PATH, train_backup)
    print(f"\nBackups: {utils_backup.name}, {train_backup.name}")

    # Apply edits
    new_utils = utils_code
    if edits.get("new_imports"):
        new_utils = apply_imports(new_utils, edits["new_imports"])
    if edits.get("utils_extract_features"):
        new_utils = apply_extract_features(new_utils, edits["utils_extract_features"])
    if edits.get("utils_calculate_score"):
        new_utils = apply_calculate_score(new_utils, edits["utils_calculate_score"])

    new_train = train_code
    if edits.get("train_feature_names"):
        new_train = apply_train_features(new_train, edits["train_feature_names"])

    UTILS_PATH.write_text(new_utils, encoding="utf-8")
    TRAIN_PATH.write_text(new_train, encoding="utf-8")
    print("Edits applied.")

    # Run tests
    print("\nRunning tests...")
    if not run_tests():
        print("\nTests failed. Reverting changes.")
        shutil.copy2(utils_backup, UTILS_PATH)
        shutil.copy2(train_backup, TRAIN_PATH)
        sys.exit(1)
    print("Tests passed.")

    # Retrain
    print("\nRetraining model...")
    model_output = str(next_model_path())
    ok, output = retrain(args.logfiles, model_output)
    if not ok:
        print("\nRetrain failed. Reverting changes.")
        shutil.copy2(utils_backup, UTILS_PATH)
        shutil.copy2(train_backup, TRAIN_PATH)
        sys.exit(1)

    # Clean up backups
    utils_backup.unlink()
    train_backup.unlink()

    print(f"\nDone. New model: {model_output}")
    print("Review the accuracy above. If it regressed, revert with git checkout.")


if __name__ == "__main__":
    main()
