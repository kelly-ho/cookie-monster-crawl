"""
Train a logistic regression model on crawl log data.

Reads one or more JSONL crawl logs, extracts (features, label) pairs
from visited URLs, and trains a logistic regression model.

Raw features (config-independent) from discover events are used for training.
Labels come from result events (is_recipe).

Usage:
    python -m cookie_monster_crawl.train <logfile> [logfile ...] [options]
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "domain_share",
    "lsh_count",
    "dead_branch",
    "anchor_word_count",
    "path_depth",
    "leaf_word_count",
    "leaf_is_infrastructure",
    "leaf_is_navigational",
    "leaf_is_recipe_related",
    "mid_infrastructure",
    "mid_nav",
    "mid_recipe",
    "has_date_in_path",
    "query_param_count",
    "slug_word_count_ratio",
    "has_numeric_id",
    "is_print_or_wprm",
]

MODEL_DIR = Path("models")


def load_events(logfiles: list[str]) -> tuple[dict, dict]:
    """Load discover and result events from JSONL logs, keyed by URL."""
    discovers = {}  # url -> raw_features
    results = {}    # url -> is_recipe

    for logfile in logfiles:
        with open(logfile, encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                url = event.get("url")
                if not url:
                    continue
                if event["type"] == "discover" and event.get("raw_features"):
                    discovers[url] = event["raw_features"]
                elif event["type"] == "result":
                    results[url] = event["is_recipe"]

    return discovers, results


def build_dataset(discovers: dict, results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Join discover features with result labels."""
    X, y = [], []
    for url, is_recipe in results.items():
        if url not in discovers:
            continue
        features = discovers[url]
        X.append([features.get(f, 0.0) for f in FEATURE_NAMES])
        y.append(1 if is_recipe else 0)
    return np.array(X), np.array(y)


def next_model_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    existing = sorted(MODEL_DIR.glob("model_v*.pkl"))
    version = len(existing) + 1
    return MODEL_DIR / f"model_v{version}.pkl"


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    return model


def print_report(model: Pipeline, X_train, y_train, X_test, y_test, filepath: Path):
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    n_recipes = int(y_train.sum()) + int(y_test.sum())
    n_non = (len(y_train) + len(y_test)) - n_recipes
    print(f"\nModel saved to: {filepath}")
    print(f"Total samples: {len(y_train) + len(y_test)}  (recipes={n_recipes}, non-recipes={n_non})")
    print(f"Train/test split: {len(y_train)}/{len(y_test)}")

    if int(y_train.sum()) == 0 or int((~y_train.astype(bool)).sum()) == 0:
        print("Warning: only one class in training data — coefficients not meaningful")
        return

    coefs = clf.coef_[0] / scaler.scale_
    print("\nFeature coefficients (positive = predicts recipe):")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s} {coef:+.4f}")

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nTrain accuracy: {train_acc:.1%}   Test accuracy: {test_acc:.1%}   Baseline: {baseline:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression on crawl log data")
    parser.add_argument("logfiles", nargs="+", help="JSONL crawl log files")
    parser.add_argument("--output", default=None, help="Output model path (default: models/model_vN.pkl)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set (default: 0.2)")
    args = parser.parse_args()

    discovers, results = load_events(args.logfiles)
    X, y = build_dataset(discovers, results)

    if len(y) == 0:
        print("No labeled samples found. Logs must contain raw_features in discover events.", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = train(X_train, y_train)

    filepath = Path(args.output) if args.output else next_model_path()
    MODEL_DIR.mkdir(exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "trained_at": datetime.now().isoformat(),
            "logfiles": args.logfiles,
            "n_samples": len(y),
        }, f)

    print_report(model, X_train, y_train, X_test, y_test, filepath)


if __name__ == "__main__":
    main()
