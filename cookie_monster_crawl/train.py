"""
Train URL scoring models on crawl log data.

Reads one or more JSONL crawl logs, extracts (features, label) pairs
from visited URLs, and trains multiple model types for comparison.

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    "is_roundup_slug",
    "anchor_has_recipe_keyword",
    "domain_harvest_rate",
    "query_param_count",
    "slug_word_count_ratio",
    "has_numeric_id",
    "is_print_or_wprm",
    "leaf_is_plural",
    "has_how_to_prefix",
    "has_what_is_prefix",
    "recipe_word_density",
    "is_listing_page",
    "path_has_comment_page",
]

MODEL_DIR = Path("models")

MODELS = {
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "random_forest": lambda: Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ]),
    "gradient_boosting": lambda: Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    ]),
    "svm": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}


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
                    if url not in discovers:
                        discovers[url] = event["raw_features"]
                elif event["type"] == "result":
                    if url not in results:
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


def train_model(name: str, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = MODELS[name]()
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model: Pipeline, name: str) -> list[tuple[str, float]]:
    """Extract feature importances/coefficients depending on model type."""
    clf = model.named_steps["clf"]

    if name == "logistic_regression":
        scaler = model.named_steps["scaler"]
        coefs = clf.coef_[0] / scaler.scale_
        return sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True)
    elif name in ("random_forest", "gradient_boosting"):
        importances = clf.feature_importances_
        return sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    elif name == "svm":
        # SVM with RBF kernel has no direct feature importances
        return []
    return []


def print_comparison(results: dict, X_test, y_test):
    """Print a side-by-side comparison of all models."""
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'Train':>8s} {'Test':>8s} {'vs Baseline':>12s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*12}")

    for name, (model, train_acc, test_acc) in sorted(results.items(), key=lambda x: -x[1][2]):
        delta = test_acc - baseline
        print(f"{name:<25s} {train_acc:>7.1%} {test_acc:>7.1%} {delta:>+11.1%}")

    print(f"{'baseline (majority)':<25s} {'':>8s} {baseline:>7.1%}")
    print(f"{'='*60}")


def print_feature_report(model: Pipeline, name: str):
    """Print feature importances for a single model."""
    importances = get_feature_importances(model, name)
    if not importances:
        return

    if name == "logistic_regression":
        print(f"\nFeature coefficients (positive = predicts recipe):")
        for feat, coef in importances:
            print(f"  {feat:25s} {coef:+.4f}")
    else:
        print(f"\nFeature importances:")
        for feat, imp in importances:
            print(f"  {feat:25s} {imp:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train URL scoring models on crawl log data")
    parser.add_argument("logfiles", nargs="+", help="JSONL crawl log files")
    parser.add_argument("--output", default=None, help="Output model path (default: models/model_vN.pkl)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Model type to save (default: best performing)")
    parser.add_argument("--balance", type=float, default=None,
                        help="Target ratio of majority class (e.g. 0.6 for 60/40). Undersamples training set only.")
    args = parser.parse_args()

    discovers, results = load_events(args.logfiles)
    X, y = build_dataset(discovers, results)

    if len(y) == 0:
        print("No labeled samples found. Logs must contain raw_features in discover events.", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    n_recipes = int(y.sum())
    n_non = len(y) - n_recipes
    print(f"Total samples: {len(y)}  (recipes={n_recipes}, non-recipes={n_non})")
    print(f"Train/test split: {len(y_train)}/{len(y_test)}")

    if args.balance:
        minority_mask = y_train == 0
        majority_mask = y_train == 1
        n_minority = minority_mask.sum()
        n_majority_target = int(n_minority * args.balance / (1 - args.balance))
        rng = np.random.RandomState(42)
        majority_idx = rng.choice(np.where(majority_mask)[0], size=n_majority_target, replace=False)
        keep_idx = np.concatenate([np.where(minority_mask)[0], majority_idx])
        rng.shuffle(keep_idx)
        X_train, y_train = X_train[keep_idx], y_train[keep_idx]
        print(f"Balanced training set: {len(y_train)}  (recipes={int(y_train.sum())}, non-recipes={len(y_train) - int(y_train.sum())})")

    # Train all models
    all_results = {}
    for name in MODELS:
        model = train_model(name, X_train, y_train)
        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (model.predict(X_test) == y_test).mean()
        all_results[name] = (model, train_acc, test_acc)

    print_comparison(all_results, X_test, y_test)

    # Pick model to save
    if args.model:
        best_name = args.model
    else:
        best_name = max(all_results, key=lambda k: all_results[k][2])

    best_model = all_results[best_name][0]
    print(f"\nSaving best model: {best_name}")
    print_feature_report(best_model, best_name)

    filepath = Path(args.output) if args.output else next_model_path()
    MODEL_DIR.mkdir(exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_type": best_name,
            "feature_names": FEATURE_NAMES,
            "trained_at": datetime.now().isoformat(),
            "logfiles": args.logfiles,
            "n_samples": len(y),
        }, f)

    print(f"Model saved to: {filepath}")


if __name__ == "__main__":
    main()
