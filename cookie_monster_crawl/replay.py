"""
Replay script for cookie-monster-crawl JSONL logs.

Reconstructs a crawl run from its event log and produces a structured
analysis suitable for human review or LLM strategy generation.

Usage:
    python -m cookie_monster_crawl.replay <logfile> [options]
"""

import argparse
import json
import math
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class URLLifecycle:
    url: str
    domain: str
    is_seed: bool = False

    # From discover event
    discovered_score: Optional[float] = None
    score_components: Dict[str, float] = field(default_factory=dict)
    raw_features: Dict[str, float] = field(default_factory=dict)
    source_url: Optional[str] = None
    anchor_text: Optional[str] = None

    # From visit event
    visited: bool = False
    visit_priority: Optional[float] = None

    # From result event
    is_recipe: Optional[bool] = None
    recipe_title: Optional[str] = None
    links_found: Optional[int] = None

    # From filter event
    filtered: bool = False
    filter_reason: Optional[str] = None
    filter_score: Optional[float] = None

    # From rescore events
    rescores: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1: Load
# ---------------------------------------------------------------------------

def load_events(logfile: str) -> List[dict]:
    events = []
    with open(logfile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return sorted(events, key=lambda e: e.get("event_id", 0))


# ---------------------------------------------------------------------------
# Phase 2: Reconstruct URL lifecycles
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    return urlparse(url).netloc


def reconstruct(events: List[dict]) -> Dict[str, URLLifecycle]:
    lifecycles: Dict[str, URLLifecycle] = {}

    def get(url: str) -> URLLifecycle:
        if url not in lifecycles:
            lifecycles[url] = URLLifecycle(url=url, domain=_domain(url))
        return lifecycles[url]

    for e in events:
        t = e.get("type")
        url = e.get("url")
        if not url:
            continue

        lc = get(url)

        if t == "seed":
            lc.is_seed = True
            lc.visited = True

        elif t == "discover":
            lc.discovered_score = e.get("score")
            lc.score_components = e.get("score_components", {})
            lc.raw_features = e.get("raw_features", {})
            lc.source_url = e.get("source_url")
            lc.anchor_text = e.get("anchor_text")

        elif t == "visit":
            lc.visited = True
            lc.visit_priority = e.get("priority_at_fetch")

        elif t == "result":
            lc.is_recipe = e.get("is_recipe")
            lc.recipe_title = e.get("recipe_title")
            lc.links_found = e.get("links_discovered")

        elif t == "filter":
            lc.filtered = True
            lc.filter_reason = e.get("reason")
            lc.filter_score = e.get("score")

        elif t == "rescore":
            lc.rescores.append({
                "old": e.get("old_priority"),
                "new": e.get("new_priority"),
            })

    return lifecycles


# ---------------------------------------------------------------------------
# Phase 2b: Re-score with a candidate model
# ---------------------------------------------------------------------------

def rescore_with_model(lifecycles: Dict[str, URLLifecycle], model_path: str):
    """Replace discovered_score with predictions from a candidate model."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    feature_names = data["feature_names"]

    for lc in lifecycles.values():
        if not lc.raw_features:
            continue
        vec = np.array([[lc.raw_features.get(f, 0.0) for f in feature_names]])
        lc.discovered_score = float(model.predict_proba(vec)[0][0])

    return data.get("model_type", "unknown")


def simulate_harvest(lifecycles: Dict[str, URLLifecycle], top_n: int) -> dict:
    """Simulate priority queue: take top N URLs by lowest score, report harvest."""
    candidates = [lc for lc in lifecycles.values() if lc.discovered_score is not None and lc.is_recipe is not None and not lc.is_seed]
    ranked = sorted(candidates, key=lambda lc: lc.discovered_score)
    selected = ranked[:top_n]

    recipes = sum(1 for lc in selected if lc.is_recipe)
    non_recipes = len(selected) - recipes

    false_positives = [(lc.url, round(lc.discovered_score, 4)) for lc in selected if not lc.is_recipe]
    missed = sum(1 for lc in ranked[top_n:] if lc.is_recipe)

    return {
        "top_n": len(selected),
        "harvest_pct": round(recipes / len(selected) * 100, 1) if selected else 0,
        "recipes": recipes,
        "non_recipes": non_recipes,
        "missed_recipes": missed,
        "total_candidates": len(candidates),
        "false_positives": false_positives,
    }


# ---------------------------------------------------------------------------
# Phase 3: Analyze
# ---------------------------------------------------------------------------

def _band_label(score: float, bands: List[float]) -> str:
    for i in range(len(bands) - 1):
        if score < bands[i + 1]:
            return f"{bands[i]:.1f}–{bands[i+1]:.1f}"
    return f"{bands[-1]:.1f}+"


def analyze(
    lifecycles: Dict[str, URLLifecycle],
    events: List[dict],
    logfile: str,
    score_bands: List[float],
    filter_sample_fraction: float,
    domain_filter: Optional[str],
) -> dict:
    lcs = list(lifecycles.values())
    if domain_filter:
        lcs = [lc for lc in lcs if domain_filter in lc.domain]

    seeds = [lc for lc in lcs if lc.is_seed]
    visited = [lc for lc in lcs if lc.visited and not lc.is_seed]
    filtered = [lc for lc in lcs if lc.filtered]
    discovered = [lc for lc in lcs if lc.discovered_score is not None]
    rescored = [lc for lc in lcs if lc.rescores]

    total_pages = len(visited)
    filter_sample_size = max(1, math.ceil(filter_sample_fraction * total_pages)) if total_pages else 1

    # --- Domain breakdown ---
    domain_stats: Dict[str, dict] = defaultdict(lambda: {
        "pages": 0, "recipes": 0, "filtered": 0,
    })
    for lc in visited:
        domain_stats[lc.domain]["pages"] += 1
        if lc.is_recipe:
            domain_stats[lc.domain]["recipes"] += 1
    for lc in filtered:
        domain_stats[lc.domain]["filtered"] += 1

    domain_breakdown = {}
    for dom, s in domain_stats.items():
        pages = s["pages"]
        recipes = s["recipes"]
        domain_breakdown[dom] = {
            "pages_visited": pages,
            "recipes_found": recipes,
            "urls_filtered": s["filtered"],
            "harvest_rate_pct": round(recipes / pages * 100, 1) if pages else 0,
        }

    # --- Score calibration ---
    # Only non-seed discovered URLs that were subsequently visited
    calibration_buckets: Dict[str, dict] = {}
    for label in [_band_label(score_bands[i], score_bands) for i in range(len(score_bands) - 1)] + [f"{score_bands[-1]:.1f}+"]:
        calibration_buckets[label] = {"discovered": 0, "visited": 0, "recipes": 0}

    for lc in discovered:
        if lc.discovered_score is None:
            continue
        label = _band_label(lc.discovered_score, score_bands)
        b = calibration_buckets[label]
        b["discovered"] += 1
        if lc.visited:
            b["visited"] += 1
            if lc.is_recipe:
                b["recipes"] += 1

    score_calibration = []
    for label, b in calibration_buckets.items():
        if b["discovered"] == 0:
            continue
        visited_count = b["visited"]
        score_calibration.append({
            "score_band": label,
            "discovered": b["discovered"],
            "visited": visited_count,
            "recipes": b["recipes"],
            "hit_rate_pct": round(b["recipes"] / visited_count * 100, 1) if visited_count else None,
        })

    # --- Filter audit ---
    by_reason: Dict[str, List[URLLifecycle]] = defaultdict(list)
    for lc in filtered:
        by_reason[lc.filter_reason or "unknown"].append(lc)

    filter_audit = {}
    for reason, items in by_reason.items():
        sample = random.sample(items, min(filter_sample_size, len(items)))
        filter_audit[reason] = {
            "count": len(items),
            "sample_urls": [
                {"url": lc.url, "score": lc.filter_score}
                for lc in sample
            ],
        }

    # --- Score component analysis ---
    # Compare avg component values for recipe vs non-recipe visited pages
    component_totals: Dict[str, Dict[str, list]] = defaultdict(lambda: {"recipe": [], "non_recipe": []})
    for lc in visited:
        if not lc.score_components:
            continue
        bucket = "recipe" if lc.is_recipe else "non_recipe"
        for comp, val in lc.score_components.items():
            component_totals[comp][bucket].append(val)

    component_analysis = {}
    for comp, buckets in component_totals.items():
        r = buckets["recipe"]
        nr = buckets["non_recipe"]
        component_analysis[comp] = {
            "avg_when_recipe": round(sum(r) / len(r), 4) if r else None,
            "avg_when_not_recipe": round(sum(nr) / len(nr), 4) if nr else None,
        }

    # --- Rescore summary ---
    all_rescores = [r for lc in rescored for r in lc.rescores]
    deltas = [r["new"] - r["old"] for r in all_rescores if r["new"] is not None and r["old"] is not None]
    rescore_summary = {
        "total_rescores": len(all_rescores),
        "urls_rescored": len(rescored),
        "avg_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
    }

    # --- Run metadata ---
    elapsed = [e.get("elapsed_secs", 0) for e in events if "elapsed_secs" in e]
    total_duration = max(elapsed) if elapsed else 0

    meta = {
        "logfile": str(logfile),
        "total_events": len(events),
        "total_duration_secs": round(total_duration, 2),
        "seeds": [lc.url for lc in seeds],
        "pages_visited": total_pages,
        "recipes_found": sum(1 for lc in visited if lc.is_recipe),
        "urls_filtered": len(filtered),
        "filter_sample_size": filter_sample_size,
    }

    return {
        "meta": meta,
        "domain_breakdown": domain_breakdown,
        "score_calibration": score_calibration,
        "filter_audit": filter_audit,
        "component_analysis": component_analysis,
        "rescore_summary": rescore_summary,
    }


# ---------------------------------------------------------------------------
# Phase 4: Output
# ---------------------------------------------------------------------------

def print_summary(analysis: dict):
    meta = analysis["meta"]
    recipes = meta["recipes_found"]
    pages = meta["pages_visited"]
    efficiency = round(recipes / pages * 100, 1) if pages else 0

    print(f"\n{'='*60}")
    print(f"CRAWL REPLAY  —  {Path(meta['logfile']).name}")
    print(f"{'='*60}")
    print(f"Duration:       {meta['total_duration_secs']}s")
    print(f"Pages visited:  {pages}")
    print(f"Recipes found:  {recipes}  ({efficiency}% efficiency)")
    print(f"URLs filtered:  {meta['urls_filtered']}")
    print(f"Seeds:          {', '.join(meta['seeds']) or '(none recorded)'}")

    print(f"\n{'─'*60}")
    print("DOMAIN BREAKDOWN")
    print(f"{'─'*60}")
    for dom, s in analysis["domain_breakdown"].items():
        print(f"  {dom}")
        print(f"    pages={s['pages_visited']}  recipes={s['recipes_found']}  "
              f"filtered={s['urls_filtered']}  harvest={s['harvest_rate_pct']}%")

    print(f"\n{'─'*60}")
    print("SCORE CALIBRATION  (lower score = higher priority)")
    print(f"{'─'*60}")
    print(f"  {'Band':<12} {'Discovered':>10} {'Visited':>8} {'Recipes':>8} {'Hit rate':>10}")
    for row in analysis["score_calibration"]:
        hit = f"{row['hit_rate_pct']}%" if row['hit_rate_pct'] is not None else "n/a"
        print(f"  {row['score_band']:<12} {row['discovered']:>10} {row['visited']:>8} "
              f"{row['recipes']:>8} {hit:>10}")

    print(f"\n{'─'*60}")
    print("FILTER AUDIT")
    print(f"{'─'*60}")
    for reason, data in analysis["filter_audit"].items():
        print(f"  {reason}: {data['count']} URLs  (sample of {len(data['sample_urls'])})")
        for item in data["sample_urls"]:
            score_str = f"  score={item['score']}" if item["score"] is not None else ""
            print(f"    - {item['url']}{score_str}")

    if analysis["component_analysis"]:
        print(f"\n{'─'*60}")
        print("SCORE COMPONENT ANALYSIS  (avg value by outcome)")
        print(f"{'─'*60}")
        print(f"  {'Component':<14} {'→ recipe':>10} {'→ not recipe':>14}")
        for comp, vals in analysis["component_analysis"].items():
            r = f"{vals['avg_when_recipe']:.4f}" if vals["avg_when_recipe"] is not None else "n/a"
            nr = f"{vals['avg_when_not_recipe']:.4f}" if vals["avg_when_not_recipe"] is not None else "n/a"
            print(f"  {comp:<14} {r:>10} {nr:>14}")

    rs = analysis["rescore_summary"]
    print(f"\n{'─'*60}")
    print("RESCORING")
    print(f"{'─'*60}")
    print(f"  URLs rescored:   {rs['urls_rescored']}")
    print(f"  Total rescores:  {rs['total_rescores']}")
    delta = f"{rs['avg_delta']:+.4f}" if rs["avg_delta"] is not None else "n/a"
    print(f"  Avg delta:       {delta}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay and analyse a cookie-monster-crawl JSONL log")
    parser.add_argument("logfile", help="Path to .jsonl crawl log")
    parser.add_argument("--mode", choices=["summary", "json"], default="summary")
    parser.add_argument("--domain", default=None, help="Filter analysis to a single domain substring")
    parser.add_argument(
        "--score-bands",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 0.7, 0.8],
        metavar="BOUND",
        help="Score band boundaries (default: 0.0 0.3 0.5 0.7 0.8)",
    )
    parser.add_argument(
        "--filter-sample-fraction",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of total pages visited used as filter sample size (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible filter sampling")
    parser.add_argument("--model", default=None, help="Path to model .pkl — re-scores URLs and simulates harvest")
    parser.add_argument("--top-n", type=int, default=1000, help="Pages to simulate fetching when using --model (default: 1000)")
    parser.add_argument("--show-misses", action="store_true", help="Print non-recipe URLs in simulated top N")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    events = load_events(args.logfile)
    lifecycles = reconstruct(events)

    if args.model:
        model_type = rescore_with_model(lifecycles, args.model)
        result = simulate_harvest(lifecycles, args.top_n)

        print(f"\n{'='*50}")
        print(f"SIMULATED HARVEST — {model_type} ({Path(args.model).name})")
        print(f"{'='*50}")
        print(f"Candidates:         {result['total_candidates']}")
        print(f"Simulated fetches:  {result['top_n']}")
        print(f"Harvest efficiency: {result['harvest_pct']}%")
        print(f"Recipes fetched:    {result['recipes']}")
        print(f"Non-recipes:        {result['non_recipes']}")
        print(f"Missed recipes:     {result['missed_recipes']}")
        print(f"{'='*50}")

        if args.show_misses and result["false_positives"]:
            print(f"\nNon-recipe URLs in top {result['top_n']}:")
            for url, score in result["false_positives"]:
                print(f"  {score:.4f}  {url}")
        return

    analysis = analyze(
        lifecycles,
        events,
        logfile=args.logfile,
        score_bands=sorted(args.score_bands),
        filter_sample_fraction=args.filter_sample_fraction,
        domain_filter=args.domain,
    )

    if args.mode == "json":
        json.dump(analysis, sys.stdout, indent=2)
        print()
    else:
        print_summary(analysis)


if __name__ == "__main__":
    main()
