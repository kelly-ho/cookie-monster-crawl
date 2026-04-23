"""
Label filtered URLs by fetching them and checking for recipe schema.

Reads crawl logs, finds URLs that were filtered (discovered but never visited),
fetches a sample of them, and writes labeled events back to a new JSONL log
that can be used for training.

Usage:
    python -m cookie_monster_crawl.label_filtered crawl_logs/crawl_*.jsonl --max-fetch 500
    python -m cookie_monster_crawl.label_filtered crawl_logs/crawl_2026-04-21_09-41-28.jsonl --max-fetch 200
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from cookie_monster_crawl.parser import get_recipe_data
from cookie_monster_crawl.replay import load_events, reconstruct

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CookieMonsterCrawler/1.0; +recipe-research)"
}


async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetch a URL, return HTML or None."""
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status == 200 and "text/html" in response.headers.get("Content-Type", ""):
                return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass
    return None


async def label_urls(urls_with_features: list[dict], max_fetch: int, concurrency: int) -> list[dict]:
    """Fetch URLs and label them as recipe or non-recipe."""
    sample = random.sample(urls_with_features, min(max_fetch, len(urls_with_features)))
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process(item):
        url = item["url"]
        async with semaphore:
            # Rate limit per domain
            await asyncio.sleep(0.5)
            html = await fetch(session, url)

        if html is None:
            return

        recipe = get_recipe_data(html, url)
        results.append({
            "url": url,
            "is_recipe": recipe is not None,
            "raw_features": item["raw_features"],
            "anchor_text": item.get("anchor_text", ""),
            "source": "label_filtered",
        })

    async with aiohttp.ClientSession() as session:
        tasks = [process(item) for item in sample]
        total = len(tasks)
        done = 0
        for batch_start in range(0, total, concurrency):
            batch = tasks[batch_start:batch_start + concurrency]
            await asyncio.gather(*batch)
            done += len(batch)
            recipes = sum(1 for r in results if r["is_recipe"])
            print(f"  Progress: {done}/{total} fetched, {len(results)} labeled ({recipes} recipes, {len(results) - recipes} non-recipes)", file=sys.stderr)

    return results


def collect_filtered_urls(logfiles: list[str]) -> list[dict]:
    """Collect filtered URLs with their features from crawl logs."""
    filtered = []
    seen = set()

    for logfile in logfiles:
        events = load_events(logfile)
        lifecycles = reconstruct(events)

        for lc in lifecycles.values():
            if lc.filtered and not lc.visited and lc.raw_features and lc.url not in seen:
                seen.add(lc.url)
                filtered.append({
                    "url": lc.url,
                    "raw_features": lc.raw_features,
                    "anchor_text": lc.anchor_text or "",
                    "filter_reason": lc.filter_reason,
                    "filter_score": lc.filter_score,
                })

    return filtered


def write_training_log(results: list[dict], output_path: str):
    """Write labeled results as a JSONL log compatible with the training pipeline."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            # Write discover event with features
            discover = {
                "type": "discover",
                "event_id": i * 3,
                "url": r["url"],
                "score": 0.5,
                "raw_features": r["raw_features"],
                "anchor_text": r["anchor_text"],
                "score_components": {},
            }
            f.write(json.dumps(discover) + "\n")

            # Write visit event
            visit = {
                "type": "visit",
                "event_id": i * 3 + 1,
                "url": r["url"],
                "priority_at_fetch": 0.5,
            }
            f.write(json.dumps(visit) + "\n")

            # Write result event
            result = {
                "type": "result",
                "event_id": i * 3 + 2,
                "url": r["url"],
                "is_recipe": r["is_recipe"],
                "recipe_title": None,
                "links_discovered": 0,
            }
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Label filtered URLs for training data")
    parser.add_argument("logfiles", nargs="+", help="Crawl log JSONL files")
    parser.add_argument("--max-fetch", type=int, default=500, help="Max URLs to fetch (default: 500)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent fetches (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Collecting filtered URLs from logs...", file=sys.stderr)
    filtered = collect_filtered_urls(args.logfiles)
    print(f"Found {len(filtered)} unique filtered URLs", file=sys.stderr)

    if not filtered:
        print("No filtered URLs found.", file=sys.stderr)
        return

    # Show filter reason breakdown
    reasons = {}
    for f in filtered:
        r = f.get("filter_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    print("Filter reasons:", file=sys.stderr)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}", file=sys.stderr)

    print(f"\nFetching up to {args.max_fetch} URLs...", file=sys.stderr)
    start = time.time()
    results = asyncio.run(label_urls(filtered, args.max_fetch, args.concurrency))
    elapsed = time.time() - start

    recipes = sum(1 for r in results if r["is_recipe"])
    non_recipes = len(results) - recipes
    print(f"\nLabeled {len(results)} URLs in {elapsed:.1f}s", file=sys.stderr)
    print(f"  Recipes: {recipes}", file=sys.stderr)
    print(f"  Non-recipes: {non_recipes}", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"crawl_logs/labeled_{timestamp}.jsonl"
    write_training_log(results, output_path)
    print(f"\nTraining log written to: {output_path}", file=sys.stderr)
    print("Include this log in your training command alongside regular crawl logs.", file=sys.stderr)


if __name__ == "__main__":
    main()
