import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CrawlLogger:
    """
    Records crawl events to a JSONL file for offline replay + analysis
        - "discover": a new URL was found and scored
        - "visit": a URL was fetched
        - "result": whether the fetched page was a recipe
        - "filter": a URL was filtered out (score too high, robots blocked, etc.)
    """

    def __init__(self, output_dir: str = "crawl_logs"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(output_dir, f"crawl_{timestamp}.jsonl")
        self._file = open(self.filepath, "w", encoding="utf-8")
        self._event_count = 0
        self._start_time = time.time()
        logger.info(f"Crawl logger writing to {self.filepath}")

    def _write(self, event: dict):
        event["event_id"] = self._event_count
        event["elapsed_secs"] = round(time.time() - self._start_time, 4)
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()
        self._event_count += 1

    def log_seed(self, url: str):
        """Log a seed URL at crawl start."""
        self._write({
            "type": "seed",
            "url": url,
        })

    def log_discover(
        self,
        url: str,
        source_url: str,
        anchor_text: str,
        score: float,
        domain_counts: Dict[str, int],
        score_components: Optional[Dict[str, float]] = None,
        raw_features: Optional[Dict] = None,
        explore: bool = False,
    ):
        """Log when a new URL is discovered from a page."""
        self._write({
            "type": "discover",
            "url": url,
            "source_url": source_url,
            "anchor_text": anchor_text,
            "score": round(score, 6),
            "score_components": score_components or {},
            "raw_features": raw_features or {},
            "domain_counts_snapshot": dict(domain_counts) if domain_counts else {},
            "explore": explore,
        })

    def log_visit(self, url: str, priority: float, pages_fetched: int):
        """Log when a URL is about to be fetched."""
        self._write({
            "type": "visit",
            "url": url,
            "priority_at_fetch": round(priority, 6),
            "pages_fetched_so_far": pages_fetched,
        })

    def log_result(
        self,
        url: str,
        is_recipe: bool,
        recipe_title: Optional[str],
        links_found: int,
        cumulative_recipes: int,
        cumulative_pages: int,
    ):
        """Log the outcome of visiting a URL."""
        self._write({
            "type": "result",
            "url": url,
            "is_recipe": is_recipe,
            "recipe_title": recipe_title,
            "links_discovered": links_found,
            "cumulative_recipes": cumulative_recipes,
            "cumulative_pages": cumulative_pages,
            "harvest_efficiency": round(
                cumulative_recipes / cumulative_pages * 100, 2
            ) if cumulative_pages > 0 else 0,
        })

    def log_filter(self, url: str, reason: str, score: Optional[float] = None):
        """Log when a URL is filtered out."""
        event = {
            "type": "filter",
            "url": url,
            "reason": reason,
        }
        if score is not None:
            event["score"] = round(score, 6)
        self._write(event)

    def log_rescore(self, url: str, old_priority: float, new_priority: float):
        """Log when a URL is rescored and requeued."""
        self._write({
            "type": "rescore",
            "url": url,
            "old_priority": round(old_priority, 6),
            "new_priority": round(new_priority, 6),
        })

    def close(self):
        self._file.close()
        logger.info(
            f"Crawl log complete: {self._event_count} events written to {self.filepath}"
        )