import asyncio
import aiohttp
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, List, Set, Dict
from collections import defaultdict
from cookie_monster_crawl.parser import get_links, get_recipe_data, get_base_domain
from cookie_monster_crawl.utils import RobotsChecker, URLPrioritizer, DEFAULT_SCORING
from cookie_monster_crawl.priority_queue import AsyncPriorityQueue
from cookie_monster_crawl.crawl_logger import CrawlLogger
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {                                                     
    "User-Agent": "CookieMonsterCrawler/0.1 (+https://github.com/kelly-ho/cookie-monster-crawler)"                  
} 

def load_crawl_config(filepath: str = None) -> dict:
    """Load crawl_config.json, falling back to DEFAULT_SCORING if not found."""
    if filepath is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, "data", "crawl_config.json")
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {filepath}, using defaults")
        return {"scoring": DEFAULT_SCORING}


class Crawler:
    def __init__(
        self,
        start_urls: List[str]=None,
        max_pages: int = 100,
        concurrency: int = 5,
        delay_secs: float = 1.0,
        timeout_secs = 15,
        infrastructure_file='infrastructure_segments.txt',
        navigational_file='navigational_segments.txt',
        recipe_related_file='recipe_related_segments.txt',
        max_score_threshold: float = None,
        enable_logging=True,
        crawl_config: dict = None,
    ):
        self.start_urls = start_urls if start_urls is not None else []
        self.concurrency = concurrency
        self.delay_secs = delay_secs
        self.timeout_secs = timeout_secs
        self.max_pages = max_pages
        self.robots_checker = RobotsChecker(headers=HEADERS)

        scoring = (crawl_config or {}).get("scoring", {})
        self.url_prioritizer = URLPrioritizer(
            infrastructure_file=infrastructure_file,
            navigational_file=navigational_file,
            recipe_related_file=recipe_related_file,
            scoring_config=scoring,
            max_score_threshold=max_score_threshold,
        )

        self.queue = AsyncPriorityQueue()
        self.visited: Set[str] = set()
        self.queued: Set[str] = set()
        self.recipes: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.blocked_domains: Set[str] = set()
        self.stop_signal = asyncio.Event()
        self.domain_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        self.pages_fetched = 0
        self.seed_file_name = None
        self.latencies: List[float] = []
        self.domain_stats: Dict[str, int] = defaultdict(int)
        self.crawl_start_time: Optional[float] = None
        self.crawl_end_time: Optional[float] = None
        self.crawl_log: Optional[CrawlLogger] = CrawlLogger() if enable_logging else None

    async def fetch(self, session: aiohttp.ClientSession, url: str, retry_count: int = 0, max_retries: int = 3) -> Optional[str]:
        '''Fetch HTML content from a URL asynchronously with domain-level locking and exponential backoff for retryable errors.'''
        domain = get_base_domain(url)
        
        async with self.domain_locks[domain]:
            if not await self.robots_checker.is_allowed(url):
                logger.info(f"Blocked by robots.txt: {url}")
                self.blocked_domains.add(domain)
                return None
            
            crawl_delay = await self.robots_checker.get_crawl_delay(domain)
            sleep_time = max(self.delay_secs, crawl_delay)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            start_time = time.time()
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=self.timeout_secs)) as response:
                    if response.status == 200 and "text/html" in response.headers.get("Content-Type", ""):
                        html = await response.text()
                        latency = time.time() - start_time
                        self.latencies.append(latency)
                        self.domain_stats[domain] += 1
                        return html
                    elif response.status in {429, 500, 502, 503, 504} and retry_count < max_retries:
                        backoff_time = (2 ** retry_count) * 2
                        logger.warning(f"Status {response.status} for {url}, retrying in {backoff_time}s (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(backoff_time)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                return None
        
        if retry_count < max_retries:
            return await self.fetch(session, url, retry_count + 1, max_retries)
        
        return None
            
    async def worker(self):
        while True:
            try:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.timeout_secs
                )
            except asyncio.TimeoutError:
                if self.stop_signal.is_set() or self.queue.empty():
                    logger.info("Worker exiting: queue idle timeout reached")
                    return
                continue

            try:
                priority, (url, anchor_text) = item
                if url is None:
                    return
                
                if self.stop_signal.is_set():
                    break

                if url in self.visited:
                    continue
                
                domain = get_base_domain(url)
                if self.domain_locks[domain].locked():
                    await self.queue.put((priority + self.url_prioritizer.lock_penalty, (url, anchor_text)))
                    continue
                
                is_seed = url in self.start_urls
                if not is_seed:
                    new_priority, _ = self.url_prioritizer.calculate_score(url, self.domain_stats, anchor_text)
                    if new_priority > (priority + self.url_prioritizer.rescore_sensitivity):
                        logger.info(f"Requeuing {url}: priority worsened from {priority:.3f} to {new_priority:.3f}")
                        if self.crawl_log:
                            self.crawl_log.log_rescore(url, priority, new_priority)
                        await self.queue.put((new_priority, (url, anchor_text)))
                        continue

                logger.info(f"Fetching: {url}")
                self.visited.add(url)

                if not is_seed:
                    self.pages_fetched += 1
                    if self.crawl_log:
                        self.crawl_log.log_visit(url, priority, self.pages_fetched)

                html = await self.fetch(self.session, url)
                if not html:
                    continue

                recipe = get_recipe_data(html, url)
                links = get_links(html, url)
                if recipe:
                    logger.info(f"Found recipe: {recipe['title']}")
                    self.recipes.append(recipe)
                    self.url_prioritizer.update_model(url, is_recipe=True)
                else:
                    self.url_prioritizer.update_model(url, is_recipe=False)

                if self.crawl_log and url not in self.start_urls:
                    self.crawl_log.log_result(
                        url=url,
                        is_recipe=bool(recipe),
                        recipe_title=recipe["title"] if recipe else None,
                        links_found=len(links),
                        cumulative_recipes=len(self.recipes),
                        cumulative_pages=self.pages_fetched,
                    )

                # Check if we've hit page limit (excluding seed URLs)
                if self.pages_fetched >= self.max_pages:
                    logger.info(f"Reached max_pages limit of {self.max_pages}")
                    self.stop_signal.set()
                    break

                for link, anchor_text in links.items():
                    if link not in self.queued:
                        if await self.robots_checker.is_allowed(link):
                            priority_score, score_components = self.url_prioritizer.calculate_score(link, self.domain_stats, anchor_text)
                            if priority_score <= self.url_prioritizer.max_score_threshold:
                                logger.debug(f"Queueing link: {link} with priority {priority_score:.3f}")
                                await self.queue.put((priority_score, (link, anchor_text)))
                                self.queued.add(link)
                                if self.crawl_log:
                                    self.crawl_log.log_discover(link, url, anchor_text, priority_score, self.domain_stats, score_components)
                            else:
                                logger.debug(f"Filtered: {link} (score {priority_score:.3f} > threshold {self.url_prioritizer.max_score_threshold})")
                                if self.crawl_log:
                                    self.crawl_log.log_filter(link, "score_threshold", priority_score)
                        else:
                            domain = get_base_domain(link)
                            self.blocked_domains.add(domain)
                            if self.crawl_log:
                                self.crawl_log.log_filter(link, "robots_blocked")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
            finally:
                self.queue.task_done()


    async def crawl(self):
        self.crawl_start_time = time.time()
        for url in self.start_urls:
            self.queue.put_nowait((-float('inf'), (url, "")))
            self.queued.add(url)
            if self.crawl_log:
                self.crawl_log.log_seed(url)
        async with aiohttp.ClientSession() as session:
            self.session = session
            workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
            # Wait for completion or cancellation due to limits
            await asyncio.gather(*workers, return_exceptions=True)
        self.crawl_end_time = time.time()
        if self.crawl_log:
            self.crawl_log.close()
        self.save_results()
        self.generate_report()

    def save_results(self):
        output_file = "recipes.json"
        with open(output_file, "w") as f:
            json.dump(self.recipes, f, indent=2)

        logger.info(f"Recipes saved to {output_file}")
        logger.info(f"\nDone. Visited {len(self.visited)} pages, found {len(self.recipes)} recipes.")
        
        recipe_urls = {recipe['url'] for recipe in self.recipes}
        seed_urls = set(self.start_urls)
        non_recipe_urls = sorted(self.visited - recipe_urls - seed_urls)
        
        if non_recipe_urls:
            logger.info(f"\n{'='*60}")
            logger.info(f"Non-recipe URLs visited ({len(non_recipe_urls)} total):")
            logger.info(f"{'='*60}")
            for url in non_recipe_urls:
                logger.info(f"  - {url}")
            logger.info(f"{'='*60}")
        
        if self.blocked_domains:
            logger.info(f"\n{'='*60}")
            logger.info(f"Websites blocked by robots.txt ({len(self.blocked_domains)} total):")
            logger.info(f"{'='*60}")
            for domain in sorted(self.blocked_domains):
                logger.info(f"  - {domain}")
            logger.info(f"{'='*60}")
        else:
            logger.info("\nNo websites were blocked by robots.txt.")

    def generate_report(self):
        """Generate a JSON report with crawl statistics."""
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if self.seed_file_name:
            report_file = f"results/run_{self.seed_file_name}_{timestamp}.json"
        else:
            report_file = f"results/run_{timestamp}.json"
        
        total_fetched = self.pages_fetched
        recipe_count = len(self.recipes)
        harvest_efficiency = (recipe_count / total_fetched * 100) if total_fetched > 0 else 0
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        total_duration = self.crawl_end_time - self.crawl_start_time if self.crawl_start_time and self.crawl_end_time else 0
        
        report = {
            "timestamp": timestamp,
            "total_fetched": total_fetched,
            "recipes_found": recipe_count,
            "harvest_efficiency_percent": round(harvest_efficiency, 2),
            "domain_breakdown": dict(self.domain_stats),
            "avg_latency_seconds": round(avg_latency, 3),
            "total_duration_seconds": round(total_duration, 2),
            "pages_per_second": round(total_fetched / total_duration, 2) if total_duration > 0 else 0,
            "blocked_domains": list(self.blocked_domains),
            "config": {
                "max_pages": self.max_pages,
                "concurrency": self.concurrency,
                "delay_secs": self.delay_secs,
                "timeout_secs": self.timeout_secs
            }
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CRAWL REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Total pages fetched: {total_fetched}")
        logger.info(f"Recipes found: {recipe_count}")
        logger.info(f"Harvest efficiency: {harvest_efficiency:.2f}%")
        logger.info(f"Average latency: {avg_latency:.3f}s")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info(f"Pages per second: {report['pages_per_second']}")
        logger.info(f"{'='*60}")

    def load_seeds(self, filepath=None):
        """Load seed URLs from JSON file. Path is relative to project root."""
        if filepath is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, "data", "static-target.json")
        
        # Extract seed file name (without extension) for report naming
        self.seed_file_name = os.path.splitext(os.path.basename(filepath))[0]
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.start_urls = data.get("seeds", [])
                logger.info(f"Loaded {len(self.start_urls)} seed URLs from {filepath}")
        except FileNotFoundError:
            logger.error(f"Error: {filepath} not found.")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error: Failed to decode JSON from {filepath}.")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cookie Monster Recipe Crawler")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl (default: 100)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers (default: 5)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--timeout", type=float, default=15, help="Request timeout in seconds (default: 15)")
    parser.add_argument("--max-score", type=float, default=0.80, help="Max score threshold for URL filtering (default: 0.80)")
    parser.add_argument("--seeds", type=str, default=None, help="Path to seed URLs JSON file (default: data/static-target.json)")
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL crawl event logging")
    parser.add_argument("--config", type=str, default=None, help="Path to crawl_config.json (default: data/crawl_config.json)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_path = os.path.join("logs", log_name)
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging_config = dict(level=logging.INFO, format=fmt, handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ])
    logging_config["force"] = True
    logging.basicConfig(**logging_config)

    crawl_config = load_crawl_config(args.config)

    cookie_monster = Crawler(
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        delay_secs=args.delay,
        timeout_secs=args.timeout,
        max_score_threshold=args.max_score if args.max_score != 0.80 else None,
        enable_logging=not args.no_log,
        crawl_config=crawl_config,
    )
    cookie_monster.load_seeds(args.seeds)
    asyncio.run(cookie_monster.crawl())
