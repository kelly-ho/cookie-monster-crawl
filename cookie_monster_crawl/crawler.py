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
from cookie_monster_crawl.utils import RobotsChecker, URLPrioritizer
from cookie_monster_crawl.priority_queue import AsyncPriorityQueue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "CookieMonsterCrawler/0.1 (+https://github.com/kelly-ho/cookie-monster-crawler)"
}

class Crawler:
    def __init__(self, start_urls: List[str]=[], max_pages: int = 100, concurrency: int = 5, delay_secs: float = 1.0, timeout_secs = 15):
        self.start_urls = start_urls
        self.concurrency = concurrency
        self.delay_secs = delay_secs
        self.timeout_secs = timeout_secs
        self.max_pages = max_pages
        self.robots_checker = RobotsChecker(user_agent=HEADERS["User-Agent"])
        self.url_prioritizer = URLPrioritizer()

        self.queue = AsyncPriorityQueue()
        self.visited: Set[str] = set()
        self.queued: Set[str] = set()
        self.recipes: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.blocked_domains: Set[str] = set()
        self.stop_signal = asyncio.Event()
        
        # Metrics tracking
        self.latencies: List[float] = []
        self.domain_stats: Dict[str, int] = defaultdict(int)
        self.crawl_start_time: Optional[float] = None
        self.crawl_end_time: Optional[float] = None

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        '''Fetch HTML content from a URL asynchronously.'''
        if not await self.robots_checker.is_allowed(url):
            logger.info(f"Blocked by robots.txt: {url}")
            domain = get_base_domain(url)
            self.blocked_domains.add(domain)
            return None
        domain = get_base_domain(url)
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
                logger.warning(f"Unexpected status {response.status} for {url}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        return None
            
    async def worker(self):
        while True:
            try:
                item = await self.queue.get()
                _, url = item
                if url is None:
                    return
                
                if self.stop_signal.is_set():
                    self.queue.task_done()
                    break

                if url in self.visited:
                    self.queue.task_done()
                    continue
                
                logger.info(f"Fetching: {url}")
                self.visited.add(url)
                
                html = await self.fetch(self.session, url)
                if not html:
                    continue
                recipe = get_recipe_data(html, url)
                if recipe:
                    logger.info(f"Found recipe: {recipe['title']}")
                    self.recipes.append(recipe)

                if len(self.visited) >= self.max_pages:
                    logger.info(f"Reached max_pages limit of {self.max_pages}")
                    self.stop_signal.set()
                    self.queue.task_done()
                    break
                
                links = get_links(html, url)
                for link in links:
                    if link not in self.queued:
                        if await self.robots_checker.is_allowed(link):
                            priority_score = self.url_prioritizer.calculate_score(link)
                            await self.queue.put((priority_score, link))
                            self.queued.add(link)
                        else:
                            domain = get_base_domain(link)
                            self.blocked_domains.add(domain)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
            finally:
                self.queue.task_done()

    async def crawl(self):
        self.crawl_start_time = time.time()
        for url in self.start_urls:
            self.queue.put_nowait((-float('inf'), url))
            self.queued.add(url)
        async with aiohttp.ClientSession() as session:
            self.session = session
            workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
            # Wait for completion or cancellation due to limits
            await asyncio.gather(*workers, return_exceptions=True)
        self.crawl_end_time = time.time()
        self.save_results()
        self.generate_report()

    def save_results(self):
        output_file = "recipes.json"
        with open(output_file, "w") as f:
            json.dump(self.recipes, f, indent=2)

        logger.info(f"Recipes saved to {output_file}")
        logger.info(f"\nDone. Visited {len(self.visited)} pages, found {len(self.recipes)} recipes.")
        
        # List non-recipe URLs
        recipe_urls = {recipe['url'] for recipe in self.recipes}
        non_recipe_urls = sorted(self.visited - recipe_urls)
        
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
        
        total_fetched = len([url for url in self.visited if any(url in recipe['url'] for recipe in self.recipes) or url in self.visited])
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
        
        report_file = f"results/run_{timestamp}.json"
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
            filepath = os.path.join(project_root, "data", "seed.json")
        
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
    os.makedirs("logs", exist_ok=True)
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_path = os.path.join("logs", log_name)
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    config = dict(level=logging.INFO, format=fmt, handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ])
    config["force"] = True
    logging.basicConfig(**config)
    cookie_monster = Crawler()
    cookie_monster.load_seeds()
    asyncio.run(cookie_monster.crawl())
