import asyncio
import aiohttp
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional, List, Set, Dict
from cookie_monster_crawl.parser import get_links, get_recipe_data, get_base_domain
from cookie_monster_crawl.utils import RobotsChecker, score_page
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

        self.queue = AsyncPriorityQueue()
        self.visited: Set[str] = set()
        self.recipes: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.blocked_domains: Set[str] = set()

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
        
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=self.timeout_secs)) as response:
                if response.status == 200 and "text/html" in response.headers.get("Content-Type", ""):
                    return await response.text()
                logger.warning(f"Unexpected status {response.status} for {url}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        return None
            
    async def worker(self):
        while True:
            try:
                item = await self.queue.get()
                _, url_object = item
                if url_object is None:
                    return
                url, parent_relevance = url_object
                logger.info(f"Fetching: {url}")
                html = await self.fetch(self.session, url)
                if not html:
                    continue
                relevance = score_page(html)
                recipe = get_recipe_data(html, url)
                if recipe:
                    logger.info(f"Found recipe: {recipe['title']}")
                    self.recipes.append(recipe)
                links = get_links(html, url)
                for link in links:
                    if link not in self.visited and len(self.visited) < self.max_pages:
                        if await self.robots_checker.is_allowed(link):
                            link_priority = (0.7 * relevance + 0.3 * parent_relevance)
                            await self.queue.put((-link_priority, (link, relevance)))
                            self.visited.add(link)
                        else:
                            domain = get_base_domain(link)
                            self.blocked_domains.add(domain)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
            finally:
                self.queue.task_done()

    async def crawl(self):
        for url in self.start_urls:
            if len(self.visited) < self.max_pages:
                self.queue.put_nowait((-1.0, (url, 1.0)))
                self.visited.add(url)
        async with aiohttp.ClientSession() as session:
            self.session = session
            workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
            await self.queue.join()
            for _ in range(self.concurrency):
                await self.queue.put((float("inf"), None))
            await asyncio.gather(*workers, return_exceptions=True)
        self.save_results()

    def save_results(self):
        output_file = "recipes.json"
        with open(output_file, "w") as f:
            json.dump(self.recipes, f, indent=2)

        logger.info(f"Recipes saved to {output_file}")
        logger.info(f"\nDone. Visited {len(self.visited)} pages, found {len(self.recipes)} recipes.")
        
        if self.blocked_domains:
            logger.info(f"\n{'='*60}")
            logger.info(f"Websites blocked by robots.txt ({len(self.blocked_domains)} total):")
            logger.info(f"{'='*60}")
            for domain in sorted(self.blocked_domains):
                logger.info(f"  - {domain}")
            logger.info(f"{'='*60}")
        else:
            logger.info("\nNo websites were blocked by robots.txt.")


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
