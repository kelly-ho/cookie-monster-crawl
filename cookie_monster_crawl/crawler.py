import asyncio
import aiohttp
import json
import logging
from typing import Optional, List, Set, Dict
from .parser import get_links, get_recipe_data, get_base_domain
from .utils import RobotsChecker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "CookieMonsterCrawler/0.1 (+https://github.com/kelly-ho/cookie-monster-crawler)"
}

class Crawler:
    def __init__(self, start_urls: List[str], max_pages: int = 100, concurrency: int = 5, delay_secs: float = 1.0, timeout_secs = 15):
        self.start_urls = start_urls
        self.concurrency = concurrency
        self.delay_secs = delay_secs
        self.timeout_secs = timeout_secs
        self.max_pages = max_pages
        self.robots_checker = RobotsChecker(user_agent=HEADERS["User-Agent"])

        self.queue = asyncio.Queue()
        self.visited: Set[str] = set()
        self.recipes: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        '''Fetch HTML content from a URL asynchronously.'''
        if not self.robots_checker.is_allowed(url):
            logger.info(f"Blocked by robots.txt: {url}")
            return None
        domain = get_base_domain(url)
        crawl_delay = self.robots_checker.get_crawl_delay(domain)
        sleep_time = max(self.delay_secs, crawl_delay)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=self.timeout_secs)) as response:
                if response.status == 200 and "text/html" in response.headers.get("Content-Type", ""):
                    return await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        return None
            
    async def worker(self):
        while True:
            try:
                url = await self.queue.get()
                if url is None: 
                    return
                logger.info(f"Fetching: {url}")
                html = await self.fetch(self.session, url)
                if not html:
                    self.queue.task_done()
                    continue
                recipe = get_recipe_data(html, url)
                if recipe:
                    logger.info(f"Found recipe: {recipe['title']}")
                    self.recipes.append(recipe)
                links = get_links(html, url)
                for link in links:
                    if link not in self.visited and len(self.visited) < self.max_pages and self.robots_checker.is_allowed(link):
                        await self.queue.put(link)
                        self.visited.add(link)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
            finally:
                self.queue.task_done()

    async def crawl(self):
        for url in self.start_urls:
            if len(self.visited) < self.max_pages:
                self.queue.put_nowait(url)
                self.visited.add(url)
        async with aiohttp.ClientSession() as session:
            self.session = session
            workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
            await self.queue.join()
            for _ in range(self.concurrency):
                await self.queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)
        self.save_results()

    def save_results(self):
        output_file = "recipes.json"
        with open(output_file, "w") as f:
            json.dump(self.recipes, f, indent=2)

        logger.info(f"\nDone. Visited {len(self.visited)} pages, found {len(self.recipes)} recipes.")
        logger.info(f"Recipes saved to {output_file}")


if __name__ == "__main__":
    START_URLS = [
        "https://www.americastestkitchen.com/", 
        "https://www.seriouseats.com/"
    ]
    cookie_monster = Crawler(start_urls=START_URLS)
    asyncio.run(cookie_monster.crawl())
