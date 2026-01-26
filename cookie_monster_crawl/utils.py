import logging
import random
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
from cookie_monster_crawl.parser import get_base_domain
import httpx


RECIPE_KEYWORDS = {
    "recipe", "ingredients", "instructions",
    "cook", "bake", "prep", "serves"
}

NON_RECIPE_KEYWORDS = {
    "guide", "review", "blog", "article"
}

def score_page(html: str) -> float:
    text = html.lower()

    recipe_hits = sum(1 for kw in RECIPE_KEYWORDS if kw in text)
    non_recipe_hits = sum(1 for kw in NON_RECIPE_KEYWORDS if kw in text)

    score = recipe_hits - 0.5 * non_recipe_hits  
    return min(1.0, score / 5)


class RobotsChecker:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: Dict[str, Optional[RobotFileParser]] = {}
        # Use browser header for the fetching robots.txt
        self.fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/plain"
        }
        self.logger = logging.getLogger(__name__)

    async def _load_robots_txt(self, domain: str) -> Optional[RobotFileParser]:
        """Asynchronously fetch robots.txt with headers."""
        robots_url = f"https://{domain}/robots.txt"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(robots_url, headers=self.fetch_headers)
                if response.status_code == 404:
                    return None
                # If we get blocked (403), we might want to log it and be conservative
                if response.status_code != 200:
                    self.logger.warning(f"Blocked or error {response.status_code} for {domain}")
                    return None
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
                return parser
            except Exception as e:
                self.logger.error(f"Unable to load robots.txt for {domain}: {e}")
                return None

    async def is_allowed(self, url: str) -> bool:
        """Check permission without blocking the event loop."""
        parsed_url = httpx.URL(url)
        domain = parsed_url.host
        if domain not in self.parsers:
            self.parsers[domain] = await self._load_robots_txt(domain)
        parser = self.parsers[domain]
        if parser is None:
            return True # Default to allowed if no rules found    
        return parser.can_fetch(self.user_agent, url)
    
    async def get_crawl_delay(self, domain: str) -> float:
        '''Get the crawl delay (in seconds) for a domain from robots.txt'''
        if domain not in self.parsers:
            self.parsers[domain] = await self._load_robots_txt(domain)
        parser = self.parsers[domain]
        if parser is None:
            return 0.0
        delay = parser.crawl_delay(self.user_agent)
        return delay if delay is not None else 0.0
