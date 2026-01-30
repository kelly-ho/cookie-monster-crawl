import logging
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
import httpx
import re
from urllib.parse import urlparse
import numpy as np
from datasketch import MinHash, MinHashLSH
from cookie_monster_crawl.parser import get_base_domain

class URLPrioritizer:
    def __init__(self, lsh_threshold=0.4, num_perm=128):
        self.recipe_patterns = re.compile(r'/(?:recipe(?!s)|cook|make|instructions|bake)/')
        self.non_recipe_patterns = re.compile(r'/(guide|review|blog|article|tag|category|author|search|member|login|cart|shop|holiday|index|how-to|recipes|budget|ideas)/')
        
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.junk_counter = 0
        self.rescore_sensitivity = 0.3

    def _get_minhash(self, url: str):
        """Creates a fingerprint based on URL path segments and character shingles."""
        m = MinHash(num_perm=self.num_perm)
        path = urlparse(url).path.lower()
        path = re.sub(r'\d+', 'ID', path)
        tokens = re.split(r'[/-]', path)
        for token in tokens:
            if not token: continue
            m.update(token.encode('utf8'))
        return m
    
    def learn_non_recipe(self, url: str):
        """Penalize page that is crawled but contains no recipe schema."""
        m = self._get_minhash(url)
        self.junk_counter += 1
        self.lsh.insert(f"junk_{self.junk_counter}", m)
    
    def calculate_score(self, url: str, domain_counts: Dict[str, int] = None, anchor_text: str = "") -> float:
        score = 0.5 
        path = urlparse(url).path.lower()

        recipe_matches = self.recipe_patterns.findall(path)
        non_recipe_matches = self.non_recipe_patterns.findall(path)

        # lower score means higher priority
        score -= (len(recipe_matches) * 0.8)
        score += (len(non_recipe_matches) * 1.2)
        
        # Penalize frequent domains for diversity
        if domain_counts:
            domain = get_base_domain(url)
            fetch_count = domain_counts.get(domain, 0)
            score += (fetch_count * 0.05)


        m = self._get_minhash(url)
        similar_junk = self.lsh.query(m)
        if similar_junk:
            score += (len(similar_junk) * 1.5)

        # TODO: Adjust score based on anchor text
        # TODO: Compare z score to sigmoid
        return 1 / (1 + np.exp(-score))


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
                # If we get blocked (403), log it and be conservative
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
            return True
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
