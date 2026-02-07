import logging
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
import re
import aiohttp
from urllib.parse import urlparse
import numpy as np
from datasketch import MinHash, MinHashLSH
from cookie_monster_crawl.parser import get_base_domain
from collections import defaultdict

FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip')

class URLPrioritizer:
    def __init__(self, lsh_threshold=0.4, num_perm=128):
        self.recipe_patterns = re.compile(r'/(?:recipe(?!s)|cook|make|instructions|bake)/')
        self.non_recipe_patterns = re.compile(
            r'/(guide|review|blog|article|tag|category|author|search|'
            r'member|login|cart|shop|holiday|index|how-to|recipes|'
            r'budget|ideas|collection|story|shopping)/'
        )
        
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.junk_counter = 0
        self.rescore_sensitivity = 0.3
        # { "domain": { "path_root": [success_count, total_count] } }
        self.domain_path_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def _get_path_info(self, url: str):
        """Helper to extract consistent segments and root."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        segments = [s for s in path.split('/') if s]
        root = segments[0] if segments else "root"
        return parsed.netloc, path, segments, root
    
    def _get_minhash(self, url: str):
        """Creates a fingerprint based on URL path segments."""
        m = MinHash(num_perm=self.num_perm)
        path = urlparse(url).path.lower()
        path = re.sub(r'\d+', 'ID', path)
        tokens = re.split(r'[/-]', path)
        for token in tokens:
            if not token: continue
            m.update(token.encode('utf8'))
        return m
    
    def update_model(self, url: str, is_recipe: bool):
        """Penalize page that is crawled but contains no recipe schema."""
        domain, _, _, root = self._get_path_info(url)
        self.domain_path_stats[domain][root][1] += 1
        if is_recipe: 
            self.domain_path_stats[domain][root][0] += 1
        else:
            m = self._get_minhash(url)
            self.junk_counter += 1
            self.lsh.insert(f"junk_{self.junk_counter}", m)

    def _score_anchor_complexity(self, anchor_text: str) -> float:
        words = anchor_text.strip().split()
        word_count = len(words)
        if word_count == 0:
            return 2.0
        elif word_count == 1:
            return 1.0
        elif word_count >= 4:
            return -0.8  # Reward specific titles
        return 0.0
    
    def calculate_score(self, url: str, domain_counts: Dict[str, int] = None, anchor_text: str = "") -> float:
        if url.lower().endswith(FILE_EXTENSIONS):
            return 0.99

        # TODO: add segment analysis
        domain, path, segments, root = self._get_path_info(url)
        score = 0.4 

        stats = self.domain_path_stats[domain].get(root)
        if stats:
            success_rate = stats[0] / stats[1]
            if success_rate < 0.2:
                score += 2  # Heavy penalty for proven dead branches

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

        if anchor_text:
            score += self._score_anchor_complexity(anchor_text)

        m = self._get_minhash(url)
        similar_junk = self.lsh.query(m)
        if similar_junk:
            score += (len(similar_junk) * 1.5)

        # TODO: Compare z score to sigmoid
        return 1 / (1 + np.exp(-score))


class RobotsChecker:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: Dict[str, Optional[RobotFileParser]] = {}
        self.fetch_headers = {
            "User-Agent": user_agent,
            "Accept": "text/plain"
        }
        self.logger = logging.getLogger(__name__)
            
    async def _load_robots_txt(self, domain: str) -> Optional[RobotFileParser]:
        """Asynchronously fetch robots.txt with headers."""
        robots_url = f"https://{domain}/robots.txt"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, headers=self.fetch_headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        self.logger.warning(f"Blocked or error {response.status} for {domain}")
                        return None
                    text = await response.text()
                    parser = RobotFileParser()
                    parser.parse(text.splitlines())
                    return parser
        except Exception as e:
            self.logger.error(f"Unable to load robots.txt for {domain}: {e}")
            return None

    async def is_allowed(self, url: str) -> bool:
        """Check permission without blocking the event loop."""
        domain = urlparse(url).hostname
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
