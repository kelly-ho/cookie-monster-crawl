import logging
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
from cookie_monster_crawl.parser import get_base_domain


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
    '''Manages robots.txt rules per domain'''
    
    def __init__(self, user_agent: str):
        '''Initialize RobotsChecker with a user agent.'''
        self.user_agent = user_agent
        self.parsers: Dict[str, Optional[RobotFileParser]] = {}
        self.logger = logging.getLogger(__name__)
    
    def _load_robots_txt(self, domain: str) -> Optional[RobotFileParser]:
        '''Fetch and parse robots.txt for a domain'''
        robots_url = f"https://{domain}/robots.txt"
        try:
            parser = RobotFileParser(robots_url)
            parser.read()
            self.logger.debug(f"Loaded robots.txt from {domain}")
            return parser
        except Exception as e:
            self.logger.warning(f"Failed to fetch robots.txt from {domain}: {e}")
            return None
    
    def is_allowed(self, url: str) -> bool:
        '''Check if URL is allowed by the domain's robots.txt'''
        domain = get_base_domain(url)
        if domain not in self.parsers:
            parser = self._load_robots_txt(domain)
            self.parsers[domain] = parser
        parser = self.parsers[domain]
        if parser is None:
            return True
        allowed = parser.can_fetch(self.user_agent, url)
        if not allowed:
            self.logger.info(f"Blocked by robots.txt: {url}")
        return allowed
    
    def get_crawl_delay(self, domain: str) -> float:
        '''Get the crawl delay (in seconds) for a domain from robots.txt'''
        if domain not in self.parsers:
            self.parsers[domain] = self._load_robots_txt(domain)
        parser = self.parsers[domain]
        if parser is None:
            return 0.0
        delay = parser.crawl_delay(self.user_agent)
        return delay if delay is not None else 0.0
    
