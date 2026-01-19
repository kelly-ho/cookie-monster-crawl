import logging
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
from .parser import get_base_domain


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
