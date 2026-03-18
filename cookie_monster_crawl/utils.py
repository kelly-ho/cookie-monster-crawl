import logging
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional, Set
import re
import aiohttp
from urllib.parse import urlparse
import numpy as np
from datasketch import MinHash, MinHashLSH
from cookie_monster_crawl.parser import get_base_domain
from collections import defaultdict
from pathlib import Path

FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip')

logger = logging.getLogger(__name__)

def _load_segments_from_file(filename: str) -> Set[str]:
    """Load URL segments from a text file in the data folder."""
    data_dir = Path(__file__).parent.parent / "data"
    filepath = data_dir / filename
    try:
        with open(filepath, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logging.warning(f"Segment file not found: {filepath}")
        return set()

class URLPrioritizer:
    def __init__(
        self, 
        lsh_threshold=0.4, 
        num_perm=128,
        infrastructure_file='infrastructure_segments.txt',
        navigational_file='navigational_segments.txt',
        recipe_related_file='recipe_related_segments.txt',
        max_score_threshold=0.80
    ):
        self.infrastructure_segments = _load_segments_from_file(infrastructure_file)
        self.navigational_segments = _load_segments_from_file(navigational_file)
        self.recipe_related_segments = _load_segments_from_file(recipe_related_file)
        
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.junk_counter = 0
        self.rescore_sensitivity = 0.3
        self.lock_penalty = 0.1
        self.max_score_threshold = max_score_threshold
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
    
    def _score_segments(self, segments: list) -> float:
        """
        Score URL based on segment-level analysis.
        - Reward longer URL slug length for leaf segments
        - Categorize mid-path segments for context (recipe related, infra, nav)
        """
        if not segments:
            return 0

        score = 0.0
        leaf = segments[-1]
        mid_path = segments[:-1]

        leaf_words = [w for w in leaf.split('-') if w]
        leaf_word_count = len(leaf_words)

        if leaf in self.infrastructure_segments:
            score += 2.0
        elif leaf in self.navigational_segments:
            score += 0.8
        elif leaf_word_count == 1:
            # Single word leaf is likely an index page
            if leaf in self.recipe_related_segments:
                score += 0.3
            else:
                score += 0.5
        elif leaf_word_count == 2:
            score += 0.1
        elif leaf_word_count == 3:
            score -= 0.5
        elif leaf_word_count >= 4:
            score -= 1.0

        for seg in mid_path:
            if seg in self.infrastructure_segments:
                score += 1.5
            elif seg in self.navigational_segments:
                score += 0.3
            elif seg in self.recipe_related_segments:
                score -= 0.3

        return score
    
    def calculate_score(self, url: str, domain_counts: Dict[str, int] = None, anchor_text: str = "") -> tuple[float, dict]:
        if url.lower().endswith(FILE_EXTENSIONS):
            return 0.99, {}

        domain, _, segments, root = self._get_path_info(url)
        stats = self.domain_path_stats[domain].get(root)

        m = self._get_minhash(url)
        similar_junk = self.lsh.query(m)

        total_domain = sum(domain_counts.values()) if domain_counts else 1
        domain_share = domain_counts.get(get_base_domain(url), 0) / total_domain if domain_counts else 0.0

        components = {
            "base":        0.4,
            "dead_branch": 2.0 if stats and (stats[0] / stats[1]) < 0.2 else 0.0,
            "segments":    self._score_segments(segments),
            "domain":      domain_share * 0.5,
            "anchor":      self._score_anchor_complexity(anchor_text) if anchor_text else 0.0,
            "lsh":         len(similar_junk) * 1.5,
        }

        raw = sum(components.values())
        final = 1 / (1 + np.exp(-raw))

        logger.debug("score_tracker: %s", {
            "url": url, **components,
            "raw_score": round(raw, 4),
            "final_score": round(final, 4),
            "path_stats": f"{stats[0]}/{stats[1]}" if stats else None,
            "anchor_text": anchor_text or None,
            "lsh_matches": len(similar_junk),
        })

        return final, {k: round(v, 6) for k, v in components.items()}


class RobotsChecker:
    def __init__(self, headers: Dict[str, str]):
        self.user_agent = headers.get("User-Agent", "UnknownBot")
        self.parsers: Dict[str, Optional[RobotFileParser]] = {}
        self.fetch_headers = headers.copy()

    async def _load_robots_txt(self, domain: str) -> Optional[RobotFileParser]:
        """Asynchronously fetch robots.txt with headers."""
        robots_url = f"https://{domain}/robots.txt"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, headers=self.fetch_headers, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        logger.warning(f"Blocked or error {response.status} for {domain}")
                        return None
                    text = await response.text()
                    parser = RobotFileParser()
                    parser.parse(text.splitlines())
                    return parser
        except Exception as e:
            logger.error(f"Unable to load robots.txt for {domain}: {e}")
            return None

    async def _get_parser(self, domain: str) -> Optional[RobotFileParser]:
        """Return cached robots parser for domain, loading if needed."""
        if domain not in self.parsers:
            self.parsers[domain] = await self._load_robots_txt(domain)
        return self.parsers[domain]

    async def is_allowed(self, url: str) -> bool:
        """Check permission without blocking the event loop."""
        domain = urlparse(url).hostname
        parser = await self._get_parser(domain)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def get_crawl_delay(self, domain: str) -> float:
        '''Get the crawl delay (in seconds) for a domain from robots.txt'''
        parser = await self._get_parser(domain)
        if parser is None:
            return 0.0
        delay = parser.crawl_delay(self.user_agent)
        return delay if delay is not None else 0.0
