import logging
import pickle
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

DEFAULT_SCORING = {
    "lsh_threshold": 0.4,
    "num_perm": 128,
    "rescore_sensitivity": 0.3,
    "lock_penalty": 0.1,
    "max_score_threshold": 0.80,
    "components": {
        "base": 0.4,
        "dead_branch_penalty": 2.0,
        "dead_branch_threshold": 0.2,
        "domain_multiplier": 0.5,
        "lsh_multiplier": 1.5,
    },
    "anchor": {
        "empty": 2.0,
        "one_word": 1.0,
        "two_three_words": 0.0,
        "four_plus_words": -0.8,
    },
    "leaf": {
        "infrastructure": 2.0,
        "navigational": 0.8,
        "recipe_single_word": 0.3,
        "single_word_default": 0.5,
        "two_words": 0.1,
        "three_words": -0.5,
        "four_plus_words": -1.0,
    },
    "mid": {
        "infrastructure": 1.5,
        "navigational": 0.3,
        "recipe_related": -0.3,
    },
}

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
        infrastructure_file='infrastructure_segments.txt',
        navigational_file='navigational_segments.txt',
        recipe_related_file='recipe_related_segments.txt',
        scoring_config: dict = None,
        max_score_threshold: float = None,
        model_path: str = None,
    ):
        self.infrastructure_segments = _load_segments_from_file(infrastructure_file)
        self.navigational_segments = _load_segments_from_file(navigational_file)
        self.recipe_related_segments = _load_segments_from_file(recipe_related_file)

        sc = DEFAULT_SCORING.copy()
        if scoring_config:
            for key, val in scoring_config.items():
                if isinstance(val, dict) and key in sc and isinstance(sc[key], dict):
                    sc[key] = {**sc[key], **val}
                else:
                    sc[key] = val
        self.sc = sc

        self.num_perm = sc["num_perm"]
        self.lsh = MinHashLSH(threshold=sc["lsh_threshold"], num_perm=self.num_perm)
        self.junk_counter = 0
        self.rescore_sensitivity = sc["rescore_sensitivity"]
        self.lock_penalty = sc["lock_penalty"]
        self.max_score_threshold = max_score_threshold if max_score_threshold is not None else sc["max_score_threshold"]
        # { "domain": { "path_root": [success_count, total_count] } }
        self.domain_path_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))

        self.model = None
        self.model_feature_names = None
        if model_path:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.model_feature_names = data["feature_names"]
            logger.info(f"Loaded scoring model from {model_path}")

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
        a = self.sc["anchor"]
        words = anchor_text.strip().split()
        word_count = len(words)
        if word_count == 0:
            return a["empty"]
        elif word_count == 1:
            return a["one_word"]
        elif word_count >= 4:
            return a["four_plus_words"]
        return a["two_three_words"]
    
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

        lf = self.sc["leaf"]
        mid = self.sc["mid"]

        if leaf in self.infrastructure_segments:
            score += lf["infrastructure"]
        elif leaf in self.navigational_segments:
            score += lf["navigational"]
        elif leaf_word_count == 1:
            # Single word leaf is likely an index page
            if leaf in self.recipe_related_segments:
                score += lf["recipe_single_word"]
            else:
                score += lf["single_word_default"]
        elif leaf_word_count == 2:
            score += lf["two_words"]
        elif leaf_word_count == 3:
            score += lf["three_words"]
        elif leaf_word_count >= 4:
            score += lf["four_plus_words"]

        for seg in mid_path:
            if seg in self.infrastructure_segments:
                score += mid["infrastructure"]
            elif seg in self.navigational_segments:
                score += mid["navigational"]
            elif seg in self.recipe_related_segments:
                score += mid["recipe_related"]

        return score
    
    def extract_features(self, url: str, domain_counts: Dict[str, int] = None, anchor_text: str = "") -> dict:
        """Extract raw, config-independent features for model training and inference."""
        if url.lower().endswith(FILE_EXTENSIONS):
            return {}

        domain, _, segments, root = self._get_path_info(url)
        stats = self.domain_path_stats[domain].get(root)
        similar_junk = self.lsh.query(self._get_minhash(url))

        total_domain = sum(domain_counts.values()) if domain_counts else 1
        domain_share = domain_counts.get(get_base_domain(url), 0) / total_domain if domain_counts else 0.0

        leaf = segments[-1] if segments else ""
        mid_path = segments[:-1]
        leaf_words = [w for w in leaf.split('-') if w]

        is_dead_branch = bool(stats and (stats[0] / stats[1]) < self.sc["components"]["dead_branch_threshold"])

        return {
            "domain_share":           round(domain_share, 6),
            "lsh_count":              len(similar_junk),
            "dead_branch":            int(is_dead_branch),
            "anchor_word_count":      len(anchor_text.strip().split()) if anchor_text.strip() else 0,
            "path_depth":             len(segments),
            "leaf_word_count":        len(leaf_words),
            "leaf_is_infrastructure": int(leaf in self.infrastructure_segments),
            "leaf_is_navigational":   int(leaf in self.navigational_segments),
            "leaf_is_recipe_related": int(leaf in self.recipe_related_segments),
            "mid_infrastructure":     sum(1 for s in mid_path if s in self.infrastructure_segments),
            "mid_nav":                sum(1 for s in mid_path if s in self.navigational_segments),
            "mid_recipe":             sum(1 for s in mid_path if s in self.recipe_related_segments),
            "is_roundup_slug": int(bool(set(leaf_words) & {'ideas', 'favorites', 'everyone', 'collection', 'roundup', 'list', 'best', 'guide', 'essential', 'ultimate', 'top'}) and any(w.endswith('s') and len(w) > 3 for w in leaf_words) or bool(re.search(r'-recipes-for-|-ideas-for-|-everyone-will-', leaf))),
            "anchor_has_recipe_keyword": int(any(w in ('recipe', 'recipes') for w in anchor_text.lower().split())),
            "has_pagination_pattern": int(any(segments[i] == 'page' and i + 1 < len(segments) and segments[i + 1].isdigit() for i in range(len(segments))) or bool(re.search(r'[?&](?:page|p|pg)=\d+', url))),
            "domain_harvest_rate": round((lambda d_stats: sum(s[0] for s in d_stats.values()) / sum(s[1] for s in d_stats.values()) if sum(s[1] for s in d_stats.values()) > 0 else 0.5)(self.domain_path_stats[domain]), 6),
            "has_date_in_path": int(bool(re.search(r'/\d{4}/\d{2}/', url))),
            "query_param_count": len(urlparse(url).query.split('&')) if urlparse(url).query else 0,
            "slug_word_count_ratio": round(len(leaf_words) / max(len(segments), 1), 6),
            "has_numeric_id": int(bool(re.search(r'\b\d{4,}\b', '-'.join(segments[-2:])) if segments else False)),
            "is_print_or_wprm": int(any(s in ('print', 'wprm_print', 'recipe-print') for s in segments)),
            "leaf_is_plural": int(leaf_words[-1].endswith('s') if leaf_words else False),
        }

    def calculate_score(self, url: str, domain_counts: Dict[str, int] = None, anchor_text: str = "") -> tuple[float, dict, dict]:
        if url.lower().endswith(FILE_EXTENSIONS):
            return 0.99, {}, {}

        domain, _, segments, root = self._get_path_info(url)
        stats = self.domain_path_stats[domain].get(root)

        m = self._get_minhash(url)
        similar_junk = self.lsh.query(m)

        total_domain = sum(domain_counts.values()) if domain_counts else 1
        domain_share = domain_counts.get(get_base_domain(url), 0) / total_domain if domain_counts else 0.0

        c = self.sc["components"]
        components = {
            "base":        c["base"],
            "dead_branch": c["dead_branch_penalty"] if stats and (stats[0] / stats[1]) < c["dead_branch_threshold"] else 0.0,
            "segments":    self._score_segments(segments),
            "domain":      domain_share * c["domain_multiplier"],
            "anchor":      self._score_anchor_complexity(anchor_text) if anchor_text else 0.0,
            "lsh":         len(similar_junk) * c["lsh_multiplier"],
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

        raw_features = {
            "domain_share":           round(domain_share, 6),
            "lsh_count":              len(similar_junk),
            "dead_branch":            int(bool(stats and (stats[0] / stats[1]) < c["dead_branch_threshold"])),
            "anchor_word_count":      len(anchor_text.strip().split()) if anchor_text.strip() else 0,
            "path_depth":             len(segments),
            "leaf_word_count":        len([w for w in (segments[-1] if segments else "").split('-') if w]),
            "leaf_is_infrastructure": int(segments[-1] in self.infrastructure_segments if segments else False),
            "leaf_is_navigational":   int(segments[-1] in self.navigational_segments if segments else False),
            "leaf_is_recipe_related": int(segments[-1] in self.recipe_related_segments if segments else False),
            "mid_infrastructure":     sum(1 for s in segments[:-1] if s in self.infrastructure_segments),
            "mid_nav":                sum(1 for s in segments[:-1] if s in self.navigational_segments),
            "mid_recipe":             sum(1 for s in segments[:-1] if s in self.recipe_related_segments),
            "is_roundup_slug": int((lambda lw, lf: bool(set(lw) & {'ideas', 'favorites', 'everyone', 'collection', 'roundup', 'list', 'best', 'guide', 'essential', 'ultimate', 'top'}) and any(w.endswith('s') and len(w) > 3 for w in lw) or bool(re.search(r'-recipes-for-|-ideas-for-|-everyone-will-', lf)))([w for w in (segments[-1] if segments else '').split('-') if w], segments[-1] if segments else '')),
            "anchor_has_recipe_keyword": int(any(w in ('recipe', 'recipes') for w in anchor_text.lower().split())),
            "has_pagination_pattern": int(any(segments[i] == 'page' and i + 1 < len(segments) and segments[i + 1].isdigit() for i in range(len(segments))) or bool(re.search(r'[?&](?:page|p|pg)=\d+', url))),
            "domain_harvest_rate": round((lambda d_stats: sum(s[0] for s in d_stats.values()) / sum(s[1] for s in d_stats.values()) if sum(s[1] for s in d_stats.values()) > 0 else 0.5)(self.domain_path_stats[domain]), 6),
            "has_date_in_path": int(bool(re.search(r'/\d{4}/\d{2}/', url))),
            "query_param_count": len(urlparse(url).query.split('&')) if urlparse(url).query else 0,
            "slug_word_count_ratio": round(len([w for w in (segments[-1] if segments else '').split('-') if w]) / max(len(segments), 1), 6),
            "has_numeric_id": int(bool(re.search(r'\b\d{4,}\b', '-'.join(segments[-2:])) if segments else False)),
            "is_print_or_wprm": int(any(s in ('print', 'wprm_print', 'recipe-print') for s in segments)),
            "leaf_is_plural": int((segments[-1].split('-')[-1] if segments else '').endswith('s')),
        }

        if self.model is not None:
            feature_vector = [[raw_features.get(f, 0.0) for f in self.model_feature_names]]
            # P(non-recipe): high = likely junk = deprioritized; low = likely recipe = high priority
            model_score = float(self.model.predict_proba(feature_vector)[0][0])
            return model_score, {}, raw_features

        return float(final), {k: round(v, 6) for k, v in components.items()}, raw_features


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
