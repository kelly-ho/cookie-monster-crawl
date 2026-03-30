import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_monster_crawl.crawler import Crawler


SEED = "https://a.com"
PAGE = "https://a.com/pasta-carbonara-recipe"
LINK = "https://a.com/some-link"
HTML = "<html></html>"
RECIPE = {"title": "Pasta Carbonara", "ingredients": [], "instructions": [], "url": PAGE}


def make_crawler(**kwargs):
    """Create a Crawler with a mocked crawl_log and a short idle timeout."""
    crawler = Crawler(start_urls=[SEED], timeout_secs=0.1, **kwargs)
    crawler.crawl_log = MagicMock()
    return crawler


async def run_worker(crawler, items):
    """Put (priority, url, anchor) items in the queue and run the worker until idle."""
    for priority, url, anchor in items:
        await crawler.queue.put((priority, (url, anchor)))
        crawler.queued.add(url)
    crawler.session = AsyncMock()
    await asyncio.wait_for(crawler.worker(), timeout=3.0)


class TestLogVisit:
    @pytest.mark.asyncio
    async def test_called_for_non_seed_url(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={}):
            await run_worker(crawler, [(0.5, PAGE, "Pasta")])

        crawler.crawl_log.log_visit.assert_called_once()
        url_arg, priority_arg, pages_arg = crawler.crawl_log.log_visit.call_args[0]
        assert url_arg == PAGE
        assert priority_arg == 0.5
        assert pages_arg == 1

    @pytest.mark.asyncio
    async def test_not_called_for_seed_url(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=None):
            await run_worker(crawler, [(-float('inf'), SEED, "")])

        crawler.crawl_log.log_visit.assert_not_called()


class TestLogResult:
    @pytest.mark.asyncio
    async def test_called_with_recipe(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=RECIPE), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={}):
            await run_worker(crawler, [(0.5, PAGE, "Pasta")])

        crawler.crawl_log.log_result.assert_called_once()
        kwargs = crawler.crawl_log.log_result.call_args[1]
        assert kwargs['url'] == PAGE
        assert kwargs['is_recipe'] is True
        assert kwargs['recipe_title'] == "Pasta Carbonara"
        assert kwargs['cumulative_recipes'] == 1
        assert kwargs['cumulative_pages'] == 1

    @pytest.mark.asyncio
    async def test_called_without_recipe(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        kwargs = crawler.crawl_log.log_result.call_args[1]
        assert kwargs['is_recipe'] is False
        assert kwargs['recipe_title'] is None

    @pytest.mark.asyncio
    async def test_links_found_count_reflects_get_links_output(self):
        crawler = make_crawler()
        two_links = {LINK: "link one", "https://a.com/other": "link two"}

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value=two_links):
            await run_worker(crawler, [(0.5, PAGE, "")])

        kwargs = crawler.crawl_log.log_result.call_args[1]
        assert kwargs['links_found'] == 2

    @pytest.mark.asyncio
    async def test_not_called_for_seed_url(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=None):
            await run_worker(crawler, [(-float('inf'), SEED, "")])

        crawler.crawl_log.log_result.assert_not_called()


class TestLogDiscover:
    @pytest.mark.asyncio
    async def test_called_when_score_below_threshold(self):
        crawler = make_crawler()
        # First call: rescore check for PAGE → no rescore
        # Second call: score for LINK → below threshold, queued
        calculate_score = MagicMock(side_effect=[(0.5, {}, {}), (0.4, {}, {})])

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', calculate_score), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={LINK: "a link"}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        crawler.crawl_log.log_discover.assert_called_once()
        url_arg, source_arg, anchor_arg, score_arg = crawler.crawl_log.log_discover.call_args[0][:4]
        assert url_arg == LINK
        assert source_arg == PAGE
        assert anchor_arg == "a link"
        assert score_arg == 0.4

    @pytest.mark.asyncio
    async def test_not_called_when_link_already_queued(self):
        crawler = make_crawler()
        crawler.queued.add(LINK)

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={LINK: "a link"}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        crawler.crawl_log.log_discover.assert_not_called()


class TestLogFilter:
    @pytest.mark.asyncio
    async def test_called_when_score_above_threshold(self):
        crawler = make_crawler()
        # First call: rescore check for PAGE → 0.5, no rescore
        # Second call: score for LINK → 0.95, above threshold
        calculate_score = MagicMock(side_effect=[(0.5, {}, {}), (0.95, {}, {})])

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', calculate_score), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={LINK: "click"}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        crawler.crawl_log.log_filter.assert_called_once()
        url_arg, reason_arg, score_arg = crawler.crawl_log.log_filter.call_args[0]
        assert url_arg == LINK
        assert reason_arg == "score_threshold"
        assert score_arg == 0.95

    @pytest.mark.asyncio
    async def test_called_when_link_is_robots_blocked(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=False), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={LINK: "click"}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        crawler.crawl_log.log_filter.assert_called_once()
        url_arg, reason_arg = crawler.crawl_log.log_filter.call_args[0][:2]
        assert url_arg == LINK
        assert reason_arg == "robots_blocked"

    @pytest.mark.asyncio
    async def test_robots_blocked_filter_has_no_score_arg(self):
        crawler = make_crawler()

        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=HTML), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=False), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={LINK: "click"}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        # log_filter(link, "robots_blocked") — no third positional arg
        assert len(crawler.crawl_log.log_filter.call_args[0]) == 2


class TestLogRescore:
    @pytest.mark.asyncio
    async def test_called_when_priority_worsens(self):
        crawler = make_crawler()
        # Queued at 0.3, rescores to 0.7:  0.7 > 0.3 + 0.3 → triggers rescore
        # Second dequeue at 0.7, rescores to 0.7: 0.7 > 0.7 + 0.3 = 1.0 → no rescore
        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=None), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.7, {}, {})), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={}):
            await run_worker(crawler, [(0.3, PAGE, "Pasta")])

        crawler.crawl_log.log_rescore.assert_called_once()
        url_arg, old_arg, new_arg = crawler.crawl_log.log_rescore.call_args[0]
        assert url_arg == PAGE
        assert old_arg == 0.3
        assert new_arg == 0.7

    @pytest.mark.asyncio
    async def test_not_called_when_priority_stable(self):
        crawler = make_crawler()
        # Queued at 0.5, rescores to 0.5: 0.5 > 0.5 + 0.3 = 0.8 → no rescore
        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=None), \
             patch.object(crawler.url_prioritizer, 'calculate_score', return_value=(0.5, {}, {})), \
             patch('cookie_monster_crawl.crawler.get_recipe_data', return_value=None), \
             patch('cookie_monster_crawl.crawler.get_links', return_value={}):
            await run_worker(crawler, [(0.5, PAGE, "")])

        crawler.crawl_log.log_rescore.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_called_for_seed_url(self):
        crawler = make_crawler()
        # Seed URLs skip the rescore block entirely
        with patch.object(crawler, 'fetch', new_callable=AsyncMock, return_value=None):
            await run_worker(crawler, [(-float('inf'), SEED, "")])

        crawler.crawl_log.log_rescore.assert_not_called()


class TestCrawlLifecycle:
    @pytest.mark.asyncio
    async def test_close_called_after_crawl(self):
        crawler = make_crawler()

        with patch.object(crawler, 'worker', new_callable=AsyncMock), \
             patch.object(crawler, 'save_results'), \
             patch.object(crawler, 'generate_report'), \
             patch('cookie_monster_crawl.crawler.aiohttp.ClientSession') as mock_cls:
            mock_cls.return_value = AsyncMock()
            await crawler.crawl()

        crawler.crawl_log.close.assert_called_once()

    def test_crawl_log_is_none_when_logging_disabled(self):
        crawler = Crawler(start_urls=[SEED], enable_logging=False)
        assert crawler.crawl_log is None
