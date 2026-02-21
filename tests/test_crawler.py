import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_monster_crawl.crawler import Crawler

class TestCrawlerInit:
    """Test Crawler initialization."""
    
    def test_crawler_init_default_params(self):
        """Test Crawler initialization with default parameters."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        assert crawler.start_urls == start_urls
        assert crawler.max_pages == 100
        assert crawler.concurrency == 5
        assert crawler.delay_secs == 1.0
        assert crawler.timeout_secs == 15
        assert len(crawler.visited) == 0
        assert len(crawler.recipes) == 0
        assert crawler.robots_checker is not None

    def test_crawler_init_custom_params(self):
        """Test Crawler initialization with custom parameters."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(
            start_urls=start_urls,
            max_pages=50,
            concurrency=10,
            delay_secs=2.0,
            timeout_secs=30
        )
        
        assert crawler.max_pages == 50
        assert crawler.concurrency == 10
        assert crawler.delay_secs == 2.0
        assert crawler.timeout_secs == 30


class TestCrawlerFetch:
    """Test Crawler.fetch() method."""
    
    @pytest.mark.asyncio
    async def test_fetch_success(self):
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        # Must use AsyncMock to support the 'async with' protocol
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text.return_value = "<html>test</html>"
        mock_session = MagicMock() 
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
             patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0), \
             patch('cookie_monster_crawl.parser.get_base_domain', return_value="smores.com"):
            
            result = await crawler.fetch(mock_session, "https://smores.com")
        
        assert result == "<html>test</html>"

    @pytest.mark.asyncio
    async def test_fetch_blocked_by_robots(self):
        """Test fetch blocked by robots.txt."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        mock_session = AsyncMock()
        
        with patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=False):
            result = await crawler.fetch(mock_session, "https://smores.com/blocked")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_non_200_status(self):
        """Test fetch with non-200 status code."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.headers = {"Content-Type": "text/html"}
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('cookie_monster_crawl.parser.get_base_domain', return_value="smores.com"), \
            patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
            patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0):
            
            result = await crawler.fetch(mock_session, "https://smores.com/notfound")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_non_html_content(self):
        """Test fetch with non-HTML content type."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True):
            with patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0):
                result = await crawler.fetch(mock_session, "https://smores.com/api")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test fetch with timeout error."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        mock_session = MagicMock()
        mock_session.get.side_effect = asyncio.TimeoutError()
        
        with patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True):
            with patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0):
                result = await crawler.fetch(mock_session, "https://smores.com")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_respects_crawl_delay(self):
        """Test that fetch respects robots.txt crawl-delay."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls, delay_secs=0)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = AsyncMock(return_value="<html>test</html>")
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True):
            with patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0.5):
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    result = await crawler.fetch(mock_session, "https://smores.com")
                    mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_fetch_429_retry(self):
        """Test fetch retries on 429 status."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        # First call returns 429, second returns 200
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Content-Type": "text/html"}
        
        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.headers = {"Content-Type": "text/html"}
        mock_response_200.text.return_value = "<html>success</html>"
        
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.side_effect = [mock_response_429, mock_response_200]
        
        with patch('cookie_monster_crawl.parser.get_base_domain', return_value="smores.com"), \
            patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
            patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0), \
            patch('asyncio.sleep', new_callable=AsyncMock):
            
            result = await crawler.fetch(mock_session, "https://smores.com")
        
        assert result == "<html>success</html>"
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_5xx_retry(self):
        """Test fetch retries on 503 status."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        mock_response_503 = AsyncMock()
        mock_response_503.status = 503
        mock_response_503.headers = {"Content-Type": "text/html"}
        
        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.headers = {"Content-Type": "text/html"}
        mock_response_200.text.return_value = "<html>recovered</html>"
        
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.side_effect = [mock_response_503, mock_response_200]
        
        with patch('cookie_monster_crawl.parser.get_base_domain', return_value="smores.com"), \
            patch.object(crawler.robots_checker, 'is_allowed', new_callable=AsyncMock, return_value=True), \
            patch.object(crawler.robots_checker, 'get_crawl_delay', new_callable=AsyncMock, return_value=0), \
            patch('asyncio.sleep', new_callable=AsyncMock):
            
            result = await crawler.fetch(mock_session, "https://smores.com")
        
        assert result == "<html>recovered</html>"


class TestCrawlerSaveResults:
    """Test Crawler.save_results() method."""
    
    def test_save_results(self, tmp_path):
        """Test saving recipes to JSON file."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        crawler.recipes = [
            {
                "title": "Chocolate Chip Cookies",
                "url": "https://smores.com/recipes/chocolate-chip",
                "ingredients": ["flour", "butter"],
                "instructions": "Mix and bake"
            }
        ]
        
        # Change to temp directory for test
        original_cwd = Path.cwd()
        import os
        os.chdir(tmp_path)
        
        try:
            crawler.save_results()
            
            output_file = tmp_path / "recipes.json"
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["title"] == "Chocolate Chip Cookies"
        finally:
            os.chdir(original_cwd)

    def test_save_results_empty(self, tmp_path):
        """Test saving empty recipes list."""
        start_urls = ["https://smores.com"]
        crawler = Crawler(start_urls=start_urls)
        
        original_cwd = Path.cwd()
        import os
        os.chdir(tmp_path)
        
        try:
            crawler.save_results()
            
            output_file = tmp_path / "recipes.json"
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert data == []
        finally:
            os.chdir(original_cwd)
