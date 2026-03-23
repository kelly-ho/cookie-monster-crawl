from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys
from urllib.robotparser import RobotFileParser
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_monster_crawl.utils import RobotsChecker
from cookie_monster_crawl.parser import get_base_domain


class TestRobotsCheckerInit:
    """Test RobotsChecker initialization."""
    
    def test_init_with_headers(self):
        """Test RobotsChecker initialization with headers dict."""
        headers = {"User-Agent": "WebCrawler/1.0", "Accept": "text/html"}
        checker = RobotsChecker(headers)
        assert checker.user_agent == "WebCrawler/1.0"
        assert len(checker.parsers) == 0
        assert checker.fetch_headers == headers

    def test_init_extracts_user_agent_from_headers(self):
        """Test that user agent is extracted from headers."""
        headers = {"User-Agent": "TestBot/2.0", "Accept-Language": "en-US"}
        checker = RobotsChecker(headers)
        assert checker.user_agent == "TestBot/2.0"

    def test_init_defaults_user_agent_when_missing(self):
        """Test default user agent when not in headers."""
        headers = {"Accept": "text/html"}
        checker = RobotsChecker(headers)
        assert checker.user_agent == "UnknownBot"


class TestRobotsCheckerIsAllowed:
    """Test RobotsChecker.is_allowed() method."""
    
    @pytest.mark.asyncio
    async def test_is_allowed_when_not_disallowed(self):
        """Test is_allowed returns True when path is not disallowed."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser):
            result = await checker.is_allowed("https://oatmealraisin.com/allowed/page")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_when_disallowed(self):
        """Test is_allowed returns False when path is disallowed."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = False
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser):
            result = await checker.is_allowed("https://oatmealraisin.com/admin/")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_is_allowed_assumes_allowed_on_error(self):
        """Test is_allowed returns True when robots.txt fails to load."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        # Simulate the internal failure by returning None, as the code does
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=None):
            result = await checker.is_allowed("https://oatmealraisin.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_caches_robots_parser(self):
        """Test is_allowed caches the robots.txt parser per domain."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser) as mock_load:
            # First call
            await checker.is_allowed("https://oatmealraisin.com/page1")
            # Second call to same domain
            await checker.is_allowed("https://oatmealraisin.com/page2")
            
            # _load_robots_txt should only be called once per domain
            assert mock_load.call_count == 1

    @pytest.mark.asyncio
    async def test_is_allowed_different_domains(self):
        """Test is_allowed handles multiple domains."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser1 = MagicMock(spec=RobotFileParser)
        mock_parser1.can_fetch.return_value = True
        mock_parser2 = MagicMock(spec=RobotFileParser)
        mock_parser2.can_fetch.return_value = False
        call_count = 0
        async def side_effect(domain):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_parser1
            else:
                return mock_parser2
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, side_effect=side_effect):
            result1 = await checker.is_allowed("https://oatmealraisin.com/page")
            result2 = await checker.is_allowed("https://other.com/page")
        
        assert result1 is True
        assert result2 is False


class TestRobotsCheckerGetCrawlDelay:
    """Test RobotsChecker.get_crawl_delay() method."""

    @pytest.mark.asyncio
    async def test_get_crawl_delay_returns_delay_when_set(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.crawl_delay.return_value = 2.5
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser):
            # Manually inject into parser cache to simulate successful load/storage
            checker.parsers["example.com"] = mock_parser
            result = await checker.get_crawl_delay("example.com")
        
        assert result == 2.5

    @pytest.mark.asyncio
    async def test_get_crawl_delay_handles_error(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        # Simulate load failure by returning None and ensuring entry is None
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=None):
            checker.parsers["example.com"] = None
            result = await checker.get_crawl_delay("example.com")
        
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_crawl_delay_caches_per_domain(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser) as mock_load:
            # First call simulates successful storage
            checker.parsers["example.com"] = await checker._load_robots_txt("example.com")
            await checker.get_crawl_delay("example.com")
            
            # If caching works, it was only "loaded" once (manually in our setup)
            assert mock_load.call_count == 1


class TestRobotsCheckerLoadRobotsTxt:
    """Test RobotsChecker._load_robots_txt() method."""
    
    @pytest.mark.asyncio
    async def test_load_robots_txt_success(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nDisallow: /admin")
        
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            parser = await checker._load_robots_txt("example.com")
            assert parser is not None
            assert isinstance(parser, RobotFileParser)

    @pytest.mark.asyncio
    async def test_load_robots_txt_handles_404(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)

        mock_response = MagicMock()
        mock_response.status = 404

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await checker._load_robots_txt("example.com")
            assert result is None

    @pytest.mark.asyncio
    async def test_load_robots_txt_handles_connection_error(self):
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await checker._load_robots_txt("example.com")
            assert result is None


class TestRobotsCheckerIntegration:
    """Integration tests for RobotsChecker."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_allowed(self):
        """Test full workflow: check if allowed and get crawl delay."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = 2.0
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser):
            is_allowed = await checker.is_allowed("https://oatmealraisin.com/allowed")
            crawl_delay = await checker.get_crawl_delay("oatmealraisin.com")
        
        assert is_allowed is True
        assert crawl_delay == 2.0

    @pytest.mark.asyncio
    async def test_full_workflow_disallowed(self):
        """Test full workflow with disallowed path."""
        headers = {"User-Agent": "TestBot/1.0"}
        checker = RobotsChecker(headers=headers)
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = False
        mock_parser.crawl_delay.return_value = 5.0
        with patch.object(checker, '_load_robots_txt', new_callable=AsyncMock, return_value=mock_parser):
            is_allowed = await checker.is_allowed("https://oatmealraisin.com/admin/")
            crawl_delay = await checker.get_crawl_delay("oatmealraisin.com")
        
        assert is_allowed is False
        assert crawl_delay == 5.0
