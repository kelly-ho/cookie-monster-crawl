from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_monster_crawl.utils import RobotsChecker
from cookie_monster_crawl.parser import get_base_domain


class TestRobotsCheckerInit:
    """Test RobotsChecker initialization."""
    
    def test_init_default_user_agent(self):
        """Test RobotsChecker initialization with default user agent."""
        checker = RobotsChecker("WebCrawler/1.0")
        assert checker.user_agent == "WebCrawler/1.0"
        assert len(checker.parsers) == 0


class TestRobotsCheckerIsAllowed:
    """Test RobotsChecker.is_allowed() method."""
    
    def test_is_allowed_when_not_disallowed(self):
        """Test is_allowed returns True when path is not disallowed."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser):
            result = checker.is_allowed("https://oatmealraisin.com/allowed/page")
        
        assert result is True

    def test_is_allowed_when_disallowed(self):
        """Test is_allowed returns False when path is disallowed."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = False
        
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser):
            result = checker.is_allowed("https://oatmealraisin.com/admin/")
        
        assert result is False

    def test_is_allowed_assumes_allowed_on_error(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        # Simulate the internal failure by returning None, as the code does
        with patch.object(checker, '_load_robots_txt', return_value=None):
            result = checker.is_allowed("https://oatmealraisin.com/page")
        assert result is True

    def test_is_allowed_caches_robots_parser(self):
        """Test is_allowed caches the robots.txt parser per domain."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser) as mock_load:
            # First call
            checker.is_allowed("https://oatmealraisin.com/page1")
            # Second call to same domain
            checker.is_allowed("https://oatmealraisin.com/page2")
            
            # _load_robots_txt should only be called once per domain
            assert mock_load.call_count == 1

    def test_is_allowed_different_domains(self):
        """Test is_allowed handles multiple domains."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser1 = MagicMock(spec=RobotFileParser)
        mock_parser1.can_fetch.return_value = True
        mock_parser2 = MagicMock(spec=RobotFileParser)
        mock_parser2.can_fetch.return_value = False
        call_count = 0
        def side_effect(domain):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_parser1
            else:
                return mock_parser2
        
        with patch.object(checker, '_load_robots_txt', side_effect=side_effect):
            result1 = checker.is_allowed("https://oatmealraisin.com/page")
            result2 = checker.is_allowed("https://other.com/page")
        
        assert result1 is True
        assert result2 is False


class TestRobotsCheckerGetCrawlDelay:
    """Test RobotsChecker.get_crawl_delay() method."""

    def test_get_crawl_delay_returns_delay_when_set(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.crawl_delay.return_value = 2.5
        
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser):
            # Manually inject into parser cache to simulate successful load/storage
            checker.parsers["example.com"] = mock_parser
            result = checker.get_crawl_delay("example.com")
        
        assert result == 2.5

    def test_get_crawl_delay_handles_error(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        # Simulate load failure by returning None and ensuring entry is None
        with patch.object(checker, '_load_robots_txt', return_value=None):
            checker.parsers["example.com"] = None
            result = checker.get_crawl_delay("example.com")
        
        assert result == 0.0

    def test_get_crawl_delay_caches_per_domain(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser) as mock_load:
            # First call simulates successful storage
            checker.parsers["example.com"] = checker._load_robots_txt("example.com")
            checker.get_crawl_delay("example.com")
            
            # If caching works, it was only "loaded" once (manually in our setup)
            assert mock_load.call_count == 1


class TestRobotsCheckerLoadRobotsTxt:
    """Test RobotsChecker._load_robots_txt() method."""
    
    def test_load_robots_txt_success(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        # Patch the parser's read method directly to avoid complex urlopen mocking
        with patch('urllib.robotparser.RobotFileParser.read', return_value=None):
            parser = checker._load_robots_txt("example.com")
            assert isinstance(parser, RobotFileParser)
            assert parser.url == "https://example.com/robots.txt"

    def test_load_robots_txt_handles_connection_error(self):
        checker = RobotsChecker(user_agent="TestBot/1.0")
        # Simulate the internal parser.read() crashing
        with patch('urllib.robotparser.RobotFileParser.read', side_effect=Exception("Timeout")):
            result = checker._load_robots_txt("example.com")
            assert result is None 

    def test_load_robots_txt_does_not_cache(self):
        """Verifies that the helper itself returns fresh objects (as designed)."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        with patch('urllib.robotparser.RobotFileParser.read', return_value=None):
            p1 = checker._load_robots_txt("example.com")
            p2 = checker._load_robots_txt("example.com")
            assert p1 is not p2


class TestRobotsCheckerIntegration:
    """Integration tests for RobotsChecker."""
    
    def test_full_workflow_allowed(self):
        """Test full workflow: check if allowed and get crawl delay."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = 2.0
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser):
            is_allowed = checker.is_allowed("https://oatmealraisin.com/allowed")
            crawl_delay = checker.get_crawl_delay("example.com")
        
        assert is_allowed is True
        assert crawl_delay == 2.0

    def test_full_workflow_disallowed(self):
        """Test full workflow with disallowed path."""
        checker = RobotsChecker(user_agent="TestBot/1.0")
        mock_parser = MagicMock(spec=RobotFileParser)
        mock_parser.can_fetch.return_value = False
        mock_parser.crawl_delay.return_value = 5.0
        with patch.object(checker, '_load_robots_txt', return_value=mock_parser):
            is_allowed = checker.is_allowed("https://oatmealraisin.com/admin/")
            crawl_delay = checker.get_crawl_delay("example.com")
        
        assert is_allowed is False
        assert crawl_delay == 5.0
