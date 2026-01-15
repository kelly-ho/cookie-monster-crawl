import requests
from unittest.mock import patch, Mock

from cookie_monster_crawl import crawler

@patch("cookie_monster_crawl.crawler.requests.get")
def test_fetch_html(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html>ok</html>"
    mock_get.return_value = mock_response
    result = crawler.fetch("https://snickerdoodle.com")
    assert result == "<html>ok</html>"

@patch("cookie_monster_crawl.crawler.requests.get")
def test_fetch_status_not_200(mock_get):
    mock_response = Mock()
    mock_response.status_code = 3
    mock_response.headers = {"Content-Type": "text/html"}
    mock_get.return_value = mock_response
    result = crawler.fetch("https://chocolate.chip.cookies.com")
    assert result is None

@patch("cookie_monster_crawl.crawler.requests.get")
def test_download_exception(mock_get):
    mock_get.side_effect = requests.RequestException()
    result = crawler.fetch("https://oatmeal.raisin.com")
    assert result is None