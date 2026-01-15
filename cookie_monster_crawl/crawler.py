import time
import requests
from typing import Optional
from collections import deque

START_URLS = [
    "https://www.americastestkitchen.com/", 
    "https://www.seriouseats.com/"
]

MAX_PAGES = 20
REQUEST_DELAY = 1.0  # seconds between requests

HEADERS = {
    "User-Agent": "CookieMonsterCrawler/0.1 (+https://github.com/kelly-ho/cookie-monster-crawler)"
}

def fetch(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        print("response", response.status_code)
        if response.status_code != 200:
            return None
        elif "text/html" not in response.headers.get("Content-Type", ""):
            return None
        return response.text
    except requests.RequestException:
        return None


def crawl():
    queue = deque(START_URLS)
    visited = set()
    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            continue
        print(f"Fetching: {url}")
        visited.add(url)
        html = fetch(url)
        if html:
            print(html)
        else:
            print("Fetch failed")
        # TODO: process text and find links to add to queue
        time.sleep(REQUEST_DELAY)
    print(f"\nDone. Visited {len(visited)} pages.")


if __name__ == "__main__":
    crawl()
