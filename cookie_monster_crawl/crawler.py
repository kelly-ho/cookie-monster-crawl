import time
import requests
import json
from typing import Optional
from collections import deque
from parser import get_links, is_recipe_page, get_recipe_data

START_URLS = [
    "https://www.americastestkitchen.com/", 
    "https://www.seriouseats.com/"
]

MAX_PAGES = 20
REQUEST_DELAY_SECS = 1.0

HEADERS = {
    "User-Agent": "CookieMonsterCrawler/0.1 (+https://github.com/kelly-ho/cookie-monster-crawler)"
}

def fetch(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
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
    recipes = []
    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            continue
        print(f"Fetching: {url}")
        visited.add(url)
        html = fetch(url)
        if html:
            recipe = get_recipe_data(html, url)
            if recipe:
                print(f" Found recipe: {recipe['title']}")
                recipes.append(recipe)
            links = get_links(html, url)
            for link in links:
                if link not in visited:
                    queue.append(link)
            print(f" Found {len(links)} links")
        else:
            print(" Fetch failed")
        time.sleep(REQUEST_DELAY_SECS)
    
    output_file = "recipes.json"
    with open(output_file, "w") as f:
        json.dump(recipes, f, indent=2)
    
    print(f"\nDone. Visited {len(visited)} pages, found {len(recipes)} recipes.")
    print(f"Recipes saved to {output_file}")


if __name__ == "__main__":
    crawl()
