import json
from typing import Set, Optional, Dict
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup


def get_base_domain(url: str) -> str:
    """Extract domain from URL for same-domain filtering."""
    parsed = urlparse(url)
    return parsed.netloc


def get_links(html: str, base_url: str) -> Dict[str, str]:
    """
    Extract all links from HTML and normalize them to absolute URLs.
    Only includes links from the same domain.
    Returns dict mapping URL -> anchor text.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = {}
    base_domain = get_base_domain(base_url)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        final_url = urljoin(base_url, href).split("#")[0].rstrip("/")
        if get_base_domain(final_url) == base_domain:
            anchor_text = a.get_text(strip=True)
            links[final_url] = anchor_text
    return links

def _try_load_json(raw_str: str) -> Optional[dict]:
    """Helper to handle standard and malformed JSON-LD."""
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        # Step 2: Self-clean literal newlines inside values
        sanitized = re.sub(
            r'("(?:[^"\\]|\\.)*")', 
            lambda m: m.group(1).replace('\n', '\\n').replace('\r', '\\r'), 
            raw_str
        )
        try:
            return json.loads(sanitized)
        except Exception:
            return None

def get_recipe_data(html: str, url: str) -> Optional[dict]:
    """
    Get recipe data from HTML with standarized json-ld Recipe schema.
    Returns a dict with title, ingredients, instructions, or None if no recipe found.
    """
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    
    for script in scripts:
        try:
            data = _try_load_json(script.get_text(strip=True))
            recipe = _extract_recipe_from_data(data)
            if recipe:
                recipe["url"] = url
                return recipe
        except (json.JSONDecodeError, TypeError):
            continue
    
    return None


def _extract_recipe_from_data(data: any) -> Optional[dict]:
    """Recursively find and extract Recipe schema from json-ld data."""
    if isinstance(data, dict):
        node_type = data.get("@type")
        types = node_type if isinstance(node_type, list) else [node_type]
        if "Recipe" in types:
            return _parse_recipe(data)
        for value in data.values():
            result = _extract_recipe_from_data(value)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _extract_recipe_from_data(item)
            if result:
                return result
    return None


def _parse_recipe(recipe_data: dict) -> dict:
    """Parse Recipe schema and extract relevant fields."""
    def extract_text_list(field):
        """Extract text from a field that could be string or list of strings."""
        value = recipe_data.get(field, [])
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in value]
        return []
    
    return {
        "title": recipe_data.get("name", "Unknown"),
        "ingredients": extract_text_list("recipeIngredient"),
        "instructions": extract_text_list("recipeInstructions"),
        "prep_time": recipe_data.get("prepTime"),
        "cook_time": recipe_data.get("cookTime"),
        "servings": recipe_data.get("recipeYield"),
    }
