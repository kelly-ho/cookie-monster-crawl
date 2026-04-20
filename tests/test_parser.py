import pytest
from cookie_monster_crawl.parser import (
    get_base_domain,
    get_links,
    get_recipe_data,
    _canonicalize_url,
    _get_link_context,
)


class TestGetBaseDomain:
    def test_extracts_domain(self):
        assert get_base_domain("https://www.example.com/path") == "www.example.com"

    def test_includes_port(self):
        assert get_base_domain("http://localhost:8080/page") == "localhost:8080"


class TestCanonicalizeUrl:
    def test_strips_utm_params(self):
        url = "https://example.com/recipe?utm_source=twitter&utm_medium=social"
        assert _canonicalize_url(url) == "https://example.com/recipe"

    def test_strips_tracking_params(self):
        url = "https://example.com/recipe?fbclid=abc123&ref=nav"
        assert _canonicalize_url(url) == "https://example.com/recipe"

    def test_preserves_meaningful_params(self):
        url = "https://example.com/search?q=cookies&page=2"
        assert "q=cookies" in _canonicalize_url(url)
        assert "page=2" in _canonicalize_url(url)

    def test_no_query_unchanged(self):
        url = "https://example.com/recipe/cookies"
        assert _canonicalize_url(url) == url


class TestGetLinkContext:
    def test_main_content_link(self):
        from bs4 import BeautifulSoup
        html = "<div><a href='/recipe'>Recipe</a></div>"
        soup = BeautifulSoup(html, "html.parser")
        a = soup.find("a")
        assert _get_link_context(a) == "main"

    def test_footer_link(self):
        from bs4 import BeautifulSoup
        html = "<footer><a href='/privacy'>Privacy</a></footer>"
        soup = BeautifulSoup(html, "html.parser")
        a = soup.find("a")
        assert _get_link_context(a) == "footer"

    def test_nav_link(self):
        from bs4 import BeautifulSoup
        html = "<nav><ul><li><a href='/about'>About</a></li></ul></nav>"
        soup = BeautifulSoup(html, "html.parser")
        a = soup.find("a")
        assert _get_link_context(a) == "nav"

    def test_header_link(self):
        from bs4 import BeautifulSoup
        html = "<header><a href='/'>Home</a></header>"
        soup = BeautifulSoup(html, "html.parser")
        a = soup.find("a")
        assert _get_link_context(a) == "header"


class TestGetLinks:
    def test_extracts_same_domain_links(self):
        html = '<a href="/recipe/cookies">Cookies</a><a href="https://other.com/page">Other</a>'
        links = get_links(html, "https://example.com/page")
        assert "https://example.com/recipe/cookies" in links
        assert len(links) == 1

    def test_returns_anchor_text_and_context(self):
        html = '<div><a href="/recipe">My Recipe</a></div>'
        links = get_links(html, "https://example.com")
        link = links["https://example.com/recipe"]
        assert link["anchor_text"] == "My Recipe"
        assert link["context"] == "main"

    def test_skips_mailto_and_javascript(self):
        html = '<a href="mailto:test@example.com">Email</a><a href="javascript:void(0)">Click</a>'
        links = get_links(html, "https://example.com")
        assert len(links) == 0

    def test_skips_fragment_only_links(self):
        html = '<a href="#section">Jump</a>'
        links = get_links(html, "https://example.com")
        assert len(links) == 0

    def test_deduplicates_by_canonical_url(self):
        html = '<a href="/recipe?utm_source=a">Link 1</a><a href="/recipe?utm_source=b">Link 2</a>'
        links = get_links(html, "https://example.com")
        assert len(links) == 1


class TestGetRecipeData:
    def test_extracts_json_ld_recipe(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Recipe", "name": "Chocolate Cake", "recipeIngredient": ["flour", "sugar"], "recipeInstructions": ["Mix", "Bake"]}
        </script>
        </head><body></body></html>
        """
        recipe = get_recipe_data(html, "https://example.com/cake")
        assert recipe is not None
        assert recipe["title"] == "Chocolate Cake"
        assert recipe["url"] == "https://example.com/cake"
        assert "flour" in recipe["ingredients"]

    def test_extracts_nested_json_ld(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@graph": [{"@type": "Recipe", "name": "Pasta", "recipeIngredient": ["noodles"]}]}
        </script>
        </head><body></body></html>
        """
        recipe = get_recipe_data(html, "https://example.com/pasta")
        assert recipe is not None
        assert recipe["title"] == "Pasta"

    def test_returns_none_for_non_recipe(self):
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "Article", "name": "Blog Post"}
        </script>
        </head><body></body></html>
        """
        assert get_recipe_data(html, "https://example.com/blog") is None

    def test_extracts_microdata_recipe(self):
        html = """
        <html><body>
        <div itemscope itemtype="https://schema.org/Recipe">
            <span itemprop="name">Soup</span>
            <span itemprop="recipeIngredient">water</span>
            <span itemprop="recipeIngredient">salt</span>
        </div>
        </body></html>
        """
        recipe = get_recipe_data(html, "https://example.com/soup")
        assert recipe is not None
        assert recipe["title"] == "Soup"
        assert "water" in recipe["ingredients"]

    def test_returns_none_for_plain_html(self):
        html = "<html><body><p>Hello world</p></body></html>"
        assert get_recipe_data(html, "https://example.com") is None
