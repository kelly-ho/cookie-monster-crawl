import pytest
from cookie_monster_crawl.utils import URLPrioritizer


@pytest.fixture
def prioritizer():
    return URLPrioritizer()


class TestExtractFeatures:
    def test_returns_dict_with_expected_keys(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/recipes/chocolate-cake")
        assert "path_depth" in features
        assert "leaf_word_count" in features
        assert "domain_harvest_rate" in features

    def test_file_extension_returns_empty(self, prioritizer):
        assert prioritizer.extract_features("https://example.com/photo.jpg") == {}
        assert prioritizer.extract_features("https://example.com/doc.pdf") == {}

    def test_path_depth(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/a/b/c")
        assert features["path_depth"] == 3

    def test_leaf_word_count(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/best-chocolate-chip-cookies")
        assert features["leaf_word_count"] == 4

    def test_infrastructure_leaf(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/privacy")
        assert features["leaf_is_infrastructure"] == 1

    def test_navigational_leaf(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/category")
        assert features["leaf_is_navigational"] == 1

    def test_recipe_related_leaf(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/recipes")
        assert features["leaf_is_recipe_related"] == 1

    def test_anchor_word_count(self, prioritizer):
        features = prioritizer.extract_features(
            "https://example.com/page", anchor_text="See the full recipe"
        )
        assert features["anchor_word_count"] == 4

    def test_anchor_has_recipe_keyword(self, prioritizer):
        features = prioritizer.extract_features(
            "https://example.com/page", anchor_text="Get the recipe"
        )
        assert features["anchor_has_recipe_keyword"] == 1

    def test_print_page_detection(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/recipe/print")
        assert features["is_print_or_wprm"] == 1

    def test_has_numeric_id(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/post/12345")
        assert features["has_numeric_id"] == 1

    def test_query_param_count(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/search?q=cookies&page=2")
        assert features["query_param_count"] == 2

    def test_underscore_normalization(self, prioritizer):
        features = prioritizer.extract_features("https://example.com/my_recipe")
        assert features["leaf_word_count"] == 2


class TestIsRoundupSlug:
    def test_roundup_word_with_plural(self, prioritizer):
        assert prioritizer._is_roundup_slug("best-cookies", ["best", "cookies"])

    def test_leading_number_with_plural(self, prioritizer):
        assert prioritizer._is_roundup_slug("75-christmas-cookies", ["75", "christmas", "cookies"])

    def test_single_recipe_not_roundup(self, prioritizer):
        assert not prioritizer._is_roundup_slug("chocolate-cake", ["chocolate", "cake"])

    def test_roundup_pattern(self, prioritizer):
        assert prioritizer._is_roundup_slug("easy-recipes-for-dinner", ["easy", "recipes", "for", "dinner"])

    def test_three_ingredient_not_roundup(self, prioritizer):
        assert not prioritizer._is_roundup_slug("3-ingredient-pasta", ["3", "ingredient", "pasta"])


class TestCalculateScore:
    def test_returns_tuple_of_three(self, prioritizer):
        score, components, features = prioritizer.calculate_score("https://example.com/recipe")
        assert isinstance(score, float)
        assert isinstance(components, dict)
        assert isinstance(features, dict)

    def test_file_extension_gets_high_score(self, prioritizer):
        score, _, _ = prioritizer.calculate_score("https://example.com/photo.jpg")
        assert score == 0.99

    def test_score_between_0_and_1(self, prioritizer):
        score, _, _ = prioritizer.calculate_score("https://example.com/recipes/chocolate-cake")
        assert 0 <= score <= 1

    def test_infrastructure_scores_higher(self, prioritizer):
        recipe_score, _, _ = prioritizer.calculate_score("https://example.com/chocolate-cake")
        infra_score, _, _ = prioritizer.calculate_score("https://example.com/privacy")
        assert infra_score > recipe_score

    def test_model_scoring_when_loaded(self, tmp_path):
        import pickle
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from cookie_monster_crawl.train import FEATURE_NAMES

        n_features = len(FEATURE_NAMES)
        X = np.random.rand(100, n_features)
        y = np.random.randint(0, 2, 100)
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": FEATURE_NAMES}, f)

        p = URLPrioritizer(model_path=str(model_path))
        score, components, features = p.calculate_score("https://example.com/recipe")
        assert 0 <= score <= 1
        assert components == {}


class TestDomainStats:
    def test_save_and_load(self, prioritizer, tmp_path):
        prioritizer.update_model("https://example.com/recipe/cake", is_recipe=True)
        prioritizer.update_model("https://example.com/recipe/pie", is_recipe=True)
        prioritizer.update_model("https://example.com/about", is_recipe=False)

        filepath = str(tmp_path / "stats.json")
        prioritizer.save_domain_stats(filepath)

        new_prioritizer = URLPrioritizer()
        new_prioritizer.load_domain_stats(filepath)
        assert "example.com" in new_prioritizer.domain_path_stats

    def test_load_nonexistent_file(self, prioritizer):
        prioritizer.load_domain_stats("/nonexistent/path.json")

    def test_harvest_rate_cold_start(self, prioritizer):
        rate = prioritizer._domain_harvest_rate("unknown.com")
        assert rate == 0.5
