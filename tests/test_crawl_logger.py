import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_monster_crawl.crawl_logger import CrawlLogger


def read_events(filepath: str) -> list:
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestCrawlLoggerDiscover:
    def test_writes_correct_fields(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_discover("https://a.com/recipe", "https://a.com", "Pasta Recipe", 0.456789, {"a.com": 3})
        log.close()

        events = read_events(log.filepath)
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "discover"
        assert e["url"] == "https://a.com/recipe"
        assert e["source_url"] == "https://a.com"
        assert e["anchor_text"] == "Pasta Recipe"
        assert e["score"] == 0.456789
        assert e["domain_counts_snapshot"] == {"a.com": 3}

    def test_score_is_rounded_to_6_places(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_discover("https://a.com/recipe", "https://a.com", "", 0.123456789, {})
        log.close()

        e = read_events(log.filepath)[0]
        assert e["score"] == round(0.123456789, 6)

    def test_empty_domain_counts(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_discover("https://a.com/recipe", "https://a.com", "", 0.5, None)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["domain_counts_snapshot"] == {}


class TestCrawlLoggerVisit:
    def test_writes_correct_fields(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/recipe", 0.35, 5)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["type"] == "visit"
        assert e["url"] == "https://a.com/recipe"
        assert e["priority_at_fetch"] == 0.35
        assert e["pages_fetched_so_far"] == 5

    def test_priority_is_rounded_to_6_places(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/recipe", 0.123456789, 1)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["priority_at_fetch"] == round(0.123456789, 6)


class TestCrawlLoggerResult:
    def test_writes_correct_fields_for_recipe(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_result("https://a.com/recipe", True, "Pasta Carbonara", 12, 3, 10)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["type"] == "result"
        assert e["url"] == "https://a.com/recipe"
        assert e["is_recipe"] is True
        assert e["recipe_title"] == "Pasta Carbonara"
        assert e["links_discovered"] == 12
        assert e["cumulative_recipes"] == 3
        assert e["cumulative_pages"] == 10

    def test_writes_correct_fields_for_non_recipe(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_result("https://a.com/about", False, None, 5, 2, 8)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["is_recipe"] is False
        assert e["recipe_title"] is None

    def test_harvest_efficiency_calculation(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_result("https://a.com/r", True, "title", 0, 2, 4)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["harvest_efficiency"] == 50.0

    def test_harvest_efficiency_zero_when_no_pages(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_result("https://a.com/r", False, None, 0, 0, 0)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["harvest_efficiency"] == 0


class TestCrawlLoggerFilter:
    def test_writes_correct_fields_with_score(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_filter("https://a.com/junk", "score_threshold", 0.95)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["type"] == "filter"
        assert e["url"] == "https://a.com/junk"
        assert e["reason"] == "score_threshold"
        assert e["score"] == 0.95

    def test_score_key_absent_when_not_provided(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_filter("https://a.com/blocked", "robots_blocked")
        log.close()

        e = read_events(log.filepath)[0]
        assert e["reason"] == "robots_blocked"
        assert "score" not in e


class TestCrawlLoggerRescore:
    def test_writes_correct_fields(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_rescore("https://a.com/recipe", 0.3, 0.7)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["type"] == "rescore"
        assert e["url"] == "https://a.com/recipe"
        assert e["old_priority"] == 0.3
        assert e["new_priority"] == 0.7

    def test_priorities_are_rounded_to_6_places(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_rescore("https://a.com/recipe", 0.123456789, 0.987654321)
        log.close()

        e = read_events(log.filepath)[0]
        assert e["old_priority"] == round(0.123456789, 6)
        assert e["new_priority"] == round(0.987654321, 6)


class TestCrawlLoggerMetadata:
    def test_event_ids_are_sequential(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/r1", 0.3, 1)
        log.log_visit("https://a.com/r2", 0.4, 2)
        log.log_visit("https://a.com/r3", 0.5, 3)
        log.close()

        events = read_events(log.filepath)
        assert [e["event_id"] for e in events] == [0, 1, 2]

    def test_elapsed_secs_is_non_negative(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/r1", 0.3, 1)
        log.log_visit("https://a.com/r2", 0.4, 2)
        log.close()

        events = read_events(log.filepath)
        assert all(e["elapsed_secs"] >= 0 for e in events)

    def test_elapsed_secs_is_non_decreasing(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/r1", 0.3, 1)
        log.log_visit("https://a.com/r2", 0.4, 2)
        log.close()

        events = read_events(log.filepath)
        assert events[0]["elapsed_secs"] <= events[1]["elapsed_secs"]

    def test_each_line_is_valid_json(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_discover("https://a.com/r", "https://a.com", "title", 0.5, {})
        log.log_visit("https://a.com/r", 0.5, 1)
        log.log_result("https://a.com/r", True, "title", 5, 1, 1)
        log.close()

        with open(log.filepath) as f:
            for line in f:
                json.loads(line)  # raises if invalid

    def test_close_allows_file_to_be_read(self, tmp_path):
        log = CrawlLogger(output_dir=str(tmp_path))
        log.log_visit("https://a.com/r", 0.3, 1)
        log.close()

        # File is readable and contains the event after close
        events = read_events(log.filepath)
        assert len(events) == 1
