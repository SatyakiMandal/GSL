"""Tests for the per-company news cache (ceia/news_cache.py).

Mirrors the fixture style already used across the project's other tests:
small, explicit builders rather than a shared fixture file.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.models import NewsItem  # noqa: E402
from ceia.news_cache import (  # noqa: E402
    CacheSegment,
    CompanyCache,
    cache_path,
    cached_items_in_window,
    compute_gaps,
    load,
    save_after_gaps,
    slugify,
)

_SOURCES = ("economic_times", "moneycontrol")


def _item(url: str, day: str, headline: str = "headline") -> NewsItem:
    return NewsItem(
        source="economic_times", url=url, headline=headline,
        published_at=datetime.fromisoformat(f"{day}T10:00:00"),
    )


class TestSlugify:
    def test_lowercases_and_replaces_spaces(self):
        assert slugify("IndusInd Bank") == "indusind_bank"

    def test_strips_punctuation(self):
        assert slugify("Tata Consultancy Services Ltd.") == "tata_consultancy_services_ltd"

    def test_collapses_repeated_separators(self):
        assert slugify("  A   B  ") == "a_b"

    def test_empty_name_falls_back_to_a_placeholder(self):
        assert slugify("") == "company"
        assert slugify("!!!") == "company"


class TestLoadAndPersist:
    def test_missing_file_returns_an_empty_cache(self, tmp_path):
        cache = load("IndusInd Bank", tmp_path)
        assert cache.segments == []
        assert cache.items == []

    def test_corrupt_file_degrades_to_empty_not_a_crash(self, tmp_path):
        path = cache_path("IndusInd Bank", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        cache = load("IndusInd Bank", tmp_path)
        assert cache.segments == []
        assert cache.items == []

    def test_round_trips_segments_and_items_through_save_and_load(self, tmp_path):
        cache = CompanyCache(company="IndusInd Bank")
        items = [_item("https://x/1", "2025-03-11")]
        gaps = [(date(2025, 3, 1), date(2025, 3, 31))]
        save_after_gaps(cache, gaps, _SOURCES, items, tmp_path)

        reloaded = load("IndusInd Bank", tmp_path)
        assert len(reloaded.segments) == 1
        assert reloaded.segments[0].start == date(2025, 3, 1)
        assert reloaded.segments[0].end == date(2025, 3, 31)
        assert reloaded.segments[0].sources == _SOURCES
        assert len(reloaded.items) == 1
        assert reloaded.items[0].url == "https://x/1"
        assert reloaded.items[0].published_at == datetime.fromisoformat("2025-03-11T10:00:00")

    def test_different_company_spellings_get_different_files(self, tmp_path):
        assert cache_path("IndusInd Bank", tmp_path) != cache_path("IndusInd", tmp_path)


class TestComputeGaps:
    def test_no_segments_is_one_full_gap(self):
        gaps = compute_gaps(date(2025, 3, 1), date(2025, 3, 31), [], _SOURCES)
        assert gaps == [(date(2025, 3, 1), date(2025, 3, 31))]

    def test_fully_covered_by_a_satisfying_segment_has_no_gaps(self):
        segments = [CacheSegment(date(2025, 3, 1), date(2025, 3, 31), _SOURCES)]
        assert compute_gaps(date(2025, 3, 10), date(2025, 3, 17), segments, _SOURCES) == []

    def test_segment_with_insufficient_sources_does_not_satisfy(self):
        """A day cached under fewer sources than requested must still be
        treated as a gap - a source added since the cached run must not be
        silently missing from the result."""
        segments = [CacheSegment(date(2025, 3, 1), date(2025, 3, 31), ("economic_times",))]
        gaps = compute_gaps(date(2025, 3, 1), date(2025, 3, 31), segments, _SOURCES)
        assert gaps == [(date(2025, 3, 1), date(2025, 3, 31))]

    def test_segment_with_a_superset_of_requested_sources_satisfies(self):
        segments = [CacheSegment(date(2025, 3, 1), date(2025, 3, 31),
                                 ("economic_times", "moneycontrol", "business_line"))]
        assert compute_gaps(date(2025, 3, 1), date(2025, 3, 31),
                            segments, _SOURCES) == []

    def test_extending_a_cached_window_forward_leaves_only_the_new_tail(self):
        segments = [CacheSegment(date(2025, 3, 1), date(2025, 3, 20), _SOURCES)]
        gaps = compute_gaps(date(2025, 3, 1), date(2025, 3, 31), segments, _SOURCES)
        assert gaps == [(date(2025, 3, 21), date(2025, 3, 31))]

    def test_extending_a_cached_window_backward_leaves_only_the_new_head(self):
        segments = [CacheSegment(date(2025, 3, 10), date(2025, 3, 31), _SOURCES)]
        gaps = compute_gaps(date(2025, 3, 1), date(2025, 3, 31), segments, _SOURCES)
        assert gaps == [(date(2025, 3, 1), date(2025, 3, 9))]

    def test_gap_between_two_disjoint_cached_segments(self):
        segments = [
            CacheSegment(date(2025, 3, 1), date(2025, 3, 5), _SOURCES),
            CacheSegment(date(2025, 3, 20), date(2025, 3, 31), _SOURCES),
        ]
        gaps = compute_gaps(date(2025, 3, 1), date(2025, 3, 31), segments, _SOURCES)
        assert gaps == [(date(2025, 3, 6), date(2025, 3, 19))]

    def test_adjacent_segments_merge_with_no_residual_gap(self):
        segments = [
            CacheSegment(date(2025, 3, 1), date(2025, 3, 15), _SOURCES),
            CacheSegment(date(2025, 3, 16), date(2025, 3, 31), _SOURCES),
        ]
        assert compute_gaps(date(2025, 3, 1), date(2025, 3, 31), segments, _SOURCES) == []


class TestCachedItemsInWindow:
    def test_returns_items_inside_the_window(self):
        cache = CompanyCache(company="X", items=[_item("u1", "2025-03-11")])
        out = cached_items_in_window(cache, date(2025, 3, 1), date(2025, 3, 31), [])
        assert [i.url for i in out] == ["u1"]

    def test_excludes_items_outside_the_window(self):
        cache = CompanyCache(company="X", items=[_item("u1", "2025-02-01")])
        out = cached_items_in_window(cache, date(2025, 3, 1), date(2025, 3, 31), [])
        assert out == []

    def test_excludes_items_whose_day_falls_in_a_gap(self):
        """A day being re-crawled (e.g. because the source list changed)
        must not have its stale, narrower-source items mixed back in."""
        cache = CompanyCache(company="X", items=[_item("u1", "2025-03-11")])
        gaps = [(date(2025, 3, 10), date(2025, 3, 12))]
        out = cached_items_in_window(cache, date(2025, 3, 1), date(2025, 3, 31), gaps)
        assert out == []

    def test_excludes_items_with_no_publication_timestamp(self):
        item = NewsItem(source="et", url="u1", headline="h", published_at=None)
        cache = CompanyCache(company="X", items=[item])
        out = cached_items_in_window(cache, date(2025, 3, 1), date(2025, 3, 31), [])
        assert out == []


class TestSaveAfterGaps:
    def test_new_items_and_segment_are_recorded(self, tmp_path):
        cache = CompanyCache(company="X")
        gaps = [(date(2025, 3, 1), date(2025, 3, 31))]
        items = [_item("u1", "2025-03-11")]
        save_after_gaps(cache, gaps, _SOURCES, items, tmp_path)
        assert [i.url for i in cache.items] == ["u1"]
        assert cache.segments == [CacheSegment(date(2025, 3, 1), date(2025, 3, 31), _SOURCES)]

    def test_stale_items_in_a_re_crawled_gap_are_dropped_even_if_replaced_by_fewer(self, tmp_path):
        """A fuller, wider-source re-crawl of a previously narrow-source day
        finding fewer (or zero) matching items must still replace the old
        entries - it must not look like the old items are still current."""
        cache = CompanyCache(
            company="X",
            segments=[CacheSegment(date(2025, 3, 1), date(2025, 3, 31), ("economic_times",))],
            items=[_item("stale", "2025-03-11")],
        )
        gaps = [(date(2025, 3, 1), date(2025, 3, 31))]
        save_after_gaps(cache, gaps, _SOURCES, [], tmp_path)
        assert cache.items == []

    def test_items_outside_the_gap_are_preserved(self, tmp_path):
        cache = CompanyCache(
            company="X",
            segments=[CacheSegment(date(2025, 3, 1), date(2025, 3, 31), _SOURCES)],
            items=[_item("kept", "2025-03-05")],
        )
        gaps = [(date(2025, 4, 1), date(2025, 4, 5))]
        save_after_gaps(cache, gaps, _SOURCES, [_item("new", "2025-04-02")], tmp_path)
        assert {i.url for i in cache.items} == {"kept", "new"}

    def test_same_url_from_a_new_crawl_replaces_the_cached_copy(self, tmp_path):
        cache = CompanyCache(
            company="X",
            segments=[CacheSegment(date(2025, 3, 1), date(2025, 3, 31), _SOURCES)],
            items=[_item("u1", "2025-03-11", headline="old headline")],
        )
        gaps = [(date(2025, 3, 1), date(2025, 3, 31))]
        save_after_gaps(cache, gaps, _SOURCES,
                        [_item("u1", "2025-03-11", headline="new headline")], tmp_path)
        assert len(cache.items) == 1
        assert cache.items[0].headline == "new headline"

    def test_adjacent_same_source_segments_coalesce(self, tmp_path):
        cache = CompanyCache(
            company="X",
            segments=[CacheSegment(date(2025, 3, 1), date(2025, 3, 15), _SOURCES)],
        )
        gaps = [(date(2025, 3, 16), date(2025, 3, 31))]
        save_after_gaps(cache, gaps, _SOURCES, [], tmp_path)
        assert cache.segments == [CacheSegment(date(2025, 3, 1), date(2025, 3, 31), _SOURCES)]

    def test_segments_with_different_sources_do_not_coalesce(self, tmp_path):
        cache = CompanyCache(
            company="X",
            segments=[CacheSegment(date(2025, 3, 1), date(2025, 3, 15), ("economic_times",))],
        )
        gaps = [(date(2025, 3, 16), date(2025, 3, 31))]
        save_after_gaps(cache, gaps, _SOURCES, [], tmp_path)
        assert len(cache.segments) == 2
