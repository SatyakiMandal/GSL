"""Tests for ceia.analyze.ingest_with_cache - the orchestration that wires
the per-company news cache (ceia/news_cache.py) into a live-scraping run:
crawl only the date gaps not already cached, then combine with whatever was
already cached for the rest of the requested window.

ceia.ingest.run() itself is not re-tested here (see test_ingest.py) - only
that ingest_with_cache calls it for the right date sub-ranges and combines
the result correctly. ceia.analyze.run_ingest is monkeypatched to a fake
that returns one NewsItem per day in the requested sub-range, so which days
were (not) re-crawled is directly observable from the returned items' URLs.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ceia.analyze as analyze_mod  # noqa: E402
from ceia.analyze import ingest_with_cache  # noqa: E402
from ceia.ingest import IngestResult  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402


class _FakeIngest:
    """Stands in for ceia.ingest.run(): one item per calendar day in the
    requested sub-range, and records every (start, end) it was called
    with, so a test can assert exactly which gaps were (not) re-crawled."""

    def __init__(self):
        self.calls: list[tuple[date, date]] = []

    def __call__(self, config: RunConfig, fetcher=None, limit=None, max_workers=8,
                skip_alias_widening=False, skip_slug_prefilter=False) -> IngestResult:
        self.calls.append((config.start, config.end))
        items = []
        day = config.start
        while day <= config.end:
            items.append(NewsItem(
                source="economic_times", url=f"https://x/{day.isoformat()}",
                headline=f"headline {day.isoformat()}",
                published_at=datetime.combine(day, datetime.min.time()),
            ))
            day += timedelta(days=1)
        return IngestResult(config=config, items=items,
                            source_status={"economic_times": "ok"})


def _config(start: str, end: str, sources=("economic_times",)) -> RunConfig:
    return RunConfig(company="IndusInd Bank", ticker="INDUSINDBK.NS",
                     start=date.fromisoformat(start), end=date.fromisoformat(end),
                     sources=list(sources))


class TestIngestWithCache:
    def test_first_run_for_a_company_crawls_the_whole_range(self, monkeypatch, tmp_path):
        fake = _FakeIngest()
        monkeypatch.setattr(analyze_mod, "run_ingest", fake)
        config = _config("2025-03-01", "2025-03-31")
        items, meta = ingest_with_cache(config, fetcher=object(), cache_dir=tmp_path)

        assert fake.calls == [(date(2025, 3, 1), date(2025, 3, 31))]
        assert len(items) == 31
        assert meta["stats"]["unique_after_dedupe"] == 31
        assert meta["news_cache"]["cached_days_reused"] == 0

    def test_a_folder_named_for_the_company_is_created(self, tmp_path):
        fake = _FakeIngest()
        config = _config("2025-03-01", "2025-03-05")
        import ceia.news_cache as news_cache_mod
        expected = news_cache_mod.cache_path("IndusInd Bank", tmp_path)
        # sanity: cache_path lives under tmp_path and is named for the company
        assert expected.parent == tmp_path
        assert "indusind_bank" in expected.name

    def test_a_fully_cached_subset_range_makes_no_new_calls(self, monkeypatch, tmp_path):
        fake = _FakeIngest()
        monkeypatch.setattr(analyze_mod, "run_ingest", fake)
        ingest_with_cache(_config("2025-03-01", "2025-03-31"), fetcher=object(),
                          cache_dir=tmp_path)

        fake.calls.clear()
        items, meta = ingest_with_cache(_config("2025-03-10", "2025-03-17"),
                                        fetcher=object(), cache_dir=tmp_path)

        assert fake.calls == []
        assert len(items) == 8
        assert {i.url for i in items} == {
            f"https://x/2025-03-{d:02d}" for d in range(10, 18)
        }
        assert meta["news_cache"]["cached_days_reused"] == 8

    def test_extending_the_range_only_crawls_the_new_tail(self, monkeypatch, tmp_path):
        fake = _FakeIngest()
        monkeypatch.setattr(analyze_mod, "run_ingest", fake)
        ingest_with_cache(_config("2025-03-01", "2025-03-31"), fetcher=object(),
                          cache_dir=tmp_path)

        fake.calls.clear()
        items, meta = ingest_with_cache(_config("2025-03-01", "2025-04-05"),
                                        fetcher=object(), cache_dir=tmp_path)

        assert fake.calls == [(date(2025, 4, 1), date(2025, 4, 5))]
        assert len(items) == 36
        assert meta["news_cache"]["cached_days_reused"] == 31

    def test_a_different_source_list_forces_a_full_re_crawl(self, monkeypatch, tmp_path):
        fake = _FakeIngest()
        monkeypatch.setattr(analyze_mod, "run_ingest", fake)
        ingest_with_cache(_config("2025-03-01", "2025-03-31", sources=("economic_times",)),
                          fetcher=object(), cache_dir=tmp_path)

        fake.calls.clear()
        items, meta = ingest_with_cache(
            _config("2025-03-01", "2025-03-31", sources=("economic_times", "moneycontrol")),
            fetcher=object(), cache_dir=tmp_path)

        assert fake.calls == [(date(2025, 3, 1), date(2025, 3, 31))]
        assert meta["news_cache"]["cached_days_reused"] == 0

    def test_repeated_runs_persist_across_separate_calls_via_disk(self, monkeypatch, tmp_path):
        """The cache must actually round-trip through disk, not just live in
        an in-memory object passed between calls - each ingest_with_cache
        call loads fresh from cache_dir."""
        fake = _FakeIngest()
        monkeypatch.setattr(analyze_mod, "run_ingest", fake)
        ingest_with_cache(_config("2025-03-01", "2025-03-10"), fetcher=object(),
                          cache_dir=tmp_path)
        fake.calls.clear()

        items, meta = ingest_with_cache(_config("2025-03-01", "2025-03-10"),
                                        fetcher=object(), cache_dir=tmp_path)
        assert fake.calls == []
        assert len(items) == 10
