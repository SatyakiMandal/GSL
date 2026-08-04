"""Concurrency-correctness tests for ingest.fetch_and_parse().

It now runs candidates through a thread pool instead of one at a time, with
items/errors/seen shared across worker threads under one lock. These tests
prove there is no lost update under concurrency - every candidate must be
accounted for exactly once, with the same final counts a sequential run
would have produced - rather than trusting that a single run "looked right".
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ceia.ingest as ingest_module  # noqa: E402
from ceia.discovery import Candidate  # noqa: E402
from ceia.fetcher import RobotsDisallowed  # noqa: E402
from ceia.ingest import fetch_and_parse  # noqa: E402


class _FakeResponse:
    def __init__(self, url: str, status: int = 200) -> None:
        self.url = url
        self.final_url = url
        self.status = status
        self.text = "<html></html>"
        self.from_cache = False
        self.fetched_at = "2025-01-01T00:00:00+00:00"
        self.sha256 = "deadbeef"


class _FakeFetcher:
    """Every URL "succeeds" after a tiny simulated network delay, so many
    worker threads are actually in flight together, giving races a real
    chance to happen if the locking were wrong."""

    def __init__(self, delay: float = 0.01, robots_disallow: set[str] | None = None,
                http_fail: set[str] | None = None) -> None:
        self.delay = delay
        self.robots_disallow = robots_disallow or set()
        self.http_fail = http_fail or set()
        self.call_count = 0
        self._lock = threading.Lock()

    def get(self, url: str) -> _FakeResponse:
        with self._lock:
            self.call_count += 1
        time.sleep(self.delay)
        if url in self.robots_disallow:
            raise RobotsDisallowed(f"{url}: disallowed")
        if url in self.http_fail:
            return _FakeResponse(url, status=500)
        return _FakeResponse(url)


@pytest.fixture(autouse=True)
def _fake_parse_article(monkeypatch):
    """Isolate concurrency correctness from real HTML parsing."""
    def fake_parse(html: str, url: str, source: str) -> dict:
        return {
            "url": url,
            "source": source,
            "headline": f"headline for {url}",
            "published_at": None,
            "timestamp_confidence": "missing",
            "body": "body text",
            "snippet": "snippet",
            "paywalled": False,
        }
    monkeypatch.setattr(ingest_module, "parse_article", fake_parse)


class TestFetchAndParseConcurrency:
    def test_every_candidate_is_accounted_for_exactly_once(self):
        candidates = [Candidate(f"https://x.example.com/a{i}", "src") for i in range(60)]
        fetcher = _FakeFetcher(delay=0.01)

        items, errors = fetch_and_parse(fetcher, candidates, max_workers=12)

        assert len(items) == 60
        assert {i.url for i in items} == {c.url for c in candidates}
        assert sum(errors.values()) == 0
        assert fetcher.call_count == 60

    def test_duplicate_urls_are_fetched_only_once(self):
        dup = Candidate("https://x.example.com/dup", "src")
        candidates = [dup] * 20 + [Candidate(f"https://x.example.com/b{i}", "src")
                                   for i in range(20)]
        fetcher = _FakeFetcher(delay=0.005)

        items, errors = fetch_and_parse(fetcher, candidates, max_workers=16)

        assert len(items) == 21  # one "dup" plus the 20 unique b-candidates
        assert len({i.url for i in items}) == 21

    def test_error_kinds_are_all_counted_correctly_under_concurrency(self):
        candidates = [Candidate(f"https://x.example.com/ok{i}", "src") for i in range(15)]
        robots_urls = {f"https://x.example.com/robots{i}" for i in range(10)}
        http_fail_urls = {f"https://x.example.com/fail{i}" for i in range(8)}
        candidates += [Candidate(u, "src") for u in robots_urls]
        candidates += [Candidate(u, "src") for u in http_fail_urls]
        fetcher = _FakeFetcher(delay=0.005, robots_disallow=robots_urls, http_fail=http_fail_urls)

        items, errors = fetch_and_parse(fetcher, candidates, max_workers=10)

        assert len(items) == 15
        assert errors["robots"] == 10
        assert errors["http"] == 8
        assert errors["parse"] == 0 and errors["empty"] == 0

    def test_limit_caps_final_items_count_under_concurrency(self):
        """Concurrency means several requests can be in flight when the limit
        is hit, so a handful more than `limit` may be *attempted* - but the
        kept item count must never exceed it."""
        candidates = [Candidate(f"https://x.example.com/c{i}", "src") for i in range(200)]
        fetcher = _FakeFetcher(delay=0.005)

        items, errors = fetch_and_parse(fetcher, candidates, limit=25, max_workers=16)

        assert len(items) <= 25
        assert len(items) > 0

    def test_repeated_runs_are_stable(self):
        """Race conditions are often intermittent - run several times rather
        than trusting one green run."""
        candidates = [Candidate(f"https://x.example.com/r{i}", "src") for i in range(40)]
        for _ in range(5):
            fetcher = _FakeFetcher(delay=0.002)
            items, errors = fetch_and_parse(fetcher, candidates, max_workers=10)
            assert len(items) == 40
            assert sum(errors.values()) == 0
