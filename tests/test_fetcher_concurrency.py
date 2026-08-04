"""Concurrency-correctness tests for Fetcher's per-origin locking.

``discover()`` and ``fetch_and_parse()`` now drive Fetcher from a thread
pool. The one property that must hold under that or the crawl stops being
polite: two threads hitting the SAME origin must never both slip past the
rate limit at once. These tests prove that directly, using a fake session
that records exactly when each call started and finished, rather than
trusting that "it worked once" means the lock is correct.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.fetcher import Fetcher  # noqa: E402


class _FakeResponse:
    def __init__(self, url: str) -> None:
        self.url = url
        self.status_code = 200
        self.text = "<html></html>"


class _RecordingSession:
    """Stands in for requests.Session; records (url, start, end) per call."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls: list[tuple[str, float, float]] = []
        self._record_lock = threading.Lock()
        self.headers: dict = {}

    def get(self, url: str, timeout=None, allow_redirects=None):
        start = time.monotonic()
        time.sleep(self.delay)
        end = time.monotonic()
        with self._record_lock:
            self.calls.append((url, start, end))
        return _FakeResponse(url)


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@pytest.fixture
def fetcher(tmp_path: Path) -> tuple[Fetcher, _RecordingSession]:
    f = Fetcher(cache_dir=tmp_path, min_interval=0.15, obey_robots=False, max_retries=1)
    session = _RecordingSession(delay=0.05)
    f._session = session
    return f, session


class TestSameOriginIsSerialised:
    def test_concurrent_requests_to_one_origin_never_overlap(self, fetcher):
        f, session = fetcher
        urls = [f"https://same-origin.example.com/article-{i}" for i in range(8)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(f.get, urls))

        assert len(session.calls) == 8
        intervals = sorted(((c[1], c[2]) for c in session.calls), key=lambda t: t[0])
        for a, b in zip(intervals, intervals[1:]):
            assert not _intervals_overlap(a, b), (
                "two requests to the same origin overlapped in time - "
                "the rate limit was violated under concurrency"
            )

    def test_concurrent_requests_respect_min_interval_spacing(self, fetcher):
        f, session = fetcher
        urls = [f"https://same-origin.example.com/spacing-{i}" for i in range(5)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(f.get, urls))

        starts = sorted(c[1] for c in session.calls)
        for earlier, later in zip(starts, starts[1:]):
            # Small tolerance for scheduling jitter, not for a real violation.
            assert later - earlier >= f.min_interval - 0.02


class TestDifferentOriginsRunConcurrently:
    def test_two_origins_overlap_in_time(self, fetcher):
        """The other half of the property: the lock must not become a single
        global lock that accidentally serialises everything."""
        f, session = fetcher
        a_urls = [f"https://origin-a.example.com/{i}" for i in range(4)]
        b_urls = [f"https://origin-b.example.com/{i}" for i in range(4)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(f.get, a_urls + b_urls))

        a_intervals = [(c[1], c[2]) for c in session.calls if "origin-a" in c[0]]
        b_intervals = [(c[1], c[2]) for c in session.calls if "origin-b" in c[0]]
        assert any(
            _intervals_overlap(a, b) for a in a_intervals for b in b_intervals
        ), "requests to two independent origins never overlapped - looks fully serialised"


class TestRobotsCacheIsRaceFree:
    def test_concurrent_first_use_fetches_robots_once(self, tmp_path: Path):
        """Many threads' first request to a never-seen origin should not each
        fire their own robots.txt fetch - the double-checked lock exists to
        prevent exactly that."""
        f = Fetcher(cache_dir=tmp_path, min_interval=0.05, max_retries=1)
        session = _RecordingSession(delay=0.02)
        f._session = session
        urls = [f"https://robots-race.example.com/page-{i}" for i in range(10)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(f.get, urls))

        robots_calls = [c for c in session.calls if c[0].endswith("/robots.txt")]
        assert len(robots_calls) == 1
