"""Tests for the Wayback Machine fallback (ceia/wayback.py).

Business Standard and Mint were re-investigated on an explicit user request
after both were previously found unreachable for a historical date range -
see the module docstring for the live findings (a same-day archived
snapshot of the Adani/Hindenburg topic page on both sites, and a second,
independent spot-check showing coverage is genuinely inconsistent for less
prominent companies). These tests pin the mechanics against a fake fetcher,
not live archive.org - see docs/ or the README for how that was verified.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.discovery import Candidate  # noqa: E402
from ceia.wayback import (  # noqa: E402
    _AVAILABILITY_URL,
    _extract_article_links,
    _probe_dates,
    _topic_slugs,
    closest_snapshot,
    discover_topic_candidates,
    fetch_and_parse_via_wayback,
)


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status = status
        self.fetched_at = "2023-01-26T00:00:00+00:00"
        self.sha256 = "deadbeef"


class _FakeWaybackFetcher:
    """Routes by URL: availability-API calls are answered from ``availability``
    (keyed by the exact target URL passed in the query string), everything
    else from ``pages`` (keyed by the exact URL requested)."""

    def __init__(self, availability: dict[str, dict] | None = None,
                 pages: dict[str, _FakeResponse] | None = None,
                 error_urls: set[str] | None = None):
        self.availability = availability or {}
        self.pages = pages or {}
        self.error_urls = error_urls or set()
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeResponse:
        self.calls.append(url)
        if url in self.error_urls:
            raise RuntimeError("simulated network failure")
        if url.startswith(_AVAILABILITY_URL):
            target = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["url"][0]
            payload = self.availability.get(
                target, {"url": target, "archived_snapshots": {}})
            return _FakeResponse(json.dumps(payload))
        if url not in self.pages:
            raise RuntimeError(f"no fake page for {url}")
        return self.pages[url]


def _availability_payload(target: str, timestamp: str, available: bool = True) -> dict:
    return {
        "url": target,
        "archived_snapshots": {
            "closest": {"status": "200", "available": available,
                       "url": f"http://web.archive.org/web/{timestamp}/{target}",
                       "timestamp": timestamp},
        } if available else {},
    }


_ARTICLE_HTML = """<html><head>
<script type="application/ld+json">
{"@type": "NewsArticle", "headline": "Adani Group says allegations baseless",
 "datePublished": "2023-01-25T22:05:00+05:30",
 "articleBody": "Adani Group called the Hindenburg report stale and baseless. """ + (
    "More detail. " * 20) + """"}
</script></head><body></body></html>"""


class TestClosestSnapshot:
    def test_returns_playback_url_and_capture_date(self):
        target = "https://www.business-standard.com/topic/adani-group"
        fetcher = _FakeWaybackFetcher(availability={
            target: _availability_payload(target, "20230125203302"),
        })
        result = closest_snapshot(fetcher, target, date(2023, 1, 25))
        assert result is not None
        playback_url, snap_date = result
        assert playback_url == f"https://web.archive.org/web/20230125203302/{target}"
        assert snap_date == date(2023, 1, 25)

    def test_no_snapshot_returns_none(self):
        target = "https://www.livemint.com/topic/some-obscure-company"
        fetcher = _FakeWaybackFetcher(availability={
            target: _availability_payload(target, "", available=False),
        })
        assert closest_snapshot(fetcher, target, date(2023, 1, 25)) is None

    def test_unavailable_flag_is_honoured(self):
        target = "https://www.business-standard.com/topic/x"
        payload = _availability_payload(target, "20230101000000")
        payload["archived_snapshots"]["closest"]["available"] = False
        fetcher = _FakeWaybackFetcher(availability={target: payload})
        assert closest_snapshot(fetcher, target, date(2023, 1, 25)) is None

    def test_fetcher_exception_degrades_to_none(self):
        target = "https://www.business-standard.com/topic/x"
        fetcher = _FakeWaybackFetcher(error_urls={f"{_AVAILABILITY_URL}?url={target}"})
        # closest_snapshot builds its own query string, so patch via a
        # fetcher whose .get always raises instead of matching exactly.
        class _AlwaysFails:
            def get(self, url):
                raise RuntimeError("boom")
        assert closest_snapshot(_AlwaysFails(), target, date(2023, 1, 25)) is None

    def test_malformed_json_degrades_to_none(self):
        class _BadJson:
            def get(self, url):
                return _FakeResponse("not json")
        assert closest_snapshot(_BadJson(), "https://x.example/", date(2023, 1, 25)) is None


class TestTopicSlugs:
    def test_lowercases_and_hyphenates(self):
        assert _topic_slugs(["Adani Enterprises"]) == ["adani-enterprises"]

    def test_dedups_case_insensitively(self):
        assert _topic_slugs(["Paytm", "PAYTM", "paytm"]) == ["paytm"]

    def test_preserves_order_and_multiple_names(self):
        assert _topic_slugs(["Adani Enterprises", "Adani Group"]) == [
            "adani-enterprises", "adani-group"]

    def test_empty_or_symbol_only_names_are_dropped(self):
        assert _topic_slugs(["", "---", "&"]) == []


class TestProbeDates:
    def test_single_day_window_yields_one_probe(self):
        assert _probe_dates(date(2023, 1, 25), date(2023, 1, 25), 7) == [date(2023, 1, 25)]

    def test_always_includes_both_ends(self):
        probes = _probe_dates(date(2023, 1, 1), date(2023, 1, 20), 7)
        assert probes[0] == date(2023, 1, 1)
        assert probes[-1] == date(2023, 1, 20)

    def test_spacing_matches_every_days(self):
        probes = _probe_dates(date(2023, 1, 1), date(2023, 1, 15), 7)
        assert probes == [date(2023, 1, 1), date(2023, 1, 8), date(2023, 1, 15)]


class TestExtractArticleLinks:
    def test_extracts_wayback_prefixed_links(self):
        html = (
            '<a href="/web/20230125203302/https://www.business-standard.com/'
            'article/companies/some-story-123012501585_1.html">Story</a>'
        )
        links = _extract_article_links(html, "business_standard")
        assert links == [
            "https://www.business-standard.com/article/companies/"
            "some-story-123012501585_1.html"
        ]

    def test_extracts_bare_unprefixed_links_too(self):
        html = (
            '<link rel="canonical" href="https://www.livemint.com/companies/'
            'news/some-story-11674710954428.html"/>'
        )
        links = _extract_article_links(html, "livemint")
        assert links == [
            "https://www.livemint.com/companies/news/some-story-11674710954428.html"
        ]

    def test_ignores_non_article_nav_links(self):
        html = '<a href="/web/20230125203302/https://www.business-standard.com/markets">Markets</a>'
        assert _extract_article_links(html, "business_standard") == []

    def test_dedups(self):
        html = (
            '<a href="/web/20230125203302/https://www.livemint.com/x/y-11674710954428.html">a</a>'
            '<a href="/web/20230125210000/https://www.livemint.com/x/y-11674710954428.html">b</a>'
        )
        assert len(_extract_article_links(html, "livemint")) == 1


class TestDiscoverTopicCandidates:
    def _topic_html(self, source_key: str, article_url: str, timestamp: str) -> str:
        return f'<a href="/web/{timestamp}/{article_url}">Story</a>'

    def test_finds_candidates_from_an_archived_topic_page(self):
        origin = "https://www.business-standard.com"
        topic_url = f"{origin}/topic/adani-group"
        article = (f"{origin}/article/companies/"
                   "hindenburg-accuses-adani-123012501585_1.html")
        playback = f"https://web.archive.org/web/20230125203302/{topic_url}"
        fetcher = _FakeWaybackFetcher(
            availability={topic_url: _availability_payload(topic_url, "20230125203302")},
            pages={playback: _FakeResponse(self._topic_html(
                "business_standard", article, "20230125203302"))},
        )
        candidates, status = discover_topic_candidates(
            fetcher, "business_standard", origin, ["Adani Group"],
            date(2023, 1, 25), date(2023, 1, 25))
        assert len(candidates) == 1
        assert candidates[0].url == article
        assert candidates[0].source == "business_standard"
        assert "1 archived topic-page snapshot" in status

    def test_snapshot_outside_margin_is_discarded(self):
        origin = "https://www.livemint.com"
        topic_url = f"{origin}/topic/some-company"
        # A capture more than 30 days from the window tells us nothing
        # about this window's coverage.
        fetcher = _FakeWaybackFetcher(
            availability={topic_url: _availability_payload(topic_url, "20220101000000")},
        )
        candidates, status = discover_topic_candidates(
            fetcher, "livemint", origin, ["Some Company"],
            date(2023, 1, 25), date(2023, 1, 25))
        assert candidates == []
        assert "no archived topic-page snapshot" in status

    def test_no_snapshot_at_all_degrades_to_empty_not_a_crash(self):
        origin = "https://www.livemint.com"
        fetcher = _FakeWaybackFetcher()  # no availability entries at all
        candidates, status = discover_topic_candidates(
            fetcher, "livemint", origin, ["Nonexistent Co"],
            date(2023, 1, 25), date(2023, 1, 25))
        assert candidates == []
        assert "no archived topic-page snapshot" in status

    def test_fetch_failure_after_a_real_snapshot_is_skipped_gracefully(self):
        origin = "https://www.business-standard.com"
        topic_url = f"{origin}/topic/adani-group"
        playback = f"https://web.archive.org/web/20230125203302/{topic_url}"
        fetcher = _FakeWaybackFetcher(
            availability={topic_url: _availability_payload(topic_url, "20230125203302")},
            error_urls={playback},
        )
        candidates, status = discover_topic_candidates(
            fetcher, "business_standard", origin, ["Adani Group"],
            date(2023, 1, 25), date(2023, 1, 25))
        assert candidates == []
        assert "no archived topic-page snapshot" in status

    def test_dedups_across_multiple_slugs(self):
        origin = "https://www.business-standard.com"
        article = f"{origin}/article/companies/x-123012501585_1.html"
        topic1 = f"{origin}/topic/adani-group"
        topic2 = f"{origin}/topic/adani-enterprises"
        playback1 = f"https://web.archive.org/web/20230125203302/{topic1}"
        playback2 = f"https://web.archive.org/web/20230125203309/{topic2}"
        fetcher = _FakeWaybackFetcher(
            availability={
                topic1: _availability_payload(topic1, "20230125203302"),
                topic2: _availability_payload(topic2, "20230125203309"),
            },
            pages={
                playback1: _FakeResponse(self._topic_html("x", article, "20230125203302")),
                playback2: _FakeResponse(self._topic_html("x", article, "20230125203309")),
            },
        )
        candidates, _ = discover_topic_candidates(
            fetcher, "business_standard", origin, ["Adani Group", "Adani Enterprises"],
            date(2023, 1, 25), date(2023, 1, 25))
        assert len(candidates) == 1


class TestFetchAndParseViaWayback:
    def test_successful_fetch_produces_a_news_item_with_the_canonical_url(self):
        canonical = ("https://www.business-standard.com/article/companies/"
                     "hindenburg-accuses-adani-123012501585_1.html")
        playback = f"https://web.archive.org/web/20230125182200/{canonical}"
        fetcher = _FakeWaybackFetcher(
            availability={canonical: _availability_payload(canonical, "20230125182200")},
            pages={playback: _FakeResponse(_ARTICLE_HTML)},
        )
        candidate = Candidate(canonical, "business_standard", None)
        items, errors = fetch_and_parse_via_wayback(fetcher, [candidate], date(2023, 1, 25))
        assert len(items) == 1
        assert items[0].url == canonical  # not the wayback-wrapped URL
        assert items[0].headline == "Adani Group says allegations baseless"
        assert sum(errors.values()) == 0

    def test_no_snapshot_counts_as_unavailable(self):
        canonical = "https://www.business-standard.com/article/companies/no-snapshot.html"
        fetcher = _FakeWaybackFetcher()  # nothing archived
        candidate = Candidate(canonical, "business_standard", None)
        items, errors = fetch_and_parse_via_wayback(fetcher, [candidate], date(2023, 1, 25))
        assert items == []
        assert errors["unavailable"] == 1

    def test_non_200_snapshot_counts_as_http_error(self):
        canonical = "https://www.business-standard.com/article/companies/broken.html"
        playback = f"https://web.archive.org/web/20230125182200/{canonical}"
        fetcher = _FakeWaybackFetcher(
            availability={canonical: _availability_payload(canonical, "20230125182200")},
            pages={playback: _FakeResponse("", status=404)},
        )
        candidate = Candidate(canonical, "business_standard", None)
        items, errors = fetch_and_parse_via_wayback(fetcher, [candidate], date(2023, 1, 25))
        assert items == []
        assert errors["http"] == 1

    def test_empty_headline_counts_as_empty(self):
        canonical = "https://www.business-standard.com/article/companies/blank.html"
        playback = f"https://web.archive.org/web/20230125182200/{canonical}"
        fetcher = _FakeWaybackFetcher(
            availability={canonical: _availability_payload(canonical, "20230125182200")},
            pages={playback: _FakeResponse("<html><body>no metadata here</body></html>")},
        )
        candidate = Candidate(canonical, "business_standard", None)
        items, errors = fetch_and_parse_via_wayback(fetcher, [candidate], date(2023, 1, 25))
        assert items == []
        assert errors["empty"] == 1

    def test_candidate_hint_date_is_used_over_the_near_fallback(self):
        """A candidate whose own hint_date is set (from discovery) should
        probe the snapshot near that date, not the generic fallback."""
        canonical = "https://www.business-standard.com/article/companies/x.html"
        fetcher = _FakeWaybackFetcher()
        candidate = Candidate(canonical, "business_standard", date(2023, 1, 25))
        fetch_and_parse_via_wayback(fetcher, [candidate], date(2020, 1, 1))
        # The availability call's timestamp param reflects the hint_date,
        # not the far-off fallback `near`.
        assert any("20230125" in call for call in fetcher.calls)
        assert not any("20200101" in call for call in fetcher.calls)
