"""Tests for the sitemap-parsing helpers in discovery.py.

Real bug caught while adding Business Today as a fifth source: its sitemap
wraps <loc>/<lastmod> content in CDATA, which the original regexes could not
see past (a CDATA opener starts with '<', outside the [^<\\s]+ character
class), so it silently matched zero URLs instead of erroring - indistinguishable
from "no coverage that day" without reading the raw response. These tests
pin both the CDATA and plain shapes so a future site's format does not
regress the other.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.discovery import (  # noqa: E402
    _loc_lastmod_pairs,
    _locs,
    _sitemap_index_pairs,
    entrackr,
    inc42,
    vccircle,
)


class TestLocsParsing:
    def test_plain_loc_no_cdata(self):
        xml = "<urlset><url><loc>https://x.com/a</loc></url></urlset>"
        assert _locs(xml) == ["https://x.com/a"]

    def test_cdata_wrapped_loc(self):
        xml = "<urlset><url><loc><![CDATA[ https://x.com/a ]]></loc></url></urlset>"
        assert _locs(xml) == ["https://x.com/a"]

    def test_mixed_cdata_and_plain_in_one_document(self):
        xml = (
            "<urlset>"
            "<url><loc><![CDATA[ https://x.com/a ]]></loc></url>"
            "<url><loc>https://x.com/b</loc></url>"
            "</urlset>"
        )
        assert _locs(xml) == ["https://x.com/a", "https://x.com/b"]


class TestLocLastmodPairs:
    def test_cdata_wrapped_loc_and_lastmod(self):
        xml = (
            "<urlset><url>"
            "<loc><![CDATA[ https://x.com/a ]]></loc>"
            "<lastmod><![CDATA[ 2025-03-05T21:36:00+05:30 ]]></lastmod>"
            "</url></urlset>"
        )
        pairs = _loc_lastmod_pairs(xml)
        assert pairs == [("https://x.com/a", date(2025, 3, 5))]

    def test_plain_loc_and_lastmod_still_works(self):
        xml = (
            "<urlset><url>"
            "<loc>https://x.com/a</loc>"
            "<lastmod>2025-03-05T21:36:00+05:30</lastmod>"
            "</url></urlset>"
        )
        pairs = _loc_lastmod_pairs(xml)
        assert pairs == [("https://x.com/a", date(2025, 3, 5))]

    def test_missing_lastmod_still_yields_the_url_with_no_date(self):
        xml = "<urlset><url><loc><![CDATA[ https://x.com/a ]]></loc></url></urlset>"
        pairs = _loc_lastmod_pairs(xml)
        assert pairs == [("https://x.com/a", None)]

    def test_multiple_urls_all_parsed(self):
        xml = (
            "<urlset>"
            "<url><loc><![CDATA[ https://x.com/a ]]></loc>"
            "<lastmod><![CDATA[ 2025-01-01T00:00:00+05:30 ]]></lastmod></url>"
            "<url><loc><![CDATA[ https://x.com/b ]]></loc>"
            "<lastmod><![CDATA[ 2025-01-02T00:00:00+05:30 ]]></lastmod></url>"
            "</urlset>"
        )
        pairs = _loc_lastmod_pairs(xml)
        assert pairs == [
            ("https://x.com/a", date(2025, 1, 1)),
            ("https://x.com/b", date(2025, 1, 2)),
        ]


class TestSitemapIndexPairs:
    def test_parses_sitemap_blocks_not_url_blocks(self):
        xml = (
            "<sitemapindex>"
            "<sitemap><loc>https://x.com/a.xml</loc>"
            "<lastmod>2025-01-01T00:00:00+00:00</lastmod></sitemap>"
            "<sitemap><loc>https://x.com/b.xml</loc>"
            "<lastmod>2025-02-01T00:00:00+00:00</lastmod></sitemap>"
            "</sitemapindex>"
        )
        assert _sitemap_index_pairs(xml) == [
            ("https://x.com/a.xml", date(2025, 1, 1)),
            ("https://x.com/b.xml", date(2025, 2, 1)),
        ]

    def test_missing_lastmod_still_yields_the_url(self):
        xml = "<sitemapindex><sitemap><loc>https://x.com/a.xml</loc></sitemap></sitemapindex>"
        assert _sitemap_index_pairs(xml) == [("https://x.com/a.xml", None)]


class _FakeXMLResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status = 200


class _FakeXMLFetcher:
    """Serves canned XML per exact URL; a URL with no fixture raises, which
    _fetch_xml's blanket except turns into a logged skip - the same failure
    mode a real 404 or network error produces."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeXMLResponse:
        self.calls.append(url)
        if url not in self.pages:
            raise RuntimeError(f"no fixture for {url}")
        return _FakeXMLResponse(self.pages[url])


def _urlset(pairs: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<url><loc>{url}</loc><lastmod>{lastmod}</lastmod></url>"
        for url, lastmod in pairs
    )
    return f"<urlset>{body}</urlset>"


def _sitemapindex(pairs: list[tuple[str, str | None]]) -> str:
    body = "".join(
        f"<sitemap><loc>{url}</loc>"
        + (f"<lastmod>{lastmod}</lastmod>" if lastmod else "")
        + "</sitemap>"
        for url, lastmod in pairs
    )
    return f"<sitemapindex>{body}</sitemapindex>"


class TestEntrackrStrategy:
    def test_fetches_one_sitemap_per_day_and_skips_missing_days(self):
        pages = {
            "https://entrackr.com/sitemap_2026-01-01.xml":
                _urlset([("https://entrackr.com/a", "2026-01-01T10:00:00+05:30")]),
            "https://entrackr.com/sitemap_2026-01-03.xml":
                _urlset([("https://entrackr.com/b", "2026-01-03T10:00:00+05:30")]),
            # 2026-01-02 has no fixture - simulates a day with no articles/404.
        }
        fetcher = _FakeXMLFetcher(pages)
        candidates = entrackr(fetcher, date(2026, 1, 1), date(2026, 1, 3))
        assert {c.url for c in candidates} == {
            "https://entrackr.com/a", "https://entrackr.com/b",
        }
        assert {c.hint_date for c in candidates} == {date(2026, 1, 1), date(2026, 1, 3)}


class TestVccircleStrategy:
    def test_walks_forward_until_a_file_is_entirely_before_the_window(self):
        pages = {
            "https://www.vccircle.com/sitemap/article-sitemap-1.xml": _urlset([
                ("https://www.vccircle.com/newest", "2026-03-10T00:00:00Z"),
            ]),
            "https://www.vccircle.com/sitemap/article-sitemap-2.xml": _urlset([
                ("https://www.vccircle.com/straddling", "2026-03-01T00:00:00Z"),
                ("https://www.vccircle.com/before-window", "2026-01-01T00:00:00Z"),
            ]),
            # File 3 is entirely before the window - must never be fetched,
            # since the walk should have already stopped at file 2.
            "https://www.vccircle.com/sitemap/article-sitemap-3.xml": _urlset([
                ("https://www.vccircle.com/way-too-old", "2025-01-01T00:00:00Z"),
            ]),
        }
        fetcher = _FakeXMLFetcher(pages)
        candidates = vccircle(fetcher, date(2026, 3, 1), date(2026, 3, 15))
        urls = {c.url for c in candidates}
        assert "https://www.vccircle.com/newest" in urls
        assert "https://www.vccircle.com/straddling" in urls
        assert "https://www.vccircle.com/before-window" not in urls
        assert "https://www.vccircle.com/sitemap/article-sitemap-3.xml" not in fetcher.calls

    def test_stops_at_max_files_when_a_fixture_is_missing(self):
        pages = {
            "https://www.vccircle.com/sitemap/article-sitemap-1.xml": _urlset([
                ("https://www.vccircle.com/only", "2026-03-10T00:00:00Z"),
            ]),
        }
        fetcher = _FakeXMLFetcher(pages)
        candidates = vccircle(fetcher, date(2026, 3, 1), date(2026, 3, 15))
        assert [c.url for c in candidates] == ["https://www.vccircle.com/only"]


class TestInc42Strategy:
    def test_skips_files_whose_newest_article_predates_the_window(self):
        index = _sitemapindex([
            ("https://inc42.com/post-sitemap.xml", "2026-08-06T00:00:00+00:00"),
            ("https://inc42.com/post-sitemap2.xml", "2025-01-01T00:00:00+00:00"),
            ("https://inc42.com/post-sitemap3.xml", "2026-01-15T00:00:00+00:00"),
        ])
        pages = {
            "https://inc42.com/sitemap_index.xml": index,
            "https://inc42.com/post-sitemap3.xml": _urlset([
                ("https://inc42.com/buzz/x", "2026-01-10T00:00:00+00:00"),
            ]),
            # post-sitemap.xml (unnumbered) and post-sitemap2.xml must never
            # be fetched: the first is the newest-file duplicate, the second's
            # own lastmod is before the window.
        }
        fetcher = _FakeXMLFetcher(pages)
        candidates = inc42(fetcher, date(2026, 1, 1), date(2026, 1, 31))
        assert [c.url for c in candidates] == ["https://inc42.com/buzz/x"]
        assert "https://inc42.com/post-sitemap.xml" not in fetcher.calls
        assert "https://inc42.com/post-sitemap2.xml" not in fetcher.calls

    def test_articles_outside_the_window_within_a_fetched_file_are_dropped(self):
        index = _sitemapindex([
            ("https://inc42.com/post-sitemap4.xml", "2026-02-01T00:00:00+00:00"),
        ])
        pages = {
            "https://inc42.com/sitemap_index.xml": index,
            "https://inc42.com/post-sitemap4.xml": _urlset([
                ("https://inc42.com/buzz/in-window", "2026-01-15T00:00:00+00:00"),
                ("https://inc42.com/buzz/too-late", "2026-03-01T00:00:00+00:00"),
            ]),
        }
        fetcher = _FakeXMLFetcher(pages)
        candidates = inc42(fetcher, date(2026, 1, 1), date(2026, 1, 31))
        assert [c.url for c in candidates] == ["https://inc42.com/buzz/in-window"]

    def test_no_index_returns_empty(self):
        fetcher = _FakeXMLFetcher({})
        assert inc42(fetcher, date(2026, 1, 1), date(2026, 1, 31)) == []
