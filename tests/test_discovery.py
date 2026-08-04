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

from ceia.discovery import _loc_lastmod_pairs, _locs  # noqa: E402


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
