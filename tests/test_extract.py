"""Tests for extract.py's timestamp extraction against non-standard meta tags.

VCCircle carries no JSON-LD articleBody or standard article:published_time
meta tag - its publish time comes from a site-specific
content_type:published_time property instead, verified directly against a
real downloaded page. This pins that one addition to _TIMESTAMP_META so a
future refactor of the list does not silently drop it.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.extract import IST, extract_timestamp  # noqa: E402


class TestVccircleTimestampMeta:
    def test_content_type_published_time_is_read(self):
        html = (
            '<html><head>'
            '<meta property="content_type:published_time" '
            'content="2026-08-06T18:27:53Z"/>'
            '</head><body></body></html>'
        )
        soup = BeautifulSoup(html, "lxml")
        value, confidence = extract_timestamp(html, soup)
        assert value == datetime(2026, 8, 6, 23, 57, 53, tzinfo=IST)
        assert confidence == "exact"

    def test_standard_article_published_time_still_wins_when_both_present(self):
        """The list is tried in order - a site with the standard tag should
        never fall through to this one, so the addition can't shadow an
        existing, better-supported source."""
        html = (
            '<html><head>'
            '<meta property="article:published_time" '
            'content="2026-08-06T10:00:00+05:30"/>'
            '<meta property="content_type:published_time" '
            'content="2026-08-06T18:27:53Z"/>'
            '</head><body></body></html>'
        )
        soup = BeautifulSoup(html, "lxml")
        value, confidence = extract_timestamp(html, soup)
        assert value == datetime(2026, 8, 6, 10, 0, 0, tzinfo=IST)
