"""Tests for the unlisted-share HTML report.

Mirrors test_report.py's concerns for the listed-company report: no unescaped
untrusted text, no external dependencies, and — specific to this template —
no borrowed statistical vocabulary (z, CAR, t, p, Robust) that a
periodically-revised indicative price cannot earn. See ceia/unlisted.py's
module docstring for why that distinction matters here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.extract import IST  # noqa: E402
from ceia.charts import news_coverage_svg, price_level_svg  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402
from ceia.report import build_unlisted_html  # noqa: E402
from ceia.unlisted import PriceMove  # noqa: E402

# Vocabulary that belongs only to the listed-company report's statistical
# apparatus - it must never appear as a claimed value here (a plain-English
# disclaimer that explains *why* it's absent is fine and expected).
FORBIDDEN_COLUMNS = ["<th>z</th>", "<th>t</th>", "<th>p**</th>",
                     "Robust***", 'CAR[', "abnormal_return_z"]


def _move(**kwargs) -> PriceMove:
    defaults = dict(start_date=date(2026, 1, 1), end_date=date(2026, 1, 15),
                    start_price=100.0, end_price=120.0,
                    headlines=[{
                        "source": "et", "headline": "Funding round announced",
                        "url": "https://example.com/a", "sentiment_label": "positive",
                        "relevance": 0.8, "summary": "A short summary.",
                    }])
    defaults.update(kwargs)
    return PriceMove(**defaults)


@dataclass
class FakeUnlistedAnalysis:
    config: RunConfig
    series: pd.DataFrame
    moves: list
    news_meta: dict
    unattributed: list = field(default_factory=list)
    items: list = field(default_factory=list)

    def ranked_moves(self, top_n=None):
        ranked = sorted(self.moves, key=lambda m: -abs(m.change))
        return ranked[:top_n] if top_n else ranked


def _item(day, sentiment=0.0, *, headline="h", url="https://example.com/x") -> NewsItem:
    from datetime import datetime
    return NewsItem(source="et", url=url, headline=headline,
                    published_at=datetime(day.year, day.month, day.day, 10, 0, tzinfo=IST),
                    sentiment_score=sentiment, relevance_score=0.8,
                    sentiment_label="positive" if sentiment > 0.15
                    else "negative" if sentiment < -0.15 else "neutral")


def make_analysis(moves=None, unattributed=None, items=None) -> FakeUnlistedAnalysis:
    days = pd.date_range("2026-01-01", periods=15)
    values = [100.0] * 14 + [120.0]
    series = pd.DataFrame({"close": values}, index=days)
    series.index.name = "date"
    return FakeUnlistedAnalysis(
        config=RunConfig(company="Test Unlisted Co", ticker="",
                         start=date(2026, 1, 1), end=date(2026, 1, 15)),
        series=series,
        moves=moves if moves is not None else [_move()],
        news_meta={"stats": {"unique_after_dedupe": 3, "relevant": 3}},
        unattributed=unattributed or [],
        items=items if items is not None else [_item(date(2026, 1, 5), 0.4)],
    )


class TestBuildUnlistedHtml:
    def test_renders_a_self_contained_document(self):
        html = build_unlisted_html(make_analysis())
        assert html.startswith("<!doctype html>")
        assert html.endswith("</body></html>")
        assert not re.search(r'<(?:script|img)[^>]+src="https?://', html)
        assert "@import" not in html
        assert "<script" not in html.lower()

    def test_no_borrowed_statistical_columns_or_vocabulary(self):
        html = build_unlisted_html(make_analysis())
        for forbidden in FORBIDDEN_COLUMNS:
            assert forbidden not in html, f"{forbidden!r} should not appear"

    def test_states_no_significance_test_is_computed(self):
        html = build_unlisted_html(make_analysis())
        assert "no z-score, market-model beta, or permutation" in html

    def test_states_the_unlistedzone_disclaimer(self):
        html = build_unlisted_html(make_analysis())
        assert "not a price feed, quote, or offer to deal" in html

    def test_move_appears_in_table_and_section(self):
        html = build_unlisted_html(make_analysis())
        assert "+20.00%" in html
        assert "Funding round announced" in html
        assert "https://example.com/a" in html

    def test_no_moves_renders_cleanly(self):
        html = build_unlisted_html(make_analysis(moves=[]))
        assert "Not enough price revisions" in html
        assert "<svg" in html

    def test_escapes_hostile_headline_text(self):
        nasty = '<img src=x onerror="alert(1)"> & "quoted"'
        move = _move(headlines=[{
            "headline": nasty, "source": "et", "url": "",
            "sentiment_label": "negative", "relevance": 0.9, "summary": "",
        }])
        html = build_unlisted_html(make_analysis(moves=[move]))
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_escapes_company_name(self):
        analysis = make_analysis()
        analysis.config.company = "<b>Evil</b> Corp"
        html = build_unlisted_html(analysis)
        assert "<b>Evil</b> Corp" not in html
        assert "&lt;b&gt;Evil" in html

    def test_unattributed_items_are_surfaced(self):
        item = NewsItem(source="et", url="u", headline="No timestamp story")
        html = build_unlisted_html(make_analysis(unattributed=[item]))
        assert "could not be placed in a price-move window" in html
        assert "No timestamp story" in html

    def test_never_claims_causation(self):
        html = build_unlisted_html(make_analysis())
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"<[^>]+>", " ", html)):
            if re.search(r"\b(?:cannot|can not|not|never|no)\b", sentence, re.I):
                continue
            for pattern in (r"\bcaused\b", r"\bdrove\b", r"\btriggered\b"):
                assert not re.search(pattern, sentence, re.I), \
                    f"causal phrasing in: {sentence!r}"

    def test_news_coverage_chart_and_table_are_present(self):
        html = build_unlisted_html(make_analysis())
        assert "news volume (bar height) and tone (colour)" in html
        assert "<h2>News coverage</h2>" in html

    def test_news_table_lists_collected_items_with_tone(self):
        item = _item(date(2026, 1, 8), 0.5, headline="Funding round expands")
        html = build_unlisted_html(make_analysis(items=[item]))
        assert "Funding round expands" in html
        assert "positive" in html
        assert "+0.50" in html

    def test_news_table_omits_items_outside_the_window(self):
        outside = _item(date(2025, 1, 1), headline="too early")
        html = build_unlisted_html(make_analysis(items=[outside]))
        assert "too early" not in html
        assert "No dated coverage found in this window" in html

    def test_news_table_omits_unattributed_items(self):
        no_date = NewsItem(source="et", url="u", headline="undated story")
        html = build_unlisted_html(make_analysis(items=[no_date]))
        assert "No dated coverage found in this window" in html


class TestPriceLevelSvg:
    def test_badges_and_real_points_render(self):
        analysis = make_analysis()
        real_dates = {date(2026, 1, 1), date(2026, 1, 15)}
        svg = price_level_svg(analysis.series, real_dates, "Test Co",
                              moves=analysis.ranked_moves())
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert 'class="real-point"' in svg
        assert "incident-badge" in svg
        assert ">1<" in svg

    def test_no_moves_means_no_badges(self):
        analysis = make_analysis()
        svg = price_level_svg(analysis.series, set(), "Test Co", moves=[])
        assert "incident-badge" not in svg
        assert 'class="chart-caption"' not in svg

    def test_empty_series_does_not_raise(self):
        empty = pd.DataFrame(columns=["close"])
        assert "No price data" in price_level_svg(empty, set(), "Test Co")


class TestNewsCoverageSvg:
    def test_bars_render_for_days_with_items(self):
        analysis = make_analysis()
        items = [_item(date(2026, 1, 5), 0.4), _item(date(2026, 1, 5), 0.6)]
        svg = news_coverage_svg(analysis.series, items, "Test Co")
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "2 item(s), tone" in svg

    def test_negative_tone_gets_hatch_overlay(self):
        analysis = make_analysis()
        items = [_item(date(2026, 1, 5), -0.6)]
        svg = news_coverage_svg(analysis.series, items, "Test Co")
        assert "tone-neg" in svg
        assert "neg-hatch-news" in svg

    def test_items_outside_the_series_window_are_ignored(self):
        analysis = make_analysis()
        items = [_item(date(2019, 1, 1), 0.5)]
        svg = news_coverage_svg(analysis.series, items, "Test Co")
        assert "1 item(s)" not in svg

    def test_empty_series_does_not_raise(self):
        empty = pd.DataFrame(columns=["close"])
        assert "No price data" in news_coverage_svg(empty, [], "Test Co")
