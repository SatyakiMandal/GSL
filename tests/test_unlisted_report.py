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

from ceia.charts import price_level_svg  # noqa: E402
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

    def ranked_moves(self, top_n=None):
        ranked = sorted(self.moves, key=lambda m: -abs(m.change))
        return ranked[:top_n] if top_n else ranked


def make_analysis(moves=None, unattributed=None) -> FakeUnlistedAnalysis:
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
