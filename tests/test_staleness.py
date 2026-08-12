"""Tests for the per-item news staleness signal (ceia/staleness.py).

Mirrors Tetlock (2011): staleness is a story's mean textual similarity to
its firm's ~10 most recent prior stories.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.models import NewsItem  # noqa: E402
from ceia.staleness import DEFAULT_LOOKBACK, score_items  # noqa: E402


def _item(headline: str, day: str, *, url=None, duplicate_of=None) -> NewsItem:
    return NewsItem(
        source="et", url=url or headline, headline=headline,
        published_at=datetime.fromisoformat(f"{day}T10:00:00"),
        duplicate_of=duplicate_of,
    )


class TestScoreItems:
    def test_first_unique_item_has_no_score(self):
        items = [_item("Company reports strong quarterly profit", "2025-03-01")]
        score_items(items)
        assert items[0].staleness_score is None

    def test_identical_headline_on_a_later_day_scores_high(self):
        items = [
            _item("Company reports strong quarterly profit", "2025-03-01"),
            _item("Company reports strong quarterly profit", "2025-03-05", url="u2"),
        ]
        score_items(items)
        assert items[1].staleness_score == pytest.approx(1.0)

    def test_unrelated_headline_scores_low(self):
        items = [
            _item("Company reports strong quarterly profit", "2025-03-01"),
            _item("Regulator opens probe into unrelated matter", "2025-03-05", url="u2"),
        ]
        score_items(items)
        assert items[1].staleness_score < 0.3

    def test_undated_items_are_left_unscored(self):
        dated = _item("A headline", "2025-03-01")
        undated = NewsItem(source="et", url="u2", headline="Another headline",
                           published_at=None)
        items = [dated, undated]
        score_items(items)
        assert undated.staleness_score is None

    def test_duplicates_are_left_unscored(self):
        items = [
            _item("Company reports strong quarterly profit", "2025-03-01"),
            _item("Company reports strong quarterly profit", "2025-03-01",
                 url="u2", duplicate_of="u1"),
        ]
        score_items(items)
        assert items[1].staleness_score is None

    def test_score_is_the_mean_similarity_to_prior_items_not_just_the_last_one(self):
        items = [
            _item("Company reports strong quarterly profit", "2025-03-01"),
            _item("Totally different unrelated regulatory story", "2025-03-02", url="u2"),
            _item("Company reports strong quarterly profit", "2025-03-05", url="u3"),
        ]
        score_items(items)
        # The third item is identical to item 1 but unrelated to item 2 -
        # the mean over both priors should sit strictly between 0 and 1.
        assert 0.0 < items[2].staleness_score < 1.0

    def test_lookback_limits_how_far_back_is_considered(self):
        """With lookback=1, only the immediately preceding item counts."""
        items = [
            _item("Company reports strong quarterly profit", "2025-03-01"),
            _item("Totally different unrelated regulatory story", "2025-03-02", url="u2"),
        ]
        score_items(items, lookback=1)
        # Only compared against item 1 (the sole allowed lookback slot).
        assert items[1].staleness_score < 0.3

    def test_order_is_by_published_at_not_list_order(self):
        """Items passed in out of chronological order must still be scored
        against what actually came *before* them in time."""
        later = _item("Company reports strong quarterly profit", "2025-03-05", url="u2")
        earlier = _item("Company reports strong quarterly profit", "2025-03-01", url="u1")
        items = [later, earlier]  # deliberately out of order
        score_items(items)
        assert earlier.staleness_score is None
        assert later.staleness_score == pytest.approx(1.0)

    def test_default_lookback_is_ten(self):
        assert DEFAULT_LOOKBACK == 10
