"""Tests for trading-day attribution and deduplication.

These cover the failure the PRD calls out in Section 10: after-hours news
credited to the same day's close, describing a reaction that happened before
the news existed.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.align import attribute, attribute_all, next_trading_day, unattributed  # noqa: E402
from ceia.dedupe import cluster_sizes, deduplicate, similarity, unique  # noqa: E402
from ceia.extract import IST, parse_timestamp  # noqa: E402
from ceia.models import NewsItem  # noqa: E402


def item(when: datetime | None, confidence: str = "exact", **kwargs) -> NewsItem:
    return NewsItem(source="et", url=kwargs.pop("url", "u"),
                    headline=kwargs.pop("headline", "h"),
                    published_at=when, timestamp_confidence=confidence, **kwargs)


class TestTradingDayAttribution:
    def test_before_close_is_same_day(self):
        result = attribute(item(datetime(2023, 1, 25, 15, 5, tzinfo=IST)))
        assert result.trading_day == date(2023, 1, 25)
        assert not result.after_close

    def test_after_close_rolls_to_next_day(self):
        result = attribute(item(datetime(2023, 1, 25, 20, 57, tzinfo=IST)))
        assert result.trading_day == date(2023, 1, 26)
        assert result.after_close

    def test_exactly_at_close_counts_as_after(self):
        result = attribute(item(datetime(2023, 1, 25, 15, 30, tzinfo=IST)))
        assert result.after_close
        assert result.trading_day == date(2023, 1, 26)

    def test_friday_evening_rolls_past_the_weekend(self):
        # 2023-01-27 was a Friday.
        result = attribute(item(datetime(2023, 1, 27, 19, 0, tzinfo=IST)))
        assert result.trading_day == date(2023, 1, 30)

    def test_saturday_news_rolls_to_monday(self):
        result = attribute(item(datetime(2023, 1, 28, 11, 0, tzinfo=IST)))
        assert result.trading_day == date(2023, 1, 30)

    def test_exchange_holiday_respected_with_calendar(self):
        """26 Jan is Republic Day: a real calendar must skip it."""
        calendar = {date(2023, 1, 25), date(2023, 1, 27), date(2023, 1, 30)}
        result = attribute(item(datetime(2023, 1, 25, 21, 34, tzinfo=IST)), calendar)
        assert result.trading_day == date(2023, 1, 27)

    def test_without_calendar_the_holiday_is_missed(self):
        """Documents the known limitation of the weekday-only fallback."""
        result = attribute(item(datetime(2023, 1, 25, 21, 34, tzinfo=IST)))
        assert result.trading_day == date(2023, 1, 26)  # a holiday, but we cannot know

    def test_utc_timestamp_converted_to_ist_before_comparing(self):
        """10:30 UTC is 16:00 IST - after the close, not before."""
        when = datetime(2023, 1, 25, 10, 30, tzinfo=timezone.utc)
        result = attribute(item(when))
        assert result.after_close
        assert result.trading_day == date(2023, 1, 26)

    def test_missing_timestamp_is_not_attributed(self):
        result = attribute(item(None, confidence="missing"))
        assert result.trading_day is None
        assert unattributed([result]) == [result]

    def test_date_only_is_attributed_but_flagged(self):
        result = attribute(item(datetime(2023, 1, 25, 0, 0, tzinfo=IST),
                                confidence="date-only"))
        assert result.trading_day == date(2023, 1, 25)
        assert not result.after_close
        assert result.timestamp_confidence == "date-only"

    def test_attribute_all_returns_every_item(self):
        items = [item(datetime(2023, 1, 25, 9, 0, tzinfo=IST)), item(None, "missing")]
        assert len(attribute_all(items)) == 2

    def test_next_trading_day_returns_none_past_calendar_horizon(self):
        assert next_trading_day(date(2023, 1, 30), {date(2023, 1, 25)}) is None


class TestTimestampParsing:
    def test_naive_timestamp_assumed_ist(self):
        value, confidence = parse_timestamp("2023-01-25T20:57:54")
        assert confidence == "exact"
        assert value.utcoffset() == timedelta(hours=5, minutes=30)

    def test_explicit_offset_preserved(self):
        value, _ = parse_timestamp("2023-01-25T20:57:54+05:30")
        assert value.hour == 20

    def test_fractional_seconds(self):
        value, confidence = parse_timestamp("2023-01-25T20:57:54.000+05:30")
        assert confidence == "exact" and value.hour == 20

    def test_date_only(self):
        value, confidence = parse_timestamp("2023-01-25")
        assert confidence == "date-only" and value.date() == date(2023, 1, 25)

    def test_garbage_is_missing(self):
        assert parse_timestamp("not a date") == (None, "missing")
        assert parse_timestamp(None) == (None, "missing")


class TestDedupe:
    def test_identical_headlines_cluster(self):
        assert similarity("Adani shares tank after fraud claim",
                          "Adani shares tank after fraud claim") == 1.0

    def test_unrelated_headlines_do_not(self):
        assert similarity("Adani shares tank after fraud claim",
                          "Infosys wins cloud deal in Europe") < 0.2

    def test_syndicated_copy_marked_duplicate(self):
        when = datetime(2023, 1, 25, 12, 0, tzinfo=IST)
        items = [
            item(when, url="et", headline="Adani shares tank after US investor calls out fraud",
                 body="x" * 500),
            item(when, url="fe", headline="Adani shares tank after US investor calls out fraud",
                 body="x" * 100),
        ]
        deduplicate(items)
        survivors = unique(items)
        assert len(survivors) == 1
        assert survivors[0].url == "et", "the fullest copy should survive"
        assert items[1].duplicate_of == "et"

    def test_different_days_never_cluster(self):
        headline = "Adani shares tank after US investor calls out fraud"
        items = [
            item(datetime(2023, 1, 25, 12, 0, tzinfo=IST), url="a", headline=headline),
            item(datetime(2023, 3, 25, 12, 0, tzinfo=IST), url="b", headline=headline),
        ]
        deduplicate(items)
        assert len(unique(items)) == 2

    def test_cluster_sizes_count_carriers(self):
        when = datetime(2023, 1, 25, 12, 0, tzinfo=IST)
        headline = "Adani shares tank after US investor calls out fraud"
        items = [item(when, url=u, headline=headline, body="x" * (500 - i * 10))
                 for i, u in enumerate(["et", "fe", "bl"])]
        deduplicate(items)
        assert cluster_sizes(items)["et"] == 3
