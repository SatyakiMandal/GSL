"""Tests for daily aggregation, incident flagging and ranking."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.dedupe import deduplicate  # noqa: E402
from ceia.eventstudy import (  # noqa: E402
    aggregate_by_day,
    attach_headlines,
    build_daily_table,
    caveats,
    rank_incidents,
    sentiment_return_correlation,
)
from ceia.extract import IST  # noqa: E402
from ceia.models import NewsItem  # noqa: E402
from ceia.returns import MarketModel, abnormal_returns, daily_returns  # noqa: E402


def item(day: date, sentiment: float, *, url="u", source="et", headline="h",
         relevance=0.9, duplicate_of=None, event="other",
         after_close=False, emotion="") -> NewsItem:
    news = NewsItem(source=source, url=url, headline=headline,
                    published_at=datetime(day.year, day.month, day.day, 10, 0, tzinfo=IST),
                    sentiment_score=sentiment, relevance_score=relevance,
                    event_category=event, after_close=after_close,
                    emotion_label=emotion)
    news.trading_day = day
    news.duplicate_of = duplicate_of
    news.sentiment_label = ("positive" if sentiment > 0.15
                            else "negative" if sentiment < -0.15 else "neutral")
    return news


def price_frame(returns_by_day: dict[date, tuple[float, float]]) -> pd.DataFrame:
    """Build a frame from {day: (company_return, benchmark_return)}."""
    days = sorted(returns_by_day)
    company = [returns_by_day[d][0] for d in days]
    benchmark = [returns_by_day[d][1] for d in days]
    frame = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.array(company)),
        "benchmark_close": 100 * np.cumprod(1 + np.array(benchmark)),
    }, index=pd.to_datetime(days))
    frame.index.name = "date"
    frame["return"] = daily_returns(frame["close"])
    frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
    return abnormal_returns(frame, MarketModel(0.0, 1.0, 0.01, 100, 0.9, True))


class TestAggregation:
    def test_counts_and_averages(self):
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.8, url="a"), item(day, -0.4, url="b"),
        ])[day]
        assert coverage.item_count == 2 and coverage.unique_count == 2
        assert coverage.mean_sentiment == pytest.approx(-0.6)
        assert coverage.min_sentiment == pytest.approx(-0.8)

    def test_duplicates_counted_once_in_sentiment(self):
        day = date(2023, 1, 25)
        items = [item(day, -0.9, url="a"),
                 item(day, -0.9, url="b", duplicate_of="a"),
                 item(day, -0.9, url="c", duplicate_of="a")]
        coverage = aggregate_by_day(items)[day]
        assert coverage.item_count == 3, "all carriers still counted"
        assert coverage.unique_count == 1, "but the story counts once"

    def test_unattributed_items_excluded(self):
        news = item(date(2023, 1, 25), -0.5)
        news.trading_day = None
        assert aggregate_by_day([news]) == {}

    def test_dominant_event_is_the_mode(self):
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", event="regulatory"),
            item(day, -0.5, url="b", event="regulatory"),
            item(day, -0.5, url="c", event="earnings"),
        ])[day]
        assert coverage.dominant_event == "regulatory"

    def test_dominant_emotion_is_the_mode(self):
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", emotion="fear"),
            item(day, -0.5, url="b", emotion="fear"),
            item(day, -0.5, url="c", emotion="anger"),
        ])[day]
        assert coverage.dominant_emotion == "fear"

    def test_dominant_emotion_blank_when_nothing_cleared_threshold(self):
        """Items where GoEmotions stayed silent must not force a label."""
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", emotion=""),
            item(day, -0.5, url="b", emotion=""),
        ])[day]
        assert coverage.dominant_emotion == ""

    def test_dominant_emotion_ignores_items_with_no_label(self):
        """A day with one confident emotion and one blank still surfaces the
        confident one, rather than being diluted to nothing."""
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", emotion="fear"),
            item(day, -0.5, url="b", emotion=""),
        ])[day]
        assert coverage.dominant_emotion == "fear"

    def test_tied_dominant_labels_break_deterministically_by_first_seen(self):
        """A real bug this pins: max(set(x), key=x.count) breaks a tied count
        via Python's per-process string hash randomisation, so the same tied
        input could report a different "dominant" label on every run -
        verified directly across nine different PYTHONHASHSEED values before
        the fix. Counter.most_common() is documented to break ties by
        first-encountered order instead, which is deterministic because the
        input order already is.
        """
        day = date(2023, 1, 25)
        # "curiosity" first, "disappointment" second, tied 1-1: this exact
        # pair previously landed on "curiosity" under some hash seeds and
        # "disappointment" under others.
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", emotion="curiosity"),
            item(day, -0.5, url="b", emotion="disappointment"),
        ])[day]
        assert coverage.dominant_emotion == "curiosity"

        reversed_order = aggregate_by_day([
            item(day, -0.5, url="a", emotion="disappointment"),
            item(day, -0.5, url="b", emotion="curiosity"),
        ])[day]
        assert reversed_order.dominant_emotion == "disappointment"

    def test_tied_dominant_event_breaks_deterministically_too(self):
        """The same fix applies to dominant_event, which had the identical
        bug pattern before dominant_emotion was added."""
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([
            item(day, -0.5, url="a", event="earnings"),
            item(day, -0.5, url="b", event="regulatory"),
        ])[day]
        assert coverage.dominant_event == "earnings"

    def test_weighted_sentiment_favours_widely_carried_stories(self):
        day = date(2023, 1, 25)
        items = [
            item(day, -1.0, url="big", relevance=1.0),
            item(day, -1.0, url="big2", duplicate_of="big"),
            item(day, -1.0, url="big3", duplicate_of="big"),
            item(day, +1.0, url="small", relevance=0.4),
        ]
        coverage = aggregate_by_day(items)[day]
        assert coverage.weighted_sentiment < coverage.mean_sentiment
        assert coverage.weighted_sentiment < 0

    def test_after_close_items_counted(self):
        day = date(2023, 1, 25)
        coverage = aggregate_by_day([item(day, -0.5, after_close=True)])[day]
        assert coverage.after_close_count == 1


class TestDailyTable:
    def _setup(self):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01),   # big company-specific fall
            days[3]: (-0.05, -0.004), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}") for i in range(6)]
        news.append(item(days[0], 0.1, url="quiet"))
        return frame, news, days

    def test_joins_prices_to_news(self):
        frame, news, days = self._setup()
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        assert len(table) == 5
        assert table.loc[days[2], "unique_count"] == 6
        assert table.loc[days[1], "unique_count"] == 0

    def test_coverage_z_marks_the_busy_day(self):
        frame, news, days = self._setup()
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        assert table.loc[days[2], "coverage_z"] > 1.0

    def test_abnormal_return_isolates_company_move(self):
        frame, news, days = self._setup()
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        row = table.loc[days[2]]
        assert row["abnormal_return"] < row["return"] * 0.99 or row["abnormal_return"] < 0
        # market fell 1%, company fell 20% -> abnormal ~ -19%
        assert row["abnormal_return"] == pytest.approx(-0.19, abs=0.01)

    def test_empty_window_returns_empty_frame(self):
        frame, news, days = self._setup()
        table = build_daily_table(frame, aggregate_by_day(news),
                                  date(2024, 1, 1), date(2024, 1, 5))
        assert table.empty


class TestRanking:
    def _setup(self):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        news.append(item(days[0], 0.1, url="quiet"))
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        return frame, table, news, days

    def test_flags_the_incident_day(self):
        frame, table, news, days = self._setup()
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        assert incidents, "the -20% day with 6 negative stories should flag"
        assert incidents[0].day == days[2]

    def test_direction_agreement_detected(self):
        frame, table, news, days = self._setup()
        top = rank_incidents(table, frame, return_threshold=1.0)[0]
        assert top.direction_agrees, "negative tone with a negative abnormal return"
        assert top.mean_sentiment < 0 and top.abnormal_return < 0

    def test_dominant_emotion_propagates_onto_the_incident(self):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory", emotion="fear")
                for i in range(6)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        top = rank_incidents(table, frame, return_threshold=1.0)[0]
        assert top.dominant_emotion == "fear"

    def test_blank_dominant_emotion_when_goemotions_was_skipped(self):
        """--skip-emotion runs must still flag incidents; the field is just ''."""
        frame, table, news, days = self._setup()  # no emotion set on these items
        top = rank_incidents(table, frame, return_threshold=1.0)[0]
        assert top.dominant_emotion == ""

    def test_quiet_days_never_flag(self):
        frame, table, news, days = self._setup()
        flagged = {i.day for i in rank_incidents(table, frame, return_threshold=1.0)}
        assert days[1] not in flagged, "no coverage that day"

    def test_big_move_without_coverage_is_not_an_incident(self):
        """An unusual return alone is a move with no visible explanation."""
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01), days[3]: (-0.01, -0.005), days[4]: (0.01, 0.005),
        })
        table = build_daily_table(frame, aggregate_by_day([]), days[0], days[-1])
        assert rank_incidents(table, frame, return_threshold=1.0) == []

    def test_coverage_without_a_move_is_not_an_incident(self):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({d: (0.001, 0.001) for d in days})
        news = [item(days[2], -0.9, url=f"n{i}") for i in range(8)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        assert rank_incidents(table, frame, return_threshold=1.5) == []

    def test_car_attached_to_each_incident(self):
        frame, table, news, days = self._setup()
        top = rank_incidents(table, frame, event_window=(-1, 3),
                             return_threshold=1.0)[0]
        assert top.car["days"] > 0
        assert top.car["car"] < 0

    def test_top_n_limits_output(self):
        frame, table, news, days = self._setup()
        assert len(rank_incidents(table, frame, return_threshold=0.1, top_n=1)) <= 1

    def test_headlines_attached(self):
        frame, table, news, days = self._setup()
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        attach_headlines(incidents, news, limit=3)
        assert incidents[0].headlines
        assert len(incidents[0].headlines) <= 3


class TestCaveats:
    def test_always_states_case_study_and_non_causation(self):
        table = pd.DataFrame({"unique_count": [1, 2, 3]})
        notes = " ".join(caveats(table, [], "market-model", "estimation-window"))
        assert "not a statistically validated causal finding" in notes
        assert "COINCIDED WITH" in notes
        assert "not evidence" in notes

    def test_warns_the_coverage_bar_was_relaxed(self):
        """A short window cannot support a coverage baseline; say so loudly."""
        table = pd.DataFrame({"unique_count": [1, 0, 0]})
        notes = " ".join(caveats(table, [], "market-model", "estimation-window"))
        assert "RELAXED" in notes
        assert "not as evidence that the coverage was itself unusual" in notes

    def test_warns_when_beta_was_assumed(self):
        table = pd.DataFrame({"unique_count": [1] * 20})
        notes = " ".join(caveats(table, [], "market-adjusted", "estimation-window"))
        assert "beta fixed at 1.0" in notes

    def test_warns_on_contaminated_scale(self):
        table = pd.DataFrame({"unique_count": [1] * 20})
        notes = " ".join(caveats(table, [], "market-model",
                                 "analysis-window SD (weaker...)"))
        assert "understates how" in notes


class TestShortWindowBehaviour:
    """A window built around a known event has no usable coverage baseline."""

    def _short_window(self):
        days = [date(2023, 1, d) for d in (24, 25, 26, 27)]
        frame = price_frame({
            days[0]: (-0.005, 0.000), days[1]: (-0.14, -0.005),
            days[2]: (0.016, 0.004), days[3]: (-0.18, -0.003),
        })
        news = ([item(days[0], -0.1, url="a1"), item(days[0], -0.2, url="a2")]
                + [item(days[1], -0.6, url=f"b{i}", event="regulatory") for i in range(3)]
                + [item(days[2], -0.3, url=f"c{i}") for i in range(6)]
                + [item(days[3], -0.5, url=f"d{i}", event="regulatory") for i in range(3)])
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        return frame, table, news, days

    def test_extreme_moves_still_flag_on_a_short_window(self):
        """The bug this covers: -14% and -18% days flagging nothing at all."""
        frame, table, news, days = self._short_window()
        incidents = rank_incidents(table, frame, return_threshold=1.5)
        flagged = {i.day for i in incidents}
        assert days[1] in flagged and days[3] in flagged

    def test_the_relaxation_is_disclosed(self):
        frame, table, news, days = self._short_window()
        incidents = rank_incidents(table, frame, return_threshold=1.5)
        notes = " ".join(caveats(table, incidents, "market-model", "estimation-window"))
        assert "RELAXED" in notes

    def test_strict_test_returns_on_a_long_window(self):
        """With enough days, coverage must actually be unusual again."""
        days = [date(2023, 1, 3) + pd.Timedelta(days=i) for i in range(20)]
        days = [d.date() if hasattr(d, "date") else d for d in days]
        frame = price_frame({d: (0.001, 0.001) for d in days})
        news = [item(d, -0.2, url=f"x{i}") for i, d in enumerate(days)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        assert rank_incidents(table, frame, return_threshold=1.5) == []


class TestSentimentReturnCorrelation:
    def _table(self, days_returns_sentiments):
        """days_returns_sentiments: list of (day, company_return, benchmark_return, sentiment).

        Prepends one price-only anchor day with no news, since the first row
        of any price series has a NaN return (pct_change() has nothing before
        it to diff against) - putting news there would make it untestable
        rather than testing the thing this class exists to test.
        """
        anchor = days_returns_sentiments[0][0] - pd.Timedelta(days=1)
        returns_by_day = {anchor: (0.0, 0.0)}
        returns_by_day.update({d: (r, b) for d, r, b, _ in days_returns_sentiments})
        frame = price_frame(returns_by_day)
        news = [item(d, s, url=f"n{i}") for i, (d, _, _, s) in enumerate(days_returns_sentiments)]
        days = [d for d, _, _, _ in days_returns_sentiments]
        return build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])

    def test_too_few_covered_days_returns_none(self):
        days = [date(2023, 1, 3 + i) for i in range(2)]
        table = self._table([(days[0], 0.02, 0.0, 0.8), (days[1], -0.02, 0.0, -0.8)])
        result = sentiment_return_correlation(table)
        assert result["r"] is None
        assert result["n"] == 2
        assert "too few" in result["note"]

    def test_perfectly_aligned_sentiment_and_return_gives_r_near_one(self):
        days = [date(2023, 1, 3 + i) for i in range(5)]
        rows = [(days[0], 0.03, 0.0, 0.9), (days[1], -0.03, 0.0, -0.9),
                (days[2], 0.05, 0.0, 0.95), (days[3], -0.05, 0.0, -0.95),
                (days[4], 0.01, 0.0, 0.2)]
        table = self._table(rows)
        result = sentiment_return_correlation(table)
        assert result["r"] is not None
        assert result["r"] > 0.9
        assert result["n"] == 5
        # r and r_squared are each independently rounded to 4dp from the
        # unrounded r, so squaring the already-rounded r only matches to
        # about that same precision, not exactly.
        assert result["r_squared"] == pytest.approx(result["r"] ** 2, abs=1e-4)

    def test_inverted_relationship_gives_negative_r(self):
        days = [date(2023, 1, 3 + i) for i in range(5)]
        rows = [(days[0], 0.03, 0.0, -0.9), (days[1], -0.03, 0.0, 0.9),
                (days[2], 0.05, 0.0, -0.95), (days[3], -0.05, 0.0, 0.95),
                (days[4], 0.01, 0.0, -0.2)]
        table = self._table(rows)
        result = sentiment_return_correlation(table)
        assert result["r"] < -0.9

    def test_zero_variance_sentiment_is_undefined_not_a_crash(self):
        days = [date(2023, 1, 3 + i) for i in range(4)]
        rows = [(d, 0.01 * (i - 1), 0.0, 0.5) for i, d in enumerate(days)]
        table = self._table(rows)
        result = sentiment_return_correlation(table)
        assert result["r"] is None
        assert "zero variance" in result["note"]

    def test_days_with_no_coverage_are_excluded_from_n(self):
        """A silent day forces sentiment to 0.0 by construction; including it
        would dilute the correlation with a manufactured non-signal point."""
        days = [date(2023, 1, 3 + i) for i in range(6)]
        returns_by_day = {days[0] - pd.Timedelta(days=1): (0.0, 0.0)}
        returns_by_day.update({d: (0.01 * (i % 3 - 1), 0.0) for i, d in enumerate(days)})
        frame = price_frame(returns_by_day)
        news = [item(days[1], 0.8, url="a"), item(days[3], -0.8, url="b"),
               item(days[5], 0.6, url="c")]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        result = sentiment_return_correlation(table)
        assert result["n"] == 3

    def test_empty_table_does_not_crash(self):
        table = pd.DataFrame(columns=[
            "close", "return", "benchmark_return", "abnormal_return",
            "unique_count", "weighted_sentiment",
        ])
        result = sentiment_return_correlation(table)
        assert result["r"] is None
        assert result["n"] == 0
