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
    emotion_valence_summary,
    flagging_diagnostics,
    rank_incidents,
    robustness_check,
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

    def test_car_dict_has_permutation_fields_even_when_series_too_short(self):
        """The 5-day fixture here is far too short for a real permutation
        test - it must degrade to p_value=None with a note, not crash or
        silently omit the keys."""
        frame, table, news, days = self._setup()
        top = rank_incidents(table, frame, return_threshold=1.0)[0]
        assert "p_value" in top.car
        assert top.car["p_value"] is None
        assert top.car.get("p_value_note")

    def test_permutations_zero_disables_the_test(self):
        frame, table, news, days = self._setup()
        top = rank_incidents(table, frame, return_threshold=1.0, permutations=0)[0]
        assert top.car["p_value"] is None
        assert "disabled" in top.car["p_value_note"]

    def test_permutation_p_value_computed_on_a_long_enough_series(self):
        """A 120-day series with one big idiosyncratic shock gives the
        permutation test enough non-flagged windows to actually run."""
        rng = np.random.default_rng(3)
        base = date(2022, 9, 1)
        days = [base + pd.Timedelta(days=i) for i in range(160)]
        days = [d for d in days if d.weekday() < 5][:120]
        market = rng.normal(0.0002, 0.006, len(days))
        company = list(market + rng.normal(0, 0.003, len(days)))
        shock_idx = 100
        company[shock_idx] -= 0.20
        returns_by_day = {d: (c, m) for d, c, m in zip(days, company, market)}
        frame = price_frame(returns_by_day)
        shock_day = days[shock_idx]
        news = [item(shock_day, -0.9, url=f"s{i}", event="regulatory") for i in range(4)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        assert incidents, "the shock day should flag"
        top = incidents[0]
        assert top.day == shock_day
        assert top.car["p_value"] is not None
        assert top.car["p_value"] < 0.1
        assert top.car["n"] > 0
        assert len(incidents[0].headlines) <= 3


class TestRobustnessCheck:
    def _setup(self):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        return frame, table, days

    def test_a_strongly_flagged_day_is_robust_at_every_grid_point(self):
        """A -20% move with 6 negative stories should clear even the
        tightest (1.3x) threshold in the default grid, so it should be
        flagged at all 9 combinations."""
        frame, table, days = self._setup()
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        result = robustness_check(table, frame, incidents, (-1, 3), 1.0, 1.0)
        key = days[2].isoformat()
        assert result["n_combos"] == 9
        assert result["days"][key]["flagged_in"] == 9
        assert result["days"][key]["fraction"] == 1.0

    def test_borderline_day_is_not_robust_at_every_grid_point(self):
        """A day that only just clears the base threshold should fail to
        flag once the return-z threshold is tightened by the grid's 1.3x."""
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.022, -0.01), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        base_incidents = rank_incidents(table, frame, return_threshold=1.0)
        assert base_incidents, "should just barely flag at the base threshold"
        z = base_incidents[0].abnormal_return_z
        assert 1.0 <= abs(z) < 1.3, f"test needs a z between 1.0 and 1.3, got {z}"
        result = robustness_check(table, frame, base_incidents, (-1, 3), 1.0, 1.0)
        key = days[2].isoformat()
        assert result["days"][key]["flagged_in"] < result["n_combos"]

    def test_no_incidents_returns_empty_result(self):
        frame, table, days = self._setup()
        result = robustness_check(table, frame, [], (-1, 3), 1.0, 1.0)
        assert result["n_combos"] == 0
        assert result["days"] == {}

    def test_base_combination_always_included_in_the_grid(self):
        """1.0x/1.0x is one of the 9 grid points, so a base incident is
        guaranteed at least 1 hit - it cannot come back as 0/9."""
        frame, table, days = self._setup()
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        result = robustness_check(table, frame, incidents, (-1, 3), 1.0, 1.0)
        for entry in result["days"].values():
            assert entry["flagged_in"] >= 1

    def test_custom_multipliers_change_the_grid_size(self):
        frame, table, days = self._setup()
        incidents = rank_incidents(table, frame, return_threshold=1.0)
        result = robustness_check(table, frame, incidents, (-1, 3), 1.0, 1.0,
                                  multipliers=(0.5, 1.0))
        assert result["n_combos"] == 4


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


class TestEmotionValenceSummary:
    def _table(self, days_returns_emotions):
        """days_returns_emotions: list of (day, company_return, benchmark_return, emotion)."""
        anchor = days_returns_emotions[0][0] - pd.Timedelta(days=1)
        returns_by_day = {anchor: (0.0, 0.0)}
        returns_by_day.update({d: (r, b) for d, r, b, _ in days_returns_emotions})
        frame = price_frame(returns_by_day)
        news = [item(d, 0.0, url=f"n{i}", emotion=e)
               for i, (d, _, _, e) in enumerate(days_returns_emotions)]
        days = [d for d, _, _, _ in days_returns_emotions]
        return build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])

    def test_groups_by_positive_negative_ambiguous(self):
        days = [date(2023, 1, 3 + i) for i in range(4)]
        table = self._table([
            (days[0], 0.03, 0.0, "joy"),
            (days[1], -0.04, 0.0, "fear"),
            (days[2], 0.01, 0.0, "admiration"),
            (days[3], 0.00, 0.0, "surprise"),
        ])
        result = emotion_valence_summary(table)
        groups = result["groups"]
        assert set(groups) == {"positive", "negative", "ambiguous"}
        assert groups["positive"]["n_days"] == 2
        assert groups["positive"]["labels_seen"] == ["admiration", "joy"]
        assert groups["negative"]["n_days"] == 1
        assert groups["negative"]["mean_abnormal_return"] == pytest.approx(
            table.loc[table["dominant_emotion"] == "fear", "abnormal_return"].iloc[0])
        assert groups["ambiguous"]["n_days"] == 1

    def test_days_with_no_emotion_are_excluded(self):
        days = [date(2023, 1, 3 + i) for i in range(3)]
        table = self._table([
            (days[0], 0.02, 0.0, "joy"),
            (days[1], -0.01, 0.0, ""),
            (days[2], 0.01, 0.0, ""),
        ])
        result = emotion_valence_summary(table)
        assert set(result["groups"]) == {"positive"}
        assert result["groups"]["positive"]["n_days"] == 1

    def test_neutral_label_is_not_a_valence_group(self):
        """'neutral' is in LABELS but excluded from all three valence groups."""
        days = [date(2023, 1, 3 + i) for i in range(2)]
        table = self._table([
            (days[0], 0.02, 0.0, "neutral"),
            (days[1], -0.01, 0.0, "joy"),
        ])
        result = emotion_valence_summary(table)
        assert set(result["groups"]) == {"positive"}

    def test_no_confident_emotion_anywhere_returns_empty_groups(self):
        days = [date(2023, 1, 3 + i) for i in range(2)]
        table = self._table([
            (days[0], 0.02, 0.0, ""),
            (days[1], -0.01, 0.0, ""),
        ])
        result = emotion_valence_summary(table)
        assert result["groups"] == {}
        assert "no day" in result["note"]

    def test_empty_table_does_not_crash(self):
        table = pd.DataFrame(columns=[
            "abnormal_return", "dominant_emotion", "weighted_sentiment",
        ])
        result = emotion_valence_summary(table)
        assert result["groups"] == {}

    def test_missing_dominant_emotion_column_does_not_crash(self):
        table = pd.DataFrame({"abnormal_return": [0.01, -0.02]})
        result = emotion_valence_summary(table)
        assert result["groups"] == {}


class TestTDistributionFlagging:
    """The professor's t-test request, reframed: day-flagging now compares
    abnormal_return_z against a t-distribution-equivalent threshold rather
    than a flat z cutoff, which widens the bar as the estimation window
    shortens (see returns.t_equivalent_threshold)."""

    def _frame(self, observations):
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        returns = {
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (0.0155, 0.0), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        }
        frame_days = sorted(returns)
        company = [returns[d][0] for d in frame_days]
        benchmark = [returns[d][1] for d in frame_days]
        frame = pd.DataFrame({
            "close": 100 * np.cumprod(1 + np.array(company)),
            "benchmark_close": 100 * np.cumprod(1 + np.array(benchmark)),
        }, index=pd.to_datetime(frame_days))
        frame.index.name = "date"
        frame["return"] = daily_returns(frame["close"])
        frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
        return abnormal_returns(frame, MarketModel(0.0, 1.0, 0.01, observations, 0.9, True)), days

    def test_short_estimation_window_is_more_conservative(self):
        """The same borderline day (abnormal_return_z ~= 1.55, just above the
        flat z=1.5 threshold) flags on a long estimation window (df=98,
        t-equivalent threshold ~= 1.51) but not on a short one (df=5,
        t-equivalent threshold ~= 1.79) - the fatter t-tails demand a bigger
        move before a short-baseline day counts as unusual."""
        long_frame, days = self._frame(observations=100)
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        table = build_daily_table(long_frame, aggregate_by_day(news), days[0], days[-1])
        incidents = rank_incidents(table, long_frame, return_threshold=1.5)
        assert {i.day for i in incidents} == {days[2]}

        short_frame, days = self._frame(observations=7)
        table_short = build_daily_table(short_frame, aggregate_by_day(news), days[0], days[-1])
        short_incidents = rank_incidents(table_short, short_frame, return_threshold=1.5)
        assert short_incidents == []

    def test_flagging_diagnostics_applies_the_same_adjustment(self):
        """flagging_diagnostics() must bucket days consistently with
        rank_incidents() - both read the same effective threshold off the
        frame's ar_scale_df, not two independently-drifting z cutoffs."""
        news_days_frame, days = self._frame(observations=100)
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        table = build_daily_table(news_days_frame, aggregate_by_day(news), days[0], days[-1])
        diag = flagging_diagnostics(table, return_threshold=1.5, frame=news_days_frame)
        assert diag["candidates"] == 1

        short_frame, days = self._frame(observations=7)
        table_short = build_daily_table(short_frame, aggregate_by_day(news), days[0], days[-1])
        diag_short = flagging_diagnostics(table_short, return_threshold=1.5, frame=short_frame)
        assert diag_short["candidates"] == 0


class TestFlaggingDiagnostics:
    """Every trading day, bucketed by which of the two flagging bars it
    cleared - the source of the report's "why the rest were dropped" text.
    """

    def _mixed_window(self):
        """15 trading days, one of each bucket, plus a run of routine days.

        Day 3 gets busy negative coverage *and* an unusual move (a
        candidate); day 6 gets the same busy negative coverage but an
        ordinary move (coverage_only); day 9 gets an unusual move but only
        the same one ordinary story every other day gets (return_only); day
        12 gets an unusual move with no coverage collected at all
        (no_coverage_big_move); everything else is routine. 14 of the 15
        days carry some news, comfortably clearing MIN_DAYS_FOR_BASELINE so
        the coverage bar is the real z-test, not the thin-baseline relaxation.
        """
        days = [date(2023, 1, d) for d in range(2, 21) if date(2023, 1, d).weekday() < 5]
        candidate_day, coverage_only_day = days[3], days[6]
        return_only_day, no_coverage_day = days[9], days[12]
        returns = {d: (0.001, 0.0005) for d in days}
        for d in (candidate_day, return_only_day, no_coverage_day):
            returns[d] = (-0.15, 0.0)
        frame = price_frame(returns)

        news = []
        for d in days:
            if d == no_coverage_day:
                continue
            busy = d in (candidate_day, coverage_only_day)
            for k in range(8 if busy else 1):
                news.append(item(d, -0.8 if busy else 0.0, url=f"{d}-{k}"))

        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        return table, {
            "candidate": candidate_day, "coverage_only": coverage_only_day,
            "return_only": return_only_day, "no_coverage": no_coverage_day,
        }

    def test_buckets_every_trading_day_correctly(self):
        table, marked = self._mixed_window()
        diag = flagging_diagnostics(table)
        assert diag["trading_days"] == 15
        assert diag["days_with_news"] == 14
        assert diag["thin_baseline"] is False
        assert diag["candidates"] == 1
        assert diag["coverage_only"] == 1
        assert diag["return_only"] == 1
        assert diag["no_coverage_big_move"] == 1
        assert diag["routine"] == 11
        total = (diag["candidates"] + diag["coverage_only"] + diag["return_only"]
                + diag["no_coverage_big_move"] + diag["routine"])
        assert total == diag["trading_days"]

    def test_candidates_count_matches_rank_incidents(self):
        """The two independent tallies must never disagree with each other."""
        table, marked = self._mixed_window()
        frame = price_frame({d: (0.001, 0.0005) for d in
                             [date(2023, 1, d) for d in range(2, 21)
                              if date(2023, 1, d).weekday() < 5]})
        diag = flagging_diagnostics(table)
        incidents = rank_incidents(table, frame, return_threshold=1.5)
        assert diag["candidates"] == len(incidents)

    def test_empty_table_returns_zeroed_dict_with_all_keys(self):
        diag = flagging_diagnostics(pd.DataFrame())
        assert diag == {
            "trading_days": 0, "days_with_news": 0, "thin_baseline": False,
            "candidates": 0, "coverage_only": 0, "return_only": 0,
            "no_coverage_big_move": 0, "routine": 0,
        }

    def test_thin_baseline_forces_any_coverage_to_count_as_unusual(self):
        """Below MIN_DAYS_FOR_BASELINE, a single ordinary story is enough to
        clear the (relaxed) coverage bar, same relaxation rank_incidents applies."""
        days = [date(2023, 1, d) for d in (23, 24, 25, 27, 30)]
        frame = price_frame({
            days[0]: (0.005, 0.004), days[1]: (0.002, 0.003),
            days[2]: (-0.20, -0.01), days[3]: (-0.02, -0.015), days[4]: (0.01, 0.005),
        })
        news = [item(days[2], -0.9, url=f"n{i}", event="regulatory") for i in range(6)]
        news.append(item(days[0], 0.1, url="quiet"))
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        diag = flagging_diagnostics(table, return_threshold=1.0)
        assert diag["thin_baseline"] is True
        assert diag["candidates"] == 1
        # days[0]'s single quiet story clears the relaxed bar but the move
        # doesn't, so it lands in coverage_only rather than routine.
        assert diag["coverage_only"] == 1

    def test_all_routine_when_nothing_is_unusual(self):
        """Uniform coverage and returns across a window long enough (>=
        MIN_DAYS_FOR_BASELINE) that the real z-test applies rather than the
        thin-baseline relaxation - identical days give every z-score 0.0,
        clearing neither bar, so every day should land as routine."""
        days = [date(2023, 1, d) for d in range(2, 21) if date(2023, 1, d).weekday() < 5]
        frame = price_frame({d: (0.001, 0.0008) for d in days})
        news = [item(d, 0.0, url=f"{d}") for d in days]
        table = build_daily_table(frame, aggregate_by_day(news), days[0], days[-1])
        diag = flagging_diagnostics(table, return_threshold=1.5)
        assert diag["thin_baseline"] is False
        assert diag["candidates"] == 0
        assert diag["routine"] == diag["trading_days"]
