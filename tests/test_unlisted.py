"""Tests for the unlisted/pre-IPO price-move pipeline (ceia/unlisted.py).

UnlistedZone's prices are indicative, team-compiled estimates held flat on
the chart between real revisions - not daily price discovery. These tests
pin the two things that make that honest rather than misleading: parsing
the real embedded series correctly, and collapsing the forward-filled
display series back down to only the days something actually changed.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.extract import IST  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402
from ceia.unlisted import (  # noqa: E402
    PriceMove,
    UnlistedCompanyNotFoundError,
    UnlistedPriceError,
    _fetch_directory,
    _match_score,
    _normalise_for_match,
    analyse_unlisted,
    attach_news_to_moves,
    fetch_price_series,
    price_moves,
    real_updates,
    resolve_unlisted_url,
)


def item(day: date, sentiment: float = 0.0, *, url="u", source="et",
        headline="h", relevance=0.9) -> NewsItem:
    return NewsItem(source=source, url=url, headline=headline,
                    published_at=datetime(day.year, day.month, day.day, 10, 0, tzinfo=IST),
                    sentiment_score=sentiment, relevance_score=relevance,
                    sentiment_label="positive" if sentiment > 0.15
                    else "negative" if sentiment < -0.15 else "neutral")


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFetcher:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def get(self, url: str) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self.text)


def _directory_page(cards: list[tuple[str, str]], last_page: int | None = None) -> str:
    """A minimal stand-in for one page of UnlistedZone's real /shares
    directory: a grid of company cards, each holding a name (class="nm")
    and a link to its product page (class="det" href="..."), plus the
    embedded ``"lastPage":N`` marker page 1's real response carries."""
    cards_html = "".join(
        f'<div class="card"><div class="nm">{name}</div>'
        f'<a class="det" href="/shares/{slug}">Details</a></div>'
        for name, slug in cards
    )
    marker = f'{{"lastPage":{last_page}}}' if last_page is not None else ""
    return f"<html><body>{cards_html}</body>{marker}</html>"


class _FakeDirectoryFetcher:
    """Unlike _FakeFetcher, serves distinct content per exact URL - needed
    to test pagination, where each page's request must return different
    cards."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.pages[url])


def _rsc_page(pairs: list[tuple[str, float]]) -> str:
    """A minimal stand-in for UnlistedZone's real page shape: the series
    embedded as a backslash-escaped JSON string inside a Next.js RSC
    streaming payload, verified directly against a real downloaded page."""
    body = ",".join(f'[\\"{d}\\",{v:g}]' for d, v in pairs)
    return (
        '<script>self.__next_f.push([1,"...,{\\"series\\":[' + body
        + '],\\"asOf\\":\\"2026-08-06\\"}..."])</script>'
    )


class TestFetchPriceSeries:
    def test_parses_the_embedded_series(self):
        html = _rsc_page([("2026-01-01", 740), ("2026-01-02", 740),
                          ("2026-01-03", 812.5)])
        fetcher = _FakeFetcher(html)
        series = fetch_price_series(fetcher, "https://unlistedzone.com/shares/x")
        assert list(series.index.date) == [date(2026, 1, 1), date(2026, 1, 2),
                                           date(2026, 1, 3)]
        assert series["close"].tolist() == [740.0, 740.0, 812.5]

    def test_no_series_found_raises(self):
        fetcher = _FakeFetcher("<html>nothing here</html>")
        with pytest.raises(UnlistedPriceError):
            fetch_price_series(fetcher, "https://unlistedzone.com/shares/x")

    def test_duplicate_dates_keep_the_last_occurrence(self):
        """The RSC payload can legitimately repeat a date across separate
        pushed chunks; the later value is the one actually rendered."""
        html = _rsc_page([("2026-01-01", 100), ("2026-01-01", 105)])
        fetcher = _FakeFetcher(html)
        series = fetch_price_series(fetcher, "https://unlistedzone.com/shares/x")
        assert len(series) == 1
        assert series["close"].iloc[0] == 105.0


class TestRealUpdates:
    def _series(self, values: list[float]) -> pd.DataFrame:
        days = pd.date_range("2026-01-01", periods=len(values))
        frame = pd.DataFrame({"close": values}, index=days)
        frame.index.name = "date"
        return frame

    def test_collapses_a_forward_filled_run(self):
        series = self._series([740, 740, 740, 740, 812.5, 812.5])
        real = real_updates(series)
        assert real["close"].tolist() == [740, 812.5]
        assert list(real.index.date) == [date(2026, 1, 1), date(2026, 1, 5)]

    def test_first_observation_always_kept(self):
        series = self._series([100])
        assert len(real_updates(series)) == 1

    def test_every_day_different_keeps_everything(self):
        series = self._series([100, 101, 99, 105])
        assert len(real_updates(series)) == 4

    def test_a_value_returning_to_a_previous_level_still_counts_as_real(self):
        """100 -> 110 -> 100 is two real moves, not a collapse back to the
        original single point - ne(shift(1)) compares only to the
        immediately preceding row, not the whole run's starting value."""
        series = self._series([100, 100, 110, 110, 100])
        real = real_updates(series)
        assert real["close"].tolist() == [100, 110, 100]


class TestPriceMoves:
    def test_one_move_per_consecutive_real_pair(self):
        days = pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01"])
        real = pd.DataFrame({"close": [100.0, 120.0, 90.0]}, index=days)
        moves = price_moves(real)
        assert len(moves) == 2
        assert moves[0].start_date == date(2026, 1, 1)
        assert moves[0].end_date == date(2026, 1, 15)
        assert moves[0].change == pytest.approx(0.20)
        assert moves[1].change == pytest.approx(-0.25)

    def test_single_observation_produces_no_moves(self):
        real = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2026-01-01"]))
        assert price_moves(real) == []


class TestAttachNewsToMoves:
    def _moves(self) -> list[PriceMove]:
        return [
            PriceMove(date(2026, 1, 1), date(2026, 1, 15), 100.0, 120.0),
            PriceMove(date(2026, 1, 15), date(2026, 2, 1), 120.0, 90.0),
        ]

    def test_news_goes_to_the_move_whose_window_it_falls_in(self):
        moves = self._moves()
        items = [item(date(2026, 1, 10), headline="mid-first-window"),
                 item(date(2026, 1, 20), headline="mid-second-window")]
        attach_news_to_moves(moves, items)
        assert [h["headline"] for h in moves[0].headlines] == ["mid-first-window"]
        assert [h["headline"] for h in moves[1].headlines] == ["mid-second-window"]

    def test_window_is_exclusive_of_start_and_inclusive_of_end(self):
        """News on the boundary date belongs to the move that just
        finished (already explained), not the one about to start - except
        at the very end date itself, which belongs to the move that ends
        there."""
        moves = self._moves()
        items = [item(date(2026, 1, 15), headline="on-the-boundary")]
        attach_news_to_moves(moves, items)
        assert [h["headline"] for h in moves[0].headlines] == ["on-the-boundary"]
        assert moves[1].headlines == []

    def test_news_outside_every_window_is_attached_nowhere(self):
        moves = self._moves()
        items = [item(date(2025, 12, 1), headline="too-early"),
                 item(date(2026, 3, 1), headline="too-late")]
        attach_news_to_moves(moves, items)
        assert moves[0].headlines == [] and moves[1].headlines == []

    def test_limit_keeps_the_strongest_signals(self):
        moves = [PriceMove(date(2026, 1, 1), date(2026, 1, 15), 100.0, 120.0)]
        items = [item(date(2026, 1, 5), sentiment=s, url=f"u{i}")
                for i, s in enumerate([0.1, 0.9, -0.9, 0.3, 0.5, -0.2])]
        attach_news_to_moves(moves, items, limit=2)
        assert len(moves[0].headlines) == 2


class TestUnlistedAnalysisIntegration:
    def test_analyse_unlisted_wires_price_and_news_together(self):
        html = _rsc_page([("2026-01-01", 740), ("2026-01-02", 740),
                          ("2026-01-15", 812.5)])
        fetcher = _FakeFetcher(html)
        config = RunConfig(company="Test Unlisted Co", ticker="",
                           start=date(2026, 1, 1), end=date(2026, 1, 31))
        items = [item(date(2026, 1, 8), headline="funding round announced")]
        analysis = analyse_unlisted(config, "https://unlistedzone.com/shares/x",
                                    fetcher=fetcher, items=items, news_meta={})
        assert len(analysis.moves) == 1
        assert analysis.moves[0].change == pytest.approx((812.5 - 740) / 740)
        assert [h["headline"] for h in analysis.moves[0].headlines] == \
            ["funding round announced"]
        assert analysis.unattributed == []

    def test_moves_are_scoped_to_the_requested_window(self):
        """UnlistedZone's chart commonly reaches back years further than any
        one run asks for - a move entirely outside the requested window must
        not appear in the output, even though the full history is still
        needed internally to compute the moves that do overlap it."""
        html = _rsc_page([
            ("2019-01-01", 300), ("2019-06-01", 350),    # entirely in 2019
            ("2026-01-01", 740), ("2026-01-15", 812.5),  # inside the window
            ("2026-06-01", 900), ("2026-06-15", 950),    # entirely after the window
        ])
        fetcher = _FakeFetcher(html)
        config = RunConfig(company="Test Unlisted Co", ticker="",
                           start=date(2026, 1, 1), end=date(2026, 1, 31))
        analysis = analyse_unlisted(config, "https://unlistedzone.com/shares/x",
                                    fetcher=fetcher, items=[], news_meta={})
        for move in analysis.moves:
            assert move.end_date >= config.start
            assert move.start_date <= config.end
        assert not any(m.end_date < date(2019, 12, 31) for m in analysis.moves)
        assert not any(m.start_date > date(2026, 6, 1) for m in analysis.moves)
        assert any(m.start_date == date(2026, 1, 1) and m.end_date == date(2026, 1, 15)
                  for m in analysis.moves)

    def test_a_move_that_started_before_the_window_but_ends_inside_it_is_kept(self):
        """The price in effect at the start of the window is still the last
        real revision before it, even if that revision itself predates the
        window - dropping it would silently discard the window's opening
        move rather than reporting it truncated at the boundary."""
        html = _rsc_page([("2025-12-01", 500), ("2026-01-10", 600)])
        fetcher = _FakeFetcher(html)
        config = RunConfig(company="Test Unlisted Co", ticker="",
                           start=date(2026, 1, 1), end=date(2026, 1, 31))
        analysis = analyse_unlisted(config, "https://unlistedzone.com/shares/x",
                                    fetcher=fetcher, items=[], news_meta={})
        assert len(analysis.moves) == 1
        assert analysis.moves[0].start_date == date(2025, 12, 1)
        assert analysis.moves[0].end_date == date(2026, 1, 10)

    def test_unattributed_items_are_separated_out(self):
        html = _rsc_page([("2026-01-01", 100), ("2026-01-15", 110)])
        fetcher = _FakeFetcher(html)
        config = RunConfig(company="Test Unlisted Co", ticker="",
                           start=date(2026, 1, 1), end=date(2026, 1, 31))
        no_timestamp = NewsItem(source="et", url="u", headline="no timestamp",
                                published_at=None)
        analysis = analyse_unlisted(config, "https://unlistedzone.com/shares/x",
                                    fetcher=fetcher, items=[no_timestamp], news_meta={})
        assert analysis.unattributed == [no_timestamp]

    def test_ranked_moves_sorts_by_absolute_change(self):
        html = _rsc_page([("2026-01-01", 100), ("2026-01-08", 105),
                          ("2026-01-15", 80), ("2026-01-22", 84)])
        fetcher = _FakeFetcher(html)
        config = RunConfig(company="Test Unlisted Co", ticker="",
                           start=date(2026, 1, 1), end=date(2026, 1, 31))
        analysis = analyse_unlisted(config, "https://unlistedzone.com/shares/x",
                                    fetcher=fetcher, items=[], news_meta={})
        ranked = analysis.ranked_moves()
        assert ranked[0].change == pytest.approx((80 - 105) / 105)  # the -23.8% move
        assert analysis.ranked_moves(top_n=1) == ranked[:1]


class TestFetchDirectory:
    def test_parses_a_single_page_of_cards(self):
        html = _directory_page([("Alpha Pvt Ltd", "alpha"),
                                ("Beta Unlisted Shares", "beta")])
        fetcher = _FakeFetcher(html)
        directory = _fetch_directory(fetcher, max_pages=1)
        assert directory == [
            ("Alpha Pvt Ltd", "https://unlistedzone.com/shares/alpha"),
            ("Beta Unlisted Shares", "https://unlistedzone.com/shares/beta"),
        ]

    def test_follows_pagination_via_the_last_page_marker(self):
        page1 = _directory_page([("Alpha", "alpha"), ("Beta", "beta")], last_page=2)
        page2 = _directory_page([("Gamma", "gamma"), ("Delta", "delta")])
        fetcher = _FakeDirectoryFetcher({
            "https://unlistedzone.com/shares": page1,
            "https://unlistedzone.com/shares?page=2": page2,
        })
        directory = _fetch_directory(fetcher, max_pages=20)
        assert [name for name, _ in directory] == ["Alpha", "Beta", "Gamma", "Delta"]
        assert fetcher.calls == [
            "https://unlistedzone.com/shares",
            "https://unlistedzone.com/shares?page=2",
        ]

    def test_stops_when_a_page_has_no_cards(self):
        """A page beyond the real directory's end (e.g. a stale lastPage
        marker) must stop the walk rather than requesting further pages
        that were never registered with the fake fetcher."""
        page1 = _directory_page([("Alpha", "alpha")], last_page=3)
        page2 = _directory_page([])
        fetcher = _FakeDirectoryFetcher({
            "https://unlistedzone.com/shares": page1,
            "https://unlistedzone.com/shares?page=2": page2,
        })
        directory = _fetch_directory(fetcher, max_pages=20)
        assert [name for name, _ in directory] == ["Alpha"]
        assert fetcher.calls == [
            "https://unlistedzone.com/shares",
            "https://unlistedzone.com/shares?page=2",
        ]

    def test_dedupes_by_product_page_href(self):
        html = _directory_page([("Alpha Pvt Ltd", "alpha"), ("ALPHA PVT LTD", "alpha")])
        fetcher = _FakeFetcher(html)
        directory = _fetch_directory(fetcher, max_pages=1)
        assert len(directory) == 1
        assert directory[0][0] == "Alpha Pvt Ltd"


class TestNormaliseForMatch:
    def test_strips_directory_boilerplate_and_lowercases(self):
        assert _normalise_for_match("NSE India Limited Unlisted Shares") == "nse india"

    def test_strips_punctuation(self):
        assert _normalise_for_match("Sonata Software Ltd.") == "sonata software"


class TestMatchScore:
    def test_containment_scores_perfect(self):
        assert _match_score("sonata", "sonata software") == 1.0

    def test_empty_inputs_score_zero(self):
        assert _match_score("", "sonata software") == 0.0
        assert _match_score("sonata", "") == 0.0

    def test_near_miss_scores_between_zero_and_one(self):
        score = _match_score("sonta software", "sonata software")
        assert 0.0 < score < 1.0

    def test_unrelated_strings_score_low(self):
        assert _match_score("sonata software", "totally different company") < 0.5


class TestResolveUnlistedUrl:
    def test_resolves_the_best_matching_company(self):
        html = _directory_page([("Sonata Software Limited", "sonata-software"),
                                ("Sona Comstar Limited", "sona-comstar")])
        fetcher = _FakeFetcher(html)
        url = resolve_unlisted_url("Sonata Software", fetcher, max_pages=1)
        assert url == "https://unlistedzone.com/shares/sonata-software"

    def test_raises_when_no_candidate_is_a_confident_match(self):
        html = _directory_page([("Totally Unrelated Company", "x")])
        fetcher = _FakeFetcher(html)
        with pytest.raises(UnlistedCompanyNotFoundError):
            resolve_unlisted_url("Sonata Software", fetcher, max_pages=1)

    def test_raises_when_the_directory_cannot_be_read(self):
        fetcher = _FakeFetcher("<html>nothing here</html>")
        with pytest.raises(UnlistedCompanyNotFoundError):
            resolve_unlisted_url("Sonata Software", fetcher, max_pages=1)
