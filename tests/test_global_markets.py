"""Tests for global market ripple-effect reference data (ceia/global_markets.py).

The core correctness question this module exists for: pairing an NSE
trading day with the *correct* prior/same session of a market that does not
share NSE's trading hours - see the module docstring for the timezone
reasoning behind same_day_available.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.global_markets import (  # noqa: E402
    GLOBAL_INDICES,
    GlobalIndexMeta,
    _align_date,
    load_global_indices,
    same_direction_rate,
)
from ceia.prices import PriceError, PriceProvider  # noqa: E402


def _frame(prices: list[float], start: str) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(prices))
    frame = pd.DataFrame({"close": prices}, index=days)
    frame.index.name = "date"
    return frame


class _MultiSymbolProvider(PriceProvider):
    name = "fake"

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if symbol not in self.frames:
            raise PriceError(f"no data for {symbol}")
        return self.frames[symbol]


class TestAlignDate:
    def test_same_day_available_uses_the_day_itself_when_present(self):
        dates = [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)]
        assert _align_date(date(2026, 5, 12), dates, same_day_available=True) \
            == date(2026, 5, 12)

    def test_same_day_available_falls_back_when_that_day_is_a_holiday(self):
        """Hong Kong closed on a day NSE was open - falls back to the most
        recent Hang Seng trading day before it, not None."""
        dates = [date(2026, 5, 11), date(2026, 5, 13)]  # HK skipped the 12th
        assert _align_date(date(2026, 5, 12), dates, same_day_available=True) \
            == date(2026, 5, 11)

    def test_not_same_day_available_uses_the_prior_day_even_when_present(self):
        """The US/London case: even though the index traded on the NSE day
        itself, that session is not yet complete by NSE's close - the prior
        session is the one actually known."""
        dates = [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)]
        assert _align_date(date(2026, 5, 12), dates, same_day_available=False) \
            == date(2026, 5, 11)

    def test_no_history_early_enough_returns_none(self):
        dates = [date(2026, 5, 12), date(2026, 5, 13)]
        assert _align_date(date(2026, 5, 11), dates, same_day_available=True) is None
        assert _align_date(date(2026, 5, 12), dates, same_day_available=False) is None

    def test_weekend_gap_falls_back_correctly(self):
        """An NSE Monday paired with a lagged (not same_day_available) index
        whose most recent session was the preceding Friday."""
        friday = date(2026, 5, 8)
        monday = date(2026, 5, 11)
        dates = [date(2026, 5, 6), date(2026, 5, 7), friday]  # no weekend sessions
        assert _align_date(monday, dates, same_day_available=False) == friday


class TestLoadGlobalIndices:
    def test_same_day_available_index_aligns_to_the_nse_day_itself(self):
        provider = _MultiSymbolProvider({
            "^HSI": _frame([100.0, 105.0, 110.0], "2026-05-11"),
        })
        nse_days = [date(2026, 5, 13)]
        out = load_global_indices(nse_days, date(2026, 5, 13), date(2026, 5, 13),
                                  providers=[provider],
                                  indices={"Hang Seng": GLOBAL_INDICES["Hang Seng"]})
        hsi = out["Hang Seng"]
        assert hsi.available
        aligned = hsi.aligned[date(2026, 5, 13)]
        assert aligned.aligned_date == date(2026, 5, 13)

    def test_lagged_index_aligns_to_the_prior_day(self):
        provider = _MultiSymbolProvider({
            "^GSPC": _frame([100.0, 105.0, 110.0], "2026-05-11"),
        })
        nse_days = [date(2026, 5, 13)]
        out = load_global_indices(nse_days, date(2026, 5, 13), date(2026, 5, 13),
                                  providers=[provider],
                                  indices={"S&P 500": GLOBAL_INDICES["S&P 500"]})
        spx = out["S&P 500"]
        aligned = spx.aligned[date(2026, 5, 13)]
        assert aligned.aligned_date == date(2026, 5, 12)

    def test_one_index_failing_does_not_sink_the_others(self):
        provider = _MultiSymbolProvider({
            "^HSI": _frame([100.0, 105.0], "2026-05-11"),
        })
        out = load_global_indices(
            [date(2026, 5, 12)], date(2026, 5, 12), date(2026, 5, 12),
            providers=[provider],
            indices={"Hang Seng": GLOBAL_INDICES["Hang Seng"],
                    "Nikkei 225": GLOBAL_INDICES["Nikkei 225"]})
        assert out["Hang Seng"].available
        assert not out["Nikkei 225"].available
        assert "unavailable" in out["Nikkei 225"].note

    def test_window_return_computed_over_the_display_window(self):
        provider = _MultiSymbolProvider({
            "^HSI": _frame([100.0, 110.0, 121.0], "2026-05-11"),
        })
        out = load_global_indices(
            [date(2026, 5, 13)], date(2026, 5, 11), date(2026, 5, 13),
            providers=[provider],
            indices={"Hang Seng": GLOBAL_INDICES["Hang Seng"]})
        assert out["Hang Seng"].window_return == pytest.approx(0.21)

    def test_z_score_reflects_the_indexs_own_return_distribution(self):
        rng = np.random.default_rng(1)
        prices = list(100 * np.cumprod(1 + rng.normal(0, 0.005, 120)))
        prices[-1] = prices[-2] * 1.10  # a genuinely unusual +10% day at the end
        provider = _MultiSymbolProvider({"^HSI": _frame(prices, "2026-01-05")})
        last_day = pd.bdate_range("2026-01-05", periods=120)[-1].date()
        out = load_global_indices(
            [last_day], date(2026, 1, 1), last_day,
            providers=[provider],
            indices={"Hang Seng": GLOBAL_INDICES["Hang Seng"]})
        assert out["Hang Seng"].aligned[last_day].return_z > 3.0

    def test_defaults_to_all_configured_indices(self):
        out = load_global_indices([date(2026, 5, 12)], date(2026, 5, 12),
                                  date(2026, 5, 12), providers=[_MultiSymbolProvider({})])
        assert set(out) == set(GLOBAL_INDICES)

    def test_generic_exception_degrades_not_crashes(self):
        """Not just PriceError - the exception-handling bug once shipped in
        ceia.nifty (see its own tests) must not recur here."""
        class _RaisingProvider(PriceProvider):
            name = "raising"

            def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
                raise ValueError("boom: not a PriceError")

        out = load_global_indices(
            [date(2026, 5, 12)], date(2026, 5, 12), date(2026, 5, 12),
            providers=[_RaisingProvider()],
            indices={"Hang Seng": GLOBAL_INDICES["Hang Seng"]})
        assert not out["Hang Seng"].available
        assert "boom: not a PriceError" in out["Hang Seng"].note


class TestSameDirectionRate:
    def _index(self, aligned_returns: dict, same_day_available=True):
        from ceia.global_markets import GlobalIndexDay, GlobalIndexSeries
        aligned = {day: GlobalIndexDay(aligned_date=day, return_=r, return_z=0.0)
                  for day, r in aligned_returns.items()}
        return GlobalIndexSeries(name="X", ticker="^X", same_day_available=same_day_available,
                                 daily=pd.DataFrame({"close": [1.0]}), aligned=aligned)

    def test_counts_agreement_and_disagreement(self):
        days = [date(2026, 5, 12), date(2026, 5, 13)]
        index = self._index({days[0]: 0.02, days[1]: -0.01})
        company = pd.DataFrame({"return": [0.03, -0.02]}, index=days)
        result = same_direction_rate(index, company, days)
        assert result == {"n": 2, "agree": 2, "rate": 1.0}

    def test_disagreement_lowers_the_rate(self):
        days = [date(2026, 5, 12), date(2026, 5, 13)]
        index = self._index({days[0]: 0.02, days[1]: -0.01})
        company = pd.DataFrame({"return": [-0.03, -0.02]}, index=days)
        result = same_direction_rate(index, company, days)
        assert result == {"n": 2, "agree": 1, "rate": 0.5}

    def test_days_with_no_alignment_are_skipped(self):
        day = date(2026, 5, 12)
        index = self._index({})  # nothing aligned
        company = pd.DataFrame({"return": [0.02]}, index=[day])
        result = same_direction_rate(index, company, [day])
        assert result == {"n": 0, "agree": 0, "rate": None}

    def test_unavailable_index_returns_none_rate(self):
        from ceia.global_markets import GlobalIndexSeries
        index = GlobalIndexSeries(name="X", ticker="^X")  # empty daily -> not available
        result = same_direction_rate(index, pd.DataFrame(), [date(2026, 5, 12)])
        assert result == {"n": 0, "agree": 0, "rate": None}
