"""Tests for the Nifty sector index reference data (ceia/nifty.py).

Mirrors the fake-provider pattern already used in test_macro.py and the
secondary-benchmark tests in test_integration.py: one index failing to load
must degrade to a note on that index alone, never sink the others or raise.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.nifty import (  # noqa: E402
    NIFTY_INDICES,
    IndexSeries,
    load_nifty_indices,
    same_direction_rate,
)
from ceia.prices import PriceError, PriceProvider  # noqa: E402


def _frame(prices: list[float], start: str = "2023-01-02") -> pd.DataFrame:
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


class TestNiftyIndicesConstant:
    def test_six_indices_configured(self):
        assert set(NIFTY_INDICES) == {
            "Nifty 50", "Nifty Bank", "Nifty Auto", "Nifty Energy",
            "Nifty IT", "Nifty Metal",
        }
        assert all(t.startswith("^") for t in NIFTY_INDICES.values())


class TestLoadNiftyIndices:
    def test_available_index_gets_rebased_daily_and_window_return(self):
        provider = _MultiSymbolProvider({
            "^NSEI": _frame([100.0, 110.0, 121.0]),
        })
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4),
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI"})
        nifty50 = out["Nifty 50"]
        assert nifty50.available
        assert nifty50.provider == "fake"
        assert nifty50.daily["level"].iloc[0] == pytest.approx(100.0)
        assert nifty50.daily["level"].iloc[-1] == pytest.approx(121.0)
        assert nifty50.window_return == pytest.approx(0.21)

    def test_unavailable_index_degrades_with_a_note(self):
        provider = _MultiSymbolProvider({})
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4),
                                 providers=[provider],
                                 indices={"Nifty Metal": "^CNXMETAL"})
        metal = out["Nifty Metal"]
        assert not metal.available
        assert "unavailable" in metal.note
        assert metal.daily.empty

    def test_one_failure_does_not_sink_the_others(self):
        provider = _MultiSymbolProvider({
            "^NSEI": _frame([100.0, 105.0]),
        })
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 3),
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI", "Nifty IT": "^CNXIT"})
        assert out["Nifty 50"].available
        assert not out["Nifty IT"].available

    def test_no_trading_days_in_window_is_reported_not_a_crash(self):
        provider = _MultiSymbolProvider({"^NSEI": _frame([100.0, 105.0])})
        out = load_nifty_indices(date(2030, 1, 1), date(2030, 1, 5),
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI"})
        assert not out["Nifty 50"].available
        assert "no trading days" in out["Nifty 50"].note

    def test_defaults_to_all_six_indices_when_none_specified(self):
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4),
                                 providers=[_MultiSymbolProvider({})])
        assert set(out) == set(NIFTY_INDICES)


class TestSameDirectionRate:
    def _index(self, returns: dict) -> IndexSeries:
        daily = pd.DataFrame({"return": list(returns.values())},
                             index=pd.DatetimeIndex(list(returns.keys())))
        return IndexSeries(name="X", ticker="^X", provider="fake", daily=daily,
                           window_return=0.0)

    def test_counts_agreement_and_disagreement(self):
        d1, d2, d3 = date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)
        index = self._index({d1: 0.01, d2: -0.02, d3: 0.03})
        company = pd.DataFrame(
            {"return": [0.02, -0.01, -0.05]},
            index=[d1, d2, d3],
        )
        result = same_direction_rate(index, company, [d1, d2, d3])
        assert result["n"] == 3
        assert result["agree"] == 2  # d1 and d2 agree in sign, d3 doesn't
        assert result["rate"] == pytest.approx(2 / 3)

    def test_no_candidate_days_returns_zero_without_crashing(self):
        index = self._index({date(2023, 1, 2): 0.01})
        result = same_direction_rate(index, pd.DataFrame(), [])
        assert result == {"n": 0, "agree": 0, "rate": None}

    def test_unavailable_index_returns_zero(self):
        index = IndexSeries(name="X", ticker="^X", note="unavailable: no data")
        result = same_direction_rate(index, pd.DataFrame(), [date(2023, 1, 2)])
        assert result == {"n": 0, "agree": 0, "rate": None}

    def test_day_missing_from_either_side_is_skipped_not_crashed(self):
        d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
        index = self._index({d1: 0.01})
        company = pd.DataFrame({"return": [0.02]}, index=[d1])
        result = same_direction_rate(index, company, [d1, d2])
        assert result["n"] == 1
        assert result["agree"] == 1
