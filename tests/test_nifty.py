"""Tests for the Nifty sector index reference data (ceia/nifty.py).

Mirrors the fake-provider pattern already used in test_macro.py and the
secondary-benchmark tests in test_integration.py: one index failing to load
must degrade to a note on that index alone, never sink the others or raise.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
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

_BENCHMARK = "^NSEI"


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


def _market_and_index_frames(n_days: int = 150, seed: int = 3):
    """A benchmark series and a correlated index series, long enough for a
    real market-model fit (needs MIN_ESTIMATION_DAYS=40 observations strictly
    before the analysis window starts) - mirrors test_returns.py's make_frame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n_days)
    market = rng.normal(0.0003, 0.008, n_days)
    index_moves = 0.0002 + 1.1 * market + rng.normal(0, 0.003, n_days)
    bench = pd.DataFrame({"close": 100 * np.cumprod(1 + market)}, index=dates)
    bench.index.name = "date"
    index = pd.DataFrame({"close": 100 * np.cumprod(1 + index_moves)}, index=dates)
    index.index.name = "date"
    return dates, bench, index


class TestNiftyIndicesConstant:
    def test_six_indices_configured(self):
        assert set(NIFTY_INDICES) == {
            "Nifty 50", "Nifty Bank", "Nifty Auto", "Nifty Energy",
            "Nifty IT", "Nifty Metal",
        }
        assert all(t.startswith("^") for t in NIFTY_INDICES.values())


class TestLoadNiftyIndices:
    """The descriptive-only path: no candidate_days passed, so no event
    study is attempted - just the rebased price path and window return."""

    def test_available_index_gets_rebased_daily_and_window_return(self):
        provider = _MultiSymbolProvider({
            "^NSEI": _frame([100.0, 110.0, 121.0]),
        })
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4), _BENCHMARK,
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI"})
        nifty50 = out["Nifty 50"]
        assert nifty50.available
        assert nifty50.provider == "fake"
        assert nifty50.daily["level"].iloc[0] == pytest.approx(100.0)
        assert nifty50.daily["level"].iloc[-1] == pytest.approx(121.0)
        assert nifty50.window_return == pytest.approx(0.21)
        assert nifty50.incident_stats == {}

    def test_unavailable_index_degrades_with_a_note(self):
        provider = _MultiSymbolProvider({})
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4), _BENCHMARK,
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
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 3), _BENCHMARK,
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI", "Nifty IT": "^CNXIT"})
        assert out["Nifty 50"].available
        assert not out["Nifty IT"].available

    def test_no_trading_days_in_window_is_reported_not_a_crash(self):
        provider = _MultiSymbolProvider({"^NSEI": _frame([100.0, 105.0])})
        out = load_nifty_indices(date(2030, 1, 1), date(2030, 1, 5), _BENCHMARK,
                                 providers=[provider],
                                 indices={"Nifty 50": "^NSEI"})
        assert not out["Nifty 50"].available
        assert "no trading days" in out["Nifty 50"].note

    def test_defaults_to_all_six_indices_when_none_specified(self):
        out = load_nifty_indices(date(2023, 1, 2), date(2023, 1, 4), _BENCHMARK,
                                 providers=[_MultiSymbolProvider({})])
        assert set(out) == set(NIFTY_INDICES)


class TestIndexEventStudy:
    """Each index gets its own real market-model event study, anchored to
    the days already flagged for the company (candidate_days) rather than
    an independent search for the index's own events - see the module
    docstring for why."""

    def _provider(self, n_days=150, seed=3):
        dates, bench, index = _market_and_index_frames(n_days=n_days, seed=seed)
        provider = _MultiSymbolProvider({_BENCHMARK: bench, "^CNXAUTO": index})
        return provider, dates

    def test_incident_stats_computed_for_each_candidate_day(self):
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        candidate_day = dates[105].date()
        out = load_nifty_indices(start, end, _BENCHMARK,
                                 candidate_days=[candidate_day],
                                 event_window=(-1, 1), providers=[provider],
                                 indices={"Nifty Auto": "^CNXAUTO"})
        auto = out["Nifty Auto"]
        assert auto.available
        assert auto.model_kind in ("market-model", "market-adjusted")
        assert candidate_day.isoformat() in auto.incident_stats
        stats = auto.incident_stats[candidate_day.isoformat()]
        assert set(stats) >= {
            "abnormal_return", "abnormal_return_z", "car", "days",
            "t_stat", "p_value_t", "p_value", "n",
        }

    def test_only_candidate_days_get_stats_not_every_day(self):
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        candidate_day = dates[105].date()
        out = load_nifty_indices(start, end, _BENCHMARK,
                                 candidate_days=[candidate_day],
                                 providers=[provider],
                                 indices={"Nifty Auto": "^CNXAUTO"})
        assert len(out["Nifty Auto"].incident_stats) == 1

    def test_index_matching_the_benchmark_gets_no_event_study(self):
        """Regressing a series on itself gives ~zero residual variance - a
        z-score there would divide by ~0, so this index is deliberately
        skipped rather than producing nonsense."""
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        candidate_day = dates[105].date()
        out = load_nifty_indices(start, end, _BENCHMARK,
                                 candidate_days=[candidate_day],
                                 providers=[provider],
                                 indices={"Nifty 50": _BENCHMARK})
        nifty50 = out["Nifty 50"]
        assert nifty50.available  # descriptive stats still populate
        assert nifty50.incident_stats == {}
        assert "same ticker as the primary benchmark" in nifty50.event_study_note

    def test_no_candidate_days_skips_the_event_study_entirely(self):
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        out = load_nifty_indices(start, end, _BENCHMARK, candidate_days=[],
                                 providers=[provider],
                                 indices={"Nifty Auto": "^CNXAUTO"})
        auto = out["Nifty Auto"]
        assert auto.available
        assert auto.incident_stats == {}
        assert auto.model_kind == ""

    def test_bad_index_ticker_degrades_without_sinking_the_run(self):
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        candidate_day = dates[105].date()
        out = load_nifty_indices(start, end, _BENCHMARK,
                                 candidate_days=[candidate_day],
                                 providers=[provider],
                                 indices={"Nifty Metal": "^CNXMETAL"})
        metal = out["Nifty Metal"]
        assert not metal.available
        assert "unavailable" in metal.note

    def test_candidate_day_outside_the_window_is_skipped_not_crashed(self):
        provider, dates = self._provider()
        start, end = dates[100].date(), dates[110].date()
        outside_day = dates[5].date()
        out = load_nifty_indices(start, end, _BENCHMARK,
                                 candidate_days=[outside_day],
                                 providers=[provider],
                                 indices={"Nifty Auto": "^CNXAUTO"})
        assert out["Nifty Auto"].incident_stats == {}


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
