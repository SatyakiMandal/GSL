"""Tests for the macro-economic backdrop (ceia/macro.py).

Repo rate changes are a small, source-cited static table - these tests pin
the window-filtering logic and the disclosed-gap list, not the historical
values themselves (those are a data-accuracy concern, not a code-correctness
one, and are cross-checked in the module docstring rather than here).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.macro import (  # noqa: E402
    NOT_AVAILABLE_INDICATORS,
    REPO_RATE_CHANGES,
    SkippedPriceProvider,
    crude_oil_series,
    macro_events_in_window,
    macro_summary,
)
from ceia.prices import PriceError, PriceProvider  # noqa: E402


class _FakeProvider(PriceProvider):
    name = "fake"

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None):
        self.frame = frame
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error
        return self.frame


def _frame(prices: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    days = pd.date_range(start, periods=len(prices))
    frame = pd.DataFrame({"close": prices}, index=days)
    frame.index.name = "date"
    return frame


class TestMacroEventsInWindow:
    def test_finds_events_inside_the_window(self):
        events = macro_events_in_window(date(2022, 12, 1), date(2023, 3, 1))
        dates = {e.day for e in events}
        assert date(2022, 12, 7) in dates
        assert date(2023, 2, 8) in dates

    def test_excludes_events_outside_the_window(self):
        events = macro_events_in_window(date(2023, 3, 1), date(2023, 12, 1))
        assert events == []

    def test_boundaries_are_inclusive(self):
        events = macro_events_in_window(date(2023, 2, 8), date(2023, 2, 8))
        assert len(events) == 1
        assert events[0].day == date(2023, 2, 8)

    def test_label_states_the_new_rate(self):
        events = macro_events_in_window(date(2023, 2, 8), date(2023, 2, 8))
        assert "6.50%" in events[0].label

    def test_every_entry_is_a_genuine_change_not_a_repeat(self):
        """A hold is not an event - consecutive table entries must differ."""
        rates = [rate for _, rate in REPO_RATE_CHANGES]
        for prev, cur in zip(rates, rates[1:]):
            assert prev != cur

    def test_table_is_chronological(self):
        dates = [day for day, _ in REPO_RATE_CHANGES]
        assert dates == sorted(dates)


class TestCrudeOilSeries:
    def test_returns_the_frame_on_success(self):
        provider = _FakeProvider(frame=_frame([80.0, 81.0, 79.5]))
        frame, note = crude_oil_series(date(2026, 1, 1), date(2026, 1, 3), provider=provider)
        assert frame is not None and note == ""
        assert provider.calls == [("BZ=F", date(2026, 1, 1), date(2026, 1, 3))]

    def test_degrades_to_none_on_price_error(self):
        provider = _FakeProvider(error=PriceError("BZ=F: HTTP 429"))
        frame, note = crude_oil_series(date(2026, 1, 1), date(2026, 1, 3), provider=provider)
        assert frame is None
        assert "unavailable" in note and "429" in note

    def test_degrades_to_none_on_empty_frame(self):
        provider = _FakeProvider(frame=pd.DataFrame(columns=["close"]))
        frame, note = crude_oil_series(date(2026, 1, 1), date(2026, 1, 3), provider=provider)
        assert frame is None
        assert "unavailable" in note


class TestSkippedPriceProvider:
    def test_raises_immediately_without_touching_the_network(self):
        with pytest.raises(PriceError):
            SkippedPriceProvider().history("BZ=F", date(2026, 1, 1), date(2026, 1, 2))

    def test_wired_through_crude_oil_series_degrades_cleanly(self):
        frame, note = crude_oil_series(date(2026, 1, 1), date(2026, 1, 2),
                                       provider=SkippedPriceProvider())
        assert frame is None
        assert "skipped" in note


class TestMacroSummary:
    def test_wires_events_and_crude_oil_together(self):
        provider = _FakeProvider(frame=_frame([80.0, 88.0]))
        summary = macro_summary(date(2022, 12, 1), date(2023, 3, 1), provider=provider)
        assert any(e["date"] == "2023-02-08" for e in summary["repo_rate_changes"])
        assert summary["crude_oil"]["change"] == pytest.approx(0.1)

    def test_crude_oil_failure_still_returns_events(self):
        provider = _FakeProvider(error=PriceError("boom"))
        summary = macro_summary(date(2022, 12, 1), date(2023, 3, 1), provider=provider)
        assert summary["repo_rate_changes"]
        assert "note" in summary["crude_oil"]

    def test_not_available_indicators_are_always_disclosed(self):
        provider = _FakeProvider(frame=_frame([80.0, 88.0]))
        summary = macro_summary(date(2026, 1, 1), date(2026, 1, 2), provider=provider)
        assert summary["not_available"] == NOT_AVAILABLE_INDICATORS
        for indicator in ("GDP growth", "CPI inflation", "IIP",
                          "Fiscal deficit", "10-year G-Sec yield"):
            assert indicator in summary["not_available"]
