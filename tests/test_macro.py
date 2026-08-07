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
    SkippedFetcher,
    SkippedPriceProvider,
    crude_oil_series,
    fiscal_deficit,
    gsec_yield,
    macro_events_in_window,
    macro_summary,
)
from ceia.prices import PriceError, PriceProvider  # noqa: E402

_NO_MACRO_FETCHER = SkippedFetcher()


class _FakeFetcherResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMacroFetcher:
    """Serves canned HTML for gsec_yield()/fiscal_deficit(), regardless of
    which URL is requested - both only ever hit one URL each."""

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeFetcherResponse:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return _FakeFetcherResponse(self.text)


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


_GSEC_HTML = (
    '<meta id="metaDesc" name="description" content="The yield on India '
    '10Y Bond Yield rose to 6.78% on August 7, 2026, marking a 0.02 '
    'percentage points increase from the previous session."/>'
)
_FISCAL_DEFICIT_HTML = (
    "<p>India's fiscal deficit for 2026-27 is budgeted at Rs 15.69 lakh "
    "crore, which equals 4.4% of GDP. This is a reduction...</p>"
)


class TestGsecYield:
    def test_parses_value_and_as_of_date(self):
        value, note = gsec_yield(fetcher=_FakeMacroFetcher(_GSEC_HTML))
        assert note == ""
        assert value["value"] == pytest.approx(6.78)
        assert value["as_of"] == "August 7, 2026"
        assert value["source"] == "tradingeconomics.com"

    def test_degrades_on_fetch_failure(self):
        value, note = gsec_yield(fetcher=_FakeMacroFetcher(error=RuntimeError("boom")))
        assert value is None
        assert "unavailable" in note

    def test_degrades_when_page_format_changes(self):
        value, note = gsec_yield(fetcher=_FakeMacroFetcher("<html>nothing here</html>"))
        assert value is None
        assert "unavailable" in note


class TestFiscalDeficit:
    def test_parses_year_amount_and_pct_gdp(self):
        value, note = fiscal_deficit(fetcher=_FakeMacroFetcher(_FISCAL_DEFICIT_HTML))
        assert note == ""
        assert value["fiscal_year"] == "2026-27"
        assert value["lakh_crore"] == pytest.approx(15.69)
        assert value["pct_gdp"] == pytest.approx(4.4)
        assert value["source"] == "govtbudget.com"

    def test_degrades_on_fetch_failure(self):
        value, note = fiscal_deficit(fetcher=_FakeMacroFetcher(error=RuntimeError("boom")))
        assert value is None
        assert "unavailable" in note

    def test_degrades_when_page_format_changes(self):
        value, note = fiscal_deficit(fetcher=_FakeMacroFetcher("<html>nothing here</html>"))
        assert value is None
        assert "unavailable" in note


class TestSkippedFetcher:
    def test_raises_immediately_without_touching_the_network(self):
        with pytest.raises(RuntimeError):
            SkippedFetcher().get("https://example.com")

    def test_wired_through_gsec_yield_degrades_cleanly(self):
        value, note = gsec_yield(fetcher=SkippedFetcher())
        assert value is None
        assert "skipped" in note

    def test_wired_through_fiscal_deficit_degrades_cleanly(self):
        value, note = fiscal_deficit(fetcher=SkippedFetcher())
        assert value is None
        assert "skipped" in note


class TestMacroSummary:
    def test_wires_events_and_crude_oil_together(self):
        provider = _FakeProvider(frame=_frame([80.0, 88.0]))
        summary = macro_summary(date(2022, 12, 1), date(2023, 3, 1),
                                provider=provider, fetcher=_NO_MACRO_FETCHER)
        assert any(e["date"] == "2023-02-08" for e in summary["repo_rate_changes"])
        assert summary["crude_oil"]["change"] == pytest.approx(0.1)

    def test_crude_oil_failure_still_returns_events(self):
        provider = _FakeProvider(error=PriceError("boom"))
        summary = macro_summary(date(2022, 12, 1), date(2023, 3, 1),
                                provider=provider, fetcher=_NO_MACRO_FETCHER)
        assert summary["repo_rate_changes"]
        assert "note" in summary["crude_oil"]

    def test_not_available_indicators_are_always_disclosed(self):
        provider = _FakeProvider(frame=_frame([80.0, 88.0]))
        summary = macro_summary(date(2026, 1, 1), date(2026, 1, 2),
                                provider=provider, fetcher=_NO_MACRO_FETCHER)
        assert summary["not_available"] == NOT_AVAILABLE_INDICATORS
        for indicator in ("GDP growth", "CPI inflation", "IIP"):
            assert indicator in summary["not_available"]
        assert "Fiscal deficit" not in summary["not_available"]
        assert "10-year G-Sec yield" not in summary["not_available"]

    def test_gsec_yield_and_fiscal_deficit_are_wired_in(self):
        provider = _FakeProvider(frame=_frame([80.0, 88.0]))
        fetcher = _FakeMacroFetcher(_GSEC_HTML + _FISCAL_DEFICIT_HTML)
        summary = macro_summary(date(2026, 1, 1), date(2026, 1, 2),
                                provider=provider, fetcher=fetcher)
        assert summary["gsec_yield"]["value"] == pytest.approx(6.78)
        assert summary["fiscal_deficit"]["fiscal_year"] == "2026-27"
