"""Tests for the price provider layer.

The network providers cannot be exercised deterministically (Yahoo rate-limits
shared egress IPs), so these cover the parts that are ours: frame shaping,
provider fallback order, and the CSV path that keeps the event study
reproducible when the network is unavailable.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.prices import (  # noqa: E402
    CsvProvider,
    PriceError,
    PriceProvider,
    _as_frame,
    load_prices,
)


@pytest.fixture
def price_csv(tmp_path: Path) -> Path:
    (tmp_path / "ADANIENT.NS.csv").write_text(
        "date,close,volume\n"
        "2023-01-20,3595.35,1200000\n"
        "2023-01-23,3442.75,2400000\n"
        "2023-01-25,3388.75,9800000\n"
        "2023-01-27,2761.45,21000000\n"
    )
    (tmp_path / "_idx_NSEI.csv").write_text(
        "date,close\n"
        "2023-01-20,18027.65\n"
        "2023-01-23,18118.30\n"
        "2023-01-25,17891.95\n"
        "2023-01-27,17604.35\n"
    )
    return tmp_path


class TestFrameShaping:
    def test_index_is_normalised_and_named(self):
        frame = _as_frame(pd.to_datetime(["2023-01-25 15:30", "2023-01-27 09:15"]), [10.0, 11.0])
        assert frame.index.name == "date"
        assert list(frame.index) == [pd.Timestamp("2023-01-25"), pd.Timestamp("2023-01-27")]

    def test_rows_with_no_close_are_dropped(self):
        frame = _as_frame(pd.to_datetime(["2023-01-25", "2023-01-26"]), [10.0, None])
        assert len(frame) == 1

    def test_duplicate_dates_keep_last(self):
        frame = _as_frame(pd.to_datetime(["2023-01-25", "2023-01-25"]), [10.0, 12.0])
        assert len(frame) == 1
        assert frame["close"].iloc[0] == 12.0

    def test_output_is_sorted(self):
        frame = _as_frame(pd.to_datetime(["2023-01-27", "2023-01-20"]), [11.0, 10.0])
        assert frame.index.is_monotonic_increasing


class TestCsvProvider:
    def test_reads_ticker(self, price_csv):
        frame = CsvProvider(price_csv).history("ADANIENT.NS", date(2023, 1, 1), date(2023, 2, 1))
        assert len(frame) == 4
        assert frame["close"].iloc[-1] == pytest.approx(2761.45)

    def test_caret_index_symbol_maps_to_safe_filename(self, price_csv):
        frame = CsvProvider(price_csv).history("^NSEI", date(2023, 1, 1), date(2023, 2, 1))
        assert len(frame) == 4

    def test_window_is_respected(self, price_csv):
        frame = CsvProvider(price_csv).history("ADANIENT.NS", date(2023, 1, 24), date(2023, 1, 26))
        assert list(frame.index.date) == [date(2023, 1, 25)]

    def test_missing_file_raises_price_error(self, tmp_path):
        with pytest.raises(PriceError, match="no CSV"):
            CsvProvider(tmp_path).history("NOPE.NS", date(2023, 1, 1), date(2023, 2, 1))


class TestFallback:
    def test_falls_through_to_the_first_working_provider(self, price_csv):
        class Broken(PriceProvider):
            name = "broken"

            def history(self, symbol, start, end):
                raise PriceError("simulated outage")

        frame, used = load_prices(
            "ADANIENT.NS", date(2023, 1, 1), date(2023, 2, 1),
            providers=[Broken(), CsvProvider(price_csv)],
        )
        assert used == "csv"
        assert len(frame) == 4

    def test_empty_result_is_treated_as_failure(self, price_csv):
        class Empty(PriceProvider):
            name = "empty"

            def history(self, symbol, start, end):
                return pd.DataFrame({"close": []})

        _, used = load_prices(
            "ADANIENT.NS", date(2023, 1, 1), date(2023, 2, 1),
            providers=[Empty(), CsvProvider(price_csv)],
        )
        assert used == "csv"

    def test_error_names_every_provider_tried(self, tmp_path):
        with pytest.raises(PriceError) as excinfo:
            load_prices("NOPE.NS", date(2023, 1, 1), date(2023, 2, 1),
                        providers=[CsvProvider(tmp_path)])
        assert "csv" in str(excinfo.value)
