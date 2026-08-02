"""Tests for the pure logic inside the Streamlit GUI.

``ceia/gui.py`` is mostly page-building calls (``st.form``, ``st.columns``,
...) that only make sense inside a running Streamlit session, so this file
does not try to drive the UI. It covers ``_price_providers``, the one function
in that module with branching logic worth pinning: which provider chain gets
built from what the form supplied.

Requires the optional ``streamlit`` dependency (the ``gui`` extra); skipped
entirely if it is not installed, same as the sentiment tests skip without
``torch``/``transformers``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.gui import _price_providers  # noqa: E402
from ceia.prices import AlphaVantageProvider, CsvProvider, YahooChartProvider, YFinanceProvider  # noqa: E402


class _FakeUpload:
    """Stands in for Streamlit's UploadedFile, which needs a live session."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def getvalue(self) -> bytes:
        return self._buf.getvalue()


class TestPriceProviders:
    def test_no_csvs_and_no_key_falls_back_to_network_chain(self):
        providers = _price_providers("ADANIENT.NS", "^NSEI", "", None, None)
        kinds = [type(p) for p in providers]
        assert kinds == [YFinanceProvider, YahooChartProvider, CsvProvider]

    def test_api_key_inserts_alpha_vantage_before_the_local_csv_fallback(self):
        providers = _price_providers("ADANIENT.NS", "^NSEI", "demo-key", None, None)
        kinds = [type(p) for p in providers]
        assert kinds == [YFinanceProvider, YahooChartProvider, AlphaVantageProvider, CsvProvider]

    def test_both_csvs_uploaded_skips_straight_to_them(self):
        ticker_csv = _FakeUpload(b"date,close\n2023-01-20,100\n")
        benchmark_csv = _FakeUpload(b"date,close\n2023-01-20,200\n")
        providers = _price_providers("ADANIENT.NS", "^NSEI", "demo-key", ticker_csv, benchmark_csv)
        assert len(providers) == 1
        assert isinstance(providers[0], CsvProvider)

    def test_only_one_csv_uploaded_still_uses_the_network_chain(self):
        """Both files are required - a lone upload should not half-activate CSV mode."""
        ticker_csv = _FakeUpload(b"date,close\n2023-01-20,100\n")
        providers = _price_providers("ADANIENT.NS", "^NSEI", "", ticker_csv, None)
        kinds = [type(p) for p in providers]
        assert kinds == [YFinanceProvider, YahooChartProvider, CsvProvider]

    def test_uploaded_csvs_are_named_for_the_ticker_and_benchmark(self):
        ticker_csv = _FakeUpload(b"date,close\n2023-01-20,100\n")
        benchmark_csv = _FakeUpload(b"date,close\n2023-01-20,200\n")
        providers = _price_providers("ADANIENT.NS", "^NSEI", "", ticker_csv, benchmark_csv)
        csv_dir = providers[0].directory
        assert (csv_dir / "ADANIENT.NS.csv").exists()
        # `^` is not a portable filename character, same convention CsvProvider uses.
        assert (csv_dir / "_idx_NSEI.csv").exists()
