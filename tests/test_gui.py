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
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.gui import _StreamlitLogHandler, _price_providers  # noqa: E402
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


class _FakePlaceholder:
    """Stands in for the st.empty() placeholder - records what would have
    been rendered without needing a live Streamlit session."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def code(self, text: str) -> None:
        self.calls.append(text)


class TestStreamlitLogHandler:
    """Regression coverage for a real bug found by actually running a live
    scrape through the GUI (Playwright driving the running app, not just
    reading the source): discovery.discover() and ingest.run() both fan out
    across a ThreadPoolExecutor, so this handler's emit() is called from
    worker threads Streamlit never spawned and that have no
    ScriptRunContext. That produced an intermittent crash inside Streamlit's
    own internals - racy enough that most calls only logged an "ignorable"
    warning while some actually raised and took the whole analysis down.
    Fixed via add_script_run_ctx(); these tests exercise the concurrent path
    that triggered it, not just the single-threaded happy path.
    """

    def test_single_threaded_emit_renders_and_caps_at_60_lines(self):
        placeholder = _FakePlaceholder()
        handler = _StreamlitLogHandler(placeholder, ctx=None)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("ceia.test.single")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for i in range(75):
                logger.info("line %d", i)
        finally:
            logger.removeHandler(handler)

        assert len(handler.lines) == 75
        rendered = placeholder.calls[-1].splitlines()
        assert len(rendered) == 60
        assert rendered[-1] == "line 74", "most recent line must survive the cap"

    def test_emit_attaches_the_captured_context_to_whichever_thread_calls_it(self):
        placeholder = _FakePlaceholder()
        sentinel_ctx = object()
        handler = _StreamlitLogHandler(placeholder, ctx=sentinel_ctx)
        handler.setFormatter(logging.Formatter("%(message)s"))

        seen = []
        with patch("ceia.gui.add_script_run_ctx",
                   side_effect=lambda thread, ctx: seen.append((thread, ctx))):
            worker_thread_holder = {}

            def emit_from_worker():
                worker_thread_holder["thread"] = threading.current_thread()
                logger = logging.getLogger("ceia.test.ctx")
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                try:
                    logger.info("from a worker thread")
                finally:
                    logger.removeHandler(handler)

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(emit_from_worker).result()

        assert seen == [(worker_thread_holder["thread"], sentinel_ctx)]

    def test_concurrent_emit_from_many_worker_threads_does_not_raise(self):
        """The actual shape of the original bug: several sources/fetches
        logging at once from a thread pool. This must not raise, and every
        line must make it into the handler even if the rendered placeholder
        only shows the most recent 60."""
        placeholder = _FakePlaceholder()
        handler = _StreamlitLogHandler(placeholder, ctx=None)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("ceia.test.concurrent")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        n_threads = 40
        errors = []

        def emit_one(i):
            try:
                logger.info("worker %d", i)
            except Exception as exc:  # noqa: BLE001 - the exact thing under test
                errors.append(exc)

        try:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                list(pool.map(emit_one, range(n_threads)))
        finally:
            logger.removeHandler(handler)

        assert errors == [], f"emit() raised from a worker thread: {errors}"
        assert len(handler.lines) == n_threads
