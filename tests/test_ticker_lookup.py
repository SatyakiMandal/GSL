"""Tests for company-name-to-ticker resolution.

Like the price providers, Yahoo's search endpoint cannot be exercised
deterministically in CI, so these drive ``resolve_ticker`` through a fake
``requests.Session`` and check the parts that are ours: exchange-suffix
ranking, non-equity filtering, and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.ticker_lookup import TickerLookupError, resolve_ticker  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    """Retries back off with time.sleep(); tests shouldn't pay for that."""
    monkeypatch.setattr("ceia.ticker_lookup.time.sleep", lambda _seconds: None)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


ADANI_QUOTES = {
    "quotes": [
        {"symbol": "ADANIENT.BO", "quoteType": "EQUITY", "longname": "Adani Enterprises Ltd",
         "exchange": "BSE"},
        {"symbol": "ADANIENT.NS", "quoteType": "EQUITY", "longname": "Adani Enterprises Ltd",
         "exchange": "NSI"},
        {"symbol": "ADANI", "quoteType": "CRYPTOCURRENCY", "longname": "Adani Coin"},
    ]
}


class TestResolveTicker:
    def test_prefers_nse_suffix_over_yahoos_own_order(self):
        session = _FakeSession([_FakeResponse(200, ADANI_QUOTES)])
        match = resolve_ticker("Adani Enterprises", exchange="NSE", session=session)
        assert match.symbol == "ADANIENT.NS"
        assert match.exact_exchange_match is True

    def test_bse_preference_picks_the_bo_suffix(self):
        session = _FakeSession([_FakeResponse(200, ADANI_QUOTES)])
        match = resolve_ticker("Adani Enterprises", exchange="BSE", session=session)
        assert match.symbol == "ADANIENT.BO"

    def test_non_equity_quotes_are_ignored(self):
        payload = {"quotes": [{"symbol": "ADANI", "quoteType": "CRYPTOCURRENCY"}]}
        session = _FakeSession([_FakeResponse(200, payload)])
        with pytest.raises(TickerLookupError, match="no equity ticker"):
            resolve_ticker("Adani", session=session)

    def test_empty_quotes_raises(self):
        session = _FakeSession([_FakeResponse(200, {"quotes": []})])
        with pytest.raises(TickerLookupError, match="no equity ticker"):
            resolve_ticker("Not A Real Company", session=session)

    def test_blank_company_raises_without_a_network_call(self):
        session = _FakeSession([])
        with pytest.raises(TickerLookupError, match="empty"):
            resolve_ticker("   ", session=session)
        assert session.calls == 0

    def test_retries_on_429_then_succeeds(self):
        session = _FakeSession([_FakeResponse(429), _FakeResponse(200, ADANI_QUOTES)])
        match = resolve_ticker("Adani Enterprises", session=session)
        assert match.symbol == "ADANIENT.NS"
        assert session.calls == 2

    def test_non_retryable_http_error_raises_immediately(self):
        session = _FakeSession([_FakeResponse(404)])
        with pytest.raises(TickerLookupError, match="HTTP 404"):
            resolve_ticker("Adani Enterprises", session=session)
        assert session.calls == 1

    def test_no_suffix_match_falls_back_to_first_equity_result(self):
        payload = {"quotes": [
            {"symbol": "FOO.L", "quoteType": "EQUITY", "longname": "Foo Ltd", "exchange": "LSE"},
        ]}
        session = _FakeSession([_FakeResponse(200, payload)])
        match = resolve_ticker("Foo", session=session)
        assert match.symbol == "FOO.L"
        assert match.exact_exchange_match is False

    def test_exhausted_retries_raise_with_last_error(self):
        session = _FakeSession([_FakeResponse(503), _FakeResponse(503),
                                _FakeResponse(503), _FakeResponse(503)])
        with pytest.raises(TickerLookupError, match="HTTP 503"):
            resolve_ticker("Adani Enterprises", session=session, max_retries=4)

    def test_request_exception_is_wrapped(self):
        class _RaisingSession:
            def get(self, *args, **kwargs):
                raise requests.ConnectionError("boom")

        with pytest.raises(TickerLookupError, match="boom"):
            resolve_ticker("Adani Enterprises", session=_RaisingSession(), max_retries=1)
