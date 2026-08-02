"""Resolve a company name to a Yahoo-style ticker symbol.

Company name is what a user actually has; every other stage of this pipeline
(news ingestion, price history, the report) wants a ticker. This queries
Yahoo Finance's public search/autocomplete endpoint -- the same host
:class:`ceia.prices.YahooChartProvider` already uses for price history, so it
inherits the same network characteristics: works on an ordinary connection,
rate-limited on shared/proxied egress (see the Phase 0 notes in the README).

Equity results are re-ranked by exchange suffix (``.NS`` for NSE, ``.BO`` for
BSE) ahead of Yahoo's own relevance order, because this project's news
sources, trading calendar and event-window logic all assume an Indian
listing. An explicit ``--ticker`` always overrides this -- resolution only
runs when the caller does not already know the symbol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

_EXCHANGE_SUFFIXES = {
    "NSE": (".NS", ".BO"),
    "BSE": (".BO", ".NS"),
}
_DEFAULT_SUFFIXES = (".NS", ".BO")


class TickerLookupError(RuntimeError):
    pass


@dataclass
class TickerMatch:
    symbol: str
    name: str
    exchange: str
    exact_exchange_match: bool


def resolve_ticker(
    company: str,
    exchange: str = "NSE",
    session: requests.Session | None = None,
    max_retries: int = 4,
    timeout: float = 15.0,
) -> TickerMatch:
    """Look up the most likely equity ticker for a company name.

    Raises :class:`TickerLookupError` rather than guessing when nothing
    matches or the endpoint cannot be reached -- silently picking the wrong
    company would poison every downstream stage.
    """
    if not company or not company.strip():
        raise TickerLookupError("company name is empty")

    session = session or requests.Session()
    suffixes = _EXCHANGE_SUFFIXES.get(exchange.upper(), _DEFAULT_SUFFIXES)

    last: str | None = None
    resp = None
    for attempt in range(max_retries):
        try:
            resp = session.get(
                _SEARCH_URL,
                params={"q": company, "quotesCount": 10, "newsCount": 0},
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last = str(exc)
            resp = None
            time.sleep(2**attempt)
            continue
        if resp.status_code == 200:
            break
        last = f"HTTP {resp.status_code}"
        if resp.status_code in (429, 500, 502, 503):
            resp = None
            time.sleep(2 ** (attempt + 1))
            continue
        raise TickerLookupError(f"ticker lookup for {company!r} failed: {last}")

    if resp is None:
        raise TickerLookupError(f"ticker lookup for {company!r} failed: {last}")

    quotes = [
        q for q in resp.json().get("quotes", [])
        if q.get("quoteType") == "EQUITY" and q.get("symbol")
    ]
    if not quotes:
        raise TickerLookupError(
            f"no equity ticker found for {company!r} -- pass --ticker explicitly"
        )

    def rank(q: dict) -> int:
        symbol = q["symbol"]
        for i, suffix in enumerate(suffixes):
            if symbol.endswith(suffix):
                return i
        return len(suffixes)

    quotes.sort(key=rank)
    best = quotes[0]
    return TickerMatch(
        symbol=best["symbol"],
        name=best.get("longname") or best.get("shortname") or company,
        exchange=best.get("exchange", ""),
        exact_exchange_match=rank(best) == 0,
    )
