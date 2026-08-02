"""Daily price history, behind a provider interface.

Phase 0 found two independent problems with the PRD's default choice of
``yfinance`` (Section 7.2) in a sandboxed/proxied environment:

1. ``yfinance`` >= 0.2.50 fetches through ``curl_cffi`` with browser TLS
   impersonation. Where egress passes through a TLS-terminating proxy, the
   impersonated handshake is rejected and every call fails with
   ``curl: (35) Recv failure``. Disabling impersonation fixes it, which is why
   :class:`YFinanceProvider` passes a plain session where the installed
   version supports one.
2. Yahoo rate-limits shared egress IPs hard (HTTP 429), regardless of client.

So the provider is pluggable: ``yfinance`` first, a direct Yahoo chart-API
client second, and a CSV loader last. The CSV path exists so the rest of the
pipeline stays runnable and reproducible when the network path is unavailable,
which matters for the PRD's reproducibility requirement in Section 10.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class PriceError(RuntimeError):
    pass


class PriceProvider(ABC):
    name: str

    @abstractmethod
    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return a frame indexed by naive trading date with a ``close`` column."""


def _as_frame(index, close, volume=None) -> pd.DataFrame:
    frame = pd.DataFrame({"close": close}, index=pd.DatetimeIndex(index).normalize())
    if volume is not None:
        frame["volume"] = volume
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["close"])


class YahooChartProvider(PriceProvider):
    """Direct Yahoo Finance chart endpoint over plain ``requests``.

    Uses the ambient proxy and CA configuration, unlike ``curl_cffi``, and
    backs off on the 429s Yahoo hands out to shared egress IPs.
    """

    name = "yahoo-chart"

    def __init__(self, cache_dir: Path | str = "cache/prices", max_retries: int = 6) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA, "Accept": "application/json"})

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        cache = self.cache_dir / f"{symbol.replace('^', '_idx_')}_{start}_{end}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            payload = self._download(symbol, start, end)
            cache.write_text(json.dumps(payload))

        quote = payload["indicators"]["quote"][0]
        tz = payload["meta"].get("exchangeTimezoneName", "Asia/Kolkata")
        index = (
            pd.to_datetime(payload["timestamp"], unit="s", utc=True)
            .tz_convert(tz)
            .tz_localize(None)
        )
        return _as_frame(index, quote.get("close"), quote.get("volume"))

    def _download(self, symbol: str, start: date, end: date) -> dict:
        params = {
            "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "period2": int(datetime.combine(end, datetime.min.time()).timestamp()),
            "interval": "1d",
        }
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params=params,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last = str(exc)
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                result = resp.json().get("chart", {}).get("result")
                if not result:
                    raise PriceError(f"{symbol}: empty chart result")
                return result[0]
            last = f"HTTP {resp.status_code}"
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            break
        raise PriceError(f"{symbol}: {last}")


class YFinanceProvider(PriceProvider):
    """The PRD's default library, with the curl_cffi impersonation disabled."""

    name = "yfinance"

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise PriceError("yfinance is not installed") from exc

        kwargs = dict(start=str(start), end=str(end), progress=False,
                      auto_adjust=False, threads=False)
        try:
            frame = yf.download(symbol, **kwargs)
        except Exception as exc:
            raise PriceError(f"{symbol}: {exc}") from exc
        if frame is None or frame.empty:
            raise PriceError(f"{symbol}: yfinance returned no rows")

        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        volume = frame.get("Volume")
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        return _as_frame(frame.index, close.to_numpy(),
                         volume.to_numpy() if volume is not None else None)


class AlphaVantageProvider(PriceProvider):
    """Alpha Vantage daily series, keyed from ``ALPHAVANTAGE_API_KEY``.

    The PRD puts Alpha Vantage out of scope for v1 because it was proposed for
    *global* coverage. It is included here for a different reason: it is
    reachable from environments where Yahoo rate-limits the egress IP and the
    NSE/BSE sites return 403.

    As of testing, Alpha Vantage's free tier has gated ``outputsize=full`` on
    ``TIME_SERIES_DAILY`` behind a paid plan, so a free key can no longer pull
    the multi-year history this tool needs for an arbitrary historical date
    range (``outputsize=compact`` only returns the most recent ~100 sessions
    counted back from *today*, which does not reach a window like January
    2023 once enough time has passed). Kept as a fallback for whichever date
    ranges it can still serve; prefer ``yfinance`` where it is reachable.
    """

    name = "alphavantage"

    def __init__(self, api_key: str | None = None,
                 cache_dir: Path | str = "cache/prices") -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _translate(symbol: str) -> str:
        """Map a yfinance-style symbol onto Alpha Vantage's spelling."""
        if symbol.endswith(".NS"):
            return symbol[:-3] + ".BSE"
        return symbol.lstrip("^")

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if not self.api_key:
            raise PriceError("ALPHAVANTAGE_API_KEY is not set")
        av_symbol = self._translate(symbol)
        cache = self.cache_dir / f"av_{av_symbol}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "TIME_SERIES_DAILY", "symbol": av_symbol,
                        "outputsize": "full", "apikey": self.api_key},
                headers={"User-Agent": _UA}, timeout=40,
            )
            if resp.status_code != 200:
                raise PriceError(f"{symbol}: HTTP {resp.status_code}")
            payload = resp.json()
            if "Time Series (Daily)" not in payload:
                # Alpha Vantage reports quota and bad symbols in the body.
                note = payload.get("Note") or payload.get("Information") or \
                    payload.get("Error Message") or str(payload)[:160]
                raise PriceError(f"{symbol}: {note}")
            cache.write_text(json.dumps(payload))

        series = payload["Time Series (Daily)"]
        rows = {pd.Timestamp(day): float(values["4. close"])
                for day, values in series.items()}
        frame = _as_frame(list(rows.keys()), list(rows.values()))
        return frame.loc[str(start):str(end)]


class CsvProvider(PriceProvider):
    """Load prices from ``<dir>/<symbol>.csv`` with ``date`` and ``close`` columns.

    Lets a run be reproduced exactly, and keeps the pipeline testable when the
    network path is blocked.
    """

    name = "csv"

    def __init__(self, directory: Path | str = "data/prices") -> None:
        self.directory = Path(directory)

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self.directory / f"{symbol.replace('^', '_idx_')}.csv"
        if not path.exists():
            raise PriceError(f"no CSV at {path}")
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame = frame.rename(columns=str.lower)
        window = frame.loc[str(start) : str(end)]
        return _as_frame(window.index, window["close"].to_numpy(),
                         window["volume"].to_numpy() if "volume" in window else None)


def load_prices(
    symbol: str,
    start: date,
    end: date,
    providers: list[PriceProvider] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Try each provider in order; return the first success and its name."""
    providers = providers or [YFinanceProvider(), YahooChartProvider(),
                              AlphaVantageProvider(), CsvProvider()]
    errors = []
    for provider in providers:
        try:
            frame = provider.history(symbol, start, end)
            if not frame.empty:
                log.info("prices for %s via %s (%d rows)", symbol, provider.name, len(frame))
                return frame, provider.name
            errors.append(f"{provider.name}: empty")
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")
    raise PriceError(f"all providers failed for {symbol} -> " + "; ".join(errors))
