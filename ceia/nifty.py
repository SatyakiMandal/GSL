"""Nifty sector index reference data (PRD extension, professor's request).

The ask: show Nifty 50 and five sector indices (Bank, Auto, Energy, IT,
Metal) alongside a company's own price analysis, "analysed the same way" as
the stock - but reusing the news/sentiment/candidate-day work already done,
not a second event study per index. What that cashes out to here is
descriptive: each index's own rebased price path and window return, plus how
often it moved in the same direction as the company on the days already
flagged as candidate incidents. None of this feeds back into flagging or
ranking - it is backdrop, the same role the macro-economic section plays.

Reuses the same provider chain as the company/benchmark price fetch
(``ceia.prices.load_prices``) rather than a special-cased fetch path, so it
degrades exactly like the existing secondary-benchmark feature does: one
index failing to load never sinks the others or the rest of the report.

Yahoo Finance tickers below are NSE's standard index symbols; live history
could not be verified against this build environment for the same reason
noted throughout Phase 0/2 - Yahoo rate-limits this sandbox's shared egress
IP regardless of which of these symbols is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .prices import PriceError, PriceProvider, load_prices
from .returns import daily_returns

NIFTY_INDICES: dict[str, str] = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Energy": "^CNXENERGY",
    "Nifty IT": "^CNXIT",
    "Nifty Metal": "^CNXMETAL",
}


@dataclass
class IndexSeries:
    name: str
    ticker: str
    provider: str = ""
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_return: float = float("nan")
    note: str = ""

    @property
    def available(self) -> bool:
        return not self.daily.empty


def load_nifty_indices(
    start: date,
    end: date,
    providers: list[PriceProvider] | None = None,
    indices: dict[str, str] | None = None,
) -> dict[str, IndexSeries]:
    """Best-effort load of each configured index over ``[start, end]``.

    One index failing to load (bad symbol, provider outage, no CSV fixture in
    tests) degrades to a note on that index alone - the same
    never-sink-the-whole-run behaviour ``analyze.py`` already applies to the
    secondary benchmark.
    """
    indices = NIFTY_INDICES if indices is None else indices
    out: dict[str, IndexSeries] = {}
    for name, ticker in indices.items():
        try:
            frame, provider_name = load_prices(ticker, start, end, providers=providers)
        except PriceError as exc:
            out[name] = IndexSeries(name=name, ticker=ticker,
                                    note=f"unavailable: {exc}")
            continue
        window = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if window.empty:
            out[name] = IndexSeries(name=name, ticker=ticker, provider=provider_name,
                                    note="no trading days in window")
            continue
        closes = window["close"].astype(float)
        daily = pd.DataFrame(index=window.index)
        daily["close"] = closes
        daily["return"] = daily_returns(closes)
        daily["level"] = closes / closes.iloc[0] * 100
        out[name] = IndexSeries(
            name=name, ticker=ticker, provider=provider_name, daily=daily,
            window_return=float(closes.iloc[-1] / closes.iloc[0] - 1),
        )
    return out


def same_direction_rate(index: IndexSeries, company_daily: pd.DataFrame,
                        candidate_days: list[date]) -> dict:
    """Of the already-flagged candidate incident days, how many did this
    index move the same direction (sign) as the company that day?

    Purely descriptive context for a flagged day - it is not itself a test
    and never changes which days are flagged.
    """
    if not index.available or not candidate_days:
        return {"n": 0, "agree": 0, "rate": None}
    agree = 0
    n = 0
    for day in candidate_days:
        ts = pd.Timestamp(day)
        if ts not in index.daily.index or day not in company_daily.index:
            continue
        idx_ret = index.daily.loc[ts, "return"]
        company_ret = company_daily.loc[day, "return"]
        if pd.isna(idx_ret) or pd.isna(company_ret):
            continue
        n += 1
        if (idx_ret > 0) == (company_ret > 0):
            agree += 1
    return {"n": n, "agree": agree, "rate": (agree / n) if n else None}
