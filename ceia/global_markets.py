"""Global market "ripple effect" reference data: how the S&P 500, Nasdaq,
Dow, FTSE 100, Hang Seng, and Nikkei 225 moved around each of this company's
flagged incident days, and across the run's window generally.

Not the same problem as ``ceia/nifty.py``'s sector indices, which trade on
the same exchange, same hours, as the company itself — same-calendar-day
comparison is exactly right there. A global market does not share NSE's
9:15am-3:30pm IST session at all, so pairing "NSE day X" with "global
index's own day X" would silently compare against a session that either
hasn't started yet or is still running — the same misattribution
``ceia/align.py`` already guards against for after-close news (PRD Section
10), applied here to whole markets instead of individual articles.

Each configured index is tagged ``same_day_available``: whether *its own*
session, for a given trading date, closes (in IST clock time) at or before
NSE's own 3:30pm close on that date:

- **Hang Seng** (HKT, UTC+8) and **Nikkei 225** (JST, UTC+9) both run mostly
  within NSE's own trading hours and close before 3:30pm IST — their own
  same-dated session is genuinely complete and known by the time NSE closes
  that day.
- **S&P 500 / Nasdaq / Dow** (US Eastern) run entirely overnight IST-time,
  closing around 1:30-2:30am IST the *next* calendar day — by NSE's open
  the next morning, the relevant, fully-known session is the *prior* US
  trading day's.
- **FTSE 100** (London) closes around 9-10pm IST, well after NSE has
  already closed for the day — so even though its own session nominally
  shares NSE's calendar date, that session is not yet complete when NSE
  closes, and the prior LSE trading day is the one actually known.

A ``same_day_available`` index is paired with its own most recent trading
day *at or before* the NSE day. Every other index is paired with its most
recent trading day *strictly before* the NSE day. Either way this degrades
gracefully across holiday-calendar mismatches: if the global market has no
session on the exact aligned date (a market-specific holiday), the lookup
falls back further, exactly as ``ceia.align``'s own trading-day lookups
already do for news.

Deliberately descriptive only, like ``ceia/nifty.py``'s window-return table:
each index's own return on its aligned day, and a z-score against *that
index's own* historical daily-return distribution — not a cross-market
market-model regression against ^NSEI, which would need to assume a
same-hours relationship these markets don't actually have. Never fed into
which days get flagged as candidate incidents.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .prices import PriceError, PriceProvider, load_prices
from .returns import daily_returns

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalIndexMeta:
    ticker: str
    same_day_available: bool
    note: str


GLOBAL_INDICES: dict[str, GlobalIndexMeta] = {
    "S&P 500": GlobalIndexMeta(
        "^GSPC", False,
        "NYSE/Nasdaq trade overnight IST-time and close ~1:30-2:30am IST "
        "the next calendar day; the prior US trading day is the session "
        "actually known by NSE's next open."),
    "Nasdaq Composite": GlobalIndexMeta(
        "^IXIC", False,
        "Same US trading hours as the S&P 500 - see that note."),
    "Dow Jones": GlobalIndexMeta(
        "^DJI", False,
        "Same US trading hours as the S&P 500 - see that note."),
    "FTSE 100": GlobalIndexMeta(
        "^FTSE", False,
        "London closes ~9-10pm IST, after NSE has already closed for the "
        "day - the prior LSE trading day is the one actually known."),
    "Hang Seng": GlobalIndexMeta(
        "^HSI", True,
        "Hong Kong trades within NSE's own hours and closes ~1:30pm IST, "
        "before NSE's 3:30pm close - its own same-dated session is known."),
    "Nikkei 225": GlobalIndexMeta(
        "^N225", True,
        "Tokyo closes ~11:30am IST, before NSE's own 3:30pm close - its "
        "own same-dated session is known."),
}


@dataclass
class GlobalIndexDay:
    aligned_date: date
    return_: float
    return_z: float


@dataclass
class GlobalIndexSeries:
    name: str
    ticker: str
    provider: str = ""
    same_day_available: bool = True
    note: str = ""
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_return: float = float("nan")
    # NSE trading day -> that index's aligned session for it.
    aligned: dict[date, GlobalIndexDay] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return not self.daily.empty


def _align_date(nse_day: date, sorted_dates: list[date],
                same_day_available: bool) -> date | None:
    """The index's own most recent trading day known by NSE's close on
    ``nse_day`` - at or before it if ``same_day_available``, strictly
    before it otherwise. ``None`` if the index has no history that early."""
    cutoff = nse_day if same_day_available else nse_day - timedelta(days=1)
    idx = bisect.bisect_right(sorted_dates, cutoff) - 1
    return sorted_dates[idx] if idx >= 0 else None


def load_global_indices(
    nse_days: list[date],
    start: date,
    end: date,
    lead_in_days: int = 200,
    providers: list[PriceProvider] | None = None,
    indices: dict[str, GlobalIndexMeta] | None = None,
) -> dict[str, GlobalIndexSeries]:
    """Best-effort load of each configured global index, aligned onto every
    date in ``nse_days`` (typically the company's own analysis-window
    trading days). One index failing to load degrades to a note on that
    index alone - the same never-sink-the-whole-run behaviour
    ``ceia.nifty.load_nifty_indices`` already follows.
    """
    indices = GLOBAL_INDICES if indices is None else indices
    fetch_start = start - timedelta(days=lead_in_days)
    out: dict[str, GlobalIndexSeries] = {}

    for name, meta in indices.items():
        try:
            frame, provider_name = load_prices(meta.ticker, fetch_start, end,
                                               providers=providers)
        except Exception as exc:
            # Not just PriceError: this index's failure must never sink the
            # others or the rest of the report (see ceia/nifty.py's own
            # widened exception handling for the real bug that pattern
            # fixed - the same discipline applies here from the start).
            out[name] = GlobalIndexSeries(name=name, ticker=meta.ticker,
                                          same_day_available=meta.same_day_available,
                                          note=f"unavailable: {exc}")
            continue
        if frame.empty:
            out[name] = GlobalIndexSeries(name=name, ticker=meta.ticker,
                                          same_day_available=meta.same_day_available,
                                          note="no data returned")
            continue

        closes = frame["close"].astype(float)
        daily = pd.DataFrame(index=frame.index)
        daily["close"] = closes
        daily["return"] = daily_returns(closes)
        sorted_dates = [ts.date() for ts in frame.index]

        returns_all = daily["return"].dropna()
        r_mean = float(returns_all.mean()) if len(returns_all) else 0.0
        r_sd = float(returns_all.std(ddof=1)) if len(returns_all) > 1 else 0.0

        aligned: dict[date, GlobalIndexDay] = {}
        for nse_day in nse_days:
            aligned_date = _align_date(nse_day, sorted_dates, meta.same_day_available)
            if aligned_date is None:
                continue
            ts = pd.Timestamp(aligned_date)
            ret = daily.loc[ts, "return"] if ts in daily.index else None
            if ret is None or pd.isna(ret):
                continue
            ret = float(ret)
            z = (ret - r_mean) / r_sd if r_sd else 0.0
            aligned[nse_day] = GlobalIndexDay(aligned_date=aligned_date,
                                              return_=ret, return_z=z)

        display = daily.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
        window_return = float("nan")
        if len(display) >= 2:
            display["level"] = display["close"] / display["close"].iloc[0] * 100
            window_return = float(display["close"].iloc[-1] / display["close"].iloc[0] - 1)

        out[name] = GlobalIndexSeries(
            name=name, ticker=meta.ticker, provider=provider_name,
            same_day_available=meta.same_day_available, note=meta.note,
            daily=display, window_return=window_return, aligned=aligned,
        )
    return out


def same_direction_rate(index: GlobalIndexSeries, company_daily: pd.DataFrame,
                        candidate_days: list[date]) -> dict:
    """Of the already-flagged candidate incident days, how many did this
    index's *aligned* session move the same direction (sign) as the company
    did on that NSE day? A quick raw-return coincidence check - purely
    descriptive, mirrors ``ceia.nifty.same_direction_rate``, never itself a
    test and never changes which days are flagged."""
    if not index.available or not candidate_days:
        return {"n": 0, "agree": 0, "rate": None}
    agree = 0
    n = 0
    for day in candidate_days:
        aligned_day = index.aligned.get(day)
        if aligned_day is None or day not in company_daily.index:
            continue
        company_ret = company_daily.loc[day, "return"]
        if pd.isna(company_ret):
            continue
        n += 1
        if (aligned_day.return_ > 0) == (company_ret > 0):
            agree += 1
    return {"n": n, "agree": agree, "rate": (agree / n if n else None)}
