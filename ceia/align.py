"""Map publication timestamps to the trading day that could react to them.

PRD Section 10: an item published after the close belongs to the *next* trading
day's price reaction, not the same day's. Getting this wrong is a quiet error —
nothing crashes, the numbers just describe a reaction that happened before the
news. Phase 0 found it is not an edge case either: the Economic Times, Financial
Express and Moneycontrol samples all published after 15:30 IST.

Items whose timestamp could not be read are left with ``trading_day = None`` and
reported separately, rather than being assigned a day on a guess.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from .models import IST_OFFSET_HOURS, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, NewsItem
from .extract import IST

CLOSE = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)


def next_trading_day(day: date, trading_days: set[date] | None) -> date | None:
    """The first trading day strictly after ``day``.

    With a real calendar (the dates the price series actually has) this handles
    exchange holidays. Without one it falls back to skipping weekends, which is
    right most of the time but will miss holidays — so a calendar is passed in
    whenever prices have been loaded.
    """
    candidate = day + timedelta(days=1)
    if trading_days:
        horizon = max(trading_days) if trading_days else None
        while horizon and candidate <= horizon:
            if candidate in trading_days:
                return candidate
            candidate += timedelta(days=1)
        return None
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate += timedelta(days=1)
    return candidate


def same_or_next_trading_day(day: date, trading_days: set[date] | None) -> date | None:
    """``day`` itself if it trades, otherwise the next day that does."""
    if trading_days is None:
        return day if day.weekday() < 5 else next_trading_day(day, None)
    if day in trading_days:
        return day
    return next_trading_day(day, trading_days)


def attribute(item: NewsItem, trading_days: set[date] | None = None) -> NewsItem:
    """Set ``trading_day`` and ``after_close`` on one item."""
    if item.published_at is None:
        item.trading_day = None
        item.after_close = False
        return item

    local = item.published_at.astimezone(IST)

    if item.timestamp_confidence == "date-only":
        # No time of day, so we cannot tell whether it beat the close. Attribute
        # to the publication day if it trades, and leave after_close False; the
        # item stays flagged via timestamp_confidence so a reader can discount it.
        item.after_close = False
        item.trading_day = same_or_next_trading_day(local.date(), trading_days)
        return item

    item.after_close = local.time() >= CLOSE
    if item.after_close:
        item.trading_day = next_trading_day(local.date(), trading_days)
    else:
        item.trading_day = same_or_next_trading_day(local.date(), trading_days)
    return item


def attribute_all(items: list[NewsItem],
                  trading_days: set[date] | None = None) -> list[NewsItem]:
    return [attribute(item, trading_days) for item in items]


def unattributed(items: list[NewsItem]) -> list[NewsItem]:
    """Items we refused to place on a trading day, for explicit reporting."""
    return [i for i in items if i.trading_day is None]
