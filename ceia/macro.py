"""Macro-economic backdrop: RBI repo rate changes and crude oil (Brent).

Added on request, alongside GDP, CPI inflation, IIP, fiscal deficit and the
10-year G-Sec yield - checked directly (the same feasibility discipline as
every other data source in this project) and left out, not silently dropped:

* **GDP / CPI inflation / IIP** — MOSPI (mospi.gov.in) is a client-rendered
  React single-page app (``<div id="root"></div>`` plus a JS module entry
  point); a plain HTTP fetch returns an empty shell for every path checked,
  including its press-release listing. This tool's fetcher is deliberately
  plain HTTP (no headless browser anywhere in the pipeline), so there is
  nothing to parse here without a much larger architectural change.
* **Fiscal deficit** — the Controller General of Accounts (cga.nic.in) does
  serve real, server-rendered HTML (not a JS shell), but no structured
  monthly-deficit page was found from its crawlable navigation in the time
  budgeted for this spike. A genuine "not yet investigated enough to trust,"
  not a hard technical wall like the two below.
* **10-year G-Sec yield** ("borrowing rate") — Yahoo Finance's chart API
  (already used for equities and Brent crude below) simply does not carry
  Indian government bond yields; checked several plausible ticker symbols
  and Yahoo's own search endpoint, no match. RBI's own database (DBIE) and
  FBIL both failed to connect from this sandbox - possibly a real block,
  possibly this environment's networking, unverified either way.
* RBI's own site (rbi.org.in) returns ``418 Unauthorised Access`` on
  ``robots.txt`` itself for every path tried - the same "permission cannot
  be established" wall Business Standard hit (see ``ceia/sources.py``) -
  which rules it out as a direct source for repo rate history too.

**What is included**, because it checked out:

* **Repo rate changes** — RBI Monetary Policy Committee decisions are always
  major, extensively covered news events, so rather than scraping RBI
  directly this is a small, source-cited static table of confirmed rate
  *changes* (not every meeting - holds are not events), cross-checked
  against multiple independent reports for the entries near this project's
  knowledge cutoff. Needs a manual update when a new change happens after
  the dates below.
* **Crude oil (Brent)** — reuses this project's existing
  :class:`ceia.prices.YahooChartProvider` against the ``BZ=F`` futures
  ticker, the same infrastructure already used for equities and the
  benchmark index. Subject to the identical Yahoo rate-limit on shared/
  proxied egress already documented in the README for equity tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .prices import PriceError, PriceProvider, YahooChartProvider

# Genuine rate *changes* only - a Monetary Policy Committee meeting that held
# the rate steady is not an event to mark. Cross-checked against multiple
# independent reports (SCC Times, Business Standard, DD News, RBI's own MPC
# schedule) rather than taken from a single source. Dates before 2022 are
# included because this project's own validation window (the Adani/Hindenburg
# case, January 2023) and other example runs fall in this range.
REPO_RATE_CHANGES: list[tuple[date, float]] = [
    (date(2015, 1, 15), 7.75),
    (date(2015, 3, 4), 7.50),
    (date(2015, 6, 2), 7.25),
    (date(2015, 9, 29), 6.75),
    (date(2016, 4, 5), 6.50),
    (date(2016, 10, 4), 6.25),
    (date(2017, 8, 2), 6.00),
    (date(2018, 6, 6), 6.25),
    (date(2018, 8, 1), 6.50),
    (date(2019, 2, 7), 6.25),
    (date(2019, 4, 4), 6.00),
    (date(2019, 6, 6), 5.75),
    (date(2019, 8, 7), 5.40),
    (date(2019, 10, 4), 5.15),
    (date(2020, 3, 27), 4.40),
    (date(2020, 5, 22), 4.00),
    (date(2022, 5, 4), 4.40),
    (date(2022, 6, 8), 4.90),
    (date(2022, 8, 5), 5.40),
    (date(2022, 9, 30), 5.90),
    (date(2022, 12, 7), 6.25),
    (date(2023, 2, 8), 6.50),
    (date(2025, 2, 7), 6.25),
    (date(2025, 4, 9), 6.00),
    (date(2025, 6, 6), 5.50),
    (date(2025, 12, 5), 5.25),
]

# Indicators looked into but not included yet - see the module docstring for
# why each one specifically. Surfaced in reports as a disclosed gap rather
# than a silent omission, the same practice this project uses for every
# other checked-and-rejected source.
NOT_AVAILABLE_INDICATORS: dict[str, str] = {
    "GDP growth": "MOSPI's site is a client-rendered React app; no data is "
                  "visible to a plain HTTP fetch.",
    "CPI inflation": "MOSPI's site is a client-rendered React app; no data "
                     "is visible to a plain HTTP fetch.",
    "IIP": "MOSPI's site is a client-rendered React app; no data is visible "
          "to a plain HTTP fetch.",
    "Fiscal deficit": "No structured monthly-deficit page found on "
                      "cga.nic.in's crawlable navigation.",
    "10-year G-Sec yield": "Not carried by Yahoo Finance; RBI's DBIE and "
                           "FBIL both failed to connect from this sandbox.",
}


@dataclass
class MacroEvent:
    day: date
    indicator: str
    label: str


def macro_events_in_window(start: date, end: date) -> list[MacroEvent]:
    """Repo rate changes whose date falls inside [start, end]."""
    events = []
    for day, rate in REPO_RATE_CHANGES:
        if start <= day <= end:
            events.append(MacroEvent(
                day=day, indicator="repo_rate",
                label=f"RBI repo rate changed to {rate:.2f}%",
            ))
    return events


class SkippedPriceProvider(PriceProvider):
    """A macro_provider that always degrades immediately, no network touched
    - what ``--skip-macro-prices`` passes in, so a run can opt out of the
    crude-oil fetch (e.g. Yahoo already rate-limiting this connection) while
    the no-network repo-rate events still show either way."""

    name = "skipped"

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raise PriceError("skipped (--skip-macro-prices)")


def crude_oil_series(
    start: date, end: date, provider: PriceProvider | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Brent crude daily closes over [start, end].

    Returns ``(frame, note)`` - ``frame`` is ``None`` on failure (a bad
    network path, the shared-egress 429 already documented for equities)
    rather than raising, so a report can degrade to "unavailable" the same
    way the secondary-benchmark comparison already does in ``ceia/analyze.py``.
    """
    provider = provider or YahooChartProvider()
    try:
        frame = provider.history("BZ=F", start, end)
    except PriceError as exc:
        return None, f"crude oil price unavailable: {exc}"
    if frame.empty:
        return None, "crude oil price unavailable: no data returned"
    return frame, ""


def macro_summary(
    start: date, end: date, provider: PriceProvider | None = None,
) -> dict:
    """Everything a report needs: events in-window, crude oil's start/end
    change, and the disclosed list of indicators not yet available."""
    events = macro_events_in_window(start, end)
    frame, note = crude_oil_series(start, end, provider=provider)
    crude = {}
    if frame is not None and not frame.empty:
        first, last = float(frame["close"].iloc[0]), float(frame["close"].iloc[-1])
        crude = {
            "start_date": frame.index[0].date().isoformat(),
            "end_date": frame.index[-1].date().isoformat(),
            "start_price": first,
            "end_price": last,
            "change": (last - first) / first if first else None,
        }
    else:
        crude = {"note": note}
    return {
        "repo_rate_changes": [
            {"date": e.day.isoformat(), "label": e.label} for e in events
        ],
        "crude_oil": crude,
        "not_available": dict(NOT_AVAILABLE_INDICATORS),
    }
