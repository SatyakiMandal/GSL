"""Macro-economic backdrop: RBI repo rate changes, crude oil (Brent), the
10-year G-Sec yield and the fiscal deficit.

Added on request, alongside GDP, CPI inflation and IIP - checked directly
(the same feasibility discipline as every other data source in this
project). Two of the seven originally requested indicators are still left
out, not silently dropped:

* **GDP / CPI inflation / IIP** — MOSPI (mospi.gov.in) is a client-rendered
  React single-page app (``<div id="root"></div>`` plus a JS module entry
  point); a plain HTTP fetch returns an empty shell for every path checked,
  including its press-release listing. This tool's fetcher is deliberately
  plain HTTP (no headless browser anywhere in the pipeline), so there is
  nothing to parse here without a much larger architectural change. (A free,
  no-API-key CSV route via the St. Louis Fed's FRED service does carry
  mirrored India CPI/IIP/GDP series, checked and confirmed working - but the
  specific series found are stale: CPI stops March 2025, industrial
  production stops January 2023, GDP is annual-only. Left out rather than
  shown as if current; revisit if fresher FRED series turn up.)
* RBI's own site (rbi.org.in) returns ``418 Unauthorised Access`` on
  ``robots.txt`` itself for every path tried - the same "permission cannot
  be established" wall Business Standard hit (see ``ceia/sources.py``) -
  which rules it out as a direct source for repo rate history, G-Sec yields,
  or anything else.

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
* **10-year G-Sec yield** ("borrowing rate") — not carried by Yahoo Finance
  (checked directly, no match), but `tradingeconomics.com`'s bond-yield page
  is real, server-rendered HTML (confirmed, not a JS shell) with the current
  value and its exact as-of date both embedded in a stable, self-describing
  ``<meta name="description">`` sentence - verified against a real fetch. Its
  ``robots.txt`` is fully unrestricted, no AI-agent block of any kind. This
  is always a *current* reading, not a value as of the report's own window -
  shown with its own fetched-on date so it is never mistaken for one.
* **Fiscal deficit** — `govtbudget.com`'s fiscal deficit tracker has a
  similarly stable sentence in its page body naming the budgeted figure and
  the fiscal year it applies to, verified against a real fetch. Its
  ``robots.txt`` blanket-blocks ``ClaudeBot`` by name — more directly than
  either UnlistedZone's or Inc42's policies (see [Phase
  5](#phase-5--unlisted--pre-ipo-shares)) — but this tool's own,
  distinct, honestly-declared user agent is not itself named anywhere in the
  file, so it falls under the unrestricted default group; raised explicitly
  rather than assumed, and the answer was the same as the two earlier cases.
  Also always the latest *budgeted* figure for a fiscal year, not a value
  scoped to the report's window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .fetcher import Fetcher
from .prices import PriceError, PriceProvider, YahooChartProvider

_GSEC_URL = "https://tradingeconomics.com/india/government-bond-yield"
_GSEC_RE = re.compile(
    r'India 10Y Bond Yield \w+ to ([\d.]+)% on ([A-Za-z]+ \d{1,2}, \d{4})')

_FISCAL_DEFICIT_URL = "https://govtbudget.com/budget-analysis/fiscal-deficit"
_FISCAL_DEFICIT_RE = re.compile(
    r"India's fiscal deficit for (\d{4}-\d{2}) is budgeted at Rs "
    r"([\d.]+) lakh crore, which equals ([\d.]+)% of GDP")

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
                  "visible to a plain HTTP fetch. A free FRED mirror exists "
                  "but is annual-only.",
    "CPI inflation": "MOSPI's site is a client-rendered React app; no data "
                     "is visible to a plain HTTP fetch. A free FRED mirror "
                     "exists but stops in March 2025.",
    "IIP": "MOSPI's site is a client-rendered React app; no data is visible "
          "to a plain HTTP fetch. A free FRED mirror exists but stops in "
          "January 2023.",
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


class SkippedFetcher:
    """What ``--skip-macro-prices`` passes for the G-Sec yield and fiscal
    deficit HTML fetches - degrades immediately, no network touched. Also
    what every test that reaches ``macro_summary()`` should inject, for the
    same reason ``SkippedPriceProvider`` exists: the real default
    (:class:`ceia.fetcher.Fetcher`) retries a failure with backoff, which is
    slow and was once a real, caught-late regression in this project's own
    test suite (see the git history for ceia/macro.py's addition)."""

    def get(self, url: str):
        raise RuntimeError("skipped (--skip-macro-prices)")


def gsec_yield(fetcher: Fetcher | None = None) -> tuple[dict | None, str]:
    """India's 10-year G-Sec yield, as tradingeconomics.com currently states
    it - a *current* reading, not a value as of the report's own window (see
    the module docstring for why no historical series is available here).

    Returns ``(value, note)`` the same shape as :func:`crude_oil_series` -
    ``value`` is ``None`` on failure rather than raising.
    """
    fetcher = fetcher or Fetcher()
    try:
        response = fetcher.get(_GSEC_URL)
    except Exception as exc:
        return None, f"10-year G-Sec yield unavailable: {exc}"
    match = _GSEC_RE.search(response.text)
    if not match:
        return None, "10-year G-Sec yield unavailable: page format changed"
    value, as_of = match.groups()
    return {"value": float(value), "as_of": as_of,
            "source": "tradingeconomics.com"}, ""


def fiscal_deficit(fetcher: Fetcher | None = None) -> tuple[dict | None, str]:
    """India's budgeted fiscal deficit, as govtbudget.com currently states
    it for the fiscal year it names - a *current* budgeted figure, not a
    value scoped to the report's own window.

    Returns ``(value, note)`` the same shape as :func:`crude_oil_series`.
    """
    fetcher = fetcher or Fetcher()
    try:
        response = fetcher.get(_FISCAL_DEFICIT_URL)
    except Exception as exc:
        return None, f"fiscal deficit unavailable: {exc}"
    match = _FISCAL_DEFICIT_RE.search(response.text)
    if not match:
        return None, "fiscal deficit unavailable: page format changed"
    fiscal_year, lakh_crore, pct_gdp = match.groups()
    return {"fiscal_year": fiscal_year, "lakh_crore": float(lakh_crore),
            "pct_gdp": float(pct_gdp), "source": "govtbudget.com"}, ""


def macro_summary(
    start: date, end: date,
    provider: PriceProvider | None = None,
    fetcher: Fetcher | None = None,
) -> dict:
    """Everything a report needs: events in-window, crude oil's start/end
    change, the latest G-Sec yield and fiscal deficit reading, and the
    disclosed list of indicators still not available.

    ``fetcher`` drives the G-Sec yield and fiscal deficit fetches; pass
    :class:`SkippedFetcher` (as ``--skip-macro-prices`` does) to skip both
    without touching the network, same role ``provider`` plays for crude oil.
    """
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

    gsec_value, gsec_note = gsec_yield(fetcher=fetcher)
    deficit_value, deficit_note = fiscal_deficit(fetcher=fetcher)

    return {
        "repo_rate_changes": [
            {"date": e.day.isoformat(), "label": e.label} for e in events
        ],
        "crude_oil": crude,
        "gsec_yield": gsec_value or {"note": gsec_note},
        "fiscal_deficit": deficit_value or {"note": deficit_note},
        "not_available": dict(NOT_AVAILABLE_INDICATORS),
    }
