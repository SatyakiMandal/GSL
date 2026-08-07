"""Company financial fundamentals: revenue growth, operating expense, NOPAT,
and order book (professor's note, added after the event-study/news-analysis
work — see the README's Phase 9 section).

Descriptive backdrop only, the same role the macro-economic and Nifty
sections already play: this never feeds into candidate-day flagging or any
significance test, and reports the company's latest reported quarterly
financials, not a value scoped to the report's own date window.

**Source: screener.in.** Checked directly before building anything here —
``robots.txt`` is fully permissive (``User-agent: *`` disallows only a
handful of unrelated query-string paths, no AI-agent block of any kind),
and a company page is real, server-rendered HTML with a genuine, dated
quarterly-results table, verified against two real companies with
deliberately different financial-statement shapes:

* **IndusInd Bank** (a bank): the table uses "Revenue" for top-line income
  and "Financing Profit" in place of an operating-profit line — banks don't
  report sales of goods or a conventional operating profit the way a
  manufacturer does.
* **Larsen & Toubro** (industrial/EPC): the table uses "Sales" and
  "Operating Profit" instead.

So which row means "top-line" and which means "operating income" is
sector-dependent, not a single hardcoded label — :func:`fetch_financials`
tries each known alternative in turn and records which one it used.

**Order Book is genuinely not available for free, for any company** — not
just a sector mismatch. It exists as a labelled row on L&T's page (an EPC
company that plausibly has one) but the row carries no values at all; the
page marks that whole data block ``Requires Premium`` — a paid screener.in
subscription, not something a plain fetch can reach. Checked directly
before disclosing this as unavailable rather than assumed. Surfaced as a
disclosed gap in every report - the same practice this project uses for
every other checked-and-rejected source (see ``ceia/macro.py``'s
``NOT_AVAILABLE_INDICATORS``) - with the *specific* reason (paywalled, vs.
simply absent from a company whose sector doesn't carry the concept at all)
kept distinct, since they are different facts about the world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from bs4 import BeautifulSoup

from .fetcher import Fetcher

_BASE = "https://www.screener.in/company"

# Screener's own row-label alternatives for the same underlying concept,
# tried in order - which one a company's page uses depends on whether it is
# a financial (bank/NBFC) or non-financial company (see module docstring).
_REVENUE_LABELS = ["Sales", "Revenue"]
_OPERATING_INCOME_LABELS = ["Operating Profit", "Financing Profit"]
_EXPENSE_LABELS = ["Expenses"]
_TAX_LABELS = ["Tax %"]

ORDER_BOOK_NOT_APPLICABLE = (
    "not applicable: this company's screener.in page carries no Order Book "
    "row at all - a concept most financial-statement templates only use "
    "for capital-goods/EPC/infrastructure companies, not this company's sector."
)
ORDER_BOOK_PAYWALLED = (
    "not available: screener.in carries an Order Book row for this company "
    "but the underlying data is gated behind a paid Premium subscription - "
    "confirmed directly (the row renders with no values, and the "
    "surrounding data block is marked 'Requires Premium')."
)


class FinancialsError(RuntimeError):
    pass


@dataclass
class QuarterlyRow:
    label_used: str
    dates: list[date]
    values: list[float | None]

    @property
    def latest(self) -> float | None:
        return self.values[-1] if self.values else None

    def change(self, periods_back: int) -> float | None:
        """Fractional change from `periods_back` quarters ago to the latest
        quarter, or None if there isn't enough history or either value is
        missing/zero."""
        if len(self.values) <= periods_back:
            return None
        latest, prior = self.values[-1], self.values[-1 - periods_back]
        if latest is None or prior is None or prior == 0:
            return None
        return (latest - prior) / abs(prior)


@dataclass
class FinancialSummary:
    ticker: str
    screener_url: str
    statement_kind: str  # "consolidated" | "standalone"
    currency_unit: str
    as_of: date | None
    revenue: QuarterlyRow | None = None
    expenses: QuarterlyRow | None = None
    operating_income: QuarterlyRow | None = None
    tax_rate: QuarterlyRow | None = None
    nopat: float | None = None
    nopat_note: str = ""
    order_book_note: str = field(default=ORDER_BOOK_NOT_APPLICABLE)
    order_book: QuarterlyRow | None = None


def _screener_symbol(ticker: str) -> str:
    """screener.in keys companies by their bare NSE/BSE symbol, not the
    Yahoo-style suffixed ticker this project uses everywhere else."""
    return ticker.split(".")[0].upper()


def _parse_number(text: str) -> float | None:
    text = text.strip().replace(",", "").rstrip("%")
    if not text or text in ("-", "—"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _quarter_dates(table) -> list[date]:
    dates = []
    for th in table.select("thead th[data-date-key]"):
        key = th.get("data-date-key")
        try:
            dates.append(date.fromisoformat(key))
        except (TypeError, ValueError):
            dates.append(None)
    return dates


def _row_by_label(table, labels: list[str], dates: list[date]) -> QuarterlyRow | None:
    """The first row (in ``labels`` preference order) that actually carries
    numeric values - a row can exist purely as a label with no data cells
    at all (see ORDER_BOOK_PAYWALLED), which is not the same as absent."""
    rows = {}
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        label = cells[0].get_text(strip=True).rstrip("+").strip()
        rows[label] = cells[1:]
    for label in labels:
        cells = rows.get(label)
        if cells is None:
            continue
        values = [_parse_number(c.get_text(strip=True)) for c in cells]
        if not any(v is not None for v in values):
            continue  # label present, but every cell is empty (paywalled)
        return QuarterlyRow(label_used=label, dates=dates, values=values)
    return None


def _order_book_status(soup) -> tuple[QuarterlyRow | None, str]:
    """Order Book lives in screener's separate "Insights" panel (per-company
    KPIs - loan book, branch count, order book, pre-sales... whichever apply
    to that company's sector), not the Quarterly Results table at all, and
    its label carries a unit suffix glued on via a nested <span> rather than
    plain text, so this needs its own row-finding logic rather than reusing
    _row_by_label. Returns ``(row_or_None, note)`` - a non-empty note means
    "not available", explaining exactly why (see the two constants above).
    """
    insights = soup.select_one("#insights")
    if insights is None:
        return None, ORDER_BOOK_NOT_APPLICABLE
    for tr in insights.select("table tbody tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        label = next(cells[0].stripped_strings, "")
        if label != "Order Book":
            continue
        values = [_parse_number(next(c.stripped_strings, "")) for c in cells[1:]]
        if any(v is not None for v in values):
            return QuarterlyRow(label_used="Order Book", dates=[], values=values), ""
        return None, ORDER_BOOK_PAYWALLED  # labelled, but every cell is masked/empty
    return None, ORDER_BOOK_NOT_APPLICABLE


def fetch_financials(ticker: str, fetcher: Fetcher | None = None) -> FinancialSummary:
    """The latest reported quarterly financials for ``ticker`` from
    screener.in - consolidated figures preferred, standalone as a fallback
    for a company that only reports standalone.

    Raises :class:`FinancialsError` on total failure (bad symbol, network
    down, page layout changed enough that no quarterly table is found) -
    callers should catch this and degrade to "unavailable", the same
    pattern every other checked-and-rejected source in this project follows.
    """
    fetcher = fetcher or Fetcher()
    symbol = _screener_symbol(ticker)

    for kind, url in (
        ("consolidated", f"{_BASE}/{symbol}/consolidated/"),
        ("standalone", f"{_BASE}/{symbol}/"),
    ):
        try:
            response = fetcher.get(url)
        except Exception as exc:
            last_error = f"{url}: {exc}"
            continue
        if response.status != 200:
            last_error = f"{url}: HTTP {response.status}"
            continue
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.select_one("#quarters table")
        if table is None:
            last_error = f"{url}: no quarterly-results table found"
            continue

        dates = _quarter_dates(table)
        currency_match = re.search(r"Figures in ([^/\n]+)", soup.get_text())
        currency_unit = currency_match.group(1).strip() if currency_match else "Rs. Crores"

        revenue = _row_by_label(table, _REVENUE_LABELS, dates)
        expenses = _row_by_label(table, _EXPENSE_LABELS, dates)
        operating_income = _row_by_label(table, _OPERATING_INCOME_LABELS, dates)
        tax_rate = _row_by_label(table, _TAX_LABELS, dates)
        order_book, order_book_note = _order_book_status(soup)

        nopat = None
        nopat_note = ""
        if operating_income is not None and operating_income.latest is not None:
            if tax_rate is not None and tax_rate.latest is not None:
                nopat = operating_income.latest * (1 - tax_rate.latest / 100)
                nopat_note = (
                    f"{operating_income.label_used} × (1 - Tax % / 100), "
                    f"both from the latest reported quarter"
                )
            else:
                nopat_note = "tax rate not available for the latest quarter"
        else:
            nopat_note = "no operating-income row (Operating Profit/Financing Profit) found"

        as_of = next((d for d in reversed(dates) if d is not None), None)

        return FinancialSummary(
            ticker=ticker, screener_url=url, statement_kind=kind,
            currency_unit=currency_unit, as_of=as_of,
            revenue=revenue, expenses=expenses, operating_income=operating_income,
            tax_rate=tax_rate, nopat=nopat, nopat_note=nopat_note,
            order_book_note=order_book_note, order_book=order_book,
        )

    raise FinancialsError(f"could not load financials for {ticker}: {last_error}")


class SkippedFinancialsFetcher:
    """Degrades immediately, no network touched - what ``--skip-financials``
    passes in, and what every test that reaches ``financials_summary()``
    should inject unless it is specifically exercising this module, for the
    same reason ``ceia.macro.SkippedFetcher`` exists."""

    def get(self, url: str):
        raise RuntimeError("skipped (--skip-financials)")


def _row_dict(row: QuarterlyRow | None) -> dict | None:
    if row is None:
        return None
    return {
        "label": row.label_used,
        "latest": row.latest,
        "qoq_change": row.change(1),
        "yoy_change": row.change(4),
    }


def financials_summary(ticker: str, fetcher: Fetcher | None = None) -> dict:
    """Everything a report needs, as a plain dict that never raises -
    degrades to a disclosed ``note`` on total failure, the same pattern
    ``ceia.macro.macro_summary()`` already follows for its own sources.
    """
    try:
        summary = fetch_financials(ticker, fetcher=fetcher)
    except FinancialsError as exc:
        return {"note": f"unavailable: {exc}"}
    return {
        "ticker": summary.ticker,
        "screener_url": summary.screener_url,
        "statement_kind": summary.statement_kind,
        "currency_unit": summary.currency_unit,
        "as_of": summary.as_of.isoformat() if summary.as_of else None,
        "revenue": _row_dict(summary.revenue),
        "expenses": _row_dict(summary.expenses),
        "operating_income": _row_dict(summary.operating_income),
        "tax_rate_pct": summary.tax_rate.latest if summary.tax_rate else None,
        "nopat": summary.nopat,
        "nopat_note": summary.nopat_note,
        "order_book": _row_dict(summary.order_book),
        "order_book_note": summary.order_book_note,
        "note": "",
    }
