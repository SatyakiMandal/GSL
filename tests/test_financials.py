"""Tests for company financial fundamentals (ceia/financials.py).

The row-label logic here is pinned against two real, structurally different
screener.in page shapes, verified live before writing any code: a bank
(IndusInd Bank - "Revenue"/"Financing Profit", no Order Book at all) and an
industrial/EPC company (Larsen & Toubro - "Sales"/"Operating Profit", an
Order Book row that exists but is Premium-gated with masked values). The
HTML fixtures below reproduce that real structure at a reduced scale, not
invented shapes.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.financials import (  # noqa: E402
    ORDER_BOOK_NOT_APPLICABLE,
    ORDER_BOOK_PAYWALLED,
    FinancialsError,
    QuarterlyRow,
    SkippedFinancialsFetcher,
    fetch_financials,
    financials_summary,
)


def _quarters_table(rows: dict[str, list[str]], dates: list[str],
                    currency: str = "Rs. Crores") -> str:
    """rows: {label: [value-per-quarter, ...]} in the same order as dates."""
    head = "".join(f'<th data-date-key="{d}">{d}</th>' for d in dates)
    body = []
    for label, values in rows.items():
        cells = "".join(f"<td>{v}</td>" for v in values)
        body.append(f'<tr><td class="text"><button>{label}</button></td>{cells}</tr>')
    return f"""
    <p>Figures in {currency} / View Standalone</p>
    <section id="quarters">
      <table>
        <thead><tr><th class="text"></th>{head}</tr></thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </section>
    """


def _insights_section(rows: dict[str, list[str] | None]) -> str:
    """rows: {label: [values...]} for real data, or None for a masked/empty
    Premium-gated row (screener renders the label with no values at all)."""
    body = []
    for label, values in rows.items():
        cells = "".join(f"<td>{v}</td>" for v in values) if values else ""
        body.append(
            f'<tr><td class="text">{label}<br>'
            f'<span class="sub">unit</span></td>{cells}</tr>'
        )
    return f'<section id="insights"><table><tbody>{"".join(body)}</tbody></table></section>'


def _page(quarters_html: str, insights_html: str = "") -> str:
    return f"<html><body>{quarters_html}{insights_html}</body></html>"


_BANK_DATES = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
_BANK_QUARTERS = {
    "Revenue": ["12,801", "10,634", "12,264", "11,609", "11,373"],
    "Expenses": ["5,726", "5,891", "6,102", "5,980", "5,082"],
    "Financing Profit": ["677", "612", "701", "650", "-397"],
    "Tax %": ["25%", "25%", "25%", "25%", "25%"],
    "Net Profit": ["2,301", "2,175", "2,402", "2,301", "1,802"],
}

_INDUSTRIAL_DATES = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
_INDUSTRIAL_QUARTERS = {
    "Sales": ["55,128", "58,201", "60,340", "82,760", "67,942"],
    "Expenses": ["47,929", "50,112", "52,003", "70,611", "59,793"],
    "Operating Profit": ["7,199", "8,089", "8,337", "12,149", "8,149"],
    "Tax %": ["25%", "26%", "24%", "27%", "28%"],
    "Net Profit": ["3,593", "4,201", "4,450", "6,881", "5,300"],
}


class _FakeResp:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status = status


class _FakeFetcher:
    def __init__(self, pages: dict[str, _FakeResp] | None = None):
        self.pages = pages or {}
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeResp:
        self.calls.append(url)
        if url not in self.pages:
            raise RuntimeError(f"no fake page for {url}")
        return self.pages[url]


class TestBankShapedCompany:
    """IndusInd Bank's real shape: Revenue/Financing Profit, no Order Book
    row anywhere on the page at all."""

    def _fetcher(self):
        html = _page(
            _quarters_table(_BANK_QUARTERS, _BANK_DATES),
            _insights_section({"Branches": ["1,000", "1,100", "1,200", "1,300", "1,400"]}),
        )
        return _FakeFetcher({
            "https://www.screener.in/company/INDUSINDBK/consolidated/": _FakeResp(html),
        })

    def test_uses_revenue_and_financing_profit_labels(self):
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        assert summary.revenue.label_used == "Revenue"
        assert summary.operating_income.label_used == "Financing Profit"

    def test_latest_values_and_as_of_date(self):
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        assert summary.revenue.latest == pytest.approx(11373.0)
        assert summary.expenses.latest == pytest.approx(5082.0)
        assert summary.as_of == date(2025, 12, 31)
        assert summary.statement_kind == "consolidated"

    def test_negative_operating_income_gives_negative_nopat(self):
        """A real, non-hypothetical case: the latest quarter's Financing
        Profit is negative, so NOPAT should be too, not clamped or hidden."""
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        assert summary.operating_income.latest == pytest.approx(-397.0)
        assert summary.nopat == pytest.approx(-397.0 * 0.75)
        assert summary.nopat < 0

    def test_order_book_is_not_applicable_not_unavailable(self):
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        assert summary.order_book is None
        assert summary.order_book_note == ORDER_BOOK_NOT_APPLICABLE

    def test_qoq_change(self):
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        # QoQ: 11,373 vs 11,609 (previous quarter)
        assert summary.revenue.change(1) == pytest.approx((11373 - 11609) / 11609)

    def test_yoy_change_needs_a_full_five_quarters(self):
        """change(4) compares against the quarter 4 back - with exactly 5
        quarters of history (indices 0..4) that's the oldest one available;
        one quarter short and it must degrade to None, not an IndexError."""
        summary = fetch_financials("INDUSINDBK.NS", fetcher=self._fetcher())
        assert summary.revenue.change(4) == pytest.approx((11373 - 12801) / 12801)

        short = QuarterlyRow(label_used="Revenue", dates=[], values=[1.0, 2.0, 3.0, 4.0])
        assert short.change(4) is None


class TestIndustrialShapedCompany:
    """Larsen & Toubro's real shape: Sales/Operating Profit, an Order Book
    row that exists but whose data is Premium-gated (masked/empty cells)."""

    def _fetcher(self, order_book_values=None):
        html = _page(
            _quarters_table(_INDUSTRIAL_QUARTERS, _INDUSTRIAL_DATES),
            _insights_section({"Order Book": order_book_values}),
        )
        return _FakeFetcher({
            "https://www.screener.in/company/LT/consolidated/": _FakeResp(html),
        })

    def test_uses_sales_and_operating_profit_labels(self):
        summary = fetch_financials("LT.NS", fetcher=self._fetcher())
        assert summary.revenue.label_used == "Sales"
        assert summary.operating_income.label_used == "Operating Profit"

    def test_positive_nopat_computed_from_latest_quarter(self):
        summary = fetch_financials("LT.NS", fetcher=self._fetcher())
        assert summary.operating_income.latest == pytest.approx(8149.0)
        assert summary.tax_rate.latest == pytest.approx(28.0)
        assert summary.nopat == pytest.approx(8149.0 * 0.72)
        assert "Operating Profit" in summary.nopat_note

    def test_order_book_row_present_but_masked_is_reported_as_paywalled(self):
        summary = fetch_financials("LT.NS", fetcher=self._fetcher(order_book_values=None))
        assert summary.order_book is None
        assert summary.order_book_note == ORDER_BOOK_PAYWALLED

    def test_order_book_with_real_values_is_used_when_actually_present(self):
        """If screener ever un-gates this (or a future company has it free),
        the parser should use the real values, not always report paywalled."""
        summary = fetch_financials(
            "LT.NS", fetcher=self._fetcher(order_book_values=["450,000", "460,000"]))
        assert summary.order_book is not None
        assert summary.order_book.values == [450000.0, 460000.0]
        assert summary.order_book_note == ""

    def test_five_quarter_yoy_change(self):
        summary = fetch_financials("LT.NS", fetcher=self._fetcher())
        assert len(summary.revenue.values) == 5
        assert summary.revenue.change(4) == pytest.approx((67942 - 55128) / 55128)


class TestFetchFallbackAndErrors:
    def test_falls_back_to_standalone_when_consolidated_missing(self):
        html = _page(_quarters_table(_BANK_QUARTERS, _BANK_DATES))
        fetcher = _FakeFetcher({
            "https://www.screener.in/company/INDUSINDBK/consolidated/": _FakeResp("", status=404),
            "https://www.screener.in/company/INDUSINDBK/": _FakeResp(html),
        })
        summary = fetch_financials("INDUSINDBK.NS", fetcher=fetcher)
        assert summary.statement_kind == "standalone"
        assert summary.revenue.latest == pytest.approx(11373.0)

    def test_no_quarterly_table_anywhere_raises_financials_error(self):
        fetcher = _FakeFetcher({
            "https://www.screener.in/company/XYZ/consolidated/": _FakeResp("<html></html>"),
            "https://www.screener.in/company/XYZ/": _FakeResp("<html></html>"),
        })
        with pytest.raises(FinancialsError):
            fetch_financials("XYZ.NS", fetcher=fetcher)

    def test_network_failure_raises_financials_error_not_a_crash(self):
        class _AlwaysFails:
            def get(self, url):
                raise RuntimeError("connection refused")
        with pytest.raises(FinancialsError):
            fetch_financials("XYZ.NS", fetcher=_AlwaysFails())

    def test_ticker_suffix_is_stripped_for_the_screener_url(self):
        html = _page(_quarters_table(_BANK_QUARTERS, _BANK_DATES))
        fetcher = _FakeFetcher({
            "https://www.screener.in/company/INDUSINDBK/consolidated/": _FakeResp(html),
        })
        fetch_financials("INDUSINDBK.NS", fetcher=fetcher)
        assert fetcher.calls == ["https://www.screener.in/company/INDUSINDBK/consolidated/"]

    def test_missing_tax_rate_leaves_nopat_none_with_a_reason(self):
        rows = {k: v for k, v in _INDUSTRIAL_QUARTERS.items() if k != "Tax %"}
        html = _page(_quarters_table(rows, _INDUSTRIAL_DATES))
        fetcher = _FakeFetcher({
            "https://www.screener.in/company/LT/consolidated/": _FakeResp(html),
        })
        summary = fetch_financials("LT.NS", fetcher=fetcher)
        assert summary.nopat is None
        assert "tax rate" in summary.nopat_note


class TestFinancialsSummary:
    """The report-facing wrapper - a plain dict that never raises, the same
    contract ceia.macro.macro_summary() already follows."""

    def test_successful_fetch_returns_a_flat_dict(self):
        html = _page(_quarters_table(_INDUSTRIAL_QUARTERS, _INDUSTRIAL_DATES),
                     _insights_section({"Order Book": None}))
        fetcher = _FakeFetcher({
            "https://www.screener.in/company/LT/consolidated/": _FakeResp(html),
        })
        result = financials_summary("LT.NS", fetcher=fetcher)
        assert result["note"] == ""
        assert result["revenue"]["label"] == "Sales"
        assert result["revenue"]["latest"] == pytest.approx(67942.0)
        assert result["order_book_note"] == ORDER_BOOK_PAYWALLED
        assert result["order_book"] is None

    def test_failure_degrades_to_a_disclosed_note_not_an_exception(self):
        class _AlwaysFails:
            def get(self, url):
                raise RuntimeError("boom")
        result = financials_summary("XYZ.NS", fetcher=_AlwaysFails())
        assert "unavailable" in result["note"]

    def test_skipped_fetcher_degrades_immediately(self):
        result = financials_summary("LT.NS", fetcher=SkippedFinancialsFetcher())
        assert "unavailable" in result["note"]


class TestQuarterlyRow:
    def test_change_needs_enough_history(self):
        row = QuarterlyRow(label_used="Revenue", dates=[], values=[100.0, 110.0])
        assert row.change(1) == pytest.approx(0.10)
        assert row.change(4) is None

    def test_change_handles_a_zero_or_missing_prior_value(self):
        row = QuarterlyRow(label_used="Revenue", dates=[], values=[0.0, 100.0])
        assert row.change(1) is None
        row2 = QuarterlyRow(label_used="Revenue", dates=[], values=[None, 100.0])
        assert row2.change(1) is None

    def test_latest_on_empty_row_is_none(self):
        assert QuarterlyRow(label_used="x", dates=[], values=[]).latest is None
