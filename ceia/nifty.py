"""Nifty sector index reference data and per-index event study (professor's
request, later widened): show Nifty 50 and five sector indices (Bank, Auto,
Energy, IT, Metal) alongside a company's own price analysis, "analysed the
same way" as the stock.

Two layers, confirmed with the user rather than guessed:

1. **Descriptive backdrop** (always computed): each index's own rebased
   price path and window return - the same role the macro-economic section
   plays, never fed back into flagging.
2. **Per-index event study** (this module's main addition): for each index,
   fit a real market model against the *same benchmark the company itself
   uses*, compute abnormal returns, and - anchored to the candidate incident
   days already flagged for the company, not an independent search for the
   index's own unrelated events - report that index's abnormal return, CAR,
   and significance (permutation + Student's-t p-value) on each of those
   days. This directly answers "did this index also move unusually on the
   day the news broke," i.e. whether an event had a sector-wide or
   market-wide footprint, not just a company-specific one. It reuses the
   exact machinery ``ceia.eventstudy.rank_incidents`` already uses for the
   company (``returns.build``, ``cumulative_abnormal_return``,
   ``permutation_test_car``) rather than inventing a parallel test.

An index whose ticker *is* the company's own benchmark (typically Nifty 50)
cannot sensibly be event-studied against itself - the residual variance of a
series regressed on itself is zero, so a z-score would divide by ~0. That
one index keeps its descriptive stats only, with a note explaining why.

Reuses the same provider chain as the company/benchmark price fetch
(``ceia.returns.build`` / ``ceia.prices.load_prices``) rather than a
special-cased fetch path, so it degrades exactly like the existing
secondary-benchmark feature does: one index failing to load never sinks the
others or the rest of the report.

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
from .returns import DEFAULT_PERMUTATIONS
from .returns import build as build_returns
from .returns import cumulative_abnormal_return, daily_returns, permutation_test_car

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
    # Per-index event study (empty when skipped, unavailable, or no
    # candidate days were passed in).
    model_kind: str = ""
    model_note: str = ""
    beta: float = float("nan")
    r_squared: float = float("nan")
    event_study_note: str = ""
    incident_stats: dict = field(default_factory=dict)  # day.isoformat() -> stats dict

    @property
    def available(self) -> bool:
        return not self.daily.empty


def _descriptive_only(name: str, ticker: str, start: date, end: date,
                      providers: list[PriceProvider] | None,
                      event_study_note: str = "") -> IndexSeries:
    """The rebase/window-return path with no market-model event study -
    used both when an index fails to load and for the index that IS the
    company's own benchmark (see module docstring)."""
    try:
        frame, provider_name = load_prices(ticker, start, end, providers=providers)
    except PriceError as exc:
        return IndexSeries(name=name, ticker=ticker, note=f"unavailable: {exc}")
    window = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        return IndexSeries(name=name, ticker=ticker, provider=provider_name,
                           note="no trading days in window")
    closes = window["close"].astype(float)
    daily = pd.DataFrame(index=window.index)
    daily["close"] = closes
    daily["return"] = daily_returns(closes)
    daily["level"] = closes / closes.iloc[0] * 100
    return IndexSeries(
        name=name, ticker=ticker, provider=provider_name, daily=daily,
        window_return=float(closes.iloc[-1] / closes.iloc[0] - 1),
        event_study_note=event_study_note,
    )


def load_nifty_indices(
    start: date,
    end: date,
    benchmark: str,
    candidate_days: list[date] | None = None,
    event_window: tuple[int, int] = (-1, 3),
    providers: list[PriceProvider] | None = None,
    indices: dict[str, str] | None = None,
    lead_in_days: int = 200,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> dict[str, IndexSeries]:
    """Best-effort load of each configured index, with a per-index event
    study anchored to ``candidate_days`` (typically the company's own
    already-flagged incident days).

    One index failing to load (bad symbol, provider outage, no CSV fixture
    in tests) degrades to a note on that index alone - the same
    never-sink-the-whole-run behaviour ``analyze.py`` already applies to the
    secondary benchmark. Pass ``candidate_days=None`` or ``[]`` to skip the
    event study entirely and get only the descriptive rebase/window-return
    stats (e.g. before any incidents have been flagged).
    """
    indices = NIFTY_INDICES if indices is None else indices
    candidate_days = candidate_days or []
    candidate_set = set(candidate_days)
    out: dict[str, IndexSeries] = {}

    for name, ticker in indices.items():
        if ticker == benchmark:
            out[name] = _descriptive_only(
                name, ticker, start, end, providers,
                event_study_note=(f"same ticker as the primary benchmark "
                                  f"({benchmark}) - an abnormal-return event "
                                  "study against itself is not meaningful."),
            )
            continue
        if not candidate_days:
            out[name] = _descriptive_only(name, ticker, start, end, providers)
            continue

        try:
            out[name] = _event_study_index(
                name, ticker, benchmark, start, end, candidate_days, candidate_set,
                event_window, providers, lead_in_days, permutations,
            )
        except Exception as exc:
            # Not just PriceError: fit_market_model/cumulative_abnormal_return/
            # permutation_test_car can raise on degenerate per-index data (e.g.
            # too little overlapping history for that specific index) that has
            # nothing to do with the price fetch itself. Any of those failures
            # must degrade to a note on this one index, per this module's own
            # documented contract - not crash load_nifty_indices (and with it
            # the whole report) for every other index and the rest of the run.
            out[name] = IndexSeries(name=name, ticker=ticker,
                                    note=f"unavailable: {exc}")
    return out


def _event_study_index(
    name: str, ticker: str, benchmark: str, start: date, end: date,
    candidate_days: list[date], candidate_set: set[date],
    event_window: tuple[int, int], providers: list[PriceProvider] | None,
    lead_in_days: int, permutations: int,
) -> IndexSeries:
    frame, model, meta = build_returns(
        ticker, benchmark, start, end,
        lead_in_days=lead_in_days, providers=providers,
    )

    window = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        return IndexSeries(name=name, ticker=ticker,
                           provider=meta["company_provider"],
                           note="no trading days in window")

    closes = window["close"].astype(float)
    daily = pd.DataFrame(index=window.index)
    daily["close"] = closes
    daily["return"] = daily_returns(closes)
    daily["level"] = closes / closes.iloc[0] * 100

    incident_stats = {}
    for day in candidate_days:
        ts = pd.Timestamp(day)
        if ts not in window.index:
            continue
        row = window.loc[ts]
        car = cumulative_abnormal_return(frame, day, event_window)
        car.update(permutation_test_car(
            frame, day, event_window, exclude_days=candidate_set,
            n_permutations=permutations,
        ))
        incident_stats[day.isoformat()] = {
            "abnormal_return": float(row["abnormal_return"]),
            "abnormal_return_z": float(row["abnormal_return_z"]),
            **car,
        }

    return IndexSeries(
        name=name, ticker=ticker, provider=meta["company_provider"],
        daily=daily, window_return=float(closes.iloc[-1] / closes.iloc[0] - 1),
        model_kind=model.kind, model_note=model.note,
        beta=model.beta, r_squared=model.r_squared,
        incident_stats=incident_stats,
    )


def same_direction_rate(index: IndexSeries, company_daily: pd.DataFrame,
                        candidate_days: list[date]) -> dict:
    """Of the already-flagged candidate incident days, how many did this
    index move the same direction (sign) as the company that day?

    A quick raw-return heuristic for the summary table - purely descriptive,
    separate from (and less rigorous than) each index's own abnormal-return
    event study in ``incident_stats`` above. Never itself a test and never
    changes which days are flagged.
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
