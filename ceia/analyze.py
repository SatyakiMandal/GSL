"""Phase 2 entry point: run the event study over a company and date range.

Reuses Phase 1 ingestion (or a saved run via ``--news``), loads prices, computes
abnormal returns, and ranks candidate incident days.

    python -m ceia.analyze --company "Adani Enterprises" --ticker ADANIENT.NS \\
        --start 2023-01-24 --end 2023-02-10 --alias Adani

One ordering detail matters: prices are loaded **before** news is attributed to
trading days, so attribution can use the exchange's real calendar rather than a
weekday approximation. That is what stops a story published on the evening of
25 January 2023 being credited to the 26th, which was Republic Day and not a
trading day at all.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import align, eventstudy, returns
from . import financials as financials_mod
from . import macro as macro_mod
from . import nifty as nifty_mod
from .fetcher import DEFAULT_USER_AGENT, Fetcher
from .ingest import DEFAULT_SOURCES, IngestResult, run as run_ingest
from .models import NewsItem, RunConfig
from .prices import PriceError, PriceProvider
from .sources import WAYBACK_SOURCES
from .ticker_lookup import TickerLookupError, resolve_ticker

# Every source this CLI can drive in one command, live-scraped sources plus
# the best-effort Wayback Machine fallback (see README Phase 8) - the
# default for --sources, so a plain `ceia.analyze` run checks everything
# without the caller needing to enumerate sources by hand.
ALL_RUNNABLE_SOURCES = DEFAULT_SOURCES + WAYBACK_SOURCES

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    config: RunConfig
    daily: pd.DataFrame
    incidents: list[eventstudy.Incident]
    price_meta: dict
    news_meta: dict
    caveats: list[str] = field(default_factory=list)
    unattributed: list[NewsItem] = field(default_factory=list)
    correlation: dict = field(default_factory=dict)
    emotion_summary: dict = field(default_factory=dict)
    secondary_daily: pd.DataFrame | None = None
    secondary_meta: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    macro_events: list = field(default_factory=list)
    macro: dict = field(default_factory=dict)
    nifty_indices: dict = field(default_factory=dict)
    financials: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        table = self.daily.reset_index()
        table["date"] = table["date"].astype(str)
        secondary = dict(self.secondary_meta)
        if self.secondary_daily is not None and not self.secondary_daily.empty:
            secondary_table = self.secondary_daily.reset_index()
            secondary_table["date"] = secondary_table["date"].astype(str)
            secondary["daily"] = json.loads(secondary_table.to_json(orient="records"))
        nifty = {}
        for name, index in self.nifty_indices.items():
            entry = {"ticker": index.ticker, "provider": index.provider,
                     "window_return": index.window_return, "note": index.note,
                     "model": index.model_kind, "model_note": index.model_note,
                     "beta": index.beta, "r_squared": index.r_squared,
                     "event_study_note": index.event_study_note,
                     "incident_stats": index.incident_stats}
            if index.available:
                idx_table = index.daily.reset_index()
                idx_table["date"] = idx_table["date"].astype(str)
                entry["daily"] = json.loads(idx_table.to_json(orient="records"))
            nifty[name] = entry
        return {
            "company": self.config.company,
            "ticker": self.config.ticker,
            "benchmark": self.config.benchmark,
            "start": self.config.start.isoformat(),
            "end": self.config.end.isoformat(),
            "event_window": list(self.config.event_window),
            "prices": self.price_meta,
            "news": self.news_meta,
            "caveats": self.caveats,
            "sentiment_return_correlation": self.correlation,
            "emotion_return_summary": self.emotion_summary,
            "secondary_benchmark": secondary,
            "threshold_robustness": self.robustness,
            "flagging_diagnostics": self.diagnostics,
            "macro": self.macro,
            "nifty_indices": nifty,
            "financials": self.financials,
            "unattributed_items": [
                {"url": i.url, "source": i.source, "headline": i.headline,
                 "reason": i.timestamp_confidence}
                for i in self.unattributed
            ],
            "daily": json.loads(table.to_json(orient="records")),
            "incidents": [i.to_dict() for i in self.incidents],
        }


def load_news_from_file(path: Path) -> tuple[list[NewsItem], dict]:
    """Rehydrate a Phase 1 run, so an analysis can be re-run without re-scraping."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for record in payload.get("items", []):
        record = dict(record)
        published = record.pop("published_at", None)
        record.pop("trading_day", None)
        item = NewsItem(**record)
        if published:
            item.published_at = pd.Timestamp(published).to_pydatetime()
        items.append(item)
    meta = {k: v for k, v in payload.items() if k != "items"}
    return items, meta


def analyse(
    config: RunConfig,
    items: list[NewsItem],
    news_meta: dict,
    lead_in_days: int = 200,
    coverage_threshold: float = eventstudy.DEFAULT_COVERAGE_Z,
    return_threshold: float = eventstudy.DEFAULT_RETURN_Z,
    providers=None,
    permutations: int = returns.DEFAULT_PERMUTATIONS,
    macro_provider: PriceProvider | None = None,
    macro_fetcher=None,
    skip_nifty_indices: bool = False,
    financials_fetcher=None,
    skip_financials: bool = False,
) -> Analysis:
    frame, model, price_meta = returns.build(
        config.ticker, config.benchmark, config.start, config.end,
        lead_in_days=lead_in_days, providers=providers,
    )

    # Real exchange calendar, now that prices are in hand.
    calendar = returns.trading_days(frame)
    align.attribute_all(items, calendar)
    price_meta["trading_days_in_window"] = len(
        [d for d in calendar if config.start <= d <= config.end])

    coverage = eventstudy.aggregate_by_day(items)
    table = eventstudy.build_daily_table(frame, coverage, config.start, config.end)
    incidents = eventstudy.rank_incidents(
        table, frame, event_window=config.event_window,
        coverage_threshold=coverage_threshold, return_threshold=return_threshold,
        permutations=permutations,
    )
    eventstudy.attach_headlines(incidents, items)
    correlation = eventstudy.sentiment_return_correlation(table)
    emotion_summary = eventstudy.emotion_valence_summary(table)
    robustness = eventstudy.robustness_check(
        table, frame, incidents, config.event_window,
        coverage_threshold, return_threshold,
    )
    diagnostics = eventstudy.flagging_diagnostics(
        table, coverage_threshold, return_threshold, frame=frame,
    )

    secondary_daily = None
    secondary_meta: dict = {}
    if config.benchmark2:
        secondary_meta["ticker"] = config.benchmark2
        try:
            frame2, model2, price_meta2 = returns.build(
                config.ticker, config.benchmark2, config.start, config.end,
                lead_in_days=lead_in_days, providers=providers,
            )
        except PriceError as exc:
            # A bad or unreachable peer ticker should not sink the whole run
            # - the primary benchmark comparison above is unaffected either
            # way, so this degrades to "not shown" rather than a hard failure.
            secondary_meta["note"] = f"secondary benchmark unavailable: {exc}"
            log.warning("secondary benchmark: %s", exc)
        else:
            secondary_daily = frame2.loc[
                pd.Timestamp(config.start):pd.Timestamp(config.end),
                ["close", "benchmark_close", "return", "benchmark_return",
                 "abnormal_return", "abnormal_return_z"],
            ].rename(columns={
                "benchmark_close": "secondary_close",
                "benchmark_return": "secondary_return",
                "abnormal_return": "secondary_abnormal_return",
                "abnormal_return_z": "secondary_abnormal_return_z",
            }).drop(columns=["close", "return"])
            # `table`'s index is plain date objects (build_daily_table sets
            # "date" as the index key, one date per row), while frame2's
            # index is a DatetimeIndex - joining or comparing the two without
            # normalising first would silently match nothing (different
            # dtypes never compare equal) rather than raise, exactly the
            # "looks fine, finds zero rows" trap this codebase has hit
            # before with the CDATA sitemap regex and the --limit truncation.
            secondary_daily.index = pd.Index(
                [ts.date() for ts in secondary_daily.index], name="date")
            secondary_meta.update({
                "provider": price_meta2.get("benchmark_provider"),
                "model": model2.kind,
                "alpha": model2.alpha,
                "beta": model2.beta,
                "r_squared": model2.r_squared,
                "model_note": model2.note,
                "note": (f"abnormal return of {config.ticker} recomputed against "
                        f"{config.benchmark2} as a second, independent benchmark "
                        "- a peer or sector index rather than the broad market."),
            })

    macro_events = macro_mod.macro_events_in_window(config.start, config.end)
    macro_summary = macro_mod.macro_summary(
        config.start, config.end, provider=macro_provider, fetcher=macro_fetcher)

    nifty_indices = ({} if skip_nifty_indices else
                     nifty_mod.load_nifty_indices(
                         config.start, config.end, config.benchmark,
                         candidate_days=[i.day for i in incidents],
                         event_window=config.event_window,
                         providers=providers, lead_in_days=lead_in_days,
                         permutations=permutations,
                     ))

    financials = ({} if skip_financials else
                 financials_mod.financials_summary(
                     config.ticker, fetcher=financials_fetcher))

    return Analysis(
        config=config,
        daily=table,
        incidents=incidents,
        price_meta=price_meta,
        news_meta=news_meta,
        caveats=eventstudy.caveats(table, incidents, model.kind,
                                   price_meta.get("ar_scale_source", "")),
        unattributed=align.unattributed(items),
        correlation=correlation,
        emotion_summary=emotion_summary,
        secondary_daily=secondary_daily,
        secondary_meta=secondary_meta,
        robustness=robustness,
        diagnostics=diagnostics,
        macro_events=macro_events,
        macro=macro_summary,
        nifty_indices=nifty_indices,
        financials=financials,
    )


def _print(analysis: Analysis) -> None:
    config = analysis.config
    print(f"\n{'=' * 74}")
    print(f"EVENT STUDY - {config.company} ({config.ticker}) vs {config.benchmark}")
    print(f"{config.start} to {config.end}")
    print("=" * 74)

    meta = analysis.price_meta
    print(f"\nPrices: {meta['company_provider']} / {meta['benchmark_provider']}, "
          f"{meta['rows_in_analysis_window']} trading days in window")
    print(f"Model : {meta['model']}  alpha={meta['alpha']:.5f} beta={meta['beta']:.3f} "
          f"R2={meta['r_squared']:.3f}" if meta["model"] == "market-model"
          else f"Model : {meta['model']} ({meta['model_note']})")

    news = analysis.news_meta.get("stats", {})
    if news:
        print(f"News  : {news.get('unique_after_dedupe', '?')} unique items, "
              f"{news.get('duplicates', 0)} duplicates, "
              f"{news.get('after_close', 0)} published after the close")
    if analysis.unattributed:
        print(f"        {len(analysis.unattributed)} item(s) had no usable "
              f"timestamp and were NOT attributed to any trading day")

    d = analysis.diagnostics
    if d.get("trading_days"):
        print(f"Flagging: {d['trading_days']} trading day(s), {d['days_with_news']} "
              f"with news; {d['candidates']} candidate(s), {d['coverage_only']} "
              f"busy/toned but ordinary move, {d['return_only']} unusual move but "
              f"ordinary coverage, {d['no_coverage_big_move']} unusual move with NO "
              f"coverage collected, {d['routine']} routine")

    corr = analysis.correlation
    if corr.get("r") is not None:
        print(f"Sentiment/return correlation: r={corr['r']:+.3f} "
              f"(R2={corr['r_squared']:.3f}, n={corr['n']}) — {corr['note']}")
    elif corr:
        print(f"Sentiment/return correlation: not computed — {corr.get('note', '')}")

    groups = analysis.emotion_summary.get("groups") or {}
    if groups:
        print("Emotion valence vs abnormal return (GoEmotions, descriptive only):")
        for valence in ("positive", "negative", "ambiguous"):
            g = groups.get(valence)
            if not g:
                continue
            print(f"   {valence:<10} n={g['n_days']:<3} "
                  f"mean abnormal={g['mean_abnormal_return'] * 100:+.2f}%  "
                  f"mean sentiment={g['mean_weighted_sentiment']:+.2f}  "
                  f"({', '.join(g['labels_seen'])})")

    sec_meta = analysis.secondary_meta
    if sec_meta.get("model") is not None:
        print(f"Secondary benchmark ({sec_meta['ticker']}): {sec_meta['model']} "
              f"beta={sec_meta['beta']:.3f} R2={sec_meta['r_squared']:.3f} — "
              f"{sec_meta['note']}")
    elif sec_meta.get("note"):
        print(f"Secondary benchmark ({sec_meta.get('ticker', '?')}): "
              f"{sec_meta['note']}")

    if analysis.nifty_indices:
        print("Nifty sector indices (descriptive backdrop, not part of flagging):")
        candidate_days = [i.day for i in analysis.incidents]
        for name, index in analysis.nifty_indices.items():
            if not index.available:
                print(f"   {name:<12} unavailable — {index.note}")
                continue
            agreement = nifty_mod.same_direction_rate(
                index, analysis.daily, candidate_days)
            agree_str = (f", moved with {config.ticker} on {agreement['agree']}/"
                        f"{agreement['n']} candidate day(s)" if agreement["n"] else "")
            print(f"   {name:<12} {index.window_return * 100:+.2f}% over the window"
                  f"{agree_str}")

    fin = analysis.financials
    if fin:
        if fin.get("note"):
            print(f"Financials: {fin['note']}")
        else:
            print(f"Financials (latest reported quarter, {fin.get('as_of', '?')}, "
                  f"{fin.get('currency_unit', '')}, {fin['statement_kind']} — "
                  f"{fin['screener_url']}):")
            rev = fin.get("revenue")
            if rev:
                qoq = f"{rev['qoq_change'] * 100:+.1f}% QoQ" if rev["qoq_change"] is not None else ""
                yoy = f"{rev['yoy_change'] * 100:+.1f}% YoY" if rev["yoy_change"] is not None else ""
                print(f"   {rev['label']}: {rev['latest']:,.0f}"
                      + (f" ({', '.join(p for p in (qoq, yoy) if p)})" if qoq or yoy else ""))
            exp = fin.get("expenses")
            if exp:
                print(f"   Expenses: {exp['latest']:,.0f}")
            if fin.get("nopat") is not None:
                print(f"   NOPAT: {fin['nopat']:,.0f} ({fin['nopat_note']})")
            else:
                print(f"   NOPAT: not computed — {fin['nopat_note']}")
            ob = fin.get("order_book")
            if ob:
                print(f"   Order Book: {ob['latest']:,.0f}")
            else:
                print(f"   Order Book: {fin['order_book_note']}")

    if analysis.daily.empty:
        print("\nNo trading days in the analysis window.")
        return

    print(f"\n{'-' * 74}\nDAILY TABLE (abnormal return = company return - expected)\n{'-' * 74}")
    print(f"{'date':<12}{'ret%':>8}{'bench%':>8}{'abn%':>8}{'z':>7}"
          f"{'news':>6}{'sent':>7}{'volZ':>7}  event")
    for day, row in analysis.daily.iterrows():
        vol_z = row.get("volume_z")
        vol_display = f"{vol_z:>7.2f}" if pd.notna(vol_z) else f"{'—':>7}"
        print(f"{str(day):<12}{row['return'] * 100:>8.2f}{row['benchmark_return'] * 100:>8.2f}"
              f"{row['abnormal_return'] * 100:>8.2f}{row['abnormal_return_z']:>7.2f}"
              f"{int(row['unique_count']):>6}{row['weighted_sentiment']:>7.2f}"
              f"{vol_display}  {row['dominant_event']}")

    print(f"\n{'-' * 74}\nCANDIDATE INCIDENT DAYS (ranked)\n{'-' * 74}")
    if not analysis.incidents:
        print("None flagged: no day had both unusual coverage and an unusual "
              "abnormal return at the configured thresholds.")
    for rank, incident in enumerate(analysis.incidents, 1):
        car = incident.car
        agree = "consistent with" if incident.direction_agrees else "OPPOSITE to"
        print(f"\n{rank}. {incident.day}  score={incident.score:.2f}")
        print(f"   abnormal return {incident.abnormal_return * 100:+.2f}% "
              f"(z={incident.abnormal_return_z:+.2f}); raw {incident.raw_return * 100:+.2f}%, "
              f"benchmark {incident.benchmark_return * 100:+.2f}%")
        if incident.volume_z and pd.notna(incident.volume):
            print(f"   volume: {incident.volume:,.0f} (z={incident.volume_z:+.2f}) "
                  "— a corroborating signal, not part of the flagging test")
        if analysis.secondary_daily is not None:
            sec_row = analysis.secondary_daily.loc[
                analysis.secondary_daily.index == incident.day]
            if not sec_row.empty:
                sec_ar = sec_row["secondary_abnormal_return"].iloc[0]
                sec_z = sec_row["secondary_abnormal_return_z"].iloc[0]
                if pd.notna(sec_ar):
                    print(f"   vs {analysis.secondary_meta['ticker']}: "
                          f"abnormal return {sec_ar * 100:+.2f}% (z={sec_z:+.2f})")
        print(f"   coverage: {incident.item_count} item(s) (z={incident.coverage_z:+.2f}), "
              f"tone {incident.mean_sentiment:+.2f} - {agree} the price move")
        if car and car.get("days"):
            t = car.get("t_stat")
            print(f"   CAR[{config.event_window[0]},+{config.event_window[1]}] "
                  f"{car['car'] * 100:+.2f}% over {car['days']} trading days "
                  f"({car['start']} to {car['end']})"
                  + (f", t={t:.2f}" if t is not None else ""))
            if car.get("p_value") is not None:
                print(f"   permutation p-value: {car['p_value']:.4f} "
                      f"(n={car['n']} placebo windows) - the fraction of "
                      "random same-length windows in this company's own "
                      "history with as extreme a CAR")
            elif car.get("p_value_note"):
                print(f"   permutation p-value: not computed - {car['p_value_note']}")
            if car.get("note"):
                print(f"   note: {car['note']}")
        day_key = incident.day.isoformat()
        for name, index in analysis.nifty_indices.items():
            stats = index.incident_stats.get(day_key)
            if not stats:
                continue
            print(f"   {name}: abnormal return {stats['abnormal_return'] * 100:+.2f}% "
                  f"(z={stats['abnormal_return_z']:+.2f}), "
                  f"CAR[{config.event_window[0]},+{config.event_window[1]}] "
                  f"{stats['car'] * 100:+.2f}%"
                  + (f", p={stats['p_value']:.3f}"
                     if stats.get("p_value") is not None else ""))
        robust = analysis.robustness.get("days", {}).get(incident.day.isoformat())
        if robust:
            print(f"   robustness: flagged in {robust['flagged_in']}/{robust['of']} "
                  f"threshold combinations tried")
        for h in incident.headlines:
            print(f"     - [{h['source']}] {h['headline'][:100]} "
                  f"({h['sentiment_label']}, rel={h['relevance']:.2f})")
            if h.get("url"):
                print(f"       {h['url']}")
            if h.get("summary"):
                print(f"       \"{h['summary'][:160]}\"")

    print(f"\n{'=' * 74}\nHOW TO READ THIS\n{'=' * 74}")
    for note in analysis.caveats:
        print(f"  * {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: event study")
    parser.add_argument("--company", required=True)
    parser.add_argument("--ticker", default=None,
                        help="e.g. ADANIENT.NS. Auto-detected from --company if omitted.")
    parser.add_argument("--exchange", default="NSE",
                        help="Preferred exchange for ticker auto-detection (NSE or BSE).")
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--benchmark2", default=None,
                        help="Optional second index/peer ticker (e.g. a sector "
                             "index or a direct competitor) for a side-by-side "
                             "abnormal-return comparison. The primary --benchmark "
                             "still drives incident detection; this is a second, "
                             "purely descriptive lens.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--sources", nargs="*", default=ALL_RUNNABLE_SOURCES,
                        help="News sources to scrape (live scraping only, "
                             "ignored with --news). Defaults to every "
                             "runnable source in one command: the 5 "
                             "always-on sources plus business_standard and "
                             "livemint's best-effort Wayback Machine "
                             "fallback (see README Phase 8), which can "
                             "sometimes find nothing for a given window - "
                             "check source_status in the JSON output or the "
                             "console log for what each source actually "
                             "found. Pass an explicit, narrower list to opt "
                             "out of any of them.")
    parser.add_argument("--news", default=None,
                        help="Reuse a Phase 1 JSON run instead of re-scraping.")
    parser.add_argument("--event-window", nargs=2, type=int, default=[-1, 3],
                        metavar=("BEFORE", "AFTER"))
    parser.add_argument("--lead-in-days", type=int, default=200)
    parser.add_argument("--coverage-z", type=float, default=eventstudy.DEFAULT_COVERAGE_Z)
    parser.add_argument("--return-z", type=float, default=eventstudy.DEFAULT_RETURN_Z)
    parser.add_argument("--permutations", type=int, default=returns.DEFAULT_PERMUTATIONS,
                        help="Placebo windows drawn per incident for the CAR "
                             "permutation-test p-value (0 disables it).")
    parser.add_argument("--min-relevance", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap articles fetched (live scraping only), evenly "
                             "spread across the whole date range rather than "
                             "just its earliest days.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent article fetches, live scraping only "
                             "(default 8). Requests to any single origin are "
                             "still serialised at the configured interval "
                             "regardless of --workers.")
    parser.add_argument("--skip-alias-widening", action="store_true",
                        help="Live scraping only. Don't try the company's "
                             "leading word as an extra alias (see "
                             "ceia.ingest.widen_aliases()). On by default; "
                             "costs one extra ticker-search request per run.")
    parser.add_argument("--skip-slug-prefilter", action="store_true",
                        help="Live scraping only. Never use the URL-slug "
                             "guess to narrow candidates before fetching, "
                             "regardless of a source's volume this run - "
                             "every candidate from every source goes "
                             "straight to full-text relevance scoring "
                             "instead. Catches a story whose slug never "
                             "names the company at all, at the cost of far "
                             "more fetches on a high-volume source.")
    parser.add_argument("--skip-macro-prices", action="store_true",
                        help="Don't fetch Brent crude, the G-Sec yield, or "
                             "the fiscal deficit for the macro-economic "
                             "backdrop section. Repo rate events (no network "
                             "needed) still show either way. Useful if Yahoo "
                             "is rate-limiting this connection - see the "
                             "README's note on shared/proxied egress.")
    parser.add_argument("--skip-nifty-indices", action="store_true",
                        help="Don't fetch Nifty 50/Bank/Auto/Energy/IT/Metal "
                             "or run their per-index event studies. Six more "
                             "lead-in price fetches and market-model fits "
                             "otherwise - useful if Yahoo is rate-limiting "
                             "this connection.")
    parser.add_argument("--skip-financials", action="store_true",
                        help="Don't fetch revenue/expense/NOPAT/order-book "
                             "fundamentals from screener.in (see README "
                             "Phase 9).")
    parser.add_argument("--price-csv", default=None,
                        help="Directory of <SYMBOL>.csv files; forces the CSV provider.")
    parser.add_argument("--api-key", default=None,
                        help="Alpha Vantage API key. Overrides ALPHAVANTAGE_API_KEY; "
                             "avoids needing to set an environment variable at all, "
                             "which on Windows PowerShell means $env:NAME = 'value', "
                             "not the cmd.exe-style 'set NAME=value'.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--out", default="out/analysis.json")
    parser.add_argument("--html", default="out/report.html",
                        help="Standalone HTML report path; --html '' to skip.")
    parser.add_argument("--pdf", default=None,
                        help="Also render the report to this PDF path. Needs "
                             'Playwright: pip install -e ".[pdf]" && '
                             "playwright install chromium.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ticker = args.ticker
    if not ticker:
        try:
            match = resolve_ticker(args.company, exchange=args.exchange)
        except TickerLookupError as exc:
            print(f"\nTicker lookup failed: {exc}")
            raise SystemExit(2)
        ticker = match.symbol
        print(f"Resolved ticker: {args.company!r} -> {ticker} ({match.name})"
              + ("" if match.exact_exchange_match
                 else f" -- not listed on {args.exchange}, using nearest match"))

    try:
        config = RunConfig(
            company=args.company, ticker=ticker, benchmark=args.benchmark,
            benchmark2=args.benchmark2,
            start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
            aliases=args.alias, sources=args.sources,
            event_window=tuple(args.event_window),
            min_relevance=args.min_relevance,
        )
    except ValueError as exc:
        print(f"\n{exc}")
        raise SystemExit(2)

    if args.news:
        items, news_meta = load_news_from_file(Path(args.news))
        log.info("loaded %d items from %s", len(items), args.news)
    else:
        fetcher = Fetcher(cache_dir=args.cache_dir, user_agent=args.user_agent)
        ingested: IngestResult = run_ingest(config, fetcher=fetcher, limit=args.limit,
                                            max_workers=args.workers,
                                            skip_alias_widening=args.skip_alias_widening,
                                            skip_slug_prefilter=args.skip_slug_prefilter)
        items, news_meta = ingested.items, ingested.to_dict()
        news_meta.pop("items", None)

    providers = None
    if args.price_csv:
        from .prices import CsvProvider
        providers = [CsvProvider(args.price_csv)]
    elif args.api_key:
        # Same provider order as the default chain, with the key injected
        # directly rather than requiring ALPHAVANTAGE_API_KEY to be set.
        from .prices import AlphaVantageProvider, CsvProvider, YahooChartProvider, YFinanceProvider
        providers = [YFinanceProvider(), YahooChartProvider(),
                    AlphaVantageProvider(api_key=args.api_key), CsvProvider()]

    macro_provider = macro_mod.SkippedPriceProvider() if args.skip_macro_prices else None
    macro_fetcher = macro_mod.SkippedFetcher() if args.skip_macro_prices else None
    try:
        analysis = analyse(
            config, items, news_meta, lead_in_days=args.lead_in_days,
            coverage_threshold=args.coverage_z, return_threshold=args.return_z,
            providers=providers, permutations=args.permutations,
            macro_provider=macro_provider, macro_fetcher=macro_fetcher,
            skip_nifty_indices=args.skip_nifty_indices,
            skip_financials=args.skip_financials,
        )
    except PriceError as exc:
        print(f"\nPrice data unavailable: {exc}")
        print("\nThe news pipeline is unaffected. Options: set "
              "ALPHAVANTAGE_API_KEY, or supply CSVs with --price-csv "
              "(columns: date,close).")
        raise SystemExit(2)

    _print(analysis)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis.to_dict(), indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")

    if args.html:
        from .report import write_report
        print(f"wrote {write_report(analysis, args.html)}")

    if args.pdf:
        from .pdf import PdfExportError, render_pdf
        from .report import build_html
        try:
            print(f"wrote {render_pdf(build_html(analysis), args.pdf)}")
        except PdfExportError as exc:
            # The JSON/HTML outputs above already succeeded - a missing or
            # broken PDF dependency should not turn a successful run into a
            # failed one, just a run with one fewer output file.
            print(f"\nPDF export skipped: {exc}")


if __name__ == "__main__":
    main()
