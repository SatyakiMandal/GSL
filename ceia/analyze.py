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
from .fetcher import DEFAULT_USER_AGENT, Fetcher
from .ingest import IngestResult, run as run_ingest
from .models import NewsItem, RunConfig
from .prices import PriceError

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

    def to_dict(self) -> dict:
        table = self.daily.reset_index()
        table["date"] = table["date"].astype(str)
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
    )
    eventstudy.attach_headlines(incidents, items)

    return Analysis(
        config=config,
        daily=table,
        incidents=incidents,
        price_meta=price_meta,
        news_meta=news_meta,
        caveats=eventstudy.caveats(table, incidents, model.kind,
                                   price_meta.get("ar_scale_source", "")),
        unattributed=align.unattributed(items),
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

    if analysis.daily.empty:
        print("\nNo trading days in the analysis window.")
        return

    print(f"\n{'-' * 74}\nDAILY TABLE (abnormal return = company return - expected)\n{'-' * 74}")
    print(f"{'date':<12}{'ret%':>8}{'bench%':>8}{'abn%':>8}{'z':>7}"
          f"{'news':>6}{'sent':>7}  event")
    for day, row in analysis.daily.iterrows():
        print(f"{str(day):<12}{row['return'] * 100:>8.2f}{row['benchmark_return'] * 100:>8.2f}"
              f"{row['abnormal_return'] * 100:>8.2f}{row['abnormal_return_z']:>7.2f}"
              f"{int(row['unique_count']):>6}{row['weighted_sentiment']:>7.2f}"
              f"  {row['dominant_event']}")

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
        print(f"   coverage: {incident.item_count} item(s) (z={incident.coverage_z:+.2f}), "
              f"tone {incident.mean_sentiment:+.2f} - {agree} the price move")
        if car and car.get("days"):
            t = car.get("t_stat")
            print(f"   CAR[{config.event_window[0]},+{config.event_window[1]}] "
                  f"{car['car'] * 100:+.2f}% over {car['days']} trading days "
                  f"({car['start']} to {car['end']})"
                  + (f", t={t:.2f}" if t is not None else ""))
            if car.get("note"):
                print(f"   note: {car['note']}")
        for headline in incident.headlines:
            print(f"     - {headline[:100]}")

    print(f"\n{'=' * 74}\nHOW TO READ THIS\n{'=' * 74}")
    for note in analysis.caveats:
        print(f"  * {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: event study")
    parser.add_argument("--company", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--news", default=None,
                        help="Reuse a Phase 1 JSON run instead of re-scraping.")
    parser.add_argument("--event-window", nargs=2, type=int, default=[-1, 3],
                        metavar=("BEFORE", "AFTER"))
    parser.add_argument("--lead-in-days", type=int, default=200)
    parser.add_argument("--coverage-z", type=float, default=eventstudy.DEFAULT_COVERAGE_Z)
    parser.add_argument("--return-z", type=float, default=eventstudy.DEFAULT_RETURN_Z)
    parser.add_argument("--min-relevance", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--price-csv", default=None,
                        help="Directory of <SYMBOL>.csv files; forces the CSV provider.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--out", default="out/analysis.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = RunConfig(
        company=args.company, ticker=args.ticker, benchmark=args.benchmark,
        start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
        aliases=args.alias, event_window=tuple(args.event_window),
        min_relevance=args.min_relevance,
    )

    if args.news:
        items, news_meta = load_news_from_file(Path(args.news))
        log.info("loaded %d items from %s", len(items), args.news)
    else:
        fetcher = Fetcher(cache_dir=args.cache_dir, user_agent=args.user_agent)
        ingested: IngestResult = run_ingest(config, fetcher=fetcher, limit=args.limit)
        items, news_meta = ingested.items, ingested.to_dict()
        news_meta.pop("items", None)

    providers = None
    if args.price_csv:
        from .prices import CsvProvider
        providers = [CsvProvider(args.price_csv)]

    try:
        analysis = analyse(
            config, items, news_meta, lead_in_days=args.lead_in_days,
            coverage_threshold=args.coverage_z, return_threshold=args.return_z,
            providers=providers,
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


if __name__ == "__main__":
    main()
