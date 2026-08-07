"""Event-timeline analysis for unlisted / pre-IPO shares (e.g. UnlistedZone).

Deliberately not an event *study* in the sense the rest of this project uses
that word. UnlistedZone's own disclaimer says its prices are "indicative
levels compiled by our team... not a price feed, quote, or offer to deal" -
verified directly against a real company's five-year chart: it carries a
dated point for almost every calendar day (1,321 of them), but only 71 are
distinct values, the rest holding the last real price flat until the next
team revision. That is not daily price discovery, so none of the listed-
stock machinery that depends on it - a market-model beta, a z-score
standardised against daily volatility, a permutation test against a daily
null - has anything valid to stand on here. Building those anyway would
produce a report that *looks* as statistically rigorous as the listed one
while resting on roughly 70 real observations dressed up as thousands.

What the data can honestly support: the real revision points, and the raw
percentage change between one and the next, attributed to the date *range*
between them rather than to a single fabricated day. That is what this
module computes, paired with whatever news the existing ingestion pipeline
(unchanged - it does not care whether a company is listed) found published
in that same range.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from html import unescape
from pathlib import Path

import pandas as pd

from .eventstudy import _excerpt
from .fetcher import DEFAULT_USER_AGENT, Fetcher
from .ingest import DEFAULT_SOURCES, IngestResult, run as run_ingest
from . import macro as macro_mod
from .models import NewsItem, RunConfig
from .sources import UNLISTED_EXTRA_SOURCES

log = logging.getLogger(__name__)

# Unlisted/pre-IPO companies get opportunistic coverage at best from the
# mainstream listed-market press DEFAULT_SOURCES draws on - real routine
# coverage of funding rounds and private-market corporate actions sits on
# startup/private-market-focused outlets instead (see sources.py's notes on
# ENTRACKR/VCCIRCLE/INC42). Only ceia.unlisted opts into these; the
# already-verified listed-company pipeline's DEFAULT_SOURCES is untouched.
UNLISTED_DEFAULT_SOURCES = DEFAULT_SOURCES + UNLISTED_EXTRA_SOURCES

# UnlistedZone's chart isn't served from a clean JSON API - the data is
# inlined into a Next.js React Server Components streaming payload
# (self.__next_f.push([1, "...escaped text..."])), where every quote inside
# the embedded JSON is backslash-escaped once. Regex rather than a real
# parser for the same reason extract.py's per-source handling is tolerant
# rather than strict: this is scraped, third-party markup, not a documented,
# stable API. Verified directly against a real page: matches every
# ["YYYY-MM-DD", price] pair in the series (1,321/1,321 on the test page).
_SERIES_POINT_RE = re.compile(r'\\"(\d{4}-\d{2}-\d{2})\\",(\d+(?:\.\d+)?)')


class UnlistedPriceError(RuntimeError):
    pass


class UnlistedCompanyNotFoundError(RuntimeError):
    pass


# UnlistedZone's own `?search=` query parameter is client-side only - a plain
# GET returns the same unfiltered listing regardless of the query, verified
# directly. What does work is its /shares directory: a paginated grid of
# every tracked company, ~24 per page, each card holding the company name
# (class="nm") and a link to its product page (class="det" href="...").
# Extracted by position rather than a real HTML parser for the same reason
# the price-series regex above is: this is scraped, third-party markup with
# no documented API, not something to build a strict parser against.
_DIRECTORY_URL = "https://unlistedzone.com/shares"
_DIRECTORY_NAME_RE = re.compile(r'class="nm">([^<]+)<')
_DIRECTORY_HREF_RE = re.compile(r'class="det" href="(/shares/[^"]+)"')
_DIRECTORY_LAST_PAGE_RE = re.compile(r'"lastPage":(\d+)')

# Boilerplate every directory listing repeats regardless of which company it
# is ("NSE India Limited Unlisted Shares", "Zepto Unlisted Shares (Equity)")
# - stripped before matching so it cannot drown out the part of the name that
# actually distinguishes one company from another.
_DIRECTORY_NOISE_RE = re.compile(
    r"\b(unlisted shares?|share price|buy\s*(?:/|and)?\s*sell(?:\s*online)?|"
    r"equity|limited|ltd)\b", re.I)

# Below this similarity score, resolve_unlisted_url() raises rather than
# guessing - the same rule ceia.ticker_lookup.resolve_ticker() follows for
# listed tickers.
_MATCH_THRESHOLD = 0.5


def _fetch_directory(fetcher: Fetcher, max_pages: int = 20) -> list[tuple[str, str]]:
    """Every (company name, product-page URL) pair UnlistedZone's own
    directory lists, across all its pages (bounded by ``max_pages`` in case
    the ``lastPage`` marker is ever missing from the response)."""
    directory: list[tuple[str, str]] = []
    seen: set[str] = set()
    last_page = max_pages
    page = 1
    while page <= last_page:
        url = _DIRECTORY_URL if page == 1 else f"{_DIRECTORY_URL}?page={page}"
        response = fetcher.get(url)
        hrefs = _DIRECTORY_HREF_RE.findall(response.text)
        if not hrefs:
            break
        if page == 1:
            match = _DIRECTORY_LAST_PAGE_RE.search(response.text)
            if match:
                last_page = min(int(match.group(1)), max_pages)
        names = _DIRECTORY_NAME_RE.findall(response.text)
        for name, href in zip(names, hrefs):
            if href not in seen:
                seen.add(href)
                directory.append((unescape(name).strip(), f"https://unlistedzone.com{href}"))
        page += 1
    return directory


def _normalise_for_match(name: str) -> str:
    name = re.sub(r"[^\w\s]", " ", name.lower())
    name = _DIRECTORY_NOISE_RE.sub(" ", name)
    return " ".join(name.split())


def _match_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query in candidate or candidate in query:
        return 1.0
    return difflib.SequenceMatcher(None, query, candidate).ratio()


def resolve_unlisted_url(company: str, fetcher: Fetcher, max_pages: int = 20) -> str:
    """Find a company's UnlistedZone product-page URL by name - the same
    role :func:`ceia.ticker_lookup.resolve_ticker` plays for listed tickers,
    so ``--url`` is optional for the common case.

    Raises rather than guessing when nothing matches confidently, same rule.
    """
    directory = _fetch_directory(fetcher, max_pages=max_pages)
    if not directory:
        raise UnlistedCompanyNotFoundError(
            "UnlistedZone's company directory could not be read")

    query = _normalise_for_match(company)
    best_name, best_url, best_score = "", "", 0.0
    for name, url in directory:
        score = _match_score(query, _normalise_for_match(name))
        if score > best_score:
            best_name, best_url, best_score = name, url, score

    if best_score < _MATCH_THRESHOLD:
        raise UnlistedCompanyNotFoundError(
            f"no confident match for {company!r} in UnlistedZone's directory "
            f"({len(directory)} companies checked, best guess was {best_name!r} "
            f"at {best_score:.2f}) -- pass --url explicitly"
        )
    log.info("resolved %r -> %s (%r, score=%.2f)", company, best_url, best_name, best_score)
    return best_url


def fetch_price_series(fetcher: Fetcher, url: str) -> pd.DataFrame:
    """The as-displayed daily series for one UnlistedZone product page.

    Every calendar day in the chart gets a row, most of them repeating the
    last real price (see module docstring) - this returns exactly what the
    page shows, forward-fill included. Use :func:`real_updates` to get back
    only the days the indicative price actually changed.
    """
    response = fetcher.get(url)
    pairs = _SERIES_POINT_RE.findall(response.text)
    if not pairs:
        raise UnlistedPriceError(f"no price series found at {url}")
    rows = {pd.Timestamp(d): float(v) for d, v in pairs}
    frame = pd.DataFrame({"close": rows}).sort_index()
    frame.index.name = "date"
    return frame[~frame.index.duplicated(keep="last")]


def real_updates(series: pd.DataFrame) -> pd.DataFrame:
    """Only the rows where the indicative price actually changed.

    Collapses runs of a forward-filled value down to their first
    occurrence - the day the price was actually revised, not every day
    after it that the chart just held flat.
    """
    close = series["close"]
    changed = close.ne(close.shift(1))
    if len(changed):
        changed.iloc[0] = True  # the first observation is always "real"
    return series[changed]


@dataclass
class PriceMove:
    """One real revision: the indicative price at the start of the range,
    what it became by the end, and the news published in between."""
    start_date: date
    end_date: date
    start_price: float
    end_price: float
    headlines: list[dict] = field(default_factory=list)

    @property
    def change(self) -> float:
        return (self.end_price - self.start_price) / self.start_price

    def to_dict(self) -> dict:
        record = asdict(self)
        record["start_date"] = self.start_date.isoformat()
        record["end_date"] = self.end_date.isoformat()
        record["change"] = self.change
        return record


def price_moves(real: pd.DataFrame) -> list[PriceMove]:
    """One move per consecutive pair of real observations, oldest first."""
    dates = [ts.date() for ts in real.index]
    prices = real["close"].tolist()
    return [
        PriceMove(start_date=dates[i], end_date=dates[i + 1],
                 start_price=prices[i], end_price=prices[i + 1])
        for i in range(len(dates) - 1)
    ]


def attach_news_to_moves(moves: list[PriceMove], items: list[NewsItem],
                         limit: int = 5) -> list[PriceMove]:
    """Hang news published inside each move's date range onto that move.

    A move's window is *(previous real date, this real date]* - news
    published the day of the previous revision already explains that
    prior move, not this one.
    """
    for move in moves:
        relevant = [
            i for i in items
            if i.published_at is not None
            and move.start_date < i.published_at.date() <= move.end_date
        ]
        relevant.sort(key=lambda i: (-abs(i.sentiment_score), -i.relevance_score))
        move.headlines = [
            {
                "source": i.source, "headline": i.headline, "url": i.url,
                "sentiment_label": i.sentiment_label,
                "relevance": round(i.relevance_score, 2),
                "summary": (i.snippet.strip() if i.snippet.strip()
                           else _excerpt(i.body) if i.body else ""),
            }
            for i in relevant[:limit]
        ]
    return moves


@dataclass
class UnlistedAnalysis:
    config: RunConfig
    url: str
    series: pd.DataFrame
    moves: list[PriceMove]
    news_meta: dict
    unattributed: list[NewsItem] = field(default_factory=list)
    items: list[NewsItem] = field(default_factory=list)
    macro_events: list = field(default_factory=list)
    macro: dict = field(default_factory=dict)

    def ranked_moves(self, top_n: int | None = None) -> list[PriceMove]:
        ranked = sorted(self.moves, key=lambda m: -abs(m.change))
        return ranked[:top_n] if top_n else ranked

    def to_dict(self) -> dict:
        series = self.series.reset_index()
        series["date"] = series["date"].astype(str)
        return {
            "company": self.config.company,
            "start": self.config.start.isoformat(),
            "end": self.config.end.isoformat(),
            "url": self.url,
            "aliases": self.config.all_aliases,
            "news": self.news_meta,
            "moves": [m.to_dict() for m in self.ranked_moves()],
            "unattributed_items": [
                {"url": i.url, "source": i.source, "headline": i.headline}
                for i in self.unattributed
            ],
            "series": json.loads(series.to_json(orient="records")),
            "macro": self.macro,
        }


def analyse_unlisted(
    config: RunConfig,
    url: str,
    fetcher: Fetcher | None = None,
    items: list[NewsItem] | None = None,
    news_meta: dict | None = None,
    headline_limit: int = 5,
    limit: int | None = None,
    skip_sentiment: bool = False,
    skip_emotion: bool = False,
    max_workers: int = 8,
    macro_provider=None,
    macro_fetcher=None,
) -> UnlistedAnalysis:
    """Wire the price-move computation to the existing news pipeline.

    ``items``/``news_meta`` let a caller reuse a previously-ingested run
    instead of always re-scraping, the same role ``--news`` plays for
    ``ceia.analyze``.
    """
    fetcher = fetcher or Fetcher()
    full_series = fetch_price_series(fetcher, url)
    # Moves are computed from the *full* history so a move that started
    # before `start` but is still the operative price at the window's open
    # is not silently dropped - then trimmed to whichever moves actually
    # overlap the requested window. UnlistedZone's chart commonly reaches
    # back years further than any one run asks for (a real company's carried
    # five years of history in testing), so without this every run would
    # report moves from long before the requested start date.
    moves = [
        m for m in price_moves(real_updates(full_series))
        if m.end_date >= config.start and m.start_date <= config.end
    ]
    series = full_series.loc[str(config.start):str(config.end)]

    if items is None:
        ingested: IngestResult = run_ingest(
            config, fetcher=fetcher, limit=limit,
            skip_sentiment=skip_sentiment, skip_emotion=skip_emotion,
            max_workers=max_workers,
        )
        items, news_meta = ingested.items, ingested.to_dict()
        news_meta.pop("items", None)

    attach_news_to_moves(moves, items, limit=headline_limit)
    unattributed = [i for i in items if i.published_at is None]

    macro_events = macro_mod.macro_events_in_window(config.start, config.end)
    macro_summary = macro_mod.macro_summary(
        config.start, config.end, provider=macro_provider, fetcher=macro_fetcher)

    return UnlistedAnalysis(config=config, url=url, series=series, moves=moves,
                            news_meta=news_meta or {}, unattributed=unattributed,
                            items=items, macro_events=macro_events, macro=macro_summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price-move/news timeline for an unlisted or pre-IPO share")
    parser.add_argument("--company", required=True)
    parser.add_argument("--url", default=None,
                        help="UnlistedZone product page, e.g. "
                             "https://unlistedzone.com/shares/<slug>. "
                             "Auto-detected from --company if omitted.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap news articles fetched, evenly spread across "
                             "the whole date range.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip FinBERT (no model download).")
    parser.add_argument("--skip-emotion", action="store_true",
                        help="Skip GoEmotions (no model download).")
    parser.add_argument("--skip-alias-widening", action="store_true",
                        help="Don't try the company's leading word as an "
                             "extra alias (see ceia.ingest.widen_aliases()).")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Override the source list. Defaults to the "
                             "listed-company sources plus entrackr, vccircle "
                             "and inc42, which cover the unlisted/pre-IPO "
                             "space the mainstream press mostly does not.")
    parser.add_argument("--skip-macro-prices", action="store_true",
                        help="Don't fetch Brent crude, the G-Sec yield, or "
                             "the fiscal deficit for the macro-economic "
                             "backdrop section. Repo rate events (no network "
                             "needed) still show either way. Useful if Yahoo "
                             "is rate-limiting this connection - see the "
                             "README's note on shared/proxied egress.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--out", default="out/unlisted.json")
    parser.add_argument("--html", default="out/unlisted_report.html",
                        help="Standalone HTML report path; --html '' to skip.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = RunConfig(
            company=args.company, ticker="",
            start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
            aliases=args.alias,
            sources=args.sources if args.sources is not None else UNLISTED_DEFAULT_SOURCES,
        )
    except ValueError as exc:
        print(f"\n{exc}")
        raise SystemExit(2)

    fetcher = Fetcher(cache_dir=args.cache_dir, user_agent=args.user_agent)

    url = args.url
    if not url:
        try:
            url = resolve_unlisted_url(args.company, fetcher)
        except UnlistedCompanyNotFoundError as exc:
            print(f"\nUnlistedZone lookup failed: {exc}")
            raise SystemExit(2)
        print(f"Resolved UnlistedZone URL: {args.company!r} -> {url}")

    macro_provider = macro_mod.SkippedPriceProvider() if args.skip_macro_prices else None
    macro_fetcher = macro_mod.SkippedFetcher() if args.skip_macro_prices else None
    try:
        analysis = analyse_unlisted(
            config, url, fetcher=fetcher, limit=args.limit,
            skip_sentiment=args.skip_sentiment, skip_emotion=args.skip_emotion,
            max_workers=args.workers, macro_provider=macro_provider,
            macro_fetcher=macro_fetcher,
        )
    except UnlistedPriceError as exc:
        print(f"\nPrice data unavailable: {exc}")
        raise SystemExit(2)

    ranked = analysis.ranked_moves()
    print(f"\n{'=' * 74}\nUNLISTED PRICE TIMELINE - {config.company}")
    print(f"{config.start} to {config.end}\n{'=' * 74}")
    print(f"\n{len(real_updates(analysis.series))} real price revision(s), "
          f"{len(analysis.moves)} gap(s) between them")
    for rank, move in enumerate(ranked, 1):
        print(f"\n{rank}. {move.start_date} to {move.end_date}  "
              f"{move.change * 100:+.2f}%  "
              f"(Rs {move.start_price:,.2f} -> Rs {move.end_price:,.2f})")
        for h in move.headlines:
            print(f"     - [{h['source']}] {h['headline'][:100]}")
            if h.get("url"):
                print(f"       {h['url']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis.to_dict(), indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")

    if args.html:
        from .report import write_unlisted_report
        print(f"wrote {write_unlisted_report(analysis, args.html)}")


if __name__ == "__main__":
    main()
