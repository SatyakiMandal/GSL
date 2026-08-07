"""Phase 1 pipeline: discover -> fetch -> parse -> filter -> dedupe -> score.

Run with ``python -m ceia.ingest --company "Adani Enterprises" ...``.

Two efficiency notes that matter at this scale. A month of Economic Times
coverage is ~13,000 URLs and Moneycontrol ~8,000; fetching all of them to find
the ~200 about one company would be both slow and rude. So candidate URLs are
**pre-filtered on their slug** before anything is fetched — these sites all put
the story's subject in the URL. Slug filtering is deliberately loose (any alias
token matches), with the real relevance scoring done on parsed text afterwards.
That trade only pays for itself on a source too large to fetch whole, though —
:func:`prefilter` skips the slug guess entirely for any source whose candidate
count *this run* is small enough to fetch in full (``PREFILTER_SKIP_THRESHOLD``),
letting the real scorer see everything instead of a URL-based guess.

Everything degrades per source: a site that blocks us or changes layout is
recorded in the run's ``source_status`` and the rest of the pipeline continues
(PRD Section 10).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import align, dedupe, relevance
from . import wayback as wayback_mod
from .discovery import Candidate, discover
from .extract import parse_article
from .extract import IST
from .fetcher import DEFAULT_USER_AGENT, Fetcher, RobotsDisallowed
from .emotion import GoEmotionScorer
from .models import NewsItem, RunConfig
from .sentiment import FinBertScorer
from .sources import ALL_SOURCES, WAYBACK_SOURCES
from .ticker_lookup import TickerLookupError, resolve_ticker

log = logging.getLogger(__name__)

DEFAULT_SOURCES = ["economic_times", "financial_express", "business_line", "moneycontrol",
                   "business_today"]

# Business Standard is deliberately absent from the normal, live-sitemap
# discovery path: its Akamai edge returns 403 for every request including
# robots.txt, so permission to crawl live cannot be established. Moneycontrol
# replaces it day to day. A best-effort Wayback Machine fallback exists as an
# explicit opt-in - see WAYBACK_SOURCES / ceia/wayback.py - not silently
# substituted here, since its coverage is never guaranteed.
DISABLED_SOURCES = {
    "business_standard": "Akamai edge returns 403 for all requests, including "
                         "robots.txt; permission to crawl live cannot be "
                         "established. Opt into the best-effort Wayback "
                         "Machine fallback instead via --sources business_standard.",
}

# origin, for building each Wayback source's /topic/<slug> discovery URL.
_WAYBACK_ORIGINS = {s.key: s.origin for s in ALL_SOURCES if s.key in WAYBACK_SOURCES}


@dataclass
class IngestResult:
    config: RunConfig
    items: list[NewsItem] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    disabled_sources: dict[str, str] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    per_source: dict[str, int] = field(default_factory=dict)
    # The aliases actually used for prefiltering/relevance this run - may
    # include one extra widened alias beyond config.all_aliases (see
    # widen_aliases()). Reported separately so a report can always say
    # exactly what was searched for, not just what was configured.
    aliases_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company": self.config.company,
            "ticker": self.config.ticker,
            "benchmark": self.config.benchmark,
            "start": self.config.start.isoformat(),
            "end": self.config.end.isoformat(),
            "aliases": self.aliases_used or self.config.all_aliases,
            "source_status": self.source_status,
            "disabled_sources": self.disabled_sources,
            "stats": self.stats,
            "per_source": self.per_source,
            "items": [i.to_dict() for i in self.items],
        }


def _slug_token_groups(aliases: list[str], ticker: str | None) -> list[frozenset[str]]:
    """Slug-matching groups: a candidate matches a group if enough of its
    words are present in the URL slug.

    One group per alias/ticker, not one flat bag of independently-OR-matched
    words. The flat-bag version matched a candidate on *any* single word from
    *any* alias, so a company whose name shares a first word with conglomerate
    siblings - "Tata" (Motors/Steel/Power/Consumer/...), "Adani"
    (Enterprises/Green/Ports/...), "Reliance", "Bajaj" - had every sibling
    company's articles pass the filter too. Verified on a real probe: for
    "Tata Consumer Products", 58 of 99 one-month prefilter matches from a
    single source turned out to be Tata Steel, Tata Motors, TCS and other
    unrelated Tata Group companies, none of which mention "consumer" or
    "products" anywhere - they only matched because "tata" alone was a valid
    token. That wastes most of a capped fetch budget on the wrong company
    before relevance scoring ever sees the candidates.

    A multi-word alias now requires at least two of its words to co-occur in
    the slug (not all of them - real slugs often drop a word, e.g. "products"
    from "tata-consumer-share-price"), which still rules out a bare "tata"
    while staying tolerant of which two words a headline happened to keep.
    Single-word aliases and the ticker symbol are unaffected: one match still
    suffices, same as before.
    """
    groups: list[frozenset[str]] = []
    for alias in aliases:
        words = {w for w in re.findall(r"[a-z0-9]+", alias.lower())
                if len(w) > 2 and w not in {"ltd", "limited", "inc", "the", "and"}}
        if words:
            groups.append(frozenset(words))
    if ticker:
        symbol = ticker.split(".")[0].lower()
        if len(symbol) > 2:
            groups.append(frozenset({symbol}))
    return groups


# Below this per-source candidate count, prefilter() fetches everything from
# that source instead of guessing from the URL - see prefilter()'s docstring.
PREFILTER_SKIP_THRESHOLD = 5000


def prefilter(candidates: list[Candidate], token_groups: list[frozenset[str]],
             skip_threshold: int = 0) -> list[Candidate]:
    """Keep candidates whose URL slug clears at least one alias group's
    match threshold (see :func:`_slug_token_groups`) - except for a source
    whose *total* candidate count this run is small enough to fetch in full
    (``skip_threshold``, 0 by default so this matches the plain slug filter
    unless a caller opts in), which skips the slug guess for that source
    entirely and lets every one of its candidates through to the real,
    text-based relevance scorer instead.

    This trades a little recall for a large reduction in fetches, but only
    where that trade is actually needed - a low-volume source gains nothing
    from a URL guess that can only ever undercount real coverage, since the
    unlisted/pre-IPO space's dedicated trackers (Entrackr, VCCircle, Inc42)
    return a few thousand candidates over a several-month window (verified
    live) versus Economic Times' 100k+ over the same window, so "fetch
    everything from a source this small" is a bounded, affordable trade the
    other high-volume sources still cannot make. A story on a source too
    large to fetch whole, whose slug never names the company, is still
    missed - that remains an accepted and disclosed cost, recorded in the
    run stats as the pre-filter ratio.
    """
    if not token_groups:
        return candidates

    by_source: dict[str, int] = {}
    for candidate in candidates:
        by_source[candidate.source] = by_source.get(candidate.source, 0) + 1

    kept = []
    for candidate in candidates:
        if by_source[candidate.source] <= skip_threshold:
            kept.append(candidate)
            continue
        slug = candidate.url.lower()
        for group in token_groups:
            threshold = min(2, len(group))
            if sum(1 for token in group if token in slug) >= threshold:
                kept.append(candidate)
                break
    return kept


def interleave(candidates: list[Candidate]) -> list[Candidate]:
    """Round-robin candidates across sources.

    Discovery returns them grouped by source, so a ``--limit`` applied to that
    order would spend the whole budget on whichever source ran first and report
    single-source coverage as if it were the full picture.
    """
    by_source: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.source, []).append(candidate)
    ordered: list[Candidate] = []
    queues = list(by_source.values())
    for index in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if index < len(queue):
                ordered.append(queue[index])
    return ordered


def widen_aliases(config: RunConfig, session=None) -> list[str]:
    """``RunConfig.all_aliases`` plus the company's single leading word -
    "Sonata" from "Sonata Software" - when an independent ticker-search
    check confirms it is safe to trust.

    Short press headlines, appointment/leadership stories especially, often
    drop every word but the first: "Sonata appoints new CEO", not "Sonata
    Software appoints new CEO". Neither the full company name nor its
    suffix-stripped short form (see ``RunConfig.all_aliases`` /
    ``_strip_corporate_suffix``) matches that, so real coverage is missed
    unless the leading word itself is trusted as an alias - but doing that
    unconditionally for every company would reopen the conglomerate-sibling
    bug this project already found and fixed once at the slug-prefilter
    level (Tata Motors/Steel/Power articles all matching a bare "Tata"
    alias meant for Tata Consumer Products; see the README).

    There is no way to tell "Sonata" and "Tata" apart from the string
    alone, so this asks the same question a human fact-checker would: search
    for the bare word via the same ticker-lookup endpoint ``--ticker``
    auto-detection already uses, and only trust it if *this* company is the
    top result. A company that shares its leading word with a well-known
    sibling (Tata, Adani, Bajaj, Reliance, Birla, ...) will not be the top
    result for its own group name, so the leading word is correctly left
    out for it - with no hardcoded list of "risky" names to write or
    maintain, which would always be one new conglomerate behind.

    Best-effort: any lookup failure (offline, rate-limited, endpoint down)
    just skips the extra alias rather than failing the run, the same
    degrade-not-fail rule ticker auto-detection already follows.
    """
    aliases = config.all_aliases
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z&]*", config.company)
             if len(w) > 2 and w.lower() != "the"]
    if not words:
        return aliases
    leading = words[0]
    if leading.lower() in {a.lower() for a in aliases}:
        return aliases
    try:
        match = resolve_ticker(leading, exchange=config.exchange, session=session)
    except TickerLookupError as exc:
        log.info("alias widening: skipped %r (%s)", leading, exc)
        return aliases
    if match.symbol.split(".")[0].upper() != config.ticker.split(".")[0].upper():
        log.info("alias widening: skipped %r (top ticker-search result was %s, "
                 "not this company)", leading, match.symbol)
        return aliases
    log.info("alias widening: added %r (confirmed top ticker-search result)", leading)
    return [*aliases, leading]


def cap_across_range(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Cap the candidate list to ``limit`` while keeping the whole date range
    represented, rather than exhausting the cap on whichever end comes first.

    Each source's own candidates arrive from discovery in roughly chronological
    order (day- or month-partitioned sitemaps), and ``interleave`` only fixes
    bias *across sources*, not across time - index 0 of every source is still
    close to ``start``. Fetching the first ``limit`` candidates off that list
    would silently stop once the budget is spent, however early that happens,
    which reads as "nothing happened after March" in the report rather than
    "the run stopped looking after March". Evenly-spaced index sampling over
    the already-interleaved list keeps every part of the window represented,
    at the cost of a shallower sample within each part.
    """
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[int(i * step)] for i in range(limit)]


def in_range(items: list[NewsItem], start: date, end: date) -> list[NewsItem]:
    """Keep items actually published inside the requested window.

    Sitemap date hints are approximate (``lastmod`` trails publication), so
    discovery deliberately over-collects at the edges. This trims back to what
    the user asked for, using the authoritative timestamp from the article.
    Items with no usable timestamp are kept and stay flagged, since dropping
    them silently would hide a gap rather than report it.
    """
    kept = []
    for item in items:
        if item.published_at is None:
            kept.append(item)
            continue
        if start <= item.published_at.astimezone(IST).date() <= end:
            kept.append(item)
    return kept


def fetch_and_parse(fetcher: Fetcher, candidates: list[Candidate],
                    limit: int | None = None,
                    max_workers: int = 8) -> tuple[list[NewsItem], dict[str, int]]:
    """Fetch and parse every candidate, ``max_workers`` at a time.

    Candidates arrive interleaved across sources (see ``interleave()``), so
    consecutive entries are usually different origins - a thread pool here
    gets several origins' rate-limited fetches running at once instead of
    one after another, without fetching any single origin any faster than
    ``Fetcher`` already allows (its per-origin lock serialises same-origin
    requests regardless of thread count). All mutable state (``items``,
    ``errors``, ``seen``) is shared across worker threads and updated inside
    one lock; contention there is negligible next to the network I/O this
    spends its time on.
    """
    items: list[NewsItem] = []
    errors = {"robots": 0, "http": 0, "parse": 0, "empty": 0}
    seen: set[str] = set()
    total = len(candidates)
    state_lock = threading.Lock()
    stop = threading.Event()
    completed = 0

    def process_one(candidate: Candidate) -> None:
        nonlocal completed
        if stop.is_set():
            return
        with state_lock:
            if candidate.url in seen:
                return
            seen.add(candidate.url)

        error_kind: str | None = None
        item: NewsItem | None = None
        try:
            response = fetcher.get(candidate.url)
        except RobotsDisallowed:
            error_kind = "robots"
        except Exception as exc:
            log.debug("fetch failed %s: %s", candidate.url, exc)
            error_kind = "http"
        else:
            if response.status != 200:
                error_kind = "http"
            else:
                try:
                    parsed = parse_article(response.text, candidate.url, candidate.source)
                except Exception as exc:
                    log.debug("parse failed %s: %s", candidate.url, exc)
                    error_kind = "parse"
                else:
                    if not parsed["headline"]:
                        error_kind = "empty"
                    else:
                        item = NewsItem(**parsed, fetched_at=response.fetched_at,
                                        content_sha256=response.sha256)

        with state_lock:
            completed += 1
            n = completed
            if error_kind is not None:
                errors[error_kind] += 1
            elif item is not None:
                if limit is not None and len(items) >= limit:
                    stop.set()
                else:
                    items.append(item)
            # Each fetch is rate-limited (>=2s/origin), so hundreds of
            # candidates can take minutes even in parallel; without this a
            # long stretch of no items kept (paywalls, off-topic slugs) looks
            # identical to the process hanging.
            if n % 20 == 0 or n == total:
                log.info("fetched %d/%d candidates, %d parsed OK, errors=%s",
                         n, total, len(items), errors)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        list(executor.map(process_one, candidates))

    return items, errors


def run(config: RunConfig, fetcher: Fetcher | None = None,
        scorer: FinBertScorer | None = None,
        emotion_scorer: GoEmotionScorer | None = None,
        limit: int | None = None,
        skip_sentiment: bool = False,
        skip_emotion: bool = False,
        skip_alias_widening: bool = False,
        skip_slug_prefilter: bool = False,
        max_workers: int = 8,
        wayback_fetcher: Fetcher | None = None) -> IngestResult:
    fetcher = fetcher or Fetcher()
    requested = config.sources or DEFAULT_SOURCES
    # business_standard/livemint are opt-in only (see WAYBACK_SOURCES /
    # ceia/wayback.py): never picked up by DEFAULT_SOURCES, and routed
    # through the Wayback fallback below rather than the normal,
    # STRATEGIES-driven discover() call, which neither source's situation fits.
    wayback_keys = [s for s in requested if s in WAYBACK_SOURCES]
    sources = [s for s in requested
              if s not in DISABLED_SOURCES and s not in WAYBACK_SOURCES]
    result = IngestResult(config=config, disabled_sources=dict(DISABLED_SOURCES))

    aliases = config.all_aliases if skip_alias_widening else widen_aliases(config)
    result.aliases_used = aliases

    candidates, status = discover(fetcher, sources, config.start, config.end)

    # Best-effort Wayback Machine discovery (and, for business_standard,
    # fetching too - its live site is fully blocked). See ceia/wayback.py's
    # module docstring: coverage is never guaranteed, so what was actually
    # found is reported in source_status exactly like any other source,
    # never silently absorbed as "no coverage".
    wayback_parsed_items: list[NewsItem] = []
    if wayback_keys:
        wb_fetcher = wayback_fetcher or wayback_mod.wayback_fetcher(
            cache_dir=str(fetcher.cache_dir))
        window_mid = config.start + (config.end - config.start) // 2
        for key in wayback_keys:
            origin = _WAYBACK_ORIGINS.get(key)
            if origin is None:
                status[key] = "unknown wayback source"
                continue
            wb_candidates, wb_status = wayback_mod.discover_topic_candidates(
                wb_fetcher, key, origin, aliases, config.start, config.end)
            if key == "business_standard":
                wb_items, wb_errors = wayback_mod.fetch_and_parse_via_wayback(
                    wb_fetcher, wb_candidates, window_mid)
                wayback_parsed_items.extend(wb_items)
                status[key] = f"{wb_status}; {len(wb_items)} article(s) fetched OK"
                log.info("wayback (%s): %s (errors=%s)", key, status[key], wb_errors)
            else:
                candidates.extend(wb_candidates)
                status[key] = wb_status
                log.info("wayback (%s): %s", key, status[key])

    result.source_status = status
    log.info("discovered %d candidate URLs", len(candidates))

    token_groups = _slug_token_groups(aliases, config.ticker)
    if skip_slug_prefilter:
        # Every candidate goes to the real, text-based relevance scorer -
        # no URL-slug guess anywhere, regardless of a source's volume this
        # run. Slower and far more requests on a high-volume source (a
        # month of Economic Times alone is ~13,000 URLs), but it is the
        # only way to catch a story whose slug never names the company at
        # all - the accepted, disclosed cost prefilter()'s docstring already
        # names for a source too large to fetch whole.
        narrowed = interleave(candidates)
        log.info("slug pre-filter skipped entirely (--skip-slug-prefilter): "
                 "%d candidate(s) all going to full-text relevance scoring",
                 len(narrowed))
    else:
        narrowed = interleave(prefilter(candidates, token_groups, PREFILTER_SKIP_THRESHOLD))
        log.info("pre-filtered to %d URLs on slug groups %s (sources under %d "
                 "candidates fetched in full, not slug-guessed)",
                 len(narrowed), [sorted(g) for g in token_groups], PREFILTER_SKIP_THRESHOLD)

    if limit is not None and len(narrowed) > limit:
        narrowed = cap_across_range(narrowed, limit)
        log.info("capped to %d candidates, spread across the full date range "
                 "rather than just its earliest days", len(narrowed))

    parsed_items, errors = fetch_and_parse(fetcher, narrowed, limit=limit,
                                           max_workers=max_workers)
    parsed_items = parsed_items + wayback_parsed_items
    items = in_range(parsed_items, config.start, config.end)
    log.info("parsed %d articles, %d inside the requested window",
             len(parsed_items), len(items))

    kept, dropped = relevance.apply(items, aliases, config.ticker,
                                    config.min_relevance)
    log.info("relevance kept %d, dropped %d", len(kept), len(dropped))

    dedupe.deduplicate(kept)
    unique_items = dedupe.unique(kept)
    log.info("%d unique after dedupe", len(unique_items))

    align.attribute_all(kept)

    if not skip_sentiment and unique_items:
        (scorer or FinBertScorer()).score_items(unique_items)

    # Emotion runs independently of sentiment: a user who wants FinBERT's
    # finance-tuned score but not a second ~500MB model download should be
    # able to skip this one without giving up the one that drives ranking.
    if not skip_emotion and unique_items:
        (emotion_scorer or GoEmotionScorer()).score_items(unique_items)

    per_source: dict[str, int] = {}
    for item in kept:
        if item.duplicate_of is None:
            per_source[item.source] = per_source.get(item.source, 0) + 1
    result.items = kept
    result.per_source = per_source
    result.stats = {
        "candidates_discovered": len(candidates),
        "candidates_after_prefilter": len(narrowed),
        "articles_parsed": len(parsed_items),
        "in_requested_window": len(items),
        "relevant": len(kept),
        "dropped_by_relevance": len(dropped),
        "unique_after_dedupe": len(unique_items),
        "duplicates": len(kept) - len(unique_items),
        "paywalled": sum(1 for i in kept if i.paywalled),
        "missing_timestamp": sum(1 for i in kept if i.timestamp_confidence == "missing"),
        "after_close": sum(1 for i in kept if i.after_close),
        "unattributed": len(align.unattributed(kept)),
        **{f"fetch_error_{k}": v for k, v in errors.items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: news ingestion and sentiment")
    parser.add_argument("--company", required=True)
    parser.add_argument("--ticker", default=None,
                        help="e.g. ADANIENT.NS. Auto-detected from --company if omitted.")
    parser.add_argument("--exchange", default="NSE",
                        help="Preferred exchange for ticker auto-detection (NSE or BSE).")
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--alias", action="append", default=[],
                        help="Repeatable. Short names, product names, misspellings.")
    parser.add_argument("--sources", nargs="*",
                        default=DEFAULT_SOURCES + WAYBACK_SOURCES,
                        help="Defaults to every runnable source in one "
                             "command: the 5 always-on sources plus "
                             "business_standard and livemint's best-effort "
                             "Wayback Machine fallback, which can sometimes "
                             "find nothing for a given window - check "
                             "source_status for what each source actually "
                             "found. Pass an explicit, narrower list to opt "
                             "out of any of them.")
    parser.add_argument("--min-relevance", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap articles fetched, evenly spread across the "
                             "whole date range rather than just its earliest "
                             "days; useful for a quick trial run.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent article fetches (default 8). Discovery "
                             "across the 4 sources always runs concurrently, "
                             "capped at one worker per source. Requests to any "
                             "single origin are still serialised at --min-interval "
                             "regardless of --workers - this only lets DIFFERENT "
                             "origins' rate-limited fetches overlap instead of "
                             "queueing behind each other.")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip FinBERT (no model download).")
    parser.add_argument("--skip-emotion", action="store_true",
                        help="Skip GoEmotions (no model download). FinBERT "
                             "sentiment, which drives incident ranking, is "
                             "unaffected either way.")
    parser.add_argument("--skip-alias-widening", action="store_true",
                        help="Don't try the company's leading word as an "
                             "extra alias (see widen_aliases()). On by "
                             "default; costs one extra ticker-search "
                             "request per run.")
    parser.add_argument("--skip-slug-prefilter", action="store_true",
                        help="Never use the URL-slug guess to narrow "
                             "candidates before fetching, regardless of a "
                             "source's volume this run - every candidate "
                             "from every source goes straight to full-text "
                             "relevance scoring instead. Catches a story "
                             "whose slug never names the company at all, at "
                             "the cost of far more fetches on a high-volume "
                             "source (a month of Economic Times alone is "
                             "~13,000 URLs). Off by default; the slug guess "
                             "already skips itself per-source below "
                             "PREFILTER_SKIP_THRESHOLD candidates.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--out", default="out/news.json")
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
            company=args.company,
            ticker=ticker,
            benchmark=args.benchmark,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            aliases=args.alias,
            sources=args.sources,
            min_relevance=args.min_relevance,
        )
    except ValueError as exc:
        print(f"\n{exc}")
        raise SystemExit(2)
    fetcher = Fetcher(cache_dir=args.cache_dir, user_agent=args.user_agent,
                      min_interval=args.min_interval)
    result = run(config, fetcher=fetcher, limit=args.limit,
                 skip_sentiment=args.skip_sentiment,
                 skip_emotion=args.skip_emotion,
                 skip_alias_widening=args.skip_alias_widening,
                 skip_slug_prefilter=args.skip_slug_prefilter,
                 max_workers=args.workers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str),
                        encoding="utf-8")

    print(f"\n{'=' * 70}\nPHASE 1 INGESTION - {config.company} ({config.ticker})")
    print(f"{config.start} to {config.end}\n{'=' * 70}")
    print("\nSources:")
    for key, state in result.source_status.items():
        print(f"  {key:20s} {state}")
    for key, why in result.disabled_sources.items():
        print(f"  {key:20s} DISABLED - {why}")
    print("\nUnique items per source:")
    for key in sorted(result.per_source):
        print(f"  {key:20s} {result.per_source[key]}")
    print("\nCounts:")
    for key, value in result.stats.items():
        print(f"  {key:32s} {value}")

    kept = [i for i in result.items if i.duplicate_of is None]
    if kept:
        print(f"\nTop items by relevance:")
        for item in sorted(kept, key=lambda i: -i.relevance_score)[:10]:
            when = item.published_at.strftime("%Y-%m-%d %H:%M") if item.published_at else "?"
            flag = " [after close]" if item.after_close else ""
            emotion = f", {item.emotion_label}" if item.emotion_label else ""
            print(f"  {item.relevance_score:.2f} {when}{flag} [{item.source}] "
                  f"{item.sentiment_label or '-'}/{item.event_category or '-'}{emotion}")
            print(f"        {item.headline[:96]}")
    print(f"\nwrote {out_path}")
    print("\nNOTE: coverage and price moves that coincide are not evidence of "
          "causation, and one company over one date range is too small a sample "
          "for statistical significance. See README > Limitations.")


if __name__ == "__main__":
    main()
