"""Turn a date range into candidate article URLs, per source.

Every source here is driven through its ``robots.txt``-declared sitemaps rather
than a search page, because Financial Express, Business Line and Moneycontrol
all disallow their search paths for every user agent (Phase 0). The sitemaps
are partitioned by date anyway, which is the axis this tool actually needs.

Each strategy yields :class:`Candidate` records carrying whatever date the
sitemap asserts. That date is a *hint* for narrowing the fetch set only — the
authoritative publish time always comes from the article page itself, since
sitemap ``lastmod`` is a modification time and can sit hours after publication.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .fetcher import Fetcher, RobotsDisallowed

log = logging.getLogger(__name__)

_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)
_URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.S | re.I)
_LASTMOD_RE = re.compile(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", re.I)

# Excel-style serial used by the Economic Times archive: days since 1899-12-30.
_ET_EPOCH = date(1899, 12, 30)


@dataclass
class Candidate:
    url: str
    source: str
    hint_date: date | None = None


def _unescape(url: str) -> str:
    return (url.replace("&amp;", "&").replace("&lt;", "<")
               .replace("&gt;", ">").replace("&quot;", '"'))


def _fetch_xml(fetcher: Fetcher, url: str) -> str | None:
    try:
        response = fetcher.get(url)
    except RobotsDisallowed as exc:
        log.warning("skipping (robots): %s", exc)
        return None
    except Exception as exc:
        log.warning("could not fetch %s: %s", url, exc)
        return None
    if response.status != 200:
        log.warning("%s returned %s", url, response.status)
        return None
    return response.text


def _locs(xml: str) -> list[str]:
    return [_unescape(u) for u in _LOC_RE.findall(xml)]


def _loc_lastmod_pairs(xml: str) -> list[tuple[str, date | None]]:
    """Pull (url, lastmod) pairs so a month sitemap can be filtered by day."""
    pairs = []
    for block in _URL_BLOCK_RE.findall(xml):
        loc = _LOC_RE.search(block)
        if not loc:
            continue
        stamp = _LASTMOD_RE.search(block)
        when = None
        if stamp:
            try:
                when = datetime.fromisoformat(stamp.group(1).replace("Z", "+00:00")).date()
            except ValueError:
                when = None
        pairs.append((_unescape(loc.group(1)), when))
    return pairs or [(u, None) for u in _locs(xml)]


def _days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _months(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current.year, current.month
        current = date(current.year + (current.month == 12),
                       current.month % 12 + 1, 1)


# --------------------------------------------------------------- strategies

def economic_times(fetcher: Fetcher, start: date, end: date) -> list[Candidate]:
    """Month sitemaps, split into parts capped at 10k URLs each."""
    index = _fetch_xml(
        fetcher,
        "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-index.xml",
    )
    if not index:
        return []
    wanted = {f"{date(y, m, 1):%Y-%B}" for y, m in _months(start, end)}
    out: list[Candidate] = []
    for part in _locs(index):
        stem = re.search(r"/(\d{4}-[A-Za-z]+)-\d+\.xml$", part)
        if not stem or stem.group(1) not in wanted:
            continue
        xml = _fetch_xml(fetcher, part)
        if not xml:
            continue
        for url, when in _loc_lastmod_pairs(xml):
            # lastmod can trail publication into the next day, so keep a
            # one-day margin and let the article page settle the real date.
            if when and not (start - timedelta(days=1) <= when <= end + timedelta(days=1)):
                continue
            out.append(Candidate(url, "economic_times", when))
    return out


def financial_express(fetcher: Fetcher, start: date, end: date) -> list[Candidate]:
    """Day sitemaps. Dated URLs resolve well beyond the index's ~92-day window."""
    out: list[Candidate] = []
    days = list(_days(start, end))
    log.info("financial_express: scanning %d day(s) of sitemaps", len(days))
    for i, day in enumerate(days, 1):
        xml = _fetch_xml(
            fetcher,
            f"https://www.financialexpress.com/sitemap.xml"
            f"?yyyy={day.year}&mm={day.month:02d}&dd={day.day:02d}",
        )
        if not xml:
            continue
        out.extend(Candidate(url, "financial_express", day) for url in _locs(xml))
        # One request per day, each spaced by the rate limit, so a wide window
        # can run minutes with no other output - a line per day makes that
        # visible progress rather than an apparent hang.
        if i % 5 == 0 or i == len(days):
            log.info("financial_express: %d/%d days done, %d URLs so far", i, len(days), len(out))
    return out


def business_line(fetcher: Fetcher, start: date, end: date) -> list[Candidate]:
    """Day sitemaps at /sitemap/archive/all/YYYYMMDD_N.xml, back to Dec 2010."""
    out: list[Candidate] = []
    days = list(_days(start, end))
    log.info("business_line: scanning %d day(s) of sitemaps", len(days))
    for i, day in enumerate(days, 1):
        # Days occasionally spill into a second part; stop at the first gap.
        for part in range(1, 4):
            xml = _fetch_xml(
                fetcher,
                f"https://www.thehindubusinessline.com/sitemap/archive/all/"
                f"{day:%Y%m%d}_{part}.xml",
            )
            if not xml:
                break
            urls = _locs(xml)
            if not urls:
                break
            out.extend(Candidate(url, "business_line", day) for url in urls)
        if i % 5 == 0 or i == len(days):
            log.info("business_line: %d/%d days done, %d URLs so far", i, len(days), len(out))
    return out


def moneycontrol(fetcher: Fetcher, start: date, end: date) -> list[Candidate]:
    """Year index -> month sitemaps (sitemap-post-YYYY-MM.xml)."""
    out: list[Candidate] = []
    wanted = {f"{y:04d}-{m:02d}" for y, m in _months(start, end)}
    for year in range(start.year, end.year + 1):
        index = _fetch_xml(
            fetcher, f"https://www.moneycontrol.com/news/index-sitemap-{year}.xml"
        )
        if not index:
            continue
        for month_url in _locs(index):
            stem = re.search(r"sitemap-post-(\d{4}-\d{2})\.xml$", month_url)
            if not stem or stem.group(1) not in wanted:
                continue
            xml = _fetch_xml(fetcher, month_url)
            if not xml:
                continue
            for url, when in _loc_lastmod_pairs(xml):
                if when and not (start - timedelta(days=1) <= when <= end + timedelta(days=1)):
                    continue
                out.append(Candidate(url, "moneycontrol", when))
    return out


STRATEGIES = {
    "economic_times": economic_times,
    "financial_express": financial_express,
    "business_line": business_line,
    "moneycontrol": moneycontrol,
}


def discover(
    fetcher: Fetcher,
    sources: list[str],
    start: date,
    end: date,
    max_workers: int = 4,
) -> tuple[list[Candidate], dict[str, str]]:
    """Collect candidates across sources, one worker thread per source.

    Returns the candidates plus a per-source status map. A source that blocks
    us or changes layout degrades to an error string instead of killing the
    run, and the report states which sources were unavailable (Section 10).

    Running sources concurrently is safe and does not make the crawl any less
    polite: each source is a different origin, and ``Fetcher`` serialises
    requests *within* an origin via its own lock regardless of how many
    threads call it (see ``fetcher.py``). What changes is wall-clock time -
    financial_express and business_line fetch one sitemap per day and were
    the dominant cost on a wide date range (a full year took ~35 minutes
    combined, sequentially); run concurrently with the two fast month-based
    sources, total discovery time drops toward whichever single source is
    slowest, not the sum of all four.
    """
    runnable = [key for key in sources if key in STRATEGIES]
    status: dict[str, str] = {
        key: "unknown source" for key in sources if key not in STRATEGIES
    }
    candidates: list[Candidate] = []
    if not runnable:
        return candidates, status

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(runnable)))) as executor:
        future_to_key = {
            executor.submit(STRATEGIES[key], fetcher, start, end): key
            for key in runnable
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                found = future.result()
            except Exception as exc:
                status[key] = f"failed: {type(exc).__name__}: {exc}"
                log.exception("discovery failed for %s", key)
                continue
            candidates.extend(found)
            status[key] = f"ok: {len(found)} candidate URLs" if found else "no URLs returned"
    return candidates, status
