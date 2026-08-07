"""Wayback Machine fallback for Business Standard and Mint (professor's
request, re-investigated - see docs/ or the README's Phase 8 section).

Two different problems, two different fixes that happen to share the same
underlying mechanism:

* **Business Standard** is fully blocked live - its Akamai edge returns 403
  for every request, including ``robots.txt`` itself, verified again (with
  both this tool's own honest user agent and a full browser one, both
  network-level 403s from ``AkamaiGHost``) before building this. Both
  *discovery* (finding what was published) and *fetching* (reading an
  article) go through an archived snapshot here.
* **Mint** was never blocked - its ``robots.txt`` is fully permissive and
  names no AI agent. Its problem is depth: the only sitemaps it advertises
  cover the last two days, so a past date range (a case study like January
  2023) has nothing to discover from. Only *discovery* goes through an
  archived snapshot (a topic page's archived link list); each article is
  then fetched live from mint's own site, which is faster and gets current
  formatting rather than Wayback's URL-rewritten copy.

**This is best-effort, not a guaranteed backfill.** archive.org's crawl
frequency tracks a page's real-world traffic: the Adani/Hindenburg story
(globally covered) had a same-day archived snapshot of both sites' topic
pages, verified directly. Two unrelated spot-checks (Paytm, Zomato) to make
sure that wasn't a one-off found "closest" snapshots anywhere from days to
15+ months off-target, and one with nothing archived at all. Every run
discloses exactly what it found - a snapshot timestamp and how many
candidate URLs it yielded, or "nothing found" - in ``source_status``,
the same place every other source's outcome is already reported, rather
than silently returning an empty list that reads as "no coverage".

**On robots.txt**: archive.org's own robots.txt (fetched, real, checked) is
permissive - ``User-agent: *`` disallows only ``/control/`` and
``/report/``, no AI-agent block of any kind. The actual playback host used
here, ``web.archive.org``, has no ``robots.txt`` file at all (a real 404,
verified). Under the standard interpretation - a missing robots.txt imposes
no restriction - that is not a wall, unlike every *403*-on-robots.txt case
this project has hit (Business Standard itself, RBI's own site), where
permission genuinely cannot be established. This project's own
:class:`ceia.robots.RobotsPolicy` is intentionally more conservative than
that standard (any non-2xx fails closed), a good default for a commercial
publisher's edge returning errors as part of blocking bots - but applying
that same conservatism to a non-profit library's missing robots.txt file
would misread an absent policy as a wall it never declared. Fetched through
a dedicated :class:`ceia.fetcher.Fetcher` with ``obey_robots=False`,
constructed only here and never used for any publisher's own origin, so
this decision stays local and visible rather than loosening the shared
default every other source is still held to.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from datetime import date, datetime, timedelta

from .discovery import Candidate
from .extract import parse_article
from .fetcher import DEFAULT_USER_AGENT, Fetcher
from .models import NewsItem

log = logging.getLogger(__name__)

_AVAILABILITY_URL = "https://archive.org/wayback/available"

# How far a topic-page snapshot's own capture date may sit from the window
# before it is discarded as telling us nothing about this window's coverage
# - Wayback only ever returns its *closest* capture, which can be calendar-
# months away for a lightly-trafficked page (see module docstring).
_SNAPSHOT_MARGIN_DAYS = 30

# Real article URLs, distinguished from a topic page's nav/section/author
# links by shape - verified against live archived pages for both sites.
_ARTICLE_LINK_PATTERNS: dict[str, re.Pattern] = {
    "business_standard": re.compile(
        r"https://www\.business-standard\.com/article/[a-z0-9/_-]+-\d{9,}_\d\.html"),
    "livemint": re.compile(
        r"https://www\.livemint\.com/[a-z0-9/_-]+-\d{5,}\.html"),
}


class SkippedWaybackFetcher:
    """Degrades immediately, no network touched - what a caller opting out
    of the Wayback fallback should inject, and what every test that reaches
    ``discover_topic_candidates``/``fetch_and_parse_via_wayback`` should
    inject unless it is specifically exercising this module, for the same
    reason ``ceia.macro.SkippedFetcher`` exists: the real default retries a
    failure with backoff, which is slow."""

    def get(self, url: str):
        raise RuntimeError("skipped (no wayback fetcher configured)")


def wayback_fetcher(cache_dir: str = "cache", user_agent: str = DEFAULT_USER_AGENT,
                    min_interval: float = 2.0) -> Fetcher:
    """A dedicated Fetcher for archive.org - see the module docstring's
    robots.txt section for why ``obey_robots=False`` is correct here and
    nowhere else in this project."""
    return Fetcher(cache_dir=cache_dir, user_agent=user_agent,
                   min_interval=min_interval, obey_robots=False)


def closest_snapshot(fetcher: Fetcher, url: str, near: date) -> tuple[str, date] | None:
    """The archived snapshot of ``url`` closest to ``near``, as
    ``(playback_url, actual_capture_date)``, or ``None`` if archive.org has
    never captured it (or the availability check itself failed)."""
    query = urllib.parse.urlencode({"url": url, "timestamp": near.strftime("%Y%m%d")})
    try:
        response = fetcher.get(f"{_AVAILABILITY_URL}?{query}")
    except Exception as exc:
        log.warning("wayback availability check failed for %s: %s", url, exc)
        return None
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError:
        return None
    closest = (payload.get("archived_snapshots") or {}).get("closest")
    if not closest or not closest.get("available"):
        return None
    timestamp = closest.get("timestamp", "")
    try:
        snap_date = datetime.strptime(timestamp[:8], "%Y%m%d").date()
    except ValueError:
        snap_date = near
    return f"https://web.archive.org/web/{timestamp}/{url}", snap_date


def _topic_slugs(names: list[str]) -> list[str]:
    """Lowercased, hyphenated slugs from company names/aliases - the same
    rough shape both sites use for their own topic pages (verified against
    real ones: 'adani-group', 'adani-enterprises', 'paytm', 'zomato')."""
    slugs, seen = [], set()
    for name in names:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if slug and slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs


def _probe_dates(start: date, end: date, every_days: int) -> list[date]:
    """A handful of evenly-spaced dates across the window, always including
    both ends - one archive.org lookup per probe, so this stays bounded on a
    wide date range rather than scanning every single day."""
    if start == end:
        return [start]
    probes = [start]
    current = start + timedelta(days=every_days)
    while current < end:
        probes.append(current)
        current += timedelta(days=every_days)
    probes.append(end)
    return probes


def _extract_article_links(html: str, source_key: str) -> list[str]:
    """Canonical (un-prefixed) article URLs found in an archived topic
    page's HTML - Wayback rewrites every href to include its own
    ``/web/<timestamp>/`` wrapper, which this strips back off."""
    pattern = _ARTICLE_LINK_PATTERNS[source_key]
    found: set[str] = set()
    for match in re.finditer(r"/web/\d+/(" + pattern.pattern + r")", html):
        found.add(match.group(1))
    for match in pattern.finditer(html):
        found.add(match.group(0))
    return sorted(found)


def discover_topic_candidates(
    fetcher: Fetcher,
    source_key: str,
    origin: str,
    names: list[str],
    start: date,
    end: date,
    probe_every_days: int = 7,
) -> tuple[list[Candidate], str]:
    """Best-effort discovery via archived topic-page snapshots near a
    handful of probe dates across ``[start, end]``.

    Returns ``(candidates, status)`` - ``status`` is a human-readable summary
    of what was actually found, for the same ``source_status`` reporting
    every other source already uses, since Wayback coverage is never
    guaranteed (see module docstring).
    """
    slugs = _topic_slugs(names)
    probes = _probe_dates(start, end, probe_every_days)
    seen_urls: set[str] = set()
    candidates: list[Candidate] = []
    snapshots_found = 0
    for slug in slugs:
        topic_url = f"{origin}/topic/{slug}"
        for probe in probes:
            snap = closest_snapshot(fetcher, topic_url, probe)
            if snap is None:
                continue
            playback_url, snap_date = snap
            if not (start - timedelta(days=_SNAPSHOT_MARGIN_DAYS) <= snap_date
                    <= end + timedelta(days=_SNAPSHOT_MARGIN_DAYS)):
                continue
            try:
                response = fetcher.get(playback_url)
            except Exception as exc:
                log.warning("wayback fetch failed for %s: %s", playback_url, exc)
                continue
            if response.status != 200:
                continue
            snapshots_found += 1
            for article_url in _extract_article_links(response.text, source_key):
                if article_url in seen_urls:
                    continue
                seen_urls.add(article_url)
                candidates.append(Candidate(article_url, source_key, None))
    if snapshots_found:
        status = (f"wayback: {snapshots_found} archived topic-page snapshot(s) "
                  f"found, {len(candidates)} candidate URL(s)")
    else:
        status = "wayback: no archived topic-page snapshot found near this window"
    return candidates, status


def fetch_and_parse_via_wayback(
    fetcher: Fetcher, candidates: list[Candidate], near: date,
) -> tuple[list[NewsItem], dict[str, int]]:
    """Business Standard's live site is fully blocked (see module docstring),
    so unlike every other source, its candidate articles are fetched through
    their own archived snapshot too, rather than live.

    ``NewsItem.url`` is always set to the real, canonical business-standard.com
    URL - the Wayback playback URL is only this tool's own retrieval path, not
    the article's identity, so a reader following the link lands on the real
    source, same as every other source in this project.
    """
    items: list[NewsItem] = []
    errors = {"unavailable": 0, "http": 0, "parse": 0, "empty": 0}
    for candidate in candidates:
        snap = closest_snapshot(fetcher, candidate.url, candidate.hint_date or near)
        if snap is None:
            errors["unavailable"] += 1
            continue
        playback_url, _ = snap
        try:
            response = fetcher.get(playback_url)
        except Exception as exc:
            log.debug("wayback article fetch failed %s: %s", playback_url, exc)
            errors["http"] += 1
            continue
        if response.status != 200:
            errors["http"] += 1
            continue
        try:
            parsed = parse_article(response.text, candidate.url, candidate.source)
        except Exception as exc:
            log.debug("parse failed %s: %s", candidate.url, exc)
            errors["parse"] += 1
            continue
        if not parsed["headline"]:
            errors["empty"] += 1
            continue
        items.append(NewsItem(**parsed, fetched_at=response.fetched_at,
                              content_sha256=response.sha256))
    return items, errors
