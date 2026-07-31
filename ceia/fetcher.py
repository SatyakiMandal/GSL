"""Polite, cached, provenance-logging HTTP client.

Every outbound request in this project goes through :class:`Fetcher`, which
enforces four things the PRD asks for in Sections 10 and 12:

* robots.txt is consulted before the first fetch to an origin and honoured on
  every fetch after that (fail-closed if robots.txt itself is unreadable);
* requests to one origin are spaced by at least ``min_interval`` seconds, and
  by the site's own ``Crawl-delay`` when it declares a longer one;
* responses are cached on disk, so re-running an analysis over the same
  company/date range does not re-scrape anything;
* every fetch is appended to a JSONL provenance log with a content hash, so a
  number in a report can be traced back to the exact bytes it came from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from .robots import RobotsPolicy

log = logging.getLogger(__name__)

# Identifies the tool honestly rather than posing as a browser. Sites that want
# to allow or refuse a research crawler can then act on it.
DEFAULT_USER_AGENT = (
    "CompanyEventImpactAnalyzer/0.1 (academic research; "
    "+https://github.com/SatyakiMandal/GSL)"
)


class RobotsDisallowed(RuntimeError):
    """Raised when robots.txt forbids a URL (or could not be read)."""


@dataclass
class Response:
    url: str
    final_url: str
    status: int
    text: str
    from_cache: bool
    fetched_at: str
    sha256: str


class Fetcher:
    def __init__(
        self,
        cache_dir: Path | str = "cache",
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval: float = 2.0,
        timeout: float = 40.0,
        max_retries: int = 3,
        obey_robots: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.http_cache = self.cache_dir / "http"
        self.http_cache.mkdir(parents=True, exist_ok=True)
        self.provenance_path = self.cache_dir / "provenance.jsonl"
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.obey_robots = obey_robots

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9",
            }
        )
        self._robots: dict[str, RobotsPolicy] = {}
        self._last_request: dict[str, float] = {}

    # ---------------------------------------------------------------- robots

    @staticmethod
    def _origin(url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"

    def robots_for(self, url: str) -> RobotsPolicy:
        origin = self._origin(url)
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            self._wait(origin)
            try:
                resp = self._session.get(robots_url, timeout=self.timeout)
                text, status = resp.text, resp.status_code
            except requests.RequestException as exc:
                log.warning("could not fetch %s: %s", robots_url, exc)
                text, status = None, None
            self._last_request[origin] = time.monotonic()
            self._robots[origin] = RobotsPolicy(origin, text, status)
            self._log_provenance(
                url=robots_url,
                final_url=robots_url,
                status=status if status is not None else -1,
                body=text or "",
                from_cache=False,
                note="robots.txt",
            )
        return self._robots[origin]

    def check(self, url: str) -> tuple[bool, str]:
        """Would this URL be fetched? Returns ``(allowed, reason)``."""
        if not self.obey_robots:
            return True, "robots checking disabled"
        return self.robots_for(url).allows(url, self.user_agent)

    # ----------------------------------------------------------------- cache

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        host = urllib.parse.urlsplit(url).netloc.replace(":", "_")
        bucket = self.http_cache / host
        bucket.mkdir(parents=True, exist_ok=True)
        return bucket / f"{digest}.json"

    def _wait(self, origin: str) -> None:
        interval = self.min_interval
        policy = self._robots.get(origin)
        if policy is not None:
            declared = policy.crawl_delay(self.user_agent)
            if declared:
                interval = max(interval, declared)
        last = self._last_request.get(origin)
        if last is not None:
            remaining = interval - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)

    # ----------------------------------------------------------- provenance

    def _log_provenance(
        self,
        url: str,
        final_url: str,
        status: int,
        body: str,
        from_cache: bool,
        note: str = "",
    ) -> str:
        digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()
        record = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "final_url": final_url,
            "status": status,
            "bytes": len(body),
            "sha256": digest,
            "from_cache": from_cache,
            "user_agent": self.user_agent,
            "note": note,
        }
        with self.provenance_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return digest

    # ------------------------------------------------------------------ get

    def get(self, url: str, force: bool = False) -> Response:
        """Fetch ``url``, honouring robots.txt, cache, and rate limits."""
        cache_path = self._cache_path(url)
        if cache_path.exists() and not force:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return Response(
                url=url,
                final_url=payload["final_url"],
                status=payload["status"],
                text=payload["text"],
                from_cache=True,
                fetched_at=payload["fetched_at"],
                sha256=payload["sha256"],
            )

        allowed, reason = self.check(url)
        if not allowed:
            raise RobotsDisallowed(f"{url}: {reason}")

        origin = self._origin(url)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait(origin)
            try:
                resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            except requests.RequestException as exc:
                last_exc = exc
                self._last_request[origin] = time.monotonic()
                time.sleep(2**attempt + random.random())
                continue
            self._last_request[origin] = time.monotonic()

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                time.sleep(2 ** (attempt + 1) + random.random())
                continue

            fetched_at = datetime.now(timezone.utc).isoformat()
            digest = self._log_provenance(
                url, resp.url, resp.status_code, resp.text, from_cache=False
            )
            payload = {
                "url": url,
                "final_url": resp.url,
                "status": resp.status_code,
                "text": resp.text,
                "fetched_at": fetched_at,
                "sha256": digest,
            }
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return Response(
                url=url,
                final_url=resp.url,
                status=resp.status_code,
                text=resp.text,
                from_cache=False,
                fetched_at=fetched_at,
                sha256=digest,
            )

        raise RuntimeError(f"{url}: giving up after {self.max_retries} attempts ({last_exc})")
