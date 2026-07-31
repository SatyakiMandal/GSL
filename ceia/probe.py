"""Phase 0 feasibility spike (PRD Section 13).

Answers, per news source and reproducibly:

* Can robots.txt be read at all, and what does it permit for our user agent?
* Does the site issue a blanket refusal to AI-agent user agents?
* Is the PRD's assumed search endpoint allowed? If not, which discovery route is?
* Do the pages we would parse arrive as server-rendered HTML, or do they need
  a headless browser?
* How far back is the archive browsable, and how much article text is visible
  without a login?

Plus: does the price provider return clean data for the ticker and benchmark?

Run with ``python -m ceia.probe``. Results are written to
``docs/phase0-findings.json`` and summarised on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .fetcher import DEFAULT_USER_AGENT, Fetcher, RobotsDisallowed
from .prices import CsvProvider, PriceError, YahooChartProvider, YFinanceProvider
from .sources import AI_AGENT_TOKENS, ALL_SOURCES, Source

log = logging.getLogger(__name__)

# Markers that a page is a shell whose content arrives via JavaScript.
_JS_SHELL_HINTS = ("__NEXT_DATA__", "window.__INITIAL", "id=\"root\"", "ng-app",
                   "requires JavaScript", "enable JavaScript")


@dataclass
class SourceFinding:
    key: str
    name: str
    robots_readable: bool = False
    robots_status: int | None = None
    ai_agents_blocked: list[str] = field(default_factory=list)
    blocks_our_agent: bool = False
    search_allowed: bool | None = None
    search_reason: str = ""
    crawl_delay: float | None = None
    sitemaps_declared: list[str] = field(default_factory=list)
    discovery: list[dict] = field(default_factory=list)
    article_check: dict = field(default_factory=dict)
    verdict: str = ""
    notes: str = ""


def _looks_javascript_rendered(html: str) -> bool:
    if len(html) < 2000:
        return True
    return any(hint in html for hint in _JS_SHELL_HINTS) and "<article" not in html


def probe_source(source: Source, fetcher: Fetcher, fetch_pages: bool) -> SourceFinding:
    finding = SourceFinding(key=source.key, name=source.name, notes=source.notes)
    policy = fetcher.robots_for(source.origin + "/")
    finding.robots_readable = policy.fetched_ok
    finding.robots_status = policy.status

    if not policy.fetched_ok:
        finding.verdict = (
            f"UNAVAILABLE - robots.txt returned {policy.status}; permission to "
            "crawl cannot be established, so this source is skipped."
        )
        return finding

    finding.ai_agents_blocked = policy.blocks_entirely(AI_AGENT_TOKENS)
    finding.sitemaps_declared = policy.sitemaps
    finding.crawl_delay = policy.crawl_delay(fetcher.user_agent)

    allowed, reason = policy.allows(source.search_url, fetcher.user_agent)
    finding.search_allowed = allowed
    finding.search_reason = reason

    ours_blocked, ours_reason = policy.allows(source.origin + "/", fetcher.user_agent)
    finding.blocks_our_agent = not ours_blocked

    for url in source.discovery:
        ok, why = policy.allows(url, fetcher.user_agent)
        entry = {"url": url, "robots_allowed": ok, "reason": why}
        if ok and fetch_pages:
            try:
                resp = fetcher.get(url)
                entry["status"] = resp.status
                entry["bytes"] = len(resp.text)
                entry["final_url"] = resp.final_url
                entry["redirected"] = resp.final_url.rstrip("/") != url.rstrip("/")
                if url.endswith(".xml"):
                    entry["sitemap_children"] = len(re.findall(r"<loc>", resp.text))
                    stamps = re.findall(r"<lastmod>(\d{4}-\d{2})", resp.text)
                    months = re.findall(r"/(\d{4}-[A-Za-z]+-\d+)\.xml", resp.text)
                    if months:
                        entry["archive_oldest_partition"] = months[-1]
                        entry["archive_newest_partition"] = months[0]
                    if stamps:
                        entry["lastmod_range"] = [min(stamps), max(stamps)]
                else:
                    entry["javascript_rendered"] = _looks_javascript_rendered(resp.text)
                    entry["article_links"] = len(
                        set(re.findall(r"articleshow/\d+\.cms|/article\d+\.ece", resp.text))
                    )
            except RobotsDisallowed as exc:
                entry["error"] = f"robots: {exc}"
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        finding.discovery.append(entry)

    if source.sample_article and fetch_pages:
        finding.article_check = _probe_article(source.sample_article, fetcher)

    finding.verdict = _verdict(finding)
    return finding


def _probe_article(url: str, fetcher: Fetcher) -> dict:
    out: dict = {"url": url}
    try:
        resp = fetcher.get(url)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    html = resp.text
    out["status"] = resp.status
    out["javascript_rendered"] = _looks_javascript_rendered(html)
    out["paywall_markers"] = sorted(
        {m.lower() for m in re.findall(r"paywall|isPrime|Subscribe Now|Premium", html)}
    )
    for block in re.findall(
        r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for obj in data if isinstance(data, list) else [data]:
            if not isinstance(obj, dict) or "NewsArticle" not in str(obj.get("@type", "")):
                continue
            body = obj.get("articleBody") or ""
            out["jsonld"] = {
                "headline": obj.get("headline"),
                "datePublished": obj.get("datePublished"),
                "dateModified": obj.get("dateModified"),
                "articleBody_chars": len(body),
            }
    return out


def _verdict(f: SourceFinding) -> str:
    if not f.robots_readable:
        return "UNAVAILABLE"
    if f.blocks_our_agent:
        return "BLOCKED for the configured user agent"
    usable = [d for d in f.discovery if d.get("robots_allowed") and d.get("status") == 200]
    if usable:
        route = "sitemap" if usable[0]["url"].endswith(".xml") else "archive/topic page"
        note = "" if f.search_allowed else " (search endpoint disallowed; using archive route)"
        return f"USABLE via {route}{note}"
    allowed_routes = [d for d in f.discovery if d.get("robots_allowed")]
    if allowed_routes:
        return "ROUTE PERMITTED BUT UNVERIFIED - not fetched in this run"
    return "NO PERMITTED DISCOVERY ROUTE FOUND"


def probe_prices(ticker: str, benchmark: str, start: date, end: date) -> dict:
    out: dict = {"ticker": ticker, "benchmark": benchmark,
                 "window": [str(start), str(end)], "providers": {}}
    for provider in (YFinanceProvider(), YahooChartProvider(), CsvProvider()):
        entry: dict = {}
        for label, symbol in (("ticker", ticker), ("benchmark", benchmark)):
            try:
                frame = provider.history(symbol, start, end)
                entry[label] = {
                    "ok": True,
                    "rows": len(frame),
                    "first": str(frame.index[0].date()) if len(frame) else None,
                    "last": str(frame.index[-1].date()) if len(frame) else None,
                    "missing_close": int(frame["close"].isna().sum()),
                }
            except PriceError as exc:
                entry[label] = {"ok": False, "error": str(exc)[:200]}
            except Exception as exc:
                entry[label] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
        entry["usable"] = all(v.get("ok") for v in entry.values() if isinstance(v, dict))
        out["providers"][provider.name] = entry
    out["chosen"] = next(
        (name for name, v in out["providers"].items() if v.get("usable")), None
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 feasibility spike")
    parser.add_argument("--ticker", default="ADANIENT.NS")
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-03-01")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="docs/phase0-findings.json")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Restrict to these source keys (default: all four).",
    )
    parser.add_argument(
        "--robots-only", action="store_true",
        help="Evaluate robots.txt without fetching any content page.",
    )
    parser.add_argument("--skip-prices", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetcher = Fetcher(cache_dir=args.cache_dir, user_agent=args.user_agent)

    selected = [s for s in ALL_SOURCES if args.only is None or s.key in args.only]
    findings = [probe_source(s, fetcher, fetch_pages=not args.robots_only) for s in selected]

    report: dict = {
        "generated_at": date.today().isoformat(),
        "user_agent": args.user_agent,
        "sources": [asdict(f) for f in findings],
    }
    if not args.skip_prices:
        report["prices"] = probe_prices(
            args.ticker, args.benchmark,
            date.fromisoformat(args.start), date.fromisoformat(args.end),
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}\nPHASE 0 FEASIBILITY SUMMARY\nuser agent: {args.user_agent}\n{'=' * 72}")
    for f in findings:
        print(f"\n[{f.name}]  {f.verdict}")
        print(f"  robots.txt readable : {f.robots_readable} (status {f.robots_status})")
        if f.ai_agents_blocked:
            print(f"  AI agents refused   : {', '.join(f.ai_agents_blocked)}")
        if f.search_allowed is not None:
            print(f"  PRD search endpoint : "
                  f"{'allowed' if f.search_allowed else 'DISALLOWED'} - {f.search_reason}")
        for d in f.discovery:
            bits = [f"robots={'ok' if d['robots_allowed'] else 'no'}"]
            for key in ("status", "sitemap_children", "javascript_rendered",
                        "archive_oldest_partition", "article_links", "error"):
                if key in d:
                    bits.append(f"{key}={d[key]}")
            print(f"    - {d['url'][:78]}\n        {' '.join(bits)}")
        if f.article_check:
            print(f"  article check       : {json.dumps(f.article_check.get('jsonld', {}))}")
            if f.article_check.get("paywall_markers"):
                print(f"  paywall markers     : {f.article_check['paywall_markers']}")

    if "prices" in report:
        print(f"\n{'=' * 72}\nPRICE DATA\n{'=' * 72}")
        for name, entry in report["prices"]["providers"].items():
            print(f"  {name:14s} usable={entry['usable']}")
            for label in ("ticker", "benchmark"):
                print(f"      {label:10s} {entry[label]}")
        print(f"  chosen provider: {report['prices']['chosen']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
