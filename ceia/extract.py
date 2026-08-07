"""Article extraction: headline, publish time, body, paywall state.

The four sources expose their metadata three different ways, so extraction
tries each mechanism in turn rather than assuming one:

* **JSON-LD** (Economic Times, Financial Express, Moneycontrol). Moneycontrol
  nests the ``NewsArticle`` inside an ``@graph``, so the walker recurses rather
  than only checking top-level objects.
* **Meta tags** (Business Line, and as a fallback everywhere). Several spellings
  are in use — ``article:published_time``, ``og:article:published_time``,
  ``publish-date``, ``itemprop="datePublished"``.
* **Visible DOM** as a last resort for the body.

A timestamp that cannot be parsed is reported as missing rather than guessed,
because a wrong timestamp silently misattributes an item to the wrong trading
day (PRD Section 10).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Ordered by trustworthiness; the first that parses wins.
_TIMESTAMP_META = [
    ("property", "article:published_time"),
    ("property", "og:article:published_time"),
    ("name", "publish-date"),
    ("itemprop", "datePublished"),
    ("name", "pubdate"),
    ("name", "article.published"),
    ("property", "og:published_time"),
    # VCCircle's non-standard tag - no JSON-LD articleBody or standard
    # article:published_time meta on that site, verified directly.
    ("property", "content_type:published_time"),
]

_PAYWALL_MARKERS = (
    "paywall", "isprime", "subscribe now", "subscriber only",
    "premium story", "for subscribers",
)

# Elements that carry chrome rather than article prose.
_STRIP_TAGS = ("script", "style", "noscript", "iframe", "form", "figure",
               "aside", "nav", "header", "footer", "svg", "button")

# Per-site body containers, best first. A generic fallback runs if none match.
BODY_SELECTORS: dict[str, list[str]] = {
    "economic_times": ["div.artText", "div.Normal", "[data-articlebody]"],
    "financial_express": ["div.pcl-full-width", "div.article-section",
                          "div.entry-content", "article"],
    "business_line": ["div.articlebodycontent", "div.contentbody",
                      "[itemprop='articleBody']"],
    "moneycontrol": ["div#contentdata", "div.content_wrapper",
                     "div.article_content"],
    # Belt-and-suspenders only: Business Today's JSON-LD already carries a
    # full articleBody, which extract_body() tries first and normally
    # succeeds on, so this DOM path is a fallback for the rare article that
    # lacks it. field--name-body is Drupal's body-field class; there are
    # usually two "text-formatted" divs on the page and this is the one that
    # actually holds the article, not a teaser.
    "business_today": ["div.field--name-body", "article"],
}


def _walk(obj: Any) -> Iterator[dict]:
    """Yield every dict nested anywhere in a JSON-LD payload."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def json_ld_articles(html: str) -> list[dict]:
    """Every ``NewsArticle``-ish object in the page, however deeply nested."""
    found = []
    for block in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.S | re.I,
    ):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            # Some sites emit trailing commas or concatenated objects.
            continue
        for node in _walk(data):
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if any(t in ("NewsArticle", "Article", "ReportageNewsArticle")
                   for t in types if isinstance(t, str)):
                found.append(node)
    return found


def parse_timestamp(raw: str | None) -> tuple[datetime | None, str]:
    """Parse an ISO-ish timestamp. Returns ``(value, confidence)``.

    A value with no timezone is assumed IST: every source here is an Indian
    outlet publishing in local time, and the alternative — treating it as UTC —
    would shift items by 5.5 hours and straddle the market close.
    """
    if not raw or not isinstance(raw, str):
        return None, "missing"
    text = raw.strip().replace("Z", "+00:00")
    # Trim fractional seconds beyond microseconds, which fromisoformat rejects.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)

    # A bare date must be caught *before* fromisoformat, which happily returns
    # midnight for "2023-01-25". Treating that as an exact time would place the
    # item confidently before the 15:30 close when we simply do not know.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.fromisoformat(text).replace(tzinfo=IST), "date-only"

    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
        if not match:
            return None, "missing"
        try:
            value = datetime.fromisoformat(match.group(1))
        except ValueError:
            return None, "missing"
        return value.replace(tzinfo=IST), "date-only"
    if value.tzinfo is None:
        value = value.replace(tzinfo=IST)
    return value, "exact"


def extract_timestamp(html: str, soup: BeautifulSoup) -> tuple[datetime | None, str]:
    for article in json_ld_articles(html):
        value, confidence = parse_timestamp(article.get("datePublished"))
        if value is not None:
            return value, confidence
    for attr, key in _TIMESTAMP_META:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            value, confidence = parse_timestamp(tag["content"])
            if value is not None:
                return value, confidence
    tag = soup.find("time")
    if tag:
        value, confidence = parse_timestamp(tag.get("datetime") or tag.get_text())
        if value is not None:
            return value, confidence
    return None, "missing"


# Outlets append their own name to the headline; it is noise for both alias
# matching and sentiment ("Moneycontrol.com" is not part of the claim).
_SITE_SUFFIX_RE = re.compile(
    r"\s*[-|–—]\s*(?:moneycontrol(?:\.com)?|the economic times|economic times|"
    r"financial express|the hindu businessline|businessline)\s*$",
    re.I,
)


def clean_headline(text: str) -> str:
    return _SITE_SUFFIX_RE.sub("", (text or "").strip()).strip()


def extract_headline(html: str, soup: BeautifulSoup) -> str:
    for article in json_ld_articles(html):
        headline = article.get("headline")
        if isinstance(headline, str) and headline.strip():
            return clean_headline(headline)
    for attr, key in (("property", "og:title"), ("name", "twitter:title")):
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            return clean_headline(tag["content"])
    if soup.h1 and soup.h1.get_text(strip=True):
        return clean_headline(soup.h1.get_text(strip=True))
    if soup.title:
        return clean_headline(soup.title.get_text(strip=True).split("|")[0])
    return ""


def _clean(soup_fragment) -> str:
    for tag in soup_fragment.find_all(_STRIP_TAGS):
        tag.decompose()
    text = soup_fragment.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def extract_body(html: str, soup: BeautifulSoup, source_key: str) -> str:
    """Full article text, preferring JSON-LD then per-site DOM containers."""
    for article in json_ld_articles(html):
        body = article.get("articleBody")
        if isinstance(body, str) and len(body.strip()) > 200:
            return re.sub(r"\s+", " ", body).strip()

    for selector in BODY_SELECTORS.get(source_key, []):
        node = soup.select_one(selector)
        if node:
            text = _clean(node)
            if len(text) > 200:
                return text

    # Generic fallback: the densest block of <p> text on the page. Business
    # Line needs this, and a naive paragraph regex would otherwise pull in
    # inline script source as if it were prose.
    best = ""
    for container in soup.find_all(["article", "div", "section"]):
        paragraphs = container.find_all("p", recursive=False) or container.find_all("p")
        if len(paragraphs) < 3:
            continue
        text = " ".join(_clean(p) for p in paragraphs)
        if len(text) > len(best):
            best = text
    return re.sub(r"\s+", " ", best).strip()


def extract_snippet(html: str, soup: BeautifulSoup) -> str:
    """The visible summary. This is the paywall fallback the PRD sanctions."""
    for article in json_ld_articles(html):
        for key in ("description", "abstract"):
            value = article.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr, key in (("property", "og:description"), ("name", "description")):
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def looks_paywalled(html: str, body: str) -> bool:
    """A gate is only meaningful if it actually cost us the text.

    Marker words appear in navigation and promo furniture on every page of
    these sites, so a marker alone is not evidence; a marker together with a
    body too short to analyse is.
    """
    lowered = html.lower()
    has_marker = any(marker in lowered for marker in _PAYWALL_MARKERS)
    return has_marker and len(body) < 600


def parse_article(html: str, url: str, source_key: str) -> dict:
    """Extract everything we need from one article page."""
    soup = BeautifulSoup(html, "lxml")
    published_at, confidence = extract_timestamp(html, soup)
    body = extract_body(html, soup, source_key)
    snippet = extract_snippet(html, soup)
    return {
        "url": url,
        "source": source_key,
        "headline": extract_headline(html, soup),
        "published_at": published_at,
        "timestamp_confidence": confidence,
        "body": body,
        "snippet": snippet,
        "paywalled": looks_paywalled(html, body),
    }
