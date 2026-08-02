"""Relevance filtering (PRD Section 8).

Not everything that names a company is *about* it. A market wrap that lists
thirty movers, or a sector piece that cites a rival in passing, should not carry
the same weight as a story whose subject is the company. The PRD asks for
keyword/alias matching plus a relevance score, not a trained classifier.

The score combines four signals:

* **where** the alias appears — headline mentions dominate, because a headline
  is a claim about what the story is about;
* **how often** it appears relative to article length;
* **how early** the first mention lands;
* **penalties** for shapes that reliably indicate a passing mention — long
  market round-ups, and articles whose headline names a *different* company.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import NewsItem

# Headlines shaped like these mention many companies without being about any.
_ROUNDUP_PATTERNS = (
    r"\btop (?:gainers|losers|picks|stocks)\b",
    r"\bmarket wrap\b", r"\bclosing bell\b", r"\bopening bell\b",
    r"\bstocks to watch\b", r"\bbuzzing stocks\b", r"\btrade spotlight\b",
    r"\bsensex\b.*\bnifty\b", r"\bmarket live\b", r"\blive updates\b",
    r"\bwhat changed for the market\b", r"\bstocks in news\b",
)
_ROUNDUP_RE = re.compile("|".join(_ROUNDUP_PATTERNS), re.I)

# Component weights. They sum to 0.97 at their theoretical maximum, so a
# perfect score is reachable in principle but rare in practice - the point is
# that scores spread across the range instead of piling up on the ceiling.
W_HEADLINE = 0.45   # named in the headline: the strongest single signal
W_BODY = 0.30       # saturating in the number of body mentions
W_DENSITY = 0.12    # mentions per 100 words
W_POSITION = 0.10   # first mention lands early
P_ROUNDUP = 0.40    # multi-company market round-up
P_NO_HEADLINE = 0.15  # subject of the headline is some other company


@dataclass
class RelevanceResult:
    score: float
    matched: list[str]
    headline_match: bool
    reason: str


def _alias_pattern(alias: str) -> re.Pattern:
    """Word-boundary match, tolerant of whitespace runs and English inflection.

    The trailing group matters more than it looks. The Indian financial press
    routinely writes the family or group form — "Adanis dismiss US firm's
    allegations" — and a bare ``\\bAdani\\b`` misses it, because the ``s`` is a
    word character. On a real corpus that cost the headline match on a story
    with forty body mentions, dropping it from ~0.9 to 0.37.
    """
    parts = [re.escape(part) for part in alias.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"(?:['’]s|s['’]|s)?\b",
                      re.I)


def score_item(
    headline: str,
    body: str,
    aliases: list[str],
    *,
    ticker: str | None = None,
) -> RelevanceResult:
    headline = headline or ""
    body = body or ""
    matched: list[str] = []
    headline_hits = 0
    body_hits = 0
    first_position: int | None = None

    searchable = list(aliases)
    if ticker:
        # Bare NSE/BSE symbol, e.g. ADANIENT from ADANIENT.NS.
        searchable.append(ticker.split(".")[0])

    for alias in searchable:
        if len(alias) < 3:
            continue
        pattern = _alias_pattern(alias)
        in_headline = len(pattern.findall(headline))
        in_body = len(pattern.findall(body))
        if not (in_headline or in_body):
            continue
        # Record only the longest matching alias family, not every substring.
        if not any(alias.lower() in seen.lower() for seen in matched):
            matched.append(alias)
        headline_hits += in_headline
        body_hits += in_body
        match = pattern.search(body)
        if match and (first_position is None or match.start() < first_position):
            first_position = match.start()

    if not matched:
        return RelevanceResult(0.0, [], False, "no alias match")

    score = 0.0
    reasons = []

    if headline_hits:
        score += W_HEADLINE
        reasons.append("named in headline")

    if body_hits:
        # Saturating curve rather than a linear ramp into a cap. A linear
        # component plus a headline match sent almost everything to the 1.0
        # ceiling: on a 137-item corpus, 121 items scored exactly 1.00, which
        # makes the score useless both for ranking and as the weight it feeds
        # into weighted sentiment. This keeps scores spread across the range.
        score += W_BODY * (1 - math.exp(-body_hits / 3))
        density = body_hits / max(len(body.split()) / 100, 1)
        score += min(W_DENSITY, density * 0.06)
        reasons.append(f"{body_hits} body mention(s)")

    if first_position is not None and body:
        # Early mentions suggest subjecthood; late ones suggest a passing cite.
        position_ratio = first_position / max(len(body), 1)
        if position_ratio < 0.15:
            score += W_POSITION
            reasons.append("mentioned early")
        elif position_ratio > 0.60:
            score -= 0.10
            reasons.append("only mentioned late")

    if _ROUNDUP_RE.search(headline):
        score -= P_ROUNDUP
        reasons.append("market round-up headline")

    # A headline naming a different company, with ours only in the body, is the
    # classic passing comparison.
    if not headline_hits and body_hits:
        score -= P_NO_HEADLINE
        reasons.append("not named in headline")

    score = max(0.0, min(1.0, score))
    return RelevanceResult(round(score, 4), matched, headline_hits > 0,
                           "; ".join(reasons))


def apply(items: list[NewsItem], aliases: list[str], ticker: str | None,
          threshold: float) -> tuple[list[NewsItem], list[NewsItem]]:
    """Score every item; return ``(kept, dropped)``."""
    kept, dropped = [], []
    for item in items:
        result = score_item(item.headline, item.body or item.snippet,
                            aliases, ticker=ticker)
        item.relevance_score = round(result.score, 3)
        item.matched_aliases = result.matched
        item.headline_match = result.headline_match
        (kept if result.score >= threshold else dropped).append(item)
    return kept, dropped
