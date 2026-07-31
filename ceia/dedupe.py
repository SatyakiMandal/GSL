"""Near-duplicate detection (PRD Section 8).

Wire-service copy runs near-identically across several outlets on the same day.
Left alone it inflates the "unusual coverage volume" signal that drives incident
ranking, so one PTI story carried by four sites would look like a burst of
independent attention rather than one story.

Matching is on token-shingle (Jaccard) similarity of the headline, restricted to
items sharing a publication day. Bodies are deliberately not used: paywalled
items have none, and syndicated copy is often re-topped and re-edited below the
headline anyway.

Duplicates are marked, not deleted — ``duplicate_of`` points at the survivor, so
the report can still show that four outlets carried a story while counting it
once.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import NewsItem

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "as",
    "is", "are", "was", "were", "be", "by", "with", "from", "its", "it", "that",
    "this", "after", "over", "amid", "says", "say", "said", "will",
}


def _tokens(headline: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", headline.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _shingles(headline: str, size: int = 2) -> frozenset[tuple[str, ...]]:
    words = [w for w in re.findall(r"[a-z0-9]+", headline.lower())
             if w not in _STOPWORDS]
    if len(words) < size:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i:i + size]) for i in range(len(words) - size + 1))


def similarity(left: str, right: str) -> float:
    """Blended unigram/bigram Jaccard similarity of two headlines."""
    a_tokens, b_tokens = _tokens(left), _tokens(right)
    if not a_tokens or not b_tokens:
        return 0.0
    unigram = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    a_shingles, b_shingles = _shingles(left), _shingles(right)
    bigram = 0.0
    if a_shingles and b_shingles:
        bigram = len(a_shingles & b_shingles) / len(a_shingles | b_shingles)
    return 0.6 * unigram + 0.4 * bigram


def deduplicate(items: list[NewsItem], threshold: float = 0.72) -> list[NewsItem]:
    """Mark near-duplicates in place and return the list.

    The survivor of a cluster is the item with the most body text — the fullest
    version of the story — so downstream sentiment scores the best copy available.
    """
    buckets: dict[object, list[NewsItem]] = defaultdict(list)
    for item in items:
        # Group by publication day so unrelated stories that reuse phrasing
        # months apart are never compared.
        buckets[item.published_at.date() if item.published_at else None].append(item)

    for bucket in buckets.values():
        # Longest body first, so the survivor is chosen before its duplicates.
        ordered = sorted(bucket, key=lambda i: len(i.body or i.snippet or ""), reverse=True)
        survivors: list[NewsItem] = []
        for item in ordered:
            match = next(
                (s for s in survivors
                 if similarity(item.headline, s.headline) >= threshold),
                None,
            )
            if match is None:
                survivors.append(item)
            else:
                item.duplicate_of = match.url
    return items


def unique(items: list[NewsItem]) -> list[NewsItem]:
    return [i for i in items if i.duplicate_of is None]


def cluster_sizes(items: list[NewsItem]) -> dict[str, int]:
    """How many outlets carried each surviving story (itself included)."""
    counts: dict[str, int] = {i.url: 1 for i in items if i.duplicate_of is None}
    for item in items:
        if item.duplicate_of in counts:
            counts[item.duplicate_of] += 1
    return counts
