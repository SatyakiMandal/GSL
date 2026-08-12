"""Per-item news staleness: how much a story rehashes this company's own
recent prior coverage, rather than reporting something new.

Mirrors Tetlock (2011, "All the News That's Fit to Reprint"): a story's
staleness is its average textual similarity to the ~10 most recent prior
stories about the same firm. That paper finds stale news gets a smaller
initial price reaction, but the little reaction there is gets partially
reversed over the following week — evidence that investors do not fully
distinguish new information from a rehash of what they already knew.

Reuses :func:`ceia.dedupe.similarity` — the same headline Jaccard/shingle
measure already used for same-day, cross-outlet duplicate detection —
applied differently: across days rather than within one, and against a
company's own prior coverage rather than same-day copies. Every item in a
single run is already about one company (``RunConfig`` scopes the whole
pipeline to it), so no separate firm-matching step is needed here the way
the original paper needed one across thousands of firms.

Scope, disclosed rather than silently assumed: "prior coverage" means prior
items *this run collected* (or had cached — see ``ceia/news_cache.py``), not
a company's full press history. A run whose window starts mid-story will
under-count staleness for its earliest days, the same edge effect the
underlying paper's own decile sort has for a firm's first-ever news event.
"""

from __future__ import annotations

from .dedupe import similarity
from .models import NewsItem

# Tetlock (2011) uses each story's ten most recent predecessors.
DEFAULT_LOOKBACK = 10


def score_items(items: list[NewsItem], lookback: int = DEFAULT_LOOKBACK) -> None:
    """Set ``staleness_score`` in place on every unique, dated item: its mean
    headline similarity to the ``lookback`` most recent prior unique items in
    ``items``, ordered by ``published_at``.

    Duplicates (``duplicate_of`` already set) and undated items are left at
    ``None`` — the former are never counted in a day's coverage anyway, and
    the latter cannot be placed in the chronological sequence a staleness
    score needs. The very first unique, dated item has no prior coverage to
    compare against and is also left at ``None`` rather than scored 0.0 —
    "no prior coverage exists" and "compared against prior coverage and
    found completely novel" are different facts, and conflating them would
    make a single early item look artificially fresh.
    """
    dated_unique = sorted(
        (i for i in items if i.duplicate_of is None and i.published_at is not None),
        key=lambda i: i.published_at,
    )
    for idx, item in enumerate(dated_unique):
        prior = dated_unique[max(0, idx - lookback):idx]
        if not prior:
            continue
        item.staleness_score = sum(
            similarity(item.headline, p.headline) for p in prior
        ) / len(prior)
