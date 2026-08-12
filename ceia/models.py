"""Core data types shared across ingestion, scoring and reporting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

# Indian market hours. Items published after the close belong to the next
# trading day's reaction (PRD Section 10).
IST_OFFSET_HOURS = 5.5
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Longest/most specific first, so "XYZ Private Limited" strips to "XYZ"
# rather than to "XYZ Private" by matching the bare "limited" suffix first.
_CORPORATE_SUFFIXES = [
    "private limited", "pvt ltd", "limited", "ltd",
    "incorporated", "inc", "corporation", "corp", "company", "co", "llc", "plc",
]


def _strip_corporate_suffix(company: str) -> str | None:
    """The company name with a trailing corporate suffix removed, or
    ``None`` if it does not end in one of the known suffixes.

    News headlines almost never carry the full legal suffix - the Indian
    financial press writes "Sonata Software", not "Sonata Software
    Limited" - so a run whose alias list is built only from the exact
    company name can silently match nothing. Verified directly on a real
    case: scoring a body saturated with "Sonata Software" mentions against
    the alias "Sonata Software Limited" alone returns 0.0, "no alias
    match"; adding the suffix-stripped "Sonata Software" scores it 0.86.
    This is why ``RunConfig.all_aliases`` always tries a stripped form
    rather than depending on the user to supply one by hand.
    """
    normalised = company.strip()
    lowered = normalised.lower()
    for suffix in _CORPORATE_SUFFIXES:
        pattern = r"[\s,]+" + r"\.?\s+".join(re.escape(w) for w in suffix.split()) + r"\.?$"
        match = re.search(pattern, lowered)
        if match:
            stripped = normalised[:match.start()].strip().rstrip(",")
            if stripped:
                return stripped
    return None


@dataclass
class NewsItem:
    """One collected article.

    ``published_at`` is timezone-aware whenever the source gave us a usable
    timestamp. When it did not, ``timestamp_confidence`` is ``"missing"`` and
    the item is flagged rather than silently attributed to a trading day, which
    is the misattribution failure Section 10 calls out.
    """

    source: str
    url: str
    headline: str
    published_at: datetime | None = None
    timestamp_confidence: str = "missing"  # exact | date-only | missing
    body: str = ""
    snippet: str = ""
    paywalled: bool = False

    # Set by the relevance filter.
    relevance_score: float = 0.0
    matched_aliases: list[str] = field(default_factory=list)
    headline_match: bool = False

    # Set by the sentiment/event tagger.
    sentiment_label: str = ""
    sentiment_score: float = 0.0  # signed: + positive, - negative
    sentiment_confidence: float = 0.0
    event_category: str = ""

    # Set by the emotion tagger (GoEmotions) - a secondary, general-purpose
    # signal read alongside sentiment_label, never a substitute for it. Empty
    # when no emotion cleared the confidence threshold (see ceia/emotion.py).
    emotion_label: str = ""
    emotion_score: float = 0.0
    emotion_secondary: list[str] = field(default_factory=list)

    # Set by the trading-day aligner.
    trading_day: date | None = None
    after_close: bool = False

    # Provenance.
    fetched_at: str = ""
    content_sha256: str = ""
    duplicate_of: str | None = None

    # Set by ceia.staleness - mean headline similarity to this company's most
    # recent prior unique stories (Tetlock (2011)'s staleness measure). None
    # when undated, a marked duplicate, or the earliest unique item in the
    # run (no prior coverage to compare against) - see ceia/staleness.py.
    staleness_score: float | None = None

    @property
    def text_for_scoring(self) -> str:
        """Headline plus body; the headline is what most drives sentiment."""
        body = self.body or self.snippet
        return f"{self.headline}. {body}".strip()

    @property
    def dedupe_key(self) -> str:
        """Normalised headline, for spotting syndicated wire copy across sites."""
        normalised = re.sub(r"[^a-z0-9 ]+", " ", self.headline.lower())
        normalised = re.sub(r"\s+", " ", normalised).strip()
        return hashlib.sha1(normalised.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["published_at"] = self.published_at.isoformat() if self.published_at else None
        record["trading_day"] = self.trading_day.isoformat() if self.trading_day else None
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "NewsItem":
        """Inverse of :meth:`to_dict` - shared by the ``--news`` file loader
        and the per-company news cache so the two don't drift apart.
        ``trading_day`` is dropped rather than parsed back: it depends on
        the trading calendar, recomputed fresh by ``align.attribute_all``
        after prices are loaded, not carried across a save/reload."""
        record = dict(record)
        published = record.pop("published_at", None)
        record.pop("trading_day", None)
        item = cls(**record)
        if published:
            item.published_at = datetime.fromisoformat(published)
        return item


@dataclass
class RunConfig:
    """The inputs from PRD Section 6."""

    company: str
    ticker: str
    exchange: str = "NSE"
    benchmark: str = "^NSEI"
    # Optional second index/peer ticker for a side-by-side abnormal-return
    # comparison — e.g. a sector peer instead of the broad market. Purely
    # additive: the primary `benchmark` above still drives incident detection
    # and scoring, this is a second lens shown alongside it.
    benchmark2: str | None = None
    start: date = date.today()
    end: date = date.today()
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    event_window: tuple[int, int] = (-1, 3)
    min_relevance: float = 0.35

    def __post_init__(self) -> None:
        # An inverted range makes every discovery strategy's date loop empty
        # (`_days`/`_months` in discovery.py never yield), so it silently
        # produces zero candidates from every source rather than an error -
        # indistinguishable from "scraping is broken" unless caught here.
        if self.start > self.end:
            raise ValueError(
                f"start date ({self.start}) is after end date ({self.end})"
            )

    @property
    def all_aliases(self) -> list[str]:
        """Company name plus user-supplied aliases, longest first, plus a
        suffix-stripped short form of the company name (see
        :func:`_strip_corporate_suffix`) so a run does not silently depend
        on the user remembering to supply one by hand.

        Longest-first matters so "Adani Enterprises" is preferred over "Adani"
        when both match, which keeps the more specific alias in
        ``matched_aliases``.
        """
        names = [self.company, *self.aliases]
        short_form = _strip_corporate_suffix(self.company)
        if short_form:
            names.append(short_form)
        seen, out = set(), []
        for name in sorted(names, key=len, reverse=True):
            key = name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(name.strip())
        return out
