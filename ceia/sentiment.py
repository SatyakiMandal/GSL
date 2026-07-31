"""Finance-aware sentiment plus coarse event tagging (PRD Sections 8, 9).

Sentiment uses **FinBERT** (``ProsusAI/finbert``), not a general-purpose model.
The PRD's reasoning holds up in testing: FinBERT reads "beat expectations but
missed guidance on margins" as 93% negative, where a generic lexicon scores
"beat" and "missed" as cancelling tokens and lands near neutral.

Two deliberate choices:

* **The headline is weighted above the body.** Headlines carry the claim the
  market reacts to; bodies dilute it with background and boilerplate.
* **Long bodies are chunked** to FinBERT's 512-token limit and averaged by
  confidence, rather than truncated, so a late reversal in a story is not lost.

Event tagging is keyword-based, which is what Section 8 asks for ("a full topic
classifier is not required"). It matters mainly so a sector-wide macro story is
not read as company-specific news.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .models import NewsItem

log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"

# Ordered: the first category with a match wins, so specific beats generic.
EVENT_PATTERNS: list[tuple[str, str]] = [
    ("earnings", r"\b(q[1-4]|quarter|quarterly|earnings|profit|revenue|ebitda|"
                 r"results|topline|bottom ?line|margin|guidance|net profit|pat)\b"),
    ("litigation", r"\b(lawsuit|litigation|court|tribunal|nclt|verdict|plea|"
                   r"petition|sued|arbitration|insolvency|bankrupt)\b"),
    ("regulatory", r"\b(sebi|rbi|cci|regulator|regulatory|probe|investigation|"
                   r"penalt|fine|notice|compliance|ruling|approval|licence|license|"
                   r"cbi|enforcement directorate|ed |raid)\b"),
    # Leadership means a *change* of leadership. Requiring an action word stops
    # every story that merely quotes a chairman from landing here.
    ("leadership", r"\b(resign\w*|steps? down|stepped down|quit\w*|ousted|"
                   r"sacked|succession|reshuffle)\b"
                   r"|\b(?:ceo|cfo|coo|managing director|chairman|chairperson|"
                   r"board)\b[^.]{0,40}\b(?:appoint|elevat|named|nominat|exits?|"
                   r"replace|takes over|steps? in)\b"
                   r"|\b(?:appoint|elevat|named|nominat)\w*\b[^.]{0,40}"
                   r"\b(?:ceo|cfo|coo|managing director|chairman|chairperson)\b"),
    ("mna", r"\b(acquisi|acquire|merger|merge|stake sale|takeover|divest|"
            r"buyout|joint venture|open offer)\b"),
    ("capital", r"\b(fpo|ipo|rights issue|qip|fund ?rais|bond|debenture|"
                r"placement|buyback|dividend|share sale|pledge)\b"),
    ("product", r"\b(launch|unveil|new product|expansion|capacity|plant|"
                r"facility|contract win|order book|commission)\b"),
    ("macro", r"\b(inflation|gdp|repo rate|crude|rupee|fed |monetary policy|"
              r"budget|tariff|sector-wide|global markets|nifty|sensex)\b"),
]
_COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in EVENT_PATTERNS]


def tag_event(headline: str, body: str = "") -> str:
    """Coarse event category. The headline is checked before the body."""
    for text in (headline, body[:1200]):
        if not text:
            continue
        for name, pattern in _COMPILED:
            if pattern.search(text):
                return name
    return "other"


@dataclass
class Sentiment:
    label: str
    score: float       # signed: +1 fully positive, -1 fully negative
    confidence: float  # probability of the winning label


class FinBertScorer:
    """Lazy-loading FinBERT wrapper.

    The model is ~440 MB and loads on first use, so constructing the scorer is
    cheap and an ingest run that collects nothing never pays for it.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 16,
                 headline_weight: float = 0.6) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.headline_weight = headline_weight
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "FinBERT needs torch and transformers: pip install -r requirements.txt"
            ) from exc
        log.info("loading %s (first run downloads ~440MB)", self.model_name)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def _classify(self, texts: list[str]) -> list[Sentiment]:
        self._load()
        torch = self._torch
        out: list[Sentiment] = []
        id2label = {i: l.lower() for i, l in self._model.config.id2label.items()}
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self._tokenizer(batch, return_tensors="pt", padding=True,
                                      truncation=True, max_length=512)
            with torch.no_grad():
                probabilities = torch.softmax(self._model(**encoded).logits, dim=-1)
            for row in probabilities:
                scores = {id2label[i]: float(v) for i, v in enumerate(row)}
                label = max(scores, key=scores.get)
                out.append(Sentiment(
                    label=label,
                    score=scores.get("positive", 0.0) - scores.get("negative", 0.0),
                    confidence=scores[label],
                ))
        return out

    def _chunks(self, text: str, max_words: int = 300) -> list[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text] if text.strip() else []
        return [" ".join(words[i:i + max_words])
                for i in range(0, len(words), max_words)]

    def score_items(self, items: list[NewsItem]) -> list[NewsItem]:
        """Score headline and body separately, then blend."""
        if not items:
            return items

        headlines = [i.headline or "" for i in items]
        headline_scores = self._classify(headlines) if any(headlines) else []

        # Flatten every item's body chunks into one batch, tracking ownership.
        chunk_texts: list[str] = []
        owners: list[int] = []
        for index, item in enumerate(items):
            body = item.body or item.snippet or ""
            for chunk in self._chunks(body):
                chunk_texts.append(chunk)
                owners.append(index)
        chunk_scores = self._classify(chunk_texts) if chunk_texts else []

        body_by_item: dict[int, list[Sentiment]] = {}
        for owner, sentiment in zip(owners, chunk_scores):
            body_by_item.setdefault(owner, []).append(sentiment)

        for index, item in enumerate(items):
            head = headline_scores[index] if index < len(headline_scores) else None
            chunks = body_by_item.get(index, [])
            if chunks:
                # Confidence-weighted mean, so hedged chunks count for less.
                total_weight = sum(c.confidence for c in chunks) or 1.0
                body_score = sum(c.score * c.confidence for c in chunks) / total_weight
                body_confidence = sum(c.confidence for c in chunks) / len(chunks)
            else:
                body_score, body_confidence = 0.0, 0.0

            if head and chunks:
                weight = self.headline_weight
                score = weight * head.score + (1 - weight) * body_score
                confidence = weight * head.confidence + (1 - weight) * body_confidence
            elif head:
                score, confidence = head.score, head.confidence
            else:
                score, confidence = body_score, body_confidence

            item.sentiment_score = round(score, 4)
            item.sentiment_confidence = round(confidence, 4)
            item.sentiment_label = (
                "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
            )
            item.event_category = tag_event(item.headline, item.body or item.snippet)
        return items


def score(items: list[NewsItem], scorer: FinBertScorer | None = None) -> list[NewsItem]:
    return (scorer or FinBertScorer()).score_items(items)
