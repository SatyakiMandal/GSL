"""GoEmotions: a secondary, general-purpose emotional register for coverage.

FinBERT (``ceia/sentiment.py``) supplies the primary signal that drives
incident detection — it is finance-tuned, and everything from the direction
test in ``eventstudy.py`` to the report's abnormal-return-agreement check
depends on its positive/negative score being right, which the validation run
in ``docs/validation-run.md`` confirmed against real prices. That pathway is
not touched here.

GoEmotions (Demszky et al., 2020; ``SamLowe/roberta-base-go_emotions`` is the
standard public checkpoint) is a 27-emotion classifier trained on Reddit
comments. It is **not** finance-tuned, and applying it to formal financial-
press prose is a genuine domain mismatch worth stating plainly rather than
glossing over — a headline like "Adani Group rejects Hindenburg allegations"
does not obviously map onto "admiration" or "amusement" the way a Reddit
comment would. So this module adds GoEmotions as texture, not as a second
vote: it never touches relevance, incident flagging, or the abnormal-return
direction check. It surfaces a dominant emotion label per item — fear, anger,
disapproval, optimism, and so on — that the narrative and report can quote
*alongside* the FinBERT sentiment, so a reader sees not just "negative" but
the flavour of negative. Fear-driven coverage of a regulatory probe reads
differently from anger-driven coverage of an FPO withdrawal, even when
FinBERT scores both similarly negative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import NewsItem

log = logging.getLogger(__name__)

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# The 28 GoEmotions labels (27 emotions + neutral), kept here so downstream
# code and tests do not need to load the model to know the label set.
LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# GoEmotions is multi-label (independent sigmoid per emotion, not softmax),
# and most labels sit near zero for any given headline. Below this confidence
# nothing is reported as "surfaced" — a low top score is noise, not signal.
DEFAULT_THRESHOLD = 0.30


@dataclass
class EmotionResult:
    top_label: str
    top_score: float
    secondary: list[str]  # other labels clearing the threshold, best first


def pick_emotions(
    scores: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    max_secondary: int = 2,
) -> EmotionResult:
    """Turn a label -> probability map into a reportable top emotion.

    ``neutral`` is excluded from consideration entirely, never eligible as
    either the top label or a secondary one. This is not a stylistic choice —
    it is what a real run against financial-press headlines forced: on formal
    news prose, ``neutral`` wins the raw argmax 70-90%+ of the time, because
    the informal markers GoEmotions was trained to key on (Reddit tone,
    exclamations, first-person voice) are simply absent from headline
    English. Reporting that "neutral" as *the* emotion would make this field
    decorative noise rather than the texture it is meant to add, and FinBERT's
    own sentiment score already owns "neutral" as a concept, so nothing is
    lost by looking past it. What is actually informative sits one rank down:
    on a real sample, a rejection-of-allegations headline scored
    neutral=0.69 but disapproval=0.40; an FPO-success headline scored
    neutral=0.77 but approval=0.32. Those are the labels worth reporting.

    Pure and unit-testable without the model: this is the logic that decides
    what "the dominant emotion" means, kept separate from the network that
    produces the raw probabilities.
    """
    ranked = sorted(
        ((label, score) for label, score in scores.items() if label != "neutral"),
        key=lambda kv: -kv[1],
    )
    if not ranked or ranked[0][1] < threshold:
        return EmotionResult("", 0.0, [])
    top_label, top_score = ranked[0]
    secondary = [label for label, score in ranked[1:] if score >= threshold][:max_secondary]
    return EmotionResult(top_label, round(top_score, 4), secondary)


class GoEmotionScorer:
    """Lazy-loading GoEmotions wrapper, scored on the headline only.

    Headline-only is deliberate. GoEmotions was trained on single Reddit
    comments a sentence or two long; feeding it a multi-hundred-word article
    the way FinBERT's chunked-body path does would dilute a signal the model
    was never built to aggregate, for a secondary field the report treats as
    colour rather than a primary driver.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.threshold = threshold
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
                "GoEmotions needs torch and transformers: pip install -r requirements.txt"
            ) from exc
        log.info("loading %s (first run downloads ~500MB)", self.model_name)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def _classify(self, texts: list[str]) -> list[dict[str, float]]:
        self._load()
        torch = self._torch
        id2label = dict(self._model.config.id2label.items())
        out: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self._tokenizer(batch, return_tensors="pt", padding=True,
                                      truncation=True, max_length=64)
            with torch.no_grad():
                # Independent sigmoid per label, not softmax: a headline can
                # register as both "anger" and "disapproval" at once, unlike
                # FinBERT's mutually exclusive positive/negative/neutral.
                probabilities = torch.sigmoid(self._model(**encoded).logits)
            for row in probabilities:
                out.append({id2label[i]: float(v) for i, v in enumerate(row)})
        return out

    def score_items(self, items: list[NewsItem]) -> list[NewsItem]:
        if not items:
            return items
        headlines = [i.headline or "" for i in items]
        raw = self._classify(headlines)
        for item, scores in zip(items, raw):
            result = pick_emotions(scores, self.threshold)
            item.emotion_label = result.top_label
            item.emotion_score = result.top_score
            item.emotion_secondary = result.secondary
        return items


def score(items: list[NewsItem], scorer: GoEmotionScorer | None = None) -> list[NewsItem]:
    return (scorer or GoEmotionScorer()).score_items(items)
