"""Tests for the GoEmotions layer.

The actual RoBERTa network is not exercised here (that would need a ~500MB
download); what is tested is the logic around it - thresholding, secondary
emotions, and how a day's dominant emotion is picked - which is the part this
project owns and the part a regression could silently break.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.emotion import (  # noqa: E402
    LABELS, VALENCE_GROUPS, EmotionResult, pick_emotions, valence_of,
)
from ceia.models import NewsItem  # noqa: E402


class TestValenceGroups:
    def test_every_group_label_is_a_real_emotion_label(self):
        all_grouped = {label for labels in VALENCE_GROUPS.values() for label in labels}
        assert all_grouped <= set(LABELS)

    def test_counts_match_the_paper_12_11_4(self):
        assert len(VALENCE_GROUPS["positive"]) == 12
        assert len(VALENCE_GROUPS["negative"]) == 11
        assert len(VALENCE_GROUPS["ambiguous"]) == 4

    def test_every_non_neutral_label_is_grouped_exactly_once(self):
        all_grouped = [label for labels in VALENCE_GROUPS.values() for label in labels]
        assert len(all_grouped) == len(set(all_grouped)), "a label appears in two groups"
        expected = set(LABELS) - {"neutral"}
        assert set(all_grouped) == expected

    def test_valence_of_known_labels(self):
        assert valence_of("joy") == "positive"
        assert valence_of("fear") == "negative"
        assert valence_of("surprise") == "ambiguous"

    def test_valence_of_neutral_and_unknown_is_blank(self):
        assert valence_of("neutral") == ""
        assert valence_of("") == ""
        assert valence_of("not-a-real-label") == ""


class TestPickEmotions:
    def test_confident_top_label_is_surfaced(self):
        result = pick_emotions({"fear": 0.82, "anger": 0.10, "neutral": 0.05})
        assert result.top_label == "fear"
        assert result.top_score == 0.82

    def test_below_threshold_surfaces_nothing(self):
        """A 0.29 top score is noise for a multi-label sigmoid model, not signal."""
        result = pick_emotions({"fear": 0.29, "anger": 0.10})
        assert result == EmotionResult("", 0.0, [])

    def test_exactly_at_threshold_counts(self):
        result = pick_emotions({"fear": 0.30})
        assert result.top_label == "fear"

    def test_empty_scores_surfaces_nothing(self):
        assert pick_emotions({}) == EmotionResult("", 0.0, [])

    def test_secondary_emotions_are_ordered_and_capped(self):
        result = pick_emotions({
            "anger": 0.9, "disapproval": 0.7, "annoyance": 0.5,
            "disgust": 0.4, "neutral": 0.35,
        }, max_secondary=2)
        assert result.top_label == "anger"
        assert result.secondary == ["disapproval", "annoyance"]

    def test_neutral_never_appears_as_a_secondary_emotion(self):
        """Neutral clearing the threshold alongside a real emotion is not
        interesting color; it would just read as noise in the narrative."""
        result = pick_emotions({"fear": 0.8, "neutral": 0.5})
        assert "neutral" not in result.secondary

    def test_neutral_is_never_the_top_label(self):
        """On real financial headlines neutral wins the raw argmax 70-90%+ of
        the time (see the module docstring), because GoEmotions was trained
        on informal Reddit text and news prose carries none of its markers.
        Reporting "neutral" as the emotion would make the field decorative
        noise, and FinBERT's own sentiment already owns that concept - so
        neutral is excluded from the ranking entirely, and the next-best real
        emotion is reported instead."""
        result = pick_emotions({"neutral": 0.93, "approval": 0.32, "fear": 0.02})
        assert result.top_label == "approval"
        assert result.top_score == 0.32

    def test_high_neutral_with_no_real_emotion_above_threshold_is_blank(self):
        result = pick_emotions({"neutral": 0.9, "fear": 0.05, "anger": 0.02})
        assert result.top_label == ""

    def test_custom_threshold_is_respected(self):
        assert pick_emotions({"fear": 0.5}, threshold=0.6).top_label == ""
        assert pick_emotions({"fear": 0.5}, threshold=0.4).top_label == "fear"

    def test_scores_are_rounded(self):
        result = pick_emotions({"fear": 0.821234567})
        assert result.top_score == 0.8212


class TestLabelSet:
    def test_has_28_labels(self):
        assert len(LABELS) == 28

    def test_neutral_is_included(self):
        assert "neutral" in LABELS

    def test_no_duplicates(self):
        assert len(LABELS) == len(set(LABELS))


class TestNewsItemDefaults:
    def test_emotion_fields_default_empty(self):
        item = NewsItem(source="et", url="u", headline="h")
        assert item.emotion_label == ""
        assert item.emotion_score == 0.0
        assert item.emotion_secondary == []

    def test_emotion_fields_round_trip_through_to_dict(self):
        item = NewsItem(source="et", url="u", headline="h",
                        emotion_label="fear", emotion_score=0.8,
                        emotion_secondary=["nervousness"])
        record = item.to_dict()
        assert record["emotion_label"] == "fear"
        assert record["emotion_secondary"] == ["nervousness"]


class TestGoEmotionScorerPostProcessing:
    """Exercise score_items() against a stubbed _classify so the batching and
    field-assignment logic is covered without loading the real network."""

    def test_assigns_fields_from_classify_output(self, monkeypatch):
        from ceia.emotion import GoEmotionScorer

        scorer = GoEmotionScorer()
        monkeypatch.setattr(
            scorer, "_classify",
            lambda texts: [{"fear": 0.9, "anger": 0.1} for _ in texts],
        )
        items = [NewsItem(source="et", url="u1", headline="a"),
                 NewsItem(source="et", url="u2", headline="b")]
        scorer.score_items(items)
        assert all(i.emotion_label == "fear" for i in items)
        assert all(i.emotion_score == 0.9 for i in items)

    def test_empty_item_list_is_a_noop(self, monkeypatch):
        from ceia.emotion import GoEmotionScorer

        scorer = GoEmotionScorer()
        calls = []
        monkeypatch.setattr(scorer, "_classify", lambda texts: calls.append(texts) or [])
        assert scorer.score_items([]) == []
        assert calls == []

    def test_low_confidence_items_get_no_label(self, monkeypatch):
        from ceia.emotion import GoEmotionScorer

        scorer = GoEmotionScorer()
        monkeypatch.setattr(scorer, "_classify",
                            lambda texts: [{"fear": 0.1, "neutral": 0.2} for _ in texts])
        items = [NewsItem(source="et", url="u", headline="a")]
        scorer.score_items(items)
        assert items[0].emotion_label == ""
