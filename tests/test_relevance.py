"""Tests for the relevance filter and event tagging."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.models import NewsItem, RunConfig  # noqa: E402
from ceia.relevance import apply, score_item  # noqa: E402
from ceia.sentiment import tag_event  # noqa: E402

ALIASES = ["Adani Enterprises", "Adani", "AEL"]


class TestRelevanceScoring:
    def test_headline_subject_scores_high(self):
        result = score_item(
            "Adani Enterprises FPO fully subscribed on final day",
            "The follow-on public offer by Adani Enterprises sailed through on Tuesday.",
            ALIASES,
        )
        assert result.score >= 0.7
        assert result.headline_match

    def test_passing_mention_scores_low(self):
        body = ("Tata Steel led the gains on Tuesday. " + "Filler commentary. " * 60
                + "Elsewhere, Adani was little changed.")
        result = score_item("Tata Steel jumps 5% on upgrade", body, ALIASES)
        assert result.score < 0.35, f"got {result.score} ({result.reason})"
        assert not result.headline_match

    def test_market_roundup_is_downweighted(self):
        subject = score_item("Adani Enterprises shares fall 20%",
                             "Adani Enterprises dropped sharply.", ALIASES)
        roundup = score_item("Top gainers and losers: Adani, Infosys, TCS in focus",
                             "Adani Enterprises dropped sharply.", ALIASES)
        assert roundup.score < subject.score

    def test_no_match_scores_zero(self):
        result = score_item("Infosys wins large deal", "Infosys signed a contract.",
                            ALIASES)
        assert result.score == 0.0
        assert result.matched == []

    def test_ticker_symbol_matches(self):
        result = score_item("ADANIENT hits lower circuit", "The stock fell.",
                            ["Adani Enterprises"], ticker="ADANIENT.NS")
        assert result.score > 0
        assert result.headline_match

    def test_longest_alias_is_reported(self):
        result = score_item("Adani Enterprises FPO news", "Adani Enterprises again.",
                            ALIASES)
        assert "Adani Enterprises" in result.matched

    def test_short_aliases_ignored(self):
        """Two-letter aliases would match everywhere; they are skipped."""
        result = score_item("Something about oil", "AB CD EF", ["AB"])
        assert result.score == 0.0

    def test_score_is_bounded(self):
        result = score_item("Adani Adani Adani Adani", "Adani " * 200, ALIASES)
        assert 0.0 <= result.score <= 1.0


class TestApply:
    def test_splits_on_threshold(self):
        items = [
            NewsItem(source="et", url="u1",
                     headline="Adani Enterprises posts higher profit",
                     body="Adani Enterprises reported results."),
            NewsItem(source="et", url="u2", headline="Weather update",
                     body="Rain expected."),
        ]
        kept, dropped = apply(items, ALIASES, "ADANIENT.NS", 0.35)
        assert [i.url for i in kept] == ["u1"]
        assert [i.url for i in dropped] == ["u2"]
        assert kept[0].relevance_score > 0


class TestEventTagging:
    def test_categories(self):
        cases = [
            ("Adani Enterprises Q3 net profit rises 12%", "earnings"),
            ("SEBI opens probe into Adani group disclosures", "regulatory"),
            ("Adani CFO resigns after five years", "leadership"),
            ("Adani Enterprises FPO opens for subscription", "capital"),
            ("Adani to acquire stake in cement maker", "mna"),
            ("Supreme Court hears plea against Adani deal", "litigation"),
            ("Adani commissions new solar plant in Gujarat", "product"),
            ("Sensex, Nifty slip as inflation worries mount", "macro"),
            ("Adani chairman visits Israel", "other"),
        ]
        for headline, expected in cases:
            assert tag_event(headline) == expected, f"{headline!r} -> {tag_event(headline)}"

    def test_headline_beats_body(self):
        assert tag_event("Adani Q3 results beat estimates",
                         "Separately, SEBI issued a notice.") == "earnings"


class TestRunConfig:
    def test_aliases_are_deduped_longest_first(self):
        config = RunConfig(company="Adani Enterprises", ticker="ADANIENT.NS",
                           aliases=["Adani", "adani enterprises", "AEL"])
        assert config.all_aliases[0] == "Adani Enterprises"
        lowered = [a.lower() for a in config.all_aliases]
        assert len(lowered) == len(set(lowered))


class TestInflection:
    """Indian financial press routinely uses the family/group plural form."""

    def test_plural_alias_matches(self):
        result = score_item("Adanis dismiss US firm's allegations",
                            "The Adani group rejected the report. " * 12, ALIASES)
        assert result.headline_match
        assert result.score > 0.8

    def test_possessive_alias_matches(self):
        result = score_item("Adani's FPO sails through",
                            "Adani Enterprises priced the issue. " * 10, ALIASES)
        assert result.headline_match

    def test_curly_apostrophe_matches(self):
        result = score_item("Adani’s FPO sails through",
                            "Adani Enterprises priced the issue. " * 10, ALIASES)
        assert result.headline_match

    def test_does_not_match_a_longer_unrelated_word(self):
        """'Adaniyar' is not 'Adani' with an inflection."""
        result = score_item("Adaniyar village wins award",
                            "Adaniyar is a village. " * 10, ["Adani"])
        assert result.score == 0.0


class TestScoreDistribution:
    """The score must discriminate, not pile up on the ceiling.

    A linear body component plus a headline match previously sent 121 of 137
    items on a real corpus to exactly 1.00, which made the score useless both
    for ranking and as the weight it feeds into weighted sentiment.
    """

    def _score(self, headline, body):
        return score_item(headline, body, ALIASES, ticker="ADANIENT.NS").score

    def test_subject_article_scores_high_but_below_the_ceiling(self):
        score = self._score(
            "Adani Enterprises FPO fully subscribed",
            "Adani Enterprises said the offer closed. " * 20)
        assert 0.7 <= score < 1.0, f"got {score}"

    def test_more_mentions_never_reach_exactly_one(self):
        score = self._score("Adani Enterprises everywhere", "Adani Enterprises " * 400)
        assert score < 1.0

    def test_marginal_and_subject_are_clearly_separated(self):
        subject = self._score("Adani Enterprises posts higher profit",
                              "Adani Enterprises reported results. " * 20)
        marginal = self._score(
            "BJP slams investor over comments",
            "Political row continued. " * 40 + "Adani was mentioned. " * 3)
        assert subject - marginal > 0.3, f"subject {subject}, marginal {marginal}"

    def test_roundup_falls_below_threshold(self):
        score = self._score(
            "Sensex tanks 600 pts, Nifty slips below 17,750 in early trade",
            "Adani Enterprises fell. " * 12)
        assert score < 0.6, "a market round-up should be clearly downweighted"
