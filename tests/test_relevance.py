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
