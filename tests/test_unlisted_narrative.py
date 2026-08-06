"""Tests for the unlisted-share narrative templates.

The important properties mirror test_report.py's for the listed-company
report: no causal language, and — specific to this module — no borrowed
statistical vocabulary (z-score, CAR, permutation, market model) that the
underlying indicative-price data cannot actually earn.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.unlisted import PriceMove  # noqa: E402
from ceia.unlisted_narrative import move_narrative, unlisted_summary_narrative  # noqa: E402

CAUSAL_WORDS = [
    r"\bcaused\b", r"\bcausing\b", r"\bdrove\b", r"\btriggered\b",
    r"\bsent the stock\b", r"\bled to a\b", r"\bresulted in a\b",
]

# None of this vocabulary is earned by a periodically-revised indicative
# price - see ceia/unlisted_narrative.py's module docstring.
BORROWED_STATS_WORDS = [
    r"\bz-score\b", r"\bstandard deviations?\b", r"\bCAR\b",
    r"\bpermutation\b", r"\bmarket model\b", r"\bp-value\b", r"\bbeta\b",
]


def _move(**kwargs) -> PriceMove:
    defaults = dict(start_date=date(2026, 1, 1), end_date=date(2026, 1, 15),
                    start_price=100.0, end_price=120.0)
    defaults.update(kwargs)
    return PriceMove(**defaults)


class TestMoveNarrative:
    def test_states_the_price_change_and_date_range(self):
        text = " ".join(move_narrative(_move(), "Test Co"))
        assert "01 January 2026" in text and "15 January 2026" in text
        assert "+20.00%" in text
        assert "₹100.00" in text and "₹120.00" in text

    def test_never_dates_the_change_more_precisely_than_the_range(self):
        text = " ".join(move_narrative(_move(), "Test Co"))
        assert "cannot be dated more precisely than this range" in text

    def test_no_causal_language(self):
        for change in (0.2, -0.2, 0.0):
            move = _move(end_price=100.0 * (1 + change))
            text = " ".join(move_narrative(move, "Test Co"))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if re.search(r"\b(?:cannot|can not|not|never|no)\b", sentence, re.I):
                    continue
                for pattern in CAUSAL_WORDS:
                    assert not re.search(pattern, sentence, re.I), \
                        f"causal phrasing {pattern!r} in: {sentence!r}"

    def test_no_borrowed_statistical_vocabulary(self):
        move = _move(headlines=[{
            "source": "et", "headline": "h", "url": "", "sentiment_label": "positive",
            "relevance": 0.9, "summary": "",
        }])
        text = " ".join(move_narrative(move, "Test Co"))
        for pattern in BORROWED_STATS_WORDS:
            assert not re.search(pattern, text, re.I), \
                f"borrowed statistical term {pattern!r} in: {text!r}"

    def test_coverage_paragraph_states_no_significance_test_is_run(self):
        move = _move(headlines=[{
            "source": "et", "headline": "h", "url": "", "sentiment_label": "positive",
            "relevance": 0.9, "summary": "",
        }])
        text = " ".join(move_narrative(move, "Test Co"))
        assert "no significance test is computed here" in text

    def test_no_coverage_is_disclosed_as_a_gap_not_hidden(self):
        text = " ".join(move_narrative(_move(headlines=[]), "Test Co"))
        assert "No collected coverage" in text
        assert "not evidence that nothing happened" in text

    def test_falling_price_uses_fell(self):
        text = " ".join(move_narrative(_move(end_price=80.0), "Test Co"))
        assert "fell from" in text

    def test_unchanged_price_uses_held_steady(self):
        text = " ".join(move_narrative(_move(end_price=100.0), "Test Co"))
        assert "held steady" in text


class TestUnlistedSummaryNarrative:
    def test_states_this_is_not_an_event_study(self):
        text = " ".join(unlisted_summary_narrative(
            "Test Co", date(2026, 1, 1), date(2026, 7, 31), [], 212, 0))
        assert "not a statistically validated event study" in text

    def test_no_moves_is_explained_not_silently_empty(self):
        text = " ".join(unlisted_summary_narrative(
            "Test Co", date(2026, 1, 1), date(2026, 7, 31), [], 212, 0))
        assert "did not contain enough revisions" in text

    def test_warns_that_large_moves_can_be_corporate_actions_not_sentiment(self):
        text = " ".join(unlisted_summary_narrative(
            "Test Co", date(2026, 1, 1), date(2026, 7, 31), [_move()], 212, 3))
        assert "bonus issues, stock splits, or rights issues" in text

    def test_reports_the_largest_move_and_revision_count(self):
        moves = [
            _move(start_price=100.0, end_price=120.0),
            _move(start_date=date(2026, 1, 15), end_date=date(2026, 2, 1),
                  start_price=120.0, end_price=90.0),
        ]
        text = " ".join(unlisted_summary_narrative(
            "Test Co", date(2026, 1, 1), date(2026, 7, 31), moves, 212, 5))
        assert "revised 3 times" in text
        assert "2 gaps" in text
        assert "-25.00%" in text  # the larger of +20% and -25% by magnitude

    def test_borrowed_statistical_vocabulary_only_appears_as_an_explicit_disclaimer(self):
        """Unlike move_narrative(), the summary is allowed to name z-score/
        beta/permutation - but only to explain that none of them are
        computed, never to state one as if it were."""
        moves = [_move()]
        text = " ".join(unlisted_summary_narrative(
            "Test Co", date(2026, 1, 1), date(2026, 7, 31), moves, 212, 3))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if re.search(r"\bno\b|\bnot computed\b|\bunlike\b", sentence, re.I):
                continue
            for pattern in BORROWED_STATS_WORDS:
                assert not re.search(pattern, sentence, re.I), \
                    f"borrowed statistical term {pattern!r} asserted in: {sentence!r}"
