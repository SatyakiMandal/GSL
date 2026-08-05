"""Tests for template-based narrative generation.

Covers the pieces that are not exercised end-to-end by test_report.py:
_drop_explanation() (why the rest of the window's days aren't candidates),
_significance_paragraph() (how much to trust the top candidate, without
overselling the z-score as a normal-tail probability), the per-category
mechanism paragraph in incident_narrative(), and the enriched opening of
summary_narrative().
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.eventstudy import Incident  # noqa: E402
from ceia.narrative import (  # noqa: E402
    _drop_explanation,
    _significance_paragraph,
    incident_narrative,
    summary_narrative,
)


def make_incident(**kwargs) -> Incident:
    defaults = dict(
        day=date(2023, 1, 25), abnormal_return=-0.1323, abnormal_return_z=-12.53,
        raw_return=-0.1379, benchmark_return=-0.0049, coverage_z=1.2,
        sentiment_z=-1.1, item_count=3, mean_sentiment=-0.42,
        dominant_event="regulatory", dominant_emotion="fear",
        volume=8_200_000.0, volume_z=2.4, score=19.1,
        direction_agrees=True,
        car={"car": -0.3034, "days": 5, "start": "2023-01-24",
             "end": "2023-01-30", "t_stat": -12.84, "truncated": False, "note": ""},
        headlines=[], sources=["business_line", "financial_express"],
    )
    defaults.update(kwargs)
    return Incident(**defaults)


class TestIncidentMechanismParagraph:
    """incident_narrative()'s new paragraph explaining *why* a category of
    news is the kind that plausibly moves a price, distinct from claiming
    this specific article did."""

    def test_known_category_gets_a_mechanism_paragraph(self):
        incident = make_incident(dominant_event="regulatory")
        paragraphs = incident_narrative(incident, "Testco", "*NSEI", (-1, 3))
        assert any("tail risk" in p for p in paragraphs)

    def test_mechanism_paragraph_is_hedged_not_a_claim_about_this_article(self):
        incident = make_incident(dominant_event="earnings")
        paragraphs = incident_narrative(incident, "Testco", "*NSEI", (-1, 3))
        mechanism = next(p for p in paragraphs if "near-term profitability" in p)
        # It describes the category's general mechanism, not this incident.
        assert "Testco" not in mechanism

    def test_other_category_still_gets_a_hedge_paragraph(self):
        incident = make_incident(dominant_event="other")
        paragraphs = incident_narrative(incident, "Testco", "*NSEI", (-1, 3))
        assert any("did not match one of the categories above" in p for p in paragraphs)

    def test_untagged_category_gets_no_mechanism_paragraph(self):
        """dominant_event == "" (nothing tagged at all) is distinct from
        "other" (tagged, but none of the known mechanisms fit)."""
        tagged = incident_narrative(make_incident(dominant_event="other"),
                                    "Testco", "*NSEI", (-1, 3))
        untagged = incident_narrative(make_incident(dominant_event=""),
                                      "Testco", "*NSEI", (-1, 3))
        assert len(untagged) == len(tagged) - 1

    def test_every_tag_event_category_has_a_mechanism_entry(self):
        """ceia.sentiment.tag_event()'s categories must all be covered, or a
        real report would silently drop the paragraph for that category."""
        from ceia.narrative import _EVENT_MECHANISM
        for category in ("earnings", "regulatory", "leadership", "litigation",
                         "mna", "capital", "product", "macro", "other"):
            assert category in _EVENT_MECHANISM


class TestDropExplanation:
    def test_no_diagnostics_returns_empty_string(self):
        assert _drop_explanation(None, candidates=1) == ""

    def test_diagnostics_missing_trading_days_returns_empty_string(self):
        assert _drop_explanation({}, candidates=0) == ""

    def test_nothing_dropped_returns_empty_string(self):
        diag = {"trading_days": 3, "candidates": 3, "coverage_only": 0,
                "return_only": 0, "no_coverage_big_move": 0, "routine": 0}
        assert _drop_explanation(diag, candidates=3) == ""

    def test_reports_real_counts_for_each_bucket(self):
        diag = {
            "trading_days": 15, "candidates": 1, "coverage_only": 1,
            "return_only": 1, "no_coverage_big_move": 1, "routine": 11,
        }
        text = _drop_explanation(diag, candidates=1)
        assert "14 trading days" in text
        assert "1 day had busy or strongly-toned coverage without a matching" in text
        assert "1 day moved unusually without correspondingly unusual coverage" in text
        assert "11 days were routine on both counts" in text

    def test_no_coverage_big_move_is_flagged_as_a_real_gap(self):
        diag = {
            "trading_days": 2, "candidates": 0, "coverage_only": 0,
            "return_only": 0, "no_coverage_big_move": 1, "routine": 1,
        }
        text = _drop_explanation(diag, candidates=0)
        assert "<strong>no coverage collected for it at all</strong>" in text
        assert "a real gap, not a finding" in text

    def test_explains_the_two_bar_rule(self):
        diag = {
            "trading_days": 2, "candidates": 0, "coverage_only": 1,
            "return_only": 0, "no_coverage_big_move": 0, "routine": 1,
        }
        text = _drop_explanation(diag, candidates=0)
        assert "unusual coverage <em>and</em> an unusual price move" in text


class TestSignificanceParagraph:
    def test_states_the_z_score_magnitude(self):
        top = make_incident(abnormal_return_z=-12.53, car={})
        text = _significance_paragraph(top, robustness=None)
        assert "12.5 standard deviations" in text

    def test_never_states_a_normal_tail_probability(self):
        """The z-score's normality assumption is exactly what the
        permutation test exists to sidestep - restating it as "1 in X" here
        would oversell a number the rest of the codebase declines to trust."""
        top = make_incident(abnormal_return_z=-12.53, car={})
        text = _significance_paragraph(top, robustness=None)
        assert "1 in" not in text
        assert "chance" not in text

    def test_permutation_clause_appears_when_p_value_present(self):
        top = make_incident(car={"p_value": 0.007, "n": 5000})
        text = _significance_paragraph(top, robustness=None)
        assert "permutation test" in text
        assert "5,000 random comparable-length windows" in text
        assert "0.7%" in text
        assert "p = 0.007" in text

    def test_permutation_clause_absent_when_p_value_is_none(self):
        top = make_incident(car={"p_value": None, "n": 0})
        text = _significance_paragraph(top, robustness=None)
        assert "permutation test" not in text

    def test_robustness_clause_appears_when_data_present_for_that_day(self):
        top = make_incident(day=date(2023, 1, 25), car={})
        robustness = {"days": {"2023-01-25": {"flagged_in": 7, "of": 9}}}
        text = _significance_paragraph(top, robustness)
        assert "7 of 9 combinations" in text

    def test_robustness_clause_absent_when_day_not_in_robustness_data(self):
        top = make_incident(day=date(2023, 1, 25), car={})
        robustness = {"days": {"2023-02-01": {"flagged_in": 9, "of": 9}}}
        text = _significance_paragraph(top, robustness)
        assert "combinations" not in text

    def test_robustness_clause_absent_when_robustness_is_none(self):
        top = make_incident(day=date(2023, 1, 25), car={})
        assert "combinations" not in _significance_paragraph(top, None)


class TestSummaryNarrativeOpening:
    """The non-causal disclaimer moved here from the deleted "what this
    report is, and is not" box - it must survive verbatim, since it's the
    single most load-bearing integrity statement in the report."""

    def test_states_non_causation_in_the_opening_paragraph(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-model")
        assert "coincided with" in paragraphs[0]
        assert "Coincidence in time is not evidence" in paragraphs[0]

    def test_market_model_wording_used_when_a_beta_was_fitted(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-model")
        assert "market model fitted on price history" in paragraphs[1]

    def test_market_adjusted_wording_used_when_beta_was_assumed(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-adjusted")
        assert "assumed to move one-for-one with the index" in paragraphs[1]

    def test_thin_baseline_weight_note_appears(self):
        diagnostics = {"trading_days": 5, "days_with_news": 2, "thin_baseline": True,
                       "candidates": 0, "coverage_only": 0, "return_only": 0,
                       "no_coverage_big_move": 0, "routine": 5}
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=5, news_count=2, model_kind="market-model",
            diagnostics=diagnostics)
        weight_para = next(p for p in paragraphs if p.startswith("How much weight"))
        assert "too few to judge which days were unusually" in weight_para

    def test_weak_scale_note_appears(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-model",
            weak_scale=True)
        weight_para = next(p for p in paragraphs if p.startswith("How much weight"))
        assert "understates how unusual the largest moves" in weight_para

    def test_no_weight_note_when_nothing_is_weak(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-model")
        assert not any(p.startswith("How much weight") for p in paragraphs)


class TestSummaryNarrativeNoIncidents:
    def test_states_no_day_met_both_tests(self):
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=20, news_count=10, model_kind="market-model")
        assert any("no day met both tests" in p for p in paragraphs)

    def test_drop_explanation_included_when_diagnostics_show_dropped_days(self):
        diagnostics = {"trading_days": 5, "days_with_news": 5, "thin_baseline": False,
                       "candidates": 0, "coverage_only": 2, "return_only": 1,
                       "no_coverage_big_move": 0, "routine": 2}
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [], daily_rows=5, news_count=10, model_kind="market-model",
            diagnostics=diagnostics)
        assert any("did not clear both bars at once" in p for p in paragraphs)


class TestSummaryNarrativeWithIncidents:
    def test_reports_candidate_count_and_top_day(self):
        top = make_incident(day=date(2023, 1, 27), abnormal_return=-0.15, car={})
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [top], daily_rows=20, news_count=10, model_kind="market-model")
        assert any("1 candidate day" in p for p in paragraphs)
        assert any("27 January 2023" in p for p in paragraphs)

    def test_significance_paragraph_is_included(self):
        top = make_incident(abnormal_return_z=-9.0, car={"p_value": 0.02, "n": 1000})
        paragraphs = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            [top], daily_rows=20, news_count=10, model_kind="market-model")
        assert any("permutation test" in p for p in paragraphs)

    def test_agreement_sentence_only_appears_with_multiple_incidents(self):
        one = [make_incident(day=date(2023, 1, 27), car={})]
        two = [make_incident(day=date(2023, 1, 27), car={}, direction_agrees=True),
               make_incident(day=date(2023, 1, 30), car={}, direction_agrees=False)]
        single_paras = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            one, daily_rows=20, news_count=10, model_kind="market-model")
        multi_paras = summary_narrative(
            "Testco", "TEST.NS", "*NSEI", date(2023, 1, 1), date(2023, 2, 1),
            two, daily_rows=20, news_count=10, model_kind="market-model")
        assert not any("flagged days" in p for p in single_paras)
        assert any("Of the 2 flagged days, 1 showed" in p for p in multi_paras)
