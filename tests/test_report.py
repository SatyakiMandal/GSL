"""Tests for narrative generation and HTML reporting.

The important assertions here are not about layout. They are that the report
cannot claim causation, cannot leak unescaped source text into the page, and
cannot quietly drop the limitations — those are the properties the PRD's
Sections 4 and 9 actually require of the output.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.charts import timeline_svg  # noqa: E402
from ceia.eventstudy import Incident  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402
from ceia.narrative import incident_narrative, summary_narrative  # noqa: E402
from ceia.report import build_html  # noqa: E402

# Words the report must never use about the relationship between coverage and
# a price move. The whole methodological claim is "coincided with".
CAUSAL_WORDS = [
    r"\bcaused\b", r"\bcausing\b", r"\bdrove\b", r"\btriggered\b",
    r"\bsent the stock\b", r"\bled to a\b", r"\bresulted in a\b",
    r"\bbecause of the (?:news|coverage|report)\b",
]


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
        headlines=[{
            "source": "business_line", "headline": "Adani shares tank",
            "url": "https://www.thehindubusinessline.com/adani-shares-tank/",
            "sentiment_label": "negative", "relevance": 1.00,
            "summary": "Adani group shares fell sharply after the report's allegations.",
        }],
        sources=["business_line", "financial_express"],
    )
    defaults.update(kwargs)
    return Incident(**defaults)


@dataclass
class FakeAnalysis:
    config: RunConfig
    daily: pd.DataFrame
    incidents: list
    price_meta: dict
    news_meta: dict
    caveats: list = field(default_factory=list)
    unattributed: list = field(default_factory=list)


def make_analysis(incidents=None, unattributed=None, daily=None) -> FakeAnalysis:
    days = pd.to_datetime(["2023-01-24", "2023-01-25", "2023-01-27"])
    frame = daily if daily is not None else pd.DataFrame({
        "close": [3400.0, 2930.0, 2400.0],
        "return": [np.nan, -0.138, -0.181],
        "benchmark_return": [np.nan, -0.005, -0.003],
        "expected_return": [np.nan, -0.006, -0.004],
        "abnormal_return": [0.0, -0.132, -0.178],
        "abnormal_return_z": [0.0, -12.5, -16.9],
        "item_count": [2, 3, 3],
        "unique_count": [2, 3, 3],
        "mean_sentiment": [-0.01, -0.22, -0.39],
        "weighted_sentiment": [-0.01, -0.22, -0.39],
        "dominant_event": ["other", "regulatory", "regulatory"],
        "sources": ["et", "et,bl", "bl"],
        "coverage_z": [-0.5, 0.3, 0.3],
        "sentiment_z": [0.9, -0.2, -0.8],
    }, index=days)
    frame.index.name = "date"
    return FakeAnalysis(
        config=RunConfig(company="Adani Enterprises", ticker="ADANIENT.NS",
                         benchmark="^NSEI", start=date(2023, 1, 24),
                         end=date(2023, 1, 28), event_window=(-1, 3)),
        daily=frame,
        incidents=incidents if incidents is not None else [make_incident()],
        price_meta={"model": "market-model", "beta": 1.2, "alpha": 0.0004,
                    "r_squared": 0.4, "company_provider": "csv",
                    "benchmark_provider": "csv", "model_note": "fitted on 120 days",
                    "ar_scale_source": "estimation-window residual SD"},
        news_meta={"stats": {"unique_after_dedupe": 24, "after_close": 13},
                   "source_status": {"economic_times": "ok: 100 candidate URLs"},
                   "disabled_sources": {"business_standard": "Akamai 403"}},
        caveats=["This is a structured case study, not a statistically validated "
                 "causal finding."],
        unattributed=unattributed or [],
    )


class TestEmotionInNarrative:
    def test_dominant_emotion_is_named_and_attributed_to_the_right_model(self):
        text = " ".join(incident_narrative(
            make_incident(dominant_emotion="fear"), "X", "^NSEI", (-1, 3)))
        assert "apprehension" in text
        assert "general-purpose emotion model" in text
        # Must not read as a second vote on the same sentiment claim.
        assert "finance-tuned sentiment score above" in text

    def test_blank_dominant_emotion_adds_nothing(self):
        """--skip-emotion runs, or a day with no confident label, must not
        fabricate an emotion sentence."""
        text = " ".join(incident_narrative(
            make_incident(dominant_emotion=""), "X", "^NSEI", (-1, 3)))
        assert "general-purpose emotion model" not in text

    def test_unknown_label_is_handled_gracefully(self):
        """Defensive: an emotion label outside the known phrase table (e.g. a
        future GoEmotions version) must not crash narrative generation."""
        text = " ".join(incident_narrative(
            make_incident(dominant_emotion="not_a_real_label"), "X", "^NSEI", (-1, 3)))
        assert "general-purpose emotion model" not in text

    def test_never_claims_causation(self):
        """The emotion clause must obey the same non-causal vocabulary rule
        as the rest of the narrative."""
        text = " ".join(incident_narrative(
            make_incident(dominant_emotion="anger"), "X", "^NSEI", (-1, 3)))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if re.search(r"\b(?:cannot|can not|not|never|no)\b", sentence, re.I):
                continue
            for pattern in CAUSAL_WORDS:
                assert not re.search(pattern, sentence, re.I), \
                    f"causal phrasing {pattern!r} in: {sentence!r}"


class TestNarrative:
    def test_separates_market_move_from_company_move(self):
        text = " ".join(incident_narrative(make_incident(), "Adani Enterprises",
                                          "^NSEI", (-1, 3)))
        assert "13.79%" in text, "raw return stated"
        assert "0.49%" in text, "benchmark move stated"
        assert "-13.23%" in text, "abnormal return stated"

    def test_never_claims_causation(self):
        """Causal verbs are allowed only inside an explicit denial.

        "cannot show that any article caused the move" is the disclaimer, not
        the claim, so the check is per sentence and skips negated ones.
        """
        for agrees in (True, False):
            text = " ".join(incident_narrative(
                make_incident(direction_agrees=agrees), "X", "^NSEI", (-1, 3)))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if re.search(r"\b(?:cannot|can not|not|never|no)\b", sentence, re.I):
                    continue
                for pattern in CAUSAL_WORDS:
                    assert not re.search(pattern, sentence, re.I), \
                        f"causal phrasing {pattern!r} asserted in: {sentence!r}"
            assert "coincidence in time" in text.lower() or "occurred together" in text.lower()

    def test_disagreement_is_explained_not_hidden(self):
        text = " ".join(incident_narrative(
            make_incident(direction_agrees=False, mean_sentiment=0.5),
            "X", "^NSEI", (-1, 3)))
        assert "opposite direction" in text
        assert "not necessarily an error" in text

    def test_car_persistence_language(self):
        extended = " ".join(incident_narrative(
            make_incident(car={"car": -0.30, "days": 5, "start": "a", "end": "b",
                               "t_stat": -1.0, "truncated": False, "note": ""}),
            "X", "^NSEI", (-1, 3)))
        assert "persisted" in extended

        reversed_ = " ".join(incident_narrative(
            make_incident(car={"car": 0.09, "days": 5, "start": "a", "end": "b",
                               "t_stat": 1.0, "truncated": False, "note": ""}),
            "X", "^NSEI", (-1, 3)))
        assert "reversed" in reversed_

    def test_truncated_window_disclosed(self):
        text = " ".join(incident_narrative(
            make_incident(car={"car": -0.1, "days": 2, "start": "a", "end": "b",
                               "t_stat": -1.0, "truncated": True, "note": "x"}),
            "X", "^NSEI", (-1, 3)))
        assert "truncated" in text

    def test_magnitude_scales_with_z_not_percent(self):
        big = " ".join(incident_narrative(make_incident(abnormal_return_z=-8.0),
                                         "X", "^NSEI", (-1, 3)))
        small = " ".join(incident_narrative(make_incident(abnormal_return_z=-0.4),
                                           "X", "^NSEI", (-1, 3)))
        assert "extraordinary" in big and "modest" in small

    def test_summary_states_the_model_assumption(self):
        text = " ".join(summary_narrative("X", "T", "^NSEI", date(2023, 1, 1),
                                         date(2023, 2, 1), [], 20, 5,
                                         "market-adjusted"))
        assert "one-for-one" in text
        assert "overstate" in text

    def test_summary_handles_no_incidents(self):
        text = " ".join(summary_narrative("X", "T", "^NSEI", date(2023, 1, 1),
                                         date(2023, 2, 1), [], 20, 5,
                                         "market-model"))
        assert "no day met both tests" in text
        assert "legitimate result" in text


class TestCharts:
    def test_empty_frame_does_not_raise(self):
        empty = pd.DataFrame(columns=["close", "benchmark_return",
                                      "abnormal_return", "unique_count",
                                      "weighted_sentiment"])
        assert "No trading days" in timeline_svg(empty, set(), "X", "^NSEI")

    def test_svg_has_all_three_panels(self):
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI")
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "rebased to 100" in svg
        assert "Abnormal return" in svg
        assert "News volume" in svg

    def test_incident_days_marked(self):
        marked = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI")
        unmarked = timeline_svg(make_analysis().daily, set(), "X", "^NSEI")
        assert marked.count("incident-rule") > unmarked.count("incident-rule")

    def test_single_row_does_not_divide_by_zero(self):
        frame = make_analysis().daily.iloc[:1]
        assert "<svg" in timeline_svg(frame, set(), "X", "^NSEI")

    def test_incidents_get_a_numbered_badge_and_tooltip(self):
        incident = make_incident(day=date(2023, 1, 25))
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI",
                           incidents=[incident])
        assert "incident-badge" in svg
        assert ">1<" in svg, "the only incident should be badge #1"
        assert "#1 25 Jan 2023" in svg

    def test_without_incidents_list_there_are_no_badges(self):
        """The plain dashed marker line still appears (incident_days alone
        drives that); only the numbered badge needs the richer incidents list."""
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI")
        assert "incident-rule" in svg
        assert "incident-badge" not in svg

    def test_incident_not_in_incidents_list_gets_no_badge(self):
        """A day flagged in incident_days but absent from incidents (a caller
        passing mismatched sets) degrades to the plain marker, not a crash."""
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 27)}, "X", "^NSEI",
                           incidents=[make_incident(day=date(2023, 1, 25))])
        assert "incident-rule" in svg
        assert "incident-badge" not in svg

    def test_baseline_100_reference_line_present(self):
        svg = timeline_svg(make_analysis().daily, set(), "X", "^NSEI")
        assert "baseline" in svg
        assert "start of window" in svg

    def test_negative_tone_bars_get_a_hatch_overlay(self):
        """Tone in panel 3 is otherwise colour-only; the hatch is a second,
        colour-independent cue for negative-tone days (fixture has two:
        weighted_sentiment -0.22 and -0.39, both <= the -0.15 tone-neg cutoff)."""
        svg = timeline_svg(make_analysis().daily, set(), "X", "^NSEI")
        assert svg.count('fill="url(#neg-hatch)"') == 2

    def test_badge_has_an_always_visible_dated_label(self):
        """SVG <title> tooltips don't render in PDF/print/screenshot contexts,
        so the date must be a real visible text element, not just a hover."""
        incident = make_incident(day=date(2023, 1, 25))
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI",
                           incidents=[incident])
        assert 'class="badge-date"' in svg
        assert 'class="badge-date-bg"' in svg
        assert ">25 Jan<" in svg

    def test_caption_explains_the_badges_when_incidents_are_present(self):
        incident = make_incident(day=date(2023, 1, 25))
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI",
                           incidents=[incident])
        assert 'class="chart-caption"' in svg
        assert "Numbered circles mark the ranked candidate incident days" in svg

    def test_no_caption_or_badge_markup_when_incidents_list_is_empty(self):
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI",
                           incidents=[])
        assert 'class="chart-caption"' not in svg
        assert 'class="badge-date"' not in svg

    def test_no_caption_or_badge_markup_when_incidents_not_passed(self):
        svg = timeline_svg(make_analysis().daily, {date(2023, 1, 25)}, "X", "^NSEI")
        assert 'class="chart-caption"' not in svg
        assert 'class="badge-date"' not in svg

    def test_multiple_badges_each_get_their_own_date_label(self):
        days = pd.to_datetime(["2023-01-24", "2023-01-25", "2023-01-27"])
        frame = make_analysis().daily
        incidents = [make_incident(day=date(2023, 1, 25)),
                    make_incident(day=date(2023, 1, 27))]
        svg = timeline_svg(frame, {date(2023, 1, 25), date(2023, 1, 27)},
                           "X", "^NSEI", incidents=incidents)
        assert ">25 Jan<" in svg
        assert ">27 Jan<" in svg
        assert svg.count('class="badge-date-bg"') == 2


class TestHtmlReport:
    def test_is_self_contained(self):
        """No external *dependency* the page needs in order to render or
        function offline (stylesheet, script, image, font). A citation link
        to a source article (<a href>) is not a rendering dependency - the
        page is still one self-contained file without it ever resolving."""
        html = build_html(make_analysis())
        assert not re.search(r'<link[^>]+href="https?://', html), "external stylesheet/font"
        assert not re.search(r'<(?:script|img)[^>]+src="https?://', html), "external script/image"
        assert "@import" not in html
        assert "<script" not in html.lower(), "no scripts needed"

    def test_permutation_p_value_renders_when_present(self):
        incident = make_incident(car={
            "car": -0.3034, "days": 5, "start": "2023-01-24", "end": "2023-01-30",
            "t_stat": -12.84, "truncated": False, "note": "",
            "p_value": 0.012, "n": 800, "p_value_note": "empirical p-value...",
        })
        html = build_html(make_analysis(incidents=[incident]))
        assert "0.012" in html
        assert "CAR permutation p=0.012" in html

    def test_missing_p_value_key_does_not_crash(self):
        """make_incident()'s default car dict predates the permutation test
        and has no p_value key; .get() must degrade to a dash, not KeyError."""
        html = build_html(make_analysis())
        assert "<svg" in html  # got all the way through without raising

    def test_robustness_column_renders_when_present(self):
        incident = make_incident(day=date(2023, 1, 25))
        analysis = make_analysis(incidents=[incident])
        analysis.robustness = {
            "n_combos": 9,
            "days": {"2023-01-25": {"flagged_in": 9, "of": 9, "fraction": 1.0}},
            "note": "each candidate day's flagging test re-run across 9 combinations...",
        }
        html = build_html(analysis)
        assert "9/9" in html
        assert "Robust" in html

    def test_missing_robustness_does_not_crash(self):
        """FakeAnalysis in these tests has no robustness attribute at all;
        getattr()'s default must keep the report rendering, not KeyError."""
        html = build_html(make_analysis())
        assert "<svg" in html

    def test_limitations_appear_before_findings(self):
        """PRD Success Metric #3: a reader must hit the non-causation
        framing before the findings, not after. It now opens the Summary
        itself (see summary_narrative) rather than living in a separate box
        a reader could skip past - so it must still land before the
        candidate-incident findings."""
        html = build_html(make_analysis())
        warning = html.index("coincided with")
        findings = html.index("Candidate incident days")
        assert warning < findings, "limitations must precede findings"

    def test_states_non_causation_prominently(self):
        html = build_html(make_analysis())
        assert "coincided with" in html
        assert "not evidence that an article caused a price move" in html
        assert "not investment advice" in html

    def test_no_longer_renders_the_removed_sections(self):
        """These two sections were deliberately removed: their content had
        already been presented elsewhere (README, this project's docs), and
        repeating a full methodology dump on every single generated report
        was judged not worth the length. This is a regression guard, not a
        preference - if either heading comes back, it should be a deliberate
        re-add, not an accidental one."""
        html = build_html(make_analysis())
        assert "What this report is, and is not" not in html
        assert "<h2>Method and provenance</h2>" not in html
        assert "<h3>Source availability</h3>" not in html

    def test_escapes_hostile_headline_text(self):
        """Headlines come from scraped pages and are untrusted input."""
        nasty = '<img src=x onerror="alert(1)"> & "quoted"'
        analysis = make_analysis(incidents=[make_incident(
            headlines=[{"headline": nasty, "source": "et", "url": "",
                       "sentiment_label": "negative", "relevance": 0.9,
                       "summary": ""}])])
        html = build_html(analysis)
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_escapes_company_name(self):
        analysis = make_analysis()
        analysis.config.company = "<b>Evil</b> Corp"
        html = build_html(analysis)
        assert "<b>Evil</b> Corp" not in html
        assert "&lt;b&gt;Evil" in html

    def test_no_incidents_renders_cleanly(self):
        html = build_html(make_analysis(incidents=[]))
        assert "No day combined notable coverage" in html
        assert "<svg" in html

    def test_unattributed_items_are_surfaced(self):
        item = NewsItem(source="et", url="u", headline="No timestamp story")
        html = build_html(make_analysis(unattributed=[item]))
        assert "could not be placed on a trading day" in html
        assert "No timestamp story" in html

    def test_after_close_count_appears_in_the_stats_grid(self):
        """The explanatory sentence lived in the now-removed Method and
        provenance section; the underlying number is still surfaced in the
        stats grid at the top of the report."""
        html = build_html(make_analysis())
        assert '<div class="k">Published after close</div><div class="v">13</div>' in html

    def test_incident_card_shows_emotion_tag_when_present(self):
        html = build_html(make_analysis(incidents=[make_incident(dominant_emotion="fear")]))
        assert "emotion: fear" in html

    def test_incident_card_omits_emotion_tag_when_blank(self):
        html = build_html(make_analysis(incidents=[make_incident(dominant_emotion="")]))
        assert "emotion:" not in html

    def test_incident_table_has_an_emotion_column(self):
        html = build_html(make_analysis(incidents=[make_incident(dominant_emotion="anger")]))
        body = html.split("<h2>Candidate incident days</h2>")[1]
        body = body.split("<h2>")[0]
        assert "Emotion" in body
        assert ">anger<" in body

    def test_emotion_tag_is_escaped(self):
        """dominant_emotion ultimately traces back to a headline; treat it as
        untrusted the same way the rest of the report does."""
        html = build_html(make_analysis(
            incidents=[make_incident(dominant_emotion='<script>evil</script>')]))
        assert "<script>evil</script>" not in html

    def test_daily_table_has_a_row_per_day(self):
        html = build_html(make_analysis())
        body = html.split("<h2>Daily detail</h2>")[1]
        assert body.count("<tr") >= 3


class TestTimestampIndexHandling:
    """pandas Timestamp subclasses datetime.date.

    An `isinstance(x, date)` guard therefore leaves Timestamps unconverted, and
    they never compare equal to the plain dates in `incident_days` — so incident
    highlighting silently disappears. Both the chart and the daily table hit
    this, so both are pinned here.
    """

    def test_daily_table_highlights_flagged_rows(self):
        html = build_html(make_analysis())
        body = html.split("<h2>Daily detail</h2>")[1]
        assert 'class="flagged"' in body, "flagged row lost its highlight"

    def test_chart_marks_incident_days_from_a_datetime_index(self):
        analysis = make_analysis()
        assert isinstance(analysis.daily.index, pd.DatetimeIndex)
        svg = timeline_svg(analysis.daily, {date(2023, 1, 25)}, "X", "^NSEI")
        assert "incident-rule" in svg
        assert "bar-incident" in svg


