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
        headlines=["[business_line] Adani shares tank (negative, rel=1.00)"],
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


class TestHtmlReport:
    def test_is_self_contained(self):
        html = build_html(make_analysis())
        assert not re.search(r'(?:src|href)="https?://', html), "external asset"
        assert "@import" not in html
        assert "<script" not in html.lower(), "no scripts needed"

    def test_limitations_appear_before_findings(self):
        html = build_html(make_analysis())
        warning = html.index("What this report is, and is not")
        findings = html.index("Candidate incident days")
        assert warning < findings, "limitations must precede findings"

    def test_states_non_causation_prominently(self):
        html = build_html(make_analysis())
        assert "coincided with" in html
        assert "not evidence that an article caused a price move" in html
        assert "not investment advice" in html

    def test_caveats_are_rendered(self):
        html = build_html(make_analysis())
        assert "structured case study" in html

    def test_escapes_hostile_headline_text(self):
        """Headlines come from scraped pages and are untrusted input."""
        nasty = '<img src=x onerror="alert(1)"> & "quoted"'
        analysis = make_analysis(incidents=[make_incident(headlines=[nasty])])
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

    def test_disabled_sources_reported(self):
        html = build_html(make_analysis())
        assert "business_standard" in html
        assert "DISABLED" in html

    def test_after_close_count_explained(self):
        html = build_html(make_analysis())
        assert "15:30 IST close" in html
        assert "13 of the collected items fell after the close" in html

    def test_finbert_disclosed(self):
        assert "FinBERT" in build_html(make_analysis())

    def test_goemotions_disclosed_as_secondary_and_not_finance_tuned(self):
        html = build_html(make_analysis())
        assert "GoEmotions" in html
        assert "not</strong> a\nfinance-tuned model" in html or \
            "not</strong> a finance-tuned model" in html
        assert "never affects relevance, incident\nflagging" in html or \
            "never affects relevance, incident flagging" in html

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


class TestModelNotePunctuation:
    """model_note values from returns.py have no terminal punctuation, and ran
    straight into "Prices came from..." with no separator. A real user's
    report showed: "fitted on 120 trading days before 2023-01-20 Prices came
    from yfinance..." with no full stop between them.
    """

    def test_period_added_when_missing(self):
        from ceia.report import _sentence
        assert _sentence("fitted on 120 trading days before 2023-01-20") == \
            "fitted on 120 trading days before 2023-01-20."

    def test_existing_punctuation_not_doubled(self):
        from ceia.report import _sentence
        assert _sentence("already ends with a period.") == "already ends with a period."
        assert _sentence("ends with a question?") == "ends with a question?"

    def test_appears_correctly_in_the_report(self):
        html = build_html(make_analysis())
        assert "days\nPrices came from" not in html  # old bug shape: no period
        assert "fitted on 120 days.\nPrices came from" in html
