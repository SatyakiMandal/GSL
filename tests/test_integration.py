"""End-to-end pipeline test.

Both bugs that shipped in Phase 2 were invisible to unit tests and only appeared
when the whole pipeline ran together: the trading calendar stopping at the
analysis window (so after-close news on the last day went unattributed), and the
coverage baseline being incoherent on short windows (so nothing flagged at all
against a −16σ move). This test wires the real modules together — extraction,
relevance, dedupe, attribution, aggregation, ranking, reporting — against
synthetic HTML and prices, so that class of bug fails here rather than in a run.

Sentiment is stubbed. FinBERT is a 440 MB download and is exercised separately;
what needs pinning here is the plumbing between stages.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia import align, dedupe, relevance  # noqa: E402
from ceia.analyze import analyse  # noqa: E402
from ceia.extract import parse_article  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402
from ceia.macro import SkippedFetcher  # noqa: E402
from ceia.prices import CsvProvider  # noqa: E402
from ceia.report import build_html  # noqa: E402

# Real 2023 NSE sessions around the window, with 26 January (Republic Day)
# absent because the exchange was shut.
SESSIONS = [
    date(2023, 1, 20), date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25),
    date(2023, 1, 27), date(2023, 1, 30), date(2023, 1, 31), date(2023, 2, 1),
    date(2023, 2, 2), date(2023, 2, 3),
]


def article_html(headline: str, published: str, body: str, *,
                 style: str = "jsonld") -> str:
    """Reproduce each source's real metadata shape."""
    if style == "jsonld":
        return f"""<html><head><script type="application/ld+json">
        {{"@type":"NewsArticle","headline":{headline!r},
          "datePublished":"{published}","articleBody":{body!r}}}
        </script></head><body></body></html>""".replace("'", '"')
    if style == "graph":  # Moneycontrol nests it inside @graph
        return f"""<html><head><script type="application/ld+json">
        {{"@graph":[{{"@type":"WebPage"}},
          {{"@type":"NewsArticle","headline":{headline!r},
            "datePublished":"{published}","articleBody":{body!r}}}]}}
        </script></head><body></body></html>""".replace("'", '"')
    # Business Line: meta tags only, no JSON-LD at all.
    return (f'<html><head><meta property="article:published_time" '
            f'content="{published}"/><meta property="og:title" content="{headline}"/>'
            f'</head><body><div class="articlebodycontent"><p>{body}</p>'
            f"</div></body></html>")


@pytest.fixture
def prices(tmp_path: Path) -> Path:
    """Company with a large idiosyncratic fall on 25 and 27 January."""
    rng = np.random.default_rng(11)
    lead = [date(2023, 1, 20) - timedelta(days=i) for i in range(220, 0, -1)]
    lead = [d for d in lead if d.weekday() < 5]
    days = lead + SESSIONS
    market = rng.normal(0.0003, 0.007, len(days))
    company = 0.0004 + 1.2 * market + rng.normal(0, 0.008, len(days))
    index = {d: i for i, d in enumerate(days)}
    company[index[date(2023, 1, 25)]] += -0.16
    company[index[date(2023, 1, 27)]] += -0.20

    pd.DataFrame({"date": pd.to_datetime(days),
                  "close": 100 * np.cumprod(1 + company)}
                 ).to_csv(tmp_path / "TEST.NS.csv", index=False)
    pd.DataFrame({"date": pd.to_datetime(days),
                  "close": 100 * np.cumprod(1 + market)}
                 ).to_csv(tmp_path / "_idx_NSEI.csv", index=False)
    peer = 0.0002 + 0.9 * market + rng.normal(0, 0.006, len(days))
    pd.DataFrame({"date": pd.to_datetime(days),
                  "close": 100 * np.cumprod(1 + peer)}
                 ).to_csv(tmp_path / "PEER.NS.csv", index=False)
    # Only one of the six Nifty sector indices gets a CSV fixture - the rest
    # must degrade gracefully (see TestNiftyIndices), same as a bad/missing
    # secondary-benchmark ticker already does elsewhere in this file.
    bank = 0.00025 + 1.1 * market + rng.normal(0, 0.007, len(days))
    pd.DataFrame({"date": pd.to_datetime(days),
                  "close": 100 * np.cumprod(1 + bank)}
                 ).to_csv(tmp_path / "_idx_NSEBANK.csv", index=False)
    return tmp_path


def build_items() -> list[NewsItem]:
    """Parse synthetic pages the way ingestion would, then score them."""
    pages = [
        ("economic_times", "jsonld", "Testco shares tank after short-seller report",
         "2023-01-25T11:20:00+05:30", "Testco Ltd fell sharply after a report. " * 12),
        ("financial_express", "jsonld", "Testco shares tank after short-seller report",
         "2023-01-25T12:05:00+05:30", "Testco Ltd fell sharply after a report. " * 4),
        ("business_line", "meta", "Testco rejects allegations in investor note",
         "2023-01-25T20:57:00+05:30", "Testco Ltd said the claims were baseless. " * 12),
        ("moneycontrol", "graph", "Regulator reviews Testco disclosures",
         "2023-01-27T14:10:00+05:30", "The regulator is reviewing Testco Ltd. " * 12),
        ("economic_times", "jsonld", "Testco bonds slide as lenders seek collateral",
         "2023-01-27T18:40:00+05:30", "Lenders asked Testco Ltd for more collateral. " * 12),
        ("economic_times", "jsonld", "Top gainers and losers: Infy, TCS, Testco in focus",
         "2023-01-30T09:30:00+05:30", "Markets were mixed. " * 20 + "Testco Ltd was flat."),
        ("business_line", "meta", "Monsoon forecast lifts farm stocks",
         "2023-01-31T10:00:00+05:30", "Rainfall is expected to be normal. " * 12),
    ]
    items = []
    for source, style, headline, published, body in pages:
        parsed = parse_article(
            article_html(headline, published, body, style=style),
            f"https://{source}.example/{abs(hash(headline))}", source,
        )
        items.append(NewsItem(**parsed))
    return items


def run_pipeline(price_dir: Path, start=date(2023, 1, 20), end=date(2023, 2, 3),
                 benchmark2: str | None = None, skip_nifty: bool = False):
    config = RunConfig(company="Testco", ticker="TEST.NS", benchmark="^NSEI",
                       benchmark2=benchmark2,
                       start=start, end=end, aliases=["Testco Ltd", "Testco"],
                       event_window=(-1, 3))
    items = build_items()
    kept, dropped = relevance.apply(items, config.all_aliases, config.ticker,
                                    config.min_relevance)
    dedupe.deduplicate(kept)
    # Stand in for FinBERT: negative unless the headline is clearly reassuring.
    for item in kept:
        item.sentiment_score = 0.4 if "rejects" in item.headline else -0.7
        item.sentiment_label = "positive" if item.sentiment_score > 0 else "negative"
        item.event_category = "regulatory"
        # Stand in for GoEmotions: a plausible, deterministic label per
        # headline shape, so the field has something real flowing through it
        # rather than being universally blank in this test.
        if "tank" in item.headline or "slide" in item.headline:
            item.emotion_label = "fear"
        elif "rejects" in item.headline:
            item.emotion_label = "annoyance"
    analysis = analyse(config, kept, {"stats": {"unique_after_dedupe": len(kept)}},
                       providers=[CsvProvider(price_dir)], return_threshold=1.5,
                       macro_provider=CsvProvider(price_dir),
                       macro_fetcher=SkippedFetcher(),
                       skip_nifty_indices=skip_nifty)
    return analysis, kept, dropped


class TestEndToEnd:
    def test_pipeline_runs_and_flags_the_shock_days(self, prices):
        analysis, kept, _ = run_pipeline(prices)
        flagged = {i.day for i in analysis.incidents}
        assert date(2023, 1, 25) in flagged
        assert date(2023, 1, 27) in flagged


class TestSecondaryBenchmark:
    def test_not_requested_leaves_secondary_empty(self, prices):
        analysis, _, _ = run_pipeline(prices)
        assert analysis.secondary_daily is None
        assert analysis.secondary_meta == {}

    def test_available_peer_populates_secondary_daily_and_meta(self, prices):
        analysis, _, _ = run_pipeline(prices, benchmark2="PEER.NS")
        assert analysis.secondary_meta["ticker"] == "PEER.NS"
        assert analysis.secondary_meta["model"] in ("market-model", "market-adjusted")
        assert isinstance(analysis.secondary_meta["beta"], float)
        assert analysis.secondary_daily is not None
        assert not analysis.secondary_daily.empty
        assert list(analysis.secondary_daily.columns) == [
            "secondary_close", "secondary_return",
            "secondary_abnormal_return", "secondary_abnormal_return_z",
        ]
        # Same trading days as the primary daily table.
        assert set(analysis.secondary_daily.index) == set(analysis.daily.index)

    def test_primary_analysis_is_unaffected_by_requesting_a_peer(self, prices):
        """Adding --benchmark2 must not change incident detection or the
        primary abnormal-return series - it is a second, additive lens."""
        without, _, _ = run_pipeline(prices)
        with_peer, _, _ = run_pipeline(prices, benchmark2="PEER.NS")
        assert {i.day for i in without.incidents} == {i.day for i in with_peer.incidents}
        pd.testing.assert_series_equal(
            without.daily["abnormal_return"], with_peer.daily["abnormal_return"])

    def test_missing_peer_ticker_degrades_gracefully(self, prices):
        """A bad/unreachable peer ticker must not sink the whole analysis -
        only the secondary comparison is unavailable."""
        analysis, _, _ = run_pipeline(prices, benchmark2="NOSUCHTICKER.NS")
        assert analysis.secondary_daily is None
        assert analysis.secondary_meta["ticker"] == "NOSUCHTICKER.NS"
        assert "unavailable" in analysis.secondary_meta["note"]
        # Primary analysis still ran to completion.
        assert not analysis.daily.empty
        flagged = {i.day for i in analysis.incidents}
        assert date(2023, 1, 25) in flagged

    def test_html_report_renders_with_secondary_benchmark(self, prices):
        analysis, _, _ = run_pipeline(prices, benchmark2="PEER.NS")
        html = build_html(analysis)
        assert "PEER.NS" in html


class TestNiftyIndices:
    """The professor's Nifty-index request, reused as reference lines/stats
    rather than a second event study - see ceia/nifty.py's module docstring."""

    def test_all_six_indices_are_attempted(self, prices):
        analysis, _, _ = run_pipeline(prices)
        assert set(analysis.nifty_indices) == {
            "Nifty 50", "Nifty Bank", "Nifty Auto", "Nifty Energy",
            "Nifty IT", "Nifty Metal",
        }

    def test_available_index_populates_daily_and_window_return(self, prices):
        analysis, _, _ = run_pipeline(prices)
        nifty50 = analysis.nifty_indices["Nifty 50"]
        bank = analysis.nifty_indices["Nifty Bank"]
        for index in (nifty50, bank):
            assert index.available
            assert not index.daily.empty
            assert index.window_return == index.window_return  # not NaN

    def test_missing_index_csv_degrades_without_sinking_the_run(self, prices):
        """Four of the six indices have no CSV fixture - each must degrade
        to its own note, exactly like a bad secondary-benchmark ticker does,
        never raise or blank out the primary analysis."""
        analysis, _, _ = run_pipeline(prices)
        auto = analysis.nifty_indices["Nifty Auto"]
        assert not auto.available
        assert "unavailable" in auto.note
        assert not analysis.daily.empty
        assert date(2023, 1, 25) in {i.day for i in analysis.incidents}

    def test_skip_nifty_indices_leaves_it_empty(self, prices):
        analysis, _, _ = run_pipeline(prices, skip_nifty=True)
        assert analysis.nifty_indices == {}

    def test_html_report_renders_the_nifty_section(self, prices):
        analysis, _, _ = run_pipeline(prices)
        html = build_html(analysis)
        assert "Nifty 50" in html
        assert "Nifty Bank" in html
        assert "Nifty Auto" in html  # shown even though unavailable

    def test_json_payload_is_serialisable_with_nifty_indices(self, prices):
        import json
        analysis, _, _ = run_pipeline(prices)
        json.dumps(analysis.to_dict(), default=str)

    def test_available_index_gets_its_own_event_study_on_candidate_days(self, prices):
        """Nifty Bank has enough lead-in history (the same ~220-day window
        used to fit the company's own market model) to get a real event
        study, anchored to the days already flagged for the company."""
        analysis, _, _ = run_pipeline(prices)
        flagged = {i.day for i in analysis.incidents}
        assert date(2023, 1, 25) in flagged and date(2023, 1, 27) in flagged
        bank = analysis.nifty_indices["Nifty Bank"]
        assert bank.model_kind in ("market-model", "market-adjusted")
        for day in flagged:
            assert day.isoformat() in bank.incident_stats
            stats = bank.incident_stats[day.isoformat()]
            assert set(stats) >= {"abnormal_return", "abnormal_return_z", "car",
                                  "t_stat", "p_value", "p_value_t"}

    def test_index_matching_the_benchmark_gets_no_event_study(self, prices):
        """The company's benchmark is ^NSEI, which is also the Nifty 50
        ticker - regressing that index on itself would be meaningless."""
        analysis, _, _ = run_pipeline(prices)
        nifty50 = analysis.nifty_indices["Nifty 50"]
        assert nifty50.available
        assert nifty50.incident_stats == {}
        assert "same ticker as the primary benchmark" in nifty50.event_study_note

    def test_html_report_renders_the_per_incident_index_breakdown(self, prices):
        analysis, _, _ = run_pipeline(prices)
        html = build_html(analysis)
        assert "How the Nifty indices moved on this same day" in html

    def test_irrelevant_story_is_dropped(self, prices):
        _, kept, dropped = run_pipeline(prices)
        assert any("Monsoon" in i.headline for i in dropped)
        assert not any("Monsoon" in i.headline for i in kept)

    def test_roundup_is_downweighted(self, prices):
        _, kept, dropped = run_pipeline(prices)
        roundup = next((i for i in kept + dropped if "Top gainers" in i.headline), None)
        assert roundup is not None
        subject = next(i for i in kept if "tank" in i.headline)
        assert roundup.relevance_score < subject.relevance_score

    def test_syndicated_copy_deduplicated(self, prices):
        _, kept, _ = run_pipeline(prices)
        tank = [i for i in kept if "tank" in i.headline]
        assert len(tank) == 2, "both outlets collected"
        assert sum(1 for i in tank if i.duplicate_of is None) == 1, "counted once"

    def test_republic_day_is_not_a_trading_day(self, prices):
        """The Phase 1 gap: a weekday fallback treats 26 Jan as tradeable."""
        analysis, _, _ = run_pipeline(prices)
        # The daily table is indexed by datetime.date, not Timestamp, so that
        # Incident.day compares equal to NewsItem.trading_day.
        assert all(isinstance(d, date) and not isinstance(d, pd.Timestamp)
                   for d in analysis.daily.index)
        assert date(2023, 1, 26) not in set(analysis.daily.index)
        assert date(2023, 1, 25) in set(analysis.daily.index)

    def test_after_close_news_rolls_to_the_next_session(self, prices):
        """20:57 on 25 Jan must land on 27 Jan, skipping the holiday."""
        _, kept, _ = run_pipeline(prices)
        evening = next(i for i in kept if "rejects" in i.headline)
        assert evening.after_close
        assert evening.trading_day == date(2023, 1, 27)

    def test_nothing_is_left_unattributed(self, prices):
        """The Phase 2 tail-buffer bug showed up exactly here."""
        analysis, _, _ = run_pipeline(prices)
        assert analysis.unattributed == []

    def test_car_extends_past_the_window_end(self, prices):
        """Needs the tail buffer; otherwise every late CAR comes back truncated."""
        analysis, _, _ = run_pipeline(prices, end=date(2023, 2, 1))
        late = [i for i in analysis.incidents if i.day >= date(2023, 1, 27)]
        assert late
        assert not late[0].car["truncated"]

    def test_market_model_was_fitted(self, prices):
        analysis, _, _ = run_pipeline(prices)
        assert analysis.price_meta["model"] == "market-model"
        assert 0.8 < analysis.price_meta["beta"] < 1.6

    def test_abnormal_return_is_not_the_raw_return(self, prices):
        analysis, _, _ = run_pipeline(prices)
        row = analysis.daily.loc[date(2023, 1, 25)]
        assert row["abnormal_return"] != pytest.approx(row["return"])
        assert row["abnormal_return"] < -0.10

    def test_report_renders_from_a_real_analysis(self, prices):
        analysis, _, _ = run_pipeline(prices)
        html = build_html(analysis)
        assert "<svg" in html
        assert "coincided with" in html
        assert 'class="flagged"' in html
        assert "incident-rule" in html

    def test_emotion_reaches_the_incident_and_the_rendered_report(self, prices):
        """Full plumbing check: extraction stub -> aggregate_by_day's mode ->
        rank_incidents' Incident.dominant_emotion -> the HTML report. The
        Phase 2 bugs this file exists to catch were exactly this shape of
        failure - correct in isolation, silently dropped somewhere in the
        chain between stages."""
        analysis, kept, _ = run_pipeline(prices)
        top_25_jan = next(i for i in analysis.incidents if i.day == date(2023, 1, 25))
        assert top_25_jan.dominant_emotion == "fear"
        html = build_html(analysis)
        assert "emotion: fear" in html
        assert "apprehension" in html  # the narrative's phrasing for "fear"

    def test_json_payload_is_serialisable(self, prices):
        import json
        analysis, _, _ = run_pipeline(prices)
        assert json.loads(json.dumps(analysis.to_dict(), default=str))


class TestDegradation:
    """A broken source must not take the run down (PRD Section 10)."""

    def test_missing_prices_raise_a_typed_error(self, tmp_path):
        from ceia.prices import PriceError
        config = RunConfig(company="Testco", ticker="TEST.NS", benchmark="^NSEI",
                           start=date(2023, 1, 20), end=date(2023, 2, 3))
        with pytest.raises(PriceError):
            analyse(config, [], {}, providers=[CsvProvider(tmp_path)])

    def test_unreadable_article_does_not_crash_extraction(self):
        parsed = parse_article("<html><body>not an article</body></html>",
                               "https://x/y", "economic_times")
        assert parsed["published_at"] is None
        assert parsed["timestamp_confidence"] == "missing"

    def test_item_without_timestamp_is_reported_not_guessed(self, prices):
        orphan = NewsItem(source="economic_times", url="https://x/orphan",
                          headline="Testco Ltd faces scrutiny", body="Testco Ltd " * 40)
        config = RunConfig(company="Testco", ticker="TEST.NS", benchmark="^NSEI",
                           start=date(2023, 1, 20), end=date(2023, 2, 3),
                           aliases=["Testco Ltd"])
        analysis = analyse(config, [orphan], {}, providers=[CsvProvider(prices)],
                          macro_provider=CsvProvider(prices),
                          macro_fetcher=SkippedFetcher())
        assert len(analysis.unattributed) == 1
        assert "could not be placed on a trading day" in build_html(analysis)
