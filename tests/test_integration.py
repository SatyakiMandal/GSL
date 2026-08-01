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


def run_pipeline(price_dir: Path, start=date(2023, 1, 20), end=date(2023, 2, 3)):
    config = RunConfig(company="Testco", ticker="TEST.NS", benchmark="^NSEI",
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
    analysis = analyse(config, kept, {"stats": {"unique_after_dedupe": len(kept)}},
                       providers=[CsvProvider(price_dir)], return_threshold=1.5)
    return analysis, kept, dropped


class TestEndToEnd:
    def test_pipeline_runs_and_flags_the_shock_days(self, prices):
        analysis, kept, _ = run_pipeline(prices)
        flagged = {i.day for i in analysis.incidents}
        assert date(2023, 1, 25) in flagged
        assert date(2023, 1, 27) in flagged

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
        analysis = analyse(config, [orphan], {}, providers=[CsvProvider(prices)])
        assert len(analysis.unattributed) == 1
        assert "could not be placed on a trading day" in build_html(analysis)
