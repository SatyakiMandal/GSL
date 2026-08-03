"""Tests for pipeline plumbing: slug pre-filter, interleaving, window trimming.

Also covers article extraction against saved fixtures of the real page shapes,
since each source exposes its metadata differently and a silent extraction
regression would poison everything downstream.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.discovery import Candidate  # noqa: E402
from ceia.extract import IST, clean_headline, json_ld_articles, parse_article  # noqa: E402
from ceia.ingest import _slug_tokens, cap_across_range, in_range, interleave, prefilter  # noqa: E402
from ceia.models import NewsItem  # noqa: E402


class TestSlugPrefilter:
    def test_tokens_drop_corporate_suffixes(self):
        tokens = _slug_tokens(["Adani Enterprises Ltd"], "ADANIENT.NS")
        assert "adani" in tokens and "enterprises" in tokens
        assert "ltd" not in tokens
        assert "adanient" in tokens

    def test_keeps_matching_slugs_only(self):
        candidates = [
            Candidate("https://x.com/news/adani-shares-fall-123.html", "et"),
            Candidate("https://x.com/news/infosys-wins-deal-456.html", "et"),
        ]
        kept = prefilter(candidates, {"adani"})
        assert len(kept) == 1 and "adani" in kept[0].url

    def test_empty_tokens_keeps_everything(self):
        candidates = [Candidate("https://x.com/a", "et")]
        assert prefilter(candidates, set()) == candidates


class TestInterleave:
    def test_round_robins_across_sources(self):
        candidates = (
            [Candidate(f"et{i}", "economic_times") for i in range(3)]
            + [Candidate(f"bl{i}", "business_line") for i in range(3)]
        )
        ordered = [c.url for c in interleave(candidates)]
        assert ordered[:2] == ["et0", "bl0"]
        assert len(ordered) == 6

    def test_a_limit_samples_every_source(self):
        """The bug this exists to prevent: one source eating the whole budget."""
        candidates = (
            [Candidate(f"et{i}", "economic_times") for i in range(50)]
            + [Candidate(f"mc{i}", "moneycontrol") for i in range(50)]
        )
        first_ten = interleave(candidates)[:10]
        assert len({c.source for c in first_ten}) == 2

    def test_uneven_sources_do_not_lose_items(self):
        candidates = ([Candidate("a", "s1")]
                      + [Candidate(f"b{i}", "s2") for i in range(4)])
        assert len(interleave(candidates)) == 5

    def test_empty_input(self):
        assert interleave([]) == []


class TestCapAcrossRange:
    """The bug this exists to prevent: a --limit run over a wide date range
    silently truncating to just its earliest days, because each source's own
    candidates arrive in roughly chronological order and interleave() only
    fixes bias across sources, not across time."""

    def test_under_the_limit_is_unchanged(self):
        candidates = [Candidate(f"a{i}", "s") for i in range(5)]
        assert cap_across_range(candidates, 10) == candidates

    def test_caps_to_exactly_the_limit(self):
        candidates = [Candidate(f"a{i}", "s") for i in range(1000)]
        assert len(cap_across_range(candidates, 100)) == 100

    def test_late_candidates_survive_a_tight_cap(self):
        """The whole point: index 999 must not be silently dropped just
        because it comes last in an interleaved, roughly chronological list."""
        candidates = [Candidate(f"a{i}", "s") for i in range(1000)]
        capped_urls = {c.url for c in cap_across_range(candidates, 100)}
        assert "a999" in capped_urls or "a0" in capped_urls
        # Stronger: the selection spans the full index range, not just one end.
        indices = sorted(int(u[1:]) for u in capped_urls)
        assert indices[0] < 100
        assert indices[-1] > 900

    def test_order_is_preserved(self):
        candidates = [Candidate(f"a{i}", "s") for i in range(1000)]
        indices = [int(c.url[1:]) for c in cap_across_range(candidates, 50)]
        assert indices == sorted(indices)


class TestWindowTrimming:
    def _item(self, when):
        return NewsItem(source="et", url=str(when), headline="h", published_at=when)

    def test_trims_outside_the_window(self):
        items = [
            self._item(datetime(2023, 1, 25, 12, 0, tzinfo=IST)),
            self._item(datetime(2023, 2, 10, 12, 0, tzinfo=IST)),
        ]
        kept = in_range(items, date(2023, 1, 24), date(2023, 1, 28))
        assert len(kept) == 1

    def test_boundaries_inclusive(self):
        items = [self._item(datetime(2023, 1, 24, 0, 5, tzinfo=IST)),
                 self._item(datetime(2023, 1, 28, 23, 55, tzinfo=IST))]
        assert len(in_range(items, date(2023, 1, 24), date(2023, 1, 28))) == 2

    def test_missing_timestamp_is_kept_not_dropped(self):
        """Dropping these would hide a gap instead of reporting it."""
        item = NewsItem(source="et", url="u", headline="h", published_at=None)
        assert in_range([item], date(2023, 1, 24), date(2023, 1, 28)) == [item]


class TestHeadlineCleaning:
    def test_strips_outlet_suffix(self):
        assert clean_headline("Adani news- Moneycontrol.com") == "Adani news"
        assert clean_headline("Adani news | The Economic Times") == "Adani news"
        assert clean_headline("Adani news - Financial Express") == "Adani news"
        assert clean_headline("Adani news – businessline") == "Adani news"

    def test_leaves_ordinary_dashes_alone(self):
        assert clean_headline("Adani vs Hindenburg - a brief story") == \
            "Adani vs Hindenburg - a brief story"


class TestJsonLdExtraction:
    def test_finds_article_nested_in_graph(self):
        """Moneycontrol nests NewsArticle inside @graph rather than at top level."""
        html = """
        <script type="application/ld+json">
        {"@context":"https://schema.org","@graph":[
          {"@type":"WebPage","name":"x"},
          {"@type":"NewsArticle","headline":"Nested headline",
           "datePublished":"2023-01-25T20:57:54+05:30"}]}
        </script>"""
        found = json_ld_articles(html)
        assert len(found) == 1
        assert found[0]["headline"] == "Nested headline"

    def test_handles_a_list_at_top_level(self):
        html = ('<script type="application/ld+json">'
                '[{"@type":"NewsArticle","headline":"A"}]</script>')
        assert json_ld_articles(html)[0]["headline"] == "A"

    def test_malformed_json_is_skipped_not_raised(self):
        html = '<script type="application/ld+json">{not json,,}</script>'
        assert json_ld_articles(html) == []

    def test_type_as_list(self):
        html = ('<script type="application/ld+json">'
                '{"@type":["NewsArticle","Article"],"headline":"B"}</script>')
        assert json_ld_articles(html)[0]["headline"] == "B"


class TestParseArticle:
    def test_meta_tag_timestamp_when_no_jsonld(self):
        """The Business Line shape."""
        html = """<html><head>
          <meta property="article:published_time" content="2023-01-25T20:57:54+05:30"/>
          <meta property="og:title" content="Adani vs Hindenburg"/>
          <meta property="og:description" content="A short summary."/>
          </head><body><div class="articlebodycontent">
          <p>%s</p></div></body></html>""" % ("Body sentence. " * 40)
        parsed = parse_article(html, "https://bl/x.ece", "business_line")
        assert parsed["headline"] == "Adani vs Hindenburg"
        assert parsed["published_at"].hour == 20
        assert parsed["timestamp_confidence"] == "exact"
        assert len(parsed["body"]) > 200
        assert parsed["snippet"] == "A short summary."

    def test_scripts_excluded_from_body(self):
        html = """<html><body><div class="articlebodycontent">
          <script>var blLoaderDiv = document.createElement('div');</script>
          <p>%s</p></div></body></html>""" % ("Real prose here. " * 30)
        parsed = parse_article(html, "u", "business_line")
        assert "createElement" not in parsed["body"]
        assert "Real prose here" in parsed["body"]

    def test_paywall_needs_a_marker_and_a_short_body(self):
        """Marker words appear in page furniture; alone they prove nothing."""
        long_body = "<p>%s</p>" % ("Full article text. " * 60)
        with_marker = f'<html><body><div class="artText">{long_body}' \
                      '<span>Subscribe Now</span></div></body></html>'
        assert not parse_article(with_marker, "u", "economic_times")["paywalled"]

        gated = ('<html><body><div class="artText"><p>Teaser only.</p>'
                 '<span>Subscribe Now</span></div></body></html>')
        assert parse_article(gated, "u", "economic_times")["paywalled"]

    def test_missing_timestamp_reported_not_guessed(self):
        html = "<html><head><title>Some headline</title></head><body><p>x</p></body></html>"
        parsed = parse_article(html, "u", "economic_times")
        assert parsed["published_at"] is None
        assert parsed["timestamp_confidence"] == "missing"
