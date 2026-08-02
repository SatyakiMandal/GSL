"""Tests for the feasibility probe.

The probe's job is to predict what a real ingestion run will get. If it reports
something different from what the pipeline actually extracts, it is worse than
useless — it hides a working source or blesses a broken one. It previously did
exactly that: it carried its own copy of the parsing logic, fell behind
`ceia.extract`, and reported "NONE FOUND" for a Moneycontrol page the pipeline
parsed perfectly. These tests pin the agreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.extract import parse_article  # noqa: E402
from ceia.probe import _looks_javascript_rendered, _probe_article, _verdict  # noqa: E402
from ceia.probe import SourceFinding  # noqa: E402

# The three metadata shapes the live sources actually use.
GRAPH_NESTED = """<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
  {"@type":"WebPage","name":"x"},
  {"@type":"NewsArticle","headline":"Regulator reviews disclosures",
   "datePublished":"2023-01-31T23:01:58+05:30",
   "articleBody":"BODY "}]}
</script></head><body><div id="contentdata"><p>%s</p></div></body></html>""" % (
    "Real prose about the company. " * 30)

OG_PREFIXED_META = """<html><head>
<meta property="og:article:published_time" content="2023-01-31T23:01:58+05:30"/>
<meta property="og:title" content="Regulator reviews disclosures"/>
</head><body><div id="contentdata"><p>%s</p></div></body></html>""" % (
    "Real prose about the company. " * 30)

PLAIN_META = """<html><head>
<meta property="article:published_time" content="2023-01-25T20:57:54+05:30"/>
<meta property="og:title" content="A brief story of short sellers"/>
</head><body><div class="articlebodycontent"><p>%s</p></div></body></html>""" % (
    "Real prose about the company. " * 30)


class FakeResponse:
    def __init__(self, text): self.text, self.status = text, 200


class FakeFetcher:
    def __init__(self, html): self.html = html
    def get(self, url, force=False): return FakeResponse(self.html)


@pytest.mark.parametrize("html,expected_source", [
    (GRAPH_NESTED, "json-ld"),
    (OG_PREFIXED_META, "meta-tag"),
    (PLAIN_META, "meta-tag"),
])
class TestProbeMatchesExtraction:
    def test_probe_finds_a_timestamp(self, html, expected_source):
        out = _probe_article("https://x/y", FakeFetcher(html), "moneycontrol")
        assert out["timestamp_source"] == expected_source
        assert out["timestamp_source"] != "NONE FOUND"
        assert out["published_at"] is not None

    def test_probe_agrees_with_the_pipeline(self, html, expected_source):
        """The property that was actually broken."""
        out = _probe_article("https://x/y", FakeFetcher(html), "moneycontrol")
        parsed = parse_article(html, "https://x/y", "moneycontrol")
        assert out["published_at"] == parsed["published_at"].isoformat()
        assert out["headline"] == parsed["headline"]
        assert out["body_chars"] == len(parsed["body"])
        assert out["paywalled"] == parsed["paywalled"]


class TestProbeArticleEdges:
    def test_graph_nested_jsonld_is_found(self):
        """Moneycontrol nests NewsArticle inside @graph."""
        out = _probe_article("https://x/y", FakeFetcher(GRAPH_NESTED), "moneycontrol")
        assert out["jsonld"]["datePublished"].startswith("2023-01-31")

    def test_reports_the_block_that_carries_the_timestamp(self):
        html = ("""<html><head>
        <script type="application/ld+json">{"@type":"Article","headline":"Nav"}</script>
        <script type="application/ld+json">{"@type":"NewsArticle","headline":"Real",
          "datePublished":"2023-01-31T10:00:00+05:30"}</script>
        </head><body><p>%s</p></body></html>""" % ("text " * 200))
        out = _probe_article("https://x/y", FakeFetcher(html), "economic_times")
        assert out["jsonld"]["headline"] == "Real"
        assert out["jsonld"]["blocks_found"] == 2

    def test_no_metadata_reports_none_found(self):
        html = "<html><body><p>%s</p></body></html>" % ("text " * 200)
        out = _probe_article("https://x/y", FakeFetcher(html), "economic_times")
        assert out["timestamp_source"] == "NONE FOUND"

    def test_fetch_failure_is_captured_not_raised(self):
        class Broken:
            def get(self, url, force=False): raise RuntimeError("blocked")
        out = _probe_article("https://x/y", Broken(), "economic_times")
        assert "error" in out and "blocked" in out["error"]


class TestJavaScriptDetection:
    def test_short_page_is_treated_as_a_shell(self):
        assert _looks_javascript_rendered("<html></html>")

    def test_server_rendered_article_is_not(self):
        assert not _looks_javascript_rendered(
            "<html><body><article>%s</article></body></html>" % ("word " * 800))

    def test_next_data_shell_is_flagged(self):
        assert _looks_javascript_rendered(
            "<html><body><div>%s</div><script>__NEXT_DATA__</script></body></html>"
            % ("x" * 3000))


class TestVerdict:
    def test_unreadable_robots_is_unavailable(self):
        assert _verdict(SourceFinding(key="bs", name="BS", robots_readable=False)) \
            == "UNAVAILABLE"

    def test_blocked_agent_is_reported(self):
        finding = SourceFinding(key="x", name="X", robots_readable=True,
                                blocks_our_agent=True)
        assert "BLOCKED" in _verdict(finding)

    def test_usable_via_sitemap_notes_the_search_block(self):
        finding = SourceFinding(
            key="fe", name="FE", robots_readable=True, search_allowed=False,
            discovery=[{"url": "https://x/sitemap.xml", "robots_allowed": True,
                        "status": 200}])
        verdict = _verdict(finding)
        assert "USABLE via sitemap" in verdict
        assert "search endpoint disallowed" in verdict

    def test_permitted_but_unfetched_is_distinguished(self):
        finding = SourceFinding(
            key="fe", name="FE", robots_readable=True, search_allowed=True,
            discovery=[{"url": "https://x/sitemap.xml", "robots_allowed": True}])
        assert "UNVERIFIED" in _verdict(finding)
