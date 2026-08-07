"""Tests for pipeline plumbing: slug pre-filter, interleaving, window trimming.

Also covers article extraction against saved fixtures of the real page shapes,
since each source exposes its metadata differently and a silent extraction
regression would poison everything downstream.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.discovery import Candidate  # noqa: E402
from ceia.extract import IST, clean_headline, json_ld_articles, parse_article  # noqa: E402
from ceia.ingest import (  # noqa: E402
    _slug_token_groups,
    cap_across_range,
    in_range,
    interleave,
    prefilter,
    widen_aliases,
)
from ceia.ingest import PREFILTER_SKIP_THRESHOLD  # noqa: E402
from ceia.models import NewsItem, RunConfig  # noqa: E402


class TestSlugTokenGroups:
    def test_drops_corporate_suffixes(self):
        groups = _slug_token_groups(["Adani Enterprises Ltd"], "ADANIENT.NS")
        alias_words = {w for g in groups for w in g}
        assert "adani" in alias_words and "enterprises" in alias_words
        assert "ltd" not in alias_words
        assert "adanient" in alias_words

    def test_one_group_per_alias_plus_one_for_the_ticker(self):
        groups = _slug_token_groups(["Adani Enterprises", "Adani Group"], "ADANIENT.NS")
        assert frozenset({"adanient"}) in groups
        assert frozenset({"adani", "enterprises"}) in groups
        assert frozenset({"adani", "group"}) in groups
        assert len(groups) == 3


class TestSlugPrefilter:
    def test_single_word_group_matches_on_one_hit(self):
        candidates = [
            Candidate("https://x.com/news/adani-shares-fall-123.html", "et"),
            Candidate("https://x.com/news/infosys-wins-deal-456.html", "et"),
        ]
        kept = prefilter(candidates, [frozenset({"adani"})])
        assert len(kept) == 1 and "adani" in kept[0].url

    def test_no_groups_keeps_everything(self):
        candidates = [Candidate("https://x.com/a", "et")]
        assert prefilter(candidates, []) == candidates

    def test_conglomerate_sibling_is_excluded(self):
        """The real bug this exists to fix: a bare "tata" token matched every
        Tata Group company's articles, not just Tata Consumer's - verified on
        a live probe where 58 of 99 one-month prefilter hits for "Tata
        Consumer Products" turned out to be Tata Steel/Motors/TCS stories."""
        candidates = [
            Candidate("https://x.com/tata-consumer-products-q3-results.html", "mc"),
            Candidate("https://x.com/tata-steel-nederland-green-transition.html", "mc"),
            Candidate("https://x.com/tata-harrier-ev-price-range.html", "mc"),
            Candidate("https://x.com/tata-consultancy-services-campus.html", "mc"),
        ]
        groups = [frozenset({"tata", "consumer", "products"})]
        kept = prefilter(candidates, groups)
        assert [c.url for c in kept] == [candidates[0].url]

    def test_two_of_three_alias_words_is_enough(self):
        """Real slugs often drop a word ("tata-consumer-share-price" has no
        "products"), so the threshold tolerates 2-of-N rather than requiring
        every word from a 3+-word company name."""
        candidates = [Candidate("https://x.com/tata-consumer-share-price.html", "mc")]
        groups = [frozenset({"tata", "consumer", "products"})]
        assert prefilter(candidates, groups) == candidates

    def test_two_word_alias_still_requires_both_words(self):
        """A 2-word alias has no slack: matching only one word is exactly the
        single-generic-word false-positive this whole fix removes."""
        candidates = [Candidate("https://x.com/tata-steel-price-target.html", "mc")]
        groups = [frozenset({"tata", "consumer"})]
        assert prefilter(candidates, groups) == []

    def test_matching_any_one_group_is_enough(self):
        candidates = [Candidate("https://x.com/tcpl-quarterly-earnings.html", "mc")]
        groups = [frozenset({"tata", "consumer", "products"}), frozenset({"tcpl"})]
        assert prefilter(candidates, groups) == candidates


class TestSlugPrefilterSkipThreshold:
    """The generalized fix for thin unlisted-space coverage: a source small
    enough to fetch entirely this run gains nothing from a URL guess that
    can only ever undercount real coverage, so it skips the slug filter and
    lets every one of its candidates through to the real, text-based
    relevance scorer instead - a rule keyed to a measurable per-run property
    (this source's own candidate count), not a hardcoded source/company list,
    so it benefits any company run against any low-volume source."""

    def test_default_threshold_matches_legacy_no_skip_behaviour(self):
        """prefilter()'s own default (skip_threshold=0) must reproduce the
        pre-existing filter exactly - the adaptive behaviour is opt-in from
        ingest.run(), not a change to the function's default contract."""
        candidates = [Candidate("https://x.com/unrelated-story.html", "et")]
        assert prefilter(candidates, [frozenset({"adani"})]) == []

    def test_a_source_under_the_threshold_skips_slug_filtering_entirely(self):
        candidates = [
            Candidate("https://entrackr.com/some-other-startup-raises-funds", "entrackr"),
            Candidate("https://entrackr.com/adani-mentioned-nowhere-in-this-slug", "entrackr"),
        ]
        kept = prefilter(candidates, [frozenset({"adani"})], skip_threshold=10)
        assert kept == candidates

    def test_a_source_over_the_threshold_still_gets_slug_filtered(self):
        candidates = [
            Candidate("https://x.com/adani-shares-rise.html", "et"),
            Candidate("https://x.com/unrelated-story.html", "et"),
        ]
        kept = prefilter(candidates, [frozenset({"adani"})], skip_threshold=1)
        assert [c.url for c in kept] == [candidates[0].url]

    def test_threshold_applies_per_source_not_across_the_whole_run(self):
        """A high-volume source next to a low-volume one must not borrow the
        other's headroom - each source's own count decides its own fate."""
        candidates = [
            Candidate(f"https://x.com/et/story-{i}.html", "et") for i in range(5)
        ] + [
            Candidate("https://entrackr.com/adani-not-in-this-slug", "entrackr"),
        ]
        kept = prefilter(candidates, [frozenset({"adani"})], skip_threshold=3)
        assert [c.source for c in kept] == ["entrackr"]

    def test_ingest_module_wires_the_named_constant_not_a_magic_number(self):
        assert PREFILTER_SKIP_THRESHOLD > 0


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Same shape test_ticker_lookup.py drives resolve_ticker through -
    widen_aliases() calls that function directly, so it takes the same fake."""

    def __init__(self, responses: list[_FakeResponse] | None = None,
                should_not_be_called: bool = False) -> None:
        self._responses = list(responses or [])
        self._should_not_be_called = should_not_be_called
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self._should_not_be_called:
            raise AssertionError("resolve_ticker should not have been called")
        return self._responses.pop(0)


def _quotes(*symbols_and_names: tuple[str, str]) -> _FakeResponse:
    return _FakeResponse(200, {"quotes": [
        {"symbol": symbol, "quoteType": "EQUITY", "longname": name}
        for symbol, name in symbols_and_names
    ]})


class TestWidenAliases:
    """The generalized fix for the Sonata Software gap: a short press
    headline that drops every word but the company's leading one ("Sonata
    appoints new CEO") matches neither the full legal name nor its
    suffix-stripped short form. Adding the leading word as an alias closes
    that, but only when an independent ticker-search confirms it is not
    also a well-known sibling's leading word (the Tata/Adani/Bajaj problem
    this project already hit once - see README)."""

    @pytest.fixture(autouse=True)
    def _no_real_sleeps(self, monkeypatch):
        """Retries back off with time.sleep(); tests shouldn't pay for that."""
        monkeypatch.setattr("ceia.ticker_lookup.time.sleep", lambda _seconds: None)

    def test_adds_leading_word_when_it_is_the_top_search_result(self):
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS")
        session = _FakeSession([_quotes(("SONATSOFTW.NS", "Sonata Software Ltd"))])
        aliases = widen_aliases(config, session=session)
        assert "Sonata" in aliases
        assert session.calls == 1

    def test_skips_leading_word_when_a_different_company_is_the_top_result(self):
        """The exact conglomerate-sibling case: a run for Tata Consumer
        Products must not pick up "Tata" as an alias when Tata's own
        ticker-search top result is a different Tata Group company."""
        config = RunConfig(company="Tata Consumer Products Limited",
                           ticker="TATACONSUM.NS")
        session = _FakeSession([_quotes(("TATAMOTORS.NS", "Tata Motors Ltd"))])
        aliases = widen_aliases(config, session=session)
        assert "Tata" not in aliases
        assert aliases == config.all_aliases

    def test_lookup_failure_degrades_silently(self):
        """Offline, rate-limited, or a dead endpoint must not fail the run -
        same rule --ticker auto-detection already follows."""
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS")
        session = _FakeSession([_FakeResponse(500)] * 5)  # exhausts all retries
        aliases = widen_aliases(config, session=session)
        assert aliases == config.all_aliases

    def test_no_equity_results_degrades_silently(self):
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS")
        session = _FakeSession([_FakeResponse(200, {"quotes": []})])
        aliases = widen_aliases(config, session=session)
        assert aliases == config.all_aliases

    def test_does_not_duplicate_an_alias_already_present(self):
        """If the user (or RunConfig.all_aliases' own suffix-stripping)
        already supplied the leading word, no lookup is needed at all."""
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS",
                           aliases=["Sonata"])
        session = _FakeSession(should_not_be_called=True)
        aliases = widen_aliases(config, session=session)
        assert aliases.count("Sonata") == 1
        assert session.calls == 0

    def test_cross_exchange_listing_still_counts_as_a_match(self):
        """Same company, different exchange suffix (.BO vs .NS) - the root
        symbol is what identifies the company, not the exchange."""
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS")
        session = _FakeSession([_quotes(("SONATSOFTW.BO", "Sonata Software Ltd"))])
        aliases = widen_aliases(config, session=session)
        assert "Sonata" in aliases

    def test_company_name_with_no_usable_leading_word_is_left_alone(self):
        config = RunConfig(company="Ltd", ticker="TEST.NS")
        session = _FakeSession(should_not_be_called=True)
        assert widen_aliases(config, session=session) == config.all_aliases


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
