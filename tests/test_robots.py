"""Tests for the robots.txt matcher.

The matcher decides whether this tool is allowed to fetch a page, so a bug here
means crawling something a site asked us not to. The cases below are taken from
the real robots.txt files of the four target sources as observed in Phase 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.robots import RobotsPolicy  # noqa: E402

FINANCIAL_EXPRESS = """
User-agent: *
Disallow: /wp-admin/
Disallow: /search/
Disallow: /*?s=
Disallow: /section/

User-agent: ClaudeBot
Disallow: /
User-agent: anthropic-ai
Disallow: /

Sitemap: https://www.financialexpress.com/sitemap.xml
Sitemap: https://www.financialexpress.com/news-sitemap.xml
"""

BUSINESS_LINE = """
User-agent: *
Disallow: /search/*
Disallow: /search/
Disallow: /*?date=*
Disallow: /todays-paper/

User-agent: ClaudeBot
Disallow: /
User-agent: GPTBot
Disallow: /
"""

ECONOMIC_TIMES = """
User-agent: *
Allow: /
Disallow: /topic/*1*
Disallow: */hindi/*
Disallow: /comments/
Crawl-delay: 5
"""

TOOL_UA = "CompanyEventImpactAnalyzer/0.1 (academic research)"


def _policy(text: str) -> RobotsPolicy:
    return RobotsPolicy("https://example.com", text, 200)


class TestWildcards:
    """The reason stdlib urllib.robotparser was not used."""

    def test_query_string_wildcard_blocks_search(self):
        policy = _policy(FINANCIAL_EXPRESS)
        allowed, reason = policy.allows("https://www.financialexpress.com/?s=adani", TOOL_UA)
        assert not allowed, "Disallow: /*?s= must block the search query"
        assert "/*?s=" in reason

    def test_trailing_wildcard_blocks_search_path(self):
        policy = _policy(BUSINESS_LINE)
        allowed, _ = policy.allows(
            "https://www.thehindubusinessline.com/search/?q=adani", TOOL_UA
        )
        assert not allowed

    def test_mid_path_wildcard(self):
        policy = _policy(ECONOMIC_TIMES)
        blocked, _ = policy.allows("https://et.com/topic/reliance-1-industries", TOOL_UA)
        assert not blocked
        allowed, _ = policy.allows("https://et.com/topic/reliance-industries", TOOL_UA)
        assert allowed

    def test_sitemaps_remain_allowed(self):
        for text, url in [
            (FINANCIAL_EXPRESS, "https://www.financialexpress.com/sitemap.xml"),
            (FINANCIAL_EXPRESS, "https://www.financialexpress.com/news-sitemap.xml"),
            (BUSINESS_LINE, "https://www.thehindubusinessline.com/sitemap/archive.xml"),
        ]:
            allowed, _ = _policy(text).allows(url, TOOL_UA)
            assert allowed, f"{url} should stay reachable"


class TestAgentGroups:
    def test_named_agent_group_beats_wildcard_group(self):
        policy = _policy(FINANCIAL_EXPRESS)
        allowed, _ = policy.allows("https://www.financialexpress.com/anything", "ClaudeBot/1.0")
        assert not allowed
        allowed, _ = policy.allows("https://www.financialexpress.com/anything", TOOL_UA)
        assert allowed

    def test_agent_token_match_is_case_insensitive(self):
        policy = _policy(FINANCIAL_EXPRESS)
        allowed, _ = policy.allows("https://www.financialexpress.com/x", "claudebot")
        assert not allowed

    def test_blocks_entirely_reports_named_ai_agents(self):
        blocked = _policy(BUSINESS_LINE).blocks_entirely(
            ["ClaudeBot", "GPTBot", "PerplexityBot"]
        )
        assert blocked == ["ClaudeBot", "GPTBot"]

    def test_crawl_delay_is_read(self):
        assert _policy(ECONOMIC_TIMES).crawl_delay(TOOL_UA) == 5.0


class TestPrecedence:
    def test_longest_match_wins(self):
        policy = _policy("User-agent: *\nDisallow: /news/\nAllow: /news/markets/\n")
        assert not policy.allows("https://x.com/news/sports/a", TOOL_UA)[0]
        assert policy.allows("https://x.com/news/markets/a", TOOL_UA)[0]

    def test_allow_wins_exact_tie(self):
        policy = _policy("User-agent: *\nDisallow: /a\nAllow: /a\n")
        assert policy.allows("https://x.com/a", TOOL_UA)[0]

    def test_empty_disallow_means_allow_all(self):
        policy = _policy("User-agent: *\nDisallow:\n")
        assert policy.allows("https://x.com/anything", TOOL_UA)[0]

    def test_default_allow_when_nothing_matches(self):
        policy = _policy(ECONOMIC_TIMES)
        allowed, _ = policy.allows("https://et.com/markets/stocks/news/x.cms", TOOL_UA)
        assert allowed

    def test_dollar_anchor(self):
        policy = _policy("User-agent: *\nDisallow: /*.pdf$\n")
        assert not policy.allows("https://x.com/a/b.pdf", TOOL_UA)[0]
        assert policy.allows("https://x.com/a/b.pdf?x=1", TOOL_UA)[0]


class TestFailClosed:
    """Business Standard returns 403 for robots.txt itself."""

    def test_unreadable_robots_is_disallowed(self):
        policy = RobotsPolicy("https://www.business-standard.com", None, 403)
        allowed, reason = policy.allows("https://www.business-standard.com/x", TOOL_UA)
        assert not allowed
        assert "could not be read" in reason

    def test_network_failure_is_disallowed(self):
        policy = RobotsPolicy("https://x.com", None, None)
        assert not policy.allows("https://x.com/", TOOL_UA)[0]

    def test_empty_robots_allows_everything(self):
        """200 with an empty body is a real answer: no restrictions."""
        policy = RobotsPolicy("https://x.com", "", 200)
        assert policy.fetched_ok
        assert policy.allows("https://x.com/anything", TOOL_UA)[0]
