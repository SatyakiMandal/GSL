"""Static description of the four news sources.

Phase 0 established that the search-page route the PRD assumed (Section 7.1)
is disallowed by robots.txt on two of the four sites, so each source records
both the *assumed* route and the route we can actually use. ``discovery``
lists candidate entry points in preference order; the probe reports which of
them robots.txt actually permits for the configured user agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    origin: str
    # The search endpoint the PRD assumed we would drive.
    search_url: str
    # Robots-sanctioned discovery entry points, best first.
    discovery: list[str] = field(default_factory=list)
    # A representative article, used to test rendering and paywall behaviour.
    sample_article: str | None = None
    notes: str = ""


ECONOMIC_TIMES = Source(
    key="economic_times",
    name="Economic Times",
    origin="https://economictimes.indiatimes.com",
    search_url="https://economictimes.indiatimes.com/searchresult.cms?query=adani",
    discovery=[
        # Month-partitioned news sitemaps, advertised in robots.txt.
        "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-index.xml",
        # Day-partitioned archive; starttime is days since 1899-12-30.
        "https://economictimes.indiatimes.com/archivelist/year-2023,month-1,starttime-44950.cms",
        "https://economictimes.indiatimes.com/topic/adani-enterprises",
    ],
    sample_article=(
        "https://economictimes.indiatimes.com/markets/stocks/news/"
        "adani-enterprises-fpo-subscribed-only-20-on-last-day-will-it-be-successful/"
        "articleshow/97481359.cms"
    ),
    notes="Company /topic/ slugs 301-redirect to the stock quote page.",
)

FINANCIAL_EXPRESS = Source(
    key="financial_express",
    name="Financial Express",
    origin="https://www.financialexpress.com",
    search_url="https://www.financialexpress.com/?s=adani",
    discovery=[
        # Day-partitioned: sitemap.xml?yyyy=YYYY&mm=MM&dd=DD. The index lists
        # only ~92 recent days, but dated URLs resolve far beyond that window.
        "https://www.financialexpress.com/sitemap.xml",
        "https://www.financialexpress.com/sitemap.xml?yyyy=2023&mm=01&dd=25",
        "https://www.financialexpress.com/news-sitemap.xml",
    ],
    sample_article=(
        "https://www.financialexpress.com/market/"
        "adani-enterprises-collects-rs-5985-crore-from-anchor-investors-ahead-of-fpo-2960438/"
    ),
    notes=(
        "robots.txt disallows /search/ and /*?s= for all agents, and issues a "
        "blanket Disallow to ClaudeBot / Claude-Web / anthropic-ai. "
        "Article pages carry JSON-LD NewsArticle."
    ),
)

BUSINESS_LINE = Source(
    key="business_line",
    name="Business Line",
    origin="https://www.thehindubusinessline.com",
    search_url="https://www.thehindubusinessline.com/search/?q=adani",
    discovery=[
        # Day-partitioned: /sitemap/archive/all/YYYYMMDD_1.xml, back to Dec 2010.
        "https://www.thehindubusinessline.com/sitemap/archive.xml",
        "https://www.thehindubusinessline.com/sitemap/archive/all/20230125_1.xml",
        "https://www.thehindubusinessline.com/sitemap/googlenews/all/all.xml",
    ],
    sample_article=(
        "https://www.thehindubusinessline.com/markets/"
        "adani-vs-hindenburg-a-brief-story-of-short-sellers/article66432953.ece"
    ),
    notes=(
        "robots.txt disallows /search/ for all agents, and issues a blanket "
        "Disallow to ClaudeBot / Claude-Web / Anthropic-ai. No JSON-LD: the "
        "publish time comes from article:published_time / publish-date meta tags."
    ),
)

MONEYCONTROL = Source(
    key="moneycontrol",
    name="Moneycontrol",
    origin="https://www.moneycontrol.com",
    search_url="https://www.moneycontrol.com/news/tags/adani.html",
    discovery=[
        # Year index -> month sitemaps (sitemap-post-YYYY-MM.xml).
        "https://www.moneycontrol.com/news/index-sitemap-2023.xml",
        "https://www.moneycontrol.com/news/sitemap/sitemap-post-2023-01.xml",
        "https://www.moneycontrol.com/news/news-sitemap.xml",
    ],
    sample_article=(
        "https://www.moneycontrol.com/news/business/"
        "indias-regulator-discussed-adani-firms-with-ratings-agencies-9975491.html"
    ),
    notes=(
        "Replaces Business Standard. robots.txt refuses GPTBot / CCBot / "
        "ChatGPT-User / Google-Extended but names no Anthropic agent. "
        "JSON-LD NewsArticle is nested inside an @graph."
    ),
)

BUSINESS_STANDARD = Source(
    key="business_standard",
    name="Business Standard",
    origin="https://www.business-standard.com",
    search_url="https://www.business-standard.com/search?q=adani",
    discovery=[
        "https://www.business-standard.com/sitemap.xml",
    ],
    sample_article=None,
    notes=("Akamai edge returns 403 for every request, including /robots.txt. "
           "Replaced by Moneycontrol; see docs/phase0-findings.md."),
)

# Business Standard is retained only so the probe keeps reporting why it is
# unavailable; ingestion uses ACTIVE_SOURCES.
ALL_SOURCES = [ECONOMIC_TIMES, FINANCIAL_EXPRESS, BUSINESS_LINE,
               MONEYCONTROL, BUSINESS_STANDARD]
ACTIVE_SOURCES = [ECONOMIC_TIMES, FINANCIAL_EXPRESS, BUSINESS_LINE, MONEYCONTROL]

# User-agent tokens that these sites use to refuse AI crawlers. The probe
# reports which of them each site blocks, because it changes who may run this
# tool and under what identity.
AI_AGENT_TOKENS = [
    "ClaudeBot",
    "Claude-Web",
    "anthropic-ai",
    "GPTBot",
    "ChatGPT-User",
    "OAI-SearchBot",
    "PerplexityBot",
    "CCBot",
    "Google-Extended",
    "Bytespider",
    "Meta-ExternalAgent",
    "Applebot-Extended",
]
