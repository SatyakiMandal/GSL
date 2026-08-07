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

BUSINESS_TODAY = Source(
    key="business_today",
    name="Business Today",
    origin="https://www.businesstoday.in",
    search_url="https://www.businesstoday.in/search?searchtext=adani",
    discovery=[
        # Day-partitioned: /rssfeeds/date-wise-story-sitemap.xml?yyyy=&mm=&dd=,
        # indexed 1,000 days deep at /rssfeeds/date-wise-stories-sitemap.xml.
        "https://www.businesstoday.in/rssfeeds/date-wise-stories-sitemap.xml",
        "https://www.businesstoday.in/rssfeeds/date-wise-story-sitemap.xml?yyyy=2025&mm=01&dd=25",
    ],
    sample_article=(
        "https://www.businesstoday.in/technology/news/story/"
        "apple-unveils-new-mac-studio-with-m4-max-and-m3-ultra-466869-2025-03-05"
    ),
    notes=(
        "robots.txt is permissive (User-agent: * / Allow: /) and names no "
        "Anthropic agent. Akamai edge (same as Business Standard and NDTV "
        "Profit) occasionally returns an Access Denied page for a single "
        "date query, verified as an intermittent, not systematic, edge "
        "hiccup - a retry or even an immediate different-date request both "
        "succeeded. Article pages carry JSON-LD NewsArticle with a full "
        "articleBody."
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

# Added for the unlisted/pre-IPO space specifically (ceia.unlisted): the five
# sources above are mainstream listed-market financial press, which cover an
# unlisted company opportunistically at best - real routine coverage of
# funding rounds, valuations and private-market corporate actions sits on
# startup/private-market-focused outlets instead. Left out of DEFAULT_SOURCES
# (ceia/ingest.py) so the already-verified listed-company pipeline is
# untouched; ceia.unlisted opts into these via its own wider default.
ENTRACKR = Source(
    key="entrackr",
    name="Entrackr",
    origin="https://entrackr.com",
    search_url="https://entrackr.com/?s=adani",
    discovery=[
        # Day-partitioned: /sitemap_YYYY-MM-DD.xml, indexed at
        # webcontent-sitemap.xml back to 2017-05-29 (verified).
        "https://entrackr.com/webcontent-sitemap.xml",
        "https://entrackr.com/sitemap_2026-08-06.xml",
        "https://entrackr.com/news-sitemap.xml",
    ],
    sample_article=(
        "https://entrackr.com/fintrackr/"
        "ixigo-posts-rs-357-cr-revenue-in-q1-fy27-profit-rises-81-12236939"
    ),
    notes=(
        "robots.txt is permissive (User-agent: * / Allow: /, only /static/* "
        "disallowed) and names no AI agent. JSON-LD NewsArticle carries "
        "datePublished and a full articleBody."
    ),
)

VCCIRCLE = Source(
    key="vccircle",
    name="VCCircle",
    origin="https://www.vccircle.com",
    search_url="https://www.vccircle.com/?s=adani",
    discovery=[
        # Numbered, reverse-chronological: article-sitemap-1.xml is the
        # newest window, article-sitemap-66.xml reaches back to 2008
        # (verified directly - there is no date in the filename or a usable
        # per-file lastmod at the index level, only inside each file).
        "https://www.vccircle.com/sitemap/article-sitemap-index.xml",
        "https://www.vccircle.com/sitemap/article-sitemap-1.xml",
    ],
    sample_article=(
        "https://www.vccircle.com/"
        "tatasons-pushed-towards-listing-following-central-bank-classification"
    ),
    notes=(
        "robots.txt is permissive (User-agent: * / Allow: /) and names no AI "
        "agent. No JSON-LD articleBody; the publish time comes from a "
        "non-standard content_type:published_time meta tag (added to "
        "extract.py's timestamp list), and body text falls back to the "
        "generic paragraph-density extractor, verified working."
    ),
)

INC42 = Source(
    key="inc42",
    name="Inc42",
    origin="https://inc42.com",
    search_url="https://inc42.com/?s=adani",
    discovery=[
        # WordPress Yoast archive: sitemap_index.xml's numbered
        # post-sitemapN.xml files run oldest (2) to newest (56, as of
        # writing), each internally per-URL <lastmod>-dated. The unnumbered
        # post-sitemap.xml duplicates the newest numbered file and is
        # skipped by discovery.inc42() accordingly.
        "https://inc42.com/sitemap_index.xml",
        "https://inc42.com/news-sitemap.xml",
    ],
    sample_article=(
        "https://inc42.com/buzz/pe-giant-tpg-offloads-shadowfax-shares-in-"
        "%e2%82%b9301-cr-bulk-deal/"
    ),
    notes=(
        "robots.txt's default User-agent: * group is Allow: / (our own "
        "user agent - CompanyEventImpactAnalyzer/0.1 - is not itself named "
        "anywhere in the file) but a separate, explicitly engineered group "
        "blocks ClaudeBot/GPTBot/etc. by name from everything except "
        "structured entity pages, while allowing Claude-Web/Claude-User "
        "('live answer-engine fetchers') everywhere. Raised directly rather "
        "than decided silently, given how deliberate that policy is; the "
        "answer was to proceed under the same rule already applied to "
        "UnlistedZone - a distinct, honestly-declared user agent is "
        "evaluated against the general group, not a block aimed at named "
        "crawlers. JSON-LD NewsArticle carries datePublished and a full "
        "articleBody."
    ),
)

# Business Standard is retained only so the probe keeps reporting why it is
# unavailable; ingestion uses ACTIVE_SOURCES.
ALL_SOURCES = [ECONOMIC_TIMES, FINANCIAL_EXPRESS, BUSINESS_LINE,
               MONEYCONTROL, BUSINESS_TODAY, BUSINESS_STANDARD,
               ENTRACKR, VCCIRCLE, INC42]
ACTIVE_SOURCES = [ECONOMIC_TIMES, FINANCIAL_EXPRESS, BUSINESS_LINE, MONEYCONTROL,
                  BUSINESS_TODAY]

# ceia.unlisted's opt-in additions to DEFAULT_SOURCES (ceia/ingest.py), keyed
# the same way discovery.STRATEGIES and RunConfig.sources already are.
UNLISTED_EXTRA_SOURCES = ["entrackr", "vccircle", "inc42"]

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
