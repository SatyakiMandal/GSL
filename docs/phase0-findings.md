# Phase 0 — Feasibility Spike

Reproduce with `python -m ceia.probe`. Everything below was observed on
2026-07-31; the raw machine-readable output is in `phase0-findings.json`.

---

## Headline result

The PRD's assumed ingestion route — drive each site's **search or archive
page** (Section 3, Section 7.1) — **is not available on three of the four named
sources.** Two disallow the search path in `robots.txt` for *every* user agent,
and one refuses all traffic outright. Business Standard was replaced by
Moneycontrol, so the effective source count is back to four.

| Source | robots.txt | PRD's search route | Usable route found | Archive depth | Verdict |
|---|---|---|---|---|---|
| **Economic Times** | 200, `Allow: /` | allowed | month sitemaps + day archive | **Oct 2001** | **Usable, verified** |
| **Financial Express** | 200 | **`Disallow: /*?s=`** | `sitemap.xml?yyyy=&mm=&dd=` | **≥ Jan 2023** | **Usable, verified** |
| **Business Line** | 200 | **`Disallow: /search/*`** | `sitemap/archive/all/YYYYMMDD_1.xml` | **Dec 2010** | **Usable, verified** |
| **Moneycontrol** | 200 | allowed | `news/index-sitemap-YYYY.xml` → month | **≥ Jan 2023** | **Usable, verified** |
| **Business Standard** | **403** | n/a | none | n/a | **Unavailable** — replaced by Moneycontrol |

All four usable sources serve **plain server-rendered HTML** — no headless
browser is needed anywhere in this project.

So the answer to "plain HTML or JavaScript-rendered?" — the question Phase 0
was chiefly meant to settle — turned out to be secondary. The binding
constraint is permission, not rendering. Where we are allowed to fetch, the
content is plain HTML and a headless browser is **not** needed.

---

## 1. Economic Times — fully verified, plain HTML

`robots.txt` opens with `User-agent: * / Allow: /`, declares no `Crawl-delay`,
and singles out only `Meta-ExternalAgent` for a blanket refusal. No
Anthropic/OpenAI-specific block.

**Discovery route (chosen): month-partitioned news sitemaps.** The
`sitemap-index.xml` advertised in `robots.txt` lists **377 monthly partitions
reaching back to October 2001** — far deeper than a search UI would page
through. Each month splits into parts capped at 10,000 URLs (January 2023 =
`-1.xml` covering the 9th–31st, `-2.xml` covering the 1st–8th; ~500 articles a
day). Entries carry `<loc>` and `<lastmod>`.

This is better than the search route the PRD assumed: it is explicitly
sanctioned by `robots.txt`, it is already partitioned by date (which is exactly
the axis a date-range tool needs), and ET's URL slugs embed the company name,
so most relevance filtering can happen on the URL list **before** fetching any
article. January 2023 held 170 Adani URLs out of 10,000.

**Secondary route: the day archive.** `/archivelist/year-YYYY,month-M,starttime-N.cms`
returns plain HTML with ~41 links. The `starttime` serial is **days since
1899-12-30** — verified by round-trip (44950 → "24 Jan, 2023"; 45310 →
"19 Jan, 2024"). Useful as a cross-check, but it is a *top-stories* list, not
full coverage: the 24 Jan 2023 page carried only one Adani mention while the
sitemap had many. Not a substitute for the sitemap.

**Dead end: `/topic/<company>`.** Company slugs 301-redirect to the stock quote
page (`/topic/adani-enterprises` → `/adani-enterprises-ltd/stocks/companyid-9074.cms`),
as does `searchresult.cms`. Both land on the same page, which is why an early
probe saw two different URLs return byte-identical HTML.

**Article pages are ideal for this tool.** Server-rendered, no JavaScript
required, and each carries a JSON-LD `NewsArticle` block with `headline`,
`datePublished`, `dateModified`, and the full `articleBody`. Sample:

```
headline:      Fully subscribed! Adani Enterprises FPO survives Hindenburg attack…
datePublished: 2023-01-31T12:46:00+05:30
dateModified:  2023-01-31T19:40:00+05:30
articleBody:   2608 chars
```

Two consequences worth carrying into Phase 1:

- **Use `datePublished`, not sitemap `lastmod`.** They differ by nearly seven
  hours in this sample (12:46 vs 19:40 IST). Section 10 asks that an item
  published after the close be attributed to the *next* trading day; using
  `lastmod` would silently push a midday story past the 15:30 IST close and
  misattribute it. The offset is explicit (`+05:30`), so no timezone guessing.
- **Paywalls exist but degrade cleanly.** ET Prime markers (`isPrime`,
  `paywall`, `Subscribe Now`) appear in the page. This sample was fully
  readable; gated ones will need the headline/timestamp/snippet fallback the
  PRD already sanctions. No attempt is made to get around the gate.

## 2. Financial Express and Business Line — verified, usable

Both carry a blanket `Disallow: /` aimed at AI-agent user agents, naming
Anthropic among others:

- **Financial Express** refuses: `ClaudeBot`, `Claude-Web`, `anthropic-ai`,
  `PerplexityBot`, `Bytespider`, `Meta-ExternalAgent`, `Applebot-Extended`.
- **Business Line** refuses those plus `GPTBot`, `ChatGPT-User`,
  `OAI-SearchBot`, `CCBot`, `Google-Extended`.

Those groups target crawlers that harvest sites wholesale. Neither site refuses
this tool's own user agent, which is declared honestly as
`CompanyEventImpactAnalyzer/0.1 (academic research; …)` and evaluated against
the `User-agent: *` group like any other client. Verification proceeded on that
basis, at the default 2 s/origin rate limit, obeying **every** `*`-group rule —
including the search-path disallows, which is why ingestion uses sitemaps.

**Financial Express.** `sitemap.xml` is an index of day-partitioned children of
the form `sitemap.xml?yyyy=2026&mm=07&dd=29`. The index lists only ~92 recent
days (2026-05-01 onward), but **dated URLs resolve well outside that window** —
`?yyyy=2023&mm=01&dd=25` returns 200 URLs, `?yyyy=2025&mm=06&dd=10` returns 175.
So FE is usable for arbitrary historical ranges even though it advertises a
three-month archive. Article pages are plain HTML with JSON-LD `NewsArticle`:

```
headline:      Adani Enterprises collects Rs 5,985 crore from anchor investors…
datePublished: 2023-01-25T21:34:23+05:30      articleBody: 1946 chars
```

That sample is itself the Section 10 attribution case: **21:34 IST is after the
15:30 close**, so it belongs to the *next* trading day's reaction, not the
25th's.

**Business Line.** `sitemap/archive.xml` indexes **6,396** day partitions of the
form `sitemap/archive/all/YYYYMMDD_1.xml`, reaching back to **2 December 2010**
— the deepest per-day archive of the three. The 2023-01-25 partition holds 114
URLs including four Adani stories, among them "Adani vs Hindenburg: a brief
story of short sellers". Pages are plain HTML.

One parser difference worth carrying into Phase 1: **Business Line publishes no
JSON-LD.** Its timestamp comes from meta tags instead —
`article:published_time`, `publish-date`, and `itemprop="datePublished"`, all
IST-offset and mutually consistent (`2023-01-25T20:57:54+05:30`). Body text also
needs a real DOM parser rather than paragraph regex, since inline scripts
otherwise leak into the extracted text. The probe now reports which timestamp
mechanism each site uses so this cannot be assumed wrong later.

## 3. Moneycontrol — the replacement for Business Standard

Added after Business Standard proved unreachable. It is the most permissive of
the four: `robots.txt` reads cleanly, the sitemap route is allowed, and the
search path is *not* disallowed (though ingestion uses sitemaps anyway, for the
date partitioning). It refuses `GPTBot`, `ChatGPT-User`, `CCBot` and
`Google-Extended`, but **names no Anthropic agent**.

**Discovery.** A year index (`news/index-sitemap-YYYY.xml`) fans out to month
sitemaps (`news/sitemap/sitemap-post-YYYY-MM.xml`). January 2023 alone holds
**8,164 URLs**, 85 of them naming Adani in the slug. Entries carry `<lastmod>`
with an IST offset.

**Parsing needed two accommodations**, both now handled:

- The `NewsArticle` JSON-LD is **nested inside an `@graph`** rather than sitting
  at the top level, so a non-recursive walker misses it entirely.
- The publish time is exposed as **`og:article:published_time`**, not the plain
  `article:published_time` the other sites use.

Sample: *"India's regulator discussed Adani firms with ratings agencies"*,
`2023-01-31T23:01:58+05:30`, 2,164 chars of body, no paywall. Note the
timestamp — 23:01 IST, long after the close.

## 4. Business Standard — unavailable

Every request, including `/robots.txt` itself, returns **403 from `AkamaiGHost`**
with an "Access Denied" interstitial. Browser-like headers do not change it;
this is edge-level bot blocking, not a `robots.txt` rule.

Because `robots.txt` is unreadable, we cannot establish that crawling is
permitted at all. The code therefore **fails closed**: `RobotsPolicy.allows()`
returns `False` when the policy could not be read, so Business Standard is
skipped rather than crawled on an assumption.

This one is not a `robots.txt` rule that a different route can satisfy, the way
the FE and BL search blocks were. Akamai is refusing the client outright, so the
only way through is to defeat the bot detection — rotating user agents,
residential proxies, or spoofing a browser's TLS/JA3 fingerprint. **That line is
not crossed here**, for three reasons that are worth stating plainly rather than
buried: it is circumventing an access control the operator deliberately put up,
it moves the project from "scraping under a site's terms" into conduct the PRD's
own Section 12 flags as legal exposure, and evasion is inherently unstable — a
fingerprint-spoofing scraper breaks on every edge-config change and silently
returns garbage rather than failing loudly.

Legitimate ways to get Business Standard back, if it matters enough:

- **Ask them.** Business Standard sells API/licensed feed access; an academic
  project is a reasonable ask.
- **Substitute a comparable source** that permits crawling — Mint, Moneycontrol,
  or Business Today all cover the same beat. This costs nothing methodologically
  and keeps the source count at four.
- **Supply items manually.** The ingestion schema is plain; a hand-collected CSV
  of headline/timestamp/URL can be dropped in for a specific incident.

**Effective source count is 4**, with Moneycontrol standing in for Business
Standard — all four verified end to end.
Section 9 already warns the sample is too small for statistical significance at
four sources; at three it is smaller still. This is reported in the tool's own
output, not just here.

## 5. Price data — `yfinance` needs help in this environment

Two independent problems, both diagnosed rather than assumed:

**(a) `yfinance` fails at the TLS layer here.** Every call dies with
`curl: (35) Recv failure: Connection reset by peer`. Isolated to a single
cause — `yfinance` ≥ 0.2.50 fetches via `curl_cffi` with browser TLS
impersonation, and this sandbox routes egress through a TLS-terminating proxy
that the impersonated handshake will not survive:

| client | result |
|---|---|
| `curl_cffi` with `impersonate="chrome"` | `SSLError` |
| `curl_cffi` with impersonation off | **200** |
| plain `requests` | **200** |

This is an artefact of *this* container. On a normal machine `yfinance` should
work unmodified, which is why it remains the first provider tried.

**(b) Yahoo rate-limits this egress IP.** Independent of client: a single cold
request returned 200, but sustained requests return **429** across both
`query1` and `query2` hosts. Shared sandbox IP, not a code defect.

`stooq` was evaluated as an alternative and rejected — it now sits behind a
JavaScript proof-of-work challenge that a plain HTTP client cannot clear.

**Resolution.** `ceia/prices.py` puts price loading behind a provider
interface and tries three in order: `yfinance` → direct Yahoo chart API over
plain `requests` (with exponential backoff and an on-disk cache) → local CSV.
The CSV provider is not a workaround for laziness; it is what keeps the event
study reproducible and testable when the network path is unavailable, which
Section 10 asks for. Ticker/benchmark symbols verified as well-formed:
`ADANIENT.NS`, `RELIANCE.NS`, `INFY.NS`, `^NSEI` (NIFTY 50), `^BSESN` (SENSEX).

**Outcome in this container, stated plainly: no provider returned data.**
The full probe, with backoff, ends:

```
yfinance      usable=False   ADANIENT.NS: yfinance returned no rows
yahoo-chart   usable=False   ADANIENT.NS: HTTP 429   ^NSEI: HTTP 429
csv           usable=False   no CSV at data/prices/ADANIENT.NS.csv
chosen provider: None
```

So the Phase 0 question "does `yfinance` return clean data for the ticker and
benchmark?" is answered **no — not from this sandbox**, and I have not verified
the shape or quality of real NSE price data. What *is* verified is the code
around it: the frame-shaping, window-slicing and provider-fallback logic is
covered by tests that run offline against the CSV provider.

This is a live risk for Phase 2, where the event-study engine needs real
returns. The likely fix is simply running it on your machine, where neither the
TLS-terminating proxy nor the shared-IP rate limit applies. If Yahoo stays
unreliable there too, the fallbacks are `nsepy`/`jugaad-data` (which the PRD
already names) or a one-time CSV export.

---

## Carried into Phase 1

1. **Three sources, all verified**: ET (2001+), Business Line (2010+), Financial
   Express (dated URLs beyond its advertised window). Business Standard stays
   disabled, with an explicit "unavailable" line in every report.
2. **Per-site article parsers.** ET and FE expose JSON-LD `NewsArticle`;
   Business Line needs meta-tag timestamps and DOM-based body extraction.
3. **Attribution uses `datePublished`, never sitemap `lastmod`.** Both the ET and
   FE samples publish after the 15:30 IST close, so this rule is load-bearing
   from the first ingested item, not an edge case.
4. **Price data is the open risk**, not news ingestion — see Section 4.
