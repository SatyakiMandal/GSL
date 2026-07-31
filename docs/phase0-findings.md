# Phase 0 — Feasibility Spike

Reproduce with `python -m ceia.probe`. Everything below was observed on
2026-07-31; the raw machine-readable output is in `phase0-findings.json`.

---

## Headline result

The PRD's assumed ingestion route — drive each site's **search or archive
page** (Section 3, Section 7.1) — **is not available on three of the four
sources.** Two disallow the search path in `robots.txt` for *every* user agent,
and one refuses all traffic outright.

| Source | robots.txt | PRD's search route | Usable route found | Verdict |
|---|---|---|---|---|
| **Economic Times** | 200, `Allow: /` | allowed | month sitemaps + day archive | **Usable, verified** |
| **Financial Express** | 200 | **`Disallow: /*?s=`** | `sitemap.xml`, `news-sitemap.xml` (declared in robots.txt) | Permitted, **unverified** |
| **Business Line** | 200 | **`Disallow: /search/*`** | `sitemap/archive.xml`, `sitemap/update.xml` | Permitted, **unverified** |
| **Business Standard** | **403** | n/a | none | **Unavailable** |

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

## 2. Financial Express and Business Line — permitted route, but I did not fetch them

Both sites are reachable and their `robots.txt` **permits the sitemap route**
for this tool's user agent. But both also carry a blanket `Disallow: /` aimed
at AI-agent user agents, and the list explicitly names Anthropic:

- **Financial Express** refuses: `ClaudeBot`, `Claude-Web`, `anthropic-ai`,
  `PerplexityBot`, `Bytespider`, `Meta-ExternalAgent`, `Applebot-Extended`.
- **Business Line** refuses those plus `GPTBot`, `ChatGPT-User`,
  `OAI-SearchBot`, `CCBot`, `Google-Extended`.

I am Claude, so I read each site's `robots.txt` (always fair game — it is the
mechanism for discovering policy) and then **stopped**, rather than fetching
their content pages to verify rendering, archive depth, and paywall behaviour.
Those three cells in the table are genuinely unverified, and I have not
guessed at them.

**This does not necessarily block you.** The refusal is addressed to
Anthropic's crawler, not to a tool you run yourself under its own user agent —
and the tool identifies itself honestly as
`CompanyEventImpactAnalyzer/0.1 (academic research; …)`, which neither site
refuses. Running `python -m ceia.probe --only financial_express business_line`
on your own machine completes the verification in about a minute. That is your
call to make, not mine, which is why it is flagged here rather than decided.

Note the sitemap route is also *better* for these two than search would have
been: both declare their sitemaps in `robots.txt` (FE: `sitemap.xml`,
`news-sitemap.xml`; BL: `sitemap/archive.xml`, `sitemap/update.xml`,
`sitemap/googlenews/all/all.xml`), so the blocked search path costs us little.

## 3. Business Standard — unavailable

Every request, including `/robots.txt` itself, returns **403 from `AkamaiGHost`**
with an "Access Denied" interstitial. Browser-like headers do not change it;
this is edge-level bot blocking, not a `robots.txt` rule.

Because `robots.txt` is unreadable, we cannot establish that crawling is
permitted at all. The code therefore **fails closed**: `RobotsPolicy.allows()`
returns `False` when the policy could not be read, so Business Standard is
skipped rather than crawled on an assumption. Getting past Akamai would mean
defeating an anti-bot control, which the build instructions rule out, so this
is treated as a permanent degradation and reported in every run's output per
the PRD's Section 10 requirement to state which sources were unavailable.

**Effective source count is 3, not 4** — and with FE/BL unverified, only 1 is
confirmed end-to-end today. Section 9 already warns the sample is too small for
statistical significance with four sources; at three it is smaller still. This
is reported in the tool's own output, not just here.

## 4. Price data — `yfinance` needs help in this environment

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

---

## Open questions for Phase 1

1. **Verify FE and Business Line yourself** (one command, above) — or tell me to
   proceed on ET alone.
2. **Is 3 sources enough to continue?** My read is yes: the PRD already frames
   the output as a case study, and ET alone gives ~500 articles/day back to
   2001. But it makes the Section 9 caveat load-bearing rather than boilerplate.
3. **Business Standard** — I propose leaving it permanently disabled with a
   clear "unavailable" line in every report.
