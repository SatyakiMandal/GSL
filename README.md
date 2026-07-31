# Company Event Impact Analyzer

A research tool that lines up what the Indian financial press wrote about a
company against how that company's stock actually behaved, and flags days where
unusual coverage coincided with an unusual **benchmark-adjusted** price move.

It is a structured case study generator, not a trading signal and not proof of
causation. See [Limitations](#limitations).

**Status: Phase 0 (feasibility spike) complete.** Phases 1–3 not yet built.

---

## Phase 0 — Feasibility spike

Phase 0 exists to answer, before any pipeline is written, whether the data this
tool needs can actually be collected: what each site's `robots.txt` permits,
whether its pages are plain HTML or JavaScript-rendered, how far back its
archive reaches, how much article text is visible without a login, and whether
the price library returns clean data.

**The full write-up is [`docs/phase0-findings.md`](docs/phase0-findings.md).**
The short version:

- **Three of four sources are usable, all verified end to end**, and all serve
  **plain HTML** — no headless browser needed anywhere.
  - **Economic Times**: month-partitioned sitemaps back to **October 2001**;
    JSON-LD with real `datePublished` and full article body.
  - **Business Line**: day-partitioned archive back to **December 2010** — the
    deepest of the three. No JSON-LD; timestamps come from meta tags.
  - **Financial Express**: day-partitioned sitemaps. Its index advertises only
    ~92 days, but dated URLs resolve far beyond that, so historical ranges work.
- **The search route the PRD assumed is disallowed** on FE (`/*?s=`) and Business
  Line (`/search/*`) for every user agent, so ingestion uses each site's
  robots.txt-declared sitemaps — which are date-partitioned and reach further
  back anyway.
- **Business Standard — unavailable.** Akamai returns 403 for everything,
  including `robots.txt`. Since permission cannot be established, the code fails
  closed and skips it. Getting through would require defeating bot detection,
  which this project does not do; see the findings doc for legitimate
  alternatives (licensed access, or substituting Mint/Moneycontrol).
- **Price data — not verified.** `yfinance` fails inside a TLS-terminating proxy
  (its `curl_cffi` browser impersonation is rejected), and Yahoo rate-limits
  shared egress IPs (`HTTP 429`). **No provider returned data from this
  sandbox**, so real NSE price quality is still unconfirmed. Price loading is
  behind a provider interface — `yfinance` → direct Yahoo chart API → local CSV
  — and the surrounding logic is covered by offline tests. Re-run the probe on
  your own machine to close this out.

### Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### Run the spike

```bash
# All four sources: robots.txt policy only, no content fetched.
python -m ceia.probe --robots-only --skip-prices

# Economic Times end to end (sitemap depth, rendering, paywall, JSON-LD).
python -m ceia.probe --only economic_times --skip-prices

# Financial Express and Business Line.
python -m ceia.probe --only financial_express business_line --skip-prices

# Check the price providers for a ticker and benchmark.
python -m ceia.probe --ticker ADANIENT.NS --benchmark ^NSEI \
                     --start 2023-01-01 --end 2023-03-01 --robots-only
```

Results print to stdout and are written to `docs/phase0-findings.json`.

Useful flags: `--user-agent` (what identity to evaluate `robots.txt` against),
`--cache-dir`, `--only <keys>`, `--robots-only`, `--skip-prices`.

### What Phase 0 shipped

| File | Purpose |
|---|---|
| `ceia/robots.py` | `robots.txt` parser and matcher |
| `ceia/fetcher.py` | Polite HTTP client: robots enforcement, rate limiting, disk cache, provenance log |
| `ceia/sources.py` | The four sources and their discovery routes |
| `ceia/prices.py` | Pluggable price providers (`yfinance` / Yahoo chart / CSV) |
| `ceia/probe.py` | The spike itself; `python -m ceia.probe` |

**`ceia/robots.py`** implements the matcher rather than using stdlib
`urllib.robotparser`, which does not support the `*` and `$` wildcards. Both
Financial Express (`Disallow: /*?s=`) and Business Line (`Disallow: /search/*`)
express their search blocks with wildcards, so the stdlib parser would have
silently reported those paths as allowed and the tool would have crawled pages
it had been asked not to. Rule precedence follows Google's spec: longest
matching pattern wins, `Allow` breaks an exact tie.

**`ceia/fetcher.py`** is the single outbound path for the whole project, so the
Section 10 obligations hold everywhere by construction:

- `robots.txt` is fetched once per origin and honoured on every request. If it
  **cannot be read**, requests to that origin fail closed — an unreadable policy
  is not treated as permission.
- Requests to one origin are spaced by at least `min_interval` (default 2s), and
  by the site's own `Crawl-delay` when it declares a longer one.
- Responses are cached to disk, so re-running the same analysis re-scrapes
  nothing.
- Every fetch appends a record to `cache/provenance.jsonl` with the URL, final
  URL after redirects, status, timestamp, byte count and **SHA-256 of the body**,
  so any figure in a report can be traced back to the exact bytes behind it.
- The client identifies itself honestly as `CompanyEventImpactAnalyzer/0.1
  (academic research; …)` rather than impersonating a browser.

Nothing in Phase 0 attempts to bypass a paywall, a login wall, or an anti-bot
control. Where content is gated, the plan is the fallback the PRD sanctions:
headline, timestamp, and visible snippet.

---

## Limitations

Stated here and repeated in the tool's own output, because it is a finding about
the method rather than a disclaimer:

- **This is a case study, not a statistical result.** One company over one date
  range yields a handful of genuinely distinct incident days — far too few for
  the sentiment/abnormal-return relationship to carry statistical significance.
  A real event study spans dozens of companies and events. The PRD (Section 9)
  anticipated this at four sources; Phase 0 found the effective count is **three**
  (Business Standard is unreachable), so the caveat binds harder, not less.
- **Coincidence in time is not causation.** The tool reports that coverage
  *coincided with* a price move. Benchmark-adjusting every return is the main
  defence against reading a market-wide move as company-specific news, but it
  does not establish that an article caused anything.
- **Not investment advice**, not a live signal, and daily closes only.

## Assumptions and judgment calls

Decisions made without asking, and the reasoning:

1. **Ingest via sitemaps, not search pages.** The PRD assumes search/archive
   pages (Sections 3, 7.1), but `robots.txt` disallows the search path on two of
   the four sites for all agents. Sitemaps are declared in `robots.txt`, are
   already partitioned by date, and reach further back. This is a real conflict
   inside the PRD — "scrape the search page" versus "respect `robots.txt`" — and
   it is resolved in favour of `robots.txt`.
2. **Unreadable `robots.txt` means "do not crawl."** Business Standard 403s even
   on `robots.txt`, so permission cannot be established and the source is
   skipped rather than crawled on an assumption.
3. **Financial Express and Business Line are crawled under the tool's own user
   agent.** Both name Anthropic's crawler in a blanket `Disallow: /`, which
   targets crawlers that harvest sites wholesale. Neither refuses this tool's
   declared agent, which is evaluated against the `User-agent: *` group like any
   other client, rate-limited, and obeys every `*`-group rule — including the
   search-path disallows.
7. **Business Standard is left disabled rather than forced.** Its block is
   edge-level, not a `robots.txt` rule, so the only way through is defeating bot
   detection. That is an access control, not a crawl preference, so the project
   does not circumvent it.
4. **`datePublished` from JSON-LD, not sitemap `lastmod`.** They differed by ~7
   hours in the sample. `lastmod` would push midday stories past the 15:30 IST
   close and misattribute them to the next trading day — exactly the quiet error
   Section 10 asks the alignment logic to prevent.
5. **A CSV price provider was added** beyond the PRD's `yfinance`/`nsepy`
   options, so the event-study engine stays reproducible and testable when the
   network path is blocked.
6. **Test case: Adani Enterprises (`ADANIENT.NS`) vs NIFTY 50 (`^NSEI`), Jan–Feb
   2023.** The Hindenburg report is an unambiguous, well-documented negative
   shock, which is what Success Metric #2 needs — a known incident that should
   surface as a top candidate with a negative abnormal return.
