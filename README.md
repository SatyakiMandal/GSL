# Company Event Impact Analyzer

A research tool that lines up what the Indian financial press wrote about a
company against how that company's stock actually behaved, and flags days where
unusual coverage coincided with an unusual **benchmark-adjusted** price move.

It is a structured case study generator, not a trading signal and not proof of
causation. See [Limitations](#limitations).

**Status: complete** — feasibility spike, news ingestion and sentiment,
event-study engine, and reporting. 191 tests passing.

One thing is deliberately unclaimed: **no run has used real price data.** Every
provider was unreachable from the build environment. The news half is verified
against live sources; the price half is verified against synthetic series with
known parameters. See [`docs/validation-run.md`](docs/validation-run.md).

---

## Phase 0 — Feasibility spike

Phase 0 exists to answer, before any pipeline is written, whether the data this
tool needs can actually be collected: what each site's `robots.txt` permits,
whether its pages are plain HTML or JavaScript-rendered, how far back its
archive reaches, how much article text is visible without a login, and whether
the price library returns clean data.

**The full write-up is [`docs/phase0-findings.md`](docs/phase0-findings.md).**
The short version:

- **Four sources are usable, all verified end to end**, and all serve
  **plain HTML** — no headless browser needed anywhere. Business Standard proved
  unreachable and was replaced by Moneycontrol, so the count is back to four.
  - **Economic Times**: month-partitioned sitemaps back to **October 2001**;
    JSON-LD with real `datePublished` and full article body.
  - **Business Line**: day-partitioned archive back to **December 2010** — the
    deepest of the four. No JSON-LD; timestamps come from meta tags.
  - **Financial Express**: day-partitioned sitemaps. Its index advertises only
    ~92 days, but dated URLs resolve far beyond that, so historical ranges work.
  - **Moneycontrol**: year index → month sitemaps; 8,164 URLs for January 2023
    alone. Nests its JSON-LD inside an `@graph` and uses
    `og:article:published_time`, both of which extraction handles.
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

# CPU-only torch first, so pip does not pull the CUDA build for FinBERT.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

That installs three commands: `ceia-probe`, `ceia-ingest`, `ceia-analyze`.
They are the same entry points as `python -m ceia.probe` and friends, which
still work without installing.

Lighter installs: `pip install -e .` for scraping only, `.[sentiment]` to add
FinBERT, `.[prices]` for `yfinance`, `.[dev]` for the tests.

```bash
pytest -q     # 191 tests, no network required
```

### Run the spike

```bash
# Every source, robots.txt policy only, no content fetched.
python -m ceia.probe --robots-only --skip-prices

# Everything: sitemap depth, rendering, paywall state, timestamp mechanism.
python -m ceia.probe --skip-prices

# One source at a time.
python -m ceia.probe --only moneycontrol --skip-prices

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
| `ceia/sources.py` | The sources and their discovery routes |
| `ceia/prices.py` | Pluggable price providers (`yfinance` / Yahoo chart / CSV) |
| `ceia/probe.py` | The spike itself; `python -m ceia.probe` |

Phase 1 adds `models.py`, `discovery.py`, `extract.py`, `relevance.py`,
`dedupe.py`, `align.py`, `sentiment.py` and `ingest.py` (see the table below).

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

## Phase 1 — News ingestion and sentiment

Collects company coverage across the four usable sources, filters it to what is
actually *about* the company, removes syndicated duplicates, places each item on
the trading day that could have reacted to it, and scores it with FinBERT.

```bash
python -m ceia.ingest \
  --company "Adani Enterprises" --ticker ADANIENT.NS \
  --alias "Adani" --alias "Adani Group" --alias "AEL" \
  --start 2023-01-24 --end 2023-01-28 \
  --out out/adani_jan2023.json
```

Useful flags: `--limit N` (cap fetches for a trial run), `--skip-sentiment` (no
model download), `--min-relevance` (default 0.35), `--min-interval` (seconds
between requests to one origin, default 2), `--sources`.

Output is JSON: run config, per-source status, counts, and every item with its
timestamp, trading-day attribution, relevance score, sentiment and event tag.

### Pipeline

| Stage | Module | What it does |
|---|---|---|
| Discovery | `ceia/discovery.py` | Date range → candidate URLs, per source, via sitemaps |
| Fetch/parse | `ceia/extract.py` | Headline, publish time, body, snippet, paywall state |
| Relevance | `ceia/relevance.py` | Alias matching + score; drops passing mentions |
| Dedupe | `ceia/dedupe.py` | Marks syndicated wire copy across outlets |
| Alignment | `ceia/align.py` | Publish time → trading day, honouring the 15:30 IST close |
| Sentiment | `ceia/sentiment.py` | FinBERT + coarse event category |
| Orchestration | `ceia/ingest.py` | Wires it together; CLI |

### Verified run

Over the Hindenburg window (24–28 Jan 2023), 6,296 discovered URLs narrowed to
208 by slug pre-filter, 48 fetched, 34 inside the window, 24 relevant — drawn
from all four active sources. FinBERT scored "Bloodbath in Adani Group stocks
leaves Rs 19,000-crore scar on LIC's book" negative and "FPO on track, no price
band change, says Adani Group" positive.

**13 of 34 items published after the 15:30 close.** The next-trading-day
attribution rule is load-bearing, not an edge case — nearly 40% of items would
have been credited to a price move that happened before the news existed.

### Design notes

**Sitemaps, not search pages.** Financial Express, Business Line and Moneycontrol
all disallow their search paths in `robots.txt` for every user agent. Their
sitemaps are date-partitioned anyway, which is the axis this tool needs.

**Slug pre-filtering before fetching.** A month of Economic Times coverage is
~13,000 URLs. Fetching all of them to find ~200 would be slow and rude, so
candidates are filtered on their URL slug first — a 97% reduction on the test
run. The cost is recall: a story that never names the company in its URL is
missed. That is disclosed in the run stats rather than hidden.

**Round-robin across sources.** Candidates are interleaved before `--limit`
applies, so a capped run samples every source instead of spending its whole
budget on whichever ran first.

**The headline outweighs the body** in both relevance (0.60 of the score) and
sentiment (0.60 of the blend). A headline is a claim about what a story is
about; bodies dilute it with background.

**Long articles are chunked, not truncated.** FinBERT takes 512 tokens; bodies
are split and averaged by confidence so a reversal late in a story is not lost.

**Paywall detection needs two signals.** Marker words like "Subscribe Now" appear
in page furniture on every article, so a marker only counts as a paywall when the
body is also too short to analyse. Gated items fall back to headline + snippet,
which is what the PRD sanctions. Nothing tries to get around a gate.

---

## Phase 2 — Financial data and event-study engine

Joins the collected coverage to price history, computes benchmark-adjusted
(abnormal) returns, and ranks candidate incident days.

```bash
# Reuse a Phase 1 news run; prices from a provider.
python -m ceia.analyze \
  --company "Adani Enterprises" --ticker ADANIENT.NS --benchmark ^NSEI \
  --start 2023-01-24 --end 2023-02-10 \
  --news out/adani_jan2023.json

# Or scrape and analyse in one pass.
python -m ceia.analyze --company "Adani Enterprises" --ticker ADANIENT.NS \
  --start 2023-01-24 --end 2023-02-10 --alias Adani
```

Flags: `--event-window BEFORE AFTER` (default `-1 3`), `--return-z`,
`--coverage-z`, `--lead-in-days`, `--price-csv DIR` (offline prices).

**Price providers** are tried in order: `yfinance` → Yahoo chart API →
Alpha Vantage (`ALPHAVANTAGE_API_KEY`) → local CSV.

### ⚠️ The live price feed is unverified

No provider was reachable from the build environment — `yfinance` fails at the
TLS layer through a proxy, Yahoo `429`s shared IPs, NSE India `403`s, and BSE
returns an SPA shell. **Alpha Vantage works but needs a free key.**

The maths is tested against synthetic series with known parameters (beta
recovered to ±0.12; a market-wide move on a beta-1 stock yields an abnormal
return of exactly 0). The end-to-end run used real scraped articles joined to a
**synthetic** price series — it demonstrates the wiring, not any finding about
Adani. **PRD Success Metric #2 cannot be claimed until this runs on real prices.**
See [`docs/phase2-notes.md`](docs/phase2-notes.md).

### Method

| Concept | Implementation |
|---|---|
| Expected return | Market model `α + β·R_m` fitted before the window; falls back to β=1 and says so |
| Abnormal return | `R_i − expected`, standardised by estimation-window residual SD |
| CAR | Summed over the event window in **trading days**, truncation flagged |
| Incident | A day with **both** unusual coverage **and** an unusual abnormal return |

The estimation window ends 5 trading days before the analysis window, so the
baseline is not contaminated by run-up to the event being measured.

### Two bugs the end-to-end run caught

Both were invisible to unit tests and only appeared when the whole pipeline ran:

1. **The trading calendar ended at the analysis window**, so after-close news on
   the last day had no session to attach to — 10 of 24 items silently became
   unattributed, and late CARs came back truncated. Fixed with a 21-day tail
   buffer.
2. **The coverage baseline was incoherent on short windows.** In a five-day
   window built around a known event every day is an event day, so no day looked
   unusual and *nothing flagged* — against abnormal returns of z = −12.5 and
   z = −16.9. The coverage test now relaxes to "has any coverage" below 10
   news-carrying days, and the output says so in bold terms rather than quietly
   changing its own bar.

---

## Phase 3 — Reporting

Produces a **single self-contained HTML file**: no external stylesheets, scripts,
fonts or images, so it can be emailed, committed or opened offline and still
render. Charts are inline SVG, so they stay crisp when zoomed or printed to PDF.

```bash
python -m ceia.analyze --company "Adani Enterprises" --ticker ADANIENT.NS \
  --start 2023-01-24 --end 2023-02-10 --alias Adani \
  --html out/report.html
```

`--html ''` skips it. A report is written by default alongside the JSON.

### What the report contains

1. **A limitations box, before any finding** — not a footnote. Success Metric #3
   asks that a reader who did not build the tool understands both the finding
   *and* its limitations, so the caveats come first.
2. **Summary narrative** — what was examined, which return model was used, and
   what was found, in plain language.
3. **Timeline** (three stacked panels): company vs benchmark rebased to 100;
   abnormal return bars; news volume coloured by tone. Flagged days are marked.
4. **Ranked incident table** with abnormal return, z, CAR and t.
5. **Per-incident narrative** — two to four paragraphs each, plus the source
   headlines behind the flag.
6. **Daily detail table**, flagged rows highlighted.
7. **Method and provenance** — price providers used, model note, timestamp
   alignment, and per-source availability including anything disabled.

### Narrative is template-based, not model-generated

Every number in a sentence comes from a computed field rather than a paraphrase
of one, and the same input produces the same prose every time — which matters for
the traceability Section 10 asks for. Magnitude wording escalates with the
z-score rather than the raw percentage, so a volatile stock is not called
"extraordinary" for a move that is ordinary by its own standards.

The vocabulary is constrained to what the method supports: coverage **coincided
with** a move, never "caused", "drove" or "triggered" it. There is a test that
scans generated narrative for causal verbs outside an explicit denial.

### Three bugs the tests caught here

- **`pandas.Timestamp` subclasses `datetime.date`**, so an `isinstance(x, date)`
  guard left Timestamps unconverted and they never matched the plain dates in
  the incident set. Incident markers on the chart and highlighting in the daily
  table both silently disappeared. Fixed in both places and pinned by a test.
- **The company name was interpolated into the narrative unescaped.** The
  templates emit raw HTML (they use `<em>`/`<strong>`), so a name containing
  markup would have been injected into the page.
- **The "tone disagrees with price" narrative branch omitted the non-causation
  caveat** that the agreement branch carried.

---

## Validation run

A 29-day run over **20 January – 17 February 2023** exercises the strict
incident test and the relevance filter against a real corpus. Full write-up in
[`docs/validation-run.md`](docs/validation-run.md).

**Real:** 200 articles fetched from the four live sources — 137 relevant, 132
unique after dedupe, 5 duplicates caught, **0 fetch errors, 0 unattributed
items**. **Synthetic:** the price series, so the flagged days are not findings
about Adani.

**59 of 189 items — 31% — published after the 15:30 IST close.** Every one would
have been credited to the wrong trading day without the attribution rule.

Two things the corpus caught that smaller runs could not:

- **The relevance score was saturating.** 121 of 137 items scored exactly 1.00,
  so it could not rank anything, and the weight it feeds into weighted sentiment
  was uniform. Rebalanced to a saturating curve with named weights; the
  distribution now runs 0.36–0.97 with nothing at the ceiling.
- **`\bAdani\b` does not match "Adanis".** The Indian press writes the family
  and group forms constantly, and one story with forty body mentions scored 0.37
  because of it. Alias patterns now accept an optional `s`, `'s` or `’s` — that
  story moved to 0.97. A negative test ensures "Adaniyar" still does not match.

---

## Limitations

Stated here and repeated in the tool's own output, because it is a finding about
the method rather than a disclaimer:

- **This is a case study, not a statistical result.** One company over one date
  range yields a handful of genuinely distinct incident days — far too few for
  the sentiment/abnormal-return relationship to carry statistical significance.
  A real event study spans dozens of companies and events. The PRD (Section 9)
  anticipated this at four sources, and four is what the tool ends up with —
  Moneycontrol standing in for the unreachable Business Standard — so the caveat
  binds exactly as the PRD framed it.
- **Coincidence in time is not causation.** The tool reports that coverage
  *coincided with* a price move. Benchmark-adjusting every return is the main
  defence against reading a market-wide move as company-specific news, but it
  does not establish that an article caused anything.
- **Not investment advice**, not a live signal, and daily closes only.
- **Short windows cannot support a coverage baseline.** Below 10 news-carrying
  trading days the tool flags on the abnormal return alone and says so. Widen the
  range for the stricter test.
- **A market-adjusted fallback assumes beta = 1**, which inflates abnormal
  returns for a high-beta stock. Every run states which model it used.

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
6. **Moneycontrol replaces Business Standard.** Its `robots.txt` is readable,
   permits the sitemap route, and names no Anthropic agent. Year→month sitemaps
   give 8,164 URLs for January 2023 alone.
7. **Relevance threshold defaults to 0.35**, tuned so a headline subject (~0.9)
   passes and a passing mention in a market round-up (~0.2) does not. It is a
   judgment call, exposed as `--min-relevance` rather than buried.
8. **Dedupe compares headlines, not bodies.** Paywalled items have no body, and
   syndicated copy is often re-edited below the headline. Duplicates are marked,
   not deleted, so a report can still say four outlets carried one story.
9. **Without price data, the trading calendar falls back to weekdays**, which
   misses exchange holidays — 26 January 2023 (Republic Day) is attributed as a
   trading day in a news-only run. Phase 2 supplies the real calendar from the
   price series. This is covered by a test that documents the gap.
10. **Test case: Adani Enterprises (`ADANIENT.NS`) vs NIFTY 50 (`^NSEI`), Jan–Feb
   2023.** The Hindenburg report is an unambiguous, well-documented negative
   shock, which is what Success Metric #2 needs — a known incident that should
   surface as a top candidate with a negative abnormal return.
