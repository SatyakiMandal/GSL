## SINGLE COMMAND RUN
[download the repository and navigate insde the folder and open the terminal. Paste the following command in the CLI. The output JSON and HTML files will be stores in the /out folder]

```text
python -m ceia.analyze --company "\<COMPANY NAME\>" --ticker \<COMPANY STOCK TICKER AS PER YAHOO FINANCE\> --benchmark ^NSEI --start \<START DATE YYYY-MM-DD\> --end \<END DATE YYYY-MM-DD\> --alias \<ALIAS\> --out out/\<OUTPUT NAME\>.json --html out/\<OUTPUT NAME\>.html
```


# Company Event Impact Analyzer

A research tool that lines up what the Indian financial press wrote about a
company against how that company's stock actually behaved, and flags days where
unusual coverage coincided with an unusual **benchmark-adjusted** price move.

It is a structured case study generator, not a trading signal and not proof of
causation. See [Limitations](#limitations).

**Status: complete and verified on real data.** All four phases, a GUI on top,
463 tests
passing, and PRD Success Metric #2 — a known incident correctly flagged with
the abnormal-return direction matching sentiment — is met. `yfinance` could
not be reached from the build sandbox (a TLS-terminating proxy broke it), so
final price verification ran on a user's own machine against the same news
corpus this repo ships: the 27 January 2023 Hindenburg-report crash flagged as
the #1 candidate, abnormal return −17.12% (z = −8.5) alongside negative
coverage. Full report and details in
[`docs/validation-run.md`](docs/validation-run.md). Sentiment is now
complemented by a secondary GoEmotions layer — see Phase 1 below.

---

## Phase 0 — Feasibility spike

Phase 0 exists to answer, before any pipeline is written, whether the data this
tool needs can actually be collected: what each site's `robots.txt` permits,
whether its pages are plain HTML or JavaScript-rendered, how far back its
archive reaches, how much article text is visible without a login, and whether
the price library returns clean data.

**The full write-up is [`docs/phase0-findings.md`](docs/phase0-findings.md).**
The short version:

- **Five sources are usable, all verified end to end**, and all serve
  **plain HTML** — no headless browser needed anywhere. Business Standard proved
  unreachable and was replaced by Moneycontrol; Business Today was added later
  as a fifth, after checking four more candidates and rejecting three of them.
  - **Economic Times**: month-partitioned sitemaps back to **October 2001**;
    JSON-LD with real `datePublished` and full article body.
  - **Business Line**: day-partitioned archive back to **December 2010** — the
    deepest of the five. No JSON-LD; timestamps come from meta tags.
  - **Financial Express**: day-partitioned sitemaps. Its index advertises only
    ~92 days, but dated URLs resolve far beyond that, so historical ranges work.
  - **Moneycontrol**: year index → month sitemaps; 8,164 URLs for January 2023
    alone. Nests its JSON-LD inside an `@graph` and uses
    `og:article:published_time`, both of which extraction handles.
  - **Business Today**: day-partitioned sitemaps 1,000 days deep, same
    `?yyyy=&mm=&dd=` query shape as Financial Express. Real JSON-LD
    `articleBody`, no extraction code changes needed. Its edge occasionally
    (not systematically — verified with a five-request burst that all
    succeeded) returns an Access Denied page for a single date; treated like
    any other single-day fetch failure, not a reason to disable the source.
    Adding it caught a real, general bug: its `<loc>`/`<lastmod>` values are
    CDATA-wrapped (`<loc><![CDATA[ https://... ]]></loc>`), which the sitemap
    regex parser couldn't see past — it silently matched zero URLs rather than
    erroring, indistinguishable from "no coverage that day" without reading
    the raw response. Fixed in the shared parser, not a per-source special
    case, so any future CDATA-wrapped source works too.
- **Four more sources were checked and rejected**, same due diligence as
  above — a name being well-known isn't enough on its own:
  - **NDTV Profit** — Akamai returns "Access Denied" for `robots.txt` itself,
    same failure as Business Standard. Permission cannot be established.
  - **LiveMint** — permissive `robots.txt`, but its declared sitemaps only
    cover `today.xml`/`yesterday.xml` plus category/commodity feeds; no
    historical date-partitioned route was found, so it can't support an
    arbitrary past date range the way the other five can.
  - **Zee Business** — permissive `robots.txt` (its bot-name-specific group
    listing ClaudeBot/GPTBot/etc. carries no actual directives, and doesn't
    match this tool's own declared identity regardless — same reasoning as
    Financial Express/Business Line's Anthropic-specific blocks not applying
    here). But its sitemap only covers a rolling few days of recent news plus
    year-archives that stop at 2019 — no route to, say, mid-2025.
  - **CNBC-TV18** — permissive `robots.txt` and a clean daily sitemap index,
    but that index only holds a rolling ~366 days, so it can't reliably serve
    a fixed historical range as "today" moves forward. A candidate worth
    revisiting for recent-window runs specifically, just not for the general
    case.
- **The search route the PRD assumed is disallowed** on FE (`/*?s=`) and Business
  Line (`/search/*`) for every user agent, so ingestion uses each site's
  robots.txt-declared sitemaps — which are date-partitioned and reach further
  back anyway.
- **Business Standard — unavailable.** Akamai returns 403 for everything,
  including `robots.txt`. Since permission cannot be established, the code fails
  closed and skips it. Getting through would require defeating bot detection,
  which this project does not do; see the findings doc for legitimate
  alternatives (licensed access, or substituting Mint/Moneycontrol).
- **Price data — verified, but not from this build environment.** `yfinance`
  fails inside a TLS-terminating proxy (its `curl_cffi` browser impersonation is
  rejected) and Yahoo rate-limits the sandbox's shared egress IP (`HTTP 429`).
  Neither problem is specific to `yfinance` itself: run unproxied, on an
  ordinary connection, it worked immediately and produced the real result in
  [`docs/validation-run.md`](docs/validation-run.md). Price loading stays
  behind a provider interface — `yfinance` → Yahoo chart API → Alpha Vantage →
  local CSV — so a blocked provider degrades rather than stops the run.

### Install

```bash
python -m venv .venv && . .venv/bin/activate

# CPU-only torch first, so pip does not pull the CUDA build for FinBERT.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

That installs four commands: `ceia-probe`, `ceia-ingest`, `ceia-analyze`,
`ceia-unlisted`. They are the same entry points as `python -m ceia.probe`
and friends, which still work without installing.

Lighter installs: `pip install -e .` for scraping only, `.[sentiment]` to add
FinBERT and GoEmotions, `.[prices]` for `yfinance`, `.[gui]` for the Streamlit
front end (see [Phase 4](#phase-4--gui)), `.[dev]` for the tests, `.[pdf]` for
PDF export (see [PDF export](#pdf-export) below — needs one extra step beyond
`pip install`, which is why it's not in `.[all]`).

```bash
pytest -q     # 463 tests, no network required
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

Useful flags: `--limit N` (cap fetches for a trial run), `--workers N`
(concurrent fetches across different sites, default 8 — see
[Concurrency](#concurrency-faster-not-less-polite)), `--skip-sentiment` /
`--skip-emotion` (skip either model independently — no download),
`--min-relevance` (default 0.35), `--min-interval` (seconds between requests
to one origin, default 2), `--sources`.

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
| Emotion | `ceia/emotion.py` | GoEmotions — secondary, general-purpose emotional texture |
| Ticker lookup | `ceia/ticker_lookup.py` | Company name → Yahoo-style ticker, when `--ticker` is omitted |
| Orchestration | `ceia/ingest.py` | Wires it together; CLI |

### Ticker auto-detection

`--ticker` is optional on both `ceia.ingest` and `ceia.analyze`. Give just a
company name and the tool looks up the symbol itself via Yahoo Finance's
public search endpoint, preferring an NSE (`.NS`) listing over BSE (`.BO`) —
or the reverse with `--exchange BSE`:

```bash
python -m ceia.analyze --company "Adani Enterprises" \
  --start 2023-01-24 --end 2023-02-10 --alias Adani
```

prints `Resolved ticker: 'Adani Enterprises' -> ADANIENT.NS (Adani
Enterprises Ltd)` before continuing exactly as if `--ticker ADANIENT.NS` had
been passed directly. Pass `--ticker` explicitly to skip the lookup — needed
if the auto-detected symbol is wrong, or the exchange isn't NSE/BSE at all.
The GUI ([Phase 4](#phase-4--gui)) does the same thing: leave its Ticker
field blank and it resolves from the Company field on submit.

This calls the same Yahoo host `YahooChartProvider` already uses for price
history, so it is subject to the identical constraint noted in
[Phase 0](#phase-0--feasibility-spike): Yahoo rate-limits shared/proxied
egress (`HTTP 429`), which is what the build sandbox for this repo hits. It
worked on an unproxied connection during development; a raised
`TickerLookupError` on a blocked network is the endpoint being unreachable,
not a bug in the matching logic — fall back to an explicit `--ticker` there.

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

**A real bug: single-word slug tokens matched every conglomerate sibling, not
just the target company.** The filter used to flatten every alias into one bag
of independently-OR-matched words, so a company whose name shares a first word
with siblings under the same group — Tata (Motors/Steel/Power/Consumer/...),
Adani (Enterprises/Green/Ports/...), Reliance, Bajaj, all extremely common in
Indian markets — had every sibling's articles pass the filter too. Verified on
a live probe: for "Tata Consumer Products", 58 of 99 one-month, one-source
prefilter matches turned out to be Tata Steel, Tata Motors and TCS stories,
none of which mention "consumer" or "products" anywhere. On a `--limit`-capped
run that silently burns most of the fetch budget on the wrong company before
relevance scoring ever sees the candidates — which is what produced the
implausible "almost no coverage" results in early full-year runs. Fixed by
matching per-alias groups instead: a multi-word alias now needs **two** of its
words to co-occur in the slug (not all of them, real slugs often drop a word),
which rules out a bare "tata" while staying tolerant of which two words a
headline kept. Re-run against the same probe: 99 candidates → 3, with zero
loss against the 7 already-confirmed-relevant articles from a real run.

**A real bug a live run caught: the full legal company name matches almost
nothing in ordinary press text.** A run for Sonata Software Limited over a
7-month window came back with **zero** relevant items — `relevant: 0` out of
10 candidates that even survived the slug pre-filter — despite genuine,
easy-to-find coverage (Q4 results, a CEO change, a 10% single-day price
jump) existing in that window. The cause: the only aliases in play were the
company name exactly as typed, `"Sonata Software Limited"`, and an unrelated
user-supplied guess, `"SONATASOF"`. Both `relevance.py`'s scoring and the
slug pre-filter above match on the literal alias text, and Indian financial
headlines essentially never write out the full legal suffix — they say
"Sonata Software," not "Sonata Software Limited." Verified directly: scoring
a paragraph saturated with "Sonata Software" mentions against the alias
`"Sonata Software Limited"` alone returns `0.0`, "no alias match"; adding
the suffix-stripped `"Sonata Software"` scores it `0.86`. Every example
command in this README works around this by hand-supplying a short alias
(`--alias Adani --alias "Adani Group" --alias AEL`), which is exactly what
silently failed to happen here. Fixed at the source rather than by
documentation alone: `RunConfig.all_aliases`
(`ceia/models.py:_strip_corporate_suffix`) now always tries a version of the
company name with a trailing corporate suffix removed — Limited, Ltd, Pvt
Ltd, Private Limited, Inc, Corp, Corporation, Co, Company, LLC, PLC — so a
run does not depend on the user remembering to add one by hand. A company
name with no such suffix (`"Adani Enterprises"`, `"Wipro"`) is unaffected.

**A generalized follow-up: even the suffix-stripped short form isn't always
enough.** Re-running the fixed Sonata Software case surfaced 8 news items and
4 candidate days where there had been none — but the CEO-transition story
from the same window still didn't come through, because press headlines
about appointments and departures often drop *every* word but the company's
leading one: "Sonata appoints Rajsekhar Datta Roy as CEO," not "Sonata
Software appoints..." Verified directly: that exact headline scores `0.216`
against the aliases `["Sonata Software Limited", "Sonata Software"]` — still
below the `0.35` threshold. The fix can't be "also add the company's short
name" per company, one at a time, forever — that is exactly the case-by-case
approach that produces silent gaps for the next company nobody thought to
check. `ceia/ingest.py:widen_aliases()` instead asks the same question a
human fact-checker would before trusting a bare first name: **is this
company the top result when you search for that word alone?** It reuses the
same Yahoo ticker-search endpoint `--ticker` auto-detection already calls
(`ceia/ticker_lookup.py:resolve_ticker`) — searching for `"Sonata"` returns
Sonata Software Limited as the top (only) match, so `"Sonata"` is added.
Searching for `"Tata"` returns whichever Tata Group company Yahoo ranks
highest — almost certainly *not* whichever specific Tata sibling a given run
is actually about — so `"Tata"` is correctly left out, with **no hardcoded
list of "risky" conglomerate names to write or maintain**, which would
always be one new group away from wrong. This is the same precision/recall
split the prefilter's 2-of-N word rule already draws, just answered with a
real external signal instead of a fixed threshold. Best-effort like ticker
auto-detection: a lookup failure (offline, rate-limited) just skips the
extra alias rather than failing the run. On by default for live scraping;
`--skip-alias-widening` opts out if the extra network round-trip per run
isn't wanted.

**Round-robin across sources.** Candidates are interleaved before `--limit`
applies, so a capped run samples every source instead of spending its whole
budget on whichever ran first.

**A real bug a full-year run caught: `--limit` was silently truncating the
date range, not sampling it.** Interleaving fixes bias *across sources* but
not *across time* — each source's own candidates still arrive in roughly
chronological order, so index 0 of every source sits near `start`. A capped
run over a wide window (e.g. `--limit 500` across a full year) exhausted the
cap on the earliest days and never looked at the rest — which read as "this
company had one quiet year" in the report rather than "the run stopped
looking after March." `cap_across_range()` now takes an evenly-spaced sample
across the whole (already interleaved) candidate list instead of the first
N, so a tight cap thins out coverage everywhere rather than deleting the back
half of the window.

**The headline outweighs the body** in both relevance (0.60 of the score) and
sentiment (0.60 of the blend). A headline is a claim about what a story is
about; bodies dilute it with background.

**Long articles are chunked, not truncated.** FinBERT takes 512 tokens; bodies
are split and averaged by confidence so a reversal late in a story is not lost.

**Paywall detection needs two signals.** Marker words like "Subscribe Now" appear
in page furniture on every article, so a marker only counts as a paywall when the
body is also too short to analyse. Gated items fall back to headline + snippet,
which is what the PRD sanctions. Nothing tries to get around a gate.

### Concurrency: faster, not less polite

A full-year run repeatedly took 30-90 minutes, mostly `financial_express` and
`business_line` fetching one sitemap per day, one request at a time.
Discovery now runs all five sources concurrently (`discover()`, one worker
thread per source) and article fetching runs `--workers` candidates at once
(default 8, `fetch_and_parse()`). Neither makes the crawl faster by hitting
any single site harder — that would trade the PRD's rate-limit obligation for
speed, which is not a trade this project makes. `Fetcher` gives every origin
its own lock: two threads hitting the *same* site are serialised exactly as
if there were only one thread (still `min_interval` seconds apart, still
honouring a declared `Crawl-delay`), while requests to *different* sites run
fully in parallel. What speeds up is wall-clock time — five sources that used
to run one after another now overlap, and the whole point of interleaving
candidates across sources before fetching (see above) is that most
consecutive candidates in that list are already different origins, so a
worker pool gets real overlap without any single site seeing a faster
request rate. Verified with dedicated concurrency tests (not just "it ran
fine once"): same-origin requests never overlap in time under load,
different-origin requests do, and shared state (`items`/`errors`/`seen`)
comes out with the identical counts a sequential run would produce, run
repeatedly to catch intermittent races.

**Honest measurement, not just theory:** a real 45-day run (`--limit 500`)
took 5m02s at `--workers 1` and 4m50s at `--workers 8` — a real but modest
~4% improvement, not the dramatic win the design might suggest. Why: discovery
is bounded by whichever single source is slowest (`business_line`'s day-by-day
sitemap fetches), and that floor doesn't move regardless of `--workers`, which
only controls the *fetch* stage. For this window, discovery dominated total
time enough that fetch-stage concurrency barely showed. The gain this was
actually built for is a **full-year** run, where `financial_express` and
`business_line` each independently take 20-35 minutes — running them
concurrently with each other (rather than after each other) turns that sum
into a max, which is where the real win shows up. That specific comparison is
expensive to run twice (each side is 30-90 minutes) and hasn't been measured
end-to-end yet; treat the full-year improvement as expected from the
mechanism, not independently confirmed the way the 45-day number above is.

### Emotion (GoEmotions) — secondary texture, not a second vote

FinBERT's finance-tuned positive/negative score is what drives incident
detection and the abnormal-return direction check; that pathway is unchanged
and remains the one verified against real prices in
[`docs/validation-run.md`](docs/validation-run.md). **GoEmotions** adds a
second, independent read of each headline — fear, anger, disapproval,
approval, and 24 other labels from the Demszky et al. (2020) taxonomy — shown
alongside FinBERT's tone, never in place of it.

**It is honestly a domain mismatch, and testing against real headlines proved
it plainly.** GoEmotions was trained on Reddit comments, and financial-press
prose carries none of the informal markers (exclamations, first-person voice)
the model keys on: on a real sample, raw top-label confidence for "neutral"
ran 69–93%. Reporting that as "the emotion" would have made the field
decorative noise, so **`neutral` is excluded from consideration entirely** —
FinBERT's own tone already owns that concept — and the best *non-neutral*
label is reported instead, only when it clears a 30% confidence floor. On the
committed validation corpus (137 real headlines) that surfaces a label for
**8%** of items. Low, but genuine: a rejection-of-allegations headline scored
disapproval 0.40, an FPO-success headline scored approval 0.32, both while
"neutral" led the raw output on each. The other 92% stay silent rather than
being forced to a guess.

**A concrete payoff.** In the validation run, the day after Adani Enterprises
withdrew its FPO — where FinBERT already flags the price/tone relationship as
"opposite" because that day's coverage skewed unexpectedly positive — GoEmotions
independently corroborates it: the dominant headline emotion that day is
**approval**, even as the stock fell 28%. Two independently-trained models agreeing
that the coverage read positively despite the crash is stronger evidence than
either alone.

**A real, pre-existing bug this surfaced.** The "dominant" label for a tied day
(e.g. one item each of two different emotions) was picked with
`max(set(x), key=x.count)`, whose tie-break depends on `set` iteration order —
which for `str` keys depends on Python's per-process hash randomisation.
Verified directly: the identical tied input returned different "dominant"
labels across nine different `PYTHONHASHSEED` values. That means the exact same
scored data could report a different incident label on every run, which is
exactly the kind of silent non-reproducibility this project has otherwise gone
out of its way to avoid (provenance hashes, deterministic templates, cached
fetches). Fixed with `collections.Counter.most_common()`, which is documented
to break ties by first-encountered order — deterministic, because the input
order already is. This bug pre-dated the emotion feature (it already affected
`dominant_event`); adding a second mode computation is what surfaced it.

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

# Or scrape and analyse in one pass — ticker auto-detected from the company name.
python -m ceia.analyze --company "Adani Enterprises" \
  --start 2023-01-24 --end 2023-02-10 --alias Adani
```

Flags: `--ticker` (skips auto-detection; see
[Ticker auto-detection](#ticker-auto-detection)), `--exchange` (`NSE` or
`BSE`, biases auto-detection, default `NSE`), `--benchmark2` (optional
second index/peer ticker for a side-by-side abnormal-return comparison —
see [Complementary signals](#complementary-signals-correlation-and-volume)),
`--event-window BEFORE AFTER`
(default `-1 3`), `--return-z`, `--coverage-z`, `--lead-in-days`,
`--price-csv DIR` (offline prices), `--api-key KEY` (Alpha Vantage key on the
command line — no environment variable needed, which sidesteps a real trap on
Windows PowerShell: `set NAME=value` is `cmd.exe` syntax and does not create
an actual environment variable in PowerShell, which needs `$env:NAME =
'value'` instead).

**Price providers** are tried in order: `yfinance` → Yahoo chart API →
Alpha Vantage (`--api-key`, or `ALPHAVANTAGE_API_KEY` if unset) → local CSV.

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

### The market-model fallback was silent until it wasn't

`ceia/returns.py` had a logger and never called it. A run that falls back
from the fitted market model to market-adjusted (`beta` fixed at 1.0) —
because there wasn't enough clean lead-in history — is a real accuracy hit
(a high-beta stock's abnormal return comes back systematically inflated),
but the reason was only ever visible by reading the finished report's
provenance section afterward. It now logs at the moment it happens: raw row
counts per price series, how many survived the inner-join alignment, and the
fitted `alpha`/`beta`/`R²` on success or the exact observation count on
fallback — the same numbers that were already being computed and thrown into
`model_note`, just surfaced live instead of only on request.

### CAR significance: from a caveated t-stat to a permutation test

Every CAR the tool prints has always shipped with a `t` column and a
disclaimer next to it: the t-stat assumes independent, normally distributed
abnormal returns over a large sample, which a single company's own handful
of trading days does not provide. That disclaimer was accurate but not
useful — it told a reader the number was unreliable without giving them
anything better.

`ceia/returns.py:permutation_test_car` replaces "trust me, it's probably
fine" with an empirical answer that doesn't need the assumption: draw many
random same-length windows from this company's own abnormal-return series —
excluding every other day already flagged as a candidate incident, so a
real event can't leak into what's supposed to be the "nothing happening"
null distribution — and report what fraction of those placebo CARs are at
least as extreme as the real one. That fraction *is* a p-value, by
construction, for this specific company, series and window length. It is
still not proof the pattern would replicate on another company or another
year — the whole exercise runs on data from one company — but it does not
inherit the independence assumption `t` does, and it says so in the report
alongside it rather than replacing it outright.

Practical details: 2000 placebo draws by default (`--permutations`, GUI
"CAR permutation-test draws"; `0` disables it), deterministic (fixed seed —
re-running the same analysis reproduces the same p-value, the same
guarantee the rest of this pipeline holds elsewhere, e.g. the
`dominant_emotion`/`dominant_event` tie-break in `eventstudy.py`), and it
degrades to "not computed" with a stated reason rather than a wrong number
when the price series is too short to draw enough non-overlapping windows.
Surfaced as a `p` column next to `t` in the CLI, HTML report and GUI.

### Sensitivity: does a flag survive a different threshold?

The ranking score orders candidates for attention, but on its own it says
nothing about how sensitive the underlying *flagging test* is to
`--coverage-z`/`--return-z` — the two thresholds a day has to clear to
become a candidate at all. A day that only flags because the thresholds
happen to be set exactly where they are is a weaker finding than one that
flags under a wide range of plausible settings, and the base run alone
can't tell a reader which is which.

`ceia/eventstudy.py:robustness_check` re-runs the same flagging test across
a 3×3 grid — each threshold at 0.7×, 1.0× and 1.3× its configured value (9
combinations, always including the exact base run) — and reports, per
candidate day, how many of those 9 combinations still flag it. The grid
only varies the two z-thresholds, not the event window: the event window
changes the CAR figure attached to an already-flagged day, but never
changes whether that day flags in the first place, so sweeping it would
just relabel the same incident set under a different heading, not test
anything.

Cheap to compute: each grid cell skips the CAR permutation test
(`permutations=0`), since robustness and CAR significance are different
questions and running placebo resampling nine times over would be pure
waste. Confirmed on the Adani/Hindenburg corpus: the two strongly-evidenced
incidents (25 and 27 January 2023) come back robust at 9/9, while a
smaller move flagged mainly through the thin-coverage-baseline relaxation
came back at only 6/9 — exactly the "solid vs. borderline" distinction this
is meant to surface. Surfaced as a `Robust` column in the CLI, HTML report
and GUI.

### Complementary signals: correlation and volume

Two additions read the same daily table from a different angle, both
deliberately kept **descriptive, not part of incident detection or ranking** —
neither changes which days get flagged or how they are scored.

* **Sentiment/return correlation** (`ceia/eventstudy.py:sentiment_return_correlation`).
  A Pearson correlation between each day's weighted sentiment and its abnormal
  return across the whole analysis window (`numpy.corrcoef`, no new
  dependency). Requires at least 3 days with both values present after
  dropping the first trading day's `NaN` return (there is nothing to diff
  against on day one) and any day with no news. Reported as `r`, `r²`, `n`,
  and a plain-language note — CLI, the HTML report's stats grid, and GUI all
  surface it. It is one Pearson coefficient over a handful of days;
  read as a single extra lens on the daily table, not a significance test.
* **Trading volume**. Fetched by every price provider already but silently
  discarded before this — `align_series()` now carries it through when
  present, and `build_daily_table()` window-relative z-scores it the same way
  `coverage_z`/`sentiment_z` are computed. Shown as a `volZ` column in the CLI
  daily table, a `Volume` column in the HTML daily table, and a per-incident
  "corroborating signal" line in the CLI, HTML incident detail, and GUI top
  candidate — a volume spike alongside a sentiment/price move is a reason to
  trust the flag more, but it is not one of the two conditions (unusual
  coverage + unusual abnormal return) that make a day a candidate in the
  first place. Confirmed unchanged: the already-validated Adani/Hindenburg
  ranking (`score` and `abnormal_return_z` on the #1 incident) was
  byte-identical before and after wiring volume through.
* **Emotion valence vs return** (`ceia/eventstudy.py:emotion_valence_summary`,
  grouping from `ceia/emotion.py:VALENCE_GROUPS`). Buckets each day's
  `dominant_emotion` into the three sentiment groups GoEmotions' own paper
  clusters its 27 labels into — **positive** (12 labels: admiration,
  amusement, approval, caring, desire, excitement, gratitude, joy, love,
  optimism, pride, relief), **negative** (11: anger, annoyance,
  disappointment, disapproval, disgust, embarrassment, fear, grief,
  nervousness, remorse, sadness), **ambiguous** (4: confusion, curiosity,
  realization, surprise) — verified against the paper text directly
  (Demszky et al., 2020, Section 5.1 / Figure 2), not assumed from the label
  names (`surprise` and `realization` read as neutral-ish in isolation, but
  the paper places both under "ambiguous"). `neutral` and days with no
  confident emotion (the common case on formal headlines — see
  `pick_emotions`) are excluded from every group rather than folded into a
  fourth bucket, since a blank label means "GoEmotions had nothing
  confident to say," not "neutral valence." Reports mean abnormal return and
  mean FinBERT sentiment per group. Same rule as correlation and volume:
  groups days that already exist in the daily table, never changes which
  days are flagged or how they're scored. CLI, HTML report, and GUI all
  show it when at least one day has a groupable emotion.
* **Secondary/peer benchmark** (`--benchmark2` / the GUI's "Secondary
  benchmark / peer" field, `ceia/analyze.py:analyse`). Recomputes the
  company's abnormal return a second time against any second ticker the
  user supplies — a sector index, a direct competitor, whatever they name —
  using the same market-model/market-adjusted machinery as the primary
  benchmark, entirely independently. The primary `--benchmark` (default
  `^NSEI`) still drives incident detection, scoring and the direction
  check; the secondary comparison is additive only, shown as an extra
  column in the daily table, a beta stat, and a per-incident "vs
  {ticker}" line in the CLI, HTML report and GUI. Deliberately **not**
  hardcoded to any specific sector-index symbol: Yahoo is unreachable from
  this build sandbox, so no sector-index ticker could be verified here —
  asking the user for the ticker sidesteps guessing at a symbol that might
  not exist. A bad or unreachable secondary ticker degrades to "not shown"
  (a `PriceError`, caught and reported in the note) rather than failing the
  whole run; the primary analysis is unaffected either way, and an
  integration test confirms the primary abnormal-return series and
  incident set are byte-identical whether or not `--benchmark2` is passed.

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

The report used to carry two boilerplate sections — "What this report is,
and is not" and "Method and provenance" — that only repeated what this
README and the CLI banner already say. Both were removed; the one
load-bearing sentence from the first ("coincidence in time is not evidence
that an article caused a price move") was folded into the Summary's opening
paragraph instead of being dropped, since it's the single most important
thing a reader needs before the numbers below it.

1. **Summary narrative** — not just what the tool does, but what this run
   actually found: how many candidate days cleared both bars, the top
   candidate's date and abnormal return, a plain-language significance
   read (the top candidate's z-score, the permutation-test p-value when
   there's a long enough price series to compute one, and how it held up
   under nearby threshold choices — see
   [CAR significance](#car-significance-from-a-caveated-t-stat-to-a-permutation-test)
   and [Sensitivity](#sensitivity-does-a-flag-survive-a-different-threshold)),
   and — the part a report that only shows the winners can't answer — real
   counts for *why the rest of the window's days aren't in the table below*:
   how many had busy or strongly-toned coverage with no matching price move,
   how many moved unusually with no matching coverage, and how many moved
   unusually with **no coverage collected for them at all** (flagged
   explicitly as a gap in this run's sources, not a finding). Backed by
   `ceia/eventstudy.py:flagging_diagnostics()`, which buckets every trading
   day against the same two-part test `rank_incidents()` uses, independently.
2. **Timeline** (three stacked panels): company vs benchmark rebased to 100,
   with a dotted "start of window" reference line; abnormal return bars, axis
   labelled in %; news volume coloured by tone, with a diagonal-hatch overlay
   on negative-tone bars so the signal isn't colour-only for colourblind
   readers. Flagged days get a dashed marker line **and** a numbered badge in
   the top panel matching their rank in the incident table below (badge #1 =
   the highest-ranked candidate). Each badge carries its date as a real,
   always-visible SVG text label underneath it, not just a hover tooltip —
   `<title>` tooltips don't render when the page is printed to PDF or
   screenshotted, so the date has to be on the page itself. An in-chart
   caption above the panels spells out what the numbers and markers mean.
   Every bar also keeps a hover tooltip with its exact date and value for
   on-screen use.
3. **Ranked incident table** with abnormal return, z, CAR, t, a
   permutation-test p-value for CAR, and a threshold-robustness fraction,
   with the column footnotes explaining each inline (there's no longer a
   separate "Method and provenance" section to point to).
4. **Per-incident narrative** — two to four paragraphs each: what the price
   did, what was being written and by how many outlets, a plain-language
   explanation of *why that category of news* (earnings, regulatory,
   leadership, litigation, M&A, capital, product, macro) is the kind that
   plausibly moves a price the way it did — a mechanism, not a claim about
   the specific article — whether tone and price direction agree, and
   whether the move persisted or reversed over the event window.
5. **Coverage behind each flag** — the source headlines, each linked to the
   original article (only when the collected URL is `http(s)`; anything else
   renders as plain text, defense-in-depth against untrusted scraped data),
   with the outlet, tone and relevance, and a real one- to two-sentence
   summary of what the article was actually about (the source's own meta
   description/JSON-LD abstract when it published one, else a plain-text
   excerpt of the extracted body).
6. **Daily detail table**, flagged rows highlighted, with a `Volume` column
   when the price provider supplied one.

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

### PDF export

`--pdf PATH` (CLI) / "Generate report.pdf" (GUI) renders the same
self-contained HTML report to a PDF via headless Chromium
(`ceia/pdf.py:render_pdf`, Playwright). A real browser engine was chosen over
a pure-Python PDF library (e.g. WeasyPrint) because the report's CSS uses
`color-mix()` for the flagged-row highlight and the charts are inline SVG
resolving CSS custom properties against `prefers-color-scheme` — a browser
renders both correctly without auditing which CSS features a lighter library
does or doesn't support.

The cost is a heavier, genuinely optional dependency: `pip install -e
".[pdf]"` alone is **not** enough — Playwright also needs `playwright install
chromium` afterward to fetch the actual browser binary, a second step none of
this project's other extras require. That's why `pdf` is its own extra and
deliberately left out of `.[all]`: bundling it in would silently ship a
"PDF" button that doesn't work until a step `pip install` never mentions. If
the dependency or the browser binary is missing, both the CLI and GUI degrade
to a clear, one-line message rather than failing the whole run — the JSON and
HTML outputs are unaffected either way.

**A real bug the GUI wiring caught:** Streamlit reruns the entire script on
every widget interaction, and a `st.form_submit_button`'s "was I just
clicked" flag is only `True` on the one rerun immediately after submission.
The whole results section — including the "Generate report.pdf" button
itself — lived inside `if submitted:`, so clicking that button (a plain,
separate widget) triggered a rerun where `submitted` was `False` again,
which silently reverted the entire page back to the bare input form,
discarding the just-computed analysis before the resulting PDF download
button could ever appear. Confirmed live (Playwright driving the actual
running app, not just reading the source): the results vanished on click.
Fixed by storing the computed `Analysis` in `st.session_state` and rendering
the results section from there rather than from a bare local variable — which
also, as a side effect, fixed the same latent problem for the pre-existing
"Download report.html"/"Download analysis.json" buttons, which had quietly
had it all along.

---

## Phase 4 — GUI

`ceia/gui.py` is a [Streamlit](https://streamlit.io) front end over the same
pipeline the CLI uses — `ceia.ingest.run`, `ceia.analyze.analyse` and
`ceia.report.build_html`. No analysis logic lives in it; it builds a form,
calls those functions, and renders the result. The CLI is unchanged and stays
the scriptable/reproducible entry point; the GUI is for interactive use.

```bash
pip install -e ".[gui]"       # streamlit, on top of whatever extras you already have
streamlit run ceia/gui.py
```

Opens at `http://localhost:8501`. Two news-data modes:

- **Bundled Adani corpus** — the same 137-item `data/adani_wide_2023.json`
  from [Reusing the collected corpus](#reusing-the-collected-corpus). Runs the
  event-study/report stage only: seconds, no network, no model download.
- **Scrape live** — the full pipeline for any company/ticker/date range,
  equivalent to `python -m ceia.ingest` followed by `ceia.analyze`. FinBERT and
  GoEmotions load lazily and are cached across runs in the same session
  (`st.cache_resource`), so only the first run pays the download/load cost.
  Progress streams into the page line by line rather than leaving a blank
  spinner for the minutes a rate-limited scrape takes.

Enter just a company name and leave the Ticker field blank to auto-detect it
(the Exchange selector next to it picks NSE vs. BSE) — see
[Ticker auto-detection](#ticker-auto-detection). Aliases are optional too;
the company name alone is always matched.

Price data uses the same provider chain as the CLI (yfinance → Yahoo chart API
→ Alpha Vantage → CSV), configurable from a "Price source" panel: an optional
Alpha Vantage key, or two uploaded CSVs (ticker and benchmark) that — if both
are present — are used directly instead of any network provider. The finished
run shows headline metrics, the top candidate incident, the full HTML report
embedded inline, and download buttons for `report.html`, `analysis.json`, and
(on request — see [PDF export](#pdf-export)) `report.pdf`.

### A thread-safety bug the live log caught

Discovery and ingestion both fan work out across a `ThreadPoolExecutor` (see
[Concurrency](#concurrency-faster-not-less-polite)), so the handler streaming
their progress into the page (`_StreamlitLogHandler`) is called from worker
threads Streamlit itself never spawned. Those threads have no
`ScriptRunContext`, and a Streamlit call from one is unsafe — confirmed by
actually running a live scrape through the GUI (Playwright driving the real
running app, not just reading the source): most such calls only logged an
"ignorable" warning, but the race was tight enough that some of them
genuinely crashed the run from inside Streamlit's own internals, non-
deterministically. Fixed with `add_script_run_ctx` (Streamlit's documented
mechanism for exactly this), attaching the context captured on the main
thread to whichever thread is calling the handler at that moment, plus a lock
around the actual render so concurrent callers cannot interleave a torn one.
Regression-tested by hammering the handler from 40 concurrent threads.

---

## Phase 5 — Unlisted / pre-IPO shares

Added on request, for shares that trade over-the-counter rather than on NSE
or BSE — priced via [UnlistedZone](https://unlistedzone.com/), an
information platform for unlisted and pre-IPO shares.

```bash
python -m ceia.unlisted --company "National Stock Exchange of India Limited" \
  --alias NSE --start 2026-01-01 --end 2026-07-31 \
  --out out/nse_unlisted.json --html out/nse_unlisted_report.html
```

### URL auto-detection

`--url` is optional, mirroring [Ticker auto-detection](#ticker-auto-detection)
for listed stocks. Give just `--company` and `resolve_unlisted_url()`
(`ceia/unlisted.py`) walks UnlistedZone's own `/shares` directory — the same
paginated listing a person browsing the site would see, ~24 companies per
page — and fuzzy-matches the company name against every card it finds
(`difflib.SequenceMatcher`, with an automatic perfect score when one name
contains the other). It prints `Resolved UnlistedZone URL: 'National Stock
Exchange of India Limited' -> https://unlistedzone.com/shares/...` before
continuing exactly as if `--url` had been passed directly, and raises rather
than guessing when nothing clears the same confidence bar `resolve_ticker()`
uses for listed tickers — pass `--url` explicitly to skip the lookup or work
around a bad match. UnlistedZone has no server-side search endpoint (its
`?search=` query parameter is client-side only, verified directly), which is
why this is a locally-scraped-directory match rather than a single lookup
call. This is purely additive: it changes nothing about how listed-stock
runs (`ceia.ingest`, `ceia.analyze`) resolve tickers.

### Three extra news sources, unlisted-space only

Running this tool on real unlisted companies (Polymatech, among others)
turned up a second gap, independent of the price data: the five listed-market
sources (`DEFAULT_SOURCES` in `ceia/ingest.py`) cover an unlisted company
opportunistically at best — mostly when there's an IPO-plans story — because
these companies aren't part of daily market coverage the way a listed
blue-chip is. Routine coverage of funding rounds, valuations and
private-market corporate actions (a CEO stepping down, for one real example
turned up live: `entrackr.com/fintrackr/nazaras-q1-fy27-results...`) sits
instead on startup/private-market-focused outlets.

`ceia.unlisted` now defaults to `DEFAULT_SOURCES` plus three such outlets —
**Entrackr**, **VCCircle**, and **Inc42** — via a new `UNLISTED_DEFAULT_SOURCES`
constant (`ceia/unlisted.py`), each checked against the same bar Phase 0 set
for the original five (`robots.txt` readable and permissive, a discoverable
historical archive, parseable timestamps and body text) before being added:

- **Entrackr** — day-partitioned archive (`sitemap_YYYY-MM-DD.xml`, verified
  back to 2017-05-29), JSON-LD `NewsArticle` with a full `articleBody`. The
  cleanest fit of the three.
- **VCCircle** — a numbered, reverse-chronological archive
  (`article-sitemap-N.xml`; file 1 is the newest window, file 66 reached back
  to 2008 when checked) discovered by walking forward from file 1 until a
  file's own content is entirely older than the wanted window
  (`discovery.py:vccircle()`), since neither the filename nor the sitemap
  index's `lastmod` encode a per-file date the way the other sources' do. No
  JSON-LD body; the publish time comes from a non-standard
  `content_type:published_time` meta tag, added to `extract.py`'s timestamp
  list (`tests/test_extract.py` pins it).
- **Inc42** — a WordPress/Yoast archive (`sitemap_index.xml`'s numbered
  `post-sitemapN.xml` files, oldest to newest) with JSON-LD `articleBody`.

**Only `ceia.unlisted` opts into these** — `DEFAULT_SOURCES` itself, and
therefore every already-verified `ceia.ingest`/`ceia.analyze` listed-company
run, is untouched. Override with `--sources` on either CLI if a run needs a
different set.

**Inc42's `robots.txt` raised the same question UnlistedZone's did, again
made explicitly rather than silently.** It names `ClaudeBot` (and other named
AI crawlers) specifically, restricting them to structured entity pages
(`/company/`, `/person/`, …) while explicitly allowing "live answer-engine
fetchers" (`Claude-Web`, `Claude-User`) everywhere and leaving the general
`User-agent: *` group — which our own, honestly-declared user agent falls
under, since it is not itself named anywhere in the file — at `Allow: /`.
More deliberately engineered than the earlier UnlistedZone case (a whole
taxonomy of bot categories, not a blanket block), so it was raised again
rather than assumed to fall under the same precedent automatically. The
answer was the same: proceed under the existing rule that a distinct,
honestly-declared user agent is evaluated against the general group, not a
block aimed at named crawlers.

**LiveMint was checked and dropped.** Its `robots.txt` is fully permissive
with no AI-agent carve-outs, but the *only* news sitemaps it declares are
`sitemap/yesterday.xml` and `sitemap/today.xml` — no month/day/numbered
archive route exists for it at all, verified directly. That covers a live
news feed, not the months-long historical backfill this tool actually needs,
so it fails the same bar Business Standard failed for a different reason
(Phase 0) and was left out rather than wired up to silently return almost
nothing on any but the most recent run. **YourStory was checked and
dropped** for a harder reason: its `robots.txt` looks permissive, but its
sitemap and article routes sit behind a Cloudflare bot-challenge (`Just a
moment...` / 403) that blocks a plain HTTP fetch regardless of policy — the
same category of wall that already disqualified Business Standard's Akamai
edge.

### This is deliberately not the same tool as Phases 2–4

An unlisted share's price is not the same kind of number as a listed one,
and pretending otherwise would have made this section a worse, quieter
version of the bug the rest of this project spends its effort avoiding.

Checked directly against UnlistedZone's own data before writing a line of
this: one real company's chart carries a dated point for **1,321** of the
calendar days between September 2021 and August 2026 — but only **156** of
those are genuine price revisions; the rest hold the last real number flat
until UnlistedZone's own team revises it again. The site says so itself:
its prices are *"indicative levels compiled by our team for information
only — not a price feed, quote, or offer to deal."* That is not daily price
discovery, so none of the listed-stock machinery that depends on one — a
market-model beta, a z-score standardised against daily volatility, a
permutation test against a daily null — has anything valid to stand on
here. Forward-filling the gaps to make the series *look* daily (an earlier
idea, tried and rejected) would not fix this: it would artificially deflate
the very baseline the z-score is measured against, since a run of repeated
zero-return days shrinks the estimation-window standard deviation, making
whatever real move eventually shows up look far more "significant" than it
is — and it would pin that whole move on whichever single day the feed
happens to update, when the real cause could be news from any point in the
weeks since the previous revision. A report built that way would look
exactly as statistically rigorous as the listed-company one while resting
on data that cannot earn any of it.

**What the data can support:** the real revision points, and the raw
percentage change between one and the next, attributed honestly to the
*date range* between them rather than to a single fabricated day —
`ceia/unlisted.py:real_updates()`/`price_moves()`. News is matched to
whichever gap it falls inside, using the exact same ingestion pipeline as
the listed-company tool (`ceia.ingest.run()`, unchanged — alias widening,
FinBERT, GoEmotions and all, since none of that depends on how the price is
priced). The report (`ceia/report.py:build_unlisted_html()`) is a distinct
template, not a bent version of the listed one: no `z`, `CAR`, `t`, `p`, or
`Robust` column, because none of them are earned. See
`ceia/unlisted.py`'s and `ceia/unlisted_narrative.py`'s module docstrings
for the full reasoning; `tests/test_unlisted_narrative.py` and
`tests/test_unlisted_report.py` scan every generated sentence to confirm
that vocabulary never leaks in as a claimed value, only ever as an explicit
disclaimer of its absence.

**News coverage, charted and tabulated — but never against an abnormal
return.** The report also carries a news-volume/tone bar chart
(`ceia/charts.py:news_coverage_svg()`) and a flat table of every dated item
collected in the window (`ceia/report.py:_news_table()`), matching the
listed-company report's coverage panel and per-day table. The distinction
this section has been drawing throughout still holds: a day's news count and
sentiment tone need nothing from a benchmark or a market model to be
honestly described, so that much is shown here exactly as it is for listed
stocks — what stays absent is any panel or column that would compare that
coverage against an abnormal return, since there is no valid one to compare
it against.

### A second real finding: large moves can be corporate actions, not news

The same live check turned up gaps of **+182%** and **−65%** in one
company's real revision history. Those line up with a bonus-share issue
UnlistedZone's own coverage separately confirms happened in that window — a
4:1 bonus mechanically resets the per-share price to roughly a fifth of
what it was, with no change in what the company is actually worth. Unlike
a listed price feed, UnlistedZone's series does not appear to be adjusted
for this. The report's Summary states this caveat explicitly rather than
letting the largest number in the table imply a dramatic market reaction
that may just be a share count changing.

### The `robots.txt` decision, made explicitly rather than silently

UnlistedZone's `robots.txt` disallows a crawler named `ClaudeBot` by name
(`Content-Signal: ai-train=no`) while allowing generic bots. This project's
existing precedent for named-crawler blocks (Financial Express/Business
Line naming "Anthropic") is that a distinct, honestly-declared user agent
is evaluated against the general `User-agent: *` group instead — which
technically permits this. But this is the first source where the named bot
is literally the one doing the building, not a generality, so it was not
this project's call to make on its own: raised directly, the answer was to
proceed on the same precedent already applied elsewhere. Recorded here
rather than left implicit.

## Reusing the collected corpus

`data/adani_wide_2023.json` holds the 137-item Adani corpus from the validation
run — headlines, URLs, timestamps, and the already-computed relevance and
sentiment scores. Point `--news` at it to re-run the event study without
re-scraping anything:

```bash
python -m ceia.analyze --company "Adani Enterprises" --ticker ADANIENT.NS \
  --start 2023-01-20 --end 2023-02-17 --alias Adani --alias "Adani Group" \
  --news data/adani_wide_2023.json --html out/report.html
```

**Article bodies are stripped from this file deliberately.** Fetching articles
for private analysis is one thing; committing 137 publishers' articles to a
repository is redistribution, which is a different act. The analysis stage never
needs the body again — sentiment is already scored — so nothing is lost. Run
`python -m ceia.ingest` to rebuild the full corpus, bodies included, yourself.

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
  anticipated this at four sources; the tool ends up with five — Moneycontrol
  standing in for the unreachable Business Standard, Business Today added
  later — but the underlying caveat binds the same way regardless of source
  count: it is still one company, one date range.
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
11. **UnlistedZone's named `ClaudeBot` disallow was raised explicitly, not
   decided silently.** See [Phase 5](#phase-5--unlisted--pre-ipo-shares) — the
   answer was to proceed on this project's existing named-crawler precedent,
   but that call was asked for rather than assumed, since this is the first
   source where the named bot is literally the one doing the building.
12. **Inc42's named `ClaudeBot` restriction was raised again, not assumed to
   fall under the UnlistedZone precedent automatically.** See
   [Phase 5](#phase-5--unlisted--pre-ipo-shares) — its policy is more
   deliberately engineered than UnlistedZone's (a taxonomy of bot categories,
   not a blanket block), so it was checked with again rather than silently
   waved through; the answer was the same as before.
