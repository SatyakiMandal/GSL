# Phase 2 — Financial data and event-study engine

## Validation status, stated up front

**The event-study maths is tested. The live NSE price feed is not.**

No price provider could be reached from the build environment:

| Source | Result |
|---|---|
| `yfinance` | fails at TLS — `curl_cffi` browser impersonation rejected by the proxy |
| Yahoo chart API (`query1`/`query2`) | `HTTP 429` on every request, sustained |
| NSE India (`nseindia.com`) | `403` — Akamai, same edge block as Business Standard |
| BSE India API | returns an SPA shell, not data |
| stooq | JavaScript proof-of-work challenge |
| Alpha Vantage | **reachable**, needs a free API key |

So the engine was verified two ways, neither of which is "it produced a correct
Adani finding":

1. **Unit tests against synthetic series with known parameters** — a series
   generated as `R_i = 0.0005 + 1.4·R_m + ε` has its beta recovered to within
   0.12, a market-wide move on a beta-1 stock yields an abnormal return of
   exactly 0, and a beta-2 stock falling 8% when the market falls 4% correctly
   yields an abnormal return of 0.
2. **An end-to-end CLI run** on the 24 real scraped Adani articles joined to a
   **synthetic** price series with shocks injected on 25 and 27 January. The
   engine recovered both days, ranked them, computed CAR, and detected that the
   coverage tone agreed with the price direction.

That second run demonstrates the wiring, **not** a finding about Adani. The
price series was generated, so the returns in it are fiction.

**To close this out**, either export `ALPHAVANTAGE_API_KEY` (free, 25 calls/day,
covers NSE) or drop CSVs in with `--price-csv`. PRD Success Metric #2 — a known
incident correctly flagged with the abnormal return direction matching sentiment
— cannot be claimed until that happens on real data.

## Methodology (PRD Section 9)

**Abnormal return.** Two models, chosen automatically:

- **market-model** — `AR = R_i − (α + β·R_m)`, α and β fitted by OLS on an
  estimation window *before* the analysis period. This is the "more rigorous
  market-model version" the PRD asks for.
- **market-adjusted** — `AR = R_i − R_m`, i.e. β fixed at 1. Used when there is
  not enough clean lead-in data, and **disclosed in the run's output**, because
  a high-beta stock shows systematically inflated abnormal returns under it.

The estimation window ends `ESTIMATION_GAP_DAYS` (5) before the analysis window
starts, so the "normal" baseline is not contaminated by run-up to the very event
being measured. There is a test asserting a −40% shock inside the analysis window
does not distort the fitted beta.

**Standardisation.** Abnormal returns are divided by the estimation window's
residual SD. When no clean window exists the analysis window's own SD is used
instead — which is weaker, because a period dominated by one large event inflates
its own denominator and *understates* how unusual that event was. Which scale was
used is reported per run.

**CAR** is summed over the event window in **trading days**, not calendar days,
so (−1, +3) around a Friday spans Thursday to the following Wednesday. Windows
truncated at the edge of the price series are flagged rather than silently
shortened.

**Incident flagging** requires *both* unusual coverage and an unusual abnormal
return. Unusual coverage alone is a busy news day; an unusual return alone is a
move with no visible explanation. See the short-window caveat below.

## Two bugs the end-to-end run caught

Both were invisible to unit tests and only appeared when the whole pipeline ran.

**1. The trading calendar ended at the analysis window.** Prices were fetched for
`[start − lead_in, end]`, so news published after the close on the *last* day had
no following session to attach to — 10 of 24 items silently became
"unattributed", and any CAR on a late incident came back truncated. Fixed with a
21-day tail buffer past `end`. The fix is why the run's CAR now reaches
2023-02-01 for a 27 January incident.

**2. The coverage baseline was incoherent on short windows.** Coverage z-scores
are computed against the company's own coverage *within the analysis window*. In
a five-day window built around a known event, every day is an event day, so no
day looks unusual and **nothing ever flagged** — the first run reported "none
flagged" against abnormal returns of z = −12.5 and z = −16.9, which is plainly
wrong.

Fixed by relaxing the coverage test to "has any coverage at all" when fewer than
`MIN_DAYS_FOR_BASELINE` (10) days carry news, and saying so explicitly in the
output:

> Only 4 trading day(s) in this window carried coverage — too few to say which
> days were unusually busy, so the coverage test was RELAXED to 'any coverage at
> all' and days were flagged on the abnormal return alone. Treat the ranking as
> 'notable price moves that had coverage', not as evidence that the coverage was
> itself unusual.

This is a genuine limitation of short windows, not a workaround. The honest fix
for a user is a wider date range, which the caveat says.

## Phase 1 gap now closed

Phase 1 attributed news to trading days using a weekday fallback, which treated
exchange holidays as tradeable. Phase 2 loads prices **before** attribution and
passes the real session calendar in, so 26 January (Republic Day) is skipped when
real data is used. The weekday fallback remains for news-only runs and is still
covered by a test that documents what it misses.

## What the ranking is not

The combined score orders candidates for a reader's attention. It is not a
p-value. The t-statistics printed alongside CAR are conventional and are printed
for that reason, but with one company over a handful of events the independence
assumptions behind them do not hold. Every run repeats this in its own output
rather than relying on this document.
