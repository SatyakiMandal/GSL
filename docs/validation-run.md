# Validation run — Adani Enterprises, 20 Jan to 17 Feb 2023

The wider run the short Phase 2 window could not support. Its purpose is to
exercise the **strict** incident test — the one that requires coverage to be
statistically unusual, not merely present — and to check the relevance filter
against a real corpus rather than a handful of articles.

## What is real here and what is not

| Component | Status |
|---|---|
| News collection, extraction, relevance, dedupe, attribution, sentiment | **Real.** 200 articles fetched from the four live sources. |
| Prices, returns, abnormal returns, CAR | **Synthetic.** No live NSE feed was reachable. |

The incident days below are therefore **not findings about Adani Enterprises**.
The price series was generated with shocks placed on chosen dates, so the
returns are fiction. What the run demonstrates is that the pipeline behaves
correctly on a real news distribution.

## Ingestion result

```
candidates_discovered       ~18,000 URLs across 4 sources
candidates_after_prefilter  narrowed on URL slug before any article fetch
articles_parsed             200
in_requested_window         189
relevant                    137        dropped_by_relevance  52
unique_after_dedupe         132        duplicates             5
paywalled                     0        missing_timestamp      0
after_close                  59        unattributed           0
fetch errors                  0  (robots / http / parse / empty)
```

**59 of 189 items — 31% — published after the 15:30 IST close.** Every one of
those would have been credited to the wrong trading day without the Section 10
attribution rule. Zero items went unattributed, and zero fetches failed.

## The strict test now engages

Twelve trading days carried coverage, above the ten-day floor, so the coverage
z-test applied rather than the relaxed "any coverage" fallback. The
corresponding caveat correctly disappeared from the output.

Three days flagged. The one worth pointing at is the second:

```
2. 2023-02-01   abnormal return -28.07% (z=-26.45)
   coverage: 3 item(s) (z=-0.37), tone +0.04 — OPPOSITE to the price move
```

It flagged on **sentiment** being unusual rather than volume: a mildly positive
tone on a day when the surrounding coverage was uniformly negative. The report
labels the direction disagreement rather than filtering it out, and the run's
caveats explain that a disagreement is not necessarily an error. That is the
behaviour the PRD asks for, and it only appears on a window long enough to have
a baseline.

A useful negative too: **3 February shows an abnormal return of −9.05%
(z = −8.52) and did not flag**, because no coverage was attributed to it. An
unusual return with nothing written about it is a move with no visible
explanation, not an incident.

## The corpus caught a real defect in the relevance filter

Scoring the 137 retained items exposed **121 of them sitting at exactly 1.00**.
A linear body-mention component plus a headline match saturated the ceiling, so
the score could not rank anything, and the weight it feeds into weighted
sentiment was effectively uniform.

Rebalanced to a saturating curve with explicit named weights
(`W_HEADLINE = 0.45`, `W_BODY = 0.30`, `W_DENSITY = 0.12`, `W_POSITION = 0.10`,
penalties `P_ROUNDUP = 0.40`, `P_NO_HEADLINE = 0.15`). The distribution now runs
0.36 to 0.97 with nothing at the ceiling, and the separation is where it should
be:

| Item | Score |
|---|---|
| "Adanis dismiss US firm's allegations" | 0.97 |
| "MFs hold investment of ₹26,388 crore in Adani Group shares" | 0.97 |
| "Sensex tanks 600 pts, Nifty slips below 17,750…" (round-up) | 0.47 |
| "BJP slams billionaire investor George Soros…" (political, mentions Adani) | 0.36 |

The same corpus exposed a second, narrower miss. **"Adanis dismiss US firm's
allegations" scored 0.37** despite forty body mentions, because `\bAdani\b` does
not match the plural — the `s` is a word character. The Indian financial press
writes the family and group forms constantly, so alias patterns now accept an
optional `s`, `'s` or `’s`. That single story moved 0.37 → 0.97. Both fixes are
pinned by tests, including a negative case ensuring "Adaniyar" still does not
match "Adani".

## Still open

**PRD Success Metric #2 remains unclaimed.** It requires a known incident to be
flagged on real data with the abnormal-return direction matching sentiment.
Everything except the price feed is now verified against real inputs; the metric
needs one run with real prices to close. Set `ALPHAVANTAGE_API_KEY` (free, 25
calls/day, covers NSE) or supply CSVs via `--price-csv`, then:

```bash
python -m ceia.analyze --company "Adani Enterprises" --ticker ADANIENT.NS \
  --start 2023-01-20 --end 2023-02-17 --alias Adani --alias "Adani Group" \
  --news out/adani_wide.json --html out/report.html
```

The news half of that command needs no re-scraping — `out/adani_wide.json` is
the corpus above, and the HTTP cache means a re-run refetches nothing.
