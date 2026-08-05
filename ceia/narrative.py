"""Template-based narrative generation (PRD Section 8).

Deliberately template-based rather than model-generated. The PRD's Section 10
asks that a report's numbers be traceable back to the data behind them, and
Success Metric #3 asks that a reader who did not build the tool can follow both
the finding and its limitations. Templates give the same input the same prose
every time, which a language model would not, and every number in a sentence
here comes from a computed field rather than a paraphrase of one.

The vocabulary is chosen to stay inside what the method supports. Coverage
"coincided with" a move; it never "caused", "drove", "triggered" or "sent" it.
"""

from __future__ import annotations

from datetime import date

from .eventstudy import Incident

# Magnitude bands for abnormal returns, in standard deviations of the
# estimation-window residual. Wording escalates with |z|, not with the raw
# percentage, so a volatile stock is not described as dramatic for a move that
# is ordinary by its own standards.
_BANDS = [
    (5.0, "an extraordinary"), (3.0, "a very large"),
    (2.0, "a large"), (1.0, "a moderate"), (0.0, "a modest"),
]

_EVENT_PHRASES = {
    "earnings": "results or guidance",
    "regulatory": "regulatory or investigative developments",
    "leadership": "a change in senior leadership",
    "litigation": "legal proceedings",
    "mna": "merger or acquisition activity",
    "capital": "a capital-raising or shareholder-return event",
    "product": "operational or product news",
    "macro": "sector-wide or macroeconomic conditions",
    "other": "assorted company news",
    "": "company news",
}

# Why this *category* of news is the kind that plausibly moves a price the
# way it did - a generic, category-level mechanism, not a claim about this
# specific article. Deliberately hedged the same way the rest of this module
# is: these describe why the category is the sort of thing markets reprice
# on quickly, not that this particular story is proven to have done it here.
# Keyed by ceia.sentiment.tag_event()'s categories, same source of truth.
_EVENT_MECHANISM = {
    "earnings": (
        "Results and guidance directly update the market's estimate of the "
        "company's near-term profitability, which is why surprises here tend to "
        "reprice a stock within the same session rather than drift in over weeks."
    ),
    "regulatory": (
        "Regulatory or investigative developments change the market's read of "
        "tail risk — potential fines, restrictions, or governance findings that "
        "were not in anyone's model the day before — which is the kind of news "
        "that typically moves a price fast and hard rather than gradually."
    ),
    "leadership": (
        "A change at the top revises how much the market trusts the people "
        "steering the company's strategy and disclosures, and an unplanned exit "
        "in particular tends to raise questions the announcement itself doesn't "
        "answer, which markets often discount for immediately."
    ),
    "litigation": (
        "Legal proceedings create a range of possible financial outcomes — "
        "damages, settlements, injunctions — that did not exist as a modelled "
        "risk before the news broke, and markets tend to price in the worst "
        "plausible case first and revise later as facts emerge."
    ),
    "mna": (
        "Merger, acquisition, or stake-sale news directly changes assumptions "
        "about the company's future ownership, balance sheet, or competitive "
        "position, all of which flow straight into a valuation model rather "
        "than needing to be inferred."
    ),
    "capital": (
        "Capital-raising or shareholder-return news (a placement, buyback, or "
        "dividend change) mechanically changes share count, leverage, or cash "
        "return to shareholders, which shows up in a valuation model directly, "
        "not just as sentiment."
    ),
    "product": (
        "Operational or product news changes expectations about future revenue "
        "or capacity, though this category usually moves a price less sharply "
        "than earnings or regulatory news unless the scale is large relative to "
        "the company's existing business."
    ),
    "macro": (
        "Sector-wide or macroeconomic news moves the whole market's or "
        "sector's baseline, not just this company specifically — worth reading "
        "against the abnormal return above, which already nets out the "
        "benchmark's own move that day."
    ),
    "other": (
        "This coverage did not match one of the categories above with a clear "
        "price-relevant mechanism (results, regulatory action, a leadership "
        "change, a deal, a capital event); read the headlines below directly "
        "for what specifically was being reported."
    ),
}

# GoEmotions labels, worded for prose. This is a secondary, general-purpose
# signal (see ceia/emotion.py) - it never changes what a paragraph concludes,
# only adds a clause naming the flavour of the tone FinBERT already scored.
_EMOTION_PHRASES = {
    "anger": "anger",
    "annoyance": "irritation",
    "disapproval": "disapproval",
    "disgust": "disgust",
    "disappointment": "disappointment",
    "fear": "apprehension",
    "nervousness": "nervousness",
    "sadness": "a subdued, downbeat register",
    "grief": "a distinctly grave register",
    "embarrassment": "embarrassment",
    "remorse": "an apologetic register",
    "confusion": "confusion",
    "surprise": "surprise",
    "curiosity": "curiosity",
    "realization": "a sense of new facts coming to light",
    "admiration": "admiration",
    "approval": "approval",
    "optimism": "optimism",
    "excitement": "excitement",
    "relief": "relief",
    "pride": "pride",
    "gratitude": "gratitude",
    "joy": "a notably upbeat register",
    "amusement": "a wry or amused register",
    "caring": "a sympathetic register",
    "desire": "anticipation",
    "love": "warmth",
}


def _band(z: float) -> str:
    magnitude = abs(z)
    for threshold, wording in _BANDS:
        if magnitude >= threshold:
            return wording
    return "a modest"


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _tone(score: float) -> str:
    if score <= -0.5:
        return "strongly negative"
    if score <= -0.15:
        return "negative"
    if score < 0.15:
        return "broadly neutral"
    if score < 0.5:
        return "positive"
    return "strongly positive"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def incident_narrative(incident: Incident, company: str, benchmark: str,
                       event_window: tuple[int, int]) -> list[str]:
    """Two to four plain-language paragraphs about one flagged day."""
    paragraphs: list[str] = []

    direction = "fell" if incident.abnormal_return < 0 else "rose"
    market_direction = "fell" if incident.benchmark_return < 0 else "rose"

    # 1. What the price did, separating the market move from the company move.
    paragraphs.append(
        f"On {incident.day:%d %B %Y}, {company} {direction} "
        f"{abs(incident.raw_return) * 100:.2f}% while {benchmark} "
        f"{market_direction} {abs(incident.benchmark_return) * 100:.2f}%. "
        f"Adjusting for what the broader market did that day leaves an abnormal "
        f"return of {_pct(incident.abnormal_return)} — {_band(incident.abnormal_return_z)} "
        f"company-specific move, {abs(incident.abnormal_return_z):.1f} standard "
        f"deviations from this stock's normal behaviour. The adjustment matters: "
        f"without it the move would read as {_pct(incident.raw_return)}, most of "
        f"which "
        + ("was" if abs(incident.benchmark_return) > abs(incident.abnormal_return)
           else "was not")
        + " the market rather than the company."
    )

    # 2. What was being written.
    topic = _EVENT_PHRASES.get(incident.dominant_event, "company news")
    outlets = len(incident.sources)
    sentence = (
        f"{incident.item_count} distinct {_plural(incident.item_count, 'item')} of "
        f"coverage {_plural(incident.item_count, 'was', 'were')} attributed to this "
        f"trading day, across {outlets} {_plural(outlets, 'outlet')}, mostly "
        f"concerning {topic}. The overall tone was {_tone(incident.mean_sentiment)} "
        f"(score {incident.mean_sentiment:+.2f} on a −1 to +1 scale)."
    )
    emotion_phrase = _EMOTION_PHRASES.get(incident.dominant_emotion)
    if emotion_phrase:
        # A separate, general-purpose model (GoEmotions), not FinBERT - worded
        # as a headline-level read, distinct from the tone score above it,
        # rather than as a second vote on the same claim.
        sentence += (
            f" Read against a general-purpose emotion model rather than the "
            f"finance-tuned sentiment score above, the headlines themselves "
            f"leaned toward {emotion_phrase}."
        )
    paragraphs.append(sentence)

    # 2b. Why this *category* of news is the kind that plausibly moves a price
    # this way - a mechanism, not a claim about the specific article (that
    # would need the article's own text, which belongs in the source list
    # below, not in template prose). Skipped for "" (no category tagged at
    # all, distinct from "other" which did get tagged and found nothing).
    mechanism = _EVENT_MECHANISM.get(incident.dominant_event)
    if mechanism:
        paragraphs.append(mechanism)

    # 3. Whether tone and price agree — and what that does and does not mean.
    if incident.direction_agrees:
        paragraphs.append(
            f"The direction of the price move is consistent with the tone of the "
            f"coverage: {_tone(incident.mean_sentiment)} coverage alongside a "
            f"{'negative' if incident.abnormal_return < 0 else 'positive'} abnormal "
            f"return. That consistency is what makes this day worth a closer look — "
            f"but it remains a coincidence in time. This analysis cannot show that "
            f"any particular article caused the move, only that the two occurred "
            f"together."
        )
    else:
        paragraphs.append(
            f"The price moved in the opposite direction to the coverage's tone: "
            f"{_tone(incident.mean_sentiment)} coverage alongside a "
            f"{'negative' if incident.abnormal_return < 0 else 'positive'} abnormal "
            f"return. This is not necessarily an error in the data. News can already "
            f"be priced in before it is published, the day's move may be driven by "
            f"something the collected coverage did not capture, or an item's tone may "
            f"be read differently by the market than by a sentiment model. As with "
            f"every day in this report, the coverage and the move are a coincidence in "
            f"time; nothing here shows that one produced the other."
        )

    # 4. Did it stick, or was it a one-day blip?
    car = incident.car or {}
    if car.get("days"):
        before, after = event_window
        cumulative = car["car"]
        same_sign = cumulative * incident.abnormal_return > 0
        if not same_sign:
            persistence = (
                "the move substantially reversed over the following sessions, which "
                "points to a short-lived reaction rather than a durable repricing"
            )
        elif abs(cumulative) >= abs(incident.abnormal_return):
            persistence = (
                "the move persisted and extended over the window, which is more "
                "consistent with a sustained repricing than a one-day blip"
            )
        else:
            persistence = (
                "the move partially retraced over the window, so some but not all of "
                "the initial reaction held"
            )
        sentence = (
            f"Cumulating abnormal returns from {abs(before)} trading "
            f"{_plural(abs(before), 'day')} before to {after} after gives a CAR of "
            f"{_pct(cumulative)} over {car['days']} trading days "
            f"({car['start']} to {car['end']}): {persistence}."
        )
        if car.get("truncated"):
            sentence += (
                " Note that this window ran past the edge of the available price "
                "series and was truncated, so it covers fewer days than requested."
            )
        paragraphs.append(sentence)

    return paragraphs


def _drop_explanation(diagnostics: dict | None, candidates: int) -> str:
    """Why the rest of the window's trading days are not in the table below.

    A report that only shows the days that made the cut looks arbitrary
    without this - it answers "were the others considered and rejected, or
    never checked at all" with real counts, bucketed by which of the two
    bars (unusual coverage, unusual price move) each day cleared.
    """
    if not diagnostics or not diagnostics.get("trading_days"):
        return ""
    d = diagnostics
    dropped = d["trading_days"] - candidates
    if dropped <= 0:
        return ""
    clauses = []
    if d.get("coverage_only"):
        n = d["coverage_only"]
        clauses.append(
            f"{n} {_plural(n, 'day')} had busy or strongly-toned coverage without a "
            f"matching unusual price move"
        )
    if d.get("return_only"):
        n = d["return_only"]
        clauses.append(
            f"{n} {_plural(n, 'day')} moved unusually without correspondingly unusual "
            f"coverage"
        )
    if d.get("no_coverage_big_move"):
        n = d["no_coverage_big_move"]
        clauses.append(
            f"{n} {_plural(n, 'day')} saw an unusually large price move with "
            f"<strong>no coverage collected for it at all</strong> in this run's "
            f"sources — a real gap, not a finding"
        )
    if d.get("routine"):
        n = d["routine"]
        clauses.append(f"{n} {_plural(n, 'day')} were routine on both counts")
    if not clauses:
        return ""
    return (
        f"The other {dropped} trading {_plural(dropped, 'day')} in the window did not "
        f"clear both bars at once: " + "; ".join(clauses) + ". Only a day that clears "
        f"both — unusual coverage <em>and</em> an unusual price move — becomes a "
        f"candidate; either alone is not enough on its own."
    )


def _significance_paragraph(top: Incident, robustness: dict | None) -> str:
    """States, in plain language, how much to trust the top candidate.

    Leans on the permutation p-value rather than the z-score's own implied
    normal-tail probability - the z-score's normality assumption is exactly
    what the permutation test exists to sidestep (see returns.py), so
    restating "N standard deviations" as a spurious "1 in X chance" here
    would oversell a number the rest of this codebase deliberately declines
    to trust.
    """
    sentence = (
        f"The top candidate's abnormal return sat {abs(top.abnormal_return_z):.1f} "
        f"standard deviations from this stock's own typical daily variation."
    )
    car = top.car or {}
    p_value = car.get("p_value")
    if p_value is not None:
        n = car.get("n", 0)
        sentence += (
            f" A permutation test — which does not depend on that z-score's normality "
            f"assumption — checked this directly: across {n:,} random comparable-length "
            f"windows drawn from this stock's own price history, "
            f"{p_value * 100:.1f}% were at least as extreme (p = {p_value:.3f})."
        )
    robust = (robustness or {}).get("days", {}).get(top.day.isoformat()) if robustness else None
    if robust:
        sentence += (
            f" It also held up under a range of nearby threshold choices, not just the "
            f"exact ones configured: flagged in {robust['flagged_in']} of "
            f"{robust['of']} combinations tried."
        )
    return sentence


def summary_narrative(company: str, ticker: str, benchmark: str,
                      start: date, end: date, incidents: list[Incident],
                      daily_rows: int, news_count: int,
                      model_kind: str, diagnostics: dict | None = None,
                      robustness: dict | None = None,
                      weak_scale: bool = False) -> list[str]:
    """The opening paragraphs of the report.

    This is a structured case study, not investment advice and not a
    statistically validated causal finding — the single most important
    thing for a reader to take from this report, so it opens here rather
    than in a separate box a reader could skip past.
    """
    paragraphs = [
        f"This report examines {company} ({ticker}) between {start:%d %B %Y} and "
        f"{end:%d %B %Y}, covering {daily_rows} trading "
        f"{_plural(daily_rows, 'day')}. It is a structured case study, not "
        f"investment advice, collecting what the Indian financial press published "
        f"about the company over that period and measuring how the stock moved "
        f"relative to {benchmark}, to identify days where notable coverage "
        f"<strong>coincided with</strong> an unusual company-specific price move. "
        f"Coincidence in time is not evidence that an article caused a price move."
    ]

    if model_kind == "market-model":
        paragraphs.append(
            "Price moves are reported as <em>abnormal returns</em>: the company's "
            "actual return minus what would have been expected given the benchmark's "
            "move that day, using a market model fitted on price history from before "
            "this window. This separates the company's own move from the market's, so "
            "a day when everything fell is not mistaken for company-specific news."
        )
    else:
        paragraphs.append(
            "Price moves are reported as <em>abnormal returns</em>: the company's "
            "actual return minus the benchmark's return that day. There was not "
            "enough price history before this window to fit a full market model, so "
            "the stock is assumed to move one-for-one with the index. For a stock "
            "that is more volatile than the market, this assumption will overstate "
            "the abnormal return."
        )

    weight_notes = []
    if diagnostics and diagnostics.get("thin_baseline"):
        weight_notes.append(
            f"only {diagnostics.get('days_with_news', 0)} trading "
            f"{_plural(diagnostics.get('days_with_news', 0), 'day')} in this window "
            f"carried any coverage at all, too few to judge which days were unusually "
            f"busy — so the coverage bar below was relaxed to 'any coverage', and a "
            f"flag here rests more heavily on the price move than on the coverage "
            f"being unusual"
        )
    if weak_scale:
        weight_notes.append(
            "abnormal returns are standardised against this analysis window's own "
            "spread rather than a clean estimation window beforehand, which "
            "understates how unusual the largest moves in it really are"
        )
    if weight_notes:
        paragraphs.append(
            "How much weight to put on the numbers below: "
            + "; ".join(weight_notes) + "."
        )

    if not incidents:
        paragraphs.append(
            f"Across {news_count} collected {_plural(news_count, 'item')} of coverage, "
            f"<strong>no day met both tests</strong> — that is, no day combined "
            f"notable coverage with an unusually large company-specific price move at "
            f"the configured thresholds. That is a legitimate result: it suggests the "
            f"period contained no single dominant, market-moving story about this "
            f"company. It can also mean the thresholds are set too strictly, or the "
            f"window is too short for the coverage baseline to mean much."
        )
        drop_note = _drop_explanation(diagnostics, candidates=0)
        if drop_note:
            paragraphs.append(drop_note)
        return paragraphs

    top = incidents[0]
    paragraphs.append(
        f"Across {news_count} collected {_plural(news_count, 'item')} of coverage, "
        f"<strong>{len(incidents)} candidate "
        f"{_plural(len(incidents), 'day')}</strong> met both tests. The most "
        f"prominent was <strong>{top.day:%d %B %Y}</strong>, with an abnormal return "
        f"of {_pct(top.abnormal_return)} alongside "
        f"{_tone(top.mean_sentiment)} coverage."
    )
    paragraphs.append(_significance_paragraph(top, robustness))
    drop_note = _drop_explanation(diagnostics, candidates=len(incidents))
    if drop_note:
        paragraphs.append(drop_note)
    agreeing = sum(1 for i in incidents if i.direction_agrees)
    if len(incidents) > 1:
        paragraphs.append(
            f"Of the {len(incidents)} flagged days, {agreeing} showed a price move in "
            f"the direction the coverage's tone would suggest and "
            f"{len(incidents) - agreeing} did not. Days where the two disagree are "
            f"kept in the report rather than filtered out, because they are often the "
            f"more interesting cases."
        )
    return paragraphs
