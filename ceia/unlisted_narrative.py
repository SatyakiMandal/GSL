"""Template-based narrative for the unlisted/pre-IPO price-move timeline.

Same discipline as ``ceia/narrative.py`` - deterministic templates, every
number traceable to a computed field, coverage "coincided with" a move and
never "caused" it - but deliberately without that module's significance
vocabulary. There is no z-score, market-model beta, CAR, or permutation
p-value here, because none of those have a valid daily baseline to stand on
against a periodically-revised indicative price (see ``ceia/unlisted.py``'s
module docstring). Printing that vocabulary anyway would borrow the
listed-company report's air of statistical rigour for data that cannot
earn it, which is the one thing this module exists to avoid.
"""

from __future__ import annotations

from datetime import date

from .narrative import _plural, _pct, _tone
from .unlisted import PriceMove


def move_narrative(move: PriceMove, company: str, rank: int | None = None) -> list[str]:
    """One or two plain-language paragraphs about one price-revision gap."""
    direction = "rose" if move.change > 0 else "fell" if move.change < 0 else "held steady"
    days = (move.end_date - move.start_date).days
    paragraphs = [
        f"Between {move.start_date:%d %B %Y} and {move.end_date:%d %B %Y} "
        f"({days} calendar {_plural(days, 'day')}), {company}'s indicative "
        f"UnlistedZone price {direction} from ₹{move.start_price:,.2f} to "
        f"₹{move.end_price:,.2f}, a change of {_pct(move.change)}. This is "
        f"the gap between two periodically revised indicative estimates, not a "
        f"daily market-quoted close, so the change cannot be dated more "
        f"precisely than this range — it may have happened all at once or "
        f"built up gradually within it."
    ]

    if move.headlines:
        outlets = len({h["source"] for h in move.headlines})
        sentiments = [1 if h["sentiment_label"] == "positive"
                     else -1 if h["sentiment_label"] == "negative" else 0
                     for h in move.headlines]
        mean_sentiment = sum(sentiments) / len(sentiments)
        paragraphs.append(
            f"{len(move.headlines)} distinct {_plural(len(move.headlines), 'item')} "
            f"of coverage {_plural(len(move.headlines), 'was', 'were')} published in "
            f"this window, across {outlets} {_plural(outlets, 'outlet')}, "
            f"{_tone(mean_sentiment)} in tone overall. This describes what was "
            f"published alongside the price change; it is not a test of whether "
            f"the two are related beyond timing, and — unlike this tool's "
            f"listed-company reports — no significance test is computed here, "
            f"because a periodically revised indicative price has no daily "
            f"baseline to measure unusualness against."
        )
    else:
        paragraphs.append(
            "No collected coverage falls inside this window in this run's "
            "sources — a gap in what was found, not evidence that nothing "
            "happened."
        )

    return paragraphs


def unlisted_summary_narrative(
    company: str, start: date, end: date,
    moves: list[PriceMove], calendar_days: int, news_count: int,
) -> list[str]:
    """The opening paragraphs of an unlisted-share timeline report."""
    paragraphs = [
        f"This report examines {company}'s indicative price on UnlistedZone "
        f"between {start:%d %B %Y} and {end:%d %B %Y}, alongside what the "
        f"Indian financial press published about the company over that "
        f"period. It is a structured timeline, not a statistically validated "
        f"event study: UnlistedZone's own indicative prices are periodically "
        f"revised estimates, not daily market-quoted closes, so a day without "
        f"a revision is not evidence the price was actually unchanged that "
        f"day — only that no new estimate was published."
    ]

    if moves:
        real_points = len(moves) + 1
        paragraphs.append(
            f"Across {calendar_days} calendar {_plural(calendar_days, 'day')} in "
            f"the window, the indicative price was revised {real_points} "
            f"{_plural(real_points, 'time')}, giving {len(moves)} "
            f"{_plural(len(moves), 'gap')} between revisions to examine — "
            f"roughly one every {calendar_days // max(real_points - 1, 1)} days "
            f"on average. Unlike this tool's listed-company reports, no "
            f"z-score, market-model beta, or permutation-test p-value is "
            f"computed here: those all assume a daily-moving price to measure "
            f"unusualness against, which this data does not provide. What "
            f"follows instead is each revision-to-revision gap, ranked by the "
            f"size of the change, alongside whatever coverage this run found "
            f"published in that same window."
        )
        paragraphs.append(
            "A large move here is not necessarily a market reaction: "
            "UnlistedZone's indicative series does not appear to be adjusted "
            "for corporate actions such as bonus issues, stock splits, or "
            "rights issues, any of which mechanically changes the per-share "
            "price without changing what the company is worth. A move that "
            "looks unusually large is worth checking against the company's "
            "own corporate-action history before reading it as sentiment."
        )
        top = max(moves, key=lambda m: abs(m.change))
        paragraphs.append(
            f"Across {news_count} collected {_plural(news_count, 'item')} of "
            f"coverage, the largest single move was {_pct(top.change)}, between "
            f"{top.start_date:%d %B %Y} and {top.end_date:%d %B %Y}."
        )
    else:
        paragraphs.append(
            f"The indicative price series in this window did not contain "
            f"enough revisions to identify any gap to examine — either the "
            f"price was not revised, or only one data point fell inside the "
            f"requested range."
        )

    return paragraphs
