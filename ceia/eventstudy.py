"""Incident detection and ranking — PRD Section 8.

For each trading day, coverage is aggregated (volume, sentiment) and compared
against that day's abnormal return. A day is a candidate "incident" when
**both** are unusual: coverage that stands out against this company's own
baseline, *and* an abnormal return that stands out against the estimation
window's residual spread.

Requiring both is what keeps the output honest. Unusual coverage alone is just a
busy news day; an unusual return alone is a move with no visible explanation.
The tool claims only that the two coincided, which is why every label in the
output says "coincided with" and never "caused".

A note on what the ranking is *not*. The combined score orders candidates for a
reader's attention. It is not a p-value and not a test statistic. The t-values
reported alongside CAR come from a single company over a handful of events,
where the independence assumptions behind them do not hold — they are printed
because they are conventional, and immediately qualified for the same reason.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .dedupe import cluster_sizes
from .emotion import valence_of
from .models import NewsItem
from .returns import cumulative_abnormal_return

log = logging.getLogger(__name__)

# A day needs to clear both bars to be a candidate.
DEFAULT_COVERAGE_Z = 1.0
DEFAULT_RETURN_Z = 1.5
# Below this many observations, z-scores against the company's own baseline are
# too unstable to lean on, and the run says so instead of pretending otherwise.
MIN_DAYS_FOR_BASELINE = 10


@dataclass
class DailyCoverage:
    day: date
    item_count: int = 0
    unique_count: int = 0
    sources: list[str] = field(default_factory=list)
    mean_sentiment: float = 0.0
    weighted_sentiment: float = 0.0
    min_sentiment: float = 0.0
    max_sentiment: float = 0.0
    dominant_event: str = ""
    dominant_emotion: str = ""
    headlines: list[str] = field(default_factory=list)
    after_close_count: int = 0


@dataclass
class Incident:
    day: date
    abnormal_return: float
    abnormal_return_z: float
    raw_return: float
    benchmark_return: float
    coverage_z: float
    sentiment_z: float
    item_count: int
    mean_sentiment: float
    dominant_event: str
    dominant_emotion: str
    volume: float
    volume_z: float
    score: float
    direction_agrees: bool
    car: dict = field(default_factory=dict)
    headlines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["day"] = self.day.isoformat()
        return record


def aggregate_by_day(items: list[NewsItem]) -> dict[date, DailyCoverage]:
    """Roll news up onto the trading day each item was attributed to.

    Duplicates are excluded from the sentiment average so one syndicated wire
    story carried by four outlets does not count four times — but the outlets
    that carried it are still recorded, since breadth of pickup is a real
    signal about how much attention a story got.
    """
    carriers = cluster_sizes(items)
    by_day: dict[date, DailyCoverage] = {}

    for item in items:
        if item.trading_day is None:
            continue  # Unattributed items are reported separately, never guessed.
        coverage = by_day.setdefault(item.trading_day, DailyCoverage(day=item.trading_day))
        coverage.item_count += 1
        if item.source not in coverage.sources:
            coverage.sources.append(item.source)
        if item.after_close:
            coverage.after_close_count += 1
        if item.duplicate_of is not None:
            continue
        coverage.unique_count += 1
        coverage.headlines.append(item.headline)

    for day, coverage in by_day.items():
        unique_items = [i for i in items
                        if i.trading_day == day and i.duplicate_of is None]
        if not unique_items:
            continue
        scores = [i.sentiment_score for i in unique_items]
        coverage.mean_sentiment = float(np.mean(scores))
        coverage.min_sentiment = float(np.min(scores))
        coverage.max_sentiment = float(np.max(scores))

        # Weight by relevance and by how many outlets carried the story: a
        # front-page story picked up everywhere should move the day's tone more
        # than a single passing item that scraped past the threshold.
        weights = [max(i.relevance_score, 0.01) * carriers.get(i.url, 1)
                   for i in unique_items]
        total = sum(weights) or 1.0
        coverage.weighted_sentiment = float(
            sum(s * w for s, w in zip(scores, weights)) / total)

        # Counter.most_common() rather than max(set(x), key=x.count): the
        # latter breaks a tied count via set iteration order, which for str
        # keys depends on Python's per-process hash randomisation (verified
        # directly - the same tied input returned different "dominant"
        # labels across nine different PYTHONHASHSEED values). That would
        # have meant a day with two items each carrying a different label
        # could report a different "dominant" event or emotion on every run
        # of identical data, undermining the reproducibility the rest of the
        # pipeline works hard for. Counter.most_common() is documented to
        # break ties by first-encountered order, which is deterministic
        # given the (already deterministic) order items were collected in.
        categories = [i.event_category for i in unique_items if i.event_category]
        if categories:
            coverage.dominant_event = Counter(categories).most_common(1)[0][0]

        # Same mode approach as dominant_event, but only over items where an
        # emotion actually cleared the confidence threshold - a day where
        # GoEmotions stayed silent on every headline should not be forced
        # into an arbitrary label.
        emotions = [i.emotion_label for i in unique_items if i.emotion_label]
        if emotions:
            coverage.dominant_emotion = Counter(emotions).most_common(1)[0][0]
    return by_day


def _z(value: float, mean: float, sd: float) -> float:
    if not sd or not math.isfinite(sd):
        return 0.0
    return (value - mean) / sd


def build_daily_table(
    frame: pd.DataFrame,
    coverage: dict[date, DailyCoverage],
    start: date,
    end: date,
) -> pd.DataFrame:
    """One row per trading day in the analysis window, prices joined to news."""
    window = frame.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
    rows = []
    for timestamp, row in window.iterrows():
        day = timestamp.date()
        day_coverage = coverage.get(day)
        rows.append({
            "date": day,
            "close": row["close"],
            "volume": row.get("volume", float("nan")),
            "return": row["return"],
            "benchmark_return": row["benchmark_return"],
            "expected_return": row.get("expected_return", float("nan")),
            "abnormal_return": row["abnormal_return"],
            "abnormal_return_z": row.get("abnormal_return_z", float("nan")),
            "item_count": day_coverage.item_count if day_coverage else 0,
            "unique_count": day_coverage.unique_count if day_coverage else 0,
            "mean_sentiment": day_coverage.mean_sentiment if day_coverage else 0.0,
            "weighted_sentiment": day_coverage.weighted_sentiment if day_coverage else 0.0,
            "dominant_event": day_coverage.dominant_event if day_coverage else "",
            "dominant_emotion": day_coverage.dominant_emotion if day_coverage else "",
            "sources": ",".join(day_coverage.sources) if day_coverage else "",
        })
    if not rows:
        # A window with no trading days at all (a bad date range, or a holiday
        # stretch). Return an empty frame with the right columns so callers can
        # treat it uniformly instead of special-casing a KeyError.
        empty = pd.DataFrame(columns=[
            "close", "volume", "return", "benchmark_return", "expected_return",
            "abnormal_return", "abnormal_return_z", "item_count", "unique_count",
            "mean_sentiment", "weighted_sentiment", "dominant_event",
            "dominant_emotion", "sources", "coverage_z", "sentiment_z", "volume_z",
        ])
        empty.index.name = "date"
        return empty

    table = pd.DataFrame(rows).set_index("date")

    # Baselines are the company's own coverage over the analysis window.
    counts = table["unique_count"].astype(float)
    table["coverage_z"] = [
        _z(v, counts.mean(), counts.std(ddof=1)) for v in counts
    ]
    sentiments = table.loc[table["unique_count"] > 0, "weighted_sentiment"]
    s_mean = float(sentiments.mean()) if len(sentiments) else 0.0
    s_sd = float(sentiments.std(ddof=1)) if len(sentiments) > 1 else 0.0
    table["sentiment_z"] = [
        _z(v, s_mean, s_sd) if c > 0 else 0.0
        for v, c in zip(table["weighted_sentiment"], table["unique_count"])
    ]
    # Window-relative, same style as coverage_z - not a claim about "normal"
    # volume from before the window, just "unusual for this company in this
    # run". NaN throughout (no volume from this provider, e.g. a bare CSV) is
    # not an error; every _z() call on it correctly comes back 0.0.
    volumes = table["volume"].astype(float)
    v_mean = float(volumes.mean()) if volumes.notna().any() else 0.0
    v_sd = float(volumes.std(ddof=1)) if volumes.notna().sum() > 1 else 0.0
    table["volume_z"] = [
        _z(v, v_mean, v_sd) if pd.notna(v) else 0.0 for v in volumes
    ]
    return table


def rank_incidents(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    event_window: tuple[int, int] = (-1, 3),
    coverage_threshold: float = DEFAULT_COVERAGE_Z,
    return_threshold: float = DEFAULT_RETURN_Z,
    top_n: int | None = None,
) -> list[Incident]:
    """Flag and rank candidate incident days."""
    incidents: list[Incident] = []
    if table.empty:
        return incidents

    # The coverage baseline is the company's own coverage across the analysis
    # window. In a window short enough to be built *around* a known event, that
    # baseline is incoherent - every day is an event day, so no day looks
    # unusual relative to its neighbours and nothing would ever flag. Below the
    # threshold, "has any coverage at all" replaces the z-test, and the run's
    # caveats say the coverage bar was relaxed.
    days_with_news = int((table["unique_count"] > 0).sum())
    thin_baseline = days_with_news < MIN_DAYS_FOR_BASELINE

    for day, row in table.iterrows():
        if row["unique_count"] == 0:
            continue
        if thin_baseline:
            coverage_unusual = True
        else:
            coverage_unusual = (row["coverage_z"] >= coverage_threshold
                                or abs(row["sentiment_z"]) >= coverage_threshold)
        return_unusual = abs(row["abnormal_return_z"]) >= return_threshold
        if not (coverage_unusual and return_unusual):
            continue

        # Does the price move the way the coverage's tone would suggest? A
        # disagreement is not a failure - it is often the interesting case
        # (good news already priced in, say) - so it is surfaced, not filtered.
        sentiment = row["weighted_sentiment"]
        abnormal = row["abnormal_return"]
        agrees = bool(sentiment * abnormal > 0) if sentiment and abnormal else False

        score = (abs(row["abnormal_return_z"])
                 * (1 + max(row["coverage_z"], 0))
                 * (1 + abs(sentiment)))
        if agrees:
            score *= 1.25  # Direction agreement makes a candidate more legible.

        incidents.append(Incident(
            day=day,
            abnormal_return=float(abnormal),
            abnormal_return_z=float(row["abnormal_return_z"]),
            raw_return=float(row["return"]),
            benchmark_return=float(row["benchmark_return"]),
            coverage_z=float(row["coverage_z"]),
            sentiment_z=float(row["sentiment_z"]),
            item_count=int(row["unique_count"]),
            mean_sentiment=float(sentiment),
            dominant_event=str(row["dominant_event"]),
            dominant_emotion=str(row["dominant_emotion"]),
            volume=float(row["volume"]) if pd.notna(row["volume"]) else float("nan"),
            volume_z=float(row["volume_z"]),
            score=float(score),
            direction_agrees=agrees,
            car=cumulative_abnormal_return(frame, day, event_window),
            sources=str(row["sources"]).split(",") if row["sources"] else [],
        ))

    incidents.sort(key=lambda i: -i.score)
    return incidents[:top_n] if top_n else incidents


def attach_headlines(incidents: list[Incident], items: list[NewsItem],
                     limit: int = 5) -> list[Incident]:
    """Hang the source items behind each flag onto the incident (Section 8)."""
    for incident in incidents:
        relevant = [i for i in items
                    if i.trading_day == incident.day and i.duplicate_of is None]
        relevant.sort(key=lambda i: (-abs(i.sentiment_score), -i.relevance_score))
        incident.headlines = [
            f"[{i.source}] {i.headline} ({i.sentiment_label}, "
            f"rel={i.relevance_score:.2f})"
            for i in relevant[:limit]
        ]
    return incidents


def sentiment_return_correlation(table: pd.DataFrame) -> dict:
    """Pearson correlation between daily sentiment and abnormal return.

    Answers the tool's own founding question directly - does sentiment track
    price - as a single number, rather than only the per-day incident flags.
    Descriptive, not inferential: over the handful of trading days a typical
    run covers, this has a wide confidence interval and is not a claim of
    statistical significance, exactly the same caveat this tool already makes
    about everything else it reports. Only news-carrying days are included; a
    silent day forces sentiment to 0 by construction, which would dilute the
    correlation with manufactured non-signal rather than real absence of it.
    """
    covered = table[table["unique_count"] > 0] if len(table) else table
    # The first row of any price series has no prior close for pct_change()
    # to work with, so its return (and everything derived from it) is NaN by
    # construction - not a data problem, just how day one of a series works.
    # A single NaN silently poisons corrcoef's result to NaN with no error,
    # so it must be dropped explicitly rather than trusted to "just work".
    covered = covered.dropna(subset=["weighted_sentiment", "abnormal_return"])
    n = len(covered)
    if n < 3:
        return {
            "n": n, "r": None, "r_squared": None,
            "note": (f"only {n} news-carrying day(s) with a usable abnormal "
                     "return - too few to compute a meaningful correlation "
                     "(need at least 3)."),
        }
    sentiment = covered["weighted_sentiment"].to_numpy(dtype=float)
    abnormal = covered["abnormal_return"].to_numpy(dtype=float)
    if np.std(sentiment) == 0 or np.std(abnormal) == 0:
        return {
            "n": n, "r": None, "r_squared": None,
            "note": ("sentiment or abnormal return has zero variance across "
                     "covered days - correlation is undefined."),
        }
    r = float(np.corrcoef(sentiment, abnormal)[0, 1])
    return {
        "n": n, "r": round(r, 4), "r_squared": round(r * r, 4),
        "note": (f"Pearson r over {n} news-carrying day(s) in this run; "
                 "descriptive only, not a significance test, and not "
                 "comparable across runs with different day counts."),
    }


def emotion_valence_summary(table: pd.DataFrame) -> dict:
    """Mean abnormal return per GoEmotions sentiment group (PRD-adjacent extra).

    ``dominant_emotion`` is already one label per day (the most common
    surfaced emotion among that day's items — see ``aggregate_by_day``). This
    buckets those labels into the paper's own positive/negative/ambiguous
    groups (Demszky et al., 2020, Section 5.1; see ``ceia/emotion.py``) and
    reports the mean abnormal return for days in each bucket, alongside
    FinBERT's own weighted sentiment for the same days as a cross-check.

    Purely descriptive, like the correlation stat above: it groups days that
    already exist in the table, it does not change which days are incidents
    or how they are scored. Days with no surfaced emotion (blank
    ``dominant_emotion`` — the common case on formal financial-press
    headlines, see ``pick_emotions``) or no news at all are excluded, not
    folded into a fourth bucket, since a blank label means "GoEmotions had
    nothing confident to say," not "neutral valence."
    """
    if table.empty or "dominant_emotion" not in table.columns:
        return {"groups": {}, "note": "no daily table to summarise."}

    covered = table[table["dominant_emotion"].astype(str).str.len() > 0].copy()
    covered = covered.dropna(subset=["abnormal_return"])
    if covered.empty:
        return {
            "groups": {},
            "note": ("no day had a confident-enough dominant emotion to "
                     "group by valence."),
        }

    covered["valence"] = covered["dominant_emotion"].map(valence_of)
    covered = covered[covered["valence"] != ""]
    if covered.empty:
        return {
            "groups": {},
            "note": "no day's dominant emotion mapped to a known valence group.",
        }

    groups = {}
    for valence in ("positive", "negative", "ambiguous"):
        subset = covered[covered["valence"] == valence]
        if subset.empty:
            continue
        groups[valence] = {
            "n_days": int(len(subset)),
            "mean_abnormal_return": round(float(subset["abnormal_return"].mean()), 5),
            "mean_weighted_sentiment": round(float(subset["weighted_sentiment"].mean()), 4),
            "labels_seen": sorted(subset["dominant_emotion"].unique().tolist()),
        }

    return {
        "groups": groups,
        "note": (f"{len(covered)} day(s) had a confident dominant emotion "
                 "(GoEmotions, headline-only, secondary to FinBERT); grouped "
                 "by the paper's own positive/negative/ambiguous clustering. "
                 "Descriptive only — small day counts per group, and this "
                 "never affects incident flagging or ranking."),
    }


def caveats(table: pd.DataFrame, incidents: list[Incident],
            model_kind: str, scale_source: str) -> list[str]:
    """The limitations this specific run has to state (PRD Section 9)."""
    notes = [
        "This is a structured case study, not a statistically validated causal "
        "finding. One company over one date range yields too few distinct "
        "incidents for the sentiment/abnormal-return relationship to carry "
        "statistical significance; a proper event study spans dozens of "
        "companies and events.",
        "Flagged days are days on which unusual coverage COINCIDED WITH an "
        "unusual benchmark-adjusted return. Coincidence in time is not evidence "
        "that an article caused a price move.",
    ]
    days_with_news = int((table["unique_count"] > 0).sum()) if len(table) else 0
    if days_with_news < MIN_DAYS_FOR_BASELINE:
        notes.append(
            f"Only {days_with_news} trading day(s) in this window carried "
            "coverage — too few to say which days were unusually busy, so the "
            "coverage test was RELAXED to 'any coverage at all' and days were "
            "flagged on the abnormal return alone. Treat the ranking as "
            "'notable price moves that had coverage', not as evidence that the "
            "coverage was itself unusual. Widen the date range to restore the "
            "stricter test."
        )
    if model_kind == "market-adjusted":
        notes.append(
            "Abnormal returns use the market-adjusted model (beta fixed at 1.0) "
            "because there was not enough clean lead-in data to fit a market "
            "model. A high-beta stock will show a systematically inflated "
            "abnormal return under this assumption."
        )
    if scale_source and "analysis-window" in scale_source:
        notes.append(
            "Abnormal returns are standardised against the analysis window's own "
            "spread rather than a clean estimation window, which understates how "
            "unusual the largest moves are."
        )
    if len(incidents) <= 2:
        notes.append(
            f"{len(incidents)} candidate incident(s) were flagged. Rankings over "
            "so few candidates are indicative only."
        )
    disagreeing = [i for i in incidents if not i.direction_agrees]
    if disagreeing:
        notes.append(
            f"{len(disagreeing)} flagged day(s) show a price move in the opposite "
            "direction to the coverage's tone. That is not necessarily an error: "
            "news can be already priced in, or the day's move driven by something "
            "the coverage did not capture."
        )
    return notes
