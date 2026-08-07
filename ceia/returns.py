"""Daily returns and benchmark-adjusted (abnormal) returns — PRD Section 9.

The point of abnormal returns is stated plainly in the PRD: if the whole market
fell 4% and the company fell 4.5%, the company-specific move is −0.5%, not
−4.5%. Reading the raw return as news-driven is the single easiest way to get a
wrong answer here.

Two models are supported:

* **market-adjusted** — ``AR = R_i − R_m``. Assumes beta of 1. Needs no
  estimation window, so it works on short date ranges.
* **market-model** — ``AR = R_i − (α + β·R_m)``, with α and β fitted by OLS over
  an estimation window *before* the analysis period. More rigorous, and the
  version the PRD calls "the more rigorous market-model version". It needs
  enough clean lead-in data, so the engine falls back to market-adjusted and
  says so rather than fitting a beta on ten observations.

The estimation window deliberately ends before the analysis window starts, so
the "normal" baseline is not contaminated by the very event being measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from .prices import PriceError, PriceProvider, load_prices

log = logging.getLogger(__name__)

# Trading days of history to fit alpha/beta on. 120 is a common choice; the
# floor is what we will accept before giving up on the market model.
DEFAULT_ESTIMATION_DAYS = 120
MIN_ESTIMATION_DAYS = 40
# Gap between the estimation window and the analysis window, so news leaking
# into the run-up does not shape the baseline.
ESTIMATION_GAP_DAYS = 5
# Calendar days of price history fetched *after* the analysis window, so that
# after-close attribution and CAR windows have sessions to land on.
TAIL_BUFFER_DAYS = 21


@dataclass
class MarketModel:
    alpha: float
    beta: float
    residual_sd: float
    observations: int
    r_squared: float
    fitted: bool
    note: str = ""

    @property
    def kind(self) -> str:
        return "market-model" if self.fitted else "market-adjusted"


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns. Simple, not log, because they aggregate across
    assets additively — which is what subtracting a benchmark requires."""
    return close.pct_change()


def align_series(company: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on trading dates both series actually have.

    An inner join is the conservative choice: a date the benchmark is missing
    cannot yield an abnormal return, and forward-filling an index level would
    invent a 0% market move and push the whole gap into the abnormal term.
    """
    columns = {
        "close": company["close"],
        "benchmark_close": benchmark["close"],
    }
    # yfinance/Yahoo chart both return volume; a bare date,close CSV does not.
    # Carried through here (previously it was fetched all the way to this
    # point and then silently dropped - never read by anything downstream)
    # so it can be shown as a corroborating signal alongside price and news.
    if "volume" in company.columns:
        columns["volume"] = company["volume"]
    frame = pd.DataFrame(columns).dropna(subset=["close", "benchmark_close"])
    frame["return"] = daily_returns(frame["close"])
    frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
    return frame


def fit_market_model(
    frame: pd.DataFrame,
    analysis_start: date,
    estimation_days: int = DEFAULT_ESTIMATION_DAYS,
    gap_days: int = ESTIMATION_GAP_DAYS,
) -> MarketModel:
    """Fit ``R_i = α + β·R_m + ε`` on data before the analysis window."""
    cutoff = pd.Timestamp(analysis_start)
    window = frame.loc[frame.index < cutoff].dropna(
        subset=["return", "benchmark_return"])
    if gap_days and len(window) > gap_days:
        window = window.iloc[:-gap_days]
    window = window.tail(estimation_days)

    if len(window) < MIN_ESTIMATION_DAYS:
        note = (f"only {len(window)} estimation observations before "
                f"{analysis_start} (need {MIN_ESTIMATION_DAYS}); "
                "using market-adjusted returns with beta fixed at 1.0")
        # This is silent otherwise - the reason only ever showed up buried in
        # the final report's provenance section. A run that falls back here
        # is a real accuracy hit (a high-beta stock gets a systematically
        # inflated abnormal return), worth seeing the moment it happens
        # rather than discovering it after the fact.
        log.warning("market model: %s", note)
        return MarketModel(
            alpha=0.0, beta=1.0, residual_sd=float("nan"),
            observations=len(window), r_squared=float("nan"), fitted=False,
            note=note,
        )

    x = window["benchmark_return"].to_numpy()
    y = window["return"].to_numpy()
    if np.std(x) == 0:
        note = ("benchmark has zero variance in the estimation window; "
                "using market-adjusted returns")
        log.warning("market model: %s", note)
        return MarketModel(0.0, 1.0, float("nan"), len(window), float("nan"),
                           False, note)

    beta, alpha = np.polyfit(x, y, 1)
    residuals = y - (alpha + beta * x)
    # ddof=2 because two parameters were estimated.
    residual_sd = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float("nan")
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")

    log.info("market model: fitted on %d trading days before %s "
             "(alpha=%.5f beta=%.3f R2=%.3f)",
             len(window), analysis_start, alpha, beta, r_squared)
    return MarketModel(
        alpha=float(alpha), beta=float(beta), residual_sd=residual_sd,
        observations=len(window), r_squared=float(r_squared), fitted=True,
        note=f"fitted on {len(window)} trading days before {analysis_start}",
    )


def abnormal_returns(frame: pd.DataFrame, model: MarketModel) -> pd.DataFrame:
    """Add ``expected_return``, ``abnormal_return`` and a standardised score."""
    out = frame.copy()
    out["expected_return"] = model.alpha + model.beta * out["benchmark_return"]
    out["abnormal_return"] = out["return"] - out["expected_return"]

    # Standardise against the estimation-window residual spread where we have
    # one, else against the analysis period's own spread. The latter is weaker:
    # a period dominated by one big event inflates its own denominator and
    # understates how unusual that event was.
    scale = model.residual_sd
    scale_source = "estimation-window residual SD"
    # Degrees of freedom behind ``scale`` - two parameters (alpha, beta) were
    # fitted out of the estimation window, matching the ddof=2 already used
    # for residual_sd above; the analysis-window fallback below loses only
    # the one degree of freedom a plain sample SD costs. Carried alongside
    # scale so a *small* estimation window - not just a missing one - is
    # reflected in how strict the t-distribution-based tests below are.
    df = max(model.observations - 2, 0)
    if not np.isfinite(scale) or scale == 0:
        scale = float(out["abnormal_return"].std(ddof=1))
        scale_source = "analysis-window SD (weaker: inflated by the events in it)"
        df = max(len(out) - 1, 0)
    out.attrs["ar_scale"] = scale
    out.attrs["ar_scale_source"] = scale_source
    out.attrs["ar_scale_df"] = df
    out["abnormal_return_z"] = (
        out["abnormal_return"] / scale if scale and np.isfinite(scale) else np.nan
    )
    return out


def t_equivalent_threshold(z_threshold: float, df: int) -> float:
    """The t-distribution critical value with the same two-tailed tail
    probability a z-threshold has under the standard normal.

    Always >= ``z_threshold`` (equal only as ``df`` -> infinity), and more so
    the smaller ``df`` is - a short estimation window makes the same
    "how many standard deviations is unusual" input a harder bar to clear,
    which is the correct behaviour: a standard deviation estimated from few
    observations is itself less certain, and a z-test silently ignores that.
    Falls back to ``z_threshold`` unchanged when ``df`` is too small for a
    t-distribution to be meaningful (there's no better answer available).
    """
    if df <= 0:
        return z_threshold
    tail_prob = stats.norm.sf(z_threshold)
    return float(stats.t.isf(tail_prob, df))


def cumulative_abnormal_return(
    frame: pd.DataFrame, event_day: date, window: tuple[int, int]
) -> dict:
    """CAR over ``window`` trading days relative to ``event_day``.

    Window offsets are in **trading days**, not calendar days, so (-1, +3)
    around a Friday spans Thursday through the following Wednesday.
    """
    index = frame.index
    target = pd.Timestamp(event_day)
    positions = index.get_indexer([target], method="bfill")
    position = int(positions[0]) if positions[0] != -1 else -1
    if position == -1:
        return {"car": float("nan"), "days": 0, "start": None, "end": None,
                "note": "event day falls after the last trading day available"}

    start = max(0, position + window[0])
    end = min(len(index) - 1, position + window[1])
    slice_ = frame.iloc[start:end + 1]
    car = float(slice_["abnormal_return"].sum())

    scale = frame.attrs.get("ar_scale")
    df = frame.attrs.get("ar_scale_df", 0)
    days = len(slice_)
    # Under the usual independence assumption the CAR's SD scales with sqrt(N).
    t_stat = (car / (scale * np.sqrt(days))
              if scale and np.isfinite(scale) and days else float("nan"))
    # A classic two-tailed Student's-t p-value for that t_stat, using the
    # estimation window's own degrees of freedom (where ``scale`` came from -
    # see abnormal_returns()) rather than the CAR window's day count, since
    # it's the SD estimate's uncertainty this is testing against. Shown
    # alongside, not instead of, the permutation-test p-value below: the
    # permutation test makes no distributional assumption about returns at
    # all, which is the more rigorous of the two - this is the familiar
    # textbook number for a reader who wants it, not a replacement.
    p_value_t = (2 * float(stats.t.sf(abs(t_stat), df))
                if np.isfinite(t_stat) and df > 0 else None)

    truncated = (position + window[0] < 0) or (position + window[1] > len(index) - 1)
    return {
        "car": car,
        "days": days,
        "start": index[start].date().isoformat(),
        "end": index[end].date().isoformat(),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else None,
        "p_value_t": p_value_t,
        "truncated": truncated,
        "note": ("window truncated at the edge of the available price series"
                 if truncated else ""),
    }


def build(
    ticker: str,
    benchmark: str,
    start: date,
    end: date,
    lead_in_days: int = 200,
    providers: list[PriceProvider] | None = None,
    estimation_days: int = DEFAULT_ESTIMATION_DAYS,
    tail_days: int = TAIL_BUFFER_DAYS,
) -> tuple[pd.DataFrame, MarketModel, dict]:
    """Load prices and return ``(frame, model, metadata)``.

    ``lead_in_days`` is calendar days of history fetched *before* ``start``, to
    give the market model something to fit on (PRD Section 8).

    ``tail_days`` extends the series *past* ``end``, which is not cosmetic: news
    published after the close on the last day of the window needs the following
    trading day to exist, and a CAR window of (-1, +3) reaches three sessions
    beyond any incident on that day. Without the tail, both silently degrade -
    the news becomes unattributed and the CAR comes back truncated.
    """
    fetch_start = start - timedelta(days=lead_in_days)
    fetch_end = end + timedelta(days=tail_days)
    company, company_provider = load_prices(ticker, fetch_start, fetch_end, providers)
    index_frame, benchmark_provider = load_prices(benchmark, fetch_start, fetch_end, providers)
    log.info("prices: %s via %s (%d rows), %s via %s (%d rows), requested "
             "%s to %s (%d lead-in day(s))",
             ticker, company_provider, len(company),
             benchmark, benchmark_provider, len(index_frame),
             fetch_start, fetch_end, lead_in_days)

    frame = align_series(company, index_frame)
    if frame.empty:
        raise PriceError(
            f"no overlapping trading dates for {ticker} and {benchmark}")
    log.info("prices: %d rows share a trading date on both series "
             "(dropped %d company-only, %d benchmark-only)",
             len(frame), len(company) - len(frame), len(index_frame) - len(frame))

    model = fit_market_model(frame, start, estimation_days=estimation_days)
    frame = abnormal_returns(frame, model)

    analysis = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    metadata = {
        "ticker": ticker,
        "benchmark": benchmark,
        "company_provider": company_provider,
        "benchmark_provider": benchmark_provider,
        "lead_in_start": fetch_start.isoformat(),
        "tail_end": fetch_end.isoformat(),
        "rows_total": len(frame),
        "rows_in_analysis_window": len(analysis),
        "model": model.kind,
        "alpha": model.alpha,
        "beta": model.beta,
        "r_squared": model.r_squared,
        "estimation_observations": model.observations,
        "model_note": model.note,
        "ar_scale_source": frame.attrs.get("ar_scale_source"),
    }
    return frame, model, metadata


# Default draws for the permutation test below. 2000 is enough for a stable
# empirical p-value to 2-3 significant figures while staying fast even on a
# multi-year price series; the function itself uses fewer when the series
# does not offer that many non-overlapping placebo windows.
DEFAULT_PERMUTATIONS = 2000


def permutation_test_car(
    frame: pd.DataFrame,
    event_day: date,
    window: tuple[int, int],
    exclude_days: set[date] = frozenset(),
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
) -> dict:
    """Empirical p-value for a CAR via placebo (pseudo-event) resampling.

    The t-stat next to CAR assumes independent, normally distributed abnormal
    returns spanning a large sample - an assumption a single company's own
    handful of trading days does not meet, and every place that t-stat is
    printed says so. This does not fix that assumption; it sidesteps it:
    draw many random same-length windows from this same abnormal-return
    series - excluding any day already flagged as a candidate, so the null
    distribution is not contaminated by the very events being tested - and
    ask what fraction of those placebo CARs are at least as extreme (two-
    sided) as the real one. That fraction *is* the p-value, by construction,
    for this specific company, series and window length - no distributional
    assumption required, at the cost of only being valid for this one run
    (it says nothing about whether the effect would replicate elsewhere).

    Deterministic by default (fixed seed): re-running the same analysis
    reproduces the same p-value, matching the reproducibility the rest of
    this pipeline works hard for (see the dominant_emotion/dominant_event
    tie-break note in ``eventstudy.py``).
    """
    if n_permutations <= 0:
        return {"p_value": None, "n": 0, "p_value_note": "permutation test disabled."}

    # Note the "p_value_note" key name rather than "note": the caller merges
    # this dict into cumulative_abnormal_return()'s own result, which already
    # has a "note" key (e.g. "window truncated..."); a same-named key here
    # would silently clobber it rather than error, exactly the kind of quiet
    # bug this project has repeatedly hunted down elsewhere.
    real = cumulative_abnormal_return(frame, event_day, window)
    real_car = real.get("car")
    days_span = real.get("days") or 0
    if real_car is None or not np.isfinite(real_car) or days_span < 1:
        return {"p_value": None, "n": 0,
                "p_value_note": "actual CAR unavailable - nothing to test against."}

    usable = frame.dropna(subset=["abnormal_return"])
    ar = usable["abnormal_return"].to_numpy()
    if len(ar) < days_span + 1:
        return {"p_value": None, "n": 0,
                "p_value_note": (f"only {len(ar)} usable trading day(s) in the "
                                f"series - too few to draw {days_span}-day "
                                "placebo windows.")}

    excluded_positions = {i for i, ts in enumerate(usable.index)
                          if ts.date() in exclude_days}
    max_start = len(ar) - days_span
    candidates = [
        s for s in range(max_start + 1)
        if not any(p in excluded_positions for p in range(s, s + days_span))
    ]
    if len(candidates) < 10:
        return {"p_value": None, "n": 0,
                "p_value_note": ("too few non-flagged windows available in "
                                 "this price series to build a null "
                                 "distribution.")}

    rng = np.random.default_rng(seed)
    if len(candidates) <= n_permutations:
        starts = candidates
    else:
        starts = rng.choice(candidates, size=n_permutations, replace=False)

    placebo_cars = np.array([ar[s:s + days_span].sum() for s in starts])
    p_value = float(np.mean(np.abs(placebo_cars) >= abs(real_car)))

    return {
        "p_value": round(p_value, 4),
        "n": len(starts),
        "p_value_note": (f"empirical p-value from {len(starts)} placebo "
                         f"window(s) of the same {days_span}-day length, "
                         "drawn from this company's own abnormal-return "
                         f"series (excluding other flagged days); "
                         f"deterministic (seed={seed})."),
    }


def trading_days(frame: pd.DataFrame) -> set[date]:
    """The real exchange calendar, for news attribution.

    This closes the Phase 1 gap where the weekday fallback treated exchange
    holidays (26 January, say) as tradeable.
    """
    return {ts.date() for ts in frame.index}
