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
    frame = pd.DataFrame({
        "close": company["close"],
        "benchmark_close": benchmark["close"],
    }).dropna()
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
        return MarketModel(
            alpha=0.0, beta=1.0, residual_sd=float("nan"),
            observations=len(window), r_squared=float("nan"), fitted=False,
            note=(f"only {len(window)} estimation observations before "
                  f"{analysis_start} (need {MIN_ESTIMATION_DAYS}); "
                  "using market-adjusted returns with beta fixed at 1.0"),
        )

    x = window["benchmark_return"].to_numpy()
    y = window["return"].to_numpy()
    if np.std(x) == 0:
        return MarketModel(0.0, 1.0, float("nan"), len(window), float("nan"),
                           False, "benchmark has zero variance in the "
                                  "estimation window; using market-adjusted returns")

    beta, alpha = np.polyfit(x, y, 1)
    residuals = y - (alpha + beta * x)
    # ddof=2 because two parameters were estimated.
    residual_sd = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float("nan")
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")

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
    if not np.isfinite(scale) or scale == 0:
        scale = float(out["abnormal_return"].std(ddof=1))
        scale_source = "analysis-window SD (weaker: inflated by the events in it)"
    out.attrs["ar_scale"] = scale
    out.attrs["ar_scale_source"] = scale_source
    out["abnormal_return_z"] = (
        out["abnormal_return"] / scale if scale and np.isfinite(scale) else np.nan
    )
    return out


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
    days = len(slice_)
    # Under the usual independence assumption the CAR's SD scales with sqrt(N).
    t_stat = (car / (scale * np.sqrt(days))
              if scale and np.isfinite(scale) and days else float("nan"))

    truncated = (position + window[0] < 0) or (position + window[1] > len(index) - 1)
    return {
        "car": car,
        "days": days,
        "start": index[start].date().isoformat(),
        "end": index[end].date().isoformat(),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else None,
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

    frame = align_series(company, index_frame)
    if frame.empty:
        raise PriceError(
            f"no overlapping trading dates for {ticker} and {benchmark}")

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


def trading_days(frame: pd.DataFrame) -> set[date]:
    """The real exchange calendar, for news attribution.

    This closes the Phase 1 gap where the weekday fallback treated exchange
    holidays (26 January, say) as tradeable.
    """
    return {ts.date() for ts in frame.index}
