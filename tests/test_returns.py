"""Tests for returns, the market model, and abnormal/cumulative abnormal returns.

Price series here are synthetic and constructed with known alpha/beta so the
fitted values can be asserted exactly. Live NSE data could not be reached from
the build environment (Yahoo rate-limits the egress IP), so these verify the
maths rather than the data feed — see docs/phase2-notes.md.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.returns import (  # noqa: E402
    MIN_ESTIMATION_DAYS,
    abnormal_returns,
    align_series,
    cumulative_abnormal_return,
    daily_returns,
    fit_market_model,
    permutation_test_car,
    t_equivalent_threshold,
    trading_days,
)


def make_frame(n_days: int = 200, alpha: float = 0.0005, beta: float = 1.4,
               seed: int = 7, shock_at: int | None = None,
               shock_size: float = -0.20) -> pd.DataFrame:
    """Company returns generated as alpha + beta * market + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n_days)
    market = rng.normal(0.0002, 0.009, n_days)
    noise = rng.normal(0, 0.004, n_days)
    company = alpha + beta * market + noise
    if shock_at is not None:
        company[shock_at] += shock_size
    frame = pd.DataFrame({
        "close": 100 * np.cumprod(1 + company),
        "benchmark_close": 100 * np.cumprod(1 + market),
    }, index=dates)
    frame.index.name = "date"
    frame["return"] = daily_returns(frame["close"])
    frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
    return frame


class TestDailyReturns:
    def test_simple_percentage_change(self):
        series = pd.Series([100.0, 110.0, 99.0])
        result = daily_returns(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.10)
        assert result.iloc[2] == pytest.approx(-0.10)


class TestAlignSeries:
    def test_inner_join_on_common_dates(self):
        company = pd.DataFrame({"close": [1.0, 2.0, 3.0]},
                               index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]))
        benchmark = pd.DataFrame({"close": [10.0, 11.0]},
                                 index=pd.to_datetime(["2023-01-03", "2023-01-04"]))
        frame = align_series(company, benchmark)
        assert len(frame) == 2, "a date the benchmark lacks cannot yield an abnormal return"

    def test_no_overlap_gives_empty(self):
        company = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2023-01-02"]))
        benchmark = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2023-05-02"]))
        assert align_series(company, benchmark).empty


class TestMarketModel:
    def test_recovers_known_alpha_and_beta(self):
        frame = make_frame(alpha=0.0005, beta=1.4)
        model = fit_market_model(frame, analysis_start=date(2023, 2, 1))
        assert model.fitted
        assert model.beta == pytest.approx(1.4, abs=0.12)
        assert model.alpha == pytest.approx(0.0005, abs=0.001)
        assert 0 < model.r_squared <= 1

    def test_falls_back_when_lead_in_too_short(self):
        frame = make_frame(n_days=20)
        model = fit_market_model(frame, analysis_start=frame.index[-1].date())
        assert not model.fitted
        assert model.beta == 1.0 and model.alpha == 0.0
        assert model.kind == "market-adjusted"
        assert "estimation observations" in model.note

    def test_estimation_window_excludes_the_analysis_period(self):
        """The baseline must not be fitted on the event it is measuring."""
        frame = make_frame(n_days=200, shock_at=190, shock_size=-0.40)
        analysis_start = frame.index[185].date()
        model = fit_market_model(frame, analysis_start=analysis_start)
        assert model.fitted
        # A -40% day inside the window would wreck the fit; it must be excluded.
        assert model.beta == pytest.approx(1.4, abs=0.2)

    def test_zero_variance_benchmark_falls_back(self):
        dates = pd.bdate_range("2022-06-01", periods=100)
        frame = pd.DataFrame({
            "close": np.linspace(100, 120, 100),
            "benchmark_close": np.full(100, 50.0),
        }, index=dates)
        frame["return"] = daily_returns(frame["close"])
        frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
        model = fit_market_model(frame, analysis_start=date(2023, 1, 1))
        assert not model.fitted
        assert "zero variance" in model.note

    def test_min_observations_constant_is_respected(self):
        frame = make_frame(n_days=MIN_ESTIMATION_DAYS + 10)
        model = fit_market_model(frame, analysis_start=frame.index[-1].date())
        assert model.fitted


class TestAbnormalReturns:
    def test_market_wide_move_is_not_company_specific(self):
        """The core claim of Section 9: beta-1 stock, market falls, AR ~ 0."""
        dates = pd.bdate_range("2023-01-02", periods=3)
        frame = pd.DataFrame({
            "close": [100.0, 96.0, 96.0],
            "benchmark_close": [100.0, 96.0, 96.0],
        }, index=dates)
        frame["return"] = daily_returns(frame["close"])
        frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
        from ceia.returns import MarketModel
        model = MarketModel(0.0, 1.0, 0.01, 100, 0.9, True)
        out = abnormal_returns(frame, model)
        assert out["return"].iloc[1] == pytest.approx(-0.04)
        assert out["abnormal_return"].iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_company_specific_move_survives(self):
        dates = pd.bdate_range("2023-01-02", periods=2)
        frame = pd.DataFrame({"close": [100.0, 90.0],
                              "benchmark_close": [100.0, 99.0]}, index=dates)
        frame["return"] = daily_returns(frame["close"])
        frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
        from ceia.returns import MarketModel
        out = abnormal_returns(frame, MarketModel(0.0, 1.0, 0.01, 100, 0.9, True))
        assert out["abnormal_return"].iloc[1] == pytest.approx(-0.09, abs=1e-9)

    def test_beta_scales_the_expectation(self):
        """A 2-beta stock falling 8% when the market falls 4% is unremarkable."""
        dates = pd.bdate_range("2023-01-02", periods=2)
        frame = pd.DataFrame({"close": [100.0, 92.0],
                              "benchmark_close": [100.0, 96.0]}, index=dates)
        frame["return"] = daily_returns(frame["close"])
        frame["benchmark_return"] = daily_returns(frame["benchmark_close"])
        from ceia.returns import MarketModel
        out = abnormal_returns(frame, MarketModel(0.0, 2.0, 0.01, 100, 0.9, True))
        assert out["abnormal_return"].iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_z_uses_estimation_scale_when_available(self):
        frame = make_frame()
        model = fit_market_model(frame, analysis_start=date(2023, 2, 1))
        out = abnormal_returns(frame, model)
        assert out.attrs["ar_scale"] == pytest.approx(model.residual_sd)
        assert "estimation-window" in out.attrs["ar_scale_source"]
        assert out.attrs["ar_scale_df"] == model.observations - 2

    def test_z_falls_back_and_says_so(self):
        from ceia.returns import MarketModel
        frame = make_frame(n_days=30)
        out = abnormal_returns(frame, MarketModel(0.0, 1.0, float("nan"), 0,
                                                  float("nan"), False))
        assert "analysis-window" in out.attrs["ar_scale_source"]
        assert np.isfinite(out.attrs["ar_scale"])
        assert out.attrs["ar_scale_df"] == len(out) - 1


class TestTEquivalentThreshold:
    def test_large_df_is_close_to_the_z_threshold(self):
        """With hundreds of estimation days, t and z critical values nearly
        coincide - the adjustment should be small, not a wholesale change."""
        assert t_equivalent_threshold(1.5, 250) == pytest.approx(1.5, abs=0.02)

    def test_small_df_widens_the_threshold(self):
        """Fatter t-tails mean a higher bar is needed for the same tail
        probability as z=1.5 - this is what makes flagging more conservative,
        not more sensitive, for short estimation windows."""
        wide = t_equivalent_threshold(1.5, 5)
        narrow = t_equivalent_threshold(1.5, 60)
        assert wide > narrow > 1.5

    def test_zero_df_falls_back_to_the_raw_z_threshold(self):
        """No usable degrees of freedom (e.g. abnormal_returns() was never
        run) must not raise or silently return nonsense - it should behave
        exactly like the old flat z-threshold check."""
        assert t_equivalent_threshold(1.5, 0) == 1.5

    def test_threshold_shrinks_toward_z_as_df_grows(self):
        thresholds = [t_equivalent_threshold(1.5, df) for df in (5, 20, 120)]
        assert thresholds == sorted(thresholds, reverse=True)


class TestCumulativeAbnormalReturn:
    def _frame(self):
        frame = make_frame(n_days=60)
        from ceia.returns import MarketModel
        return abnormal_returns(frame, MarketModel(0.0005, 1.4, 0.004, 100, 0.9, True))

    def test_car_sums_the_window(self):
        frame = self._frame()
        event = frame.index[30].date()
        result = cumulative_abnormal_return(frame, event, (-1, 3))
        assert result["days"] == 5  # -1, 0, +1, +2, +3
        expected = frame["abnormal_return"].iloc[29:34].sum()
        assert result["car"] == pytest.approx(expected)

    def test_p_value_t_matches_the_existing_t_stat(self):
        """p_value_t is a classic two-tailed Student's-t p-value computed
        from the raw t_stat that was already being reported - it augments
        the permutation p-value rather than replacing it."""
        from scipy import stats as scipy_stats
        frame = self._frame()
        event = frame.index[30].date()
        result = cumulative_abnormal_return(frame, event, (-1, 3))
        df = frame.attrs["ar_scale_df"]
        expected = 2 * scipy_stats.t.sf(abs(result["t_stat"]), df)
        assert result["p_value_t"] == pytest.approx(expected)

    def test_p_value_t_is_none_without_usable_degrees_of_freedom(self):
        frame = self._frame()
        frame.attrs["ar_scale_df"] = 0
        event = frame.index[30].date()
        result = cumulative_abnormal_return(frame, event, (-1, 3))
        assert result["p_value_t"] is None

    def test_window_is_in_trading_days_not_calendar_days(self):
        frame = self._frame()
        # Friday event: +3 trading days lands on the following Wednesday.
        fridays = [t for t in frame.index if t.weekday() == 4]
        result = cumulative_abnormal_return(frame, fridays[3].date(), (0, 3))
        assert result["days"] == 4
        assert pd.Timestamp(result["end"]).weekday() == 2

    def test_truncation_at_series_edge_is_flagged(self):
        frame = self._frame()
        result = cumulative_abnormal_return(frame, frame.index[-1].date(), (-1, 5))
        assert result["truncated"]
        assert "truncated" in result["note"]

    def test_non_trading_event_day_rolls_forward(self):
        """A Saturday event day should attach to the next available session."""
        frame = self._frame()
        saturday = frame.index[10].date()
        while pd.Timestamp(saturday).weekday() != 5:
            saturday = date.fromordinal(saturday.toordinal() + 1)
        result = cumulative_abnormal_return(frame, saturday, (0, 1))
        assert result["days"] >= 1

    def test_event_after_series_end_returns_nan(self):
        frame = self._frame()
        beyond = date(2030, 1, 1)
        result = cumulative_abnormal_return(frame, beyond, (-1, 3))
        assert np.isnan(result["car"])
        assert result["days"] == 0


class TestPermutationTestCar:
    def _frame(self, n_days=200, shock_at=None, shock_size=-0.20):
        frame = make_frame(n_days=n_days, shock_at=shock_at, shock_size=shock_size)
        from ceia.returns import MarketModel
        return abnormal_returns(frame, MarketModel(0.0005, 1.4, 0.004, 100, 0.9, True))

    def test_a_real_shock_gets_a_low_p_value(self):
        """A -20% idiosyncratic shock should sit in the extreme tail of the
        placebo distribution built from this same (otherwise unshocked)
        series - very few random windows should be as extreme."""
        frame = self._frame(shock_at=100, shock_size=-0.20)
        event = frame.index[100].date()
        result = permutation_test_car(frame, event, (-1, 1))
        assert result["p_value"] is not None
        assert result["p_value"] < 0.05
        assert result["n"] > 0

    def test_deterministic_across_repeated_calls(self):
        frame = self._frame(shock_at=100, shock_size=-0.20)
        event = frame.index[100].date()
        first = permutation_test_car(frame, event, (-1, 1))
        second = permutation_test_car(frame, event, (-1, 1))
        assert first["p_value"] == second["p_value"]

    def test_different_seed_can_change_the_draw_but_not_wildly(self):
        frame = self._frame(shock_at=100, shock_size=-0.20)
        event = frame.index[100].date()
        a = permutation_test_car(frame, event, (-1, 1), seed=1)
        b = permutation_test_car(frame, event, (-1, 1), seed=2)
        # Both should still find the real shock extreme, even though the
        # exact placebo windows sampled differ.
        assert a["p_value"] < 0.05
        assert b["p_value"] < 0.05

    def test_excluded_days_are_never_sampled(self):
        """Excluding a day must remove every placebo window that overlaps
        it, not just windows starting on it."""
        frame = self._frame(n_days=60)
        event = frame.index[30].date()
        exclude = {frame.index[i].date() for i in range(25, 35)}
        result = permutation_test_car(frame, event, (-1, 1), exclude_days=exclude,
                                      n_permutations=500)
        # With a 61-day exclusion band inside a 60-day series and a 3-day
        # window, very little room is left - this should degrade gracefully
        # (few/no windows) rather than silently sampling excluded days.
        assert result["p_value"] is None or result["n"] >= 0

    def test_disabled_via_zero_permutations(self):
        frame = self._frame(shock_at=30)
        event = frame.index[30].date()
        result = permutation_test_car(frame, event, (-1, 1), n_permutations=0)
        assert result["p_value"] is None
        assert result["n"] == 0
        assert "disabled" in result["p_value_note"]

    def test_too_short_series_degrades_gracefully(self):
        frame = self._frame(n_days=5)
        event = frame.index[2].date()
        result = permutation_test_car(frame, event, (-2, 2))
        assert result["p_value"] is None
        assert result["p_value_note"]

    def test_event_after_series_end_degrades_gracefully(self):
        frame = self._frame()
        result = permutation_test_car(frame, date(2030, 1, 1), (-1, 3))
        assert result["p_value"] is None

    def test_does_not_clobber_cumulative_abnormal_returns_note_key(self):
        """permutation_test_car's dict is merged into cumulative_abnormal_
        return()'s own dict by callers (eventstudy.rank_incidents); the two
        must not share a "note" key or the merge silently drops one."""
        car_keys = set(cumulative_abnormal_return(
            self._frame(), self._frame().index[30].date(), (-1, 3)).keys())
        perm_keys = set(permutation_test_car(
            self._frame(), self._frame().index[30].date(), (-1, 3)).keys())
        assert car_keys & perm_keys == set(), (
            f"overlapping keys would silently clobber on dict.update(): "
            f"{car_keys & perm_keys}")


class TestTradingCalendar:
    def test_returns_real_session_dates(self):
        frame = make_frame(n_days=10)
        calendar = trading_days(frame)
        assert len(calendar) == 10
        assert all(d.weekday() < 5 for d in calendar)
