"""Tests for RunConfig's own validation.

An inverted date range doesn't raise anywhere in discovery.py - every
strategy's date loop (_days/_months) just never yields, so every source
silently reports zero candidates. That reads exactly like "scraping is
broken" rather than "the date range is wrong", so RunConfig catches it at
construction time instead.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.models import RunConfig  # noqa: E402


class TestRunConfigDateValidation:
    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="after end date"):
            RunConfig(company="Testco", ticker="TEST.NS",
                     start=date(2023, 2, 1), end=date(2023, 1, 1))

    def test_start_equal_to_end_is_allowed(self):
        config = RunConfig(company="Testco", ticker="TEST.NS",
                           start=date(2023, 1, 1), end=date(2023, 1, 1))
        assert config.start == config.end

    def test_start_before_end_is_allowed(self):
        config = RunConfig(company="Testco", ticker="TEST.NS",
                           start=date(2023, 1, 1), end=date(2023, 1, 31))
        assert config.start < config.end
