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

from ceia.models import RunConfig, _strip_corporate_suffix  # noqa: E402


class TestStripCorporateSuffix:
    """News headlines almost never carry a company's full legal suffix - the
    Indian financial press writes "Sonata Software", not "Sonata Software
    Limited" - so an alias list built only from the exact company name can
    silently match nothing. This is the fix: derive a short form by default.
    """

    def test_strips_limited(self):
        assert _strip_corporate_suffix("Sonata Software Limited") == "Sonata Software"

    def test_strips_ltd_with_trailing_period(self):
        assert _strip_corporate_suffix("Tata Consultancy Services Ltd.") == \
            "Tata Consultancy Services"

    def test_strips_private_limited_as_one_unit(self):
        """Must not stop at the bare "Limited" suffix and leave "Private"
        dangling on the end of the stripped name."""
        assert _strip_corporate_suffix("ABC Private Limited") == "ABC"

    def test_strips_pvt_ltd_abbreviation(self):
        assert _strip_corporate_suffix("XYZ Pvt Ltd") == "XYZ"
        assert _strip_corporate_suffix("XYZ Pvt. Ltd.") == "XYZ"

    def test_strips_inc(self):
        assert _strip_corporate_suffix("Infosys BPO Inc.") == "Infosys BPO"

    def test_name_with_no_suffix_returns_none(self):
        """Must not change existing behaviour for names that already work,
        e.g. the validated "Adani Enterprises" case."""
        assert _strip_corporate_suffix("Adani Enterprises") is None
        assert _strip_corporate_suffix("Reliance Industries") is None
        assert _strip_corporate_suffix("Wipro") is None

    def test_bare_suffix_alone_returns_none(self):
        """Stripping "Limited" down to "" would add a useless empty alias."""
        assert _strip_corporate_suffix("Limited") is None

    def test_suffix_only_matches_at_the_end(self):
        """"Limited" appearing mid-name (an unusual but possible case) must
        not be stripped from the middle."""
        assert _strip_corporate_suffix("Limited Edition Motors") is None


class TestRunConfigAllAliasesSuffixStripping:
    def test_short_form_is_added_automatically(self):
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS")
        assert "Sonata Software" in config.all_aliases

    def test_short_form_does_not_duplicate_a_user_supplied_alias(self):
        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS",
                           aliases=["Sonata Software"])
        lowered = [a.lower() for a in config.all_aliases]
        assert lowered.count("sonata software") == 1

    def test_no_extra_alias_when_company_name_has_no_suffix(self):
        config = RunConfig(company="Adani Enterprises", ticker="ADANIENT.NS",
                           aliases=["Adani", "AEL"])
        assert config.all_aliases == ["Adani Enterprises", "Adani", "AEL"]

    def test_the_real_case_this_was_built_for(self):
        """Reproduces the exact run that missed all of Sonata Software's
        coverage: the user's aliases were just the auto-included full legal
        name plus a mismatched ticker guess, neither of which appears in
        ordinary press text. Confirms the fix end-to-end through the real
        relevance scorer, not just checking the alias list shape."""
        from ceia.relevance import score_item

        config = RunConfig(company="Sonata Software Limited", ticker="SONATSOFTW.NS",
                           aliases=["SONATASOF"])
        headline = "Sonata Software Q4 results: Net profit rises 21.4% YoY"
        body = ("Sonata Software delivered a resilient FY26 despite macroeconomic "
                "challenges. Sonata Software shares jumped 10% after the results.")

        before = score_item(headline, body, ["Sonata Software Limited", "SONATASOF"],
                            ticker=config.ticker)
        assert before.score == 0.0, "the original bug: nothing matched at all"

        after = score_item(headline, body, config.all_aliases, ticker=config.ticker)
        assert after.score >= 0.35, "the fix: the auto short form now matches"
        assert "Sonata Software" in after.matched


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
