"""Concurrency-correctness test for discovery.discover().

Proves sources actually run at the same time rather than one after another -
total wall-clock time should track the slowest single source, not the sum of
all of them - using fake strategy functions with a controlled, deterministic
delay instead of real network calls.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ceia.discovery as discovery_module  # noqa: E402
from ceia.discovery import Candidate, discover  # noqa: E402


def _slow_strategy(delay: float, source: str):
    def strategy(fetcher, start, end):
        time.sleep(delay)
        return [Candidate(f"https://x.example.com/{source}/1", source)]
    return strategy


class TestDiscoverConcurrency:
    def test_four_sources_run_concurrently_not_sequentially(self, monkeypatch):
        per_source_delay = 0.15
        fake_strategies = {
            "economic_times": _slow_strategy(per_source_delay, "economic_times"),
            "financial_express": _slow_strategy(per_source_delay, "financial_express"),
            "business_line": _slow_strategy(per_source_delay, "business_line"),
            "moneycontrol": _slow_strategy(per_source_delay, "moneycontrol"),
        }
        monkeypatch.setattr(discovery_module, "STRATEGIES", fake_strategies)

        start = time.monotonic()
        candidates, status = discover(
            fetcher=None, sources=list(fake_strategies), start=None, end=None,
        )
        elapsed = time.monotonic() - start

        assert len(candidates) == 4
        assert all(s.startswith("ok:") for s in status.values())
        # Sequential would take ~4 * per_source_delay = 0.6s; concurrent
        # should track a single source's delay plus scheduling overhead.
        assert elapsed < per_source_delay * 2.5, (
            f"took {elapsed:.2f}s for 4 sources at {per_source_delay}s each - "
            "looks sequential, not concurrent"
        )

    def test_a_failing_source_does_not_block_the_others(self, monkeypatch):
        def failing(fetcher, start, end):
            raise RuntimeError("simulated source outage")

        fake_strategies = {
            "economic_times": _slow_strategy(0.05, "economic_times"),
            "financial_express": failing,
        }
        monkeypatch.setattr(discovery_module, "STRATEGIES", fake_strategies)

        candidates, status = discover(
            fetcher=None, sources=list(fake_strategies), start=None, end=None,
        )

        assert len(candidates) == 1
        assert status["economic_times"].startswith("ok:")
        assert status["financial_express"].startswith("failed:")

    def test_unknown_source_is_reported_without_running_anything(self, monkeypatch):
        monkeypatch.setattr(discovery_module, "STRATEGIES", {})
        candidates, status = discover(
            fetcher=None, sources=["not_a_real_source"], start=None, end=None,
        )
        assert candidates == []
        assert status == {"not_a_real_source": "unknown source"}
