"""Tests for PDF export.

The happy path needs a real Chromium binary, which is not guaranteed to be
present (or resolvable at Playwright's expected path) in every environment
this test suite runs in - it self-skips rather than failing hard when no
working browser can be found, via the same code path ``render_pdf`` itself
uses. The error-handling paths (missing dependency, broken render) are
tested without a browser at all, since those are exactly the paths a real
user without ``playwright install chromium`` run will hit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ceia.pdf import PdfExportError, render_pdf  # noqa: E402

# Set by CI/this sandbox's provisioning when Playwright's pip package and its
# pre-fetched browser binary are pinned to mismatched revisions - see
# render_pdf's own docstring. Real installs following the documented
# `playwright install chromium` step never need this.
_SANDBOX_CHROMIUM = "/opt/pw-browsers/chromium"


def _working_browser_path() -> str | None:
    """The path render_pdf's happy-path tests should launch, or None to skip
    them - never guess a browser exists, ask Playwright directly."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    candidates = [None, _SANDBOX_CHROMIUM if os.path.exists(_SANDBOX_CHROMIUM) else None]
    with sync_playwright() as pw:
        for candidate in candidates:
            if candidate is False:
                continue
            try:
                kwargs = {"executable_path": candidate} if candidate else {}
                browser = pw.chromium.launch(**kwargs)
                browser.close()
                return candidate
            except Exception:
                continue
    return None


def _playwright_importable() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


_BROWSER_PATH = _working_browser_path()
_PLAYWRIGHT_IMPORTABLE = _playwright_importable()
_SKIP_REASON = "no working Chromium binary found for a live PDF render"


class TestPdfExportErrors:
    def test_missing_playwright_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        with pytest.raises(PdfExportError, match="pip install"):
            render_pdf("<html><body>hi</body></html>", "/tmp/should-not-be-written.pdf")

    def test_missing_playwright_does_not_write_a_file(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        out = tmp_path / "report.pdf"
        with pytest.raises(PdfExportError):
            render_pdf("<html></html>", out)
        assert not out.exists()

    @pytest.mark.skipif(not _PLAYWRIGHT_IMPORTABLE, reason="playwright not installed")
    def test_broken_browser_path_raises_pdf_export_error(self, tmp_path):
        with pytest.raises(PdfExportError, match="chromium"):
            render_pdf("<html></html>", tmp_path / "out.pdf",
                       browser_path="/no/such/chromium/binary")


@pytest.mark.skipif(_BROWSER_PATH is None, reason=_SKIP_REASON)
class TestPdfExportRenders:
    def test_renders_a_real_multi_page_pdf_with_report_content(self, tmp_path):
        html = """<!doctype html><html><head><meta charset="utf-8">
        <style>body{background:#123456;color:#fff}
        .flagged{background:color-mix(in srgb, red 10%, transparent)}</style>
        </head><body><h1>Testco - news and price impact</h1>
        <p>Candidate incident days</p>
        <p>permutation p-value: 0.0000</p>
        </body></html>"""
        out = tmp_path / "report.pdf"
        result = render_pdf(html, out, browser_path=_BROWSER_PATH)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 1000, "a real rendered PDF should not be near-empty"
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_output_directory_is_created(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "report.pdf"
        render_pdf("<html><body>x</body></html>", out, browser_path=_BROWSER_PATH)
        assert out.exists()

    def test_content_survives_the_round_trip(self, tmp_path):
        """Not just 'a PDF got written' - the actual report text must be
        extractable from it, or print_background/rendering silently failed."""
        pypdf = pytest.importorskip("pypdf")
        html = "<html><body><h1>Unique-Marker-48213</h1></body></html>"
        out = tmp_path / "report.pdf"
        render_pdf(html, out, browser_path=_BROWSER_PATH)
        reader = pypdf.PdfReader(str(out))
        text = "".join(page.extract_text() for page in reader.pages)
        assert "Unique-Marker-48213" in text
