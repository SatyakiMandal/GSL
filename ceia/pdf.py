"""PDF export of the self-contained HTML report.

Renders through headless Chromium (Playwright) rather than a pure-Python PDF
library (e.g. WeasyPrint): the report's CSS uses ``color-mix()`` for the
flagged-row highlight and the charts are inline SVG with CSS custom
properties resolved against ``prefers-color-scheme`` - a real browser engine
renders both correctly without hand-auditing which CSS features a
lighter-weight library does or doesn't support. The cost is a heavier,
opt-in dependency: ``pip install -e ".[pdf]"`` alone is not enough, since
Playwright also needs ``playwright install chromium`` to fetch the browser
binary - a second step none of this project's other extras require, which
is why this is kept out of the ``all`` extra rather than bundled into it.
"""

from __future__ import annotations

from pathlib import Path


class PdfExportError(Exception):
    """Raised when the PDF export dependency is missing or the render fails."""


def render_pdf(html: str, path: Path | str, browser_path: str | None = None) -> Path:
    """Render a self-contained HTML report string to a PDF at ``path``.

    ``browser_path`` overrides which Chromium binary Playwright launches.
    Leave it unset in normal use - Playwright resolves its own bundled
    browser correctly on its own. It exists so this function's own render
    path (not just its error handling) can be exercised end-to-end in an
    environment where Playwright's pip package and its pre-provisioned
    browser binary are pinned to different revisions.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PdfExportError(
            "PDF export needs Playwright, which is not installed. Run: "
            'pip install -e ".[pdf]" && playwright install chromium'
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    launch_kwargs = {"executable_path": browser_path} if browser_path else {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page()
                # wait_until="load" (not the default "commit"): the charts are
                # inline SVG with no external resources, so there is nothing
                # left to finish loading past that point - no arbitrary sleep
                # needed to "let the charts render" the way a JS-drawn chart
                # library would require.
                page.set_content(html, wait_until="load")
                page.pdf(
                    path=str(out),
                    format="A4",
                    print_background=True,  # the report's whole look is background colour
                    margin={"top": "14mm", "bottom": "14mm",
                           "left": "10mm", "right": "10mm"},
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        # Playwright's own error text for a missing browser binary includes a
        # multi-line decorative box ("Looks like Playwright was just
        # installed...") - useful in a terminal, noisy embedded in a report
        # UI. The diagnostic content is always on the first line; the rest is
        # covered by the actionable message added below regardless.
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else str(exc)
        raise PdfExportError(
            f"PDF export failed: {first_line}. If this is the first run, "
            "the Chromium binary may be missing - try: playwright install chromium"
        ) from exc

    return out
