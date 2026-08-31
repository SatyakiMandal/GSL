from pathlib import Path
from playwright.sync_api import sync_playwright
from ceia.pdf_report import build_pdf_document_html
import json

data = json.loads(Path("json/TCS_analysis.json").read_text(encoding="utf-8"))
html = build_pdf_document_html(data)
Path("scratch_preview.html").write_text(html, encoding="utf-8")

from ceia.pdf import _find_system_chromium

with sync_playwright() as p:
    chrome = _find_system_chromium()
    browser = p.chromium.launch(executable_path=chrome) if chrome else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 1600})
    page.set_content(html, wait_until="load")
    page.screenshot(path="scratch_tcs_page1.png", full_page=False)
    browser.close()

print("Screenshot saved to scratch_tcs_page1.png")
