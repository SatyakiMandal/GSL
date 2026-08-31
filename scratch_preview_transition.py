from pathlib import Path
from playwright.sync_api import sync_playwright
from ceia.pdf_report import build_pdf_document_html
from ceia.pdf import _find_system_chromium
import json

data = json.loads(Path("json/TCS_analysis.json").read_text(encoding="utf-8"))
html = build_pdf_document_html(data)
Path("scratch_preview.html").write_text(html, encoding="utf-8")

with sync_playwright() as p:
    chrome = _find_system_chromium()
    browser = p.chromium.launch(executable_path=chrome) if chrome else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 3000})
    page.set_content(html, wait_until="load")
    # Screenshot where dossiers meet Section 11
    dossier_elem = page.locator(".grid-2").nth(4) # last grid-2 is dossiers
    dossier_elem.screenshot(path="scratch_dossiers_preview.png")
    # Take screenshot of Section 11 top
    sec11 = page.locator(".sec-box").last
    sec11.screenshot(path="scratch_sec11_preview.png")
    browser.close()

print("Screenshots saved successfully!")
