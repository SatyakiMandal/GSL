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
    # A4 ratio: 794 x 1123 px at 96dpi or 1200 x 1700
    page = browser.new_page(viewport={"width": 960, "height": 1360})
    page.set_content(html, wait_until="load")
    
    # Screenshot top portion (Page 1)
    page.screenshot(path="scratch_p1.png", clip={"x": 0, "y": 0, "width": 960, "height": 1360})
    
    # Screenshot middle portion (Page 2)
    page.screenshot(path="scratch_p2.png", clip={"x": 0, "y": 1360, "width": 960, "height": 1360})
    
    browser.close()

print("Visual pages captured successfully!")
