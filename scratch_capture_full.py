from pathlib import Path
from playwright.sync_api import sync_playwright
from ceia.pdf import _find_system_chromium

with sync_playwright() as p:
    chrome = _find_system_chromium()
    browser = p.chromium.launch(executable_path=chrome) if chrome else p.chromium.launch()
    page = browser.new_page(viewport={"width": 960, "height": 800})
    html = Path("scratch_preview.html").read_text(encoding="utf-8")
    page.set_content(html, wait_until="load")
    page.screenshot(path="scratch_full.png", full_page=True)
    browser.close()

print("Full page screenshot saved successfully!")
