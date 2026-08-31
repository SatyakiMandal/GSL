from pathlib import Path
from pypdf import PdfReader

pdf = PdfReader("out/PDF/TCS_report.pdf")
print(f"Total pages in TCS_report.pdf: {len(pdf.pages)}")
for i, page in enumerate(pdf.pages):
    text = page.extract_text()
    lines = [l for l in text.splitlines() if l.strip()]
    print(f"Page {i+1}: {len(lines)} lines of text | First line: {lines[0] if lines else 'EMPTY'}")
