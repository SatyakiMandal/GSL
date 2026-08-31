from ceia.pdf_extract import extract_text_from_pdf
from pathlib import Path

text = extract_text_from_pdf(Path("out/PDF/TCS_report.pdf"))
print(f"Extracted {len(text)} characters of text from out/PDF/TCS_report.pdf")
print("First 500 chars:")
print(text[:500])
