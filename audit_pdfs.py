from pathlib import Path
import time

pdf_dir = Path("out/PDF")
pdfs = sorted(pdf_dir.glob("*.pdf"))

print(f"Total PDFs in {pdf_dir}: {len(pdfs)}")
for p in pdfs:
    age = time.time() - p.stat().st_mtime
    size_kb = p.stat().st_size / 1024.0
    print(f" - {p.name:<38} : {size_kb:>7.1f} KB  (modified {age:>5.1f}s ago)")
