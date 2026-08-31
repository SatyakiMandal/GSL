import json
from pathlib import Path
from ceia.pdf import render_analysis_pdf

json_file = Path("json/TCS_analysis.json")
data = json.loads(json_file.read_text(encoding="utf-8"))
out_pdf = Path("out/PDF/TCS_report.pdf")

print("Rendering TCS dedicated PDF from JSON model...")
render_analysis_pdf(data, out_pdf)
print(f"Generated: {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
