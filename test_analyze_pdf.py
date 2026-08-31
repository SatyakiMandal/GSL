import traceback
import json
from pathlib import Path
from ceia.pdf import render_analysis_pdf

try:
    data = json.loads(Path("json/TCS_analysis.json").read_text(encoding="utf-8"))
    res = render_analysis_pdf(data, "out/PDF/TCS_report.pdf")
    print("SUCCESS JSON:", res)
except Exception as e:
    print("ERROR JSON:", e)
    traceback.print_exc()
