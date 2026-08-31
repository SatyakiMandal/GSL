import sys
import traceback
from ceia.analyze import main

sys.argv = [
    "ceia.analyze",
    "--company", "Tata Consultancy Services Limited",
    "--ticker", "TCS.NS",
    "--start", "2026-01-01",
    "--end", "2026-08-23",
    "--html", "out/TCS_report.html",
    "--pdf", "out/PDF/TCS_report.pdf",
    "--xlsx", "Excel/TCS_model_report.xlsx",
    "--out", "json/TCS_analysis.json",
    "--permutations", "5",
]

try:
    main()
    print("\n[+] CLI MAIN SUCCEEDED!")
except Exception as e:
    print("\n[-] CLI MAIN FAILED:")
    traceback.print_exc()
