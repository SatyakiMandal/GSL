from pathlib import Path
import time

f = Path("out/PDF/TCS_report.pdf")
if f.exists():
    age = time.time() - f.stat().st_mtime
    print(f"out/PDF/TCS_report.pdf exists! Size: {f.stat().st_size / 1024:.1f} KB, Modified {age:.1f}s ago")
else:
    print("out/PDF/TCS_report.pdf does not exist yet.")
