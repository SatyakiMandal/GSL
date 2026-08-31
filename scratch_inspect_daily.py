import json
from pathlib import Path

data = json.loads(Path("json/TCS_analysis.json").read_text(encoding="utf-8"))
daily = data.get("daily", [])
print(f"Total daily rows: {len(daily)}")
if daily:
    print("Sample daily row 0 keys:", list(daily[0].keys()))
    print("Sample daily row 0:", daily[0])
