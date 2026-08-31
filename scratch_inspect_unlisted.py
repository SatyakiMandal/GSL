import json
from pathlib import Path

for p in [Path("json/Goa_Shipyard_Limited_unlisted.json"), Path("json/Polymatech_unlisted.json")]:
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(p.name, "keys:", list(d.keys())[:10])
        series = d.get("series", [])
        print(p.name, "series count:", len(series))
        if series:
            print("Sample series 0:", series[0])
