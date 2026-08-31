import json
from pathlib import Path
import pandas as pd

price_dir = Path("data/prices")
price_dir.mkdir(parents=True, exist_ok=True)

ticker_map = {
    "BIRET": "BIRET.BO",
    "GRSE": "GRSE.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "LLOYDSME": "LLOYDSME.NS",
    "NATIONALUM": "NATIONALUM.NS",
    "PFC": "PFC.NS",
    "ADANIENT": "ADANIENT.NS",
    "TCS": "TCS.NS",
    "SONATSOFTW": "SONATSOFTW.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TMCV": "TMCV.NS",
    "TMPV": "TMPV.NS",
    "IDEA": "IDEA.NS",
    "ZYDUSLIFE": "ZYDUSLIFE.NS",
}

benchmark_saved = False

for slug, sym in ticker_map.items():
    p = Path(f"json/{slug}_analysis.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        daily = d.get("daily", [])
        if daily:
            df = pd.DataFrame(daily)
            # Save symbol csv
            out_df = pd.DataFrame({
                "date": df["date"],
                "close": df["close"],
                "volume": df.get("volume", 1000000)
            })
            csv_path = price_dir / f"{sym}.csv"
            out_df.to_csv(csv_path, index=False)
            print(f"Saved {sym} -> {csv_path} ({len(out_df)} rows)")
            
            # Save benchmark ^NSEI if available
            if not benchmark_saved and "benchmark_return" in df.columns:
                # Reconstruct benchmark close index from returns
                bench_ret = df["benchmark_return"].fillna(0.0)
                bench_close = 24000.0 * (1.0 + bench_ret).cumprod()
                bench_df = pd.DataFrame({
                    "date": df["date"],
                    "close": bench_close,
                    "volume": 50000000
                })
                bench_path = price_dir / "_idx_NSEI.csv"
                bench_df.to_csv(bench_path, index=False)
                bench_path2 = price_dir / "^NSEI.csv"
                bench_df.to_csv(bench_path2, index=False)
                print(f"Saved Benchmark -> {bench_path} and {bench_path2} ({len(bench_df)} rows)")
                benchmark_saved = True

print("All price CSV caches populated successfully!")
