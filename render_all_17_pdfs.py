import json
import sys
from pathlib import Path
from datetime import datetime
from ceia.pdf import render_analysis_pdf, render_pdf
from ceia.pdf_report import build_pdf_document_html

LISTED_COMPANIES = [
    ("TCS", "Tata Consultancy Services Limited", "TCS.NS"),
    ("ADANIENT", "Adani Enterprises Limited", "ADANIENT.NS"),
    ("BIRET", "Brookfield India Real Estate Trust", "BIRET.RR"),
    ("GRSE", "Garden Reach Shipbuilders & Engineers Limited", "GRSE.NS"),
    ("IDEA", "Vodafone Idea Limited", "IDEA.NS"),
    ("INDUSINDBK", "IndusInd Bank Limited", "INDUSINDBK.NS"),
    ("LLOYDSME", "Lloyds Metals and Energy Limited", "LLOYDSME.NS"),
    ("NATIONALUM", "National Aluminium Company Limited", "NATIONALUM.NS"),
    ("PFC", "Power Finance Corporation Limited", "PFC.NS"),
    ("SONATSOFTW", "Sonata Software Limited", "SONATSOFTW.NS"),
    ("TATACONSUM", "Tata Consumer Products Limited", "TATACONSUM.NS"),
    ("TATASTEEL", "Tata Steel Limited", "TATASTEEL.NS"),
    ("TMCV", "Tata Motors Commercial Vehicles Limited", "TATAMOTORS.NS"),
    ("TMPV", "Tata Motors Passenger Vehicles Limited", "TMPV.NS"),
    ("ZYDUSLIFE", "Zydus Lifesciences Limited", "ZYDUSLIFE.NS"),
]

UNLISTED_COMPANIES = [
    ("Goa_Shipyard_Limited", "Goa Shipyard Limited"),
    ("Polymatech", "Polymatech Electronics Limited"),
]

pdf_out_dir = Path("out/PDF")
pdf_out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"COMMENCING DEDICATED INSTITUTIONAL SURVEILLANCE PDF BATCH: 17 REPORTS")
print(f"Target Directory: {pdf_out_dir.resolve()}")
print("=" * 70)

success_count = 0
failed = []

# 1. Process Listed Companies
for slug, company, ticker in LISTED_COMPANIES:
    json_path = Path(f"json/{slug}_analysis.json")
    if not json_path.exists():
        print(f"[-] Missing JSON payload for {slug}: {json_path}")
        failed.append(slug)
        continue

    pdf_path = pdf_out_dir / f"{slug}_report.pdf"
    print(f"\n[*] Rendering Dedicated PDF for {company} ({slug})...")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        render_analysis_pdf(data, pdf_path)
        size_kb = pdf_path.stat().st_size / 1024.0
        print(f"    [+] SUCCESS: {pdf_path} ({size_kb:.1f} KB)")
        success_count += 1
    except Exception as exc:
        print(f"    [-] ERROR for {slug}: {exc}")
        import traceback
        traceback.print_exc()
        failed.append(slug)

# 2. Process Unlisted Companies
for slug, company in UNLISTED_COMPANIES:
    json_path = Path(f"json/{slug}_unlisted.json")
    if not json_path.exists():
        print(f"[-] Missing unlisted JSON payload for {slug}: {json_path}")
        failed.append(slug)
        continue

    pdf_path = pdf_out_dir / f"{slug}_report.pdf"
    print(f"\n[*] Rendering Dedicated PDF for Unlisted: {company} ({slug})...")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Standardize unlisted data for PDF report generator
        unlisted_analysis = {
            "company": company,
            "ticker": "PRE-IPO / UNLISTED",
            "benchmark": "^NSEI",
            "start": data.get("start", "2026-01-01"),
            "end": data.get("end", "2026-08-23"),
            "daily": [],
            "incidents": [],
            "news_items": [],
            "price_meta": {},
            "model_meta": {"beta": 1.0, "r_squared": 0.0},
            "financials": {},
            "macro": data.get("macro", {}),
            "distance_to_default": {},
            "var": {},
            "forecasting": {},
            "microstructure": {},
            "regime": {},
            "xai": {},
            "portfolio": {},
            "spillover": {},
            "correlation": {"r": 0.0},
            "lagged_correlation": {"1": {"r": 0.0}},
            "extremity_volume_correlation": {"r": 0.0},
        }
        
        # Convert moves to incidents for dossier reporting
        moves = data.get("moves", [])
        for m in moves:
            s_date = m.get("start_date")
            chg = m.get("change", 0.0)
            hdls = m.get("headlines", [])
            inc = {
                "day": s_date,
                "abnormal_return": chg,
                "abnormal_return_z": 2.0 if abs(chg) > 0.05 else 1.0,
                "unique_count": len(hdls),
                "weighted_sentiment": 0.2 if chg > 0 else -0.2,
                "dominant_event": "Private Market Valuation Revision",
                "dominant_emotion": "Conviction",
                "car": chg,
                "permutation_p": 0.05,
                "lodr_classification": "Unlisted Private Valuation",
                "trajectory_type": "Private Step-Up" if chg > 0 else "Private Step-Down",
            }
            unlisted_analysis["incidents"].append(inc)
            for h in hdls:
                unlisted_analysis["news_items"].append({
                    "source": h.get("source", "Unlisted Press"),
                    "headline": h.get("headline", ""),
                    "sentiment_label": "positive" if chg > 0 else "negative",
                    "sentiment_score": 0.3 if chg > 0 else -0.3,
                    "relevance_score": 1.0,
                    "trading_day": s_date,
                })
                
        render_analysis_pdf(unlisted_analysis, pdf_path)
        size_kb = pdf_path.stat().st_size / 1024.0
        print(f"    [+] SUCCESS: {pdf_path} ({size_kb:.1f} KB)")
        success_count += 1
    except Exception as exc:
        print(f"    [-] ERROR for {slug}: {exc}")
        import traceback
        traceback.print_exc()
        failed.append(slug)

print("\n" + "=" * 70)
print(f"BATCH SUMMARY: {success_count}/17 PDFs Successfully Generated")
if failed:
    print(f"Failed: {failed}")
print("=" * 70)
