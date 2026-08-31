"""Full end-to-end multi-asset institutional pipeline runner for all 17 companies.

Generates:
1. JSON analytical models in /json
2. Institutional HTML reports in /out
3. Comprehensive Excel financial models in /Excel
4. Dedicated high-density Institutional PDF reports in /out/PDF
"""

import sys
import json
import logging
from pathlib import Path
from datetime import date
import pandas as pd

from ceia.config import Config
from ceia.analysis import run_analysis
from ceia.report import build_html
from ceia.excel_export import generate_institutional_excel_model
from ceia.pdf import generate_pdf_from_html
from ceia.pdf_report import build_pdf_document_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 8, 23)

LISTED_COMPANIES = [
    ("Tata Consultancy Services Limited", "TCS", "^NSEI"),
    ("Adani Enterprises Limited", "ADANIENT", "^NSEI"),
    ("Brookfield India Real Estate Trust", "BIRET", "^NSEI"),
    ("Garden Reach Shipbuilders & Engineers Limited", "GRSE", "^NSEI"),
    ("Vodafone Idea Limited", "IDEA", "^NSEI"),
    ("IndusInd Bank Limited", "INDUSINDBK", "^NSEI"),
    ("Lloyds Metals and Energy Limited", "LLOYDSME", "^NSEI"),
    ("National Aluminium Company Limited", "NATIONALUM", "^NSEI"),
    ("Power Finance Corporation Limited", "PFC", "^NSEI"),
    ("Sonata Software Limited", "SONATSOFTW", "^NSEI"),
    ("Tata Consumer Products Limited", "TATACONSUM", "^NSEI"),
    ("Tata Steel Limited", "TATASTEEL", "^NSEI"),
    ("Tata Motors Commercial Vehicles Limited", "TMCV", "^NSEI"),
    ("Tata Motors Passenger Vehicles Limited", "TMPV", "^NSEI"),
    ("Zydus Lifesciences Limited", "ZYDUSLIFE", "^NSEI"),
]

UNLISTED_FILES = [
    "json/Goa_Shipyard_Limited_unlisted.json",
    "json/Polymatech_unlisted.json",
]


def run_listed(company: str, ticker: str, benchmark: str):
    log.info(f"===> Processing Listed: {company} ({ticker})")
    cfg = Config(
        company=company,
        ticker=ticker,
        benchmark=benchmark,
        start=START_DATE,
        end=END_DATE,
    )
    
    # 1. Run full analysis pipeline
    analysis = run_analysis(cfg)
    
    # 2. Save JSON model
    json_path = Path(f"json/{ticker}_analysis.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        # Serialise dataclass
        from ceia.storage import analysis_to_dict
        d = analysis_to_dict(analysis)
        json.dump(d, f, indent=2, default=str)
    log.info(f"Saved JSON: {json_path}")
    
    # 3. Save HTML report
    html_content = build_html(analysis)
    html_path = Path(f"out/{ticker}_report.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_content, encoding="utf-8")
    log.info(f"Saved HTML: {html_path}")
    
    # 4. Save Excel Financial Model
    excel_path = Path(f"Excel/{ticker}_financial_model.xlsx")
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_institutional_excel_model(analysis, str(excel_path))
        log.info(f"Saved Excel: {excel_path}")
    except Exception as exc:
        log.warning(f"Excel generation failed for {ticker}: {exc}")
        
    # 5. Save Dedicated Institutional PDF
    pdf_html = build_pdf_document_html(analysis)
    pdf_path = Path(f"out/PDF/{ticker}_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_pdf_from_html(pdf_html, str(pdf_path))
        log.info(f"Saved PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.warning(f"PDF generation failed for {ticker}: {exc}")


def run_unlisted(json_file_path: str):
    p = Path(json_file_path)
    if not p.exists():
        log.warning(f"Unlisted json not found: {json_file_path}")
        return
    log.info(f"===> Processing Unlisted: {p.name}")
    raw_data = json.loads(p.read_text(encoding="utf-8"))
    
    company = raw_data.get("company", p.stem.replace("_unlisted", ""))
    safe_name = p.stem.replace("_unlisted", "")
    
    # PDF generation from unlisted model
    pdf_html = build_pdf_document_html(raw_data)
    pdf_path = Path(f"out/PDF/{safe_name}_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_pdf_from_html(pdf_html, str(pdf_path))
        log.info(f"Saved Unlisted PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.warning(f"Unlisted PDF generation failed for {safe_name}: {exc}")


def main():
    Path("out").mkdir(exist_ok=True)
    Path("out/PDF").mkdir(parents=True, exist_ok=True)
    Path("Excel").mkdir(exist_ok=True)
    Path("json").mkdir(exist_ok=True)
    
    log.info("Starting Full 17-Company Institutional Pipeline Execution...")
    
    for comp, tick, bench in LISTED_COMPANIES:
        try:
            run_listed(comp, tick, bench)
        except Exception as e:
            log.error(f"Error processing {tick}: {e}", exc_info=True)
            
    for unl in UNLISTED_FILES:
        try:
            run_unlisted(unl)
        except Exception as e:
            log.error(f"Error processing unlisted {unl}: {e}", exc_info=True)
            
    log.info("Full 17-Company Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
