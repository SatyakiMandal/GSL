"""Complete synchronization & output generator for all 17 companies.

Window: Strictly 1st Jan 2026 to 23rd August 2026 (157 Sessions).
Generates:
1. Updated HTML reports in out/
2. Multi-tab dynamic financial Excel models in Excel/
3. High-density Institutional PDF reports in out/PDF/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import date, datetime
import pandas as pd

from ceia.pdf import render_analysis_pdf, render_pdf
from ceia.pdf_report import build_pdf_document_html
from ceia.export_excel import export_analysis_to_excel
from ceia.report import build_html, build_unlisted_html
from ceia.models import RunConfig, NewsItem
from ceia.eventstudy import Incident
from ceia.analyze import Analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generator")

LISTED_SLUGS = [
    ("Tata Consultancy Services Limited", "TCS.NS", "TCS"),
    ("Adani Enterprises Limited", "ADANIENT.NS", "ADANIENT"),
    ("Brookfield India Real Estate Trust", "BIRET.BO", "BIRET"),
    ("Garden Reach Shipbuilders & Engineers Limited", "GRSE.NS", "GRSE"),
    ("Vodafone Idea Limited", "IDEA.NS", "IDEA"),
    ("IndusInd Bank Limited", "INDUSINDBK.NS", "INDUSINDBK"),
    ("Lloyds Metals and Energy Limited", "LLOYDSME.NS", "LLOYDSME"),
    ("National Aluminium Company Limited", "NATIONALUM.NS", "NATIONALUM"),
    ("Power Finance Corporation Limited", "PFC.NS", "PFC"),
    ("Sonata Software Limited", "SONATSOFTW.NS", "SONATSOFTW"),
    ("Tata Consumer Products Limited", "TATACONSUM.NS", "TATACONSUM"),
    ("Tata Steel Limited", "TATASTEEL.NS", "TATASTEEL"),
    ("Tata Motors Commercial Vehicles Limited", "TMCV.NS", "TMCV"),
    ("Tata Motors Passenger Vehicles Limited", "TMPV.NS", "TMPV"),
    ("Zydus Lifesciences Limited", "ZYDUSLIFE.NS", "ZYDUSLIFE"),
]

UNLISTED_SLUGS = [
    ("Goa Shipyard Limited", "Goa_Shipyard_Limited"),
    ("Polymatech Electronics Limited", "Polymatech"),
]


def analysis_from_dict(data: dict) -> Analysis:
    cfg_raw = data.get("config", {})
    cfg = RunConfig(
        company=cfg_raw.get("company", ""),
        ticker=cfg_raw.get("ticker", ""),
        exchange=cfg_raw.get("exchange", "NSE"),
        benchmark=cfg_raw.get("benchmark", "^NSEI"),
        benchmark2=cfg_raw.get("benchmark2"),
        start=date.fromisoformat(cfg_raw["start"]) if isinstance(cfg_raw.get("start"), str) else (cfg_raw.get("start") or date(2026, 1, 1)),
        end=date.fromisoformat(cfg_raw["end"]) if isinstance(cfg_raw.get("end"), str) else (cfg_raw.get("end") or date(2026, 8, 23)),
        aliases=cfg_raw.get("aliases", []),
        sources=cfg_raw.get("sources", []),
        event_window=tuple(cfg_raw.get("event_window", (-1, 3))),
        min_relevance=float(cfg_raw.get("min_relevance", 0.35)),
    )

    daily_rows = data.get("daily", [])
    if daily_rows:
        df = pd.DataFrame(daily_rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.set_index("date")
    else:
        df = pd.DataFrame()

    sec_daily_rows = data.get("secondary_daily")
    if sec_daily_rows:
        sec_df = pd.DataFrame(sec_daily_rows)
        if "date" in sec_df.columns:
            sec_df["date"] = pd.to_datetime(sec_df["date"]).dt.date
            sec_df = sec_df.set_index("date")
    else:
        sec_df = None

    incidents = []
    for inc_d in data.get("incidents", []):
        if isinstance(inc_d, dict):
            i_day = date.fromisoformat(inc_d["day"]) if isinstance(inc_d.get("day"), str) else inc_d.get("day")
            inc = Incident(
                day=i_day,
                abnormal_return=float(inc_d.get("abnormal_return", 0.0)),
                abnormal_return_z=float(inc_d.get("abnormal_return_z", 0.0)),
                raw_return=float(inc_d.get("raw_return", 0.0)),
                benchmark_return=float(inc_d.get("benchmark_return", 0.0)),
                coverage_z=float(inc_d.get("coverage_z", 0.0)),
                sentiment_z=float(inc_d.get("sentiment_z", 0.0)),
                item_count=int(inc_d.get("item_count", 0)),
                mean_sentiment=float(inc_d.get("mean_sentiment", 0.0)),
                dominant_event=str(inc_d.get("dominant_event", "")),
                dominant_emotion=str(inc_d.get("dominant_emotion", "")),
                volume=float(inc_d.get("volume", 0.0)),
                volume_z=float(inc_d.get("volume_z", 0.0)),
                score=float(inc_d.get("score", 0.0)),
                direction_agrees=bool(inc_d.get("direction_agrees", True)),
                mean_staleness=inc_d.get("mean_staleness"),
                adi_score=inc_d.get("adi_score"),
                is_clustered=bool(inc_d.get("is_clustered", False)),
                cluster_id=inc_d.get("cluster_id"),
                bmp_stat=inc_d.get("bmp_stat"),
                trajectory_type=str(inc_d.get("trajectory_type", "Permanent Repricing")),
                car=inc_d.get("car", {}),
                headlines=inc_d.get("headlines", []),
                sources=inc_d.get("sources", []),
            )
            incidents.append(inc)

    news_items = []
    for n_d in data.get("news_meta", {}).get("items", []) or data.get("news", []):
        if isinstance(n_d, dict):
            pub = datetime.fromisoformat(n_d["published_at"]) if n_d.get("published_at") and isinstance(n_d["published_at"], str) else n_d.get("published_at")
            t_day = date.fromisoformat(n_d["trading_day"]) if n_d.get("trading_day") and isinstance(n_d["trading_day"], str) else n_d.get("trading_day")
            item = NewsItem(
                source=n_d.get("source", ""),
                url=n_d.get("url", ""),
                headline=n_d.get("headline", ""),
                published_at=pub,
                timestamp_confidence=n_d.get("timestamp_confidence", "date-only"),
                body=n_d.get("body", ""),
                snippet=n_d.get("snippet", ""),
                paywalled=bool(n_d.get("paywalled", False)),
                relevance_score=float(n_d.get("relevance_score", 0.0)),
                matched_aliases=n_d.get("matched_aliases", []),
                headline_match=bool(n_d.get("headline_match", False)),
                sentiment_label=n_d.get("sentiment_label", ""),
                sentiment_score=float(n_d.get("sentiment_score", 0.0)),
                sentiment_confidence=float(n_d.get("sentiment_confidence", 0.0)),
                event_category=n_d.get("event_category", ""),
                emotion_label=n_d.get("emotion_label", ""),
                emotion_score=float(n_d.get("emotion_score", 0.0)),
                trading_day=t_day,
                after_close=bool(n_d.get("after_close", False)),
                fetched_at=n_d.get("fetched_at", ""),
                content_sha256=n_d.get("content_sha256", ""),
                duplicate_of=n_d.get("duplicate_of"),
            )
            news_items.append(item)

    news_meta = data.get("news_meta", {})
    news_meta["items"] = news_items

    return Analysis(
        config=cfg,
        daily=df,
        incidents=incidents,
        price_meta=data.get("price_meta", {}),
        news_meta=news_meta,
        caveats=data.get("caveats", []),
        unattributed=[],
        correlation=data.get("correlation", {}),
        extremity_volume_correlation=data.get("extremity_volume_correlation", {}),
        lagged_correlation=data.get("lagged_correlation", {}),
        emotion_summary=data.get("emotion_summary", {}),
        secondary_daily=sec_df,
        secondary_meta=data.get("secondary_meta", {}),
        robustness=data.get("robustness", {}),
        diagnostics=data.get("diagnostics", {}),
        macro_events=data.get("macro_events", []),
        macro=data.get("macro", {}),
        nifty_indices=data.get("nifty_indices", {}),
        global_indices=data.get("global_indices", {}),
        financials=data.get("financials", {}),
        distance_to_default=data.get("distance_to_default", {}),
        var=data.get("var", {}) or data.get("var_analysis", {}),
        forecasting=data.get("forecasting", {}),
        portfolio=data.get("portfolio", {}),
        xai=data.get("xai", {}),
        sdid=data.get("sdid", {}),
        spillover=data.get("spillover", {}),
        microstructure=data.get("microstructure", {}),
        regime=data.get("regime", {}),
    )


def process_listed(name: str, ticker: str, slug: str):
    json_path = Path(f"json/{slug}_analysis.json")
    if not json_path.exists():
        log.warning(f"JSON model not found: {json_path}")
        return
    log.info(f"===> Processing {name} ({slug}) [2026-01-01 to 2026-08-23]")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    analysis = analysis_from_dict(data)

    # 1. Generate Dedicated Institutional PDF
    pdf_path = Path(f"out/PDF/{slug}_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_analysis_pdf(analysis, pdf_path)
        log.info(f"[+] Generated PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.error(f"[-] PDF generation failed for {slug}: {exc}", exc_info=True)

    # 2. Generate HTML Report with updated graphs
    html_path = Path(f"out/{slug}_report.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        html_out = build_html(analysis)
        html_path.write_text(html_out, encoding="utf-8")
        log.info(f"[+] Generated HTML: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.error(f"[-] HTML generation failed for {slug}: {exc}", exc_info=True)

    # 3. Generate Excel Financial Model
    xlsx_path = Path(f"Excel/{slug}_model_report.xlsx")
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_analysis_to_excel(analysis, xlsx_path)
        log.info(f"[+] Generated Excel: {xlsx_path} ({xlsx_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.error(f"[-] Excel generation failed for {slug}: {exc}", exc_info=True)


def process_unlisted(name: str, slug: str):
    json_path = Path(f"json/{slug}_unlisted.json")
    if not json_path.exists():
        log.warning(f"Unlisted JSON model not found: {json_path}")
        return
    log.info(f"===> Processing Unlisted: {name} ({slug}) [2026-01-01 to 2026-08-23]")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 1. Generate Institutional PDF
    pdf_path = Path(f"out/PDF/{slug}_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_analysis_pdf(data, pdf_path)
        log.info(f"[+] Generated Unlisted PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.error(f"[-] Unlisted PDF generation failed for {slug}: {exc}", exc_info=True)

    # 2. Generate HTML Report
    html_path = Path(f"out/{slug}_unlisted_report.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ceia.unlisted import UnlistedAnalysis, PriceMove
        series_df = pd.DataFrame(data.get("series", []))
        if not series_df.empty and "date" in series_df.columns:
            series_df["date"] = pd.to_datetime(series_df["date"]).dt.date
            series_df = series_df.set_index("date")
        
        cfg = RunConfig(
            company=name,
            ticker=slug,
            start=date(2026, 1, 1),
            end=date(2026, 8, 23)
        )
        
        moves = []
        for m_d in data.get("moves", []):
            if isinstance(m_d, dict):
                s_dt = date.fromisoformat(m_d["start_date"]) if isinstance(m_d.get("start_date"), str) else m_d.get("start_date")
                e_dt = date.fromisoformat(m_d["end_date"]) if isinstance(m_d.get("end_date"), str) else m_d.get("end_date")
                mv = PriceMove(
                    start_date=s_dt,
                    end_date=e_dt,
                    start_price=float(m_d.get("start_price", 100.0)),
                    end_price=float(m_d.get("end_price", 100.0)),
                    headlines=m_d.get("headlines", [])
                )
                moves.append(mv)
                
        unl_analysis = UnlistedAnalysis(
            config=cfg,
            url=data.get("url", "https://unlistedzone.com"),
            series=series_df,
            moves=moves,
            news_meta=data.get("news", {}) or data.get("news_meta", {}),
            unattributed=[],
            items=[],
            macro_events=data.get("macro_events", []),
            macro=data.get("macro", {}),
        )
        html_out = build_unlisted_html(unl_analysis)
        html_path.write_text(html_out, encoding="utf-8")
        log.info(f"[+] Generated Unlisted HTML: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        log.error(f"[-] Unlisted HTML generation failed for {slug}: {exc}", exc_info=True)


def main():
    Path("out").mkdir(exist_ok=True)
    Path("out/PDF").mkdir(parents=True, exist_ok=True)
    Path("Excel").mkdir(exist_ok=True)

    print("=" * 80)
    print("STARTING FULL SYNCHRONIZATION FOR ALL 17 COMPANIES (1ST JAN 2026 - 23RD AUG 2026)")
    print("=" * 80)

    for comp_name, ticker, slug in LISTED_SLUGS:
        process_listed(comp_name, ticker, slug)

    for comp_name, slug in UNLISTED_SLUGS:
        process_unlisted(comp_name, slug)

    print("=" * 80)
    print("ALL 17 COMPANIES PROCESSED AND SYNCHRONIZED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
