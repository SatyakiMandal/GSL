from pathlib import Path
import json
import pandas as pd
from datetime import date
from ceia.models import RunConfig, NewsItem
from ceia.analyze import Analysis
from ceia.eventstudy import Incident
from ceia.pdf import render_analysis_pdf

json_path = Path("json/TCS_analysis.json")
data = json.loads(json_path.read_text(encoding="utf-8"))

cfg_dict = data.get("config", {})
config = RunConfig(
    company=cfg_dict.get("company", "Tata Consultancy Services Limited"),
    ticker=cfg_dict.get("ticker", "TCS.NS"),
    benchmark=cfg_dict.get("benchmark", "^NSEI"),
    start=date.fromisoformat(cfg_dict.get("start", "2026-01-01")),
    end=date.fromisoformat(cfg_dict.get("end", "2026-08-23")),
)

# Convert daily rows
daily_rows = data.get("daily", [])
daily_df = pd.DataFrame(daily_rows)
if not daily_df.empty and "date" in daily_df.columns:
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df = daily_df.set_index("date")

incidents = []
for inc_dict in data.get("incidents", []):
    inc_copy = dict(inc_dict)
    if "day" in inc_copy and isinstance(inc_copy["day"], str):
        inc_copy["day"] = date.fromisoformat(inc_copy["day"])
    inc = Incident(
        day=inc_copy["day"],
        abnormal_return=inc_copy.get("abnormal_return", 0.0),
        abnormal_return_z=inc_copy.get("abnormal_return_z", 0.0),
        coverage_z=inc_copy.get("coverage_z", 0.0),
        item_count=inc_copy.get("item_count", 0),
        unique_count=inc_copy.get("unique_count", 0),
        combined_score=inc_copy.get("combined_score", 0.0),
        mean_sentiment=inc_copy.get("mean_sentiment", 0.0),
        weighted_sentiment=inc_copy.get("weighted_sentiment", 0.0),
        dominant_event=inc_copy.get("dominant_event", ""),
        dominant_emotion=inc_copy.get("dominant_emotion", ""),
        car=inc_copy.get("car"),
        t_stat=inc_copy.get("t_stat"),
        t_stat_unconditional=inc_copy.get("t_stat_unconditional"),
        permutation_p=inc_copy.get("permutation_p"),
        t_equiv_p=inc_copy.get("t_equiv_p"),
        robustness_passed=inc_copy.get("robustness_passed", 0),
        robustness_total=inc_copy.get("robustness_total", 9),
        volume=inc_copy.get("volume"),
        volume_z=inc_copy.get("volume_z", 0.0),
    )
    inc.lodr_classification = inc_copy.get("lodr_classification", "Tier 3: Statutory")
    inc.trajectory_type = inc_copy.get("trajectory_type", "Permanent Repricing")
    incidents.append(inc)

news_items = []
for item_dict in data.get("news_items", []):
    item = NewsItem.from_dict(item_dict)
    news_items.append(item)

analysis = Analysis(
    config=config,
    daily=daily_df,
    incidents=incidents,
    news_items=news_items,
    news_meta=data.get("news_meta", {}),
    price_meta=data.get("price_meta", {}),
    model_meta=data.get("model_meta", {}),
    macro=data.get("macro", {}),
    nifty=data.get("nifty", {}),
    global_markets=data.get("global_markets", {}),
    financials=data.get("financials", {}),
    risk_metrics=data.get("risk_metrics", {}),
    distance_to_default=data.get("distance_to_default", {}),
    var_analysis=data.get("var_analysis", {}),
    forecasting=data.get("forecasting", {}),
    microstructure=data.get("microstructure", {}),
    regime=data.get("regime", {}),
    xai=data.get("xai", {}),
    portfolio=data.get("portfolio", {}),
    spillover=data.get("spillover", {}),
)

out_pdf = Path("out/PDF/TCS_report.pdf")
print("Rendering dedicated PDF report...")
render_analysis_pdf(analysis, out_pdf)
print(f"Successfully generated: {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
