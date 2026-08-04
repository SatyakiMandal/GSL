"""HTML report generation (PRD Section 8, reporting module).

Produces a single self-contained file: no external stylesheets, scripts, fonts
or images, so it can be emailed, committed, or opened offline and still render.
Charts are inline SVG (see ``ceia/charts.py``).

The structure follows what Success Metric #3 asks for — a reader who did not
build the tool should follow both the finding and its limitations. So the
limitations sit near the top, before the findings, rather than in a footnote
after them.
"""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from pathlib import Path

import pandas as pd

from .charts import timeline_svg
from .eventstudy import Incident
from .narrative import incident_narrative, summary_narrative

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1b19;--muted:#6b6862;--line:#e3e1dc;--card:#fff;
--accent:#1a56a8;--bench:#9a958c;--pos:#1f7a4d;--neg:#b3261e;--warn-bg:#fdf6e3;
--warn-br:#d9a441;--plot:#f5f4f1;}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#e9e7e2;--muted:#9e9a92;
--line:#33313a;--card:#1e1d23;--accent:#7fb0f0;--bench:#7d7970;--pos:#5cc48d;
--neg:#f2837a;--warn-bg:#2b2416;--warn-br:#a8802f;--plot:#212027;}}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1000px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 6px}
h2{font-size:1.3rem;margin:44px 0 14px;padding-bottom:7px;border-bottom:2px solid var(--line)}
h3{font-size:1.05rem;margin:26px 0 8px}
p{margin:0 0 13px}
.sub{color:var(--muted);font-size:.95rem;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:18px 20px;margin:16px 0}
.warn{background:var(--warn-bg);border-left:4px solid var(--warn-br);
border-radius:0 7px 7px 0;padding:15px 19px;margin:18px 0}
.warn h3{margin-top:0}
.warn ul{margin:8px 0 0;padding-left:20px}
.warn li{margin-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:16px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.stat .k{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.05em}
.stat .v{font-size:1.28rem;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:640px}
th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap;font-variant-numeric:tabular-nums}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em;text-align:right}
th:first-child,td:first-child{text-align:left}
td.txt,th.txt{text-align:left;white-space:normal}
tr.flagged{background:color-mix(in srgb,var(--warn-br) 11%,transparent)}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.note{color:var(--muted);font-size:.85rem;font-style:italic}
.timeline{width:100%;height:auto;display:block}
.plot-bg{fill:var(--plot)}
.gridline{stroke:var(--line);stroke-width:1}
.axis-zero{stroke:var(--muted);stroke-width:1.2}
.tick{fill:var(--muted);font-size:11px}
.panel-title{fill:var(--fg);font-size:12.5px;font-weight:600}
.line-company{fill:none;stroke:var(--accent);stroke-width:2.1;
stroke-linejoin:round;stroke-linecap:round}
.line-benchmark{fill:none;stroke:var(--bench);stroke-width:1.7;stroke-dasharray:5 3}
.incident-rule{stroke:var(--warn-br);stroke-width:1.4;stroke-dasharray:3 3;opacity:.85}
.bar-pos{fill:var(--pos)}.bar-neg{fill:var(--neg)}
.bar-incident{stroke:var(--warn-br);stroke-width:1.4}
.tone-neg{fill:var(--neg);opacity:.82}.tone-pos{fill:var(--pos);opacity:.82}
.tone-neutral{fill:var(--muted);opacity:.6}
.spark{width:110px;height:26px}
.spark-neg{fill:none;stroke:var(--neg);stroke-width:1.6}
.spark-pos{fill:none;stroke:var(--pos);stroke-width:1.6}
.incident{border:1px solid var(--line);border-left:4px solid var(--warn-br);
border-radius:0 9px 9px 0;padding:18px 22px;margin:20px 0;background:var(--card)}
.rank{display:inline-block;background:var(--warn-br);color:#fff;border-radius:5px;
padding:1px 9px;font-size:.82rem;font-weight:700;margin-right:9px}
.src{list-style:none;padding:0;margin:12px 0 0}
.src li{padding:7px 0;border-top:1px solid var(--line);font-size:.9rem}
.tag{display:inline-block;font-size:.72rem;padding:1px 7px;border-radius:4px;
background:var(--plot);color:var(--muted);margin-right:7px;border:1px solid var(--line)}
.foot{color:var(--muted);font-size:.85rem;margin-top:52px;padding-top:16px;
border-top:1px solid var(--line)}
code{background:var(--plot);padding:1px 5px;border-radius:4px;font-size:.87em}
.empty{color:var(--muted);font-style:italic}
"""


def _sentence(text: str) -> str:
    """Ensure a fragment ends with terminal punctuation before it is followed
    by more prose. ``model_note`` values (e.g. "fitted on 120 trading days
    before 2023-01-20") have none, which ran straight into the next sentence
    with no separator."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _cls(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


def _pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _stat(key: str, value: str) -> str:
    return f'<div class="stat"><div class="k">{escape(key)}</div><div class="v">{value}</div></div>'


def _volume_cell(row: pd.Series) -> str:
    volume = row.get("volume")
    if volume is None or pd.isna(volume):
        return "<td>—</td>"
    volume_z = row.get("volume_z")
    z_part = f" (z={float(volume_z):+.1f})" if volume_z is not None and pd.notna(volume_z) else ""
    return f"<td>{float(volume):,.0f}{z_part}</td>"


def _daily_table(daily: pd.DataFrame, incident_days: set[date],
                 secondary_ticker: str | None = None) -> str:
    if daily.empty:
        return '<p class="empty">No trading days in the analysis window.</p>'
    has_volume = "volume" in daily.columns
    has_secondary = secondary_ticker and "secondary_abnormal_return" in daily.columns
    rows = []
    for day, row in daily.iterrows():
        # pandas Timestamp subclasses date, so an isinstance guard would leave
        # it as a Timestamp and the incident_days membership test would never
        # match - flagged rows would silently stop highlighting.
        day = pd.Timestamp(day).date()
        flagged = ' class="flagged"' if day in incident_days else ""
        abnormal = float(row["abnormal_return"])
        secondary_cell = ""
        if has_secondary:
            sec = row.get("secondary_abnormal_return")
            secondary_cell = (f"<td class=\"{_cls(sec)}\">{_pct(sec)}</td>"
                              if pd.notna(sec) else "<td>—</td>")
        rows.append(
            f"<tr{flagged}><td>{day:%d %b %Y}</td>"
            f"<td>{float(row['close']):,.2f}</td>"
            f"<td class=\"{_cls(row['return'])}\">{_pct(row['return'])}</td>"
            f"<td class=\"{_cls(row['benchmark_return'])}\">{_pct(row['benchmark_return'])}</td>"
            f"<td class=\"{_cls(abnormal)}\"><strong>{_pct(abnormal)}</strong></td>"
            f"<td>{float(row['abnormal_return_z']):+.2f}</td>"
            + secondary_cell
            + (_volume_cell(row) if has_volume else "")
            + f"<td>{int(row['unique_count'])}</td>"
            f"<td class=\"{_cls(row['weighted_sentiment'])}\">"
            f"{float(row['weighted_sentiment']):+.2f}</td>"
            f"<td class=\"txt\">{escape(str(row['dominant_event'] or '—'))}</td></tr>"
        )
    volume_header = "<th>Volume</th>" if has_volume else ""
    secondary_header = (f"<th>Abnormal vs {escape(secondary_ticker)}</th>"
                        if has_secondary else "")
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Date</th><th>Close</th><th>Return</th><th>Benchmark</th>"
        f"<th>Abnormal</th><th>z</th>{secondary_header}{volume_header}"
        "<th>Items</th><th>Tone</th>"
        '<th class="txt">Main topic</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


def _emotion_valence_table(emotion_summary: dict) -> str:
    groups = (emotion_summary or {}).get("groups") or {}
    if not groups:
        return ""
    rows = []
    for valence in ("positive", "negative", "ambiguous"):
        g = groups.get(valence)
        if not g:
            continue
        rows.append(
            f"<tr><td class=\"txt\">{valence}</td><td>{g['n_days']}</td>"
            f"<td class=\"{_cls(g['mean_abnormal_return'])}\">"
            f"{_pct(g['mean_abnormal_return'])}</td>"
            f"<td class=\"{_cls(g['mean_weighted_sentiment'])}\">"
            f"{g['mean_weighted_sentiment']:+.2f}</td>"
            f"<td class=\"txt\">{escape(', '.join(g['labels_seen']))}</td></tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="scroll"><table><thead><tr><th class="txt">Valence</th>'
        "<th>Days</th><th>Mean abnormal return</th><th>Mean sentiment</th>"
        '<th class="txt">Labels seen</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


def _incident_table(incidents: list[Incident], window: tuple[int, int]) -> str:
    if not incidents:
        return ('<p class="empty">No day combined notable coverage with an unusual '
                "abnormal return at the configured thresholds.</p>")
    rows = []
    for rank, inc in enumerate(incidents, 1):
        car = inc.car or {}
        car_value = car.get("car")
        t_stat = car.get("t_stat")
        rows.append(
            f"<tr><td>{rank}</td><td>{inc.day:%d %b %Y}</td>"
            f"<td class=\"{_cls(inc.abnormal_return)}\"><strong>"
            f"{_pct(inc.abnormal_return)}</strong></td>"
            f"<td>{inc.abnormal_return_z:+.1f}</td>"
            f"<td>{inc.item_count}</td>"
            f"<td class=\"{_cls(inc.mean_sentiment)}\">{inc.mean_sentiment:+.2f}</td>"
            f"<td class=\"{_cls(car_value or 0)}\">"
            f"{_pct(car_value) if car_value is not None else '—'}</td>"
            f"<td>{f'{t_stat:.2f}' if t_stat is not None else '—'}</td>"
            f'<td class="txt">{"consistent" if inc.direction_agrees else "opposite"}</td>'
            f'<td class="txt">{escape(inc.dominant_event or "—")}</td>'
            f'<td class="txt">{escape(inc.dominant_emotion or "—")}</td></tr>'
        )
    before, after = window
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>#</th><th>Date</th><th>Abnormal return</th><th>z</th><th>Items</th>"
        f"<th>Tone</th><th>CAR[{before},+{after}]</th><th>t</th>"
        '<th class="txt">Tone vs price</th><th class="txt">Main topic</th>'
        '<th class="txt">Emotion*</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _incident_sections(incidents: list[Incident], company: str, benchmark: str,
                       window: tuple[int, int]) -> str:
    if not incidents:
        return ""
    blocks = []
    for rank, inc in enumerate(incidents, 1):
        paragraphs = "".join(
            f"<p>{text}</p>"
            for text in incident_narrative(inc, company, benchmark, window)
        )
        sources = "".join(
            f"<li>{escape(headline)}</li>" for headline in inc.headlines
        )
        source_block = (
            f'<h4 style="margin:16px 0 4px;font-size:.92rem">Coverage behind this flag</h4>'
            f'<ul class="src">{sources}</ul>' if sources else ""
        )
        volume_note = ""
        if inc.volume and pd.notna(inc.volume):
            volume_z_part = (f" (z={inc.volume_z:+.2f})"
                             if inc.volume_z is not None and pd.notna(inc.volume_z) else "")
            volume_note = (
                f'<p class="note">Volume: {inc.volume:,.0f}{volume_z_part} — a '
                "corroborating signal, not part of the flagging test.</p>"
            )
        blocks.append(
            f'<div class="incident"><h3><span class="rank">#{rank}</span>'
            f"{inc.day:%d %B %Y}</h3>"
            f'<div><span class="tag">abnormal {_pct(inc.abnormal_return)}</span>'
            f'<span class="tag">z {inc.abnormal_return_z:+.1f}</span>'
            f'<span class="tag">{inc.item_count} item(s)</span>'
            f'<span class="tag">{escape(inc.dominant_event or "other")}</span>'
            + (f'<span class="tag">emotion: {escape(inc.dominant_emotion)}</span>'
               if inc.dominant_emotion else "")
            + "</div>"
            f"{paragraphs}{volume_note}{source_block}</div>"
        )
    return "".join(blocks)


def build_html(analysis) -> str:
    """Render an :class:`ceia.analyze.Analysis` as a standalone HTML document."""
    config = analysis.config
    daily = analysis.daily
    incidents = analysis.incidents
    incident_days = {i.day for i in incidents}
    price = analysis.price_meta
    news_stats = analysis.news_meta.get("stats", {}) or {}
    # Prefer the deduplicated count: one wire story carried by four outlets is
    # one item of news, not four.
    news_count = news_stats.get("unique_after_dedupe")
    if news_count is None:
        news_count = news_stats.get("relevant", 0)

    # The narrative templates emit raw HTML (they use <em> and <strong>), so
    # every value interpolated into them must be escaped here rather than at
    # render time. Company and ticker are user-supplied on the command line.
    safe_company = escape(config.company)
    safe_benchmark = escape(config.benchmark)

    summary = "".join(
        f"<p>{text}</p>" for text in summary_narrative(
            safe_company, escape(config.ticker), safe_benchmark,
            config.start, config.end, incidents, len(daily), news_count,
            price.get("model", "market-adjusted"),
        )
    )

    correlation = getattr(analysis, "correlation", {}) or {}
    corr_display = (f"r = {correlation['r']:+.3f}"
                    if correlation.get("r") is not None else "n/a")
    emotion_summary = getattr(analysis, "emotion_summary", {}) or {}
    emotion_table = _emotion_valence_table(emotion_summary)

    secondary_meta = getattr(analysis, "secondary_meta", {}) or {}
    secondary_daily = getattr(analysis, "secondary_daily", None)
    secondary_ticker = secondary_meta.get("ticker") if secondary_daily is not None else None
    daily_display = daily
    if secondary_ticker:
        daily_display = daily.join(secondary_daily[["secondary_abnormal_return"]])

    stats = "".join([
        _stat("Trading days", str(len(daily))),
        _stat("News items", str(news_count)),
        _stat("Candidate days", str(len(incidents))),
        _stat("Return model", price.get("model", "—")),
        _stat("Beta", f"{price.get('beta', float('nan')):.2f}"),
        _stat("Published after close", str(news_stats.get("after_close", 0))),
        _stat("Sentiment/return correlation", corr_display),
    ] + ([_stat(f"Beta vs {secondary_ticker}", f"{secondary_meta.get('beta', float('nan')):.2f}")]
        if secondary_ticker and secondary_meta.get("beta") is not None else []))

    caveats = "".join(f"<li>{escape(note)}</li>" for note in analysis.caveats)

    # Per-source availability, including anything that failed or is disabled.
    source_rows = []
    for key, state in (analysis.news_meta.get("source_status") or {}).items():
        source_rows.append(f"<tr><td>{escape(key)}</td>"
                           f'<td class="txt">{escape(str(state))}</td></tr>')
    for key, why in (analysis.news_meta.get("disabled_sources") or {}).items():
        source_rows.append(f"<tr><td>{escape(key)}</td>"
                           f'<td class="txt">DISABLED — {escape(str(why))}</td></tr>')
    source_table = (
        '<div class="scroll"><table><thead><tr><th>Source</th>'
        '<th class="txt">Status this run</th></tr></thead><tbody>'
        + "".join(source_rows) + "</tbody></table></div>"
    ) if source_rows else '<p class="empty">No source status recorded.</p>'

    unattributed = ""
    if analysis.unattributed:
        items = "".join(
            f"<li>[{escape(i.source)}] {escape(i.headline)}</li>"
            for i in analysis.unattributed[:20]
        )
        unattributed = (
            f'<div class="warn"><h3>{len(analysis.unattributed)} item(s) could not be '
            f"placed on a trading day</h3><p>These had no readable publication "
            f"timestamp. They are excluded from the daily aggregation rather than "
            f"attributed on a guess, since a wrong timestamp would credit coverage to "
            f"a price move that happened before it.</p>"
            f'<ul class="src">{items}</ul></div>'
        )

    generated = datetime.now().strftime("%d %B %Y at %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(config.company)} — event impact analysis</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>{escape(config.company)} — news and price impact</h1>
<div class="sub">{escape(config.ticker)} vs {escape(config.benchmark)} &middot;
{config.start:%d %B %Y} to {config.end:%d %B %Y} &middot; generated {generated}</div>

<div class="warn">
<h3>What this report is, and is not</h3>
<p>This is a <strong>structured case study</strong>, not a statistically validated
causal finding, and not investment advice. It identifies days where notable news
coverage <strong>coincided with</strong> an unusual company-specific price move.
Coincidence in time is not evidence that an article caused a price move.</p>
<ul>{caveats}</ul>
</div>

<h2>Summary</h2>
{summary}
<div class="grid">{stats}</div>

<h2>Timeline</h2>
<p>The top panel rebases both the company and the benchmark to 100 at the start of
the window, so they can be compared on percentage terms. The middle panel shows the
<strong>abnormal return</strong> — the company's move once the market's move that day
is removed. The bottom panel shows how many distinct news items were attributed to
each trading day, coloured by tone. Dashed vertical lines mark flagged days.</p>
{timeline_svg(daily, incident_days, config.company, config.benchmark)}

<h2>Candidate incident days</h2>
<p>Ranked by the combination of an unusual abnormal return and notable coverage.
The ranking orders days for attention; it is not a significance test.
<em>*Emotion</em> is a secondary, general-purpose signal (GoEmotions) read
alongside tone, not a substitute for it — see Method and provenance below.</p>
{_incident_table(incidents, config.event_window)}
{_incident_sections(incidents, safe_company, safe_benchmark, config.event_window)}

<h2>Daily detail</h2>
<p>Every trading day in the window. Highlighted rows are flagged days.</p>
{_daily_table(daily_display, incident_days, secondary_ticker)}

{unattributed}

<h2>Method and provenance</h2>
<div class="card">
<p><strong>Abnormal return.</strong> {escape(_sentence(price.get('model_note', '')))}
Prices came from <code>{escape(str(price.get('company_provider', '?')))}</code>
(company) and <code>{escape(str(price.get('benchmark_provider', '?')))}</code>
(benchmark). Abnormal returns are standardised against the
{escape(str(price.get('ar_scale_source', 'unknown scale')))}.</p>
<p><strong>Timestamp alignment.</strong> An item published after the 15:30 IST close is
attributed to the <em>next</em> trading day, since it could not have moved that day's
close. {news_stats.get('after_close', 0)} of the collected items fell after the close.</p>
<p><strong>Sentiment.</strong> Scored with FinBERT, a finance-tuned model, rather than a
general-purpose sentiment library — ordinary financial phrasing such as "beat
expectations but missed guidance" is read incorrectly by generic tools. This
score is what drives incident detection and the abnormal-return-direction
check above.</p>
<p><strong>Emotion.</strong> A secondary tag from GoEmotions (Demszky et al.,
2020), a 27-emotion model trained on Reddit comments — <strong>not</strong> a
finance-tuned model, and reading formal financial-press prose is a genuine
domain mismatch. It is included as texture (fear vs. anger vs. disapproval
alongside a shared "negative" FinBERT score can distinguish, say, a regulatory
probe from a hostile FPO withdrawal) and never affects relevance, incident
flagging, or the direction check. Scored on the headline only, and left blank
below a 30% confidence threshold rather than forced to a low-confidence guess.</p>
<p><strong>Sentiment/return correlation.</strong>
{escape(correlation.get('note', 'Not computed.'))} A Pearson correlation
across a handful of trading days is descriptive, not a significance test —
treat it as a single additional lens on the same daily table above, not as
proof that sentiment predicts price.</p>
{f'<p><strong>Emotion valence vs return.</strong> {escape(emotion_summary.get("note", ""))}</p>{emotion_table}' if emotion_table else ''}
{f'<p><strong>Secondary benchmark ({escape(secondary_ticker)}).</strong> {escape(secondary_meta.get("note", ""))}</p>' if secondary_ticker else (f'<p><strong>Secondary benchmark.</strong> {escape(secondary_meta["note"])}</p>' if secondary_meta.get("note") else '')}
</div>

<h3>Source availability</h3>
{source_table}

<div class="foot">
Generated by the Company Event Impact Analyzer. Descriptive research only —
not investment advice, and not a claim of causation.
</div>

</div></body></html>"""


def write_report(analysis, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(analysis), encoding="utf-8")
    return out
