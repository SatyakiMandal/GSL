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

from .charts import index_sparkline_svg, news_coverage_svg, price_level_svg, timeline_svg
from .eventstudy import Incident
from .narrative import incident_narrative, summary_narrative
from .unlisted import real_updates
from .unlisted_narrative import move_narrative, unlisted_summary_narrative

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1b19;--muted:#6b6862;--line:#e3e1dc;--card:#fff;
--accent:#1a56a8;--bench:#9a958c;--pos:#1f7a4d;--neg:#b3261e;--warn-bg:#fdf6e3;
--warn-br:#d9a441;--plot:#f5f4f1;--macro:#7a4fb5;}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#e9e7e2;--muted:#9e9a92;
--line:#33313a;--card:#1e1d23;--accent:#7fb0f0;--bench:#7d7970;--pos:#5cc48d;
--neg:#f2837a;--warn-bg:#2b2416;--warn-br:#a8802f;--plot:#212027;--macro:#b79aef;}}
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
.stat .v.long{font-size:.82rem;font-weight:500;line-height:1.5;font-variant-numeric:normal}
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
.baseline{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3;opacity:.55}
.incident-badge circle{fill:var(--warn-br);stroke:var(--card);stroke-width:1.5}
.incident-badge text{fill:#fff;font-size:10px;font-weight:700;font-family:inherit}
.badge-date-bg{fill:var(--plot);opacity:.88}
.badge-date{fill:var(--warn-br);font-size:9px;font-weight:700;font-family:inherit}
.real-point{fill:var(--accent);stroke:var(--card);stroke-width:1}
.macro-marker path{fill:var(--macro);stroke:var(--card);stroke-width:1}
.chart-caption{fill:var(--muted);font-size:10.5px;font-style:italic}
.bar-pos{fill:var(--pos)}.bar-neg{fill:var(--neg)}
.bar-incident{stroke:var(--warn-br);stroke-width:1.4}
.tone-neg{fill:var(--neg);opacity:.82}.tone-pos{fill:var(--pos);opacity:.82}
.tone-neutral{fill:var(--muted);opacity:.6}
.hatch-line{stroke:var(--bg);stroke-width:1.6;opacity:.6}
.spark{width:110px;height:26px}
.spark-neg{fill:none;stroke:var(--neg);stroke-width:1.6}
.spark-pos{fill:none;stroke:var(--pos);stroke-width:1.6}
.incident{border:1px solid var(--line);border-left:4px solid var(--warn-br);
border-radius:0 9px 9px 0;padding:18px 22px;margin:20px 0;background:var(--card)}
.rank{display:inline-block;background:var(--warn-br);color:#fff;border-radius:5px;
padding:1px 9px;font-size:.82rem;font-weight:700;margin-right:9px}
.src{list-style:none;padding:0;margin:12px 0 0}
.src li{padding:7px 0;border-top:1px solid var(--line);font-size:.9rem}
.src a{color:var(--accent);text-decoration:none}
.src a:hover{text-decoration:underline}
.src-meta{color:var(--muted);font-size:.8rem}
.src-summary{color:var(--muted);font-size:.85rem;font-style:italic;margin-top:3px;line-height:1.5}
.tag{display:inline-block;font-size:.72rem;padding:1px 7px;border-radius:4px;
background:var(--plot);color:var(--muted);margin-right:7px;border:1px solid var(--line)}
.foot{color:var(--muted);font-size:.85rem;margin-top:52px;padding-top:16px;
border-top:1px solid var(--line)}
code{background:var(--plot);padding:1px 5px;border-radius:4px;font-size:.87em}
.empty{color:var(--muted);font-style:italic}
"""


def _cls(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


def _headline_item(h: dict) -> str:
    """One <li> for the source list behind an incident flag.

    Links to the real article and shows its own summary (a genuine meta
    description/abstract from the source, or a plain-text excerpt of the
    body - see eventstudy.attach_headlines) so a reader can see what the
    story actually said without leaving the report, and follow the link to
    verify it. The URL scheme is checked before being used as an href -
    scraped data, so treated as untrusted input, not just escaped text.
    """
    headline = escape(h.get("headline") or "(no headline)")
    url = h.get("url") or ""
    title = (
        f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{headline}</a>'
        if url.startswith(("http://", "https://")) else headline
    )
    meta = f'[{escape(h.get("source", "?"))}, {escape(h.get("sentiment_label") or "—")}, ' \
          f'rel={h.get("relevance", 0):.2f}]'
    summary = h.get("summary") or ""
    summary_html = f'<div class="src-summary">{escape(summary)}</div>' if summary else ""
    return f'<li>{title} <span class="src-meta">{meta}</span>{summary_html}</li>'


def _pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _stat(key: str, value: str, long: bool = False) -> str:
    """A .grid stat card. ``long=True`` renders the value in a smaller,
    regular-weight style for a full sentence (an unavailability reason, a
    disclosed caveat) rather than the large bold style meant for a short
    number - a long sentence in that style overflowed its card and blew up
    the whole section's height (a real bug this project shipped)."""
    css_class = "v long" if long else "v"
    return f'<div class="stat"><div class="k">{escape(key)}</div><div class="{css_class}">{value}</div></div>'


def _macro_section(macro: dict) -> str:
    """RBI repo rate changes and Brent crude across the window, the latest
    G-Sec yield and fiscal deficit reading, plus the disclosed list of
    indicators checked and still not available (GDP, CPI inflation, IIP) -
    see ``ceia/macro.py``'s module docstring for why each one specifically.
    Context alongside the analysis above, never framed as having driven it.
    """
    if not macro:
        return ""
    events = macro.get("repo_rate_changes") or []
    crude = macro.get("crude_oil") or {}
    gsec = macro.get("gsec_yield") or {}
    deficit = macro.get("fiscal_deficit") or {}
    not_available = macro.get("not_available") or {}

    if events:
        rows = "".join(
            f'<tr><td>{escape(e["date"])}</td>'
            f'<td class="txt">{escape(e["label"])}</td></tr>'
            for e in events
        )
        events_block = (
            '<div class="scroll"><table><thead><tr><th>Date</th>'
            '<th class="txt">Repo rate change</th></tr></thead><tbody>'
            + rows + "</tbody></table></div>"
        )
    else:
        events_block = '<p class="empty">No RBI repo rate change in this window.</p>'

    if crude.get("change") is not None:
        crude_stat = _stat(
            "Brent crude",
            f'{_pct(crude["change"])} (${crude["start_price"]:,.2f} '
            f'→ ${crude["end_price"]:,.2f})',
        )
    else:
        crude_stat = _stat("Brent crude", escape(crude.get("note") or "unavailable"), long=True)

    if gsec.get("value") is not None:
        gsec_stat = _stat(
            "10Y G-Sec yield (latest, not window-scoped)",
            f'{gsec["value"]:.2f}% as of {escape(gsec["as_of"])}',
        )
    else:
        gsec_stat = _stat("10Y G-Sec yield", escape(gsec.get("note") or "unavailable"),
                          long=True)

    if deficit.get("lakh_crore") is not None:
        deficit_stat = _stat(
            "Fiscal deficit (latest budgeted figure, not window-scoped)",
            f'₹{deficit["lakh_crore"]:.2f} lakh crore ({deficit["pct_gdp"]:.1f}% '
            f'of GDP), FY {escape(deficit["fiscal_year"])}',
        )
    else:
        deficit_stat = _stat("Fiscal deficit", escape(deficit.get("note") or "unavailable"),
                             long=True)

    gaps_block = ""
    if not_available:
        items = "".join(
            f"<li><strong>{escape(k)}</strong>: {escape(v)}</li>"
            for k, v in not_available.items()
        )
        gaps_block = (
            '<p class="note">Checked and not included in this report, rather '
            f"than silently omitted:</p><ul class=\"note\">{items}</ul>"
        )

    return f"""
<h2>Macro-economic backdrop</h2>
<p>RBI repo rate changes in this window (also marked on the timeline above as
small diamonds), and Brent crude's move across it — context alongside the
price action above, not a claim that either one drove it. The G-Sec yield
and fiscal deficit stats below are the latest available reading in either
case, not a value as of this report's own window — each is labelled with the
date or fiscal year it actually applies to.</p>
<div class="grid">{crude_stat}{gsec_stat}{deficit_stat}</div>
{events_block}
{gaps_block}
"""


def _nifty_section(nifty_indices: dict, incidents: list, daily: pd.DataFrame,
                   ticker: str) -> str:
    """Nifty 50 and five sector indices as reference lines/stats alongside
    the company's own analysis - the professor's request, reused rather than
    re-run as a second event study: each index's own rebased trend and
    window return, plus how often it moved the same direction as the
    company on the days already flagged as candidate incidents. Descriptive
    only, like the macro-economic section above it - never part of flagging.
    """
    if not nifty_indices:
        return ""
    from .nifty import same_direction_rate
    candidate_days = [i.day for i in incidents]
    rows = []
    for name, index in nifty_indices.items():
        if not index.available:
            rows.append(
                f"<tr><td>{escape(name)}</td><td class=\"txt\">{escape(index.ticker)}</td>"
                f'<td colspan="3" class="txt note">{escape(index.note or "unavailable")}</td></tr>'
            )
            continue
        agreement = same_direction_rate(index, daily, candidate_days)
        agree_cell = (f"{agreement['agree']}/{agreement['n']}"
                     if agreement["n"] else "—")
        rows.append(
            f"<tr><td>{escape(name)}</td><td class=\"txt\">{escape(index.ticker)}</td>"
            f'<td class="{_cls(index.window_return)}">{_pct(index.window_return)}</td>'
            f"<td>{index_sparkline_svg(index.daily)}</td>"
            f"<td>{agree_cell}</td></tr>"
        )
    return f"""
<h2>Nifty sector indices</h2>
<p>Nifty 50 and five sector indices, rebased and read the same window as
{escape(ticker)}'s own price action above. This summary table is
descriptive only — none of it changes which days are flagged as candidate
incidents below. <em>Moved with</em> is a quick raw-return coincidence
check: of the candidate incident days already flagged, how many this index
also moved on (same sign of daily return). Each ranked candidate incident
below carries a fuller, statistical version of this same question — that
index's own market-model abnormal return, CAR, and significance on that
specific day, not just its raw return's sign.</p>
<div class="scroll"><table><thead><tr><th class="txt">Index</th>
<th class="txt">Ticker</th><th>Window return</th><th>Trend</th>
<th>Moved with {escape(ticker)}</th></tr></thead><tbody>
{"".join(rows)}</tbody></table></div>
"""


def _global_markets_section(global_indices: dict, incidents: list, daily: pd.DataFrame,
                            ticker: str) -> str:
    """S&P 500/Nasdaq/Dow/FTSE 100/Hang Seng/Nikkei 225 as a "ripple effect"
    reference backdrop - same descriptive shape as the Nifty section above,
    but each index's own trading day is timezone-aligned onto this
    company's NSE trading day first (see ceia/global_markets.py's module
    docstring for why a naive same-calendar-date comparison would be wrong
    for a market that does not share NSE's hours). Descriptive only, like
    every other backdrop section - never part of flagging.
    """
    if not global_indices:
        return ""
    from .global_markets import same_direction_rate
    candidate_days = [i.day for i in incidents]
    rows = []
    for name, index in global_indices.items():
        if not index.available:
            rows.append(
                f"<tr><td>{escape(name)}</td><td class=\"txt\">{escape(index.ticker)}</td>"
                f'<td colspan="3" class="txt note">{escape(index.note or "unavailable")}</td></tr>'
            )
            continue
        agreement = same_direction_rate(index, daily, candidate_days)
        agree_cell = (f"{agreement['agree']}/{agreement['n']}"
                     if agreement["n"] else "—")
        alignment = ("same NSE day" if index.same_day_available
                    else "prior trading day")
        rows.append(
            f"<tr><td>{escape(name)}</td><td class=\"txt\">{escape(index.ticker)}</td>"
            f'<td class="{_cls(index.window_return)}">{_pct(index.window_return)}</td>'
            f"<td>{index_sparkline_svg(index.daily)}</td>"
            f'<td class="txt">{escape(alignment)}</td>'
            f"<td>{agree_cell}</td></tr>"
        )
    return f"""
<h2>Global markets</h2>
<p>How major overseas indices moved around {escape(ticker)}'s own trading
days — context, not a claim that a move overseas drove anything here.
Because none of these markets share NSE's 9:15am–3:30pm IST hours, each
index is paired with whichever of *its own* sessions was actually complete
and known by the time that NSE day mattered: Hang Seng and Nikkei 225 close
before NSE does, so their own same-dated session applies; the S&amp;P 500,
Nasdaq, Dow, and FTSE 100 close overnight or after NSE's own close, so the
*prior* trading day's session is the one actually known (see the
Alignment column). <em>Moved with</em> is the same descriptive coincidence
check the Nifty table above uses — of the flagged candidate days, how often
this index's aligned session moved the same direction.</p>
<div class="scroll"><table><thead><tr><th class="txt">Index</th>
<th class="txt">Ticker</th><th>Window return</th><th>Trend</th>
<th class="txt">Alignment</th><th>Moved with {escape(ticker)}</th></tr></thead><tbody>
{"".join(rows)}</tbody></table></div>
"""


def _financials_row_stat(key: str, row: dict | None) -> str:
    if not row:
        return _stat(key, "unavailable")
    parts = [f"{row['latest']:,.0f}"]
    if row.get("qoq_change") is not None:
        parts.append(f"{row['qoq_change'] * 100:+.1f}% QoQ")
    if row.get("yoy_change") is not None:
        parts.append(f"{row['yoy_change'] * 100:+.1f}% YoY")
    return _stat(f"{key} ({escape(row['label'])})", " · ".join(parts))


def _financials_section(financials: dict) -> str:
    """Revenue growth, operating expense, NOPAT and order book (professor's
    note, added after the news/price event-study work) - descriptive
    backdrop only, like the macro-economic and Nifty sections above, never
    tied to candidate-day flagging or any significance test. Always the
    latest reported quarter, not a value scoped to this report's own
    window - see ``ceia/financials.py``'s module docstring for the
    screener.in source, and why "which row means what" is sector-dependent
    (a bank's page uses different labels than an industrial company's) and
    why Order Book is disclosed as unavailable for most companies rather
    than silently omitted.
    """
    if not financials:
        return ""
    if financials.get("note"):
        return (
            "<h2>Financial fundamentals</h2>"
            f'<p class="empty">{escape(financials["note"])}</p>'
        )

    revenue_stat = _financials_row_stat("Revenue", financials.get("revenue"))
    expense_stat = _financials_row_stat("Operating expense", financials.get("expenses"))

    if financials.get("nopat") is not None:
        nopat_stat = _stat(
            "NOPAT", f"{financials['nopat']:,.0f} {escape(financials['currency_unit'])}")
    else:
        nopat_stat = _stat("NOPAT", escape(financials.get("nopat_note") or "not computed"),
                          long=True)

    order_book = financials.get("order_book")
    if order_book:
        order_book_stat = _stat(
            "Order book", f"{order_book['latest']:,.0f} {escape(financials['currency_unit'])}")
    else:
        order_book_stat = _stat(
            "Order book", escape(financials.get("order_book_note") or "unavailable"),
            long=True)

    return f"""
<h2>Financial fundamentals</h2>
<p>Latest reported quarter ({escape(str(financials.get("as_of") or "?"))},
{escape(financials["statement_kind"])} figures, {escape(financials["currency_unit"])}) from
<a href="{escape(financials["screener_url"])}" target="_blank" rel="noopener">screener.in</a> —
context alongside the price/news analysis above, never scoped to this
report's own date window and never part of candidate-day flagging. NOPAT is
computed as operating income × (1 − tax rate), both from the same latest
quarter — which row counts as "operating income" is sector-dependent (a
bank's page uses a different label than an industrial company's), named
alongside the figure below.</p>
<div class="grid">{revenue_stat}{expense_stat}{nopat_stat}{order_book_stat}</div>
"""


def _volume_cell(row: pd.Series) -> str:
    volume = row.get("volume")
    if volume is None or pd.isna(volume):
        return "<td>—</td>"
    volume_z = row.get("volume_z")
    z_part = f" (z={float(volume_z):+.1f})" if volume_z is not None and pd.notna(volume_z) else ""
    return f"<td>{float(volume):,.0f}{z_part}</td>"


def _staleness_cell(row: pd.Series) -> str:
    staleness = row.get("mean_staleness")
    if staleness is None or pd.isna(staleness):
        return "<td>—</td>"
    return f"<td>{float(staleness):.2f}</td>"


def _daily_table(daily: pd.DataFrame, incident_days: set[date],
                 secondary_ticker: str | None = None) -> str:
    if daily.empty:
        return '<p class="empty">No trading days in the analysis window.</p>'
    has_volume = "volume" in daily.columns
    has_secondary = secondary_ticker and "secondary_abnormal_return" in daily.columns
    has_staleness = "mean_staleness" in daily.columns
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
            + (_staleness_cell(row) if has_staleness else "")
            + f"<td class=\"txt\">{escape(str(row['dominant_event'] or '—'))}</td></tr>"
        )
    volume_header = "<th>Volume</th>" if has_volume else ""
    secondary_header = (f"<th>Abnormal vs {escape(secondary_ticker)}</th>"
                        if has_secondary else "")
    staleness_header = "<th>Staleness</th>" if has_staleness else ""
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Date</th><th>Close</th><th>Return</th><th>Benchmark</th>"
        f"<th>Abnormal</th><th>z</th>{secondary_header}{volume_header}"
        f"<th>Items</th><th>Tone</th>{staleness_header}"
        '<th class="txt">Main topic</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


def _incident_table(incidents: list[Incident], window: tuple[int, int],
                    robustness: dict | None = None) -> str:
    if not incidents:
        return ('<p class="empty">No day combined notable coverage with an unusual '
                "abnormal return at the configured thresholds.</p>")
    robust_days = (robustness or {}).get("days") or {}
    rows = []
    for rank, inc in enumerate(incidents, 1):
        car = inc.car or {}
        car_value = car.get("car")
        t_stat = car.get("t_stat")
        p_value = car.get("p_value")
        p_value_t = car.get("p_value_t")
        robust = robust_days.get(inc.day.isoformat())
        robust_cell = (f"{robust['flagged_in']}/{robust['of']}" if robust else "—")
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
            f"<td>{f'{p_value:.3f}' if p_value is not None else '—'}</td>"
            f"<td>{f'{p_value_t:.3f}' if p_value_t is not None else '—'}</td>"
            f"<td>{robust_cell}</td>"
            f'<td class="txt">{"consistent" if inc.direction_agrees else "opposite"}</td>'
            f'<td class="txt">{escape(inc.dominant_event or "—")}</td>'
            f'<td class="txt">{escape(inc.dominant_emotion or "—")}</td></tr>'
        )
    before, after = window
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>#</th><th>Date</th><th>Abnormal return</th><th>z</th><th>Items</th>"
        f"<th>Tone</th><th>CAR[{before},+{after}]</th><th>t</th><th>p**</th><th>p(t)†</th>"
        '<th>Robust***</th>'
        '<th class="txt">Tone vs price</th><th class="txt">Main topic</th>'
        '<th class="txt">Emotion*</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _index_breakdown_table(day, window: tuple[int, int], nifty_indices: dict) -> str:
    """For one incident day, how each Nifty index moved - its own
    market-model abnormal return, CAR and significance over the same event
    window, anchored to this already-flagged day rather than an independent
    search for the index's own events (see ``ceia/nifty.py``). Empty when no
    index has an event study computed for this day (unavailable, skipped as
    the primary benchmark itself, or the event study wasn't requested).
    """
    day_key = day.isoformat()
    rows = []
    for name, index in nifty_indices.items():
        stats = index.incident_stats.get(day_key)
        if not stats:
            continue
        car_value = stats.get("car")
        t_stat = stats.get("t_stat")
        p_value = stats.get("p_value")
        p_value_t = stats.get("p_value_t")
        rows.append(
            f"<tr><td class=\"txt\">{escape(name)}</td>"
            f'<td class="{_cls(stats["abnormal_return"])}">'
            f'{_pct(stats["abnormal_return"])}</td>'
            f"<td>{stats['abnormal_return_z']:+.1f}</td>"
            f'<td class="{_cls(car_value or 0)}">'
            f"{_pct(car_value) if car_value is not None and pd.notna(car_value) else '—'}</td>"
            f"<td>{f'{t_stat:.2f}' if t_stat is not None else '—'}</td>"
            f"<td>{f'{p_value:.3f}' if p_value is not None else '—'}</td>"
            f"<td>{f'{p_value_t:.3f}' if p_value_t is not None else '—'}</td></tr>"
        )
    if not rows:
        return ""
    before, after = window
    return (
        '<h4 style="margin:16px 0 4px;font-size:.92rem">How the Nifty indices moved '
        "on this same day — each index's own abnormal return against "
        "the same benchmark the company is measured against, not evidence "
        "either moved the other</h4>"
        '<div class="scroll"><table><thead><tr><th class="txt">Index</th>'
        f"<th>Abnormal</th><th>z</th><th>CAR[{before},+{after}]</th>"
        "<th>t</th><th>p**</th><th>p(t)†</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _global_market_breakdown_table(day, global_indices: dict) -> str:
    """For one incident day, how each overseas index's *aligned* session
    moved (see ``ceia/global_markets.py`` for why that's not simply "the
    same calendar date"). A self-relative z-score (against that index's own
    historical daily-return distribution), not a market-model abnormal
    return against ^NSEI - the two markets don't share trading hours, so a
    cross-market regression would assume a relationship that isn't there.
    Empty when no configured index has aligned data for this day.
    """
    rows = []
    for name, index in global_indices.items():
        aligned = index.aligned.get(day)
        if aligned is None:
            continue
        alignment = ("same NSE day" if index.same_day_available
                    else "prior trading day")
        rows.append(
            f"<tr><td class=\"txt\">{escape(name)}</td>"
            f"<td class=\"txt\">{aligned.aligned_date:%d %b %Y}</td>"
            f'<td class="txt">{escape(alignment)}</td>'
            f'<td class="{_cls(aligned.return_)}">{_pct(aligned.return_)}</td>'
            f"<td>{aligned.return_z:+.1f}</td></tr>"
        )
    if not rows:
        return ""
    return (
        '<h4 style="margin:16px 0 4px;font-size:.92rem">How global markets moved '
        "around this day — each index's own aligned-session return and its own "
        "z-score, not evidence either moved the other</h4>"
        '<div class="scroll"><table><thead><tr><th class="txt">Index</th>'
        '<th class="txt">Aligned session</th><th class="txt">Alignment</th>'
        "<th>Return</th><th>z</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _incident_sections(incidents: list[Incident], company: str, benchmark: str,
                       window: tuple[int, int], nifty_indices: dict | None = None,
                       global_indices: dict | None = None) -> str:
    if not incidents:
        return ""
    nifty_indices = nifty_indices or {}
    global_indices = global_indices or {}
    blocks = []
    for rank, inc in enumerate(incidents, 1):
        paragraphs = "".join(
            f"<p>{text}</p>"
            for text in incident_narrative(inc, company, benchmark, window)
        )
        sources = "".join(_headline_item(h) for h in inc.headlines)
        source_block = (
            f'<h4 style="margin:16px 0 4px;font-size:.92rem">Coverage behind this flag — '
            f'linked to the original article, where a source stated one</h4>'
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
        p_value = (inc.car or {}).get("p_value")
        index_block = (_index_breakdown_table(inc.day, window, nifty_indices)
                      + _global_market_breakdown_table(inc.day, global_indices))
        blocks.append(
            f'<div class="incident"><h3><span class="rank">#{rank}</span>'
            f"{inc.day:%d %B %Y}</h3>"
            f'<div><span class="tag">abnormal {_pct(inc.abnormal_return)}</span>'
            f'<span class="tag">z {inc.abnormal_return_z:+.1f}</span>'
            f'<span class="tag">{inc.item_count} item(s)</span>'
            f'<span class="tag">{escape(inc.dominant_event or "other")}</span>'
            + (f'<span class="tag">emotion: {escape(inc.dominant_emotion)}</span>'
               if inc.dominant_emotion else "")
            + (f'<span class="tag">CAR permutation p={p_value:.3f}</span>'
               if p_value is not None else "")
            + "</div>"
            f"{paragraphs}{volume_note}{index_block}{source_block}</div>"
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

    diagnostics = getattr(analysis, "diagnostics", {}) or {}
    robustness = getattr(analysis, "robustness", {}) or {}
    macro_events = getattr(analysis, "macro_events", []) or []
    weak_scale = "analysis-window" in str(price.get("ar_scale_source", ""))

    summary = "".join(
        f"<p>{text}</p>" for text in summary_narrative(
            safe_company, escape(config.ticker), safe_benchmark,
            config.start, config.end, incidents, len(daily), news_count,
            price.get("model", "market-adjusted"),
            diagnostics=diagnostics, robustness=robustness, weak_scale=weak_scale,
        )
    )

    correlation = getattr(analysis, "correlation", {}) or {}
    corr_display = (f"r = {correlation['r']:+.3f}"
                    if correlation.get("r") is not None else "n/a")

    extremity_corr = getattr(analysis, "extremity_volume_correlation", {}) or {}
    extremity_stats = (
        [_stat("Sentiment extremity/volume correlation",
              f"r = {extremity_corr['r']:+.3f}")]
        if extremity_corr.get("r") is not None else []
    )
    lagged_horizons = (getattr(analysis, "lagged_correlation", {}) or {}).get("horizons") or {}
    lagged_stats = [
        _stat(f"Sentiment → return {h}d later", f"r = {result['r']:+.3f}")
        for h, result in sorted(lagged_horizons.items())
        if result.get("r") is not None
    ]

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
    ] + extremity_stats + lagged_stats
    + ([_stat(f"Beta vs {secondary_ticker}", f"{secondary_meta.get('beta', float('nan')):.2f}")]
        if secondary_ticker and secondary_meta.get("beta") is not None else []))

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

<h2>Summary</h2>
{summary}
<div class="grid">{stats}</div>

<h2>Timeline</h2>
<p>The top panel rebases both the company and the benchmark to 100 at the start of
the window, so they can be compared on percentage terms; the dotted line marks that
starting level. The middle panel shows the <strong>abnormal return</strong> — the
company's move once the market's move that day is removed. The bottom panel shows how
many distinct news items were attributed to each trading day, coloured by tone (a
hatched fill marks negative-tone bars as a colour-independent cue). Dashed vertical
lines and numbered badges mark the ranked candidate incident days below — badge
<strong>#1</strong> is the highest-ranked candidate, and so on. Small diamonds
along the bottom of the top panel mark RBI repo rate changes — economic
context, not a candidate incident, so they are not numbered or ranked below.
Hover any bar or marker for its exact date and value.</p>
{timeline_svg(daily, incident_days, config.company, config.benchmark, incidents=incidents, macro_events=macro_events)}
{_macro_section(getattr(analysis, "macro", {}) or {})}
{_nifty_section(getattr(analysis, "nifty_indices", {}) or {}, incidents, daily, config.ticker)}
{_global_markets_section(getattr(analysis, "global_indices", {}) or {}, incidents, daily, config.ticker)}
{_financials_section(getattr(analysis, "financials", {}) or {})}

<h2>Candidate incident days</h2>
<p>Ranked by the combination of an unusual abnormal return and notable coverage.
The ranking orders days for attention; it is not a significance test.
<em>*Emotion</em> is a secondary, general-purpose signal (GoEmotions) read
alongside tone, not a substitute for it. <em>**p</em> is a permutation-test
p-value for the CAR — the fraction of random comparable-length windows in
this stock's own price history with as extreme a move, an alternative to
the <code>t</code> column that doesn't need to assume a large, independent,
normally distributed sample. <em>†p(t)</em> is the classic two-tailed
Student's-t p-value for the same <code>t</code> statistic, shown alongside
(not instead of) the permutation p-value; it also accounts for how many
estimation-window observations the residual scale was fitted on, so it
widens on short estimation windows rather than assuming a large sample.
<em>***Robust</em> counts how many of a 3×3
grid of nearby coverage/return threshold choices still flag this day (9 is
the most robust; a day flagged in only 1–2 is threshold-sensitive). Where a
Nifty index event study was run (see below), each flagged day also shows how
that index moved on the same day, using the same <code>t</code>/<em>p**</em>/
<em>p(t)†</em> statistics computed for that index against the same
benchmark — a coincidence check for whether the move reached beyond this one
stock, not evidence either one drove the other.</p>
{_incident_table(incidents, config.event_window, robustness)}
{_incident_sections(incidents, safe_company, safe_benchmark, config.event_window, getattr(analysis, "nifty_indices", {}) or {}, getattr(analysis, "global_indices", {}) or {})}

<h2>Daily detail</h2>
<p>Every trading day in the window. Highlighted rows are flagged days.
<em>Staleness</em> is this day's coverage's average textual similarity to
this company's own most recent prior stories (0 = entirely new content, 1 =
a near-exact rehash) — high staleness is associated in the literature with
a smaller, more easily reversed price reaction (Tetlock, 2011); it is
descriptive context, never part of the flagging test.</p>
{_daily_table(daily_display, incident_days, secondary_ticker)}

{unattributed}

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


def _move_table(moves: list) -> str:
    if not moves:
        return ('<p class="empty">Not enough price revisions in this window '
                "to identify a move.</p>")
    rows = []
    for rank, move in enumerate(moves, 1):
        rows.append(
            f"<tr><td>{rank}</td>"
            f"<td>{move.start_date:%d %b %Y} – {move.end_date:%d %b %Y}</td>"
            f"<td>₹{move.start_price:,.2f}</td>"
            f"<td>₹{move.end_price:,.2f}</td>"
            f"<td class=\"{_cls(move.change)}\"><strong>{_pct(move.change)}</strong></td>"
            f"<td>{len(move.headlines)}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>#</th><th>Date range</th><th>Start</th><th>End</th>"
        "<th>Change</th><th>Items</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _move_sections(moves: list, company: str) -> str:
    if not moves:
        return ""
    blocks = []
    for rank, move in enumerate(moves, 1):
        paragraphs = "".join(f"<p>{text}</p>" for text in move_narrative(move, company))
        sources = "".join(_headline_item(h) for h in move.headlines)
        source_block = (
            '<h4 style="margin:16px 0 4px;font-size:.92rem">Coverage in this '
            'window — linked to the original article, where a source stated '
            f'one</h4><ul class="src">{sources}</ul>' if sources else ""
        )
        blocks.append(
            f'<div class="incident"><h3><span class="rank">#{rank}</span>'
            f"{move.start_date:%d %b %Y} – {move.end_date:%d %b %Y}</h3>"
            f'<div><span class="tag">{_pct(move.change)}</span>'
            f'<span class="tag">{len(move.headlines)} item(s)</span></div>'
            f"{paragraphs}{source_block}</div>"
        )
    return "".join(blocks)


def _news_table(items: list, start: date, end: date) -> str:
    """Every collected item published inside the window, oldest first — the
    same underlying data the per-move ``Coverage in this window`` lists
    already show, but as one flat table so a reader can scan tone and
    relevance across the whole run without opening each move section."""
    dated = sorted(
        (i for i in items if i.published_at is not None
         and start <= i.published_at.date() <= end),
        key=lambda i: i.published_at,
    )
    if not dated:
        return '<p class="empty">No dated coverage found in this window.</p>'

    rows = []
    for i in dated:
        headline = escape(i.headline or "(no headline)")
        title = (
            f'<a href="{escape(i.url)}" target="_blank" rel="noopener noreferrer">{headline}</a>'
            if i.url.startswith(("http://", "https://")) else headline
        )
        tone_cls = ("pos" if i.sentiment_score > 0.15
                   else "neg" if i.sentiment_score < -0.15 else "")
        rows.append(
            f"<tr><td>{i.published_at:%d %b %Y}</td>"
            f'<td class="txt">{title}</td>'
            f"<td>{escape(i.source)}</td>"
            f'<td class="{tone_cls}">{escape(i.sentiment_label or "—")} '
            f"({i.sentiment_score:+.2f})</td>"
            f"<td>{i.relevance_score:.2f}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        '<th>Date</th><th class="txt">Headline</th><th>Source</th>'
        "<th>Tone</th><th>Relevance</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def build_unlisted_html(analysis) -> str:
    """Render a :class:`ceia.unlisted.UnlistedAnalysis` as a standalone HTML
    document.

    A distinct template from :func:`build_html`, not a bent version of it —
    see ``ceia/unlisted.py``'s module docstring for why: no z, CAR, t, p, or
    Robust column here, because none of those are earned by a periodically
    revised indicative price.
    """
    config = analysis.config
    safe_company = escape(config.company)
    ranked = analysis.ranked_moves()

    news_stats = analysis.news_meta.get("stats", {}) or {}
    news_count = news_stats.get("unique_after_dedupe")
    if news_count is None:
        news_count = news_stats.get("relevant", 0)

    calendar_days = (config.end - config.start).days + 1

    summary = "".join(
        f"<p>{text}</p>" for text in unlisted_summary_narrative(
            safe_company, config.start, config.end, ranked, calendar_days, news_count,
        )
    )

    real = real_updates(analysis.series)
    real_dates = {pd.Timestamp(d).date() for d in real.index}

    stats = "".join([
        _stat("Calendar days", str(calendar_days)),
        _stat("Price revisions", str(len(real))),
        _stat("Notable moves", str(len(ranked))),
        _stat("News items", str(news_count)),
    ])

    unattributed = ""
    if analysis.unattributed:
        items = "".join(
            f"<li>[{escape(i.source)}] {escape(i.headline)}</li>"
            for i in analysis.unattributed[:20]
        )
        unattributed = (
            f'<div class="warn"><h3>{len(analysis.unattributed)} item(s) could '
            f"not be placed in a price-move window</h3><p>These had no readable "
            f"publication timestamp, so there is no date to compare against the "
            f"price series.</p>"
            f'<ul class="src">{items}</ul></div>'
        )

    generated = datetime.now().strftime("%d %B %Y at %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(config.company)} — unlisted price timeline</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>{escape(config.company)} — indicative price and news timeline</h1>
<div class="sub">Unlisted / pre-IPO &middot;
{config.start:%d %B %Y} to {config.end:%d %B %Y} &middot; generated {generated}</div>

<h2>Summary</h2>
{summary}
<div class="grid">{stats}</div>

<h2>Price timeline</h2>
<p>The line shows the indicative price exactly as UnlistedZone displays it —
held flat between revisions, not observed daily. Small dots mark the dates the
price was actually revised; everywhere else is a forward-filled display value,
not a new observation. Dashed vertical lines and numbered badges mark the
ranked price moves below. Small diamonds along the bottom mark RBI repo rate
changes — economic context, not a price move, so not part of the ranking.</p>
{price_level_svg(analysis.series, real_dates, config.company, moves=ranked, macro_events=getattr(analysis, "macro_events", []) or [])}
{_macro_section(getattr(analysis, "macro", {}) or {})}

<h2>News coverage</h2>
<p>Every collected item's publication day, across the whole window — bar
height is the day's item count, colour is the mean sentiment tone that day.
This is purely descriptive: unlike the price panel above, it needs nothing
from a benchmark or a market model, so it exists here even though this
report computes no abnormal return (see the Summary above for why).</p>
{news_coverage_svg(analysis.series, analysis.items, config.company)}
{_news_table(analysis.items, config.start, config.end)}

<h2>Notable price moves</h2>
<p>Ranked by the size of the raw change between one price revision and the
next. This is a description of what changed and what was published in the
same window, not a significance test — see the Summary above for why no
z-score, market-model beta, or permutation p-value is computed here.</p>
{_move_table(ranked)}
{_move_sections(ranked, safe_company)}

{unattributed}

<div class="foot">
Generated by the Company Event Impact Analyzer. Descriptive research only —
not investment advice, and not a claim of causation. Indicative prices via
UnlistedZone; not a price feed, quote, or offer to deal.
</div>

</div></body></html>"""


def write_unlisted_report(analysis, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_unlisted_html(analysis), encoding="utf-8")
    return out
