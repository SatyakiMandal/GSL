"""Inline SVG chart generation.

Charts are hand-built SVG rather than matplotlib PNGs for three reasons: the
report stays a single self-contained file with no image assets, the output is
vector (so it stays crisp when a reader zooms or prints to PDF), and it adds no
dependency beyond what ingestion already needs.

Colours are supplied via CSS custom properties and `currentColor` where possible
so the same markup works in the report's light and dark themes.
"""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd

# Panel geometry. The viewBox makes everything scale to the container width.
WIDTH = 960
PAD_LEFT = 62
PAD_RIGHT = 18
PAD_TOP = 26
LABEL_BAND = 34


def _x_positions(n: int, width: int) -> list[float]:
    """Evenly spaced band centres across the plot area."""
    if n <= 0:
        return []
    inner = width - PAD_LEFT - PAD_RIGHT
    if n == 1:
        return [PAD_LEFT + inner / 2]
    step = inner / n
    return [PAD_LEFT + step * (i + 0.5) for i in range(n)]


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round-ish tick values spanning [low, high]."""
    if low == high:
        return [low]
    span = high - low
    raw = span / max(count - 1, 1)
    magnitude = 10 ** (len(f"{int(abs(raw))}") - 1) if abs(raw) >= 1 else 1
    while magnitude > abs(raw) * 2 and magnitude > 1e-9:
        magnitude /= 10
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if step >= raw:
            break
    start = step * (low // step)
    ticks, value = [], start
    while value <= high + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _date_labels(dates: list[date], xs: list[float], max_labels: int = 9) -> str:
    if not dates:
        return ""
    stride = max(1, len(dates) // max_labels)
    parts = []
    for i, (day, x) in enumerate(zip(dates, xs)):
        if i % stride and i != len(dates) - 1:
            continue
        parts.append(
            f'<text x="{x:.1f}" y="0" class="tick" text-anchor="middle" '
            f'transform="rotate(-38 {x:.1f} 0)">{day:%d %b}</text>'
        )
    return "".join(parts)


def _panel_frame(y0: float, height: float, title: str) -> str:
    return (
        f'<text x="{PAD_LEFT}" y="{y0 - 8:.1f}" class="panel-title">{escape(title)}</text>'
        f'<rect x="{PAD_LEFT}" y="{y0:.1f}" width="{WIDTH - PAD_LEFT - PAD_RIGHT}" '
        f'height="{height:.1f}" class="plot-bg"/>'
    )


def _y_axis(y0: float, height: float, low: float, high: float,
            fmt: str = "{:.0f}") -> str:
    parts = []
    for tick in _nice_ticks(low, high):
        if high == low:
            continue
        y = y0 + height - (tick - low) / (high - low) * height
        if y < y0 - 1 or y > y0 + height + 1:
            continue
        parts.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}" '
            f'class="gridline"/>'
            f'<text x="{PAD_LEFT - 8}" y="{y + 3.5:.1f}" class="tick" '
            f'text-anchor="end">{fmt.format(tick)}</text>'
        )
    return "".join(parts)


def timeline_svg(daily: pd.DataFrame, incident_days: set[date],
                 company: str, benchmark: str, incidents: list | None = None) -> str:
    """Three stacked panels: rebased prices, abnormal returns, coverage.

    Prices are rebased to 100 at the window's first session so a stock priced in
    thousands and an index priced in tens can share an axis and be compared on
    percentage terms, which is what the eye should be reading here.

    ``incidents`` (optional) is the ranked candidate list from
    ``eventstudy.rank_incidents`` — when supplied, each flagged day gets a
    numbered badge in the top panel matching its rank in the incident table
    below, and richer hover text on its abnormal-return bar, so a reader can
    tell *which* flagged day is which without leaving the chart. Without it
    (or for a day in ``incident_days`` that isn't in ``incidents``), the day
    still gets its dashed marker line and outlined bar, just no number.
    """
    if daily.empty:
        return '<p class="empty">No trading days in the analysis window.</p>'

    incident_rank: dict[date, int] = {}
    incident_tooltip: dict[date, str] = {}
    for rank, incident in enumerate(incidents or [], 1):
        incident_rank[incident.day] = rank
        incident_tooltip[incident.day] = (
            f"#{rank} {incident.day:%d %b %Y}: abnormal return "
            f"{incident.abnormal_return * 100:+.2f}%, {incident.item_count} "
            "news item(s) — see the ranked list below"
        )

    # pandas Timestamp subclasses datetime.date, so an isinstance check would
    # leave Timestamps untouched and they never compare equal to the plain
    # dates in ``incident_days`` — silently dropping every incident marker.
    # Normalise unconditionally instead.
    dates = [pd.Timestamp(d).date() for d in daily.index]
    n = len(dates)
    xs = _x_positions(n, WIDTH)

    # Reserved only when there is something to caption. The caption exists
    # because a numbered badge with no on-chart explanation is only
    # meaningful to a reader who also has the paragraph above the chart in
    # front of them - true in a browser, not true once the SVG is printed,
    # screenshotted, or embedded in a PDF on its own (see ceia/pdf.py).
    has_badges = bool(incidents)
    top_caption_h = 18.0 if has_badges else 0.0

    price_h, abn_h, cov_h = 190.0, 130.0, 110.0
    gap = 46.0
    price_y = PAD_TOP + top_caption_h
    abn_y = price_y + price_h + gap
    cov_y = abn_y + abn_h + gap
    total_h = cov_y + cov_h + LABEL_BAND + 14

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {total_h:.0f}" class="timeline" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Price, abnormal return and coverage timeline">'
        # Negative-tone coverage bars (panel 3) are otherwise distinguished
        # from positive ones by colour alone - this hatch overlay gives a
        # second, colour-independent cue for readers who can't rely on the
        # red/green difference.
        '<defs><pattern id="neg-hatch" width="6" height="6" '
        'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
        '<line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>'
    ]
    if has_badges:
        parts.append(
            f'<text x="{PAD_LEFT}" y="{PAD_TOP - 4:.1f}" class="chart-caption">'
            "Numbered circles mark the ranked candidate incident days below, "
            "each dated underneath — hover any bar for its exact date and "
            "value.</text>"
        )

    # ---- Panel 1: rebased price vs benchmark -----------------------------
    closes = daily["close"].astype(float).tolist()
    bench = (daily["benchmark_return"].fillna(0) + 1).cumprod().tolist()
    base_close = closes[0] or 1.0
    company_idx = [c / base_close * 100 for c in closes]
    bench_idx = [b / (bench[0] or 1.0) * 100 for b in bench]

    low = min(min(company_idx), min(bench_idx))
    high = max(max(company_idx), max(bench_idx))
    span = (high - low) or 1.0
    low, high = low - span * 0.08, high + span * 0.08

    def price_y_of(v: float) -> float:
        return price_y + price_h - (v - low) / (high - low) * price_h

    parts.append(_panel_frame(price_y, price_h,
                              f"{company} vs {benchmark} — rebased to 100"))
    parts.append(_y_axis(price_y, price_h, low, high))

    # A reference line at the rebased starting level (100 = "unchanged since
    # day one"). Without it, "is the stock up or down since the window
    # started" requires reading the y-axis scale; with it, it is a glance.
    if low <= 100 <= high:
        base_y = price_y_of(100.0)
        parts.append(
            f'<line x1="{PAD_LEFT}" y1="{base_y:.1f}" x2="{WIDTH - PAD_RIGHT}" '
            f'y2="{base_y:.1f}" class="baseline"/>'
            f'<text x="{PAD_LEFT + 4}" y="{base_y - 3:.1f}" class="tick">'
            "start of window</text>"
        )

    for values, css in ((bench_idx, "line-benchmark"), (company_idx, "line-company")):
        points = " ".join(f"{x:.1f},{price_y_of(v):.1f}" for x, v in zip(xs, values))
        parts.append(f'<polyline points="{points}" class="{css}"/>')

    badge_r = 9.0
    badge_cy = price_y + 15.0
    for x, day in zip(xs, dates):
        if day not in incident_days:
            continue
        parts.append(
            f'<line x1="{x:.1f}" y1="{price_y:.1f}" x2="{x:.1f}" '
            f'y2="{price_y + price_h:.1f}" class="incident-rule"/>'
        )
        rank = incident_rank.get(day)
        if rank is None:
            continue
        tooltip = escape(incident_tooltip.get(day, f"{day:%d %b %Y}: flagged incident"))
        parts.append(
            f'<g class="incident-badge">'
            f'<circle cx="{x:.1f}" cy="{badge_cy:.1f}" r="{badge_r}"/>'
            f'<text x="{x:.1f}" y="{badge_cy + 3.5:.1f}" text-anchor="middle">{rank}</text>'
            f"<title>{tooltip}</title></g>"
        )
        # The date each badge refers to, always visible - not only in the
        # <title> tooltip, which needs a mouse hover and so renders as
        # nothing at all once this SVG is screenshotted or exported to PDF
        # (see ceia/pdf.py). A small background rect keeps it legible where
        # it crosses the price lines behind it.
        date_label = f"{day:%d %b}"
        label_y = badge_cy + badge_r + 11.0
        label_w = len(date_label) * 5.4 + 6.0
        parts.append(
            f'<rect x="{x - label_w / 2:.1f}" y="{label_y - 9.0:.1f}" '
            f'width="{label_w:.1f}" height="12" rx="2" class="badge-date-bg"/>'
            f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'class="badge-date">{date_label}</text>'
        )

    # The legend sits in the title row rather than inside the plot: a flat
    # benchmark line runs along the top of the panel and an in-plot legend
    # lands straight on top of it.
    label = escape(company if len(company) <= 22 else company[:21] + "…")
    offset = 150 + len(label) * 6.2
    parts.append(
        f'<g class="legend" transform="translate({WIDTH - PAD_RIGHT - offset:.0f},'
        f'{price_y - 12})">'
        f'<line x1="0" y1="0" x2="22" y2="0" class="line-company"/>'
        f'<text x="28" y="4" class="tick">{label}</text>'
        f'<line x1="{40 + len(label) * 6.2:.0f}" y1="0" '
        f'x2="{62 + len(label) * 6.2:.0f}" y2="0" class="line-benchmark"/>'
        f'<text x="{68 + len(label) * 6.2:.0f}" y="4" class="tick">'
        f'{escape(benchmark)}</text></g>'
    )

    # ---- Panel 2: abnormal returns ---------------------------------------
    abnormal = [float(v) * 100 for v in daily["abnormal_return"].fillna(0)]
    bound = max((abs(v) for v in abnormal), default=1.0) * 1.2 or 1.0

    def abn_y_of(v: float) -> float:
        return abn_y + abn_h / 2 - (v / bound) * (abn_h / 2)

    parts.append(_panel_frame(abn_y, abn_h,
                              "Abnormal return — company move with the market's move removed (%)"))
    parts.append(_y_axis(abn_y, abn_h, -bound, bound, "{:+.1f}%"))

    band = (WIDTH - PAD_LEFT - PAD_RIGHT) / n
    bar_w = max(2.0, min(band * 0.62, 26.0))
    zero = abn_y_of(0)
    for x, value, day in zip(xs, abnormal, dates):
        y = abn_y_of(value)
        top, height = min(y, zero), abs(y - zero)
        css = "bar-neg" if value < 0 else "bar-pos"
        flag = " bar-incident" if day in incident_days else ""
        tooltip = (incident_tooltip.get(day) or f"{day:%d %b %Y}: {value:+.2f}%")
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(height, 0.8):.1f}" class="{css}{flag}">'
            f"<title>{escape(tooltip)}</title></rect>"
        )
    parts.append(
        f'<line x1="{PAD_LEFT}" y1="{zero:.1f}" x2="{WIDTH - PAD_RIGHT}" '
        f'y2="{zero:.1f}" class="axis-zero"/>'
    )

    # ---- Panel 3: coverage volume, coloured by tone ----------------------
    counts = [int(v) for v in daily["unique_count"].fillna(0)]
    tones = [float(v) for v in daily["weighted_sentiment"].fillna(0)]
    max_count = max(counts) or 1

    parts.append(_panel_frame(cov_y, cov_h,
                             "News volume (bar height) and tone (colour)"))
    parts.append(_y_axis(cov_y, cov_h, 0, max_count))

    for x, count, tone, day in zip(xs, counts, tones, dates):
        if count <= 0:
            continue
        height = count / max_count * cov_h
        css = ("tone-neg" if tone <= -0.15 else
               "tone-pos" if tone >= 0.15 else "tone-neutral")
        bar_y = cov_y + cov_h - height
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{bar_y:.1f}" '
            f'width="{bar_w:.1f}" height="{height:.1f}" class="{css}">'
            f'<title>{day:%d %b %Y}: {count} item(s), tone {tone:+.2f}</title></rect>'
        )
        if css == "tone-neg":
            parts.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" '
                f'height="{height:.1f}" fill="url(#neg-hatch)" pointer-events="none"/>'
            )

    parts.append(
        f'<g transform="translate(0,{cov_y + cov_h + 16:.1f})">'
        f'{_date_labels(dates, xs)}</g>'
    )
    parts.append("</svg>")
    return "".join(parts)


def price_level_svg(series: pd.DataFrame, real_dates: set[date],
                    company: str, moves: list | None = None) -> str:
    """A single panel: an unlisted share's indicative price level over time.

    ``series`` is the as-displayed daily frame (forward-fill included) -
    the same convention UnlistedZone's own chart uses, so the line matches
    what a reader would see there. ``real_dates`` marks which of those days
    were genuine revisions with a small dot; everywhere else the line is
    held flat, not observed. ``moves`` (optional, ranked) gets the same
    numbered-badge-with-dated-label treatment ``timeline_svg`` uses for
    incidents, so a reader can tell which ranked price-move gap is which.
    """
    if series.empty:
        return '<p class="empty">No price data in the analysis window.</p>'

    dates = [pd.Timestamp(d).date() for d in series.index]
    n = len(dates)
    xs = _x_positions(n, WIDTH)
    closes = series["close"].astype(float).tolist()

    move_rank: dict[date, int] = {}
    move_tooltip: dict[date, str] = {}
    for rank, move in enumerate(moves or [], 1):
        move_rank[move.end_date] = rank
        move_tooltip[move.end_date] = (
            f"#{rank} {move.start_date:%d %b %Y} to {move.end_date:%d %b %Y}: "
            f"{move.change * 100:+.2f}% — see the ranked list below"
        )

    has_badges = bool(moves)
    top_caption_h = 18.0 if has_badges else 0.0
    price_h = 260.0
    price_y = PAD_TOP + top_caption_h
    total_h = price_y + price_h + LABEL_BAND + 14

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {total_h:.0f}" class="timeline" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Indicative price timeline">'
    ]
    if has_badges:
        parts.append(
            f'<text x="{PAD_LEFT}" y="{PAD_TOP - 4:.1f}" class="chart-caption">'
            "Numbered circles mark the ranked price moves below, each dated "
            "underneath. Dots are real revisions — elsewhere the line is flat, "
            "not observed.</text>"
        )

    low, high = min(closes), max(closes)
    span = (high - low) or 1.0
    low, high = low - span * 0.08, high + span * 0.08

    def price_y_of(v: float) -> float:
        return price_y + price_h - (v - low) / (high - low) * price_h

    parts.append(_panel_frame(price_y, price_h, f"{company} — indicative price (₹)"))
    parts.append(_y_axis(price_y, price_h, low, high, "{:,.0f}"))

    points = " ".join(f"{x:.1f},{price_y_of(v):.1f}" for x, v in zip(xs, closes))
    parts.append(f'<polyline points="{points}" class="line-company"/>')

    badge_r = 9.0
    badge_cy = price_y + 15.0
    for x, day, value in zip(xs, dates, closes):
        if day in real_dates:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{price_y_of(value):.1f}" r="2.5" '
                f'class="real-point"><title>{day:%d %b %Y}: ₹{value:,.2f} '
                "(revised)</title></circle>"
            )
        rank = move_rank.get(day)
        if rank is None:
            continue
        parts.append(
            f'<line x1="{x:.1f}" y1="{price_y:.1f}" x2="{x:.1f}" '
            f'y2="{price_y + price_h:.1f}" class="incident-rule"/>'
        )
        tooltip = escape(move_tooltip.get(day, f"{day:%d %b %Y}: price move"))
        parts.append(
            f'<g class="incident-badge">'
            f'<circle cx="{x:.1f}" cy="{badge_cy:.1f}" r="{badge_r}"/>'
            f'<text x="{x:.1f}" y="{badge_cy + 3.5:.1f}" text-anchor="middle">{rank}</text>'
            f"<title>{tooltip}</title></g>"
        )
        date_label = f"{day:%d %b}"
        label_y = badge_cy + badge_r + 11.0
        label_w = len(date_label) * 5.4 + 6.0
        parts.append(
            f'<rect x="{x - label_w / 2:.1f}" y="{label_y - 9.0:.1f}" '
            f'width="{label_w:.1f}" height="12" rx="2" class="badge-date-bg"/>'
            f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'class="badge-date">{date_label}</text>'
        )

    parts.append(
        f'<g transform="translate(0,{price_y + price_h + 16:.1f})">'
        f'{_date_labels(dates, xs)}</g>'
    )
    parts.append("</svg>")
    return "".join(parts)


def news_coverage_svg(series: pd.DataFrame, items: list, company: str) -> str:
    """A single panel: collected news volume (bar height) and tone (colour),
    one bar per calendar day, over the same day axis ``price_level_svg``
    plots for the same window - so the two panels line up and a reader can
    see what was published against the price line directly above it.

    Deliberately not part of ``price_level_svg`` itself and not gated on any
    price statistic: unlike an abnormal-return panel, a day's news count and
    tone need nothing from the price series to be honestly described, so
    this exists for unlisted reports even though a market-model abnormal
    return does not (see ``ceia/unlisted.py``'s module docstring).
    """
    if series.empty:
        return '<p class="empty">No price data in the analysis window.</p>'

    dates = [pd.Timestamp(d).date() for d in series.index]
    n = len(dates)
    xs = _x_positions(n, WIDTH)

    counts = {d: 0 for d in dates}
    tone_sum = {d: 0.0 for d in dates}
    for item in items:
        if item.published_at is None:
            continue
        day = item.published_at.date()
        if day not in counts:
            continue
        counts[day] += 1
        tone_sum[day] += item.sentiment_score

    cov_h = 130.0
    cov_y = PAD_TOP
    total_h = cov_y + cov_h + LABEL_BAND + 14
    max_count = max(counts.values(), default=0) or 1

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {total_h:.0f}" class="timeline" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="News coverage timeline">'
        '<defs><pattern id="neg-hatch-news" width="6" height="6" '
        'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
        '<line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>'
    ]
    parts.append(_panel_frame(
        cov_y, cov_h, f"{company} — news volume (bar height) and tone (colour)"))
    parts.append(_y_axis(cov_y, cov_h, 0, max_count))

    band = (WIDTH - PAD_LEFT - PAD_RIGHT) / n
    bar_w = max(2.0, min(band * 0.62, 26.0))
    for x, day in zip(xs, dates):
        count = counts[day]
        if count <= 0:
            continue
        tone = tone_sum[day] / count
        height = count / max_count * cov_h
        css = ("tone-neg" if tone <= -0.15 else
               "tone-pos" if tone >= 0.15 else "tone-neutral")
        bar_y = cov_y + cov_h - height
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{bar_y:.1f}" '
            f'width="{bar_w:.1f}" height="{height:.1f}" class="{css}">'
            f'<title>{day:%d %b %Y}: {count} item(s), tone {tone:+.2f}</title></rect>'
        )
        if css == "tone-neg":
            parts.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" '
                f'height="{height:.1f}" fill="url(#neg-hatch-news)" pointer-events="none"/>'
            )

    parts.append(
        f'<g transform="translate(0,{cov_y + cov_h + 16:.1f})">'
        f'{_date_labels(dates, xs)}</g>'
    )
    parts.append("</svg>")
    return "".join(parts)
