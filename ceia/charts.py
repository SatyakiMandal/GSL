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
                 company: str, benchmark: str) -> str:
    """Three stacked panels: rebased prices, abnormal returns, coverage.

    Prices are rebased to 100 at the window's first session so a stock priced in
    thousands and an index priced in tens can share an axis and be compared on
    percentage terms, which is what the eye should be reading here.
    """
    if daily.empty:
        return '<p class="empty">No trading days in the analysis window.</p>'

    # pandas Timestamp subclasses datetime.date, so an isinstance check would
    # leave Timestamps untouched and they never compare equal to the plain
    # dates in ``incident_days`` — silently dropping every incident marker.
    # Normalise unconditionally instead.
    dates = [pd.Timestamp(d).date() for d in daily.index]
    n = len(dates)
    xs = _x_positions(n, WIDTH)

    price_h, abn_h, cov_h = 190.0, 130.0, 110.0
    gap = 46.0
    price_y = PAD_TOP
    abn_y = price_y + price_h + gap
    cov_y = abn_y + abn_h + gap
    total_h = cov_y + cov_h + LABEL_BAND + 14

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {total_h:.0f}" class="timeline" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Price, abnormal return and coverage timeline">'
    ]

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

    for values, css in ((bench_idx, "line-benchmark"), (company_idx, "line-company")):
        points = " ".join(f"{x:.1f},{price_y_of(v):.1f}" for x, v in zip(xs, values))
        parts.append(f'<polyline points="{points}" class="{css}"/>')

    for x, day in zip(xs, dates):
        if day in incident_days:
            parts.append(
                f'<line x1="{x:.1f}" y1="{price_y:.1f}" x2="{x:.1f}" '
                f'y2="{price_y + price_h:.1f}" class="incident-rule"/>'
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
    parts.append(_y_axis(abn_y, abn_h, -bound, bound, "{:+.1f}"))

    band = (WIDTH - PAD_LEFT - PAD_RIGHT) / n
    bar_w = max(2.0, min(band * 0.62, 26.0))
    zero = abn_y_of(0)
    for x, value, day in zip(xs, abnormal, dates):
        y = abn_y_of(value)
        top, height = min(y, zero), abs(y - zero)
        css = "bar-neg" if value < 0 else "bar-pos"
        flag = " bar-incident" if day in incident_days else ""
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(height, 0.8):.1f}" class="{css}{flag}">'
            f'<title>{day:%d %b %Y}: {value:+.2f}%</title></rect>'
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
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{cov_y + cov_h - height:.1f}" '
            f'width="{bar_w:.1f}" height="{height:.1f}" class="{css}">'
            f'<title>{day:%d %b %Y}: {count} item(s), tone {tone:+.2f}</title></rect>'
        )

    parts.append(
        f'<g transform="translate(0,{cov_y + cov_h + 16:.1f})">'
        f'{_date_labels(dates, xs)}</g>'
    )
    parts.append("</svg>")
    return "".join(parts)
