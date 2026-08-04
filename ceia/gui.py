"""Local web GUI for the Company Event Impact Analyzer.

Run with:

    streamlit run ceia/gui.py

This is a thin presentation layer over the existing pipeline — every number it
shows comes from the same ``ceia.ingest.run`` / ``ceia.analyze.analyse`` /
``ceia.report.build_html`` calls the CLI uses. No analysis logic lives here;
this file only builds a form, calls those functions, and renders the result.

Two data-source modes, since a live run takes minutes (rate-limited scraping,
model downloads) and not every session needs that:

* **Bundled corpus** — the 137-item, real, committed Adani corpus
  (``data/adani_wide_2023.json``) with sentiment and emotion already scored.
  Runs the event-study/report stage only; seconds, no network.
* **Scrape live** — the full pipeline for any company/ticker/date range,
  identical to ``python -m ceia.ingest`` followed by ``ceia.analyze``.

Long-running steps stream their progress into the page via a small logging
handler (``_StreamlitLogHandler``) rather than leaving the user watching a
blank spinner for minutes.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run ceia/gui.py` executes this file as a standalone script, not
# as part of the ceia package, so relative imports fail with "attempted
# relative import with no known parent package" - confirmed by actually
# running it, not just importing it. Absolute imports plus this bootstrap
# work the same way whether ceia is pip-installed or run straight from a
# checkout, matching how `python -m ceia.ingest` needs no install either.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ceia import ingest  # noqa: E402
from ceia.analyze import Analysis, analyse, load_news_from_file  # noqa: E402
from ceia.fetcher import DEFAULT_USER_AGENT, Fetcher  # noqa: E402
from ceia.models import RunConfig  # noqa: E402
from ceia.prices import (  # noqa: E402
    AlphaVantageProvider,
    CsvProvider,
    PriceError,
    YahooChartProvider,
    YFinanceProvider,
)
from ceia.report import build_html  # noqa: E402
from ceia.sentiment import FinBertScorer  # noqa: E402

BUNDLED_CORPUS = Path(__file__).resolve().parent.parent / "data" / "adani_wide_2023.json"

st.set_page_config(page_title="Company Event Impact Analyzer", layout="wide")


# --------------------------------------------------------------- log streaming

class _StreamlitLogHandler(logging.Handler):
    """Appends each log record to a placeholder, so a multi-minute scrape
    shows live progress instead of a spinner with no detail underneath."""

    def __init__(self, placeholder) -> None:
        super().__init__()
        self.placeholder = placeholder
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        # Cap what is rendered; a live scrape can log hundreds of lines and
        # redrawing all of them every time gets slow well before that.
        self.placeholder.code("\n".join(self.lines[-60:]))


def _run_with_live_log(fn, *args, **kwargs):
    placeholder = st.empty()
    handler = _StreamlitLogHandler(placeholder)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("ceia")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        return fn(*args, **kwargs)
    finally:
        logger.removeHandler(handler)


# ----------------------------------------------------------- cached resources

@st.cache_resource(show_spinner="Loading FinBERT (first run downloads ~440MB)…")
def _finbert_scorer() -> FinBertScorer:
    return FinBertScorer()


@st.cache_resource(show_spinner="Loading GoEmotions (first run downloads ~500MB)…")
def _emotion_scorer():
    from ceia.emotion import GoEmotionScorer
    return GoEmotionScorer()


# --------------------------------------------------------------------- helpers

def _price_providers(ticker: str, benchmark: str, api_key: str,
                     ticker_csv, benchmark_csv) -> list | None:
    """Build the provider chain from what the form actually supplied.

    Uploaded CSVs take priority — a user who went to the trouble of exporting
    their own price data almost certainly wants it used, not silently
    overridden by whichever network provider happens to answer first.
    """
    if ticker_csv is not None and benchmark_csv is not None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ceia_prices_"))
        (tmp_dir / f"{ticker.replace('^', '_idx_')}.csv").write_bytes(ticker_csv.getvalue())
        (tmp_dir / f"{benchmark.replace('^', '_idx_')}.csv").write_bytes(
            benchmark_csv.getvalue())
        return [CsvProvider(tmp_dir)]

    providers = [YFinanceProvider(), YahooChartProvider()]
    if api_key:
        providers.append(AlphaVantageProvider(api_key=api_key))
    providers.append(CsvProvider())  # data/prices/, in case it exists locally
    return providers


def _run_analysis(config: RunConfig, source_mode: str, limit: int | None,
                  skip_sentiment: bool, skip_emotion: bool,
                  api_key: str, ticker_csv, benchmark_csv, workers: int = 8) -> Analysis:
    if source_mode == "Bundled Adani corpus (instant, no scraping)":
        if not BUNDLED_CORPUS.exists():
            raise FileNotFoundError(f"{BUNDLED_CORPUS} is missing from this checkout")
        items, news_meta = load_news_from_file(BUNDLED_CORPUS)
        st.info(f"Loaded {len(items)} items from the bundled corpus "
               "(headlines/timestamps real; article bodies are not "
               "redistributed — see the README).")
    else:
        fetcher = Fetcher(cache_dir="cache", user_agent=DEFAULT_USER_AGENT)
        scorer = None if skip_sentiment else _finbert_scorer()
        emotion_scorer = None if skip_emotion else _emotion_scorer()
        result = _run_with_live_log(
            ingest.run, config, fetcher=fetcher, scorer=scorer,
            emotion_scorer=emotion_scorer, limit=limit,
            skip_sentiment=skip_sentiment, skip_emotion=skip_emotion,
            max_workers=workers,
        )
        items = result.items
        news_meta = result.to_dict()
        news_meta.pop("items", None)
        for key, why in result.disabled_sources.items():
            st.caption(f"Source disabled: {key} — {why}")
        for key, state in result.source_status.items():
            if state.startswith("failed") or "0 " in state:
                st.warning(f"{key}: {state}")

        # A zero (or low) final count could die at any of several stages -
        # discovery, the slug pre-filter, fetch/parse, the date window, or
        # the relevance filter - and without this breakdown that's a black
        # box. Shown every run, not just on failure, since "why is this
        # number lower than I expected" matters on a partial success too.
        with st.expander("Ingestion funnel (why the final count is what it is)"):
            funnel = pd.Series(result.stats, name="count").to_frame()
            st.dataframe(funnel, use_container_width=True)
            if result.stats.get("candidates_discovered", 0) == 0:
                st.caption("Zero candidates discovered - check the per-source "
                          "status above; if every source shows 'ok: 0 URLs "
                          "returned' rather than 'failed', the date range "
                          "genuinely has nothing in these sitemaps yet.")
            elif result.stats.get("candidates_after_prefilter", 0) == 0:
                st.caption("Candidates were found but none survived the slug "
                          "pre-filter - the company name/aliases/ticker may "
                          "not match how this company's URLs are worded. Try "
                          "adding an alias.")

    providers = _price_providers(config.ticker, config.benchmark, api_key,
                                 ticker_csv, benchmark_csv)
    return _run_with_live_log(analyse, config, items, news_meta, providers=providers)


# ------------------------------------------------------------------------ UI

st.title("Company Event Impact Analyzer")
st.caption(
    "Lines up Indian financial-press coverage against benchmark-adjusted stock "
    "returns. A structured case study generator, not a trading signal and not "
    "proof of causation — see the full caveats in the generated report below."
)

with st.form("run_config"):
    col1, col2 = st.columns(2)
    with col1:
        source_mode = st.radio(
            "News data",
            ["Bundled Adani corpus (instant, no scraping)", "Scrape live"],
            help="Live scraping respects each source's robots.txt and rate "
                 "limits, so it takes minutes, not seconds.",
        )
        company = st.text_input("Company", value="Adani Enterprises", key="company")
        ticker_col, exchange_col = st.columns([3, 1])
        with ticker_col:
            ticker = st.text_input(
                "Ticker (Yahoo-style, e.g. ADANIENT.NS)", value="", key="ticker",
                help="Leave blank to auto-detect from the company name above.",
            )
        with exchange_col:
            exchange = st.selectbox("Exchange", ["NSE", "BSE"],
                                    help="Used to pick between .NS/.BO when "
                                         "auto-detecting the ticker.")
        benchmark = st.text_input("Benchmark", value="^NSEI", key="benchmark")
        aliases_raw = st.text_input(
            "Aliases (comma-separated, optional)", value="",
            help="Short names, product names, misspellings — anything the "
                 "press might use besides the company's full legal name. "
                 "The company name itself is always included.",
        )
    with col2:
        default_start = date(2023, 1, 20)
        default_end = date(2023, 2, 17)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=default_end)
        window_before, window_after = st.columns(2)
        with window_before:
            event_before = st.number_input("Event window: days before", value=-1, step=1)
        with window_after:
            event_after = st.number_input("Event window: days after", value=3, step=1)

    with st.expander("Price source"):
        api_key = st.text_input(
            "Alpha Vantage API key (optional)", type="password",
            help="Only used if yfinance and the Yahoo chart API are both "
                 "unreachable. Free key: alphavantage.co/support/#api-key",
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            ticker_csv = st.file_uploader("Ticker price CSV (date,close)", type="csv")
        with pc2:
            benchmark_csv = st.file_uploader("Benchmark price CSV (date,close)", type="csv")
        st.caption("Providers are tried in order: yfinance → Yahoo chart API → "
                  "Alpha Vantage → uploaded CSV / local CSV. Uploading both "
                  "CSVs above skips straight to them.")

    with st.expander("Advanced"):
        min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.35, 0.05)
        coverage_z = st.number_input("Coverage z-score threshold", value=1.0, step=0.1)
        return_z = st.number_input("Abnormal-return z-score threshold", value=1.5, step=0.1)
        limit = st.number_input(
            "Fetch limit (live scraping only; 0 = no cap)", value=200, step=50,
            help="Caps articles fetched, for a quick trial run. Evenly spread "
                 "across the whole date range rather than just its earliest "
                 "days, so a low cap thins out coverage everywhere instead of "
                 "silently truncating the end of the window.",
        )
        workers = st.number_input(
            "Concurrent fetches (live scraping only)", value=8, min_value=1,
            max_value=32, step=1,
            help="How many articles to fetch at once. Discovery across the 4 "
                 "sources always runs concurrently, one per source. Requests "
                 "to any single site are still rate-limited exactly as "
                 "before regardless of this setting - it only lets DIFFERENT "
                 "sites' fetches overlap instead of queueing behind each "
                 "other.",
        )
        skip_sentiment = st.checkbox("Skip FinBERT sentiment (no model download)")
        skip_emotion = st.checkbox("Skip GoEmotions (no model download)")

    submitted = st.form_submit_button("Run analysis", type="primary")

if submitted:
    resolved_ticker = ticker.strip()
    if not resolved_ticker:
        from ceia.ticker_lookup import TickerLookupError, resolve_ticker
        try:
            with st.spinner(f"Looking up a ticker for {company!r}…"):
                match = resolve_ticker(company, exchange=exchange)
        except TickerLookupError as exc:
            st.error(f"Ticker lookup failed: {exc}\n\nType the ticker in directly instead.")
            st.stop()
        resolved_ticker = match.symbol
        st.info(f"Resolved ticker: {company!r} → **{resolved_ticker}** ({match.name})"
               + ("" if match.exact_exchange_match
                  else f" — not listed on {exchange}, using the nearest match"))

    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
    try:
        config = RunConfig(
            company=company, ticker=resolved_ticker, benchmark=benchmark, exchange=exchange,
            start=start, end=end, aliases=aliases,
            event_window=(int(event_before), int(event_after)),
            min_relevance=min_relevance,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    try:
        with st.spinner("Running…"):
            analysis = _run_analysis(
                config, source_mode, limit=int(limit) or None,
                skip_sentiment=skip_sentiment, skip_emotion=skip_emotion,
                api_key=api_key, ticker_csv=ticker_csv, benchmark_csv=benchmark_csv,
                workers=int(workers),
            )
    except PriceError as exc:
        st.error(
            f"Price data unavailable: {exc}\n\n"
            "The news pipeline is unaffected — try an Alpha Vantage key or "
            "upload CSVs under 'Price source' above."
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001 — surface it in the page, not a traceback
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    st.success("Done.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trading days", len(analysis.daily))
    m2.metric("News items",
             analysis.news_meta.get("stats", {}).get("unique_after_dedupe", "—"))
    m3.metric("Candidate incidents", len(analysis.incidents))
    m4.metric("Return model", analysis.price_meta.get("model", "—"))

    if analysis.incidents:
        top = analysis.incidents[0]
        st.subheader(f"Top candidate: {top.day:%d %B %Y}")
        st.write(
            f"Abnormal return **{top.abnormal_return * 100:+.2f}%** "
            f"(z = {top.abnormal_return_z:+.1f}), {top.item_count} item(s), "
            f"tone {top.mean_sentiment:+.2f} — "
            f"**{'consistent with' if top.direction_agrees else 'opposite to'}** "
            "the price move."
        )
        if top.volume and pd.notna(top.volume):
            volume_z_part = (f" (z = {top.volume_z:+.2f})"
                             if top.volume_z is not None and pd.notna(top.volume_z) else "")
            st.caption(
                f"Volume: {top.volume:,.0f}{volume_z_part} — a corroborating "
                "signal, not part of the flagging test."
            )

    corr = analysis.correlation
    if corr.get("r") is not None:
        st.caption(f"Sentiment/return correlation: r = **{corr['r']:+.3f}** "
                  f"(R² = {corr['r_squared']:.3f}, n = {corr['n']}) — {corr['note']}")
    elif corr.get("note"):
        st.caption(f"Sentiment/return correlation not computed — {corr['note']}")

    html_report = build_html(analysis)
    st.download_button("Download report.html", data=html_report,
                       file_name="report.html", mime="text/html")
    st.download_button("Download analysis.json",
                       data=json.dumps(analysis.to_dict(), indent=2, default=str),
                       file_name="analysis.json", mime="application/json")

    st.subheader("Full report")
    st.components.v1.html(html_report, height=1400, scrolling=True)
