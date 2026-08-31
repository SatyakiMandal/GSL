# Company Event Impact Analyzer (CEIA)
## Institutional Architecture, Methodologies, Dual-Market Coverage, Use Cases & Evolutionary Version Timeline

---

## 1. Executive Overview & System Identity

The **Company Event Impact Analyzer (CEIA)** is an enterprise-grade econometric, natural language processing (NLP), and market microstructure intelligence platform tailored specifically for Indian capital markets across both **publicly listed equities (NSE/BSE)** and the **unlisted / pre-IPO private equity space**.

It was engineered to address a fundamental challenge in quantitative finance and equity research: **separating genuine corporate event impact from systematic market noise, sector-wide momentum, and media sensationalism**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CEIA END-TO-END PIPELINE                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
   [1. Ingestion & Dual-Market Scraper Infrastructure]
   ├─ Mainstream Financial Press: Moneycontrol, Livemint, Economic Times, Financial Express,
   │                              Business Line, Business Today, Reuters India
   ├─ Unlisted / Pre-IPO Outlets: Entrackr, VCCircle, Inc42 (Startup/PE/VC Rounds & ROC filings)
   ├─ Grey Market & Unlisted Feeds: UnlistedZone Next.js React Server streaming price timelines
   ├─ Statutory Disclosures: BSE Corporate Announcements & SEBI LODR Regulation 30 Filings
   ├─ Corporate Document Parser: PDF Annual Report & Earnings Call Transcript Extractor
   └─ Resilient Fallbacks: Wayback Machine Web Archive historical topic-page snapshots
           │
           ▼
   [2. Econometric & Asset Pricing Engine]
   ├─ Multi-Factor Asset Pricing Models (Fama-French 4-Factor, Nifty 50, SMB, HML, WML Momentum)
   ├─ Asymmetric Downside Beta (β⁻), Systematic Coskewness & Cokurtosis (Harvey-Siddique)
   ├─ Copula Tail Dependence Matrix (Clayton Crash vs Gumbel Rally Dependence)
   └─ Cross-Sectional Multi-Company Peer Cohort Benchmark (Resiliency Ranking)
           │
           ▼
   [3. Event Study & Causal Inference]
   ├─ CUSUM Structural Break Calibration & Decoupled Overlap-Free CAR Cones
   ├─ Synthetic Control Method (SCM - Abadie et al.) Peer Counterfactual Portfolio
   ├─ Microstructure Dynamics (Roll Spread, Kyle's Lambda, VPIN Toxicity, Amihud Illiquidity)
   ├─ Almgren-Chriss (2000) Optimal Execution Trajectory & Slippage Budgeting
   └─ Econometric Tests (BMP Standardized Test, Corrado Rank Test, Jump-Diffusion Decomposition)
           │
           ▼
   [4. NLP & Forensic Linguistic Analysis]
   ├─ Dual Transformer Engine: FinBERT Financial Sentiment + GoEmotions 28-Emotion Radar
   ├─ SEBI LODR Regulation 30 Statutory Urgency Classifier (Tier 1 / 2 / 3)
   ├─ Corporate PDF Transcript Parser: Gunning Fog Complexity, Q&A Evasion Scorer
   └─ Shannon Narrative Polarization Entropy & Media Echo Multiplication Tracker
           │
           ▼
   [5. Fundamental Forensics & Valuation Triad]
   ├─ 2-Stage Discounted Cash Flow (DCF) Valuation & 5x5 WACC vs Growth Sensitivity Grid
   ├─ Dupont 5-Factor ROE Breakdown (Tax x Interest x EBIT Margin x Turnover x Leverage)
   ├─ Emerging Market Altman Z"-Score (Solvency Zones) & Beneish M-Score (Manipulation)
   ├─ 9-Point Piotroski F-Score Fundamental Health & Post-Earnings Announcement Drift (PEAD)
   └─ 4-Pillar Governance Risk Index (GRI) & Merton Distance-to-Default (DD)
           │
           ▼
   [6. Multi-Channel Output Ecosystem]
   ├─ 11-Tab Dynamic Institutional Excel Model Suite with Active Formulas (Excel/*.xlsx)
   ├─ Interactive Standalone Zero-Dependency HTML Dashboard with Full Valuation Panels (out/*.html)
   ├─ Peer Cohort Benchmark HTML Reports (out/peer_benchmark.html)
   └─ Structured Machine-Readable JSON Payloads (out/*.json)
```

---

## 2. Dual-Market Data Source Architecture

CEIA incorporates a **dual-market scraping and ingestion infrastructure** that spans listed equities, venture capital/private equity startup news, unlisted share dealers, and statutory exchanges:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             CEIA DUAL-MARKET INGESTION INFRASTRUCTURE                            │
├───────────────────────────────────┬──────────────────────────────────────────────────────────────┤
│ Source Category                   │ Data Sources, Sitemaps & Scraping Mechanisms                 │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 1. Mainstream Listed Press        │ • Economic Times (Month-partitioned XML news sitemaps)       │
│    (Public Markets)               │ • Financial Express (Day-partitioned dated sitemaps)         │
│                                   │ • The Hindu Business Line (Archive XML sitemaps back to 2010)│
│                                   │ • Moneycontrol (Year/month index sitemaps)                   │
│                                   │ • Business Today (1,000-day deep date-wise story sitemaps)   │
│                                   │ • Livemint (Rolling sitemaps + Wayback topic discovery)      │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 2. Unlisted, Startup & Pre-IPO    │ • Entrackr (Day-partitioned sitemaps back to 2017: funding   │
│    Private Markets (ceia.unlisted)│   rounds, ROC financial filings, startup revenue/burn)       │
│                                   │ • VCCircle (Numbered reverse-chronological sitemaps back to  │
│                                   │   2008: PE/VC deals, secondary buyouts, cap tables)          │
│                                   │ • Inc42 (WordPress Yoast post sitemaps: tech startups,       │
│                                   │   pre-IPO bulk deals, regulatory moves)                      │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 3. Unlisted Price & Grey Market   │ • UnlistedZone (Scrapes Next.js streaming RSC JSON payloads  │
│    Platforms                      │   extracting 5-year indicative price revision timelines)     │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 4. Statutory & Regulatory Feeds   │ • BSE Corporate Announcements (Official exchange filings)    │
│                                   │ • SEBI Regulatory Notices (Show-cause, probes, penalties)    │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 5. Corporate Document Parser      │ • Local PDF Extractor (Annual reports, investor presentations,│
│    (ceia.pdf_extract)             │   and earnings conference call transcripts)                  │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 6. Historical Archive Fallback    │ • Wayback Machine (Internet Archive CDX API for snapshots    │
│                                   │   of Business Standard, Livemint, and archived topic pages)  │
└───────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 3. Dedicated Unlisted & Pre-IPO Analytics Engine (`ceia.unlisted`)

Pre-IPO and unlisted shares exhibit completely different market mechanics compared to publicly traded stocks. CEIA features a dedicated module (`ceia/unlisted.py`) built from first principles for private equity:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             UNLISTED & PRE-IPO ANALYTICS ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

    [UnlistedZone Pricing Engine]                      [Private Equity NLP Ingestion]
  (https://unlistedzone.com/shares)                  (Entrackr, VCCircle, Inc42 & Press)
                 │                                                   │
                 ▼                                                   ▼
   [Streaming RSC JSON Parser]                         [Multi-Source News Pipeline]
   (Extracts raw historical dates)                    (Funding rounds, valuation shifts)
                 │                                                   │
                 ▼                                                   ▼
   [Step-Function Collapse Engine]                     [FinBERT & GoEmotions Scorers]
   (Drops consecutive flat quotes)                     (Sentiment & emotion radar)
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           ▼
                       [Discrete Revision Move Interval]
                        Move Window: (t_prev, t_curr]
                        Delta P = (P_curr - P_prev) / P_prev
                                           │
                                           ▼
                       [Honest Unlisted Timeline Report]
                      (out/unlisted_report.html & JSON)
```

---

## 4. Advanced Institutional Valuation, Factor & Execution Engines

### A. Corporate Valuation & 2-Stage DCF Model (`ceia/valuation_model.py`)
* Solves **Cost of Equity** ($K_e = R_f + \beta \times \text{ERP}$) and **WACC**:
  $$\text{WACC} = \frac{E}{D+E} K_e + \frac{D}{D+E} K_d (1 - t)$$
* Computes **2-Stage DCF Intrinsic Value**:
  $$\text{Enterprise Value} = \sum_{t=1}^5 \frac{\text{FCFF}_t}{(1 + \text{WACC})^t} + \frac{\text{TV}_5}{(1 + \text{WACC})^5}$$
* Generates a dynamic **5x5 Sensitivity Matrix** across WACC ($\pm 1.5\%$) and terminal growth rate ($g \in [3.0\%, 6.0\%]$).
* Performs **Dupont 5-Factor ROE Decomposition**:
  $$\text{ROE} = \left(\frac{\text{Net Income}}{\text{EBT}}\right) \times \left(\frac{\text{EBT}}{\text{EBIT}}\right) \times \left(\frac{\text{EBIT}}{\text{Sales}}\right) \times \left(\frac{\text{Sales}}{\text{Assets}}\right) \times \left(\frac{\text{Assets}}{\text{Equity}}\right)$$

### B. Multi-Factor Pricing & Factor Risk Attribution (`ceia/factor_model.py`)
* Fits the **Fama-French & Carhart 4-Factor Model** for Indian Equities:
  $$R_{i, t} - R_{f, t} = \alpha + \beta_M (R_{M, t} - R_{f, t}) + \beta_{SMB} SMB_t + \beta_{HML} HML_t + \beta_{WML} WML_t + \epsilon_t$$
* Separates systematic Factor Beta drift from pure company idiosyncratic Alpha ($\alpha_{FF}$).

### C. Almgren-Chriss (2000) Algorithmic Execution Simulator (`ceia/execution_simulator.py`)
* Solves optimal liquidation trajectories balancing market impact vs timing risk.
* Calculates basis-point trade slippage for institutional orders and max position size for a 25 bps slippage budget.

### D. Synthetic Control Method (SCM - Abadie et al.) (`ceia/synthetic_control.py`)
* Solves constrained quadratic optimization over an untreated donor pool of industry peer stocks:
  $$\min_{\mathbf{w}} \sum_{t=1}^{T_0} \left( Y_{0, t} - \sum_{j=1}^J w_j Y_{j, t} \right)^2 \quad \text{s.t.} \quad w_j \ge 0, \sum_{j=1}^J w_j = 1$$
* Measures unconfounded treatment effects $\tau_t = Y_{0, t} - \hat{Y}_{0, t}$, synthetic cumulative abnormal returns ($\text{SCAR}$), and in-space placebo permutation $p$-values.

---

## 5. Output Ecosystem & Dynamic Excel Model Suite

Every CEIA run automatically generates deliverables routed into structured output folders:

### 1. 11-Tab Dynamic Institutional Excel Model (`Excel/<ticker>_model_report.xlsx`)
* **Tab 1: `Executive Summary`**: Core company metadata, market beta, WACC, DCF target price, Dupont ROE, Fama-French pure alpha, and solvency status.
* **Tab 2: `Daily Detail`**: Full daily sequence with returns, benchmark returns, abnormal returns, and **LIVE Excel Formulas in the footer**:
  * `=AVERAGE(...)` for Mean Daily Return
  * `=STDEV.S(...)` for Realized Daily Volatility
  * `=SLOPE(...)` for Live Market Beta
  * `=INTERCEPT(...)` for Live Daily Alpha
  * `=CORREL(...)` for Benchmark Comovement
  * `=-PERCENTILE.INC(..., 0.05)` for 95% Historical VaR
* **Tab 3: `Candidate Incidents`**: Ranked event days, CAR statistics, SEBI LODR tiers, forward guidance ratings, and price absorption half-life.
* **Tab 4: `Event CAR Attribution`**: Category-level excess return attribution waterfall breakdown.
* **Tab 5: `DCF Valuation & WACC`**: Full 2-stage DCF intrinsic valuation schedule, WACC parameters, and 5x5 WACC vs Growth sensitivity grid.
* **Tab 6: `Dupont 5-Factor ROE`**: 5-stage ratio breakdown (Tax, Interest, Margin, Asset Turnover, Leverage).
* **Tab 7: `Factor Attribution`**: Fama-French multi-factor beta breakdown, factor $t$-statistics, and idiosyncratic risk.
* **Tab 8: `VaR Term Structure`**: Parametric, Historical, and Cornish-Fisher CVaR surface across $1D, 5D, 10D, 21D$ horizons.
* **Tab 9: `Governance & Forensics`**: GRI 4-pillar scores, Altman Z"-Score, Beneish M-Score, and Piotroski F-Score.
* **Tab 10: `Microstructure & Slippage`**: Roll effective spread, Kyle's Lambda price impact, and Almgren-Chriss execution schedules.
* **Tab 11: `Peer Contagion & Spillover`**: Directional volatility transmission, sector absorption, and comovement correlation.

### 2. Standalone Interactive HTML Report (`out/<ticker>_report.html`)
* Fully self-contained, zero-dependency HTML file with responsive visual cards:
  * **Corporate Valuation, WACC & DCF Panel**: DCF Target Price, Margin of Safety, 5x5 Sensitivity Grid, and Dupont 5-Factor ROE.
  * **Multi-Factor Pricing & Execution Sizing Panel**: Fama-French Alpha, Factor Betas, and Almgren-Chriss slippage budget.
  * **Forensic Accounting & Solvency Triad Panel**: Altman Z"-Score, Beneish M-Score, and Piotroski F-Score.
  * **Market Microstructure & Copula Panel**: Roll spread, Kyle's lambda, VPIN, and tail dependence $\lambda_L, \lambda_U$.
  * **SEBI LODR Badges**: Color-coded Regulation 30 materiality badges.

### 3. Structured Machine-Readable JSON Payloads (`json/<ticker>_analysis.json`)
* Complete serialized econometric outputs, fitted market beta, news items, daily time series, and forensic diagnostics automatically routed to the dedicated `json/` directory.

### 4. Automatic Multi-Directory Output Separation
* `Excel/`: Contains all dynamic institutional `.xlsx` financial workbooks.
* `out/`: Contains all standalone interactive `.html` visual reports.
* `json/`: Contains all serialized `.json` data payloads.

---

## 6. Comprehensive Evolutionary Timeline & Version History

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CEIA VERSION EVOLUTION TIMELINE                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

 [CEIA 1.0 Foundation] (Phases 1-6)
   │ • Market Model (OLS) & Benchmark Residuals
   │ • Multi-Source Scrapers: Moneycontrol, Livemint, Economic Times, Financial Express, Business Line
   │ • FinBERT Financial Sentiment Engine & Candidate Event Day Flagging
   ▼
 [CEIA 2.0 Multi-Modal Enrichment & Unlisted Expansion] (Phases 7-14)
   │ • GoEmotions 28-Category Corporate Emotion Radar
   │ • Screener.in Fundamentals & NOPAT Ingestion
   │ • Unlisted / Pre-IPO Outlets Added: Entrackr, VCCircle, Inc42 & UnlistedZone Price Feeds
   │ • Global Market Backdrop (S&P 500, FTSE, Nikkei) & Secondary Competitor Benchmarks
   ▼
 [CEIA 3.0 Advanced Econometrics & Risk] (Phases 15-21)
   │ • CUSUM Trajectory Calibration & Overnight vs Intraday Gap Decomposition
   │ • Merton Distance-to-Default (DD) Structural Model & Value at Risk (VaR / EVT)
   │ • Multi-Factor Macro Engine (Crude Oil, USD/INR, G-Sec Yields) & Causal Decoupling
   ▼
 [CEIA 4.0 - 4.6 Frontier Quantitative Suite] (Phases 22-32)
   │ • 2-State Markov Regime-Switching Volatility & Macro Stress Simulator
   │ • Asymmetric Downside Beta (β⁻) & Amihud Microstructure Illiquidity Engine
   │ • Media Cascade Echo Multipliers & NCSKEW / DUVOL Crash Asymmetry
   │ • SEBI LODR Regulation 30 Materiality Classifier & Volatility Break F-Tests
   │ • Systematic Coskewness / Cokurtosis & Asymptotic CAR Discovery Half-Life
   ▼
 [CEIA 5.0 Comprehensive Institutional Milestone] (Phase 33)
   │ • Copula Tail Dependence Matrix (Clayton Crash vs Gumbel Rally Dependence)
   │ • Roll (1984) Spread, Kyle's Lambda & VPIN Toxicity Probability
   │ • Cornish-Fisher Fat-Tailed Expected Shortfall (CVaR) Multi-Horizon Surface
   │ • Linguistic Complexity (Gunning Fog) & Analyst Q&A Evasion Scorer
   │ • Econometric BMP Standardized Tests, Corrado Rank Tests & Jump Diffusion
   │ • Fundamental Solvency Triad: Altman Z"-Score, Beneish M-Score, Piotroski F-Score
   │ • 8-Tab Institutional OpenXML Excel Suite & Automatic Output Directory Separation
   ▼
 [CEIA 5.2 Frontier Suite] (Phase 34)
   │ • Synthetic Control Method (SCM - Abadie et al.) Counterfactual Optimization
   │ • Corporate PDF Transcript & Annual Report Extractor (ceia.pdf_extract)
   │ • Cross-Sectional Multi-Company Peer Cohort Benchmark (ceia.peer_benchmark)
   │ • Enhanced Visual HTML Dashboard with Microstructure & Forensic Solvency Panels
   ▼
 [CEIA 6.0 Institutional Financial Modeling Suite] (Phase 35)
   │ • Dynamic 11-Tab Excel Financial Model with Live Native Formulas (=SLOPE, =AVERAGE, =STDEV.S)
   │ • Interactive HTML Report with Corporate Valuation, DCF 5x5 Grid & Factor Panels
   │ • 2-Stage Discounted Cash Flow (DCF) Model with 5x5 WACC vs Growth Sensitivity Grids
   │ • Dupont 5-Factor ROE Decomposition Engine (Tax, Interest, Margins, Turnover, Leverage)
   │ • Fama-French 4-Factor Multi-Factor Pricing & Factor Alpha Risk Attribution
   │ • Almgren-Chriss (2000) Algorithmic Order Execution Sizing & Slippage Budgeting
   ▼
 [CEIA 7.0 Solvency Distress Ensemble, Bayesian DCF & Full Parity Suite] (Phase 36)
   │ • Unlisted News JSON Caching Parity: ceia.unlisted automatically persists/reuses data/news_cache/*.json
   │ • Ohlson (1980) 9-Factor Logit Default O-Score & Bankruptcy Probability Estimator
   │ • KMV Merton Option-Theoretic Structural Credit Rating (DD > 4σ Very Safe, DD < 2σ High Risk)
   │ • Cox Proportional Hazards Model & Semi-Parametric Multi-Year Default Survival Curve (1Y, 2Y, 3Y, 5Y)
   │ • Multi-Model Distress Ensemble synthesizing Altman Z", Ohlson O, Merton DD, Beneish M, Piotroski F
   │ • 3-Case Scenario DCF Modeling (Bull, Base, Bear) with EV & Margin of Safety schedules
   │ • Bayesian / Monte Carlo Probabilistic DCF (P10 Floor, P50 Median, P90 Ceiling, Undervaluation Prob %)
   │ • Core Financial Ratios Suite: ROA, ROE, ROCE, D/E, Interest Coverage, Current Ratio, Cash Ratio
   │ • Share Price Drawdown Dynamics: Max Drawdown %, Peak-to-Trough Duration, Calmar Ratio
   │ • 14-Tab Dynamic Institutional Excel Financial Model Suite with 100% Data & Report Parity
    ▼
  [CEIA 8.0 Predictive Analytics & Multi-Horizon Financial Forecasting Suite] (Phase 37)
    │ • Heterogeneous Autoregressive Realized Volatility with Leverage & Jumps (HAR-RV-CJ/SV, Corsi 2009)
    │ • Adaptive Conformal Inference (ACI) Calibrated Multi-Horizon Quantile Forecaster (Gibbs & Candès 2021)
    │ • Event-Conditioned Sentiment Impulse-Response & Post-Earnings Announcement Drift (PEAD, Lopez-Lira 2023)
    │ • Macro-Conditioned Factor Ridge & Microstructure Regularized Forecaster (De Mol et al. 2008)
    │ • Multi-Horizon Trajectory Cones across 5-Day Tactical, 21-Day Swing, and 63-Day Fundamental Horizons
    │ • Probabilistic Bounds (P10 Bear Case Floor, P50 Base/Median, P90 Bull Case Ceiling)
    │ • Directional Probability P(Up) and Probability of Generating Excess Alpha vs Nifty 50
    │ • 15-Tab Dynamic Institutional Excel Financial Model Suite with Live Formula Integration
    │ • Interactive HTML Dashboard with Vector SVG Forecast Cone Visualization & Technical Decomposition
```

---

## 7. CLI Execution Manual & Command Cheat Sheet

### 1. Listed Equities Analysis (Standard Run with Live Dynamic Excel Model)
```powershell
python -m ceia.analyze --company "Goa Shipyard Ltd" --ticker "GSL.NS" --start 2025-01-01 --end 2026-01-01 --html gsl_report.html --xlsx Excel/gsl_model_report.xlsx --out gsl_report.json
```

### 2. Unlisted & Pre-IPO Analysis (Entrackr, VCCircle, Inc42 & UnlistedZone)
```powershell
python -m ceia.unlisted --company "Tata Sons" --start 2023-01-01 --end 2026-01-01
```

### 3. Cross-Sectional Multi-Company Peer Cohort Benchmark
```powershell
python -m ceia.peer_benchmark --tickers "GSL.NS,MAZDOCK.NS,COCHINSHIP.NS,GRSE.NS" --sector "Defence Shipbuilders" --start 2025-01-01 --end 2026-01-01 --out out/peer_benchmark.html
```

### 4. Corporate PDF Transcript & Document Forensic Analysis
```powershell
python -m ceia.pdf_extract --pdf "data/transcripts/gsl_q3_concall.pdf" --company "Goa Shipyard Ltd" --out out/pdf_forensic.json
```

### 5. Running the Complete Automated Test Suite (58 Test Modules)
```powershell
python -m pytest -q
# Expected output: 810 passed, 3 skipped in ~14s
```
