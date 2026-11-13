# 📈 NIFTY 50 Portfolio Risk & Sector Allocation Optimizer
**Quantitative Risk Modeling & Modern Portfolio Theory (2021–2026 NSE Equities)**

## 📌 Project Overview
This interactive quantitative finance dashboard answers a core portfolio management question:
> *"How can an investor maximize risk-adjusted returns across top NIFTY 50 sectors (Banking, IT, Auto, Pharma) while minimizing maximum drawdown during market corrections?"*

Using 5 years of split-adjusted historical National Stock Exchange (NSE) data (**1,235 trading days from November 2021 to November 2026**), this application simulates 3,000 sector allocations, computes institutional tail-risk metrics (CAGR, Maximum Drawdown, and 95% Historical Value at Risk), and solves for the mathematically optimal Max-Sharpe portfolio using constrained optimization.

---

## 🛠️ Tech Stack & Methodology
* **Data Extraction & Cleaning:** Python, `yfinance`, `Pandas` (split/dividend adjustment via `auto_adjust=True`, exchange holiday alignment across tickers, and NSE circuit-limit integrity checks)
* **Quantitative Risk & Optimization Engine:** `NumPy`, `SciPy` (`SLSQP` constrained optimization maximizing Sharpe Ratio above India's 6.5% 10-Year G-Sec risk-free rate), Monte Carlo simulation (3,000 portfolios), Day-1 anchored Maximum Drawdown, and 95% Historical VaR
* **Interactive Visualization:** `Plotly` & `Streamlit`

### Sector Basket Analyzed
* **Banking & Financials:** `HDFCBANK.NS`, `ICICIBANK.NS`
* **Information Technology (IT):** `TCS.NS`, `INFY.NS`
* **Automobile:** `M&M.NS`, `MARUTI.NS`
* **Pharmaceuticals:** `SUNPHARMA.NS`, `CIPLA.NS`
* **Benchmark Index:** NIFTY 50 (`^NSEI`)

### Repository Structure
```text
nifty-portfolio-optimizer/
├── data/                        # Pre-compiled 5-year NSE datasets for fast cloud rendering
│   ├── nifty_sector_prices.csv  # Split-adjusted daily closing prices for 8 NIFTY leaders
│   └── nifty50_benchmark.csv    # Split-adjusted daily closing prices for NIFTY 50 (^NSEI)
├── data_loader.py               # ETL & financial data cleaning pipeline
├── optimizer.py                 # MPT, Monte Carlo, CAGR, VaR & Drawdown math engine
├── app.py                       # Interactive Streamlit + Plotly simulator dashboard
└── requirements.txt             # Production dependencies
```

---

## 📊 Key Features
* **Interactive Allocation Simulator:** Adjust sector weight sliders in real time (auto-normalized to 100%) or toggle to the **SciPy Optimal (Max-Sharpe)** allocation, with customizable `₹ INR` starting capital.
* **Executive Risk KPI Cards:** Instant calculation of Compounded Annual Return (CAGR), Annualized Volatility, Sharpe Ratio, and 5-Year Maximum Drawdown (in `%` and `₹`).
* **Monte Carlo Efficient Frontier:** Plots 3,000 simulated portfolios colored by Sharpe Ratio alongside your current portfolio star and the optimal max-sharpe diamond.
* **5-Year Wealth & Underwater Drawdown Backtest:** Tracks compounded `₹` growth and peak-to-trough drawdowns against the NIFTY 50 benchmark.
* **Cross-Sector Correlation & 95% VaR Analysis:** Visualizes sector diversification benefits and daily tail-risk loss thresholds.

---

## 💡 Key Findings (2021–2026 NSE Cycle)
* **Equal-Weight Baseline vs. Benchmark:** An equal-weight basket (25% each across Banking, IT, Auto, and Pharma) delivered **11.75%** annualized return (**0.38** Sharpe Ratio) compared to **7.65%** annualized return for the NIFTY 50 (`^NSEI`).
* **SciPy Max-Sharpe Optimization:** Overweighting high-momentum **Auto** (`54.1%`) and low-correlation defensive **Pharma** (`35.9%`) alongside `5.0%` Banking and `5.0%` IT doubled the Sharpe Ratio to **0.76** (**18.49%** arithmetic return) while reducing 5-year Maximum Drawdown from `-21.69%` to **-21.35%**.
* **Diversification Takeaway:** Pairing defensive Pharma with cyclical Auto and Banking lowers portfolio correlation during broad market sell-offs without sacrificing long-term compounded growth.

---

## 🚀 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd nifty-portfolio-optimizer
   ```
2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Launch the Streamlit simulator:**
   ```bash
   streamlit run app.py
   ```
