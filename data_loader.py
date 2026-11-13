import os
import yfinance as yf
import pandas as pd
import numpy as np

# 8 Liquid NIFTY 50 Leaders across 4 Core Sectors
SECTOR_TICKERS = {
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'TCS.NS': 'IT',
    'INFY.NS': 'IT',
    'M&M.NS': 'Auto',
    'MARUTI.NS': 'Auto',
    'SUNPHARMA.NS': 'Pharma',
    'CIPLA.NS': 'Pharma'
}
BENCHMARK_TICKER = '^NSEI'  # NIFTY 50 Index

def extract_and_clean_nse_data(period="5y"):
    os.makedirs('data', exist_ok=True)
    tickers = list(SECTOR_TICKERS.keys())

    print(f"Downloading {period} of split-adjusted NSE data for {len(tickers)} NIFTY leaders...")
    # auto_adjust=True automatically adjusts historical prices for stock splits & dividends
    raw_stocks = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']

    print("Downloading NIFTY 50 (^NSEI) benchmark data...")
    raw_bench = yf.download(BENCHMARK_TICKER, period=period, auto_adjust=True, progress=False)['Close']

    # Ensure raw_bench is a single-column DataFrame named NIFTY50
    if isinstance(raw_bench, pd.Series):
        bench_df = raw_bench.to_frame(name='NIFTY50')
    else:
        bench_df = raw_bench.rename(columns={BENCHMARK_TICKER: 'NIFTY50'})

    # --- FINANCIAL DATA CLEANING PIPELINE ---
    # 1. Remove zero or negative corrupt prices
    raw_stocks = raw_stocks.where(raw_stocks > 0, np.nan)
    bench_df = bench_df.where(bench_df > 0, np.nan)

    # 2. Align trading dates across all stocks and forward-fill isolated 1-2 day exchange gaps
    stocks_clean = raw_stocks.ffill(limit=2).dropna()
    bench_clean = bench_df.ffill(limit=2).dropna()

    # 3. Intersect exact trading dates between stocks and benchmark
    common_dates = stocks_clean.index.intersection(bench_clean.index)
    stocks_clean = stocks_clean.loc[common_dates].round(2)
    bench_clean = bench_clean.loc[common_dates].round(2)

    # 4. Sanity check for impossible API spike glitches (> 25% single-day jump outside NSE circuit limits)
    daily_returns = stocks_clean.pct_change().dropna()
    max_move = daily_returns.abs().max().max()
    if max_move > 0.25:
        print(f"Warning: Detected an extreme single-day move of {max_move:.1%}. Please inspect raw tickers.")
    else:
        print(f"Data integrity check passed (Max single-day move across 5 years: {max_move:.2%}).")

    # Save clean datasets to data/
    stocks_clean.index.name = 'Date'
    bench_clean.index.name = 'Date'

    stocks_clean.to_csv('data/nifty_sector_prices.csv')
    bench_clean.to_csv('data/nifty50_benchmark.csv')

    print(f"Success! Saved {len(stocks_clean)} trading days ({stocks_clean.index.min().date()} to {stocks_clean.index.max().date()}) into 'data/'.")

if __name__ == "__main__":
    extract_and_clean_nse_data()
