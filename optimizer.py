import numpy as np
import pandas as pd
from scipy.optimize import minimize

SECTOR_MAP = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS'],
    'IT': ['TCS.NS', 'INFY.NS'],
    'Auto': ['M&M.NS', 'MARUTI.NS'],
    'Pharma': ['SUNPHARMA.NS', 'CIPLA.NS']
}
SECTORS = list(SECTOR_MAP.keys())
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065  # ~6.5% India 10-Year G-Sec benchmark yield

def load_market_data():
    """Loads pre-compiled NSE stock and NIFTY 50 benchmark prices."""
    stocks = pd.read_csv('data/nifty_sector_prices.csv', index_col='Date', parse_dates=True)
    bench = pd.read_csv('data/nifty50_benchmark.csv', index_col='Date', parse_dates=True)

    stock_returns = stocks.pct_change().dropna()
    bench_returns = bench['NIFTY50'].pct_change().dropna()

    # Build sector-level daily returns (equal-weighted across the 2 sector leaders)
    sector_returns = pd.DataFrame(index=stock_returns.index)
    for sector, tickers in SECTOR_MAP.items():
        sector_returns[sector] = stock_returns[tickers].mean(axis=1)

    return stocks, bench, sector_returns, bench_returns

def calculate_portfolio_stats(weights, sector_returns, capital=1000000):
    """
    Computes arithmetic return, CAGR, volatility, Sharpe ratio, Day-1 anchored
    Max Drawdown, 95% daily VaR, and cumulative wealth curve.
    """
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    daily_port_ret = sector_returns.dot(weights)

    # 1. Arithmetic Annualized Return (for Sharpe Ratio consistency) & Volatility
    ann_return = daily_port_ret.mean() * TRADING_DAYS
    ann_vol = daily_port_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    # 2. Compounded Annual Growth Rate (CAGR) matching final compounded wealth
    cum_growth = (1 + daily_port_ret).cumprod()
    num_years = len(daily_port_ret) / TRADING_DAYS
    cagr = (cum_growth.iloc[-1]) ** (1 / num_years) - 1
    wealth_series = cum_growth * capital

    # 3. Day-1 Anchored Maximum Drawdown (MDD)
    running_max = np.maximum(1.0, cum_growth.cummax())
    drawdown_series = (cum_growth - running_max) / running_max
    max_drawdown = drawdown_series.min()

    # 4. 95% Historical Value at Risk (Daily 5th percentile loss)
    var_95_pct = np.percentile(daily_port_ret, 5)
    var_95_inr = abs(var_95_pct) * capital

    return {
        'weights': weights,
        'ann_return_pct': round(ann_return * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'max_drawdown_inr': round(abs(max_drawdown) * capital, 0),
        'var_95_pct': round(abs(var_95_pct) * 100, 2),
        'var_95_inr': round(var_95_inr, 0),
        'daily_returns': daily_port_ret,
        'wealth_series': wealth_series,
        'drawdown_series': drawdown_series * 100
    }

def find_optimal_max_sharpe(sector_returns):
    """Uses SciPy SLSQP optimization with convergence check to maximize Sharpe Ratio."""
    num_sectors = len(SECTORS)
    mean_ret = sector_returns.mean() * TRADING_DAYS
    cov_matrix = sector_returns.cov() * TRADING_DAYS

    def neg_sharpe(weights):
        p_ret = np.dot(weights, mean_ret)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - RISK_FREE_RATE) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.05, 0.60) for _ in range(num_sectors))
    init_guess = np.array([1.0 / num_sectors] * num_sectors)

    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Portfolio optimization failed to converge: {result.message}")

    return np.round(result.x, 4)

def run_monte_carlo_simulation(sector_returns, num_portfolios=3000, seed=42):
    """Generates 3,000 random sector allocations to visualize the Efficient Frontier."""
    np.random.seed(seed)
    mean_ret = sector_returns.mean() * TRADING_DAYS
    cov_matrix = sector_returns.cov() * TRADING_DAYS

    results = []
    for _ in range(num_portfolios):
        w = np.random.dirichlet(np.ones(len(SECTORS)))
        p_ret = np.dot(w, mean_ret)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (p_ret - RISK_FREE_RATE) / p_vol
        results.append({
            'Return (%)': round(p_ret * 100, 2),
            'Volatility (%)': round(p_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Allocation': f"Bank {w[0]:.0%} | IT {w[1]:.0%} | Auto {w[2]:.0%} | Pharma {w[3]:.0%}"
        })

    return pd.DataFrame(results)
