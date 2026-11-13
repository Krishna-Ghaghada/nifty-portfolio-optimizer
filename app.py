import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimizer import (
    SECTORS, SECTOR_MAP, TRADING_DAYS, RISK_FREE_RATE,
    load_market_data, calculate_portfolio_stats,
    find_optimal_max_sharpe, run_monte_carlo_simulation
)

# 1. Page Configuration
st.set_page_config(
    page_title="NIFTY 50 Portfolio Risk & Sector Optimizer",
    page_icon="📈",
    layout="wide"
)

# Cache data and Monte Carlo simulation so slider adjustments respond instantaneously
@st.cache_data
def get_cached_market_data():
    stocks, bench, sector_ret, bench_ret = load_market_data()
    opt_weights = find_optimal_max_sharpe(sector_ret)
    mc_df = run_monte_carlo_simulation(sector_ret, num_portfolios=3000)
    return stocks, bench, sector_ret, bench_ret, opt_weights, mc_df

stocks_df, bench_df, sector_returns, bench_returns, opt_weights, mc_df = get_cached_market_data()
bench_returns = bench_returns.reindex(sector_returns.index).fillna(0.0)

# 2. Sidebar Simulator Controls
st.sidebar.header("⚙️ Portfolio Simulator Controls")
alloc_mode = st.sidebar.radio(
    "Allocation Mode:",
    options=["Custom Sector Sliders", "SciPy Optimal (Max Sharpe)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sector Allocation Weights")

if alloc_mode == "SciPy Optimal (Max Sharpe)":
    raw_weights = opt_weights
    for sec, w in zip(SECTORS, raw_weights):
        st.sidebar.slider(f"{sec} (%)", 0, 100, int(round(w * 100)), disabled=True)
else:
    w_bank = st.sidebar.slider("🏦 Banking (HDFCBANK, ICICIBANK)", 0, 100, 25)
    w_it = st.sidebar.slider("💻 IT (TCS, INFY)", 0, 100, 25)
    w_auto = st.sidebar.slider("🚗 Auto (M&M, MARUTI)", 0, 100, 25)
    w_pharma = st.sidebar.slider("💊 Pharma (SUNPHARMA, CIPLA)", 0, 100, 25)
    raw_weights = [w_bank, w_it, w_auto, w_pharma]
    if sum(raw_weights) == 0:
        raw_weights = [1, 1, 1, 1]

norm_weights = np.array(raw_weights, dtype=float) / np.sum(raw_weights)
st.sidebar.caption(
    f"**Normalized Weights:** Bank {norm_weights[0]:.0%} | IT {norm_weights[1]:.0%} | "
    f"Auto {norm_weights[2]:.0%} | Pharma {norm_weights[3]:.0%}"
)

st.sidebar.markdown("---")
capital = st.sidebar.number_input(
    "Initial Capital (₹ INR):",
    min_value=100000,
    max_value=100000000,
    value=1000000,
    step=100000
)
st.sidebar.info(
    f"**Period:** 5 Years ({sector_returns.index.min().date()} to {sector_returns.index.max().date()})\n\n"
    f"**Risk-Free Rate:** {RISK_FREE_RATE:.1%} (India 10Y G-Sec)\n\n"
    "**Benchmark:** NIFTY 50 (`^NSEI`)"
)

# Calculate stats for Current Portfolio, Optimal Portfolio, and NIFTY 50 Benchmark
curr_stats = calculate_portfolio_stats(norm_weights, sector_returns, capital)
opt_stats = calculate_portfolio_stats(opt_weights, sector_returns, capital)

bench_ann_ret = round(bench_returns.mean() * TRADING_DAYS * 100, 2)
bench_cum = (1 + bench_returns).cumprod()
bench_cagr = round((bench_cum.iloc[-1] ** (1 / (len(bench_returns) / TRADING_DAYS)) - 1) * 100, 2)
bench_ann_vol = round(bench_returns.std() * np.sqrt(TRADING_DAYS) * 100, 2)
bench_wealth = bench_cum * capital
bench_dd = (bench_cum / np.maximum(1.0, bench_cum.cummax()) - 1) * 100

# 3. Header & Core Question
st.title("📈 NIFTY 50 Portfolio Risk & Sector Allocation Optimizer")
st.markdown(
    "**Core Question:** *How can an investor maximize risk-adjusted returns across top NIFTY 50 sectors "
    "(Banking, IT, Auto, Pharma) while minimizing maximum drawdown during market corrections?*"
)
st.markdown("---")

# 4. Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    cagr_diff = round(curr_stats['cagr_pct'] - bench_cagr, 2)
    st.metric(
        label="Compounded Return (CAGR)",
        value=f"{curr_stats['cagr_pct']}%",
        delta=f"{cagr_diff:+.2f}% vs NIFTY 50 ({bench_cagr}%)"
    )
with col2:
    vol_diff = round(curr_stats['ann_vol_pct'] - bench_ann_vol, 2)
    st.metric(
        label="Annualized Volatility (Risk)",
        value=f"{curr_stats['ann_vol_pct']}%",
        delta=f"{vol_diff:+.2f}% vs NIFTY 50",
        delta_color="inverse"
    )
with col3:
    sharpe_diff = round(curr_stats['sharpe_ratio'] - opt_stats['sharpe_ratio'], 2)
    st.metric(
        label="Sharpe Ratio",
        value=f"{curr_stats['sharpe_ratio']}",
        delta=f"{sharpe_diff:+.2f} vs Optimal ({opt_stats['sharpe_ratio']})"
    )
with col4:
    st.metric(
        label="Maximum Drawdown (5Y Worst)",
        value=f"{curr_stats['max_drawdown_pct']}%",
        delta=f"-₹{curr_stats['max_drawdown_inr']:,.0f} Peak-to-Trough",
        delta_color="inverse"
    )

st.markdown("---")

# 5. Four Interactive Plotly Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Efficient Frontier (Monte Carlo)",
    "💰 Cumulative Wealth & Drawdown",
    "🔗 Sector Correlation Matrix",
    "⚠️ 95% Value at Risk (Tail Risk)"
])

# --- TAB 1: EFFICIENT FRONTIER ---
with tab1:
    st.subheader("Modern Portfolio Theory: 3,000 Simulated Sector Allocations")
    fig_ef = px.scatter(
        mc_df,
        x="Volatility (%)",
        y="Return (%)",
        color="Sharpe Ratio",
        color_continuous_scale="Viridis",
        hover_data=["Allocation"],
        opacity=0.65,
        height=470
    )

    # Plot Current Portfolio Star
    fig_ef.add_trace(go.Scatter(
        x=[curr_stats['ann_vol_pct']],
        y=[curr_stats['ann_return_pct']],
        mode='markers+text',
        marker=dict(color='#FFD700', size=18, symbol='star', line=dict(color='black', width=1.5)),
        name='Your Current Portfolio',
        text=['Your Portfolio'],
        textposition='top center'
    ))

    # Plot Optimal Max-Sharpe Diamond with exact sector breakdown in legend
    opt_label = f"Optimal: Bank {opt_weights[0]:.0%}, IT {opt_weights[1]:.0%}, Auto {opt_weights[2]:.0%}, Pharma {opt_weights[3]:.0%}"
    fig_ef.add_trace(go.Scatter(
        x=[opt_stats['ann_vol_pct']],
        y=[opt_stats['ann_return_pct']],
        mode='markers+text',
        marker=dict(color='#00CC96', size=16, symbol='diamond', line=dict(color='black', width=1.5)),
        name=opt_label,
        text=['Optimal Max-Sharpe'],
        textposition='bottom center'
    ))

    fig_ef.update_layout(
        hovermode="closest",
        legend=dict(
            yanchor="bottom",
            y=0.94,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(0, 0, 0, 0)",
            bordercolor="lightgray",
            borderwidth=1
        )
    )
    st.plotly_chart(fig_ef, use_container_width=True)


# --- TAB 2: CUMULATIVE WEALTH & DRAWDOWN ---
with tab2:
    st.subheader(f"5-Year Backtest: Growth of ₹{capital:,.0f} vs. NIFTY 50 Benchmark")
    wealth_df = pd.DataFrame({
        'Your Portfolio': curr_stats['wealth_series'],
        'Optimal Max-Sharpe': opt_stats['wealth_series'],
        'NIFTY 50 (^NSEI)': bench_wealth
    })
    fig_wealth = px.line(
        wealth_df,
        labels={'value': 'Portfolio Value (₹ INR)', 'variable': 'Strategy'},
        color_discrete_map={
            'Your Portfolio': '#1f77b4',
            'Optimal Max-Sharpe': '#00CC96',
            'NIFTY 50 (^NSEI)': '#FF7F0E'
        },
        height=380
    )
    fig_wealth.update_layout(hovermode="x unified")
    st.plotly_chart(fig_wealth, use_container_width=True)

    st.subheader("Underwater Drawdown Curve (% Below Peak)")
    dd_df = pd.DataFrame({
        'Your Portfolio DD (%)': curr_stats['drawdown_series'],
        'NIFTY 50 DD (%)': bench_dd
    })
    fig_dd = px.area(
        dd_df,
        labels={'value': 'Drawdown (%)', 'variable': 'Series'},
        color_discrete_map={'Your Portfolio DD (%)': '#EF553B', 'NIFTY 50 DD (%)': '#636EFA'},
        height=260
    )
    fig_dd.update_layout(hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

# --- TAB 3: CORRELATION HEATMAP ---
with tab3:
    st.subheader("Cross-Sector Daily Return Correlation Matrix")
    corr_matrix = sector_returns.corr().round(2)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        height=430
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Lower correlation between defensive Pharma and cyclical sectors (Banking, Auto) reduces overall portfolio volatility during broad market corrections."
    )

# --- TAB 4: VALUE AT RISK (VaR) ---
with tab4:
    st.subheader("Daily Return Distribution & 95% Value at Risk (VaR)")
    daily_pct = curr_stats['daily_returns'] * 100
    var_cutoff = -curr_stats['var_95_pct']

    fig_var = px.histogram(
        x=daily_pct,
        nbins=60,
        labels={'x': 'Daily Portfolio Return (%)', 'y': 'Trading Days'},
        color_discrete_sequence=['#636EFA'],
        height=400
    )
    fig_var.add_vline(
        x=var_cutoff,
        line_dash="dash",
        line_color="red",
        annotation_text=f"95% Daily VaR: {var_cutoff:.2f}% (-₹{curr_stats['var_95_inr']:,.0f})"
    )
    st.plotly_chart(fig_var, use_container_width=True)

# 6. Dynamic Quantitative Verdict
st.markdown("---")
st.subheader("💡 Quantitative Allocation Verdict & Risk Summary")
if alloc_mode == "SciPy Optimal (Max Sharpe)":
    st.success(
        f"**Optimal Allocation Active:** The SciPy SLSQP optimizer allocates "
        f"**{opt_weights[2]:.1%} Auto, {opt_weights[3]:.1%} Pharma, {opt_weights[0]:.1%} Banking, and {opt_weights[1]:.1%} IT**, "
        f"achieving a **{opt_stats['sharpe_ratio']} Sharpe Ratio** (**{opt_stats['cagr_pct']}% CAGR** / "
        f"{opt_stats['ann_return_pct']}% arithmetic return vs. {bench_cagr}% CAGR for NIFTY 50). "
        f"At a 95% confidence level, daily losses do not exceed **₹{opt_stats['var_95_inr']:,.0f} ({opt_stats['var_95_pct']}%)**."
    )
else:
    sharpe_gain = round(opt_stats['sharpe_ratio'] - curr_stats['sharpe_ratio'], 2)
    if sharpe_gain > 0:
        st.info(
            f"**Current Custom Allocation:** Your portfolio yields **{curr_stats['cagr_pct']}% CAGR** "
            f"({curr_stats['ann_return_pct']}% arithmetic return) with **{curr_stats['ann_vol_pct']}%** volatility "
            f"(Sharpe Ratio: **{curr_stats['sharpe_ratio']}**) and a 95% daily Value at Risk of **₹{curr_stats['var_95_inr']:,.0f}**. "
            f"Switching to the **SciPy Optimal Max-Sharpe** allocation improves your Sharpe Ratio by **{sharpe_gain:+.2f}** "
            f"(to **{opt_stats['sharpe_ratio']}**) by overweighting high-momentum Auto and low-correlation defensive Pharma."
        )
    else:
        st.success(
            f"**Outperforming Constrained Optimal:** Your custom allocation yields **{curr_stats['cagr_pct']}% CAGR** "
            f"({curr_stats['ann_return_pct']}% arithmetic return) with a **{curr_stats['sharpe_ratio']} Sharpe Ratio**—beating "
            f"the 5%-minimum-constrained SciPy portfolio (**{opt_stats['sharpe_ratio']}**) because it allows **0% allocation** "
            f"to lagging sectors. Daily 95% VaR is **₹{curr_stats['var_95_inr']:,.0f}**."
        )

