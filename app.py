import sys
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from simulation_engine import SimConfig, run_simulation
from historic_returns import get_historic_return_matrix

st.set_page_config(page_title="Swiss Early Retirement Simulator", layout="wide")

st.markdown("""
<style>
    /* Condense the sidebar vertical spacing without overlapping */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem !important;
    }
    /* Condense headers but leave breathing room */
    section[data-testid="stSidebar"] h3 {
        padding-top: 1rem !important;
        padding-bottom: 0.25rem !important;
    }
    /* Condense dividers */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.5em !important;
        margin-bottom: 0.5em !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Swiss Early Retirement Simulator (Zurich Phase 1)")

st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# 1. Demographics
st.sidebar.subheader("Demographics")
start_age = st.sidebar.number_input("Retirement Age", value=40, min_value=30, max_value=65, help="Your current age. This is when the simulation begins.")
duration = st.sidebar.number_input("Simulation Duration (Years)", value=50, min_value=10, max_value=70, help="How many years into the future the simulation should run.")

# Success Criteria
st.sidebar.subheader("Success Criteria")
success_pct = st.sidebar.number_input(
    "Target Ending NW (% of Inflation-Adj. Start NW)",
    value=50.0,
    step=5.0,
    format="%.1f",
    help="The percentage of inflation-adjusted starting net worth you want to preserve at the end of the simulation. 0.0% means you just want to avoid going broke (survival)."
)


# 2. Initial Assets
st.sidebar.subheader("Initial Assets (CHF)")
initial_liquid_wealth = st.sidebar.number_input("Taxable Liquid Wealth (CHF)", value=2_400_000, step=100_000, help="Your easily accessible taxable investments (stocks, bonds, cash). Do not include your primary residence.")
initial_pillar_2 = st.sidebar.number_input("Pillar 2 (Freizügigkeitskonto)", value=450_000, step=50_000, help="The current balance of your Swiss Pillar 2 pension. Assumed to be 100% invested in equities (proportional to your US vs Non-US target allocation). Withdrawn at age 65.")

num_pillar_3a = st.sidebar.number_input("Number of Pillar 3a Accounts", value=5, min_value=0, max_value=5, help="How many separate Pillar 3a accounts you hold. Liquidated sequentially starting 5 years before age 65 to minimize taxes.")
pillar_3a_accounts = []
for i in range(int(num_pillar_3a)):
    val = st.sidebar.number_input(f"Pillar 3a Account {i+1}", value=20_000, step=5000, help="The balance of this specific Pillar 3a account. Assumed to be 100% invested in equities (proportional to your US vs Non-US target allocation).")
    pillar_3a_accounts.append(val)

# 3. Asset Allocation
st.sidebar.subheader("Target Asset Allocation (%)")
alloc_us = st.sidebar.number_input("US Stocks", value=50.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio invested in US Stocks.")
alloc_non_us = st.sidebar.number_input("Non-US Stocks", value=30.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio invested in Non-US Stocks.")
alloc_cash = st.sidebar.number_input("CHF Cash", value=10.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio held in CHF Cash.")
alloc_gold = st.sidebar.number_input("Gold", value=3.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio held in Gold.")
alloc_btc = st.sidebar.number_input("Bitcoin", value=7.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio held in Bitcoin.")

total_alloc = alloc_us + alloc_non_us + alloc_cash + alloc_gold + alloc_btc
if not np.isclose(total_alloc, 100.0, atol=0.01):
    st.sidebar.error(f"Asset allocation must sum to exactly 100%. Currently at {total_alloc:.2f}%. Please adjust.")
    st.stop()

# 4. Rebalancing
st.sidebar.subheader("Rebalancing Strategy")
rebalance_strategy = st.sidebar.selectbox("Frequency", ["Never", "Quarterly", "Yearly", "Monthly", "Threshold"], index=2, help="How often to rebalance your portfolio back to your target asset allocation.")
rebalance_threshold = 0.0
if rebalance_strategy == "Threshold":
    rebalance_threshold = st.sidebar.number_input("Threshold Drift (%)", value=1.0, step=0.1, format="%.1f", help="If rebalancing based on threshold, the max absolute drift allowed before rebalancing.") / 100.0

st.sidebar.subheader("Withdrawal Strategy")
enable_smart_selling = st.sidebar.checkbox("Smart Cash Buffer", value=True, help="During market downturns (net worth < inflation-adjusted start), skip rebalancing and spend down Cash first. Only sell other assets if Cash is fully depleted. Once recovered, normal rebalancing resumes.")

# 5. Economics
st.sidebar.subheader("Economics")

annual_expenses = st.sidebar.number_input("Annual Base Expenses (CHF)", value=85_000, step=5000, help="Your expected base living expenses in today's CHF. This will automatically inflate each year.")
enable_dynamic_expenses = st.sidebar.checkbox("Enable Dynamic Expenses", value=True, help="If enabled, base expenses are reduced when your net worth falls below your inflation-adjusted starting net worth.")
dynamic_expense_floor_pct = 100.0
if enable_dynamic_expenses:
    dynamic_expense_floor_pct = st.sidebar.number_input("Reduced Expense Floor (%)", value=85.0, step=1.0, format="%.1f", help="The percentage of your base expenses you will spend when your net worth is below the watermark.")

monthly_ahv = st.sidebar.number_input("Expected Monthly AHV Pension from age 65 (CHF)", value=2000, step=100, help="The monthly AHV pension you expect to receive starting at age 65 (in today's CHF, adjusted annually for CPI inflation in the simulation).")
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", value=1.5, step=0.1, format="%.1f", help="Expected annual dividend yield of the portfolio.") / 100.0
inflation_mean = st.sidebar.number_input("Inflation Mean (%)", value=2.5, step=0.1, format="%.1f", help="Expected average annual inflation rate.") / 100.0
inflation_std = st.sidebar.number_input("Inflation Volatility (%)", value=1.0, step=0.1, format="%.1f", help="Expected volatility of inflation.") / 100.0

st.sidebar.subheader("Monte Carlo Parameters")
st.sidebar.caption("Note: Returns must be Nominal (unadjusted for inflation) and in CHF terms. E.g., historic US Stock returns are ~9.5% in USD, but ~7.0% in CHF due to currency drag.")
ret_us = st.sidebar.number_input("US Stocks Nominal Mean (%)", value=7.0, step=0.1, format="%.1f", help="Expected nominal mean return for US Stocks in CHF. Note: Historic S&P500 returns are ~9.5% in USD, but ~7.0% in CHF due to the appreciating Franc.") / 100.0
ret_non_us = st.sidebar.number_input("Non-US Stocks Nominal Mean (%)", value=6.0, step=0.1, format="%.1f", help="Expected nominal mean return for Non-US Stocks in CHF terms.") / 100.0
ret_cash = st.sidebar.number_input("CHF Cash Nominal Mean (%)", value=1.0, step=0.1, format="%.1f", help="Expected nominal mean return for CHF Cash.") / 100.0
ret_gold = st.sidebar.number_input("Gold Nominal Mean (%)", value=6.0, step=0.1, format="%.1f", help="Expected nominal mean return for Gold in CHF terms.") / 100.0
ret_btc = st.sidebar.number_input("Bitcoin Nominal Mean (%)", value=10.0, step=0.1, format="%.1f", help="Expected nominal mean return for Bitcoin in CHF terms.") / 100.0

vol_eq = st.sidebar.number_input("Equities Volatility (%)", value=15.0, step=0.1, format="%.1f", help="Expected volatility for Stocks.") / 100.0
vol_gold = st.sidebar.number_input("Gold Volatility (%)", value=15.0, step=0.1, format="%.1f", help="Expected volatility for Gold.") / 100.0
vol_btc = st.sidebar.number_input("Bitcoin Volatility (%)", value=60.0, step=0.1, format="%.1f", help="Expected volatility for Bitcoin.") / 100.0

mc_num_runs = int(st.sidebar.number_input("Number of Monte Carlo Runs", value=1000, min_value=100, max_value=10000, step=100, help="How many distinct future paths to simulate."))

try:
    dummy = get_historic_return_matrix(int(duration))
    hist_num_runs = dummy.shape[0]
    st.sidebar.info(f"Using 100-year historic/synthetic CHF returns. Available overlapping runs: {hist_num_runs}")
except ValueError as e:
    st.sidebar.error(str(e))
    hist_num_runs = 0



# 6. Tax Location
st.sidebar.subheader("Zurich Tax Location")
cantonal_multiplier = 0.95 # Zurich Cantonal Steuerfuss for 2026
municipal_multiplier = st.sidebar.number_input("Municipal Multiplier (Steuerfuss, %)", value=119.0, step=0.1, format="%.1f", help="Your municipal tax multiplier (Steuerfuss) in Zurich (e.g. 119% for City of Zurich).") / 100.0



if True:
    if hist_num_runs == 0:
        st.error("Cannot run simulation. Duration is too long for the available historic data.")
        st.stop()

    def create_config(num_runs):
        return SimConfig(
            num_runs=num_runs,
            duration_years=int(duration),
            inflation_mean=inflation_mean,
            inflation_std=inflation_std,
            start_age=int(start_age),
            dividend_yield=dividend_yield,
            enable_dynamic_expenses=enable_dynamic_expenses,
            dynamic_expense_floor_pct=dynamic_expense_floor_pct / 100.0,
            initial_liquid_wealth=initial_liquid_wealth,
            initial_pillar_2=initial_pillar_2,
            initial_pillar_3a_accounts=pillar_3a_accounts,
            alloc_us_stocks=alloc_us / 100.0,
            alloc_non_us_stocks=alloc_non_us / 100.0,
            alloc_chf_cash=alloc_cash / 100.0,
            alloc_gold=alloc_gold / 100.0,
            alloc_bitcoin=alloc_btc / 100.0,
            rebalance_strategy=rebalance_strategy,
            rebalance_threshold=rebalance_threshold,
            annual_base_expenses=annual_expenses,
            monthly_ahv_pension=float(monthly_ahv),
            enable_smart_selling=enable_smart_selling,
            cantonal_multiplier=cantonal_multiplier,
            municipal_multiplier=municipal_multiplier
        )
    
    config_mc = create_config(mc_num_runs)
    config_hist = create_config(hist_num_runs)
    
    from simulation_engine import generate_monte_carlo_returns
    rng = np.random.default_rng(42)
    
    mc_return_matrix = generate_monte_carlo_returns(
        num_runs=config_mc.num_runs,
        duration_years=config_mc.duration_years,
        ret_us=ret_us,
        ret_non_us=ret_non_us,
        ret_cash=ret_cash,
        ret_gold=ret_gold,
        ret_btc=ret_btc,
        vol_eq=vol_eq,
        vol_gold=vol_gold,
        vol_btc=vol_btc,
        seed=42
    )
    
    mc_inflation_matrix = rng.normal(inflation_mean, inflation_std, (config_mc.num_runs, config_mc.duration_years))
    
    hist_return_matrix = get_historic_return_matrix(config_hist.duration_years)
    hist_inflation_matrix = rng.normal(inflation_mean, inflation_std, (config_hist.num_runs, config_hist.duration_years))
    
    with st.spinner('Running Monte Carlo simulations...'):
        history_mc = run_simulation(config_mc, mc_return_matrix, mc_inflation_matrix)
    with st.spinner('Running Historic simulations...'):
        history_hist = run_simulation(config_hist, hist_return_matrix, hist_inflation_matrix)
        
    def render_results(history, config, num_runs, title, inflation_matrix, success_pct):
        st.header(title)
        net_worth_history = history['net_worth']
        
        final_net_worth = net_worth_history[-1, :]
        initial_nw = history['initial_net_worth']
        
        run_final_inflation_factor = np.prod(1 + inflation_matrix, axis=1)
        run_inf_adj_start_nw = initial_nw * run_final_inflation_factor
        target_ending_nw = (success_pct / 100.0) * run_inf_adj_start_nw
        success_mask = final_net_worth > target_ending_nw
        
        if success_pct == 0.0:
            tooltip_text = "Success is defined as ending net worth > 0 CHF (not going broke)."
        else:
            tooltip_text = f"Success is defined as ending net worth > {success_pct}% of the inflation-adjusted starting net worth."
            
        success_rate = np.mean(success_mask) * 100
        median_final = np.median(final_net_worth)
        
        final_nw_inf_adj = final_net_worth / run_final_inflation_factor
        median_final_inf_adj = np.median(final_nw_inf_adj)
        
        cum_inflation = np.cumprod(1 + inflation_matrix, axis=1)
        median_cum_inflation = np.median(cum_inflation, axis=0)
        inf_adj_start_nw_trajectory = initial_nw * median_cum_inflation
        st.markdown(f"**Nominal Starting NW:** {initial_nw:,.0f} CHF")
        
        rich_threshold = initial_nw * median_cum_inflation[-1] * 3.0
        
        if median_final <= 0:
            tldr_status = "🛑 **BROKE**"
        elif median_final >= rich_threshold:
            tldr_status = "🚀 **RICH**"
        else:
            tldr_status = "🪦 **DEAD**"
            
        st.markdown(f"### {tldr_status}")
        
        # Row 1: Success Rate & Watermark Metric
        col_r1_1, col_r1_2 = st.columns(2)
        col_r1_1.metric("Probability of Success", f"{success_rate:.1f}%", help=tooltip_text)
        
        avg_years_below_watermark = np.mean(np.sum(history['below_watermark'], axis=0))
        pct_years_below_watermark = (avg_years_below_watermark / config.duration_years) * 100.0
        col_r1_2.metric("Avg Years Below Start NW", f"{avg_years_below_watermark:.1f} ({pct_years_below_watermark:.1f}%)", help="Average number of years per simulation where the portfolio drops below the inflation-adjusted starting net worth. If the Dynamic Expense Floor feature is enabled, this is exactly equal to the number of times the expenses are reduced.")
        
        # Row 2: Median Ending NW (Real vs Nominal)
        col_r2_1, col_r2_2 = st.columns(2)
        col_r2_1.metric("Median Ending NW (Real)", f"{median_final_inf_adj:,.0f} CHF")
        col_r2_2.metric("Median Ending NW (Nominal)", f"{median_final:,.0f} CHF")
        
        # Plotly Chart
        years = np.arange(config.start_age + 1, config.start_age + config.duration_years + 1)
        fig = go.Figure()
        
        percentiles = [5, 25, 50, 75, 95]
        colors = ['crimson', 'orange', 'forestgreen', 'royalblue', 'purple']
        
        simulation_years = np.arange(1, config.duration_years + 1)
        
        max_traces_to_plot = int(num_runs) if "Historic" in title else min(100, int(num_runs))
        for i in range(max_traces_to_plot):
            if "Historic" in title:
                start_year = 1928 + i
                end_year = start_year + config.duration_years - 1
                trace_name = f"Cohort: {start_year} - {end_year}"
                custom_text = [f"Calendar Year: {start_year + y - 1}" for y in simulation_years]
            else:
                trace_name = f"Run {i+1}"
                custom_text = [f"Year N: {y}" for y in simulation_years]
                
            fig.add_trace(go.Scattergl(
                x=years, 
                y=net_worth_history[:, i],
                customdata=custom_text,
                mode='lines', 
                line=dict(color='#555555', width=0.5), 
                opacity=0.25, 
                showlegend=False, 
                hovertemplate="<b>" + trace_name + "</b><br>%{customdata} (Age: %{x})<br>Net Worth: %{y:,.0f} CHF<extra></extra>",
                name=trace_name
            ))
            
        for p, c in zip(percentiles, colors):
            p_vals = np.percentile(net_worth_history, p, axis=1)
            custom_text_pct = [f"Year N: {y}" for y in simulation_years]
            fig.add_trace(go.Scatter(
                x=years, 
                y=p_vals,
                customdata=custom_text_pct,
                mode='lines', 
                name=f'{p}th Pct', 
                line=dict(color=c, width=3 if p != 50 else 5),
                hovertemplate="<b>" + f"{p}th Percentile" + "</b><br>%{customdata} (Age: %{x})<br>Net Worth: %{y:,.0f} CHF<extra></extra>"
            ))
        
        # Add Inflation-Adjusted Starting Net Worth Reference Line
        fig.add_trace(go.Scatter(
            x=years, 
            y=inf_adj_start_nw_trajectory, 
            mode='lines', 
            name='Inflation-Adj Start NW', 
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate="<b>Inflation-Adj Start NW</b><br>Age: %{x}<br>Value: %{y:,.0f} CHF<extra></extra>"
        ))
            
        fig.update_layout(xaxis_title="Age", yaxis_title="Net Worth (CHF)", yaxis=dict(tickformat=",.0f"), hovermode="closest", margin=dict(t=15, b=40))
        st.subheader("Net Worth Trajectory", help="This chart displays the value of your assets over time.")
        st.plotly_chart(fig, width='stretch')

        # Income Breakdown Chart
        median_divs = np.median(history['income_dividends'], axis=1)
        median_ahv = np.median(history['income_ahv'], axis=1)
        median_expenses = np.median(history['expenses_paid'], axis=1)
        median_taxes = np.median(history['taxes_paid'], axis=1)
        
        capital_sold = np.maximum(0, median_expenses + median_taxes - median_divs - median_ahv)
        
        fig_income = go.Figure()
        fig_income.add_trace(go.Bar(x=years, y=median_divs, name='Dividends', marker_color='blue'))
        fig_income.add_trace(go.Bar(x=years, y=median_ahv, name='AHV Pension', marker_color='orange'))
        fig_income.add_trace(go.Bar(x=years, y=capital_sold, name='Capital Sold', marker_color='red'))
        
        fig_income.add_trace(go.Scatter(x=years, y=median_expenses + median_taxes, mode='lines', name='Total Cash Needed', line=dict(color='black', width=2, dash='dash')))
        
        fig_income.update_layout(xaxis_title="Age", yaxis_title="Amount (CHF)", barmode='stack', yaxis=dict(tickformat=",.0f"), hovermode="x", margin=dict(t=15, b=40))
        st.subheader("Income vs Required Cash", help="This chart displays the median cash flows across all simulated portfolio paths for each year.")
        st.plotly_chart(fig_income, width='stretch')

        # Withdrawal Rate Chart
        safe_net_worth = np.maximum(net_worth_history, 1.0)
        withdrawal_rate_history = ((history['expenses_paid'] + history['taxes_paid']) / safe_net_worth) * 100.0
        
        fig_wr = go.Figure()
        for p, c in zip(percentiles, colors):
            p_vals = np.percentile(withdrawal_rate_history, p, axis=1)
            # Cap values for visualization purposes when net worth approaches zero
            p_vals = np.minimum(p_vals, 100.0) 
            fig_wr.add_trace(go.Scatter(x=years, y=p_vals, mode='lines', name=f'{p}th Pct', line=dict(color=c, width=3 if p != 50 else 5)))
            
        fig_wr.update_layout(
            xaxis_title="Age", 
            yaxis_title="Withdrawal Rate (%)", 
            yaxis=dict(tickformat=".1f", range=[0, min(100, np.max(np.percentile(withdrawal_rate_history, 75, axis=1))*1.5 + 5)]), 
            hovermode="x",
            margin=dict(t=15, b=40)
        )
        st.subheader("Withdrawal Rate", help="This chart displays the percentage of your current net worth consumed by expenses and taxes each year.")
        st.plotly_chart(fig_wr, width='stretch')

        # Taxes Breakdown Chart
        fig2 = go.Figure(data=[
            go.Bar(name='Expenses', x=years, y=median_expenses, marker_color='blue'),
            go.Bar(name='Taxes', x=years, y=median_taxes, marker_color='red')
        ])
        fig2.update_layout(barmode='stack', xaxis_title="Age", yaxis_title="CHF", yaxis=dict(tickformat=",.0f"), margin=dict(t=15, b=40))
        st.subheader("Expenses & Taxes", help="This chart displays the median expenses and taxes paid across all simulated portfolio paths for each year.")
        st.plotly_chart(fig2, width='stretch')

        is_historic = "Historic" in title
        analysis_name = "Cohort Analysis" if is_historic else "Run Analysis"
        id_col_name = "Cohort" if is_historic else "Run"
        
        st.subheader(analysis_name, help=f"Best and worst paths based on the final net worth.")
        import pandas as pd
        
        final_nw = net_worth_history[-1, :]
        min_nw = np.min(net_worth_history, axis=0)
        years_below = np.sum(history['below_watermark'], axis=0)
        
        # Find best and worst indices efficiently
        sorted_indices = np.argsort(final_nw)
        worst_indices = sorted_indices[:10]
        best_indices = sorted_indices[::-1][:10]
        
        def build_df(indices):
            data = []
            for idx in indices:
                idx = int(idx)
                if is_historic:
                    cohort_year = 1928 + idx
                    run_id = f"{cohort_year} - {cohort_year + config.duration_years - 1}"
                else:
                    run_id = f"Run {idx + 1}"
                data.append({
                    id_col_name: run_id,
                    "Final NW (Real)": final_nw_inf_adj[idx],
                    "Final NW (Nom)": final_nw[idx],
                    "Min NW (Nom)": min_nw[idx],
                    "Yrs Below": int(years_below[idx])
                })
            return pd.DataFrame(data)
            
        st.markdown(f"##### Top 10 Best {id_col_name}s" if is_historic else "##### Top 10 Best Runs")
        df_best = build_df(best_indices)
        st.dataframe(df_best.style.format({
            "Final NW (Real)": "{:,.0f}",
            "Final NW (Nom)": "{:,.0f}",
            "Min NW (Nom)": "{:,.0f}"
        }), hide_index=True, width='stretch')
        
        st.markdown(f"##### Top 10 Worst {id_col_name}s" if is_historic else "##### Top 10 Worst Runs")
        df_worst = build_df(worst_indices)
        st.dataframe(df_worst.style.format({
            "Final NW (Real)": "{:,.0f}",
            "Final NW (Nom)": "{:,.0f}",
            "Min NW (Nom)": "{:,.0f}"
        }), hide_index=True, width='stretch')
            
    col_hist, col_mc = st.columns(2)
    with col_hist:
        render_results(history_hist, config_hist, hist_num_runs, "Historic Returns", hist_inflation_matrix, success_pct)
    with col_mc:
        render_results(history_mc, config_mc, mc_num_runs, "Monte Carlo", mc_inflation_matrix, success_pct)
else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation'.")
