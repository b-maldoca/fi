import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.historic_returns import (
    BITCOIN_NOMINAL_MEAN,
    BITCOIN_VOL,
    HISTORIC_REAL_CHF_APPRECIATION,
    HISTORIC_RETURNS_CASH_CHF,
    HISTORIC_RETURNS_GOLD_CHF,
    HISTORIC_RETURNS_NON_US_CHF,
    HISTORIC_RETURNS_US_CHF,
    HISTORIC_RETURNS_US_USD,
    HISTORIC_SWISS_INFLATION,
    HISTORIC_YEARS,
    POLITIS_WHITE_BLOCK_YEARS,
    compute_effective_sample_size,
    generate_bootstrapped_data,
    get_historic_inflation_matrix,
    get_historic_return_matrix,
)
from src.metrics import (
    beginning_of_year_withdrawal_rate,
    classify_outcome,
    compute_success_mask,
    cumulative_inflation,
    real_final_net_worth,
)
from src.simulation_engine import (
    SimConfig,
    estimate_year_0_taxes,
    generate_monte_carlo_inflation,
    generate_monte_carlo_returns,
    historic_lognormal_params,
    run_simulation,
)
from src.tax_engine import ZURICH_CANTONAL_MULTIPLIER, ZURICH_CITY_MUNICIPAL_MULTIPLIER

st.set_page_config(page_title="Zurich Early Retirement Simulator", layout="wide")

st.markdown("""
<style>
    /* Maximize content area by reducing side margins and padding */
    .block-container,
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    section.main > div.block-container {
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Tighten gap between columns */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }

    /* Condense sidebar internal horizontal padding to reduce grey space on left/right edges */
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] div.block-container,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
        padding-top: 1.25rem !important;
    }

    /* Minimize only empty space at the very top of the sidebar above the Configuration header */
    div[data-testid="stSidebarHeader"] {
        min-height: 0px !important;
        height: 0px !important;
        padding: 0px !important;
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 0.25rem !important;
    }
    section[data-testid="stSidebar"] h2:first-of-type,
    section[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"]:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Condense the sidebar vertical spacing without overlapping */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.6rem !important;
    }
    /* Condense headers but leave breathing room */
    section[data-testid="stSidebar"] h3 {
        padding-top: 0.75rem !important;
        padding-bottom: 0.2rem !important;
    }
    /* Condense dividers */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.4em !important;
        margin-bottom: 0.4em !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Zurich Early Retirement Simulator")

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
    min_value=0.0,
    step=5.0,
    format="%.1f",
    help="The percentage of inflation-adjusted starting net worth you want to preserve at the end of the simulation. 0.0% means you just want to avoid going broke (survival)."
)

# 2. Initial Assets
st.sidebar.subheader("Initial Assets (CHF)")
initial_liquid_wealth = st.sidebar.number_input("Taxable Liquid Wealth (CHF)", value=2_450_000, min_value=0, step=50_000, help="Your easily accessible taxable investments (stocks, bonds, cash). Do not include your primary residence.")
initial_pillar_2 = st.sidebar.number_input("Pillar 2 (Freizügigkeitskonto)", value=450_000, min_value=0, step=50_000, help="The current balance of your Swiss Pillar 2 pension. Assumed to be 100% invested in equities (proportional to your US vs Non-US target allocation). Withdrawn at age 65.")

num_pillar_3a = st.sidebar.number_input(
    "Number of Pillar 3a Accounts",
    value=5,
    min_value=0,
    max_value=5,
    help="How many separate Pillar 3a accounts you hold. Liquidated sequentially starting 5 years before age 65 to minimize taxes.",
)
pillar_3a_balance = st.sidebar.number_input(
    "Balance per Pillar 3a Account (CHF)",
    value=20_000,
    min_value=0,
    step=5_000,
    disabled=(int(num_pillar_3a) == 0),
    help="The balance of each Pillar 3a account (all accounts hold this equal amount). Assumed to be 100% invested in equities (proportional to your US vs Non-US target allocation).",
)
pillar_3a_accounts = [pillar_3a_balance] * int(num_pillar_3a)

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
rebalance_strategy = st.sidebar.selectbox("Frequency", ["Cash Tent", "Monthly", "Quarterly", "Yearly", "Threshold", "Never"], index=0, help="How often to rebalance your portfolio. 'Cash Tent' (Pfau Glidepath) assumes a cash buffer of [Tent Duration × (Annual Expenses + Estimated Taxes)] is already built up at retirement (Year 0) and glides down to your base cash target weight over time.")
rebalance_threshold = 0.0
tent_duration_years = 7

if rebalance_strategy == "Threshold":
    rebalance_threshold = st.sidebar.number_input("Threshold Drift (%)", value=1.0, min_value=0.0, step=0.1, format="%.1f", help="If rebalancing based on threshold, the max absolute drift allowed before rebalancing.") / 100.0
elif rebalance_strategy == "Cash Tent":
    tent_duration_years = int(st.sidebar.number_input("Tent Duration (Years)", value=7, min_value=1, max_value=30, step=1, help="Number of years of living expenses + estimated taxes held as cash buffer at retirement (Year 0), which glides down to your base cash target weight over the tent duration."))

st.sidebar.subheader("Withdrawal Strategy")
enable_smart_selling = st.sidebar.checkbox("Smart Cash Buffer", value=True, help="During market downturns (net worth < inflation-adjusted start), skip rebalancing and spend down Cash first. Only sell other assets if Cash is fully depleted. Once recovered, normal rebalancing resumes.")

# 5. Economics & Spending Strategy
st.sidebar.subheader("Spending Strategy")
spending_strategy = st.sidebar.selectbox(
    "Spending Model",
    ["Static", "Dynamic (Floor & Ceiling)", "Vanguard Dynamic"],
    index=2,
    help="Select how annual living expenses adjust over time:\n"
         "- Static: Base expenses grow strictly with inflation.\n"
         "- Dynamic (Floor & Ceiling): Base expenses decrease when net worth falls below the starting watermark, and increase when above.\n"
         "- Vanguard Dynamic: Spending is recalculated annually as a percentage of total portfolio net worth, bounded by a maximum cut (floor) and maximum raise (ceiling) relative to prior year's inflation-adjusted spending."
)

# Bi-directional synchronization for Vanguard Dynamic Spending
total_start_nw = initial_liquid_wealth + initial_pillar_2 + sum(pillar_3a_accounts)
cantonal_multiplier = ZURICH_CANTONAL_MULTIPLIER  # Zurich Cantonal Steuerfuss for 2026


def get_year0_taxes(exp: float) -> float:
    try:
        live_div_yield = float(st.session_state.get("dividend_yield_pct", 1.5)) / 100.0
        live_cash_rate = float(st.session_state.get("ret_cash_pct", 1.0)) / 100.0
        live_ahv = float(st.session_state.get("monthly_ahv_input", 2000))
        live_muni_mult = float(st.session_state.get("municipal_multiplier_pct", ZURICH_CITY_MUNICIPAL_MULTIPLIER * 100.0)) / 100.0
        temp_config = SimConfig(
            num_runs=1,
            duration_years=1,
            inflation_mean=0.025,
            inflation_std=0.01,
            start_age=int(start_age),
            dividend_yield=live_div_yield,
            cash_rate=live_cash_rate,
            initial_liquid_wealth=initial_liquid_wealth,
            initial_pillar_2=initial_pillar_2,
            initial_pillar_3a_accounts=pillar_3a_accounts,
            alloc_us_stocks=alloc_us / 100.0,
            alloc_non_us_stocks=alloc_non_us / 100.0,
            alloc_chf_cash=alloc_cash / 100.0,
            alloc_gold=alloc_gold / 100.0,
            alloc_bitcoin=alloc_btc / 100.0,
            rebalance_strategy='Never',
            rebalance_threshold=0.0,
            annual_base_expenses=exp,
            monthly_ahv_pension=live_ahv,
            cantonal_multiplier=cantonal_multiplier,
            municipal_multiplier=live_muni_mult
        )
        return estimate_year_0_taxes(temp_config)
    except ValueError:
        # Transiently invalid widget state (e.g. mid-edit); a real error surfaces
        # when the full SimConfig is built below. Anything else is a bug and must raise.
        return 0.0


if 'annual_expenses' not in st.session_state:
    st.session_state['annual_expenses'] = 85_000

_vanguard_sync_sig = (
    int(start_age),
    float(initial_liquid_wealth),
    float(initial_pillar_2),
    tuple(float(x) for x in pillar_3a_accounts),
    round(float(alloc_us), 2),
    round(float(alloc_non_us), 2),
    round(float(alloc_cash), 2),
    round(float(alloc_gold), 2),
    round(float(alloc_btc), 2),
    round(float(st.session_state.get("dividend_yield_pct", 1.5)), 2),
    round(float(st.session_state.get("ret_cash_pct", 1.0)), 2),
    round(float(st.session_state.get("monthly_ahv_input", 2000)), 2),
    round(float(st.session_state.get("municipal_multiplier_pct", 119.0)), 2),
)

if 'vanguard_target_rate_pct' not in st.session_state or st.session_state.get('_last_vanguard_sync_sig') != _vanguard_sync_sig:
    st.session_state['_last_vanguard_sync_sig'] = _vanguard_sync_sig
    if total_start_nw > 0:
        est_tax = get_year0_taxes(float(st.session_state['annual_expenses']))
        st.session_state['vanguard_target_rate_pct'] = round(((st.session_state['annual_expenses'] + est_tax) / total_start_nw) * 100.0, 2)
    else:
        st.session_state['vanguard_target_rate_pct'] = 3.5


def on_expenses_change():
    nw = total_start_nw
    if nw > 0:
        exp = float(st.session_state['annual_expenses'])
        est_tax = get_year0_taxes(exp)
        total_outflow = exp + est_tax
        st.session_state['vanguard_target_rate_pct'] = round((total_outflow / nw) * 100.0, 2)


def on_twr_change():
    nw = total_start_nw
    if nw > 0:
        rate = float(st.session_state['vanguard_target_rate_pct'])
        target_total_outflow = (rate / 100.0) * nw
        # Fixed point iteration to solve: living_exp + taxes(living_exp) = target_total_outflow
        living_exp = max(0.0, target_total_outflow - get_year0_taxes(float(st.session_state.get('annual_expenses', 85_000))))
        for _ in range(5):
            est_tax = get_year0_taxes(living_exp)
            living_exp = max(0.0, target_total_outflow - est_tax)
        st.session_state['annual_expenses'] = int(round(living_exp, -2))


annual_expenses = float(st.sidebar.number_input(
    "Annual Base Expenses (CHF)",
    key="annual_expenses",
    min_value=0,
    step=5000,
    format="%d",
    on_change=on_expenses_change,
    help="Your expected net living expenses (excluding taxes) in today's CHF. Under Vanguard Dynamic Spending, this is bi-directionally synchronized with Target Withdrawal Rate (%), accounting for estimated taxes."
))

dynamic_expense_floor_pct = 100.0
dynamic_expense_ceiling_pct = 100.0
vanguard_target_rate = (annual_expenses / total_start_nw) if total_start_nw > 0 else (st.session_state['vanguard_target_rate_pct'] / 100.0)
vanguard_floor_pct = 0.050
vanguard_ceiling_pct = 0.050

if spending_strategy == "Dynamic (Floor & Ceiling)":
    dynamic_expense_floor_pct = st.sidebar.number_input("Reduced Expense Floor (%)", value=85.0, min_value=0.0, step=1.0, format="%.1f", help="The percentage of your base expenses you will spend when your net worth is below the watermark.")
    dynamic_expense_ceiling_pct = st.sidebar.number_input("Expanded Expense Ceiling (%)", value=115.0, min_value=0.0, step=1.0, format="%.1f", help="The percentage of your base expenses you will spend when your net worth is above the watermark.")
elif spending_strategy == "Vanguard Dynamic":
    twr_pct = st.sidebar.number_input(
        "Target Withdrawal Rate (%)",
        key="vanguard_target_rate_pct",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        on_change=on_twr_change,
        help="Target annual total withdrawal rate (covering both net living expenses and estimated taxes) as a % of total portfolio net worth. Bi-directionally synchronized with Annual Base Expenses (CHF)."
    )
    # Convert tax-inclusive UI withdrawal rate into net living expense rate of net worth for SimConfig
    vanguard_target_rate = (annual_expenses / total_start_nw) if total_start_nw > 0 else (twr_pct / 100.0)
    vanguard_floor_pct = st.sidebar.number_input("Max Annual Cut / Floor (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f", help="Maximum allowable reduction in spending compared to prior year's inflation-adjusted spending (Vanguard default: 5.0%).") / 100.0
    vanguard_ceiling_pct = st.sidebar.number_input("Max Annual Raise / Ceiling (%)", value=5.0, min_value=0.0, step=0.5, format="%.1f", help="Maximum allowable increase in spending compared to prior year's inflation-adjusted spending (Vanguard default: 5.0%).") / 100.0

st.sidebar.subheader("Income & Yield")
monthly_ahv = st.sidebar.number_input("Expected Monthly AHV Pension from age 65 (CHF)", value=2000, min_value=0, step=100, key="monthly_ahv_input", help="The monthly AHV pension you expect to receive starting at age 65 (in today's CHF, adjusted annually for CPI inflation in the simulation).")
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", value=1.5, min_value=0.0, step=0.1, format="%.1f", key="dividend_yield_pct", help="Expected annual dividend yield of the portfolio.") / 100.0


def _cagr_pct(annual_returns: np.ndarray) -> float:
    return (float(np.prod(1.0 + annual_returns)) ** (1.0 / len(annual_returns)) - 1.0) * 100.0


_hist_span = f"{HISTORIC_YEARS[0]}–{HISTORIC_YEARS[-1]}"
_us_usd_cagr = _cagr_pct(HISTORIC_RETURNS_US_USD)
_us_chf_cagr = _cagr_pct(HISTORIC_RETURNS_US_CHF)
_exus_chf_cagr = _cagr_pct(HISTORIC_RETURNS_NON_US_CHF)
# Historic (arithmetic mean, log-vol) pairs in the exact parameterisation of generate_monte_carlo_returns.
_hist_us_mean, _hist_us_vol = historic_lognormal_params(HISTORIC_RETURNS_US_CHF)
_hist_exus_mean, _hist_exus_vol = historic_lognormal_params(HISTORIC_RETURNS_NON_US_CHF)
_hist_gold_mean, _hist_gold_vol = historic_lognormal_params(HISTORIC_RETURNS_GOLD_CHF)
_hist_cash_mean = float(np.mean(HISTORIC_RETURNS_CASH_CHF))
_hist_infl_mean = float(np.mean(HISTORIC_SWISS_INFLATION))
_hist_infl_std = float(np.std(HISTORIC_SWISS_INFLATION, ddof=1))
# One volatility input drives both equity sleeves, so default it to the average of the two histories.
_hist_eq_vol = (_hist_us_vol + _hist_exus_vol) / 2.0


def _hist_help(mean: float | None = None, vol: float | None = None) -> str:
    parts = []
    if mean is not None:
        parts.append(f"arithmetic mean {mean * 100:.1f}%")
    if vol is not None:
        parts.append(f"log-return volatility {vol * 100:.1f}%")
    return f" Historic {_hist_span} in CHF: " + ", ".join(parts) + "."


st.sidebar.subheader("Monte Carlo Parameters")
st.sidebar.caption(
    "Note: Returns and inflation must be Nominal (unadjusted for inflation) and in CHF terms, and the means are "
    f"arithmetic. For reference ({_hist_span}, geometric CAGR): US Stocks {_us_usd_cagr:.1f}% in USD but "
    f"{_us_chf_cagr:.1f}% in CHF due to currency drag; Non-US Stocks {_exus_chf_cagr:.1f}% in CHF. "
    "Stock means, volatilities and gold defaults are calibrated to that history (arithmetic means, log-return vols); "
    "CHF cash and inflation stay forward-looking (today's near-zero savings rates, SNB 0–2% target), with historic "
    "values in each field's help."
)
_min_ret_pct = -99.0  # generate_monte_carlo_returns requires returns > -100%
inflation_mean = st.sidebar.number_input("Inflation Mean (%)", value=2.5, min_value=_min_ret_pct, step=0.1, format="%.1f", help="Expected average annual inflation rate for Monte Carlo." + _hist_help(_hist_infl_mean)) / 100.0
inflation_std = st.sidebar.number_input("Inflation Volatility (%)", value=1.0, min_value=0.0, step=0.1, format="%.1f", help="Expected volatility of inflation for Monte Carlo." + f" Historic {_hist_span}: {_hist_infl_std * 100:.1f}% (dominated by the 1920s deflation and the 1940s/1970s inflation spikes).") / 100.0
ret_us = st.sidebar.number_input("US Stocks Nominal Mean (%)", value=round(_hist_us_mean * 100.0, 1), min_value=_min_ret_pct, step=0.1, format="%.1f", help="Expected nominal arithmetic mean return for US Stocks in CHF. Default = history." + _hist_help(_hist_us_mean) + f" (S&P 500 CAGR: {_us_usd_cagr:.1f}% in USD, {_us_chf_cagr:.1f}% in CHF.)") / 100.0
ret_non_us = st.sidebar.number_input("Non-US Stocks Nominal Mean (%)", value=round(_hist_exus_mean * 100.0, 1), min_value=_min_ret_pct, step=0.1, format="%.1f", help="Expected nominal arithmetic mean return for Non-US Stocks in CHF terms. Default = history." + _hist_help(_hist_exus_mean)) / 100.0
ret_cash = st.sidebar.number_input("CHF Cash Nominal Mean (%)", value=1.0, min_value=_min_ret_pct, step=0.1, format="%.1f", key="ret_cash_pct", help="Expected nominal mean return for CHF Cash (also used as the contractual taxable savings interest floor in Year 0 tax sync and Monte Carlo)." + _hist_help(_hist_cash_mean)) / 100.0
ret_gold = st.sidebar.number_input("Gold Nominal Mean (%)", value=round(_hist_gold_mean * 100.0, 1), min_value=_min_ret_pct, step=0.1, format="%.1f", help="Expected nominal arithmetic mean return for Gold in CHF terms. Default = history." + _hist_help(_hist_gold_mean)) / 100.0
ret_btc = st.sidebar.number_input(
    "Bitcoin Nominal Mean (%)",
    value=float(BITCOIN_NOMINAL_MEAN * 100.0),
    min_value=_min_ret_pct,
    step=0.1,
    format="%.1f",
    help="Expected nominal arithmetic mean return for synthetic Bitcoin in CHF terms (applied across Historic Backtesting, Stationary Bootstrapping, and Monte Carlo).",
) / 100.0

vol_eq = st.sidebar.number_input("Equities Volatility (%)", value=round(_hist_eq_vol * 100.0, 1), min_value=0.0, step=0.1, format="%.1f", help="Annualized log-return volatility applied to both US and Non-US Stocks. Default = average of the two histories." + f" Historic {_hist_span} in CHF: US {_hist_us_vol * 100:.1f}%, Non-US {_hist_exus_vol * 100:.1f}%.") / 100.0
vol_gold = st.sidebar.number_input("Gold Volatility (%)", value=round(_hist_gold_vol * 100.0, 1), min_value=0.0, step=0.1, format="%.1f", help="Annualized log-return volatility for Gold. Default = history." + _hist_help(vol=_hist_gold_vol)) / 100.0
vol_btc = st.sidebar.number_input(
    "Bitcoin Volatility (%)",
    value=float(BITCOIN_VOL * 100.0),
    min_value=0.0,
    step=0.1,
    format="%.1f",
    help="Expected annualized volatility for synthetic Bitcoin (applied across Historic Backtesting, Stationary Bootstrapping, and Monte Carlo).",
) / 100.0
btc_equity_corr = st.sidebar.slider(
    "Bitcoin–Equity Correlation (ρ)",
    min_value=-0.50,
    max_value=0.90,
    value=0.50,
    step=0.05,
    format="%.2f",
    help=(
        "Correlation between synthetic Bitcoin monthly log-returns and US Equity log-returns "
        "across all three engines (Historic Backtesting, Stationary Bootstrapping, and Monte Carlo). "
        "Default 0.50 reflects post-2020 institutional co-movement and prevents a 0-correlation "
        "rebalancing free lunch during equity crashes."
    ),
)

mc_num_runs = int(st.sidebar.number_input("Number of Monte Carlo Runs", value=1000, min_value=100, max_value=10000, step=100, help="How many distinct future paths to simulate."))
boot_num_runs = int(st.sidebar.number_input("Number of Bootstrapping Runs", value=1000, min_value=100, max_value=10000, step=100, help="How many Politis-Romano stationary block-bootstrap paths (with circular wrap-around) to simulate."))
boot_block_years = int(st.sidebar.slider(
    "Stationary Bootstrap Mean Block Length (Years)",
    min_value=1,
    max_value=15,
    value=int(POLITIS_WHITE_BLOCK_YEARS),
    step=1,
    help=(
        "Mean block length L (in years) for the Politis-Romano (1994) Stationary Bootstrap "
        "(geometric block lengths with transition probability p = 1/L and circular wrap-around). "
        "Default 5 years matches the Politis-White (2004) spectral optimal block length."
    ),
))
random_seed = int(st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1, help="Random seed for reproducible Monte Carlo, Bootstrapping, and synthetic asset simulation paths."))

st.sidebar.subheader("Currency / PPP Assumption")
real_chf_appreciation = st.sidebar.slider(
    "Real CHF Appreciation beyond PPP (%/yr)",
    min_value=-1.00,
    max_value=2.00,
    value=0.00,
    step=0.01,  # HISTORIC_REAL_CHF_APPRECIATION is rounded to 0.01pp and must be selectable
    format="%.2f%%",
    help=(
        "Expected long-run real appreciation of the Swiss Franc beyond inflation differentials "
        "(Relative Purchasing Power Parity) applied to foreign-priced sleeves (US Stocks, Non-US Stocks, Gold) "
        "in Historic Backtesting and Block Bootstrapping.\n\n"
        "- 0.00% (default): Relative PPP holds going forward (no excess real currency drag).\n"
        f"- +{HISTORIC_REAL_CHF_APPRECIATION * 100:.2f}%: Reproduces the raw {_hist_span} historical real CHF "
        "appreciation beyond US-CH CPI differentials."
    ),
) / 100.0

try:
    hist_return_matrix = get_historic_return_matrix(
        int(duration),
        seed=random_seed,
        real_chf_appreciation=real_chf_appreciation,
        btc_equity_corr=btc_equity_corr,
        btc_mean=ret_btc,
        btc_vol=vol_btc,
    )
    hist_num_runs = hist_return_matrix.shape[0]
    st.sidebar.info(f"Using {len(HISTORIC_YEARS)}-year historic Swiss market data ({HISTORIC_YEARS[0]}–{HISTORIC_YEARS[-1]}). Available contiguous cohorts: {hist_num_runs}")
except ValueError as e:
    st.sidebar.error(str(e))
    hist_return_matrix = None
    hist_num_runs = 0

# 6. Tax Location
st.sidebar.subheader("Zurich Tax Location")
municipal_multiplier = st.sidebar.number_input("Municipal Multiplier (Steuerfuss, %)", value=ZURICH_CITY_MUNICIPAL_MULTIPLIER * 100.0, min_value=0.0, step=0.1, format="%.1f", key="municipal_multiplier_pct", help="Your municipal tax multiplier (Steuerfuss) in Zurich (e.g. 119% for City of Zurich).") / 100.0
bracket_indexation = st.sidebar.number_input(
    "Tax Bracket Indexation (% of CPI)",
    value=100.0,
    min_value=0.0,
    max_value=100.0,
    step=5.0,
    format="%.0f",
    help="How much of each year's inflation is passed through to the tax bracket edges and the AHV contribution table.\n\n"
         "- 100%: Full indexation. This is what Swiss law mandates — Art. 39 DBG requires the federal tariff to be adjusted to the CPI annually, and § 48 StG ZH indexes the Zurich income and wealth tariffs.\n"
         "- 0%: Brackets frozen in nominal terms, so inflation alone pushes you into higher brackets ('kalte Progression'). A worst case, not the legal default.\n\n"
         "Note that the Steuerfuss multipliers are political and never indexed. A portfolio growing faster than inflation still climbs the wealth tax schedule in real terms at any setting."
) / 100.0

if hist_num_runs == 0 or hist_return_matrix is None:
    st.error("Cannot run simulation. Duration is too long for the available historic data.")
    st.stop()


def create_config(num_runs: int) -> SimConfig:
    return SimConfig(
        num_runs=num_runs,
        duration_years=int(duration),
        inflation_mean=inflation_mean,
        inflation_std=inflation_std,
        start_age=int(start_age),
        dividend_yield=dividend_yield,
        spending_strategy=spending_strategy,
        dynamic_expense_floor_pct=dynamic_expense_floor_pct / 100.0,
        dynamic_expense_ceiling_pct=dynamic_expense_ceiling_pct / 100.0,
        vanguard_target_rate=vanguard_target_rate,
        vanguard_floor_pct=vanguard_floor_pct,
        vanguard_ceiling_pct=vanguard_ceiling_pct,
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
        enable_smart_selling=enable_smart_selling,
        annual_base_expenses=annual_expenses,
        monthly_ahv_pension=monthly_ahv,
        cantonal_multiplier=cantonal_multiplier,
        municipal_multiplier=municipal_multiplier,
        bracket_indexation=bracket_indexation,
        tent_duration_years=tent_duration_years,
        real_chf_appreciation=real_chf_appreciation,
        cash_rate=ret_cash,
        btc_equity_corr=btc_equity_corr,
        seed=random_seed
    )


config_mc = create_config(mc_num_runs)
config_boot = create_config(boot_num_runs)
config_hist = create_config(hist_num_runs)

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
    seed=random_seed,
    btc_equity_corr=btc_equity_corr,
)
mc_inflation_matrix = generate_monte_carlo_inflation(
    num_runs=config_mc.num_runs,
    duration_years=config_mc.duration_years,
    inflation_mean=inflation_mean,
    inflation_std=inflation_std,
    seed=random_seed
)

boot_return_matrix, boot_inflation_matrix = generate_bootstrapped_data(
    num_runs=config_boot.num_runs,
    duration_years=config_boot.duration_years,
    seed=random_seed,
    block_size_years=boot_block_years,
    real_chf_appreciation=real_chf_appreciation,
    btc_equity_corr=btc_equity_corr,
    btc_mean=ret_btc,
    btc_vol=vol_btc,
    stationary=True,
)

hist_inflation_matrix = get_historic_inflation_matrix(config_hist.duration_years)

with st.spinner('Running Monte Carlo simulations...'):
    history_mc = run_simulation(config_mc, mc_return_matrix, mc_inflation_matrix)
with st.spinner('Running Bootstrapping simulations...'):
    history_boot = run_simulation(config_boot, boot_return_matrix, boot_inflation_matrix)
with st.spinner('Running Historic Backtesting simulations...'):
    history_hist = run_simulation(config_hist, hist_return_matrix, hist_inflation_matrix)


def _build_hist_eff_caption(history, config, inflation_matrix, success_pct) -> str:
    success_mask = compute_success_mask(
        history['net_worth'][-1, :], history['initial_net_worth'], inflation_matrix, success_pct
    )
    eff_stats = compute_effective_sample_size(
        len(HISTORIC_YEARS), config.duration_years, float(np.mean(success_mask) * 100.0)
    )
    return (
        f"⚠️ <b>Effective Sample Size (<i>N</i><sub>eff</sub>): {eff_stats['n_eff']:.1f}</b> independent cohorts "
        f"(from {int(eff_stats['n_cohorts'])} overlapping {config.duration_years}y windows) · "
        f"<b>90% CI: {eff_stats['ci_low_pct']:.0f}%–{eff_stats['ci_high_pct']:.0f}%</b>"
    )


_TLDR_BADGES = {"BROKE": "🛑 **BROKE**", "RICH": "🚀 **RICH**", "DEAD": "🪦 **DEAD**"}


def render_results(history, config, num_runs, title, inflation_matrix, success_pct, hist_eff_caption: str = ""):
    header_tooltips = {
        "Historic Backtesting": f"Replays exact contiguous historical sequences (e.g. 1928–1978, 1929–1979) from {len(HISTORIC_YEARS)} years of historical Swiss-adjusted market data. Because rolling multi-decade cohorts overlap heavily, effective sample size N_eff = T / D is surfaced alongside a 90% Wilson confidence band, and tail lines show Worst/Best Cohort instead of unsupported 5th/95th percentiles.",
        "Historic Bootstrapping": "Creates thousands of distinct retirement scenarios using the Politis–Romano (1994) Stationary Bootstrap with circular wrap-around and geometric block lengths (default mean 5 years). Preserves intra-year and multi-year macroeconomic cycles while sampling all historical years with equal 1/T probability.",
        "Monte Carlo": "Generates thousands of stochastic future paths using correlated parametric lognormal distributions based on user-configured nominal means, standard deviations, and inflation parameters."
    }
    st.header(title, help=header_tooltips.get(title))
    net_worth_history = history['net_worth']
    is_historic_backtest = "Historic Backtesting" in title

    final_net_worth = net_worth_history[-1, :]
    initial_nw = history['initial_net_worth']

    success_mask = compute_success_mask(final_net_worth, initial_nw, inflation_matrix, success_pct)

    if success_pct == 0.0:
        tooltip_text = "Success is defined as ending net worth > 0 CHF (not going broke)."
    else:
        tooltip_text = f"Success is defined as ending net worth > {success_pct}% of the inflation-adjusted starting net worth."

    success_rate = np.mean(success_mask) * 100
    median_final = np.median(final_net_worth)

    if is_historic_backtest:
        eff_stats = compute_effective_sample_size(
            len(HISTORIC_YEARS), config.duration_years, success_rate
        )
        tooltip_text += (
            f" Effective independent sample size N_eff = {eff_stats['n_eff']:.1f} "
            f"(from {int(eff_stats['n_cohorts'])} overlapping {config.duration_years}y windows). "
            f"90% Wilson confidence interval: {eff_stats['ci_low_pct']:.0f}%–{eff_stats['ci_high_pct']:.0f}%."
        )

    final_nw_inf_adj = real_final_net_worth(final_net_worth, inflation_matrix)
    median_final_inf_adj = np.median(final_nw_inf_adj)

    cum_inflation = cumulative_inflation(inflation_matrix)
    median_cum_inflation = np.median(cum_inflation, axis=0)
    inf_adj_start_nw_trajectory = initial_nw * median_cum_inflation
    st.markdown(f"**Nominal Starting NW:** {initial_nw:,.0f} CHF")
    if hist_eff_caption:
        visibility_style = "visible" if is_historic_backtest else "hidden"
        st.markdown(
            f'<div style="font-size: 0.875rem; color: rgba(49, 51, 63, 0.75); line-height: 1.4; '
            f'margin-top: -0.25rem; margin-bottom: 0.25rem; visibility: {visibility_style};" '
            f'aria-hidden="{"false" if is_historic_backtest else "true"}">'
            f'{hist_eff_caption}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"### {_TLDR_BADGES[classify_outcome(final_net_worth, initial_nw, inflation_matrix)]}")

    # Row 1: Success Rate & Watermark Metric
    col_r1_1, col_r1_2 = st.columns(2)
    col_r1_1.metric("Probability of Success", f"{success_rate:.1f}%", help=tooltip_text)

    avg_years_below_watermark = np.mean(np.sum(history['below_watermark'], axis=0))
    pct_years_below_watermark = (avg_years_below_watermark / config.duration_years) * 100.0
    col_r1_2.metric("Avg Years Below Start NW", f"{avg_years_below_watermark:.1f} ({pct_years_below_watermark:.1f}%)", help="Average number of years per simulation where the portfolio drops below the inflation-adjusted starting net worth. If the Dynamic Expense Floor feature is enabled, this is exactly equal to the number of times the expenses are reduced.")

    total_outflows_per_run = np.sum(history['expenses_paid'] + history['taxes_paid'], axis=0)
    median_total_withdrawals = np.median(total_outflows_per_run)

    total_outflows_real_per_run = np.sum((history['expenses_paid'] + history['taxes_paid']) / cum_inflation.T, axis=0)
    median_total_withdrawals_real = np.median(total_outflows_real_per_run)

    # Pre-65 (<65) vs Post-65 (>=65) breakdown
    sim_ages = np.arange(config.start_age, config.start_age + config.duration_years)
    pre_65_mask = sim_ages < 65
    post_65_mask = sim_ages >= 65

    if np.any(pre_65_mask):
        pre_65_outflows = np.sum(history['expenses_paid'][pre_65_mask, :] + history['taxes_paid'][pre_65_mask, :], axis=0)
        median_pre_65_nominal = np.median(pre_65_outflows)
        pre_65_outflows_real = np.sum((history['expenses_paid'][pre_65_mask, :] + history['taxes_paid'][pre_65_mask, :]) / cum_inflation.T[pre_65_mask, :], axis=0)
        median_pre_65_real = np.median(pre_65_outflows_real)
    else:
        median_pre_65_nominal = 0.0
        median_pre_65_real = 0.0

    if np.any(post_65_mask):
        post_65_outflows = np.sum(history['expenses_paid'][post_65_mask, :] + history['taxes_paid'][post_65_mask, :], axis=0)
        median_post_65_nominal = np.median(post_65_outflows)
        post_65_outflows_real = np.sum((history['expenses_paid'][post_65_mask, :] + history['taxes_paid'][post_65_mask, :]) / cum_inflation.T[post_65_mask, :], axis=0)
        median_post_65_real = np.median(post_65_outflows_real)
    else:
        median_post_65_nominal = 0.0
        median_post_65_real = 0.0

    # Row 2: Median Ending NW (Real vs Nominal)
    col_r2_1, col_r2_2 = st.columns(2)
    col_r2_1.metric("Median Ending NW (Real)", f"{median_final_inf_adj:,.0f} CHF")
    col_r2_2.metric("Median Ending NW (Nominal)", f"{median_final:,.0f} CHF")

    # Row 3: Median Total Cumulative Withdrawals (Real vs Nominal)
    col_r3_1, col_r3_2 = st.columns(2)
    col_r3_1.metric("Median Total Withdrawals (Real)", f"{median_total_withdrawals_real:,.0f} CHF", help="Total cumulative real purchasing power spent on living expenses and taxes over the full simulation period.")
    col_r3_2.metric("Median Total Withdrawals (Nominal)", f"{median_total_withdrawals:,.0f} CHF", help="Total cumulative nominal cash spent on living expenses and taxes over the full simulation period.")

    # Row 4: Pre-AHV (<65) vs Post-65 (>=65) Outflows
    col_r4_1, col_r4_2 = st.columns(2)
    pre_65_years_cnt = int(np.sum(pre_65_mask))
    post_65_years_cnt = int(np.sum(post_65_mask))
    col_r4_1.metric(
        f"Pre-AHV Outflow (< Age 65, {pre_65_years_cnt}y)",
        f"{median_pre_65_real:,.0f} CHF",
        help=f"Median total money required to cover living expenses and taxes during the early retirement gap before age 65 ({pre_65_years_cnt} years). Nominal: {median_pre_65_nominal:,.0f} CHF."
    )
    col_r4_2.metric(
        f"Post-65 Outflow (Age 65+, {post_65_years_cnt}y)",
        f"{median_post_65_real:,.0f} CHF",
        help=f"Median total money required to cover living expenses and taxes from age 65 through end-of-life ({post_65_years_cnt} years). Nominal: {median_post_65_nominal:,.0f} CHF."
    )

    # Plotly Chart
    years = np.arange(config.start_age + 1, config.start_age + config.duration_years + 1)
    fig = go.Figure()

    if is_historic_backtest:
        # With N_eff ~ 2.1–2.6 independent cohorts, P5/P95 are statistically unsupported;
        # show Empirical Min (Worst Cohort), quartiles (25th/50th/75th), and Empirical Max (Best Cohort).
        percentiles = [0, 25, 50, 75, 100]
        pct_labels = ['Worst Cohort (Min)', '25th Pct', '50th Pct', '75th Pct', 'Best Cohort (Max)']
    else:
        percentiles = [5, 25, 50, 75, 95]
        pct_labels = ['5th Pct', '25th Pct', '50th Pct', '75th Pct', '95th Pct']
    colors = ['crimson', 'orange', 'forestgreen', 'royalblue', 'purple']

    simulation_years = np.arange(1, config.duration_years + 1)

    max_traces_to_plot = int(num_runs) if is_historic_backtest else min(100, int(num_runs))
    for i in range(max_traces_to_plot):
        if is_historic_backtest:
            start_year = int(HISTORIC_YEARS[i])
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

    for p, label, c in zip(percentiles, pct_labels, colors):
        p_vals = np.percentile(net_worth_history, p, axis=1)
        custom_text_pct = [f"Year N: {y}" for y in simulation_years]
        fig.add_trace(go.Scatter(
            x=years,
            y=p_vals,
            customdata=custom_text_pct,
            mode='lines',
            name=label,
            line=dict(color=c, width=3 if p != 50 else 5),
            hovertemplate="<b>" + label + "</b><br>%{customdata} (Age: %{x})<br>Net Worth: %{y:,.0f} CHF<extra></extra>"
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

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Net Worth (CHF)",
        yaxis=dict(tickformat=",.0f"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
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

    fig_income.update_layout(
        xaxis_title="Age",
        yaxis_title="Amount (CHF)",
        barmode='stack',
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Income vs Required Cash", help="This chart displays the median cash flows across all simulated portfolio paths for each year.")
    st.plotly_chart(fig_income, width='stretch')

    # Annual Withdrawal Breakdown Chart
    fig_withdrawal = go.Figure(data=[
        go.Bar(name='Living Expenses', x=years, y=median_expenses, marker_color='royalblue'),
        go.Bar(name='Taxes Paid', x=years, y=median_taxes, marker_color='crimson')
    ])
    fig_withdrawal.update_layout(
        barmode='stack',
        xaxis_title="Age",
        yaxis_title="Annual Withdrawal (CHF)",
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Annual Withdrawal Breakdown", help="This stacked chart displays the median annual withdrawals (living expenses and taxes paid) across all simulated portfolio paths over time.")
    st.plotly_chart(fig_withdrawal, width='stretch')

    # Withdrawal Rate Chart (relative to beginning-of-year portfolio value)
    withdrawal_rate_history = beginning_of_year_withdrawal_rate(history)

    fig_wr = go.Figure()
    max_p_val = 5.0
    for p, label, c in zip(percentiles, pct_labels, colors):
        p_vals = np.percentile(withdrawal_rate_history, p, axis=1)
        # Cap values for visualization purposes when net worth approaches zero
        p_vals = np.minimum(p_vals, 100.0)
        max_p_val = max(max_p_val, float(np.max(p_vals)))
        fig_wr.add_trace(go.Scatter(x=years, y=p_vals, mode='lines', name=label, line=dict(color=c, width=3 if p != 50 else 5)))

    y_upper = min(25.0, max_p_val * 1.15)
    fig_wr.update_layout(
        xaxis_title="Age",
        yaxis_title="Withdrawal Rate (%)",
        yaxis=dict(tickformat=".1f", range=[0, y_upper]),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Withdrawal Rate", help="This chart displays the percentage of your current net worth consumed by expenses and taxes each year.")
    st.plotly_chart(fig_wr, width='stretch')

    # Asset Allocation Development Chart (Liquid Assets + Pillar 2 + Pillar 3a)
    median_assets_by_class = np.median(history['liquid_assets_by_class'], axis=1)  # shape: (years, 5)
    median_p2 = np.median(history['pillar_2'], axis=1) if 'pillar_2' in history else np.zeros(len(years))
    median_p3a = np.median(history['pillar_3a'], axis=1) if 'pillar_3a' in history else np.zeros(len(years))

    fig_alloc = go.Figure()

    all_series = [
        ('CHF Cash', median_assets_by_class[:, 2], 'forestgreen'),
        ('US Stocks', median_assets_by_class[:, 0], 'royalblue'),
        ('Non-US Stocks', median_assets_by_class[:, 1], 'darkcyan'),
        ('Gold', median_assets_by_class[:, 3], 'gold'),
        ('Bitcoin', median_assets_by_class[:, 4], 'purple'),
        ('Pillar 3a', median_p3a, 'mediumpurple'),
        ('Pillar 2', median_p2, 'darkorange'),
    ]

    for name, values, color in all_series:
        if np.max(values) > 0:
            fig_alloc.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines',
                name=name,
                stackgroup='one',
                line=dict(width=0.5),
                marker=dict(color=color)
            ))

    fig_alloc.update_layout(
        xaxis_title="Age",
        yaxis_title="Asset Value (CHF)",
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Asset Allocation Development", help="This stacked chart visualizes the median nominal balance of all asset categories (taxable liquid investments, Pillar 2, and Pillar 3a) over time, showing how your net worth breakdown glides and rebalances throughout retirement.")
    st.plotly_chart(fig_alloc, width='stretch')

    analysis_name = "Cohort Analysis" if is_historic_backtest else "Run Analysis"
    id_col_name = "Cohort" if is_historic_backtest else "Run"

    st.subheader(analysis_name, help="Best and worst paths based on the final net worth.")

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
            if is_historic_backtest:
                cohort_year = int(HISTORIC_YEARS[idx])
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

    st.markdown(f"##### Top 10 Best {id_col_name}s" if is_historic_backtest else "##### Top 10 Best Runs")
    df_best = build_df(best_indices)
    st.dataframe(df_best.style.format({
        "Final NW (Real)": "{:,.0f}",
        "Final NW (Nom)": "{:,.0f}",
        "Min NW (Nom)": "{:,.0f}"
    }), hide_index=True, width='stretch')

    st.markdown(f"##### Top 10 Worst {id_col_name}s" if is_historic_backtest else "##### Top 10 Worst Runs")
    df_worst = build_df(worst_indices)
    st.dataframe(df_worst.style.format({
        "Final NW (Real)": "{:,.0f}",
        "Final NW (Nom)": "{:,.0f}",
        "Min NW (Nom)": "{:,.0f}"
    }), hide_index=True, width='stretch')


hist_eff_caption = _build_hist_eff_caption(history_hist, config_hist, hist_inflation_matrix, success_pct)
col_hist, col_boot, col_mc = st.columns(3)
with col_hist:
    render_results(history_hist, config_hist, hist_num_runs, "Historic Backtesting", hist_inflation_matrix, success_pct, hist_eff_caption)
with col_boot:
    render_results(history_boot, config_boot, boot_num_runs, "Historic Bootstrapping", boot_inflation_matrix, success_pct, hist_eff_caption)
with col_mc:
    render_results(history_mc, config_mc, mc_num_runs, "Monte Carlo", mc_inflation_matrix, success_pct, hist_eff_caption)
