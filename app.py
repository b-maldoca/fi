from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.historic_returns as _hr_chf
import src.historic_returns_eur as _hr_eur
from src.historic_returns import (
    BITCOIN_NOMINAL_MEAN,
    BITCOIN_VOL,
    HISTORIC_RETURNS_US_USD,
    HISTORIC_YEARS,
    POLITIS_WHITE_BLOCK_YEARS,
    compute_effective_sample_size,
)
from src.jurisdiction import COUNTRY_DEFAULTS
from src.metrics import (
    RICH_MULTIPLE,
    beginning_of_year_withdrawal_rate,
    classify_outcome,
    classify_path_outcomes,
    compute_success_mask,
    compute_survival_curves,
    cumulative_inflation,
    format_depletion_label,
    format_shortfall_label,
    min_real_net_worth,
    real_final_net_worth,
    summarize_failure_timing,
    worst_path_order,
    years_in_shortfall,
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
from src.tax_engine_at import HEALTH_SELF_INSURANCE_MONTHLY

# Historic return data per country. Both generated modules share the calendar and the function
# API; only the series names and the PPP keyword differ (CHF: real_chf_appreciation, EUR:
# real_eur_appreciation). SimConfig.real_chf_appreciation carries the home-currency value.
_HISTORY = {
    "CH": SimpleNamespace(
        module=_hr_chf,
        us=_hr_chf.HISTORIC_RETURNS_US_CHF,
        non_us=_hr_chf.HISTORIC_RETURNS_NON_US_CHF,
        cash=_hr_chf.HISTORIC_RETURNS_CASH_CHF,
        gold=_hr_chf.HISTORIC_RETURNS_GOLD_CHF,
        inflation=_hr_chf.HISTORIC_SWISS_INFLATION,
        real_appreciation=_hr_chf.HISTORIC_REAL_CHF_APPRECIATION,
        ppp_kw="real_chf_appreciation",
        currency_name="Swiss Franc",
        data_label="Swiss market data",
    ),
    "AT": SimpleNamespace(
        module=_hr_eur,
        us=_hr_eur.HISTORIC_RETURNS_US_EUR,
        non_us=_hr_eur.HISTORIC_RETURNS_NON_US_EUR,
        cash=_hr_eur.HISTORIC_RETURNS_CASH_EUR,
        gold=_hr_eur.HISTORIC_RETURNS_GOLD_EUR,
        inflation=_hr_eur.HISTORIC_AUSTRIAN_INFLATION,
        real_appreciation=_hr_eur.HISTORIC_REAL_EUR_APPRECIATION,
        ppp_kw="real_eur_appreciation",
        currency_name="Euro",
        data_label="EUR (Austria) market data",
    ),
}
COUNTRY_LABELS = {"Switzerland (Zurich)": "CH", "Austria": "AT"}
SWISS_PROXY_NOTE = (
    f"Austrian history: {HISTORIC_YEARS[0]}–{_hr_eur.SWISS_PROXY_LAST_YEAR} uses Swiss inflation and Swiss "
    "savings-rate data (no usable Austrian data exists) with a PPP-implied EUR/USD rate; "
    f"{_hr_eur.SWISS_PROXY_LAST_YEAR + 1}–{_hr_eur.CASH_SWISS_PROXY_LAST_YEAR} cash is the Swiss savings rate "
    "translated to EUR under relative PPP. Applies to Historic Backtesting, Bootstrapping and the Monte Carlo defaults."
)


@st.cache_data(show_spinner=False)
def _run_cached_mc(
    config: SimConfig,
    ret_us: float,
    ret_non_us: float,
    ret_cash: float,
    ret_gold: float,
    ret_btc: float,
    vol_eq: float,
    vol_gold: float,
    vol_btc: float,
) -> tuple[dict, np.ndarray]:
    ret_mat = generate_monte_carlo_returns(
        num_runs=config.num_runs,
        duration_years=config.duration_years,
        ret_us=ret_us,
        ret_non_us=ret_non_us,
        ret_cash=ret_cash,
        ret_gold=ret_gold,
        ret_btc=ret_btc,
        vol_eq=vol_eq,
        vol_gold=vol_gold,
        vol_btc=vol_btc,
        seed=config.seed,
        btc_equity_corr=config.btc_equity_corr,
    )
    inf_mat = generate_monte_carlo_inflation(
        num_runs=config.num_runs,
        duration_years=config.duration_years,
        inflation_mean=config.inflation_mean,
        inflation_std=config.inflation_std,
        seed=config.seed,
    )
    return run_simulation(config, ret_mat, inf_mat), inf_mat


@st.cache_data(show_spinner=False)
def _run_cached_boot(
    config: SimConfig,
    block_size_years: int,
    ret_btc: float,
    vol_btc: float,
) -> tuple[dict, np.ndarray]:
    hist = _HISTORY[config.country]
    ret_mat, inf_mat = hist.module.generate_bootstrapped_data(
        num_runs=config.num_runs,
        duration_years=config.duration_years,
        seed=config.seed,
        block_size_years=block_size_years,
        btc_equity_corr=config.btc_equity_corr,
        btc_mean=ret_btc,
        btc_vol=vol_btc,
        stationary=True,
        **{hist.ppp_kw: config.real_chf_appreciation},
    )
    return run_simulation(config, ret_mat, inf_mat), inf_mat


@st.cache_data(show_spinner=False)
def _run_cached_hist(
    config: SimConfig,
    ret_btc: float,
    vol_btc: float,
) -> tuple[dict, np.ndarray]:
    hist = _HISTORY[config.country]
    ret_mat = hist.module.get_historic_return_matrix(
        config.duration_years,
        seed=config.seed,
        btc_equity_corr=config.btc_equity_corr,
        btc_mean=ret_btc,
        btc_vol=vol_btc,
        **{hist.ppp_kw: config.real_chf_appreciation},
    )
    inf_mat = hist.module.get_historic_inflation_matrix(config.duration_years)
    return run_simulation(config, ret_mat, inf_mat), inf_mat

st.set_page_config(page_title="Early Retirement Simulator", layout="wide")

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

st.sidebar.header("Configuration")
st.sidebar.markdown("---")

country_label = st.sidebar.selectbox(
    "Country",
    list(COUNTRY_LABELS),
    key="country_select",
    help=(
        "Tax jurisdiction and home currency.\n\n"
        "- Switzerland (Zurich): CHF, Federal + Canton Zurich income and wealth tax, AHV, Pillar 2 / 3a.\n"
        "- Austria: EUR, 27.5% capital gains / dividend tax (KESt) with the Regelbesteuerungsoption, "
        "progressive tax on the state pension, mandatory health self-insurance until the pension starts; "
        "no wealth tax, no Pillar 2 / 3a. "
        + SWISS_PROXY_NOTE
    ),
)
COUNTRY = COUNTRY_LABELS[country_label]
IS_AT = COUNTRY == "AT"
CDEF = COUNTRY_DEFAULTS[COUNTRY]
CCY = CDEF.currency
HIST = _HISTORY[COUNTRY]
PENSION_LABEL = "State Pension" if IS_AT else "AHV Pension"
PRE_PENSION = "Pre-Pension" if IS_AT else "Pre-AHV"
CPI_NAME = "Austrian CPI (Swiss CPI before 1958)" if IS_AT else "Swiss CPI"
CCY_NAME = "euros" if IS_AT else "Swiss Francs"

# Country-dependent widget defaults are (re)initialised through Session State when the country
# changes (and on first load), so switching never leaves e.g. the CHF cash rate in the EUR model.
if st.session_state.get("_last_country") != COUNTRY:
    st.session_state["_last_country"] = COUNTRY
    st.session_state["ret_cash_pct"] = round(CDEF.cash_rate * 100.0, 1)
    st.session_state["bracket_indexation_pct"] = round(CDEF.bracket_indexation * 100.0, 1)
    st.session_state["_last_ppp_for_mc"] = None
    for _flag in ("_user_edited_mc_ret_us", "_user_edited_mc_ret_exus", "_user_edited_mc_ret_gold"):
        st.session_state[_flag] = False

st.title("Austria Early Retirement Simulator" if IS_AT else "Zurich Early Retirement Simulator")
if IS_AT:
    st.warning("⚠️ " + SWISS_PROXY_NOTE)

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
st.sidebar.subheader(f"Initial Assets ({CCY})")
initial_liquid_wealth = st.sidebar.number_input(f"Taxable Liquid Wealth ({CCY})", value=2_450_000, min_value=0, step=50_000, help="Your easily accessible taxable investments (stocks, bonds, cash). Do not include your primary residence.")
if IS_AT:
    # Swiss-only pension vehicles (Austrian occupational pensions are out of scope).
    initial_pillar_2, num_pillar_3a, pillar_3a_balance = 0, 0, 0
else:
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
alloc_cash = st.sidebar.number_input(f"{CCY} Cash", value=10.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help=f"Percentage of your portfolio held in {CCY} Cash.")
alloc_gold = st.sidebar.number_input("Gold", value=3.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Percentage of your portfolio held in Gold." + (" Held as a gold ETC: gains are taxed at 27.5% like other capital income." if IS_AT else ""))
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
enable_smart_selling = st.sidebar.checkbox("Smart Cash Buffer", value=True, help="During market downturns (after-tax net worth < inflation-adjusted after-tax start), skip rebalancing and spend down Cash first. Only sell other assets if Cash is fully depleted. Once recovered, normal rebalancing resumes.")

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
        live_cash_rate = float(st.session_state.get("ret_cash_pct", CDEF.cash_rate * 100.0)) / 100.0
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
            rebalance_strategy=rebalance_strategy,
            rebalance_threshold=rebalance_threshold,
            tent_duration_years=tent_duration_years,
            annual_base_expenses=exp,
            monthly_ahv_pension=live_ahv,
            cantonal_multiplier=cantonal_multiplier,
            municipal_multiplier=live_muni_mult,
            country=COUNTRY,
            health_insurance_monthly=HEALTH_SELF_INSURANCE_MONTHLY,
        )
        return estimate_year_0_taxes(temp_config)
    except ValueError:
        # Transiently invalid widget state (e.g. mid-edit); a real error surfaces
        # when the full SimConfig is built below. Anything else is a bug and must raise.
        return 0.0


if 'annual_expenses' not in st.session_state:
    st.session_state['annual_expenses'] = 85_000

_vanguard_sync_sig = (
    COUNTRY,
    int(start_age),
    float(initial_liquid_wealth),
    float(initial_pillar_2),
    tuple(float(x) for x in pillar_3a_accounts),
    round(float(alloc_us), 2),
    round(float(alloc_non_us), 2),
    round(float(alloc_cash), 2),
    round(float(alloc_gold), 2),
    round(float(alloc_btc), 2),
    rebalance_strategy,
    int(tent_duration_years) if rebalance_strategy == "Cash Tent" else 0,
    round(float(st.session_state.get("dividend_yield_pct", 1.5)), 2),
    round(float(st.session_state.get("ret_cash_pct", 1.0)), 2),
    round(float(st.session_state.get("monthly_ahv_input", 2000)), 2),
    round(float(st.session_state.get("municipal_multiplier_pct", 119.0)), 2),
)

if '_synced_twr_pct' not in st.session_state or st.session_state.get('_last_vanguard_sync_sig') != _vanguard_sync_sig:
    st.session_state['_last_vanguard_sync_sig'] = _vanguard_sync_sig
    if total_start_nw > 0:
        est_tax = get_year0_taxes(float(st.session_state['annual_expenses']))
        st.session_state['_synced_twr_pct'] = round(((st.session_state['annual_expenses'] + est_tax) / total_start_nw) * 100.0, 2)
    else:
        st.session_state['_synced_twr_pct'] = 3.5
    if spending_strategy == "Vanguard Dynamic":
        st.session_state['vanguard_target_rate_pct'] = st.session_state['_synced_twr_pct']
elif spending_strategy == "Vanguard Dynamic" and 'vanguard_target_rate_pct' not in st.session_state:
    st.session_state['vanguard_target_rate_pct'] = st.session_state['_synced_twr_pct']


def on_expenses_change():
    nw = total_start_nw
    if nw > 0:
        exp = float(st.session_state['annual_expenses'])
        est_tax = get_year0_taxes(exp)
        total_outflow = exp + est_tax
        synced = round((total_outflow / nw) * 100.0, 2)
        st.session_state['_synced_twr_pct'] = synced
        if 'vanguard_target_rate_pct' in st.session_state:
            st.session_state['vanguard_target_rate_pct'] = synced


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
        rounded_exp = int(round(living_exp, -2))
        st.session_state['annual_expenses'] = rounded_exp
        actual_tax = get_year0_taxes(float(rounded_exp))
        actual_rate = round(((rounded_exp + actual_tax) / nw) * 100.0, 2)
        st.session_state['_synced_twr_pct'] = actual_rate
        if rounded_exp == 0 and rate < actual_rate:
            st.session_state['vanguard_target_rate_pct'] = actual_rate


annual_expenses = float(st.sidebar.number_input(
    f"Annual Base Expenses ({CCY})",
    key="annual_expenses",
    min_value=0,
    step=5000,
    format="%d",
    on_change=on_expenses_change,
    help=f"Your expected net living expenses (excluding taxes) in today's {CCY}. Under Vanguard Dynamic Spending, this is bi-directionally synchronized with Target Withdrawal Rate (%), accounting for estimated taxes."
))

dynamic_expense_floor_pct = 100.0
dynamic_expense_ceiling_pct = 100.0
vanguard_target_rate = (annual_expenses / total_start_nw) if total_start_nw > 0 else (st.session_state['_synced_twr_pct'] / 100.0)
vanguard_floor_pct = 0.050
vanguard_ceiling_pct = 0.050

if spending_strategy == "Dynamic (Floor & Ceiling)":
    dynamic_expense_floor_pct = st.sidebar.number_input("Reduced Expense Floor (%)", value=85.0, min_value=0.0, step=1.0, format="%.1f", help="The percentage of your base expenses you will spend when your after-tax net worth is below the starting watermark.")
    dynamic_expense_ceiling_pct = st.sidebar.number_input("Expanded Expense Ceiling (%)", value=max(115.0, float(dynamic_expense_floor_pct)), min_value=float(dynamic_expense_floor_pct), step=1.0, format="%.1f", help="The percentage of your base expenses you will spend when your after-tax net worth is above the starting watermark (must be ≥ Reduced Expense Floor).")
elif spending_strategy == "Vanguard Dynamic":
    twr_pct = st.sidebar.number_input(
        "Target Withdrawal Rate (%)",
        key="vanguard_target_rate_pct",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        on_change=on_twr_change,
        help="Target annual total withdrawal rate (covering both net living expenses and estimated taxes) as a % of total portfolio net worth. Bi-directionally synchronized with Annual Base Expenses (" + CCY + ")."
    )
    # Convert tax-inclusive UI withdrawal rate into net living expense rate of net worth for SimConfig
    vanguard_target_rate = (annual_expenses / total_start_nw) if total_start_nw > 0 else (twr_pct / 100.0)
    vanguard_floor_pct = st.sidebar.number_input("Max Annual Cut / Floor (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f", help="Maximum allowable reduction in spending compared to prior year's inflation-adjusted spending (Vanguard default: 5.0%).") / 100.0
    vanguard_ceiling_pct = st.sidebar.number_input("Max Annual Raise / Ceiling (%)", value=5.0, min_value=0.0, step=0.5, format="%.1f", help="Maximum allowable increase in spending compared to prior year's inflation-adjusted spending (Vanguard default: 5.0%).") / 100.0

st.sidebar.subheader("Income & Yield")
if IS_AT:
    monthly_ahv = st.sidebar.number_input(
        "Expected Monthly Gross State Pension from age 65 (EUR, paid 14×/yr)",
        value=2000,
        min_value=0,
        step=100,
        key="monthly_ahv_input",
        help=(
            "The gross monthly ASVG state pension you expect from age 65, in today's EUR (adjusted annually for CPI "
            "inflation in the simulation). Paid 14 times a year (12 regular payments + 13th/14th special payments). "
            "6% health insurance is deducted; the 12 regular payments are taxed at the progressive tariff (with the "
            "pensioner tax credit), the 13th/14th at 6% above EUR 620 (tax-free for pensions up to EUR 1,307.50/month). "
            "Verified against the BMF Brutto-Netto-Rechner. Health self-insurance stops when the pension starts."
        ),
    )
else:
    monthly_ahv = st.sidebar.number_input("Expected Monthly AHV Pension from age 65 (CHF)", value=2000, min_value=0, step=100, key="monthly_ahv_input", help="The monthly AHV pension you expect to receive starting at age 65 (in today's CHF, adjusted annually for CPI inflation in the simulation).")
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", value=1.5, min_value=0.0, step=0.1, format="%.1f", key="dividend_yield_pct", help="Expected annual dividend yield on US and Non-US equity holdings.") / 100.0


def _cagr_pct(annual_returns: np.ndarray) -> float:
    return (float(np.prod(1.0 + annual_returns)) ** (1.0 / len(annual_returns)) - 1.0) * 100.0


_hist_span = f"{HISTORIC_YEARS[0]}–{HISTORIC_YEARS[-1]}"
_us_usd_cagr = _cagr_pct(HISTORIC_RETURNS_US_USD)
_us_home_cagr = _cagr_pct(HIST.us)
_exus_home_cagr = _cagr_pct(HIST.non_us)

# The PPP assumption is read before the Monte Carlo defaults are derived so that all
# three engines share one FX assumption: Historic Backtesting and Bootstrapping rescale
# the foreign-priced monthly series by it, and the Monte Carlo default means for the
# same sleeves are calibrated on that rescaled history.
st.sidebar.subheader("Currency / PPP Assumption")
real_fx_appreciation = st.sidebar.slider(
    f"Real {CCY} Appreciation beyond PPP (%/yr)",
    min_value=-1.00,
    max_value=2.00,
    value=0.00,
    step=0.01,  # HISTORIC_REAL_{CHF,EUR}_APPRECIATION is rounded to 0.01pp and must be selectable
    format="%.2f%%",
    help=(
        f"Expected long-run real appreciation of the {HIST.currency_name} beyond inflation differentials "
        "(Relative Purchasing Power Parity) applied to foreign-priced sleeves (US Stocks, Non-US Stocks, Gold) "
        "in Historic Backtesting and Block Bootstrapping, and used to calibrate the Monte Carlo default means "
        "for the same sleeves so all three engines share one FX assumption.\n\n"
        "- 0.00% (default): Relative PPP holds going forward (no excess real currency drag).\n"
        f"- +{HIST.real_appreciation * 100:.2f}%: Reproduces the raw {_hist_span} historical real {CCY} "
        f"appreciation beyond {'US-AT' if IS_AT else 'US-CH'} CPI differentials."
    ),
) / 100.0

# Historic (arithmetic mean, log-vol) pairs in the exact parameterisation of generate_monte_carlo_returns,
# with the foreign-priced sleeves rescaled to the PPP assumption above (log-vol is invariant to that).
_hist_us_mean, _hist_us_vol = historic_lognormal_params(
    HIST.module.ppp_adjusted_annual_returns(HIST.us, real_fx_appreciation)
)
_hist_exus_mean, _hist_exus_vol = historic_lognormal_params(
    HIST.module.ppp_adjusted_annual_returns(HIST.non_us, real_fx_appreciation)
)
_hist_gold_mean, _hist_gold_vol = historic_lognormal_params(
    HIST.module.ppp_adjusted_annual_returns(HIST.gold, real_fx_appreciation)
)
_hist_cash_mean = float(np.mean(HIST.cash))
_hist_infl_mean = float(np.mean(HIST.inflation))
_hist_infl_std = float(np.std(HIST.inflation, ddof=1))
# One volatility input drives both equity sleeves, so default it to the average of the two histories.
_hist_eq_vol = (_hist_us_vol + _hist_exus_vol) / 2.0
_ppp_basis = f"at {real_fx_appreciation * 100:+.2f}%/yr real {CCY} appreciation beyond PPP"


def _hist_help(mean: float | None = None, vol: float | None = None, foreign: bool = False) -> str:
    parts = []
    if mean is not None:
        parts.append(f"arithmetic mean {mean * 100:.1f}%")
    if vol is not None:
        parts.append(f"log-return volatility {vol * 100:.1f}%")
    basis = f" ({_ppp_basis})" if foreign else ""
    return f" Historic {_hist_span} in {CCY}{basis}: " + ", ".join(parts) + "."


st.sidebar.subheader(
    "Monte Carlo Parameters",
    help=(
        f"Note: Returns and inflation must be Nominal (unadjusted for inflation) and in {CCY} terms, and the means are "
        f"arithmetic. For reference ({_hist_span}, raw geometric CAGR): US Stocks {_us_usd_cagr:.1f}% in USD but "
        f"{_us_home_cagr:.1f}% in {CCY} due to currency drag; Non-US Stocks {_exus_home_cagr:.1f}% in {CCY}. "
        "Stock means, volatilities, gold and inflation defaults are calibrated to that history (arithmetic means, "
        "log-return vols for assets); the US, Non-US and Gold means are taken from the history rescaled to the "
        f"Currency / PPP Assumption above, so all three engines share one FX assumption. Only {CCY} cash stays "
        "forward-looking (today's near-zero savings rates and Year 0 taxable interest), with historic values in each "
        "field's help."
    ),
)
_min_ret_pct = -99.0  # generate_monte_carlo_returns requires returns > -100%

# Sync PPP-calibrated defaults into stable session_state keys only while the user has not
# manually overridden that specific Monte Carlo mean input.
if st.session_state.get("_last_ppp_for_mc") != real_fx_appreciation:
    st.session_state["_last_ppp_for_mc"] = real_fx_appreciation
    for _key, _flag, _mean in (
        ("mc_ret_us_pct", "_user_edited_mc_ret_us", _hist_us_mean),
        ("mc_ret_exus_pct", "_user_edited_mc_ret_exus", _hist_exus_mean),
        ("mc_ret_gold_pct", "_user_edited_mc_ret_gold", _hist_gold_mean),
    ):
        if _key not in st.session_state or not st.session_state.get(_flag, False):
            st.session_state[_key] = round(_mean * 100.0, 1)
else:
    for _key, _mean in (
        ("mc_ret_us_pct", _hist_us_mean),
        ("mc_ret_exus_pct", _hist_exus_mean),
        ("mc_ret_gold_pct", _hist_gold_mean),
    ):
        if _key not in st.session_state:
            st.session_state[_key] = round(_mean * 100.0, 1)


def _mark_mc_us_edited():
    st.session_state["_user_edited_mc_ret_us"] = True


def _mark_mc_exus_edited():
    st.session_state["_user_edited_mc_ret_exus"] = True


def _mark_mc_gold_edited():
    st.session_state["_user_edited_mc_ret_gold"] = True


inflation_mean = st.sidebar.number_input("Inflation Mean (%)", value=round(_hist_infl_mean * 100.0, 1), min_value=_min_ret_pct, step=0.1, format="%.1f", help="Expected average annual inflation rate for Monte Carlo. Default = history." + _hist_help(_hist_infl_mean)) / 100.0
inflation_std = st.sidebar.number_input("Inflation Volatility (%)", value=round(_hist_infl_std * 100.0, 1), min_value=0.0, step=0.1, format="%.1f", help="Expected volatility of inflation for Monte Carlo. Default = history." + f" Historic {_hist_span}: {_hist_infl_std * 100:.1f}% (dominated by the 1920s deflation and the 1940s/1970s inflation spikes).") / 100.0
ret_us = st.sidebar.number_input("US Stocks Nominal Mean (%)", key="mc_ret_us_pct", min_value=_min_ret_pct, step=0.1, format="%.1f", on_change=_mark_mc_us_edited, help=f"Expected nominal arithmetic mean return for US Stocks in {CCY}. Default = history under the PPP assumption above." + _hist_help(_hist_us_mean, foreign=True) + f" (Raw S&P 500 CAGR: {_us_usd_cagr:.1f}% in USD, {_us_home_cagr:.1f}% in {CCY}.)") / 100.0
ret_non_us = st.sidebar.number_input("Non-US Stocks Nominal Mean (%)", key="mc_ret_exus_pct", min_value=_min_ret_pct, step=0.1, format="%.1f", on_change=_mark_mc_exus_edited, help=f"Expected nominal arithmetic mean return for Non-US Stocks in {CCY} terms. Default = history under the PPP assumption above." + _hist_help(_hist_exus_mean, foreign=True)) / 100.0
ret_cash = st.sidebar.number_input(f"{CCY} Cash Nominal Mean (%)", min_value=_min_ret_pct, step=0.1, format="%.1f", key="ret_cash_pct", help=f"Expected nominal mean return for {CCY} Cash (also used as the contractual taxable savings interest floor in Year 0 tax sync and Monte Carlo). Default {CDEF.cash_rate * 100:.1f}%" + (" (ECB, Austrian household deposits, Sep 2026: agreed maturity ≤ 1y 2.18%, notice deposits 2.05%)." if IS_AT else ".") + _hist_help(_hist_cash_mean)) / 100.0
ret_gold = st.sidebar.number_input("Gold Nominal Mean (%)", key="mc_ret_gold_pct", min_value=_min_ret_pct, step=0.1, format="%.1f", on_change=_mark_mc_gold_edited, help=f"Expected nominal arithmetic mean return for Gold in {CCY} terms. Default = history under the PPP assumption above." + _hist_help(_hist_gold_mean, foreign=True)) / 100.0
ret_btc = st.sidebar.number_input(
    "Bitcoin Nominal Mean (%)",
    value=float(BITCOIN_NOMINAL_MEAN * 100.0),
    min_value=_min_ret_pct,
    step=0.1,
    format="%.1f",
    help=f"Expected nominal arithmetic mean return for synthetic Bitcoin in {CCY} terms (applied across Historic Backtesting, Stationary Bootstrapping, and Monte Carlo).",
) / 100.0

vol_eq = st.sidebar.number_input("Equities Volatility (%)", value=round(_hist_eq_vol * 100.0, 1), min_value=0.0, step=0.1, format="%.1f", help="Annualized log-return volatility applied to both US and Non-US Stocks. Default = average of the two histories." + f" Historic {_hist_span} in {CCY}: US {_hist_us_vol * 100:.1f}%, Non-US {_hist_exus_vol * 100:.1f}%.") / 100.0
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

if int(duration) <= len(HISTORIC_YEARS):
    hist_num_runs = len(HISTORIC_YEARS) - int(duration) + 1
    st.sidebar.info(
        f"Using {len(HISTORIC_YEARS)}-year historic {HIST.data_label} ({HISTORIC_YEARS[0]}–{HISTORIC_YEARS[-1]}). "
        f"Available contiguous cohorts: {hist_num_runs}"
        + (f" (Swiss inflation and cash data {HISTORIC_YEARS[0]}–{_hr_eur.SWISS_PROXY_LAST_YEAR}, Swiss-based cash to "
           f"{_hr_eur.CASH_SWISS_PROXY_LAST_YEAR})" if IS_AT else "")
    )
else:
    st.sidebar.error(f"Duration {int(duration)} exceeds available historic data ({len(HISTORIC_YEARS)} years).")
    hist_num_runs = 0

# 6. Tax Location / Assumptions
if IS_AT:
    st.sidebar.subheader("Austrian Tax Assumptions")
    municipal_multiplier = ZURICH_CITY_MUNICIPAL_MULTIPLIER  # unused outside Switzerland
    bracket_indexation = st.sidebar.number_input(
        "Tax Bracket Indexation (% of CPI)",
        key="bracket_indexation_pct",
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        format="%.1f",
        help="How much of each year's inflation is passed through to the income tax bracket edges, the pensioner "
             "tax credit and the EUR 620 allowance on the 13th/14th pension payment.\n\n"
             "- 66.7% (default): the automatic indexation by two thirds of inflation under § 33a EStG; the remaining "
             "third is discretionary and not assumed.\n"
             "- 100%: full indexation.\n"
             "- 0%: brackets frozen in nominal terms ('kalte Progression').\n\n"
             "The flat 27.5% / 25% capital income tax rates are not affected.",
    ) / 100.0
    health_insurance_monthly = st.sidebar.number_input(
        "Health Self-Insurance (EUR/month, § 16 ASVG)",
        value=float(HEALTH_SELF_INSURANCE_MONTHLY),
        disabled=True,
        format="%.2f",
        help="Mandatory voluntary health insurance (Selbstversicherung in der Krankenversicherung, ÖGK 2026 rate), "
             "paid from retirement until the state pension starts (for life if no pension is entered), indexed with "
             "CPI. Not tax-deductible. From 65 the 6% health contribution on the pension applies instead.",
    )
else:
    st.sidebar.subheader("Zurich Tax Location")
    municipal_multiplier = st.sidebar.number_input("Municipal Multiplier (Steuerfuss, %)", value=ZURICH_CITY_MUNICIPAL_MULTIPLIER * 100.0, min_value=0.0, step=0.1, format="%.1f", key="municipal_multiplier_pct", help="Your municipal tax multiplier (Steuerfuss) in Zurich (e.g. 119% for City of Zurich).") / 100.0
    bracket_indexation = st.sidebar.number_input(
        "Tax Bracket Indexation (% of CPI)",
        key="bracket_indexation_pct",
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        format="%.0f",
        help="How much of each year's inflation is passed through to the tax bracket edges and the AHV contribution table.\n\n"
             "- 100%: Full indexation. This is what Swiss law mandates — Art. 39 DBG requires the federal tariff to be adjusted to the CPI annually, and § 48 StG ZH indexes the Zurich income and wealth tariffs.\n"
             "- 0%: Brackets frozen in nominal terms, so inflation alone pushes you into higher brackets ('kalte Progression'). A worst case, not the legal default.\n\n"
             "Note that the Steuerfuss multipliers are political and never indexed. A portfolio growing faster than inflation still climbs the wealth tax schedule in real terms at any setting."
    ) / 100.0
    health_insurance_monthly = HEALTH_SELF_INSURANCE_MONTHLY  # unused outside Austria

if hist_num_runs == 0:
    st.error("Cannot run simulation. Duration is too long for the available historic data.")
    st.stop()


def create_config(num_runs: int, *, mc_inflation: bool = True) -> SimConfig:
    return SimConfig(
        num_runs=num_runs,
        duration_years=int(duration),
        inflation_mean=inflation_mean if mc_inflation else 0.0,
        inflation_std=inflation_std if mc_inflation else 0.0,
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
        real_chf_appreciation=real_fx_appreciation,  # home-currency (CHF or EUR) real appreciation beyond PPP
        cash_rate=ret_cash,
        btc_equity_corr=btc_equity_corr,
        seed=random_seed,
        country=COUNTRY,
        health_insurance_monthly=float(health_insurance_monthly),
    )


config_mc = create_config(mc_num_runs, mc_inflation=True)
config_boot = create_config(boot_num_runs, mc_inflation=False)
config_hist = create_config(hist_num_runs, mc_inflation=False)

with st.spinner('Running Monte Carlo simulations...'):
    history_mc, mc_inflation_matrix = _run_cached_mc(
        config_mc, ret_us, ret_non_us, ret_cash, ret_gold, ret_btc, vol_eq, vol_gold, vol_btc
    )
with st.spinner('Running Bootstrapping simulations...'):
    history_boot, boot_inflation_matrix = _run_cached_boot(
        config_boot, boot_block_years, ret_btc, vol_btc
    )
with st.spinner('Running Historic Backtesting simulations...'):
    history_hist, hist_inflation_matrix = _run_cached_hist(
        config_hist, ret_btc, vol_btc
    )


def _build_hist_eff_caption(history, config, inflation_matrix, success_pct) -> str:
    success_mask = compute_success_mask(
        history['net_worth'][-1, :],
        history['initial_net_worth'],
        inflation_matrix,
        success_pct,
        net_worth_history=history['net_worth'],
    )
    eff_stats = compute_effective_sample_size(
        len(HISTORIC_YEARS), config.duration_years, float(np.mean(success_mask) * 100.0)
    )
    return (
        f"⚠️ <b>Effective Sample Size (<i>N</i><sub>eff</sub>): {eff_stats['n_eff']:.1f}</b> independent cohorts "
        f"(from {int(eff_stats['n_cohorts'])} overlapping {config.duration_years}y windows) · "
        f"<b>90% CI: {eff_stats['ci_low_pct']:.0f}%–{eff_stats['ci_high_pct']:.0f}%</b>"
    )


_TLDR_BADGES = {
    "BROKE": "🛑 **BROKE**",
    "SHORTFALL": "⚠️ **SHORTFALL**",
    "DEAD": "🪦 **DEAD**",
    "RICH": "🚀 **RICH**",
}


def _tldr_help(success_pct: float) -> str:
    target_desc = (
        f"{success_pct:g}% of inflation-adjusted starting net worth"
        if success_pct > 0.0
        else f"0 {CCY} (plain survival)"
    )
    return (
        "TL;DR summary of the median portfolio outcome at the end of the retirement horizon:\n"
        f"- 🚀 RICH: Median real (inflation-adjusted) ending net worth is ≥ {RICH_MULTIPLE:.0f}× starting net worth (and above target).\n"
        f"- 🪦 DEAD: Median real ending net worth hits your success criterion (> {target_desc}) but is < {RICH_MULTIPLE:.0f}× starting net worth.\n"
        f"- ⚠️ SHORTFALL: Median real ending net worth is > 0 {CCY} (solvent) but does not hit your success criterion (≤ {target_desc}).\n"
        f"- 🛑 BROKE: Median ending net worth is ≤ 0 {CCY} (the median path runs out of money before the end of the horizon)."
    )


def render_results(history, config, num_runs, title, inflation_matrix, success_pct, hist_eff_caption: str = ""):
    header_tooltips = {
        "Historic Backtesting": f"Replays exact contiguous historical sequences (e.g. 1928–1978, 1929–1979) from {len(HISTORIC_YEARS)} years of historical {HIST.data_label}{(" (" + SWISS_PROXY_NOTE + ")") if IS_AT else ""}. Because rolling multi-decade cohorts overlap heavily, effective sample size N_eff = T / D is surfaced alongside a 90% Wilson confidence band, and tail lines show Worst/Best Cohort instead of unsupported 5th/95th percentiles.",
        "Historic Bootstrapping": "Creates thousands of distinct retirement scenarios using the Politis–Romano (1994) Stationary Bootstrap with circular wrap-around and geometric block lengths (default mean 5 years). Preserves intra-year and multi-year macroeconomic cycles while sampling all historical years with equal 1/T probability.",
        "Monte Carlo": "Generates thousands of stochastic future paths using correlated parametric lognormal distributions based on user-configured nominal means, standard deviations, and inflation parameters."
    }
    st.header(title, help=header_tooltips.get(title))
    net_worth_history = history['net_worth']
    is_historic_backtest = "Historic Backtesting" in title

    final_net_worth = net_worth_history[-1, :]
    initial_nw = history['initial_net_worth']

    success_mask = compute_success_mask(
        final_net_worth, initial_nw, inflation_matrix, success_pct, net_worth_history=net_worth_history
    )

    if success_pct == 0.0:
        tooltip_text = f"Success is defined as never depleting net worth (NW > 0 {CCY} every year) and ending net worth > 0 {CCY}."
    else:
        tooltip_text = (
            f"Success is defined as never depleting net worth (NW > 0 {CCY} every year) and ending net worth > "
            f"{success_pct:g}% of the inflation-adjusted starting net worth."
        )

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
    initial_withdrawal = config.annual_base_expenses + estimate_year_0_taxes(config)
    inf_adj_start_withdrawal_trajectory = initial_withdrawal * median_cum_inflation
    st.markdown(f"**Nominal Starting NW:** {initial_nw:,.0f} {CCY}")
    if hist_eff_caption:
        visibility_style = "visible" if is_historic_backtest else "hidden"
        st.markdown(
            f'<div style="font-size: 0.875rem; color: rgba(49, 51, 63, 0.75); line-height: 1.4; '
            f'margin-top: -0.25rem; margin-bottom: 0.25rem; visibility: {visibility_style};" '
            f'aria-hidden="{"false" if is_historic_backtest else "true"}">'
            f'{hist_eff_caption}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"### {_TLDR_BADGES[classify_outcome(final_net_worth, initial_nw, inflation_matrix, success_pct)]}",
        help=_tldr_help(success_pct),
    )

    # Row 1: Success Rate & Watermark Metric
    col_r1_1, col_r1_2 = st.columns(2)
    col_r1_1.metric("Probability of Success", f"{success_rate:.1f}%", help=tooltip_text)

    avg_years_below_watermark = np.mean(np.sum(history['below_watermark'], axis=0))
    pct_years_below_watermark = (avg_years_below_watermark / config.duration_years) * 100.0
    col_r1_2.metric("Avg Years Below Start NW", f"{avg_years_below_watermark:.1f} ({pct_years_below_watermark:.1f}%)", help="Average number of years per simulation where after-tax net worth drops below the inflation-adjusted after-tax starting net worth watermark. Under Dynamic (Floor & Ceiling) spending, this equals the number of years expenses are reduced to the floor.")

    # Row 1b: Failure Decomposition (Depletion Rate & Age)
    failure_stats = summarize_failure_timing(history, inflation_matrix, success_pct, config.start_age)
    col_r1b_1, col_r1b_2 = st.columns(2)
    col_r1b_1.metric(
        "Depletion Rate (NW ≤ 0)",
        f"{failure_stats['depletion_rate_pct']:.1f}%",
        help=(
            "Decomposes non-successful runs by severity:\n"
            f"- Total Net Worth Depletion (NW ≤ 0): {failure_stats['depletion_rate_pct']:.1f}% (portfolio runs completely out of money)\n"
            + (
                ""
                if IS_AT
                else f"- Pre-AHV Liquid Depletion (Liquid ≤ 0 before Age 65): {failure_stats['pre_65_depletion_rate_pct']:.1f}% (accessible liquid wealth exhausted while Pillar 2/3a are still locked)\n"
            )
            + f"- Target Shortfall (0 < Final Real NW ≤ {success_pct:.1f}%): {failure_stats['shortfall_rate_pct']:.1f}% (survives full {config.duration_years}y horizon without going broke, but finishes below target ending wealth)"
        ),
    )
    if failure_stats['earliest_depletion_age'] is not None:
        depletion_age_val = f"Age {int(failure_stats['earliest_depletion_age'])} / {failure_stats['median_depletion_age']:.0f}"
        depletion_age_help = (
            f"Among runs that hit Net Worth ≤ 0 ({failure_stats['depletion_rate_pct']:.1f}% of runs): "
            f"Earliest depletion occurs at Age {int(failure_stats['earliest_depletion_age'])} (Year {failure_stats['earliest_depletion_year']}), "
            f"and Median depletion occurs at Age {failure_stats['median_depletion_age']:.1f} (Year {failure_stats['median_depletion_year']:.1f})."
        )
    else:
        depletion_age_val = "None (0 broke)"
        depletion_age_help = (
            f"No simulation runs depleted total net worth (NW ≤ 0) over the {config.duration_years}-year horizon. "
            + ("" if IS_AT else f"Pre-AHV liquid depletion rate (< Age 65): {failure_stats['pre_65_depletion_rate_pct']:.1f}%.")
        )
    col_r1b_2.metric("Depletion Age (Min / Med)", depletion_age_val, help=depletion_age_help)

    # Row 1c: Target Shortfall (Rate & Age)
    col_r1c_1, col_r1c_2 = st.columns(2)
    col_r1c_1.metric(
        "Shortfall Rate (Real NW ≤ Target)",
        f"{failure_stats['shortfall_occurrence_rate_pct']:.1f}%",
        help=(
            f"Share of runs where real net worth dropped to or below the {success_pct:g}% success criterion (% of inflation-adjusted starting NW) "
            f"at any point during the {config.duration_years}-year retirement horizon.\n\n"
            f"- Target Shortfall Occurrence: {failure_stats['shortfall_occurrence_rate_pct']:.1f}% (breached target at least once)\n"
            f"- Final Target Shortfall: {failure_stats['shortfall_rate_pct']:.1f}% (ended horizon below target without total ruin)\n"
            f"- Total Depletion (NW ≤ 0): {failure_stats['depletion_rate_pct']:.1f}% (exhausted all wealth)"
        ),
    )
    if failure_stats['earliest_shortfall_age'] is not None:
        shortfall_age_val = f"Age {int(failure_stats['earliest_shortfall_age'])} / {failure_stats['median_shortfall_age']:.0f}"
        shortfall_age_help = (
            f"Among runs where real net worth dropped to or below the {success_pct:g}% success criterion "
            f"({failure_stats['shortfall_occurrence_rate_pct']:.1f}% of runs): "
            f"First breached at earliest Age {int(failure_stats['earliest_shortfall_age'])} (Year {failure_stats['earliest_shortfall_year']}), "
            f"with median shortfall at Age {failure_stats['median_shortfall_age']:.1f} (Year {failure_stats['median_shortfall_year']:.1f})."
        )
    else:
        shortfall_age_val = "None (100% kept)"
        shortfall_age_help = (
            f"Real net worth stayed strictly above {success_pct:g}% of inflation-adjusted starting wealth "
            f"across all {config.duration_years} simulation years for every run."
        )
    col_r1c_2.metric("Shortfall Age (Min / Med)", shortfall_age_val, help=shortfall_age_help)

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
    col_r2_1.metric("Median Ending NW (Real)", f"{median_final_inf_adj:,.0f} {CCY}")
    col_r2_2.metric("Median Ending NW (Nominal)", f"{median_final:,.0f} {CCY}")

    # Row 3: Median Total Cumulative Withdrawals (Real vs Nominal)
    col_r3_1, col_r3_2 = st.columns(2)
    col_r3_1.metric("Median Total Withdrawals (Real)", f"{median_total_withdrawals_real:,.0f} {CCY}", help="Total cumulative real purchasing power spent on living expenses and taxes over the full simulation period.")
    col_r3_2.metric("Median Total Withdrawals (Nominal)", f"{median_total_withdrawals:,.0f} {CCY}", help="Total cumulative nominal cash spent on living expenses and taxes over the full simulation period.")

    # Row 4: Pre-AHV (<65) vs Post-65 (>=65) Outflows
    col_r4_1, col_r4_2 = st.columns(2)
    pre_65_years_cnt = int(np.sum(pre_65_mask))
    post_65_years_cnt = int(np.sum(post_65_mask))
    col_r4_1.metric(
        f"{PRE_PENSION} Outflow (< Age 65, {pre_65_years_cnt}y)",
        f"{median_pre_65_real:,.0f} {CCY}",
        help=f"Median cumulative real (today's {CCY}) outflows required to cover living expenses and taxes during the early retirement gap before age 65 ({pre_65_years_cnt} years). Nominal: {median_pre_65_nominal:,.0f} {CCY}."
    )
    col_r4_2.metric(
        f"Post-65 Outflow (Age 65+, {post_65_years_cnt}y)",
        f"{median_post_65_real:,.0f} {CCY}",
        help=f"Median cumulative real (today's {CCY}) outflows required to cover living expenses and taxes from age 65 through end-of-life ({post_65_years_cnt} years). Nominal: {median_post_65_nominal:,.0f} {CCY}."
    )

    # Plotly Chart (x-axis uses the retirement age during each simulated year: start_age .. start_age + duration_years - 1)
    years = np.arange(config.start_age, config.start_age + config.duration_years)
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
            hovertemplate="<b>" + trace_name + "</b><br>%{customdata} (Age: %{x})<br>Net Worth: %{y:,.0f} " + CCY + "<extra></extra>",
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
            hovertemplate="<b>" + label + "</b><br>%{customdata} (Age: %{x})<br>Net Worth: %{y:,.0f} " + CCY + "<extra></extra>"
        ))

    # Add Inflation-Adjusted Starting Net Worth Reference Line
    fig.add_trace(go.Scatter(
        x=years,
        y=inf_adj_start_nw_trajectory,
        mode='lines',
        name='Inflation-Adj Start NW',
        line=dict(color='black', width=2, dash='dash'),
        hovertemplate="<b>Inflation-Adj Start NW</b><br>Age: %{x}<br>Value: %{y:,.0f} " + CCY + "<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title=f"Net Worth ({CCY})",
        yaxis=dict(tickformat=",.0f"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Net Worth Trajectory", help="This chart displays the value of your assets over time.")
    st.plotly_chart(fig, width='stretch')

    # Portfolio Survival by Age Chart (Kaplan-Meier style survival curve)
    survival_curves = compute_survival_curves(history, inflation_matrix, success_pct)
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=years,
        y=survival_curves["solvent_pct"],
        mode='lines',
        name='Solvent (NW > 0)',
        line=dict(color='forestgreen', width=3),
        hovertemplate="<b>Solvent (NW > 0)</b><br>Age: %{x}<br>Survival: %{y:.1f}%<extra></extra>",
    ))
    if not IS_AT:
        fig_surv.add_trace(go.Scatter(
            x=years,
            y=survival_curves["liquid_solvent_pct"],
            mode='lines',
            name='Liquid Solvent (Liquid > 0)',
            line=dict(color='darkorange', width=2, dash='dot'),
            hovertemplate="<b>Liquid Solvent (Liquid > 0)</b><br>Age: %{x}<br>Survival: %{y:.1f}%<extra></extra>",
        ))
    fig_surv.add_trace(go.Scatter(
        x=years,
        y=survival_curves["above_target_pct"],
        mode='lines',
        name=f'Above Target ({success_pct:g}% Start NW)',
        line=dict(color='royalblue', width=2.5, dash='dash'),
        hovertemplate=f"<b>Above Target ({success_pct:g}% Start NW)</b><br>Age: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>",
    ))
    if config.start_age < 65 < config.start_age + config.duration_years:
        fig_surv.add_vline(
            x=65,
            line_width=1.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Age 65 (State Pension)" if IS_AT else "Age 65 (AHV + P2)",
            annotation_position="bottom left",
        )
    fig_surv.update_layout(
        xaxis_title="Age",
        yaxis_title="Share of Runs (%)",
        yaxis=dict(tickformat=".0f", range=[0, 102]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10),
    )
    st.subheader(
        "Portfolio Survival by Age",
        help=(
            "Shows when portfolio failures occur over retirement rather than only at the final year:\n"
            "- Solvent (NW > 0): % of paths that have never run out of total net worth up to each age.\n"
            + (
                ""
                if IS_AT
                else "- Liquid Solvent (Liquid > 0): % of paths that have never exhausted accessible liquid wealth (captures Pre-AHV bridge insolvency before Age 65 while Pillar 2/3a are locked).\n"
            )
            + "- Above Target: % of never-depleted paths whose real net worth exceeds your target ending wealth criterion at each age."
        ),
    )
    st.plotly_chart(fig_surv, width='stretch')

    # Income Breakdown Chart
    median_divs = np.median(history['income_dividends'], axis=1)
    median_interest = np.median(history.get('income_interest', np.zeros_like(history['income_dividends'])), axis=1)
    median_ahv = np.median(history['income_ahv'], axis=1)
    median_expenses = np.median(history['expenses_paid'], axis=1)
    median_taxes = np.median(history['taxes_paid'], axis=1)

    capital_sold = np.maximum(0, median_expenses + median_taxes - median_divs - median_interest - median_ahv)

    fig_income = go.Figure()
    fig_income.add_trace(go.Bar(x=years, y=median_divs, name='Dividends', marker_color='blue'))
    fig_income.add_trace(go.Bar(x=years, y=median_interest, name='Cash Interest', marker_color='teal'))
    fig_income.add_trace(go.Bar(x=years, y=median_ahv, name=PENSION_LABEL, marker_color='orange'))
    fig_income.add_trace(go.Bar(x=years, y=capital_sold, name='Capital Sold', marker_color='red'))

    fig_income.add_trace(go.Scatter(x=years, y=inf_adj_start_withdrawal_trajectory, mode='lines', name='Inflation-Adj Start', line=dict(color='forestgreen', width=2.5)))

    fig_income.update_layout(
        xaxis_title="Age",
        yaxis_title=f"Amount ({CCY})",
        barmode='stack',
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Income vs Required Cash", help="This chart displays the median cash flows across all simulated portfolio paths for each year (Dividends, " + CCY + " Cash Interest, " + PENSION_LABEL + ", and Capital Sold), along with the inflation-adjusted starting withdrawal trajectory.")
    st.plotly_chart(fig_income, width='stretch')

    # Annual Withdrawal Breakdown Chart (50th Pct bars + single bad-case path stacked lines on median-CPI scale + Inflation-Adj Start)
    ranked_paths = worst_path_order(history, inflation_matrix, success_pct)
    if is_historic_backtest:
        bad_idx = int(ranked_paths[0])
        bad_label = 'Worst Cohort'
    else:
        bad_idx = int(ranked_paths[int(round(0.05 * (int(num_runs) - 1)))])
        bad_label = '5th Pct'

    exp_path = history['expenses_paid'][:, bad_idx]
    tax_path = history['taxes_paid'][:, bad_idx]
    unfunded_path = (
        history['unfunded_deficit'][:, bad_idx]
        if 'unfunded_deficit' in history
        else np.zeros_like(exp_path)
    )
    funded_exp_path = np.maximum(0.0, exp_path - unfunded_path)
    rem_unfunded = np.maximum(0.0, unfunded_path - exp_path)
    funded_tax_path = np.maximum(0.0, tax_path - rem_unfunded)
    cpi_scale_to_median = median_cum_inflation / cum_inflation[bad_idx, :]
    bad_expenses = funded_exp_path * cpi_scale_to_median
    bad_taxes = funded_tax_path * cpi_scale_to_median

    tax_component_labels = {
        "income_tax": "Income tax",
        "wealth_tax": "Wealth tax",
        "ahv_contributions": "AHV contributions",
        "kest": "KESt (capital income tax)",
        "health_insurance": "Health insurance",
    }
    components = history.get("tax_components", {})
    component_lines = [
        (tax_component_labels.get(name, name), np.median(values, axis=1)) for name, values in components.items()
    ]
    if components:
        residual = history['taxes_paid'] - sum(components.values())
        if np.any(np.abs(residual) > 0.5):
            component_lines.append(("Capital withdrawal tax", np.median(residual, axis=1)))
    tax_hover = [
        "<br>".join(f"{label}: {vals[t]:,.0f} {CCY}" for label, vals in component_lines) for t in range(len(years))
    ]

    fig_withdrawal = go.Figure(data=[
        go.Bar(name='Living Expenses (50th Pct)', x=years, y=median_expenses, marker_color='royalblue'),
        go.Bar(
            name='Taxes Paid (50th Pct)',
            x=years,
            y=median_taxes,
            marker_color='crimson',
            customdata=tax_hover,
            hovertemplate="<b>Taxes Paid (50th Pct)</b><br>Age: %{x}<br>Total: %{y:,.0f} " + CCY
            + "<br><i>Component medians:</i><br>%{customdata}<extra></extra>",
        ),
        go.Scatter(
            name=f'Living Expenses ({bad_label})',
            x=years,
            y=bad_expenses,
            mode='lines',
            stackgroup='bad',
            fill='none',
            line=dict(color='navy', width=2.5, dash='dash'),
        ),
        go.Scatter(
            name=f'Taxes Paid ({bad_label})',
            x=years,
            y=bad_taxes,
            mode='lines',
            stackgroup='bad',
            fill='none',
            line=dict(color='darkred', width=2.5, dash='dot'),
        ),
        go.Scatter(
            name='Inflation-Adj Start',
            x=years,
            y=inf_adj_start_withdrawal_trajectory,
            mode='lines',
            line=dict(color='forestgreen', width=2.5),
        ),
    ])
    fig_withdrawal.update_layout(
        barmode='stack',
        xaxis_title="Age",
        yaxis_title=f"Annual Withdrawal ({CCY})",
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader(
        "Annual Withdrawal Breakdown",
        help=(
            "This stacked chart displays the median (50th Pct, bars) and bad-case path "
            f"({bad_label}, stacked lines from the worst/5th-percentile outcome path re-expressed on the median CPI scale) "
            "annual funded withdrawals (living expenses and taxes paid) over time, along with the inflation-adjusted starting withdrawal reference line. "
            "Hover the taxes bar for the median of each tax component."
        ),
    )
    st.plotly_chart(fig_withdrawal, width='stretch')

    # Withdrawal Rate Chart (net portfolio withdrawals relative to beginning-of-year net worth)
    withdrawal_rate_history = beginning_of_year_withdrawal_rate(history)
    wr_pct_labels = (
        ['Best Cohort (Min)', '25th Pct', '50th Pct', '75th Pct', 'Worst Cohort (Max)']
        if is_historic_backtest
        else pct_labels
    )
    wr_colors = colors[::-1]

    fig_wr = go.Figure()
    max_p_val = 5.0
    for p, label, c in zip(percentiles, wr_pct_labels, wr_colors):
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
    st.subheader("Withdrawal Rate", help="This chart displays net annual portfolio withdrawals (expenses + taxes minus " + PENSION_LABEL.lower() + ", floored at 0) as a percentage of beginning-of-year net worth.")
    st.plotly_chart(fig_wr, width='stretch')

    # Asset Allocation Development Chart (Liquid Assets + Pillar 2 & 3a)
    median_assets_by_class = np.median(history['liquid_assets_by_class'], axis=1)  # shape: (years, 5)
    p2_hist = history['pillar_2'] if 'pillar_2' in history else np.zeros((len(years), 1))
    p3a_hist = history['pillar_3a'] if 'pillar_3a' in history else np.zeros((len(years), 1))
    median_pensions = np.median(p2_hist + p3a_hist, axis=1)

    fig_alloc = go.Figure()

    all_series = [
        (f'{CCY} Cash', median_assets_by_class[:, 2], 'forestgreen'),
        ('US Stocks', median_assets_by_class[:, 0], 'royalblue'),
        ('Non-US Stocks', median_assets_by_class[:, 1], 'darkcyan'),
        ('Gold', median_assets_by_class[:, 3], 'gold'),
        ('Bitcoin', median_assets_by_class[:, 4], 'purple'),
        ('Pillar 2 & 3a', median_pensions, 'darkorange'),
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
        yaxis_title=f"Asset Value ({CCY})",
        yaxis=dict(tickformat=",.0f"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=15, b=65, l=10, r=10)
    )
    st.subheader("Asset Allocation Development", help="This stacked chart visualizes the median nominal balance of all asset categories (taxable liquid investments and combined Pillar 2 & 3a pensions) over time, showing how your net worth breakdown glides and rebalances throughout retirement.")
    st.plotly_chart(fig_alloc, width='stretch')

    analysis_name = "Cohort Analysis" if is_historic_backtest else "Run Analysis"
    id_col_name = "Cohort" if is_historic_backtest else "Run"

    worst_order_help = (
        "Worst paths are ranked by outcome severity (shown in the **Outcome** column), most severe group first:\n"
        "1. **Depleted** (net worth reached ≤ 0): earliest Depletion Age first.\n"
        f"2. **Failed** (survived, but Final NW (Real) ≤ {success_pct:g}% of starting NW): lowest Final NW (Real) first.\n"
        f"3. **Recovered** (dropped to ≤ {success_pct:g}% during retirement but finished above it): most Yrs in Shortfall first.\n"
        "4. **Never breached**: lowest Final NW (Real) first.\n"
        "Remaining ties are broken by lowest Final NW (Real)."
    )
    st.subheader(
        analysis_name,
        help=(
            "Best paths are sorted by Final NW (Real), highest first.\n\n"
            + worst_order_help
            + "\n\n"
            f"- **{id_col_name}**: {'Historical retirement cohort calendar years' if is_historic_backtest else 'Simulation run identifier'}.\n"
            f"- **Outcome**: Severity group of the path relative to the {success_pct:g}% success criterion (Depleted / Failed / Recovered / Never breached).\n"
            f"- **Depletion Age**: Age and simulation year when total net worth first depleted to ≤ 0 {CCY} ('Age X (Yr Y)'), 'Liq Age X (Yr Y)' if accessible liquid wealth was exhausted before Age 65 while pensions were locked, or 'Survived'.\n"
            f"- **Yrs in Shortfall**: Number of years real net worth was at or below the {success_pct:g}% success criterion.\n"
            f"- **Shortfall Age**: Age and simulation year when real net worth first dropped to or below the {success_pct:g}% success criterion ('Age X (Yr Y)'), even if the path later recovered, or 'Never' if the criterion was maintained throughout retirement.\n"
            f"- **Final NW (Real)**: Ending net worth in today's purchasing power (deflated by this path's realized cumulative {CPI_NAME} inflation).\n"
            f"- **Final NW (Nom)**: Ending net worth in future nominal {CCY_NAME} (unadjusted for inflation).\n"
            "- **Min NW (Real)**: Lowest real net worth (today's purchasing power) reached at any year-end during retirement.\n"
            "- **Yrs Below**: Total years spent below the inflation-adjusted after-tax starting net worth watermark."
        ),
    )

    final_nw = net_worth_history[-1, :]
    min_nw_real = min_real_net_worth(net_worth_history, inflation_matrix)
    years_below = np.sum(history['below_watermark'], axis=0)
    yrs_in_shortfall = years_in_shortfall(net_worth_history, initial_nw, inflation_matrix, success_pct)
    outcomes = classify_path_outcomes(history, inflation_matrix, success_pct)
    depletion_years = failure_stats["depletion_years"]
    liquid_depletion_years = failure_stats["liquid_depletion_years"]
    shortfall_years = failure_stats["shortfall_years"]

    worst_indices = worst_path_order(history, inflation_matrix, success_pct)[:10]
    best_indices = np.argsort(final_nw_inf_adj)[::-1][:10]

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
                "Outcome": str(outcomes[idx]),
                "Depletion Age": format_depletion_label(idx, depletion_years, liquid_depletion_years, config.start_age),
                "Yrs in Shortfall": int(yrs_in_shortfall[idx]),
                "Shortfall Age": format_shortfall_label(idx, shortfall_years, config.start_age),
                "Final NW (Real)": final_nw_inf_adj[idx],
                "Final NW (Nom)": final_nw[idx],
                "Min NW (Real)": min_nw_real[idx],
                "Yrs Below": int(years_below[idx])
            })
        return pd.DataFrame(data)

    column_config = {
        id_col_name: st.column_config.TextColumn(
            id_col_name,
            help=(
                "Calendar years spanned by this contiguous historical retirement cohort (e.g. 1928 - 1977)."
                if is_historic_backtest
                else "Simulation run identifier for this stochastic future path."
            ),
        ),
        "Outcome": st.column_config.TextColumn(
            "Outcome",
            help=(
                "Severity group used to rank the worst paths: "
                "'Depleted' = net worth reached ≤ 0; "
                f"'Failed' = survived but finished at or below the {success_pct:g}% success criterion; "
                f"'Recovered' = dropped to or below the criterion during retirement but finished above it; "
                "'Never breached' = stayed above the criterion every year."
            ),
        ),
        "Depletion Age": st.column_config.TextColumn(
            "Depletion Age",
            help=(
                f"Retirement age and simulation year when total net worth first reached ≤ 0 {CCY} ('Age X (Yr Y)'). "
                "'Liq Age X (Yr Y)' indicates accessible liquid wealth was exhausted before Age 65 while Pillar 2/3a pensions were locked. "
                "'Survived' means portfolio remained solvent throughout the entire retirement horizon."
            ),
        ),
        "Shortfall Age": st.column_config.TextColumn(
            "Shortfall Age",
            help=(
                f"Retirement age and simulation year when real net worth first dropped to or below the success "
                f"criterion ({success_pct:g}% of inflation-adjusted starting net worth), shown as 'Age X (Yr Y)'. "
                f"Counts any breach, even if the path later recovered. 'Never' if net worth stayed strictly above "
                f"the criterion throughout the entire retirement horizon."
            ),
        ),
        "Yrs in Shortfall": st.column_config.NumberColumn(
            "Yrs in Shortfall",
            format="%d",
            help=f"Number of years (not necessarily consecutive) this path's real net worth was at or below the {success_pct:g}% success criterion.",
        ),
        "Final NW (Real)": st.column_config.NumberColumn(
            "Final NW (Real)",
            format="%,.0f " + CCY,
            help=f"Ending net worth in today's purchasing power (deflated by this path's realized cumulative {CPI_NAME} inflation).",
        ),
        "Final NW (Nom)": st.column_config.NumberColumn(
            "Final NW (Nom)",
            format="%,.0f " + CCY,
            help=f"Ending net worth in future nominal {CCY_NAME} (unadjusted for inflation). Bankrupt paths include accrued 5% annual debt penalty interest.",
        ),
        "Min NW (Real)": st.column_config.NumberColumn(
            "Min NW (Real)",
            format="%,.0f " + CCY,
            help="Lowest real net worth (today's purchasing power) reached at any year-end during the retirement horizon, i.e. the depth of the worst trough.",
        ),
        "Yrs Below": st.column_config.NumberColumn(
            "Yrs Below",
            format="%d",
            help="Total number of years this path spent below the inflation-adjusted after-tax starting net worth watermark (triggering defensive spending cuts or smart cash buffer rules).",
        ),
    }

    st.markdown(f"##### Top 10 Best {id_col_name}s")
    st.caption("Sorted by **Final NW (Real)**, highest first.")
    df_best = build_df(best_indices)
    st.dataframe(df_best, column_config=column_config, hide_index=True, width='stretch')

    st.markdown(f"##### Top 10 Worst {id_col_name}s")
    st.caption(
        "Sorted by **Outcome** severity, then within each group: "
        "**Depleted** → earliest Depletion Age · "
        "**Failed** → lowest Final NW (Real) · "
        "**Recovered** → most Yrs in Shortfall · "
        "**Never breached** → lowest Final NW (Real). "
        "Remaining ties: lowest Final NW (Real).",
        help=worst_order_help,
    )
    df_worst = build_df(worst_indices)
    st.dataframe(df_worst, column_config=column_config, hide_index=True, width='stretch')


hist_eff_caption = _build_hist_eff_caption(history_hist, config_hist, hist_inflation_matrix, success_pct)
col_hist, col_boot, col_mc = st.columns(3)
with col_hist:
    render_results(history_hist, config_hist, hist_num_runs, "Historic Backtesting", hist_inflation_matrix, success_pct, hist_eff_caption)
with col_boot:
    render_results(history_boot, config_boot, boot_num_runs, "Historic Bootstrapping", boot_inflation_matrix, success_pct, hist_eff_caption)
with col_mc:
    render_results(history_mc, config_mc, mc_num_runs, "Monte Carlo", mc_inflation_matrix, success_pct, hist_eff_caption)
