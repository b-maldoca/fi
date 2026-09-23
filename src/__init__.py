"""Zurich Early Retirement Simulator - simulation engine package.

This package is deliberately decoupled from the Streamlit UI (``app.py``) so it
can be driven programmatically. From the repository root, no ``sys.path``
manipulation is required::

    from src import SimConfig, run_simulation

    config = SimConfig(
        num_runs=100,
        duration_years=50,
        inflation_mean=0.02,
        inflation_std=0.01,
        start_age=40,
        dividend_yield=0.015,
        initial_liquid_wealth=3_000_000.0,
        alloc_us_stocks=1.0,
    )
    results = run_simulation(config, returns, inflation)

The individual modules remain importable directly if you only need part of the
surface::

    from src.tax_engine import calculate_income_tax
"""

from .historic_returns import (
    BITCOIN_NOMINAL_MEAN,
    BITCOIN_VOL,
    DEFAULT_BTC_EQUITY_CORR,
    HISTORIC_REAL_CHF_APPRECIATION,
    HISTORIC_RETURNS,
    HISTORIC_RETURNS_CASH_CHF,
    HISTORIC_RETURNS_GOLD_CHF,
    HISTORIC_RETURNS_NON_US_CHF,
    HISTORIC_RETURNS_NON_US_USD,
    HISTORIC_RETURNS_US_CHF,
    HISTORIC_RETURNS_US_USD,
    HISTORIC_SWISS_INFLATION,
    HISTORIC_USD_CHF_FX,
    HISTORIC_YEARS,
    MONTHLY_CASH_CHF,
    POLITIS_WHITE_BLOCK_YEARS,
    compute_effective_sample_size,
    generate_bootstrapped_data,
    generate_bootstrapped_inflation,
    generate_bootstrapped_returns,
    get_historic_inflation_matrix,
    get_historic_return_matrix,
)
from .simulation_engine import (
    VALID_REBALANCE_STRATEGIES,
    VALID_SPENDING_STRATEGIES,
    SimConfig,
    estimate_year_0_taxes,
    generate_monte_carlo_inflation,
    generate_monte_carlo_returns,
    get_target_weights,
    run_simulation,
)
from .tax_engine import (
    calculate_ahv_non_worker,
    calculate_capital_withdrawal_tax,
    calculate_income_tax,
    calculate_wealth_tax,
)

__all__ = [
    # simulation_engine
    "SimConfig",
    "run_simulation",
    "estimate_year_0_taxes",
    "get_target_weights",
    "generate_monte_carlo_returns",
    "generate_monte_carlo_inflation",
    "VALID_SPENDING_STRATEGIES",
    "VALID_REBALANCE_STRATEGIES",
    # tax_engine
    "calculate_income_tax",
    "calculate_wealth_tax",
    "calculate_capital_withdrawal_tax",
    "calculate_ahv_non_worker",
    # historic_returns
    "HISTORIC_YEARS",
    "HISTORIC_REAL_CHF_APPRECIATION",
    "BITCOIN_NOMINAL_MEAN",
    "BITCOIN_VOL",
    "DEFAULT_BTC_EQUITY_CORR",
    "POLITIS_WHITE_BLOCK_YEARS",
    "HISTORIC_RETURNS",
    "HISTORIC_RETURNS_US_CHF",
    "HISTORIC_RETURNS_NON_US_CHF",
    "HISTORIC_RETURNS_CASH_CHF",
    "HISTORIC_RETURNS_GOLD_CHF",
    "HISTORIC_RETURNS_US_USD",
    "HISTORIC_RETURNS_NON_US_USD",
    "HISTORIC_SWISS_INFLATION",
    "HISTORIC_USD_CHF_FX",
    "MONTHLY_CASH_CHF",
    "compute_effective_sample_size",
    "get_historic_return_matrix",
    "get_historic_inflation_matrix",
    "generate_bootstrapped_data",
    "generate_bootstrapped_returns",
    "generate_bootstrapped_inflation",
]
