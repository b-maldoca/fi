import numpy as np

from src.simulation_engine import SimConfig, run_simulation, get_target_weights, estimate_year_0_taxes
from src.historic_returns import (
    get_historic_return_matrix,
    get_historic_inflation_matrix,
    generate_bootstrapped_data,
    generate_bootstrapped_returns,
    generate_bootstrapped_inflation,
    HISTORIC_RETURNS,
    HISTORIC_SWISS_INFLATION,
    HISTORIC_YEARS
)

def test_simulation_engine_basic_run():
    config = SimConfig(
        num_runs=2,
        duration_years=5,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=60,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=1050000.0,
        initial_pillar_2=500000.0,
        initial_pillar_3a_accounts=[50000.0, 50000.0],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.3,
        alloc_chf_cash=0.2,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Yearly',
        rebalance_threshold=0.05,
        annual_base_expenses=100000.0,
        monthly_ahv_pension=2450.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Setup a simple deterministic return matrix (0% return)
    # shape: (num_runs, duration_months, 5)
    return_matrix = np.zeros((config.num_runs, config.duration_years * 12, 5))
    
    history = run_simulation(config, return_matrix)
    
    # Verify outputs
    assert history['net_worth'].shape == (5, 2)
    assert history['liquid_assets'].shape == (5, 2)

def test_simulation_engine_pillar_2_liquidation():
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=64, # Year 0: 64, Year 1: 65
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=500000.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    return_matrix = np.zeros((config.num_runs, config.duration_years * 12, 5))
    history = run_simulation(config, return_matrix)
    
    # liquid assets should jump by ~500k minus significant capital withdrawal taxes (approx 40k)
    assert history['liquid_assets'][1, 0] > 550000.0
    assert history['liquid_assets'][1, 0] < 600000.0

def test_simulation_engine_bankruptcy_outflow():
    # Start with 0 assets, high expenses, to force severe bankruptcy.
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=0.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=100_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    return_matrix = np.zeros((config.num_runs, config.duration_years * 12, 5))
    history = run_simulation(config, return_matrix)
    
    # Year 0 end: expenses are 100k + 530 CHF mandatory minimum AHV non-worker tax.
    assert np.isclose(history['liquid_assets'][0, 0], -100_530.0)
    
    # Year 1 end: Previous debt incurs 5% penalty for 12 months, then another 100.5k expense is added.
    # -100530 * 1.05 = -105556.5. Plus -100530 = -206086.5
    assert np.isclose(history['liquid_assets'][1, 0], -206_086.5, rtol=0.01)

def test_simulation_engine_capital_withdrawal_timing():
    # If a 500k Pillar 2 is liquidated, the tax should not earn market returns for 11 months.
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=64, # Year 0: age 64, Year 1: age 65
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=0.0,
        initial_pillar_2=500_000.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0, # All in stocks
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    # 10% monthly return (huge, to test if tax money grows incorrectly)
    return_matrix = np.full((config.num_runs, config.duration_years * 12, 5), 0.10)
    history = run_simulation(config, return_matrix)
    
    # Pillar 2 liquidated at start of year 1 (age 65).
    # Pillar 2 liquidated at start of year 1 (age 65).
    # Starts at 500k. Since it tracks equities, it grows by 10% monthly in year 0 -> ~1.569M.
    # Tax on 1.569M is ~83k. Net is ~1.48M.
    # 1.48M grows by 1.10^12 (3.138) -> ~4.65M.
    # If tax was deferred to the end of the year, 1.569M would grow to 4.92M, then 83k deducted -> 4.84M.
    # Thus, if the tax is correctly deducted immediately, the final amount should be safely < 4.75M.
    assert history['liquid_assets'][1, 0] < 4_750_000.0

def test_simulation_engine_int_casting():
    # Verify that passing strictly integers from the UI does not crash the numpy arrays with UFuncOutputCastingError.
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=0, # Integer
        initial_pillar_2=500000, # Integer
        initial_pillar_3a_accounts=[20000, 20000], # Integers
        alloc_us_stocks=1.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0, # Integer
        monthly_ahv_pension=0, # Integer
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    # Market return is a float
    return_matrix = np.full((config.num_runs, config.duration_years * 12, 5), 0.05)
    
    # This should not raise an exception
    history = run_simulation(config, return_matrix)
    assert history is not None

def test_simulation_engine_dynamic_expenses():
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=True,
        dynamic_expense_floor_pct=0.8,
        enable_smart_selling=False,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=1.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=10000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    # Market return is negative (-10% each month), dropping net worth below 100k
    return_matrix = np.full((config.num_runs, config.duration_years * 12, 5), -0.10)
    history = run_simulation(config, return_matrix)
    # Expenses should be exactly 8000.0 (80% of 10000.0) because NW drops below initial 100k.
    assert np.isclose(history['expenses_paid'][0, 0], 8000.0)


def test_simulation_engine_smart_cash_buffer():
    def run_with_smart_selling(enabled: bool):
        config = SimConfig(
            num_runs=1,
            duration_years=2,
            inflation_mean=0.0,
            inflation_std=0.0,
            start_age=40,
            dividend_yield=0.0,
            enable_dynamic_expenses=False,
            dynamic_expense_floor_pct=1.0,
            enable_smart_selling=enabled,
            initial_liquid_wealth=100000.0,
            initial_pillar_2=0.0,
            initial_pillar_3a_accounts=[],
            alloc_us_stocks=0.5,
            alloc_non_us_stocks=0.0,
            alloc_chf_cash=0.5,
            alloc_gold=0.0,
            alloc_bitcoin=0.0,
            rebalance_strategy='Monthly',
            rebalance_threshold=0.05,
            annual_base_expenses=10000.0,
            monthly_ahv_pension=0.0,
            cantonal_multiplier=1.0,
            municipal_multiplier=1.19
        )
        
        # Consistent market crash (-5% a month)
        return_matrix = np.full((config.num_runs, config.duration_years * 12, 5), -0.05)
        return_matrix[:, :, 2] = 0.0 # Cash does not drop
        
        return run_simulation(config, return_matrix)
        
    history_dumb = run_with_smart_selling(False)
    history_smart = run_with_smart_selling(True)
    
    final_nw_dumb = history_dumb['net_worth'][-1, 0]
    final_nw_smart = history_smart['net_worth'][-1, 0]
    
    # Smart selling prevents rebalancing falling stocks with safe cash,
    # and forces expenses to be paid from cash, preserving more overall net worth.
    assert final_nw_smart > final_nw_dumb


def test_simulation_engine_negative_market_returns():
    # Verify that asset balances do not go negative due to extreme negative market returns.
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0, # 100% stocks
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0.0, # No expenses
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Return of -150% in month 0
    return_matrix = np.zeros((config.num_runs, config.duration_years * 12, 5))
    return_matrix[:, 0, 0] = -1.5
    
    # We need to capture the state after month 0.
    # Since we can't easily capture monthly state, we look at the end of the year.
    # If the asset went negative (-50k) and then had 0% returns, it would be -50k.
    # If it is clipped to 0, it should be 0.
    # Note: since it goes negative, it would be treated as bankrupt next month and moved to cash,
    # then penalized with 5% APY.
    # If it's clipped to 0, it stays 0.
    history = run_simulation(config, return_matrix)
    # The asset drops to 0, but at the end of the year they owe the minimum AHV non-worker contribution (530 CHF),
    # which goes into negative cash.
    assert history['net_worth'][0, 0] == -530.0


def test_simulation_engine_staggered_3a_at_62():
    # If starting at age 62 with three 3a accounts:
    # Year 0 (age 62): Liquidate account 0 (50k). Tax should be cap_tax(50k) + AHV(50k).
    # Year 1 (age 63): Liquidate account 1 (50k). Tax should be cap_tax(50k) + AHV(100k).
    # Year 2 (age 64): Liquidate account 2 (50k). Tax should be cap_tax(50k) + AHV(150k).
    #
    # Without staggering fix, all 3 liquidate in Year 0.
    # Year 0 tax = cap_tax(150k) + AHV(150k).
    # Year 1 tax = 0 + AHV(150k) = 530.
    #
    # Capital withdrawal tax on 50k is approx:
    # Fed base: calculate_bracket_tax(50k) = 18.8k*0 + 14.5k*0.0077 + 10.5k*0.0088 + 6.2k*0.0264 = 111.65 + 92.4 + 163.68 = 367.73.
    # Fed cap tax = 367.73 / 5 = 73.55.
    # Zurich base: calculate_bracket_tax(50k) = 7.3k*0 + 5.2k*0.02 + 5.5k*0.03 + 7k*0.04 + 9k*0.05 + 11k*0.06 + 5k*0.07 = 104 + 165 + 280 + 450 + 660 + 350 = 2009.
    # Zurich cap tax = 2009 / 10 = 200.9.
    # Combined Zurich (2.19 multiplier) = 200.9 * 2.19 = 440.
    # Total cap tax on 50k = 73.55 + 440 = 513.55 CHF.
    #
    # So Year 1 tax with staggering should be: 513.55 (cap tax) + 530 (AHV) = 1043.55 CHF.
    # Without staggering, it would be just 530 CHF (AHV only).
    config = SimConfig(
        num_runs=1,
        duration_years=3,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=62,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=0.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[50000.0, 50000.0, 50000.0],
        alloc_us_stocks=1.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    return_matrix = np.zeros((config.num_runs, config.duration_years * 12, 5))
    history = run_simulation(config, return_matrix)
    
    # Year 1 tax should be around 1043 CHF (must be > 800)
    assert history['taxes_paid'][1, 0] > 800.0


def test_simulation_engine_invalid_allocation():
    import pytest
    with pytest.raises(ValueError, match="Asset allocations must sum to exactly 1.0"):
        SimConfig(
            num_runs=1,
            duration_years=1,
            inflation_mean=0.0,
            inflation_std=0.0,
            start_age=40,
            dividend_yield=0.0,
            enable_dynamic_expenses=False,
            dynamic_expense_floor_pct=1.0,
            enable_smart_selling=False,
            initial_liquid_wealth=100000.0,
            initial_pillar_2=0.0,
            initial_pillar_3a_accounts=[],
            alloc_us_stocks=0.5, # 50%
            alloc_non_us_stocks=0.4, # 40% (sums to 90%)
            alloc_chf_cash=0.0,
            alloc_gold=0.0,
            alloc_bitcoin=0.0,
            rebalance_strategy='Never',
            rebalance_threshold=0.05,
            annual_base_expenses=10000.0,
            monthly_ahv_pension=0.0,
            cantonal_multiplier=1.0,
            municipal_multiplier=1.19
        )


def test_generate_monte_carlo_returns():
    from src.simulation_engine import generate_monte_carlo_returns
    
    num_runs = 10
    duration_years = 5
    matrix = generate_monte_carlo_returns(
        num_runs=num_runs,
        duration_years=duration_years,
        ret_us=0.07,
        ret_non_us=0.06,
        ret_cash=0.01,
        ret_gold=0.05,
        ret_btc=0.20,
        vol_eq=0.15,
        vol_gold=0.15,
        vol_btc=0.60,
        seed=100
    )
    
    assert matrix.shape == (num_runs, duration_years * 12, 5)
    # Lognormal returns must be strictly greater than -1.0
    assert np.all(matrix > -1.0)


def test_simulation_engine_smart_cash_buffer_stasis():
    # Case A: Cash is sufficient. Stocks should be completely untouched.
    config_a = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=True,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.5,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Monthly', # Try to rebalance monthly
        rebalance_threshold=0.05,
        annual_base_expenses=10000.0, # Deficit = 10000 + 530 = 10530
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Stocks drop 10% in month 0, then 0%. Cash 0%.
    return_matrix = np.zeros((1, 12, 5))
    return_matrix[0, 0, 0] = -0.10
    
    history_a = run_simulation(config_a, return_matrix)
    
    # Year 0 end assets (before tax/expenses):
    # US Stocks: 50k * 0.9 = 45k
    # Cash: 50k
    # Since Cash (50k) > Deficit (10530), Stocks should remain EXACTLY 45k!
    # Cash should be 50k - 10530 = 39470.
    # index 0 is US Stocks, index 2 is Cash.
    final_assets_a = history_a['liquid_assets_by_class'][0, 0]
    assert np.isclose(final_assets_a[0], 45000.0)
    assert np.isclose(final_assets_a[2], 39464.525)
    
    # Case B: Cash is insufficient (60k expenses).
    config_b = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=True,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.5,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Monthly',
        rebalance_threshold=0.05,
        annual_base_expenses=60000.0, # Deficit = 60000 + 530 = 60530
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    history_b = run_simulation(config_b, return_matrix)
    
    # Cash (50k) < Deficit (60530).
    # Cash becomes exactly 0.0.
    # Remaining 10530 is taken from Stocks.
    # Stocks becomes 45000 - 10530 = 34470.
    final_assets_b = history_b['liquid_assets_by_class'][0, 0]
    assert np.isclose(final_assets_b[2], 0.0)
    assert np.isclose(final_assets_b[0], 34470.0)

def test_simulation_engine_quarterly_rebalancing():
    """Test that quarterly rebalancing occurs on months 2, 5, 8, 11."""
    return_matrix = np.zeros((1, 12, 5))
    
    # Month 0: US stocks +50%
    return_matrix[0, 0, 0] = 0.5
    # Month 3: Non-US stocks +50%
    return_matrix[0, 3, 1] = 0.5
    
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.5,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Quarterly',
        rebalance_threshold=0.0,
        enable_smart_selling=False,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    history = run_simulation(config, return_matrix)
    final_assets = history['liquid_assets_by_class'][0, 0]
    
    # Expected:
    # Month 0: US=75k, NonUS=50k
    # Month 2 (rebalance): US=62.5k, NonUS=62.5k
    # Month 3: US=62.5k, NonUS=93.75k (62.5 * 1.5)
    # Month 11 (rebalance): US=78.125k, NonUS=78.125k (minus wealth tax)
    
    assert np.isclose(final_assets[0], 77818.253125)
    assert np.isclose(final_assets[1], 77818.253125)

def test_simulation_engine_withdrawal_rate():
    """Test that we correctly compute a withdrawal rate including both expenses and taxes."""
    return_matrix = np.zeros((1, 12, 5))
    
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        initial_liquid_wealth=1000000.0, # 1M
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=1.0, # 100% cash
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        enable_smart_selling=False,
        annual_base_expenses=50000.0, # 5% base expenses
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    history = run_simulation(config, return_matrix)
    
    # Extract values for year 0
    expenses = history['expenses_paid'][0, 0]
    taxes = history['taxes_paid'][0, 0]
    net_worth = history['net_worth'][0, 0]
    
    # Calculate withdrawal rate manually as UI does
    safe_nw = max(net_worth, 1)
    withdrawal_rate = ((expenses + taxes) / safe_nw) * 100.0
    
    # Expenses should be exactly 50000
    assert np.isclose(expenses, 50000.0)
    
    # Taxes: Wealth tax on 1M (minus expenses) + AHV Non-worker tax.
    # It should be greater than 0.
    assert taxes > 0.0
    
    # The withdrawal rate should logically be higher than just expenses / net_worth
    rate_without_taxes = (expenses / safe_nw) * 100.0
    assert withdrawal_rate > rate_without_taxes

def test_simulation_engine_below_watermark_tracking():
    """Test that below_watermark is accurately tracked when net worth drops."""
    # 2 years simulation. Year 1: 0% return. Year 2: -50% return.
    return_matrix = np.zeros((1, 24, 5))
    return_matrix[0, 12:, :] = -0.057 # ~ -50% compounded over 12 months
    
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=True,
        dynamic_expense_floor_pct=0.5,
        initial_liquid_wealth=1000000.0, # 1M
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0, # 100% US stocks to get hit by the -50%
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        enable_smart_selling=False,
        annual_base_expenses=50000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # 0% inflation
    inflation_matrix = np.zeros((1, 2))
    
    history = run_simulation(config, return_matrix, inflation_matrix)
    
    # Year 0: Starts at 1M, expenses 50k. Should not be below watermark.
    assert history['below_watermark'][0, 0] == False
    
    # Year 1: Takes ~ -50% hit. Should be well below 1M watermark.
    assert history['below_watermark'][1, 0] == True
    
    # Verify dynamic expenses were triggered in year 1
    # Year 0 expenses should be 50k
    assert np.isclose(history['expenses_paid'][0, 0], 50000.0)
    # Year 1 expenses should be cut in half (dynamic_expense_floor_pct = 0.5)
    assert np.isclose(history['expenses_paid'][1, 0], 25000.0)


def test_simulation_engine_threshold_rebalancing():
    # Target: 50% US Stocks, 50% Cash. Threshold: 10% (0.10)
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.5,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Threshold',
        rebalance_threshold=0.10,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Month 0: Stocks +15% -> Stocks 57.5k, Cash 50k (Total 107.5k). Weight 53.49% (Drift 3.49%)
    # Month 1: Stocks +30% -> Stocks 74.75k, Cash 50k (Total 124.75k). Weight 59.92% (Drift 9.92%)
    # Month 2: Stocks +10% -> Stocks 82.225k, Cash 50k (Total 132.225k). Weight 62.18% (Drift 12.18% > 10%) -> Rebalances to 50/50 (66.1125k each)
    # Month 3..11: 0% -> stays 66.1125k each.
    return_matrix = np.zeros((1, 12, 5))
    return_matrix[0, 0, 0] = 0.15
    return_matrix[0, 1, 0] = 0.30
    return_matrix[0, 2, 0] = 0.10
    
    history = run_simulation(config, return_matrix)
    
    final_assets = history['liquid_assets_by_class'][0, 0]
    # Total should be close to 132k (some taxes deducted)
    total = final_assets[0] + final_assets[2]
    assert total > 130000.0
    assert total < 132225.0
    # Should be rebalanced to equal amounts (ratio 50/50)
    assert np.isclose(final_assets[0], final_assets[2])
    assert np.isclose(final_assets[0] / total, 0.5, atol=1e-4)

def test_simulation_engine_threshold_rebalancing_no_trigger():
    # Target: 50% US Stocks, 50% Cash. Threshold: 10% (0.10)
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=100000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.5,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Threshold',
        rebalance_threshold=0.10,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Month 0: Stocks +15% -> Stocks 57.5k, Cash 50k (Total 107.5k). Weight 53.49% (Drift 3.49% < 10%) -> No Rebalance
    # Month 1..11: 0% -> stays 57.5k Stocks, 50k Cash.
    return_matrix = np.zeros((1, 12, 5))
    return_matrix[0, 0, 0] = 0.15
    
    history = run_simulation(config, return_matrix)
    
    final_assets = history['liquid_assets_by_class'][0, 0]
    # Total should be close to 107.5k (some taxes deducted)
    total = final_assets[0] + final_assets[2]
    assert total > 105000.0
    assert total < 107500.0
    # Should NOT be rebalanced (ratio should stay ~53.49% for stocks)
    expected_ratio = 57.5 / 107.5
    assert not np.isclose(final_assets[0], final_assets[2])
    assert np.isclose(final_assets[0] / total, expected_ratio, atol=1e-4)
    assert np.isclose(final_assets[2] / total, 1 - expected_ratio, atol=1e-4)


def test_simulation_engine_ahv_pre_65_inflation():
    # Verify that AHV pension is adjusted for inflation that occurs *before* age 65.
    # Start age: 63. Duration: 3 years.
    # We receive AHV in Year 2 (Age 65).
    # Inflation: 5% annually (constant).
    # bracket_indexation is pinned to 0.0 so the AHV non-worker minimum stays at exactly
    # 530 CHF and the cash arithmetic below remains hand-checkable. Indexed brackets are
    # covered separately by test_bracket_indexation_*.
    config = SimConfig(
        num_runs=1,
        duration_years=3,
        inflation_mean=0.05,
        inflation_std=0.0,
        start_age=63,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=0.001,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=1.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.05,
        annual_base_expenses=0.0,
        monthly_ahv_pension=1000.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0,
        bracket_indexation=0.0
    )
    
    # 0% returns, 5% inflation
    return_matrix = np.zeros((1, 36, 5))
    inflation_matrix = np.full((1, 3), 0.05)
    
    history = run_simulation(config, return_matrix, inflation_matrix)
    
    cash_y0 = history['liquid_assets_by_class'][0, 0, 2]
    cash_y1 = history['liquid_assets_by_class'][1, 0, 2]
    cash_y2 = history['liquid_assets_by_class'][2, 0, 2]
    
    # Year 0: paid minimum AHV (530). Cash: 0.001 - 530 = -529.999
    assert np.isclose(cash_y0, -529.999)
    # Year 1: starts at -529.999, incurs 5% penalty (-26.5), pays 530 tax. Cash: -529.999 * 1.05 - 530 = -1086.49895
    assert np.isclose(cash_y1, -1086.49895)
    # Year 2 (Age 65): starts at -1086.49895.
    # Month 0: incurs 1 month penalty (-4.43), cash becomes -1090.93. Pension added (+1157.63), cash becomes positive (66.70).
    # Month 1..11: pension added monthly. No more penalty.
    # Final cash should be ~12800.57
    assert np.isclose(cash_y2, 12800.5745, atol=1e-2)


def test_get_target_weights_cash_tent():
    config = SimConfig(
        num_runs=1,
        duration_years=10,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=1_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.50,
        alloc_non_us_stocks=0.30,
        alloc_chf_cash=0.10,
        alloc_gold=0.10,
        alloc_bitcoin=0.00,
        rebalance_strategy='Cash Tent',
        rebalance_threshold=0.0,
        annual_base_expenses=40_000.0, # 10 * 40k = 400k peak cash (40% of 1M)
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0,
        tent_duration_years=10
    )
    
    # Year 0: Cash peak = 40%, Non-cash total = 60%. Base non-cash total = 90%.
    # US = 0.50 * (60/90) = 0.33333, Non-US = 0.30 * (60/90) = 0.20, Gold = 0.10 * (60/90) = 0.06667
    w0 = get_target_weights(config, 0)
    assert np.isclose(w0[2], 0.40)
    assert np.isclose(w0[0], 0.50 * (0.60 / 0.90))
    assert np.isclose(np.sum(w0), 1.0)
    
    # Year 5 (Halfway): Cash = 40% - 0.5 * (40% - 10%) = 25%.
    w5 = get_target_weights(config, 5)
    assert np.isclose(w5[2], 0.25)
    assert np.isclose(np.sum(w5), 1.0)
    
    # Year 10 (End of tent): Cash = 10% (base allocation)
    w10 = get_target_weights(config, 10)
    assert np.isclose(w10[2], 0.10)
    assert np.isclose(w10[0], 0.50)
    assert np.isclose(np.sum(w10), 1.0)
    
    # Year 15 (Post-tent): Cash = 10%
    w15 = get_target_weights(config, 15)
    assert np.isclose(w15[2], 0.10)
    assert np.isclose(np.sum(w15), 1.0)


def test_simulation_engine_cash_tent_glidepath():
    # Verify that run_simulation respects Cash Tent glidepath over time
    config = SimConfig(
        num_runs=1,
        duration_years=4,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=100_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.80,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.20,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Cash Tent',
        rebalance_threshold=0.0,
        annual_base_expenses=15_000.0, # 4 * 15k = 60k peak cash (60% of 100k)
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0,
        tent_duration_years=4
    )
    
    # 0% returns, 0% inflation
    return_matrix = np.zeros((1, 48, 5))
    history = run_simulation(config, return_matrix)
    
    # Glidepath: w(y) = 0.60 - (y / 4) * 0.40 -> 60%, 50%, 40%, 30%, then 20% base.
    # The month-11 rebalance sets the allocation HELD DURING THE NEXT YEAR, so it
    # must target w(y + 1). (Previously it targeted w(y), lagging the tent by a year.)

    # Year 0 end: 15k expenses paid -> 85k remaining, rebalanced to w(1) = 50%
    assets_y0 = history['liquid_assets_by_class'][0, 0]
    assert np.isclose(assets_y0[2], 42_500.0)
    assert np.isclose(assets_y0[0], 42_500.0)
    
    # Year 2 end: 45k paid -> 55k remaining, rebalanced to w(3) = 30% Cash (16.5k), 70% US (38.5k)
    assets_y2 = history['liquid_assets_by_class'][2, 0]
    assert np.isclose(assets_y2[2], 16_500.0)
    assert np.isclose(assets_y2[0], 38_500.0)
    
    # Year 3 end (last year): 60k paid -> 40k remaining. No next year exists, so it
    # rebalances to its own w(3) = 30% Cash (12k), 70% US (28k)
    assets_y3 = history['liquid_assets_by_class'][3, 0]
    assert np.isclose(assets_y3[2], 12_000.0)
    assert np.isclose(assets_y3[0], 28_000.0)


def test_cash_tent_reaches_base_weights_after_tent_duration():
    # With a 3-year tent over a 6-year horizon, the allocation held during year 3
    # (i.e. after the year-2 month-11 rebalance) must already be the base weights.
    config = SimConfig(
        num_runs=1, duration_years=6, inflation_mean=0.0, inflation_std=0.0,
        start_age=65, dividend_yield=0.0, enable_smart_selling=False,
        initial_liquid_wealth=100_000.0, alloc_us_stocks=0.80, alloc_chf_cash=0.20,
        rebalance_strategy='Cash Tent', annual_base_expenses=10_000.0,
        cantonal_multiplier=0.0, municipal_multiplier=0.0, tent_duration_years=3,
    )
    history = run_simulation(config, np.zeros((1, 72, 5)))
    by_class = history['liquid_assets_by_class'][:, 0, :]
    cash_share = by_class[:, 2] / by_class.sum(axis=1)
    # End of year y holds w(y + 1): 3/3 * ... -> [w1, w2, w3=base, base, base, base(last)]
    w_peak = 0.30  # 3 * 10k / 100k
    expected = [w_peak - (1 / 3) * (w_peak - 0.20), w_peak - (2 / 3) * (w_peak - 0.20), 0.20, 0.20, 0.20, 0.20]
    assert np.allclose(cash_share, expected)


def test_estimate_year_0_taxes_includes_taxes():
    # Verify that get_target_weights includes estimated taxes in cash tent peak
    config = SimConfig(
        num_runs=1,
        duration_years=10,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=50, # under 65 -> AHV non-worker tax applicable
        dividend_yield=0.02,
        enable_dynamic_expenses=False,
        dynamic_expense_floor_pct=1.0,
        enable_smart_selling=False,
        initial_liquid_wealth=1_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.50,
        alloc_non_us_stocks=0.30,
        alloc_chf_cash=0.10,
        alloc_gold=0.10,
        alloc_bitcoin=0.00,
        rebalance_strategy='Cash Tent',
        rebalance_threshold=0.0,
        annual_base_expenses=50_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19,
        tent_duration_years=10
    )
    
    w0 = get_target_weights(config, 0)
    # Peak cash = 10 * (50,000 expenses + est_taxes). Must be strictly > 10 * 50,000 / 1,000,000 = 0.50
    assert w0[2] > 0.50
    assert np.isclose(np.sum(w0), 1.0)


def test_simulation_engine_vanguard_dynamic_spending():
    # Test Vanguard Dynamic Spending strategy
    # Base expenses = 40,000 CHF. Target rate = 4% (0.04), Floor = 2.5%, Ceiling = 5.0%
    config = SimConfig(
        num_runs=1,
        duration_years=3,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        spending_strategy="Vanguard Dynamic",
        vanguard_target_rate=0.04,
        vanguard_floor_pct=0.025,
        vanguard_ceiling_pct=0.05,
        initial_liquid_wealth=1_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=1.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        annual_base_expenses=40_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0
    )

    # Return matrix:
    # Year 0: 0% returns -> Net worth before expenses ~ 1,000,000. Target spending = 0.04 * 1M = 40,000.
    # Prior inflated spending = 40,000. Floor = 39,000. Ceiling = 42,000. Target 40k -> Expenses = 40,000.
    return_matrix = np.zeros((1, 36, 5))
    history = run_simulation(config, return_matrix)
    
    exp_y0 = history['expenses_paid'][0, 0]
    assert np.isclose(exp_y0, 40_000.0)


def test_simulation_engine_dynamic_floor_and_ceiling():
    # Test Dynamic (Floor & Ceiling) strategy
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        spending_strategy="Dynamic (Floor & Ceiling)",
        dynamic_expense_floor_pct=0.80,
        dynamic_expense_ceiling_pct=1.20,
        initial_liquid_wealth=1_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=1.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        annual_base_expenses=50_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0
    )

    return_matrix = np.zeros((1, 24, 5))
    history = run_simulation(config, return_matrix)
    
    # Net worth equals initial net worth -> spending equals base expenses = 50,000
    exp_y0 = history['expenses_paid'][0, 0]
    assert np.isclose(exp_y0, 50_000.0)


def test_estimate_year_0_taxes_inclusion():
    # Verify that estimate_year_0_taxes includes wealth tax and income tax for tax-inclusive TWR
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=50,
        dividend_yield=0.02,
        initial_liquid_wealth=2_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.3,
        alloc_chf_cash=0.2,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        annual_base_expenses=80_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )

    est_tax = estimate_year_0_taxes(config)
    assert est_tax > 0.0
    
    total_outflow = config.annual_base_expenses + est_tax
    twr_pct = (total_outflow / config.initial_liquid_wealth) * 100.0
    assert twr_pct > (config.annual_base_expenses / config.initial_liquid_wealth) * 100.0


def test_simulation_engine_start_at_age_65_immediate_liquidation():
    # Starting at age 65 should liquidate all Pillar 2 and Pillar 3a in Year 0 Month 0
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        initial_liquid_wealth=100_000.0,
        initial_pillar_2=300_000.0,
        initial_pillar_3a_accounts=[50_000.0, 50_000.0],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.5,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        annual_base_expenses=0.0,
        monthly_ahv_pension=2_000.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    return_matrix = np.zeros((1, 24, 5))
    history = run_simulation(config, return_matrix)
    
    # Pillar 2 and 3a should be 0 at end of year 0 (liquidated)
    assert history['pillar_2'][0, 0] == 0.0
    assert history['pillar_3a'][0, 0] == 0.0
    # Liquid assets should have absorbed 400k (minus capital withdrawal taxes) plus initial 100k + AHV
    assert history['liquid_assets'][0, 0] > 400_000.0
    assert history['taxes_paid'][0, 0] > 0.0


def test_simulation_engine_staggered_pillar_3a_multiple_accounts():
    # Start at age 60 with 3 accounts: Account 0 liquidates at 60 (yr 0), Account 1 at 61 (yr 1), Account 2 at 62 (yr 2)
    config = SimConfig(
        num_runs=1,
        duration_years=4,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=60,
        dividend_yield=0.0,
        initial_liquid_wealth=100_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[20_000.0, 20_000.0, 20_000.0],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.5,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        rebalance_threshold=0.0,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    return_matrix = np.zeros((1, 48, 5))
    history = run_simulation(config, return_matrix)
    
    # Year 0 (age 60): 1 account liquidated, 2 remain (40k)
    assert np.isclose(history['pillar_3a'][0, 0], 40_000.0)
    # Year 1 (age 61): 2nd account liquidated, 1 remains (20k)
    assert np.isclose(history['pillar_3a'][1, 0], 20_000.0)
    # Year 2 (age 62): 3rd account liquidated, 0 remain (0k)
    assert np.isclose(history['pillar_3a'][2, 0], 0.0)
    assert np.isclose(history['pillar_3a'][3, 0], 0.0)


def test_historic_returns_matrix_valid():
    from src.historic_returns import MONTHLY_CASH_CHF
    # 50-year duration
    matrix = get_historic_return_matrix(50)
    expected_runs = len(HISTORIC_RETURNS) - 50 + 1
    assert matrix.shape == (expected_runs, 50 * 12, 5)
    # Check that cash return matches empirical Swiss retail cash rate series (floored at 0%)
    assert np.allclose(matrix[0, :, 2], MONTHLY_CASH_CHF[: 50 * 12])
    assert np.all(matrix[:, :, 2] >= 0.0)


def test_historic_returns_matrix_invalid():
    import pytest
    with pytest.raises(ValueError, match="Duration must be a positive integer"):
        get_historic_return_matrix(0)
    with pytest.raises(ValueError, match="Duration must be a positive integer"):
        get_historic_return_matrix(-5)
    with pytest.raises(ValueError, match="exceeds available historic data"):
        get_historic_return_matrix(len(HISTORIC_RETURNS) + 10)


def test_sim_config_validation_invalid_runs_or_duration():
    import pytest
    with pytest.raises(ValueError, match="must be positive integers"):
        SimConfig(num_runs=0, duration_years=5, inflation_mean=0.0, inflation_std=0.0, start_age=40, dividend_yield=0.0)
    with pytest.raises(ValueError, match="must be positive integers"):
        SimConfig(num_runs=10, duration_years=-1, inflation_mean=0.0, inflation_std=0.0, start_age=40, dividend_yield=0.0)


def test_sim_config_validation_negative_allocations():
    import pytest
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(
            num_runs=1, duration_years=5, inflation_mean=0.0, inflation_std=0.0, start_age=40, dividend_yield=0.0,
            alloc_us_stocks=1.2, alloc_non_us_stocks=-0.2, alloc_chf_cash=0.0, alloc_gold=0.0, alloc_bitcoin=0.0
        )


def test_sim_config_validation_negative_age_and_vanguard():
    import pytest
    with pytest.raises(ValueError, match="start_age.*cannot be negative"):
        SimConfig(num_runs=1, duration_years=5, inflation_mean=0.0, inflation_std=0.0, start_age=-5, dividend_yield=0.0, alloc_us_stocks=1.0)
    with pytest.raises(ValueError, match="tent_duration_years.*cannot be negative"):
        SimConfig(num_runs=1, duration_years=5, inflation_mean=0.0, inflation_std=0.0, start_age=40, dividend_yield=0.0, tent_duration_years=-1, alloc_us_stocks=1.0)
    with pytest.raises(ValueError, match="Vanguard Dynamic Spending parameters cannot be negative"):
        SimConfig(num_runs=1, duration_years=5, inflation_mean=0.0, inflation_std=0.0, start_age=40, dividend_yield=0.0, vanguard_floor_pct=-0.1, alloc_us_stocks=1.0)


def test_estimate_year_0_taxes_zero_and_edge_values():
    # Zero liquid wealth
    cfg_zero = SimConfig(
        num_runs=1, duration_years=1, inflation_mean=0.0, inflation_std=0.0,
        start_age=40, dividend_yield=0.0, initial_liquid_wealth=0.0, alloc_us_stocks=1.0
    )
    assert estimate_year_0_taxes(cfg_zero) >= 0.0

    # Start age >= 65 (no AHV non-worker tax, AHV pension received)
    cfg_ret = SimConfig(
        num_runs=1, duration_years=1, inflation_mean=0.0, inflation_std=0.0,
        start_age=65, dividend_yield=0.0, initial_liquid_wealth=1_000_000.0,
        alloc_us_stocks=1.0, monthly_ahv_pension=2000.0, cantonal_multiplier=1.0, municipal_multiplier=1.19
    )
    tax = estimate_year_0_taxes(cfg_ret)
    assert tax > 0.0


def test_vanguard_spending_depleted_portfolio_floor():
    # In a depleted portfolio scenario, Vanguard Dynamic spending still respects the floor
    config = SimConfig(
        num_runs=1,
        duration_years=3,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=50,
        dividend_yield=0.0,
        spending_strategy="Vanguard Dynamic",
        vanguard_target_rate=0.04,
        vanguard_floor_pct=0.05,
        vanguard_ceiling_pct=0.05,
        annual_base_expenses=50_000.0,
        initial_liquid_wealth=0.0, # Zero wealth
        alloc_us_stocks=1.0
    )
    return_matrix = np.zeros((1, 36, 5))
    history = run_simulation(config, return_matrix)
    
    # Year 0 spending is bounded by floor: 50,000 * 0.95 = 47,500
    assert np.isclose(history['expenses_paid'][0, 0], 47_500.0)
    # Year 1 spending is 47,500 * 0.95 = 45,125
    assert np.isclose(history['expenses_paid'][1, 0], 45_125.0)


def test_generate_bootstrapped_returns_shape_and_values():
    from src.historic_returns import MONTHLY_CASH_CHF
    matrix = generate_bootstrapped_returns(num_runs=50, duration_years=30, seed=123)
    assert matrix.shape == (50, 360, 5)
    assert not np.isnan(matrix).any()
    # Check cash return is drawn from empirical Swiss retail rates (non-negative and bounded by historical max)
    assert np.all(matrix[:, :, 2] >= 0.0)
    assert np.all(matrix[:, :, 2] <= np.max(MONTHLY_CASH_CHF) + 1e-9)
    # Check US and Non-US equities have valid return ranges
    assert np.all(matrix[:, :, 0] > -1.0)
    assert np.all(matrix[:, :, 1] > -1.0)
    
    # Check empirical inflation bootstrapping
    inf_matrix = generate_bootstrapped_inflation(num_runs=50, duration_years=30, seed=123)
    assert inf_matrix.shape == (50, 30)
    assert not np.isnan(inf_matrix).any()
    assert np.all(inf_matrix > -0.5)


def test_historic_inflation_matrix_valid():
    inf_matrix = get_historic_inflation_matrix(50)
    expected_runs = len(HISTORIC_SWISS_INFLATION) - 50 + 1
    assert inf_matrix.shape == (expected_runs, 50)
    assert not np.isnan(inf_matrix).any()


def test_generate_bootstrapped_returns_reproducibility():
    m1 = generate_bootstrapped_returns(num_runs=20, duration_years=10, seed=42)
    m2 = generate_bootstrapped_returns(num_runs=20, duration_years=10, seed=42)
    assert np.array_equal(m1, m2)


def test_generate_bootstrapped_returns_invalid_inputs():
    import pytest
    with pytest.raises(ValueError, match="must be positive integers"):
        generate_bootstrapped_returns(num_runs=0, duration_years=10)
    with pytest.raises(ValueError, match="must be positive integers"):
        generate_bootstrapped_returns(num_runs=10, duration_years=-2)


def test_bootstrapping_simulation_integration():
    config = SimConfig(
        num_runs=100,
        duration_years=25,
        inflation_mean=0.02,
        inflation_std=0.01,
        start_age=45,
        dividend_yield=0.015,
        initial_liquid_wealth=1_500_000.0,
        initial_pillar_2=300_000.0,
        initial_pillar_3a_accounts=[50_000.0, 50_000.0],
        alloc_us_stocks=0.6,
        alloc_non_us_stocks=0.3,
        alloc_chf_cash=0.1,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Yearly',
        annual_base_expenses=70_000.0,
        monthly_ahv_pension=2200.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    boot_return_matrix = generate_bootstrapped_returns(config.num_runs, config.duration_years, seed=42)
    inflation_matrix = generate_bootstrapped_inflation(config.num_runs, config.duration_years, seed=42)
    
    history = run_simulation(config, boot_return_matrix, inflation_matrix)
    
    assert history['net_worth'].shape == (25, 100)
    assert history['liquid_assets'].shape == (25, 100)
    assert not np.isnan(history['net_worth']).any()
    
    # Verify success rate computation
    run_final_inf_factor = np.prod(1 + inflation_matrix, axis=1)
    target_ending_nw = 0.5 * (history['initial_net_worth'] * run_final_inf_factor)
    success_rate = np.mean(history['net_worth'][-1, :] > target_ending_nw) * 100.0
    assert 0.0 <= success_rate <= 100.0


def test_estimate_year_0_taxes_age_65_with_pensions():
    # If starting at age 65 with 0 liquid wealth but 1M in Pillar 2,
    # estimate_year_0_taxes should compute taxes on the liquidated Pillar 2 balance.
    config = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.015,
        initial_liquid_wealth=0.0,
        initial_pillar_2=1_000_000.0,
        initial_pillar_3a_accounts=[50_000.0],
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.3,
        alloc_chf_cash=0.2,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        annual_base_expenses=40_000.0,
        monthly_ahv_pension=2000.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    taxes = estimate_year_0_taxes(config)
    # Must be strictly positive (income tax from dividends/interest/AHV + wealth tax)
    assert taxes > 0.0


def test_get_target_weights_cash_tent_age_65():
    # If starting at age 65 with 0 liquid wealth and 1M in Pillar 2 under Cash Tent
    config = SimConfig(
        num_runs=1,
        duration_years=10,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.015,
        initial_liquid_wealth=0.0,
        initial_pillar_2=1_000_000.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.6,
        alloc_non_us_stocks=0.2,
        alloc_chf_cash=0.2,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Cash Tent',
        annual_base_expenses=50_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19,
        tent_duration_years=5
    )
    
    w0 = get_target_weights(config, 0)
    # Peak cash weight should be > base cash (0.2) because 5 years * (50k + taxes) is ~30%
    assert w0[2] > 0.20
    assert np.isclose(np.sum(w0), 1.0)


def test_simulation_engine_no_rebalance_when_bankrupt():
    # If a portfolio is in deep debt (-50k), rebalancing must not multiply debt by target weights
    # into US stocks, Non-US stocks, Gold, or Bitcoin. All debt must remain in CHF cash.
    strategies = ['Monthly', 'Quarterly', 'Yearly', 'Cash Tent', 'Threshold']
    
    for strat in strategies:
        config = SimConfig(
            num_runs=1,
            duration_years=1,
            inflation_mean=0.0,
            inflation_std=0.0,
            start_age=40,
            dividend_yield=0.0,
            initial_liquid_wealth=0.0,
            initial_pillar_2=0.0,
            initial_pillar_3a_accounts=[],
            alloc_us_stocks=0.5,
            alloc_non_us_stocks=0.3,
            alloc_chf_cash=0.2,
            alloc_gold=0.0,
            alloc_bitcoin=0.0,
            rebalance_strategy=strat,
            rebalance_threshold=0.01,
            enable_smart_selling=False,
            annual_base_expenses=50_000.0,
            monthly_ahv_pension=0.0,
            cantonal_multiplier=1.0,
            municipal_multiplier=1.19
        )
        
        return_matrix = np.zeros((1, 12, 5))
        history = run_simulation(config, return_matrix)
        
        final_assets = history['liquid_assets_by_class'][0, 0]
        # Non-cash assets must remain 0
        assert np.isclose(final_assets[0], 0.0), f"Failed for strategy {strat}"
        assert np.isclose(final_assets[1], 0.0), f"Failed for strategy {strat}"
        assert np.isclose(final_assets[3], 0.0), f"Failed for strategy {strat}"
        assert np.isclose(final_assets[4], 0.0), f"Failed for strategy {strat}"
        # All debt in CHF cash
        assert final_assets[2] < 0.0, f"Failed for strategy {strat}"


def test_vanguard_spending_dynamic_bounds_progression():
    # Verify that Vanguard Dynamic spending recalculates spending in year 1 and 2
    # with floor (-5%) and ceiling (+5%) bounds
    config = SimConfig(
        num_runs=1,
        duration_years=3,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=50,
        dividend_yield=0.0,
        spending_strategy="Vanguard Dynamic",
        vanguard_target_rate=0.04, # 4% target
        vanguard_floor_pct=0.05,   # max -5%
        vanguard_ceiling_pct=0.05, # max +5%
        initial_liquid_wealth=1_000_000.0, # 1M
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=1.0,
        alloc_non_us_stocks=0.0,
        alloc_chf_cash=0.0,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Never',
        annual_base_expenses=40_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=1.0,
        municipal_multiplier=1.19
    )
    
    # Year 0: Market booms +50% in month 0 -> Net worth before expenses is 1.5M.
    # Target spending = 0.04 * 1.5M = 60,000.
    # Prior expense = 40,000. Ceiling = 40,000 * 1.05 = 42,000.
    # Spending should be capped at ceiling = 42,000!
    return_matrix = np.zeros((1, 36, 5))
    return_matrix[0, 0, 0] = 0.50
    
    history = run_simulation(config, return_matrix)
    
    assert np.isclose(history['expenses_paid'][0, 0], 42_000.0)


def test_beginning_of_year_withdrawal_rate_calculation():
    # Verify that withdrawal rate computed against beginning-of-year net worth is stable and accurate
    net_worth_end = np.array([[10_000.0]])
    expenses = np.array([[40_000.0]])
    taxes = np.array([[10_000.0]])
    
    # Beginning of year portfolio value was 10k + 40k + 10k = 60k
    safe_start_nw = np.maximum(net_worth_end + expenses + taxes, 1.0)
    wr = ((expenses + taxes) / safe_start_nw) * 100.0
    
    # Withdrawal rate should be (50k / 60k) * 100 = 83.33%, NOT (50k / 10k) * 100 = 500%
    assert np.isclose(wr[0, 0], (50_000.0 / 60_000.0) * 100.0)


def test_trace_labeling_bootstrapping_vs_backtesting():
    # Verify that Bootstrapping (e.g. 1000 runs) does not try to index HISTORIC_YEARS beyond 104
    duration_years = 50
    
    # Simulate the UI trace generation logic
    for title, num_runs in [("Historic Backtesting", 55), ("Historic Bootstrapping", 1000), ("Monte Carlo", 1000)]:
        is_historic_backtest = "Historic Backtesting" in title or "Historic Returns" in title
        max_traces_to_plot = int(num_runs) if is_historic_backtest else min(100, int(num_runs))
        
        traces = []
        for i in range(max_traces_to_plot):
            if is_historic_backtest:
                start_year = int(HISTORIC_YEARS[i])
                end_year = start_year + duration_years - 1
                trace_name = f"Cohort: {start_year} - {end_year}"
            else:
                trace_name = f"Run {i+1}"
            traces.append(trace_name)
            
        if is_historic_backtest:
            assert len(traces) == 55
            assert traces[0] == f"Cohort: {HISTORIC_YEARS[0]} - {HISTORIC_YEARS[0] + duration_years - 1}"
        else:
            assert len(traces) == 100
            assert traces[0] == "Run 1"
            assert traces[99] == "Run 100"


def test_generate_bootstrapped_data_joint_alignment():
    num_runs = 50
    duration_years = 20
    ret_matrix, inf_matrix = generate_bootstrapped_data(num_runs, duration_years, seed=123)
    
    assert ret_matrix.shape == (50, 240, 5)
    assert inf_matrix.shape == (50, 20)
    assert not np.isnan(ret_matrix).any()
    assert not np.isnan(inf_matrix).any()
    
    # Check consistency with individual wrappers
    ret_ind = generate_bootstrapped_returns(num_runs, duration_years, seed=123)
    inf_ind = generate_bootstrapped_inflation(num_runs, duration_years, seed=123)
    assert np.array_equal(ret_matrix, ret_ind)
    assert np.array_equal(inf_matrix, inf_ind)


def test_sim_config_validation_extended_negative_inputs():
    import pytest
    base_args = dict(
        num_runs=10, duration_years=5, inflation_mean=0.02, inflation_std=0.01,
        start_age=40, dividend_yield=0.015, alloc_us_stocks=1.0
    )
    
    # Negative multipliers
    with pytest.raises(ValueError, match="Tax multipliers cannot be negative"):
        SimConfig(**base_args, cantonal_multiplier=-0.1)
    with pytest.raises(ValueError, match="Tax multipliers cannot be negative"):
        SimConfig(**base_args, municipal_multiplier=-0.5)
        
    # Negative expenses and income
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(**base_args, annual_base_expenses=-1000.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(**base_args, monthly_ahv_pension=-500.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        args = dict(base_args)
        args['dividend_yield'] = -0.01
        SimConfig(**args)
        
    # Negative assets
    with pytest.raises(ValueError, match="Initial asset balances cannot be negative"):
        SimConfig(**base_args, initial_liquid_wealth=-10_000.0)
    with pytest.raises(ValueError, match="Initial asset balances cannot be negative"):
        SimConfig(**base_args, initial_pillar_2=-50_000.0)
    with pytest.raises(ValueError, match="Initial asset balances cannot be negative"):
        SimConfig(**base_args, initial_pillar_3a_accounts=[10_000.0, -500.0])
        
    # Negative threshold and dynamic spending bounds
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(**base_args, rebalance_threshold=-0.05)
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(**base_args, dynamic_expense_floor_pct=-0.1)
    with pytest.raises(ValueError, match="cannot be negative"):
        SimConfig(**base_args, dynamic_expense_ceiling_pct=-0.2)


def test_seed_reproducibility_and_variation():
    from src.simulation_engine import generate_monte_carlo_returns

    # 1. Bootstrapping reproducibility with same seed
    b_ret1, b_inf1 = generate_bootstrapped_data(num_runs=20, duration_years=10, seed=42)
    b_ret2, b_inf2 = generate_bootstrapped_data(num_runs=20, duration_years=10, seed=42)
    assert np.array_equal(b_ret1, b_ret2)
    assert np.array_equal(b_inf1, b_inf2)

    # Bootstrapping variation with different seed
    b_ret3, b_inf3 = generate_bootstrapped_data(num_runs=20, duration_years=10, seed=999)
    assert not np.array_equal(b_ret1, b_ret3)
    assert not np.array_equal(b_inf1, b_inf3)

    # 2. Monte Carlo reproducibility with same seed
    mc_args = dict(
        num_runs=20,
        duration_years=10,
        ret_us=0.07,
        ret_non_us=0.06,
        ret_cash=0.01,
        ret_gold=0.06,
        ret_btc=0.10,
        vol_eq=0.15,
        vol_gold=0.15,
        vol_btc=0.60,
    )
    mc_ret1 = generate_monte_carlo_returns(**mc_args, seed=42)
    mc_ret2 = generate_monte_carlo_returns(**mc_args, seed=42)
    assert np.array_equal(mc_ret1, mc_ret2)

    # Monte Carlo variation with different seed
    mc_ret3 = generate_monte_carlo_returns(**mc_args, seed=999)
    assert not np.array_equal(mc_ret1, mc_ret3)


def test_monte_carlo_inflation_rng_independence_and_validation():
    import pytest
    from src.simulation_engine import generate_monte_carlo_inflation, generate_monte_carlo_returns

    num_runs = 200
    duration_years = 20
    seed = 42

    mc_ret = generate_monte_carlo_returns(
        num_runs=num_runs,
        duration_years=duration_years,
        ret_us=0.07,
        ret_non_us=0.06,
        ret_cash=0.01,
        ret_gold=0.06,
        ret_btc=0.10,
        vol_eq=0.15,
        vol_gold=0.15,
        vol_btc=0.60,
        seed=seed
    )
    mc_inf = generate_monte_carlo_inflation(
        num_runs=num_runs,
        duration_years=duration_years,
        inflation_mean=0.025,
        inflation_std=0.01,
        seed=seed
    )

    assert mc_inf.shape == (num_runs, duration_years)
    # Reconstruct standard normal Z variates from US Stock month 0..19 and compare with inflation Z variates
    drift_us = np.log(1 + 0.07) - 0.5 * 0.15**2
    us_log_ret = np.log(mc_ret[:, :duration_years, 0] + 1.0)
    z_us = (us_log_ret - drift_us / 12.0) / (0.15 / np.sqrt(12.0))
    z_inf = (mc_inf - 0.025) / 0.01

    # Correlation must NOT be 1.0 (which occurred when both used default_rng(seed) without stream offset)
    corr = np.corrcoef(z_us.flatten(), z_inf.flatten())[0, 1]
    assert abs(corr) < 0.2

    # Validation checks
    with pytest.raises(ValueError, match="num_runs must be positive"):
        generate_monte_carlo_inflation(0, 10)
    with pytest.raises(ValueError, match="duration_years must be positive"):
        generate_monte_carlo_inflation(10, 0)
    with pytest.raises(ValueError, match="inflation_std cannot be negative"):
        generate_monte_carlo_inflation(10, 10, inflation_std=-0.01)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        generate_monte_carlo_inflation(10, 10, seed=-1)
    with pytest.raises(ValueError, match="Volatilities cannot be negative"):
        generate_monte_carlo_returns(10, 10, 0.07, 0.06, 0.01, 0.06, 0.10, -0.15, 0.15, 0.60)
    with pytest.raises(ValueError, match="Expected annual returns must be greater than -100%"):
        generate_monte_carlo_returns(10, 10, -1.0, 0.06, 0.01, 0.06, 0.10, 0.15, 0.15, 0.60)


def test_historic_lognormal_params_round_trips_through_mc_generator():
    """Params estimated from history must regenerate that history's arithmetic mean and log-vol."""
    import pytest
    from src.historic_returns import HISTORIC_RETURNS_US_CHF
    from src.simulation_engine import generate_monte_carlo_returns, historic_lognormal_params

    mean, vol = historic_lognormal_params(HISTORIC_RETURNS_US_CHF)
    assert mean == pytest.approx(float(np.mean(HISTORIC_RETURNS_US_CHF)))
    # Log-vol is below simple-return vol for a volatile series (the old UI conflated them).
    assert vol < float(np.std(HISTORIC_RETURNS_US_CHF, ddof=1))

    mc = generate_monte_carlo_returns(4000, 20, mean, 0.06, 0.01, 0.05, 0.07, vol, 0.15, 0.5, seed=7)
    annual = np.prod(1.0 + mc[:, :, 0].reshape(4000, 20, 12), axis=2) - 1.0
    assert float(annual.mean()) == pytest.approx(mean, abs=0.003)
    assert float(np.log1p(annual).std()) == pytest.approx(vol, abs=0.003)


def test_historic_lognormal_params_validation():
    import pytest
    from src.simulation_engine import historic_lognormal_params

    with pytest.raises(ValueError, match="at least 2"):
        historic_lognormal_params(np.array([0.05]))
    with pytest.raises(ValueError, match="greater than -100%"):
        historic_lognormal_params(np.array([0.05, -1.0]))


def test_year_0_liquid_wealth_and_taxes_at_age_60_to_64():
    from src.simulation_engine import _get_year_0_liquid_wealth, estimate_year_0_taxes, get_target_weights
    from src.tax_engine import calculate_capital_withdrawal_tax

    # At start_age=62, the first eligible Pillar 3a account (60 + 0 <= 62) is liquidated in Year 0 Month 0
    config_with_3a = SimConfig(
        num_runs=1,
        duration_years=5,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=62,
        dividend_yield=0.015,
        initial_liquid_wealth=500_000.0,
        initial_pillar_2=200_000.0,
        initial_pillar_3a_accounts=[50_000.0, 50_000.0],
        alloc_us_stocks=0.60,
        alloc_non_us_stocks=0.20,
        alloc_chf_cash=0.20,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy="Cash Tent",
        rebalance_threshold=0.0,
        annual_base_expenses=40_000.0,
        monthly_ahv_pension=2000.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19,
        tent_duration_years=5
    )
    config_no_3a = SimConfig(
        num_runs=1,
        duration_years=5,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=62,
        dividend_yield=0.015,
        initial_liquid_wealth=500_000.0,
        initial_pillar_2=200_000.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.60,
        alloc_non_us_stocks=0.20,
        alloc_chf_cash=0.20,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy="Cash Tent",
        rebalance_threshold=0.0,
        annual_base_expenses=40_000.0,
        monthly_ahv_pension=2000.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19,
        tent_duration_years=5
    )

    cap_tax_50k = float(calculate_capital_withdrawal_tax(50_000.0, 0.95, 1.19))
    assert np.isclose(_get_year_0_liquid_wealth(config_with_3a), 500_000.0 + 50_000.0 - cap_tax_50k)

    tax_with_3a = estimate_year_0_taxes(config_with_3a)
    tax_no_3a = estimate_year_0_taxes(config_no_3a)
    # Ongoing Year 0 taxes (income, wealth, AHV) are higher with the liquidated 3a account added to liquid wealth
    assert tax_with_3a > tax_no_3a

    # Verify Cash Tent target weights account for the higher Year 0 liquid wealth
    w0 = get_target_weights(config_with_3a, 0)
    assert np.isclose(np.sum(w0), 1.0)

    # Verify estimate_year_0_taxes + capital_withdrawal_tax matches actual run_simulation Year 0 taxes_paid under base weights
    config_yearly = SimConfig(
        **{**config_with_3a.__dict__, "rebalance_strategy": "Yearly", "cash_rate": 0.0}
    )
    return_matrix = np.zeros((1, 60, 5))
    inflation_matrix = np.zeros((1, 5))
    history = run_simulation(config_yearly, return_matrix, inflation_matrix)
    assert np.isclose(estimate_year_0_taxes(config_yearly) + cap_tax_50k, history['taxes_paid'][0, 0], rtol=1e-5)


def test_sim_config_strategy_and_seed_validation():
    import pytest
    base_args = dict(
        num_runs=5, duration_years=5, inflation_mean=0.02, inflation_std=0.01,
        start_age=40, dividend_yield=0.015, alloc_us_stocks=1.0
    )
    with pytest.raises(ValueError, match="Invalid spending_strategy"):
        SimConfig(**base_args, spending_strategy="InvalidStrategy")
    with pytest.raises(ValueError, match="Invalid rebalance_strategy"):
        SimConfig(**base_args, rebalance_strategy="BiWeekly")
    with pytest.raises(ValueError, match="inflation_std.*cannot be negative"):
        args = dict(base_args)
        args["inflation_std"] = -0.01
        SimConfig(**args)
    with pytest.raises(ValueError, match="seed.*cannot be negative"):
        SimConfig(**base_args, seed=-5)


def test_run_simulation_deterministic_fallback_inflation_and_deflation_clamp():
    config1 = SimConfig(
        num_runs=5,
        duration_years=3,
        inflation_mean=0.02,
        inflation_std=0.015,
        start_age=45,
        dividend_yield=0.01,
        alloc_us_stocks=1.0,
        annual_base_expenses=50_000.0,
        seed=77
    )
    config2 = SimConfig(
        num_runs=5,
        duration_years=3,
        inflation_mean=0.02,
        inflation_std=0.015,
        start_age=45,
        dividend_yield=0.01,
        alloc_us_stocks=1.0,
        annual_base_expenses=50_000.0,
        seed=77
    )
    return_matrix = np.zeros((5, 36, 5))
    # When inflation_matrix is None, run_simulation must be 100% deterministic via config.seed
    h1 = run_simulation(config1, return_matrix, inflation_matrix=None)
    h2 = run_simulation(config2, return_matrix, inflation_matrix=None)
    assert np.array_equal(h1['net_worth'], h2['net_worth'])
    assert np.array_equal(h1['expenses_paid'], h2['expenses_paid'])

    # Extreme deflation (-150% inflation) must be clamped so inflation_factors stay strictly positive (> 0)
    extreme_deflation = np.full((5, 3), -1.50)
    h_defl = run_simulation(config1, return_matrix, inflation_matrix=extreme_deflation)
    assert np.all(h_defl['expenses_paid'] > 0.0)


def test_historic_and_bootstrapped_lognormal_bounds_and_seeds():
    from src.historic_returns import MONTHLY_CASH_CHF
    # Synthetic Bitcoin (idx 4) must use lognormal returns (> -1.0) and respect seed
    hist_m1 = get_historic_return_matrix(10, seed=42)
    hist_m2 = get_historic_return_matrix(10, seed=42)
    hist_m3 = get_historic_return_matrix(10, seed=99)

    assert np.array_equal(hist_m1, hist_m2)
    assert not np.array_equal(hist_m1[:, :, 4], hist_m3[:, :, 4])
    assert np.all(hist_m1 > -1.0)

    boot_ret, _ = generate_bootstrapped_data(200, 30, seed=42)
    assert np.all(boot_ret > -1.0)
    # Cash monthly return (idx 2) must match empirical Swiss retail cash rates (floored at 0.0)
    assert np.allclose(hist_m1[0, :, 2], MONTHLY_CASH_CHF[: 10 * 12])
    assert np.all(boot_ret[:, :, 2] >= 0.0)


def test_vanguard_dynamic_net_target_rate_no_tax_double_counting():
    from src.simulation_engine import estimate_year_0_taxes

    # Verify that setting vanguard_target_rate = annual_base_expenses / initial_net_worth
    # produces Year 0 living expenses equal to annual_base_expenses (85,000 CHF)
    # and total outflow equal to 85,000 + Year 0 taxes (no tax double-counting)
    total_nw = 3_000_000.0
    base_exp = 85_000.0
    config = SimConfig(
        num_runs=1,
        duration_years=2,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.015,
        cash_rate=0.0,
        spending_strategy="Vanguard Dynamic",
        vanguard_target_rate=base_exp / total_nw,
        vanguard_floor_pct=0.05,
        vanguard_ceiling_pct=0.05,
        initial_liquid_wealth=2_450_000.0,
        initial_pillar_2=450_000.0,
        initial_pillar_3a_accounts=[20_000.0] * 5,
        alloc_us_stocks=0.50,
        alloc_non_us_stocks=0.30,
        alloc_chf_cash=0.10,
        alloc_gold=0.03,
        alloc_bitcoin=0.07,
        rebalance_strategy="Never",
        rebalance_threshold=0.0,
        annual_base_expenses=base_exp,
        monthly_ahv_pension=2000.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19
    )
    return_matrix = np.zeros((1, 24, 5))
    inflation_matrix = np.zeros((1, 2))
    history = run_simulation(config, return_matrix, inflation_matrix)

    assert np.isclose(history['expenses_paid'][0, 0], base_exp)
    est_tax = estimate_year_0_taxes(config)
    assert np.isclose(history['taxes_paid'][0, 0], est_tax, rtol=1e-3)


def test_bootstrapping_five_year_contiguous_blocks():
    import pytest
    from src.historic_returns import (
        HISTORIC_RETURNS_NON_US_CHF,
        HISTORIC_RETURNS_US_CHF,
        HISTORIC_SWISS_INFLATION,
        generate_bootstrapped_data,
        generate_bootstrapped_inflation,
        generate_bootstrapped_returns,
    )

    num_runs = 25
    duration_years = 12  # Non-multiple of 5 (2 full 5-year blocks + 2 years from a 3rd block)
    ret_matrix, inf_matrix = generate_bootstrapped_data(
        num_runs=num_runs, duration_years=duration_years, seed=42, stationary=False
    )

    assert ret_matrix.shape == (num_runs, duration_years * 12, 5)
    assert inf_matrix.shape == (num_runs, duration_years)

    # Reconstruct annual returns by compounding each year's 12 REAL monthly returns.
    def _to_annual(monthly_col):
        reshaped = monthly_col.reshape(num_runs, duration_years, 12)
        return np.prod(1.0 + reshaped, axis=2) - 1.0

    sampled_us_annual = _to_annual(ret_matrix[:, :, 0])
    sampled_non_us_annual = _to_annual(ret_matrix[:, :, 1])

    # For every run, each 5-year block (years 0..4 and years 5..9) must match an exact contiguous 5-year slice
    # in HISTORIC_RETURNS_US_CHF, HISTORIC_RETURNS_NON_US_CHF, and HISTORIC_SWISS_INFLATION
    total_years = len(HISTORIC_RETURNS_US_CHF)
    for r in range(num_runs):
        for block_start in (0, 5):
            us_block = sampled_us_annual[r, block_start : block_start + 5]
            non_us_block = sampled_non_us_annual[r, block_start : block_start + 5]
            inf_block = inf_matrix[r, block_start : block_start + 5]

            matched = False
            for h_idx in range(total_years - 5 + 1):
                if (
                    np.allclose(us_block, HISTORIC_RETURNS_US_CHF[h_idx : h_idx + 5], atol=1e-6)
                    and np.allclose(non_us_block, HISTORIC_RETURNS_NON_US_CHF[h_idx : h_idx + 5], atol=1e-6)
                    and np.allclose(inf_block, HISTORIC_SWISS_INFLATION[h_idx : h_idx + 5], atol=1e-6)
                ):
                    matched = True
                    break
            assert matched, f"Run {r} block starting at year {block_start} is not a contiguous 5-year historical slice"

        # Also verify the trailing 2-year partial block (years 10..11) is a contiguous 2-year slice
        us_tail = sampled_us_annual[r, 10:12]
        inf_tail = inf_matrix[r, 10:12]
        tail_matched = any(
            np.allclose(us_tail, HISTORIC_RETURNS_US_CHF[h_idx : h_idx + 2], atol=1e-6)
            and np.allclose(inf_tail, HISTORIC_SWISS_INFLATION[h_idx : h_idx + 2], atol=1e-6)
            for h_idx in range(total_years - 5 + 1)
        )
        assert tail_matched

    # Wrapper consistency with explicit block_size_years
    assert np.array_equal(
        ret_matrix,
        generate_bootstrapped_returns(num_runs, duration_years, seed=42, block_size_years=5, stationary=False),
    )
    assert np.array_equal(
        inf_matrix,
        generate_bootstrapped_inflation(num_runs, duration_years, seed=42, block_size_years=5, stationary=False),
    )

    # Validation for invalid block_size_years
    with pytest.raises(ValueError, match="block_size_years"):
        generate_bootstrapped_data(10, 10, seed=42, block_size_years=0)
    with pytest.raises(ValueError, match="block_size_years"):
        generate_bootstrapped_data(10, 10, seed=42, block_size_years=total_years + 1)


def _indexation_config(bracket_indexation, inflation_mean=0.04, start_age=45, duration_years=10, **overrides):
    """Builds a deterministic single-run config used by the bracket indexation tests."""
    params = dict(
        num_runs=1,
        duration_years=duration_years,
        inflation_mean=inflation_mean,
        inflation_std=0.0,
        start_age=start_age,
        dividend_yield=0.02,
        enable_smart_selling=False,
        initial_liquid_wealth=3_000_000.0,
        initial_pillar_2=0.0,
        initial_pillar_3a_accounts=[],
        alloc_us_stocks=0.0,
        alloc_non_us_stocks=0.6,
        alloc_chf_cash=0.4,
        alloc_gold=0.0,
        alloc_bitcoin=0.0,
        rebalance_strategy='Yearly',
        annual_base_expenses=60_000.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19,
        bracket_indexation=bracket_indexation,
    )
    params.update(overrides)
    return SimConfig(**params)


def test_bracket_indexation_matches_homogeneity_identity():
    # Full indexation must be exactly equivalent to deflating the tax base, applying the
    # unscaled tariff, and re-inflating: T_indexed(x, f) == f * T_nominal(x / f).
    # Verified against a tariff reconstructed by hand from the raw tax_engine functions.
    from src.tax_engine import calculate_income_tax, calculate_wealth_tax, calculate_ahv_non_worker

    duration_years = 8
    inflation = 0.04
    config = _indexation_config(1.0, inflation_mean=inflation, duration_years=duration_years)

    return_matrix = np.zeros((1, duration_years * 12, 5))
    inflation_matrix = np.full((1, duration_years), inflation)
    history = run_simulation(config, return_matrix, inflation_matrix)

    # Re-derive the final year's tax bill by hand from the recorded state.
    year = duration_years - 1
    f = (1.0 + inflation) ** duration_years  # bracket factor after N annual updates

    assets = history['liquid_assets_by_class'][year, 0, :]
    expenses = history['expenses_paid'][year, 0]
    taxes = history['taxes_paid'][year, 0]

    # Undo the selling that happened after taxes were assessed to recover the tax base.
    total_liquid_end = np.sum(assets) + expenses + taxes
    equities = assets[1] + taxes * (assets[1] / np.sum(assets)) + expenses * (assets[1] / np.sum(assets))

    # Since return_matrix has 0% cash return, realized cash interest is 0.0 (no phantom 1% tax drag).
    taxable_income = equities * config.dividend_yield
    taxable_wealth = max(0.0, total_liquid_end - expenses)

    expected = (
        f * calculate_income_tax(taxable_income / f, config.cantonal_multiplier, config.municipal_multiplier)
        + f * calculate_wealth_tax(taxable_wealth / f, config.cantonal_multiplier, config.municipal_multiplier)
        + f * calculate_ahv_non_worker(taxable_wealth / f)
    )

    assert np.isclose(taxes, expected, rtol=1e-6)

    # And the identity must differ from the frozen-bracket tariff under real inflation.
    unindexed = (
        calculate_income_tax(taxable_income, config.cantonal_multiplier, config.municipal_multiplier)
        + calculate_wealth_tax(taxable_wealth, config.cantonal_multiplier, config.municipal_multiplier)
        + calculate_ahv_non_worker(taxable_wealth)
    )
    assert unindexed > expected


def test_bracket_indexation_is_noop_without_inflation():
    # With zero inflation the bracket factor stays at 1.0, so the setting must not matter.
    duration_years = 6
    return_matrix = np.zeros((1, duration_years * 12, 5))
    inflation_matrix = np.zeros((1, duration_years))

    frozen = run_simulation(
        _indexation_config(0.0, inflation_mean=0.0, duration_years=duration_years),
        return_matrix, inflation_matrix
    )
    indexed = run_simulation(
        _indexation_config(1.0, inflation_mean=0.0, duration_years=duration_years),
        return_matrix, inflation_matrix
    )

    assert np.allclose(frozen['taxes_paid'], indexed['taxes_paid'])
    assert np.allclose(frozen['net_worth'], indexed['net_worth'])


def test_bracket_indexation_monotonic_between_extremes():
    # Under positive inflation, more indexation must mean strictly less tax, with partial
    # indexation landing between the two extremes.
    duration_years = 12
    inflation = 0.04
    return_matrix = np.zeros((1, duration_years * 12, 5))
    inflation_matrix = np.full((1, duration_years), inflation)

    results = {}
    for p in (0.0, 0.5, 1.0):
        h = run_simulation(
            _indexation_config(p, inflation_mean=inflation, duration_years=duration_years),
            return_matrix, inflation_matrix
        )
        results[p] = np.sum(h['taxes_paid'][:, 0])

    assert results[0.0] > results[0.5] > results[1.0]

    # Year 0 is barely affected (one year of indexation) while the gap compounds over time.
    frozen = run_simulation(
        _indexation_config(0.0, inflation_mean=inflation, duration_years=duration_years),
        return_matrix, inflation_matrix
    )
    full = run_simulation(
        _indexation_config(1.0, inflation_mean=inflation, duration_years=duration_years),
        return_matrix, inflation_matrix
    )
    gap_year_0 = frozen['taxes_paid'][0, 0] / full['taxes_paid'][0, 0]
    gap_final = frozen['taxes_paid'][-1, 0] / full['taxes_paid'][-1, 0]
    assert gap_final > gap_year_0 > 1.0


def test_bracket_indexation_scales_ahv_step_function():
    # The AHV non-worker table is a step function, not a piecewise-linear tariff. It still
    # indexes exactly, because floor division is scale invariant: (f*a) // (f*b) == a // b.
    # A retiree pinned at the 530 CHF minimum must therefore pay 530 * f once indexed.
    duration_years = 5
    inflation = 0.10
    return_matrix = np.zeros((1, duration_years * 12, 5))
    inflation_matrix = np.full((1, duration_years), inflation)

    # Tiny wealth keeps the contribution pinned to the minimum at every bracket factor.
    common = dict(
        inflation_mean=inflation,
        duration_years=duration_years,
        start_age=50,
        initial_liquid_wealth=1.0,
        annual_base_expenses=0.0,
        dividend_yield=0.0,
        cantonal_multiplier=0.0,
        municipal_multiplier=0.0,
    )

    frozen = run_simulation(_indexation_config(0.0, **common), return_matrix, inflation_matrix)
    indexed = run_simulation(_indexation_config(1.0, **common), return_matrix, inflation_matrix)

    for year in range(duration_years):
        f = (1.0 + inflation) ** (year + 1)
        assert np.isclose(frozen['taxes_paid'][year, 0], 530.0)
        assert np.isclose(indexed['taxes_paid'][year, 0], 530.0 * f)


def test_bracket_indexation_validation_and_default():
    import pytest

    # Defaults to full indexation, matching Art. 39 DBG / § 48 StG ZH.
    config = SimConfig(
        num_runs=1, duration_years=5, inflation_mean=0.02, inflation_std=0.0,
        start_age=40, dividend_yield=0.0, alloc_us_stocks=1.0
    )
    assert config.bracket_indexation == 1.0

    with pytest.raises(ValueError, match="bracket_indexation"):
        SimConfig(
            num_runs=1, duration_years=5, inflation_mean=0.02, inflation_std=0.0,
            start_age=40, dividend_yield=0.0, alloc_us_stocks=1.0,
            bracket_indexation=-0.1
        )


def test_real_chf_appreciation_ppp_adjustment_and_validation():
    import pytest
    from src.historic_returns import (
        HISTORIC_REAL_CHF_APPRECIATION,
        generate_bootstrapped_data,
        get_historic_return_matrix,
    )

    # Default SimConfig has real_chf_appreciation = 0.0 (PPP neutrality)
    config = SimConfig(
        num_runs=1, duration_years=5, inflation_mean=0.02, inflation_std=0.0,
        start_age=40, dividend_yield=0.0, alloc_us_stocks=1.0
    )
    assert config.real_chf_appreciation == 0.0

    with pytest.raises(ValueError, match="real_chf_appreciation"):
        SimConfig(
            num_runs=1, duration_years=5, inflation_mean=0.02, inflation_std=0.0,
            start_age=40, dividend_yield=0.0, alloc_us_stocks=1.0,
            real_chf_appreciation=1.0
        )

    # Raw history (None) matches HISTORIC_REAL_CHF_APPRECIATION (+0.68%/yr) identically
    raw_mat = get_historic_return_matrix(10, seed=42, real_chf_appreciation=None)
    hist_mat = get_historic_return_matrix(
        10, seed=42, real_chf_appreciation=HISTORIC_REAL_CHF_APPRECIATION
    )
    assert np.allclose(raw_mat, hist_mat, atol=1e-12)

    # PPP-neutral (0.0) scales foreign-priced sleeves (cols 0, 1, 3) by
    # ann_adj = 1.0 / (1.0 - HISTORIC_REAL_CHF_APPRECIATION) over 12 months,
    # while leaving CHF cash (col 2) and synthetic BTC (col 4) untouched.
    ppp_mat = get_historic_return_matrix(10, seed=42, real_chf_appreciation=0.0)
    expected_ann_adj = 1.0 / (1.0 - HISTORIC_REAL_CHF_APPRECIATION)
    for col in (0, 1, 3):
        raw_yr0 = np.prod(1.0 + raw_mat[0, :12, col])
        ppp_yr0 = np.prod(1.0 + ppp_mat[0, :12, col])
        assert np.isclose(ppp_yr0 / raw_yr0, expected_ann_adj, atol=1e-10)
    for col in (2, 4):
        assert np.allclose(raw_mat[:, :, col], ppp_mat[:, :, col], atol=1e-12)

    # Same property holds for generate_bootstrapped_data
    boot_raw, _ = generate_bootstrapped_data(5, 10, seed=7, real_chf_appreciation=None)
    boot_ppp, _ = generate_bootstrapped_data(5, 10, seed=7, real_chf_appreciation=0.0)
    for col in (0, 1, 3):
        raw_yr0 = np.prod(1.0 + boot_raw[0, :12, col])
        ppp_yr0 = np.prod(1.0 + boot_ppp[0, :12, col])
        assert np.isclose(ppp_yr0 / raw_yr0, expected_ann_adj, atol=1e-10)


def test_stationary_bootstrap_politis_romano_and_effective_sample_size():
    from src.historic_returns import (
        HISTORIC_RETURNS_US_CHF,
        POLITIS_WHITE_BLOCK_YEARS,
        compute_effective_sample_size,
        generate_bootstrapped_data,
    )

    assert POLITIS_WHITE_BLOCK_YEARS == 5

    # Verify effective sample size and 90% Wilson confidence interval for 40y horizon over 104y dataset
    stats = compute_effective_sample_size(total_years=104, duration_years=40, success_rate_pct=90.0)
    assert stats["n_cohorts"] == 65
    assert np.isclose(stats["n_eff"], 2.6)
    assert 0.0 <= stats["ci_low_pct"] < 90.0 < stats["ci_high_pct"] <= 100.0
    # At N_eff = 2.6, a 90% sample rate has a wide 90% Wilson CI (~41% to ~99%)
    assert stats["ci_high_pct"] - stats["ci_low_pct"] > 40.0

    # Verify Politis-Romano stationary bootstrap (default stationary=True) supports circular wrap-around
    # by checking that year 2025 (last historical index) is followed by year 1922 (index 0) whenever a block continues
    total_years = len(HISTORIC_RETURNS_US_CHF)
    ret_mat, inf_mat = generate_bootstrapped_data(
        num_runs=200, duration_years=20, seed=42, block_size_years=5, stationary=True
    )
    assert ret_mat.shape == (200, 240, 5)
    assert inf_mat.shape == (200, 20)

    # Reconstruct annual US returns to identify sampled historical indices
    us_annual = np.prod(1.0 + ret_mat[:, :, 0].reshape(200, 20, 12), axis=2) - 1.0

    wrap_around_observed = False
    for r in range(200):
        for t in range(19):
            idx_t = int(np.argmin(np.abs(HISTORIC_RETURNS_US_CHF - us_annual[r, t])))
            idx_next = int(np.argmin(np.abs(HISTORIC_RETURNS_US_CHF - us_annual[r, t + 1])))
            if idx_t == total_years - 1 and idx_next == 0:
                wrap_around_observed = True
                break
        if wrap_around_observed:
            break
    assert wrap_around_observed, "Circular wrap-around from year T-1 to year 0 must occur under stationary bootstrap"


def test_bitcoin_equity_correlation_and_cash_tax_interest():
    from src.simulation_engine import generate_monte_carlo_returns

    # 1. Verify positive vs negative Bitcoin-equity correlation across all 3 engines
    hist_pos = get_historic_return_matrix(30, seed=42, btc_equity_corr=0.80)
    hist_neg = get_historic_return_matrix(30, seed=42, btc_equity_corr=-0.80)
    corr_hist_pos = np.corrcoef(np.log1p(hist_pos[0, :, 0]), np.log1p(hist_pos[0, :, 4]))[0, 1]
    corr_hist_neg = np.corrcoef(np.log1p(hist_neg[0, :, 0]), np.log1p(hist_neg[0, :, 4]))[0, 1]
    assert corr_hist_pos > 0.65
    assert corr_hist_neg < -0.65

    mc_pos = generate_monte_carlo_returns(
        100, 30, 0.08, 0.06, 0.01, 0.02, 0.07, 0.16, 0.15, 0.50, seed=42, btc_equity_corr=0.80
    )
    mc_neg = generate_monte_carlo_returns(
        100, 30, 0.08, 0.06, 0.01, 0.02, 0.07, 0.16, 0.15, 0.50, seed=42, btc_equity_corr=-0.80
    )
    corr_mc_pos = np.corrcoef(np.log1p(mc_pos[:, :, 0].ravel()), np.log1p(mc_pos[:, :, 4].ravel()))[0, 1]
    corr_mc_neg = np.corrcoef(np.log1p(mc_neg[:, :, 0].ravel()), np.log1p(mc_neg[:, :, 4].ravel()))[0, 1]
    assert corr_mc_pos > 0.75
    assert corr_mc_neg < -0.75

    # 2. Verify taxable cash interest scales with actual realized cash return (0% return -> 0 CHF interest tax)
    cfg = SimConfig(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=65,
        dividend_yield=0.0,
        alloc_chf_cash=1.0,
        initial_liquid_wealth=1_000_000.0,
        annual_base_expenses=0.0,
        monthly_ahv_pension=0.0,
        cantonal_multiplier=0.95,
        municipal_multiplier=1.19,
    )
    ret_zero_cash = np.zeros((1, 12, 5))
    ret_high_cash = np.zeros((1, 12, 5))
    ret_high_cash[:, :, 2] = (1.03) ** (1.0 / 12.0) - 1.0  # 3% cash return

    h_zero = run_simulation(cfg, ret_zero_cash)
    h_high = run_simulation(cfg, ret_high_cash)
    # Under 3% cash return, income tax on 30,000 CHF interest + wealth tax is strictly higher than under 0% cash return
    assert h_high["taxes_paid"][0, 0] > h_zero["taxes_paid"][0, 0] + 1_000.0


def test_bitcoin_defaults_and_custom_params_across_engines():
    import pytest
    from src.historic_returns import (
        BITCOIN_NOMINAL_MEAN,
        BITCOIN_VOL,
        generate_bootstrapped_data,
        get_historic_return_matrix,
    )

    assert np.isclose(BITCOIN_NOMINAL_MEAN, 0.07)
    assert np.isclose(BITCOIN_VOL, 0.50)

    # Zero-volatility custom BTC return produces exact compound monthly return (1 + btc_mean)**(1/12) - 1
    custom_mean = 0.12
    expected_monthly = (1.0 + custom_mean) ** (1.0 / 12.0) - 1.0
    hist_zero_vol = get_historic_return_matrix(10, seed=42, btc_mean=custom_mean, btc_vol=0.0)
    assert np.allclose(hist_zero_vol[:, :, 4], expected_monthly, atol=1e-12)

    boot_zero_vol, _ = generate_bootstrapped_data(5, 10, seed=42, btc_mean=custom_mean, btc_vol=0.0)
    assert np.allclose(boot_zero_vol[:, :, 4], expected_monthly, atol=1e-12)

    # Invalid btc_mean or btc_vol raises ValueError
    with pytest.raises(ValueError, match="must be greater than -1.0"):
        get_historic_return_matrix(10, seed=42, btc_mean=-1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        get_historic_return_matrix(10, seed=42, btc_vol=-0.1)


def test_gaussian_copula_path_standardization_no_drift_leak():
    from src.historic_returns import _generate_lognormal_monthly_returns

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    # Create two synthetic US equity paths with identical standardized monthly shocks
    # but wildly different secular drifts (-5%/mo vs +5%/mo).
    base_shocks = np.sin(np.linspace(0.1, 20.0, 240)) * 0.04
    us_crash_cohort = np.expm1(-0.05 + base_shocks)[np.newaxis, :]
    us_boom_cohort = np.expm1(0.05 + base_shocks)[np.newaxis, :]

    btc_crash = _generate_lognormal_monthly_returns(
        rng1, 0.07, 0.50, (1, 240), us_monthly_slice=us_crash_cohort, btc_equity_corr=0.80
    )
    btc_boom = _generate_lognormal_monthly_returns(
        rng2, 0.07, 0.50, (1, 240), us_monthly_slice=us_boom_cohort, btc_equity_corr=0.80
    )
    # Because per-path standardization centers each cohort's log-returns to mean 0 and std 1,
    # the US cohort's secular drift does not leak into Bitcoin's expected return.
    assert np.allclose(btc_crash, btc_boom, atol=1e-12)


def test_run_simulation_matrix_shape_validation():
    import pytest

    cfg = SimConfig(
        num_runs=2,
        duration_years=5,
        inflation_mean=0.02,
        inflation_std=0.01,
        start_age=50,
        dividend_yield=0.015,
        alloc_us_stocks=1.0,
    )
    with pytest.raises(ValueError, match="return_matrix shape"):
        run_simulation(cfg, np.zeros((2, 48, 5)))
    with pytest.raises(ValueError, match="inflation_matrix shape"):
        run_simulation(cfg, np.zeros((2, 60, 5)), np.zeros((2, 4)))


def test_sim_config_defaults_to_zurich_city_multipliers():
    # Previously both defaulted to 0.0, so programmatic users (incl. the package
    # docstring example) silently got zero cantonal/municipal income & wealth tax.
    cfg = SimConfig(num_runs=1, duration_years=1, inflation_mean=0.0, inflation_std=0.0,
                    start_age=50, dividend_yield=0.0, alloc_us_stocks=1.0)
    assert (cfg.cantonal_multiplier, cfg.municipal_multiplier) == (0.95, 1.19)
    assert cfg.initial_pillar_3a_accounts == []
    # default_factory: instances must not share one mutable list
    other = SimConfig(num_runs=1, duration_years=1, inflation_mean=0.0, inflation_std=0.0,
                      start_age=50, dividend_yield=0.0, alloc_us_stocks=1.0)
    assert cfg.initial_pillar_3a_accounts is not other.initial_pillar_3a_accounts


def test_sim_config_rejects_out_of_range_rates():
    import pytest
    base = dict(num_runs=1, duration_years=1, inflation_mean=0.0, inflation_std=0.0,
                start_age=50, dividend_yield=0.0, alloc_us_stocks=1.0)
    with pytest.raises(ValueError, match="vanguard_floor_pct"):
        SimConfig(**base | {"vanguard_floor_pct": 1.01})
    with pytest.raises(ValueError, match="inflation_mean"):
        SimConfig(**base | {"inflation_mean": -1.0})
    with pytest.raises(ValueError, match="cash_rate"):
        SimConfig(**base | {"cash_rate": -1.5})
    # Boundary values are accepted
    SimConfig(**base | {"vanguard_floor_pct": 1.0, "inflation_mean": -0.5, "cash_rate": -0.0075})


def test_historic_real_chf_appreciation_is_derived_from_data():
    # The PPP residual must match what the CSVs imply. It was hard-coded at 0.0068
    # (arithmetic -1.78% - -1.09% on stale FX data); the geometric value from the
    # rebuilt data is ~0.70%/yr.
    import os
    import pandas as pd
    from src.historic_returns import HISTORIC_REAL_CHF_APPRECIATION, HISTORIC_YEARS

    data = os.path.join(os.path.dirname(__file__), "..", "data")

    def dec(name):
        df = pd.read_csv(os.path.join(data, name), header=None, names=["m", "y", "v"])
        return df[df.m == 12].set_index("y").v

    fx, ch, us = dec("usd_chf.csv"), dec("ch_inflation.csv"), dec("us_cpi.csv")
    y0, y1 = int(HISTORIC_YEARS[0]) - 1, int(HISTORIC_YEARS[-1])
    n = y1 - y0
    fx_drift = (fx[y1] / fx[y0]) ** (1 / n)
    ppp_drift = ((ch[y1] / ch[y0]) / (us[y1] / us[y0])) ** (1 / n)
    expected = 1.0 - fx_drift / ppp_drift
    assert abs(HISTORIC_REAL_CHF_APPRECIATION - expected) < 5e-5
    assert 0.006 < HISTORIC_REAL_CHF_APPRECIATION < 0.008


def test_target_weights_computed_once_per_year(monkeypatch):
    # get_target_weights re-runs the Year 0 tax estimate; run_simulation must not
    # call it every month (it used to: 12x per year).
    import src.simulation_engine as se
    calls = []
    real = se.get_target_weights
    monkeypatch.setattr(se, "get_target_weights", lambda c, y: calls.append(y) or real(c, y))
    cfg = SimConfig(num_runs=1, duration_years=10, inflation_mean=0.0, inflation_std=0.0,
                    start_age=50, dividend_yield=0.0, initial_liquid_wealth=1e6,
                    alloc_us_stocks=0.8, alloc_chf_cash=0.2, rebalance_strategy="Cash Tent",
                    annual_base_expenses=40_000.0)
    se.run_simulation(cfg, np.zeros((1, 120, 5)))
    assert calls == list(range(10))

