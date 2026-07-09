import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulation_engine import SimConfig, run_simulation

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
    from simulation_engine import generate_monte_carlo_returns
    
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
        municipal_multiplier=0.0
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
