import numpy as np
from dataclasses import dataclass
from tax_engine import calculate_income_tax, calculate_wealth_tax, calculate_capital_withdrawal_tax, calculate_ahv_non_worker

@dataclass
class SimConfig:
    num_runs: int
    duration_years: int
    inflation_mean: float
    inflation_std: float
    
    start_age: int
    dividend_yield: float
    
    # Dynamic Expenses
    enable_dynamic_expenses: bool
    dynamic_expense_floor_pct: float
    
    # Initial Assets
    initial_liquid_wealth: float
    initial_pillar_2: float
    initial_pillar_3a_accounts: list[float]
    
    # Target Asset Allocation (must sum to 1.0)
    # Indices: 0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin
    alloc_us_stocks: float
    alloc_non_us_stocks: float
    alloc_chf_cash: float
    alloc_gold: float
    alloc_bitcoin: float
    
    # Rebalance strategy: 'never', 'yearly', 'monthly', 'threshold'
    rebalance_strategy: str
    rebalance_threshold: float 
    enable_smart_selling: bool
    
    # Expenses & Income
    annual_base_expenses: float
    monthly_ahv_pension: float
    
    # Taxes
    cantonal_multiplier: float
    municipal_multiplier: float

    def __post_init__(self):
        total_alloc = (self.alloc_us_stocks + self.alloc_non_us_stocks + 
                       self.alloc_chf_cash + self.alloc_gold + self.alloc_bitcoin)
        if not np.isclose(total_alloc, 1.0, atol=0.0001):
            raise ValueError(f"Asset allocations must sum to exactly 1.0. Currently: {total_alloc:.4f}")


def run_simulation(config: SimConfig, return_matrix: np.ndarray, inflation_matrix: np.ndarray = None) -> dict:
    """
    Executes the multi-asset monthly simulation loop.
    return_matrix shape: (num_runs, duration_months, 5)
    """
    num_runs = config.num_runs
    duration_months = config.duration_years * 12
    
    target_weights = np.array([
        config.alloc_us_stocks,
        config.alloc_non_us_stocks,
        config.alloc_chf_cash,
        config.alloc_gold,
        config.alloc_bitcoin
    ])
    
    # Assets (0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin)
    liquid_assets = np.zeros((num_runs, 5))
    liquid_assets[:] = config.initial_liquid_wealth * target_weights
    
    pillar_2 = np.full(num_runs, config.initial_pillar_2, dtype=float)
    pillar_3a = [np.full(num_runs, acc, dtype=float) for acc in config.initial_pillar_3a_accounts]
    
    initial_net_worth = config.initial_liquid_wealth + config.initial_pillar_2 + sum(config.initial_pillar_3a_accounts)
    
    history_net_worth = np.zeros((duration_months // 12, num_runs))
    history_liquid = np.zeros((duration_months // 12, num_runs))
    history_liquid_by_class = np.zeros((duration_months // 12, num_runs, 5))
    history_taxes = np.zeros((duration_months // 12, num_runs))
    history_expenses = np.zeros((duration_months // 12, num_runs))
    history_income_divs = np.zeros((duration_months // 12, num_runs))
    history_income_ahv = np.zeros((duration_months // 12, num_runs))
    history_below_watermark = np.zeros((duration_months // 12, num_runs), dtype=bool)
    
    inflation_factors = np.ones(num_runs)
    taxable_liquidation_amount = np.zeros(num_runs)
    capital_withdrawal_tax_this_year = np.zeros(num_runs)
    
    for m in range(duration_months):
        year = m // 12
        month_of_year = m % 12
        current_age = config.start_age + year
        
        # 1. Inflation (Update once a year at month 0)
        if month_of_year == 0:
            if inflation_matrix is not None:
                inflation = inflation_matrix[:, year]
            else:
                inflation = np.random.normal(config.inflation_mean, config.inflation_std, num_runs)
            inflation_factors *= (1 + inflation)
            taxable_liquidation_amount.fill(0)
            capital_withdrawal_tax_this_year.fill(0)
            
        # 2. Liquidations (Yearly at month 0)
        if month_of_year == 0:
            if current_age < 65:
                has_liquidated = np.zeros(num_runs, dtype=bool)
                for i, p3a in enumerate(pillar_3a):
                    liquidation_age = 60 + i
                    mask = (current_age >= liquidation_age) & (p3a > 0) & (~has_liquidated)
                    amount = p3a[mask]
                    taxable_liquidation_amount[mask] += amount
                    liquid_assets[mask] += np.outer(amount, target_weights)
                    p3a[mask] = 0
                    has_liquidated[mask] = True
            else:
                # Age >= 65: Liquidate ALL remaining accounts
                for p3a in pillar_3a:
                    mask = p3a > 0
                    amount = p3a[mask]
                    taxable_liquidation_amount[mask] += amount
                    liquid_assets[mask] += np.outer(amount, target_weights)
                    p3a[mask] = 0
                
            mask_p2 = (current_age >= 65) & (pillar_2 > 0)
            amount_p2 = pillar_2[mask_p2]
            taxable_liquidation_amount[mask_p2] += amount_p2
            liquid_assets[mask_p2] += np.outer(amount_p2, target_weights)
            pillar_2[mask_p2] = 0
            
            # Calculate Capital Withdrawal Tax immediately upon liquidation
            has_liquidation = taxable_liquidation_amount > 0
            if np.any(has_liquidation):
                cap_tax = np.zeros(num_runs)
                cap_tax[has_liquidation] = calculate_capital_withdrawal_tax(
                    taxable_liquidation_amount[has_liquidation],
                    config.cantonal_multiplier,
                    config.municipal_multiplier
                )
                
                # Deduct immediately at source (proportionally as it was added)
                liquid_assets -= np.outer(cap_tax, target_weights)
                
                # Track it for the annual history log
                capital_withdrawal_tax_this_year += cap_tax
        
        # 3. Market Returns
        monthly_returns = return_matrix[:, m, :] # (num_runs, 5)
        
        # Apply 100% Equity returns to Pillar 2 and 3a (proportional to US/Non-US target alloc)
        total_stocks = config.alloc_us_stocks + config.alloc_non_us_stocks
        if total_stocks > 0:
            weight_us = config.alloc_us_stocks / total_stocks
            weight_non_us = config.alloc_non_us_stocks / total_stocks
        else:
            weight_us, weight_non_us = 0.5, 0.5
            
        equity_monthly_returns = monthly_returns[:, 0] * weight_us + monthly_returns[:, 1] * weight_non_us
        
        pillar_2 *= (1 + equity_monthly_returns)
        for p3a in pillar_3a:
            p3a *= (1 + equity_monthly_returns)
        
        # Handle Bankruptcy (Total liquid < 0)
        total_liquid = np.sum(liquid_assets, axis=1, keepdims=True)
        is_bankrupt = (total_liquid.flatten() < 0)
        
        if np.any(is_bankrupt):
            # Move all debt to CHF Cash (index 2), zero out other assets
            liquid_assets[is_bankrupt, :] = 0
            liquid_assets[is_bankrupt, 2] = total_liquid[is_bankrupt].flatten()
            
        # Positive balances get market returns. Negative balances get a 5% APY penalty (converted to monthly)
        penalty_monthly = (1.05)**(1/12) - 1
        returns_to_apply = np.where(liquid_assets > 0, monthly_returns, penalty_monthly)
        returns_to_apply = np.maximum(-1.0, returns_to_apply)
        liquid_assets *= (1 + returns_to_apply)
        
        # 4. Rebalancing Check
        do_rebalance = np.zeros(num_runs, dtype=bool)
        
        # 4.5. Monthly AHV Pension Addition
        if current_age >= 65:
            monthly_ahv = config.monthly_ahv_pension * inflation_factors
            liquid_assets[:, 2] += monthly_ahv # Add directly to CHF Cash

        
        if config.rebalance_strategy == 'Monthly':
            do_rebalance[:] = True
        elif config.rebalance_strategy == 'Quarterly' and month_of_year % 3 == 2:
            do_rebalance[:] = True
        elif config.rebalance_strategy == 'Yearly' and month_of_year == 11:
            do_rebalance[:] = True
        elif config.rebalance_strategy == 'Threshold':
            current_total = np.sum(liquid_assets, axis=1, keepdims=True)
            current_total_safe = np.where(current_total > 0, current_total, 1)
            current_weights = liquid_assets / current_total_safe
            drift = np.max(np.abs(current_weights - target_weights), axis=1)
            do_rebalance = (drift > config.rebalance_threshold) & (current_total.flatten() > 0)
            
        if config.enable_smart_selling:
            current_p3a = sum(p3a for p3a in pillar_3a) if len(pillar_3a) > 0 else 0
            current_nw = np.sum(liquid_assets, axis=1) + pillar_2 + current_p3a
            inflation_adjusted_initial_nw = initial_net_worth * inflation_factors
            is_downturn = current_nw < inflation_adjusted_initial_nw
            do_rebalance[is_downturn] = False
            
        if np.any(do_rebalance):
            total_to_rebalance = np.sum(liquid_assets[do_rebalance], axis=1, keepdims=True)
            liquid_assets[do_rebalance] = total_to_rebalance * target_weights
            
        # 5. Annual Taxation and Expenses
        if month_of_year == 11:
            current_expenses = config.annual_base_expenses * inflation_factors
            
            # Dynamic Expense Adjustment
            total_p3a_current = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            current_nw_before_expenses = np.sum(liquid_assets, axis=1) + pillar_2 + total_p3a_current
            
            inflation_adjusted_initial_nw = initial_net_worth * inflation_factors
            is_below_watermark = current_nw_before_expenses < inflation_adjusted_initial_nw
            history_below_watermark[year, :] = is_below_watermark
            
            if config.enable_dynamic_expenses:
                current_expenses = np.where(is_below_watermark, 
                                            current_expenses * config.dynamic_expense_floor_pct, 
                                            current_expenses)
                                            
            annual_ahv_received = np.where(current_age >= 65, 12 * config.monthly_ahv_pension * inflation_factors, 0.0)
            
            # Taxable Income = Dividends (yield on Equity portion)
            # US Stocks (0) and Non-US Stocks (1) generate dividends
            equities = np.maximum(0, liquid_assets[:, 0] + liquid_assets[:, 1])
            dividends = equities * config.dividend_yield
            
            # Cash (index 2) generates taxable interest (assumed 1% APY as per return matrix)
            cash = np.maximum(0, liquid_assets[:, 2])
            interest = cash * 0.01
            
            taxable_income = dividends + interest + annual_ahv_received
            
            income_tax = calculate_income_tax(taxable_income, config.cantonal_multiplier, config.municipal_multiplier)
            
            total_liquid_end = np.sum(liquid_assets, axis=1)
            # Wealth Tax and AHV non-worker tax are assessed on wealth *after* deducting living expenses
            taxable_wealth = np.maximum(0, total_liquid_end - current_expenses)
            
            wealth_tax = calculate_wealth_tax(taxable_wealth, config.cantonal_multiplier, config.municipal_multiplier)
            
            ahv_mask = current_age < 65
            ahv_contrib = np.zeros(num_runs)
            ahv_contrib[ahv_mask] = calculate_ahv_non_worker(taxable_wealth[ahv_mask])
            
            # Capital withdrawal tax was already deducted at source in month 0
            total_taxes = income_tax + wealth_tax + ahv_contrib
            net_cash_flow = current_expenses + total_taxes # AHV already added to cash monthly
            
            deficit = np.maximum(0, net_cash_flow)
            
            # Amount we can actually pay from current positive assets
            payable = np.minimum(deficit, np.maximum(0, total_liquid_end))
            
            # Any remaining deficit goes directly to negative Cash
            remaining_deficit = deficit - payable
            
            if config.enable_smart_selling:
                inflation_adjusted_initial_nw = initial_net_worth * inflation_factors
                is_downturn = current_nw_before_expenses < inflation_adjusted_initial_nw
                
                # Pay from cash first for downturn runs
                downturn_payable = payable[is_downturn]
                available_cash = np.maximum(0, liquid_assets[is_downturn, 2])
                sell_cash = np.minimum(downturn_payable, available_cash)
                
                liquid_assets[is_downturn, 2] -= sell_cash
                payable[is_downturn] -= sell_cash
            
            # Sell proportionally from remaining positive assets
            positive_assets = np.maximum(0, liquid_assets)
            total_positive = np.sum(positive_assets, axis=1)
            safe_total_pos = np.where(total_positive > 0, total_positive, 1.0)
            
            fractions = payable / safe_total_pos
            sell_amounts = positive_assets * fractions[:, None]
            liquid_assets -= sell_amounts
            
            liquid_assets[:, 2] -= remaining_deficit
            
            # Record
            total_p3a = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            net_worth = np.sum(liquid_assets, axis=1) + pillar_2 + total_p3a
            
            history_net_worth[year, :] = net_worth
            history_liquid[year, :] = np.sum(liquid_assets, axis=1)
            history_liquid_by_class[year, :, :] = liquid_assets
            history_taxes[year, :] = total_taxes + capital_withdrawal_tax_this_year
            history_expenses[year, :] = current_expenses
            history_income_divs[year, :] = dividends
            history_income_ahv[year, :] = annual_ahv_received

    return {
        'net_worth': history_net_worth,
        'liquid_assets': history_liquid,
        'liquid_assets_by_class': history_liquid_by_class,
        'taxes_paid': history_taxes,
        'expenses_paid': history_expenses,
        'income_dividends': history_income_divs,
        'income_ahv': history_income_ahv,
        'below_watermark': history_below_watermark,
        'initial_net_worth': initial_net_worth
    }


def generate_monte_carlo_returns(
    num_runs: int,
    duration_years: int,
    ret_us: float,
    ret_non_us: float,
    ret_cash: float,
    ret_gold: float,
    ret_btc: float,
    vol_eq: float,
    vol_gold: float,
    vol_btc: float,
    seed: int = 42
) -> np.ndarray:
    """Generates monthly returns using a lognormal model to prevent negative asset prices."""
    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    
    def get_monthly_returns(ann_ret, ann_vol):
        # Ito drift correction: mu = ln(1+R) - 0.5 * sigma^2
        drift = np.log(1 + ann_ret) - 0.5 * ann_vol**2
        log_ret = rng.normal(drift/12, ann_vol/np.sqrt(12), (num_runs, duration_months))
        return np.exp(log_ret) - 1.0

    matrix[:, :, 0] = get_monthly_returns(ret_us, vol_eq)
    matrix[:, :, 1] = get_monthly_returns(ret_non_us, vol_eq)
    # Cash is constant nominal return (geometric monthly rate)
    matrix[:, :, 2] = (1 + ret_cash)**(1/12) - 1.0
    matrix[:, :, 3] = get_monthly_returns(ret_gold, vol_gold)
    matrix[:, :, 4] = get_monthly_returns(ret_btc, vol_btc)
    
    return matrix

