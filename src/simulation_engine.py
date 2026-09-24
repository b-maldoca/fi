import numpy as np
from dataclasses import dataclass, field
from .tax_engine import (
    ZURICH_CANTONAL_MULTIPLIER,
    ZURICH_CITY_MUNICIPAL_MULTIPLIER,
    calculate_ahv_non_worker,
    calculate_capital_withdrawal_tax,
    calculate_income_tax,
    calculate_wealth_tax,
)

VALID_SPENDING_STRATEGIES = {"Static", "Dynamic (Floor & Ceiling)", "Vanguard Dynamic"}
VALID_REBALANCE_STRATEGIES = {"Cash Tent", "Monthly", "Quarterly", "Yearly", "Threshold", "Never"}


@dataclass
class SimConfig:
    num_runs: int
    duration_years: int
    inflation_mean: float
    inflation_std: float
    
    start_age: int
    dividend_yield: float
    
    # Spending Strategy: 'Static', 'Dynamic (Floor & Ceiling)', 'Vanguard Dynamic'
    spending_strategy: str = "Static"
    enable_dynamic_expenses: bool = False
    dynamic_expense_floor_pct: float = 1.0
    dynamic_expense_ceiling_pct: float = 1.0
    
    # Vanguard Dynamic Spending settings
    vanguard_target_rate: float = 0.035
    vanguard_floor_pct: float = 0.05
    vanguard_ceiling_pct: float = 0.05
    
    # Initial Assets
    initial_liquid_wealth: float = 0.0
    initial_pillar_2: float = 0.0
    initial_pillar_3a_accounts: list[float] = field(default_factory=list)
    
    # Target Asset Allocation (must sum to 1.0)
    # Indices: 0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin
    alloc_us_stocks: float = 0.0
    alloc_non_us_stocks: float = 0.0
    alloc_chf_cash: float = 0.0
    alloc_gold: float = 0.0
    alloc_bitcoin: float = 0.0
    
    # Rebalance strategy: 'Cash Tent', 'Monthly', 'Quarterly', 'Yearly', 'Threshold', 'Never'
    rebalance_strategy: str = "Yearly"
    rebalance_threshold: float = 0.0
    enable_smart_selling: bool = True
    
    # Expenses & Income
    annual_base_expenses: float = 0.0
    monthly_ahv_pension: float = 0.0
    
    # Taxes (Steuerfuss). Defaults are Canton Zurich 2026 (0.95) + City of Zurich (1.19).
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER

    # Fraction of realized CPI applied annually to the tax bracket edges.
    # 1.0 = full indexation (Art. 39 DBG / § 48 StG ZH); 0.0 = frozen nominal
    # brackets, i.e. the retiree absorbs the full bracket creep.
    bracket_indexation: float = 1.0

    # Cash Tent Strategy (Pfau Glidepath)
    tent_duration_years: int = 7

    # Expected real CHF appreciation beyond Relative Purchasing Power Parity (PPP).
    # 0.0 = PPP neutrality (default); historic_returns.HISTORIC_REAL_CHF_APPRECIATION
    # (derived from the data at build time, ~0.70%/yr) reproduces raw 1922-2025 history.
    real_chf_appreciation: float = 0.0

    # Expected CHF Cash yield used for Year 0 tax estimation (and Cash Tent buffer sizing).
    cash_rate: float = 0.01

    # Correlation between synthetic Bitcoin monthly log-returns and US equity log-returns.
    btc_equity_corr: float = 0.50

    # Random Seed for reproducible stochastic simulation
    seed: int = 42

    def __post_init__(self):
        if self.num_runs <= 0 or self.duration_years <= 0:
            raise ValueError(f"num_runs ({self.num_runs}) and duration_years ({self.duration_years}) must be positive integers.")
            
        if self.start_age < 0:
            raise ValueError(f"start_age ({self.start_age}) cannot be negative.")

        if self.tent_duration_years < 0:
            raise ValueError(f"tent_duration_years ({self.tent_duration_years}) cannot be negative.")

        if self.inflation_std < 0.0:
            raise ValueError(f"inflation_std ({self.inflation_std}) cannot be negative.")

        if self.inflation_mean <= -1.0 or self.cash_rate <= -1.0:
            raise ValueError(
                f"inflation_mean ({self.inflation_mean}) and cash_rate ({self.cash_rate}) must be greater than -1.0."
            )

        if self.seed < 0:
            raise ValueError(f"seed ({self.seed}) cannot be negative.")

        if self.real_chf_appreciation >= 1.0:
            raise ValueError(f"real_chf_appreciation ({self.real_chf_appreciation}) must be strictly less than 1.0.")

        if not (-1.0 <= self.btc_equity_corr <= 1.0):
            raise ValueError(f"btc_equity_corr ({self.btc_equity_corr}) must be between -1.0 and 1.0.")
            
        if self.vanguard_floor_pct < 0.0 or self.vanguard_ceiling_pct < 0.0 or self.vanguard_target_rate < 0.0:
            raise ValueError("Vanguard Dynamic Spending parameters cannot be negative.")

        if self.vanguard_floor_pct > 1.0:
            # A cut of more than 100% would allow negative spending floors.
            raise ValueError(f"vanguard_floor_pct ({self.vanguard_floor_pct}) cannot exceed 1.0 (a 100% cut).")

        if self.initial_pillar_3a_accounts is None:
            self.initial_pillar_3a_accounts = []
            
        if self.initial_liquid_wealth < 0.0 or self.initial_pillar_2 < 0.0 or any(a < 0.0 for a in self.initial_pillar_3a_accounts):
            raise ValueError("Initial asset balances cannot be negative.")

        if self.annual_base_expenses < 0.0 or self.monthly_ahv_pension < 0.0 or self.dividend_yield < 0.0:
            raise ValueError("Annual base expenses, AHV pension, and dividend yield cannot be negative.")

        if self.cantonal_multiplier < 0.0 or self.municipal_multiplier < 0.0:
            raise ValueError("Tax multipliers cannot be negative.")

        if self.bracket_indexation < 0.0:
            raise ValueError(f"bracket_indexation ({self.bracket_indexation}) cannot be negative.")

        if self.rebalance_threshold < 0.0 or self.dynamic_expense_floor_pct < 0.0 or self.dynamic_expense_ceiling_pct < 0.0:
            raise ValueError("Rebalance threshold and dynamic expense floor/ceiling percentages cannot be negative.")

        allocs = [self.alloc_us_stocks, self.alloc_non_us_stocks, self.alloc_chf_cash, self.alloc_gold, self.alloc_bitcoin]
        if any(a < 0.0 for a in allocs):
            raise ValueError(f"Asset allocations cannot be negative. Current: {allocs}")
            
        total_alloc = sum(allocs)
        if not np.isclose(total_alloc, 1.0, atol=0.0001):
            raise ValueError(f"Asset allocations must sum to exactly 1.0. Currently: {total_alloc:.4f}")
            
        # Backwards compatibility for enable_dynamic_expenses flag
        if self.enable_dynamic_expenses and self.spending_strategy == "Static":
            self.spending_strategy = "Dynamic (Floor & Ceiling)"

        if self.spending_strategy not in VALID_SPENDING_STRATEGIES:
            raise ValueError(f"Invalid spending_strategy '{self.spending_strategy}'. Must be one of {sorted(VALID_SPENDING_STRATEGIES)}.")

        if self.rebalance_strategy not in VALID_REBALANCE_STRATEGIES:
            raise ValueError(f"Invalid rebalance_strategy '{self.rebalance_strategy}'. Must be one of {sorted(VALID_REBALANCE_STRATEGIES)}.")


def _get_year_0_liquid_wealth(config: SimConfig) -> float:
    """
    Computes Year 0 initial liquid wealth including any immediate Month 0 pension liquidations
    (all Pillar 2 + Pillar 3a at age >= 65, or the first eligible Pillar 3a account at ages 60-64)
    net of immediate Capital Withdrawal Tax.
    """
    init_wealth = config.initial_liquid_wealth
    total_pension_liq = 0.0

    if config.start_age >= 65:
        p2 = config.initial_pillar_2
        p3a = sum(config.initial_pillar_3a_accounts) if config.initial_pillar_3a_accounts else 0.0
        total_pension_liq = p2 + p3a
    elif 60 <= config.start_age < 65 and config.initial_pillar_3a_accounts:
        for i, acc in enumerate(config.initial_pillar_3a_accounts):
            if config.start_age >= 60 + i and acc > 0.0:
                total_pension_liq = acc
                break

    if total_pension_liq > 0.0:
        cap_tax = float(calculate_capital_withdrawal_tax(
            total_pension_liq,
            config.cantonal_multiplier,
            config.municipal_multiplier
        ))
        init_wealth += max(0.0, total_pension_liq - cap_tax)

    return init_wealth


def estimate_year_0_taxes(config: SimConfig) -> float:
    """Estimates Year 0 total taxes (income, wealth, AHV non-worker) for initial Cash Tent and TWR calculation."""
    init_wealth = _get_year_0_liquid_wealth(config)

    equities = max(0.0, init_wealth * (config.alloc_us_stocks + config.alloc_non_us_stocks))
    dividends = equities * config.dividend_yield
    cash = max(0.0, init_wealth * config.alloc_chf_cash)
    interest = cash * max(0.0, config.cash_rate)
    annual_ahv = 12 * config.monthly_ahv_pension if config.start_age >= 65 else 0.0
    
    taxable_income = dividends + interest + annual_ahv
    income_tax = float(calculate_income_tax(taxable_income, config.cantonal_multiplier, config.municipal_multiplier))
    
    taxable_wealth = max(0.0, init_wealth - config.annual_base_expenses)
    wealth_tax = float(calculate_wealth_tax(taxable_wealth, config.cantonal_multiplier, config.municipal_multiplier))
    
    ahv_contrib = float(calculate_ahv_non_worker(taxable_wealth)) if config.start_age < 65 else 0.0
    
    return income_tax + wealth_tax + ahv_contrib


def get_target_weights(config: SimConfig, year: int) -> np.ndarray:
    """
    Returns the target asset allocation weights for a given simulation year.
    Supports constant target weights or dynamic Cash Tent glidepath weights.
    """
    base_weights = np.array([
        config.alloc_us_stocks,
        config.alloc_non_us_stocks,
        config.alloc_chf_cash,
        config.alloc_gold,
        config.alloc_bitcoin
    ], dtype=float)
    
    if config.rebalance_strategy != 'Cash Tent':
        return base_weights
        
    T = config.tent_duration_years
    w_base_cash = config.alloc_chf_cash
    
    if T <= 0 or year >= T:
        return base_weights
        
    init_wealth = _get_year_0_liquid_wealth(config)

    if init_wealth <= 0:
        return base_weights
        
    est_taxes = estimate_year_0_taxes(config)
    annual_outflow = config.annual_base_expenses + est_taxes
    peak_cash_amount = T * annual_outflow
    w_peak = min(1.0, peak_cash_amount / init_wealth)
    
    if w_peak <= w_base_cash:
        return base_weights
        
    # Linear glide of cash weight from w_peak at year 0 down to w_base_cash at year T
    cash_weight = w_peak - (year / T) * (w_peak - w_base_cash)
    
    non_cash_base_total = 1.0 - w_base_cash
    if non_cash_base_total <= 0:
        return base_weights
        
    non_cash_scale = (1.0 - cash_weight) / non_cash_base_total
    
    weights = base_weights * non_cash_scale
    weights[2] = cash_weight  # CHF Cash is index 2
    return weights


def _indexed_tax(tax_fn, taxable_amount: np.ndarray, bracket_factors: np.ndarray, *args) -> np.ndarray:
    """
    Applies a progressive tariff whose bracket edges are scaled by `bracket_factors`.

    Swiss law indexes tax bracket edges to the CPI so that purely nominal growth
    does not push a taxpayer into higher brackets: Art. 39 DBG requires the EFD to
    adjust the federal tariff annually, and § 48 StG ZH indexes the Canton Zurich
    income *and* wealth tariffs.

    Every tariff in `tax_engine` is positively homogeneous of degree 1 under a
    simultaneous scaling of the tax base and the bracket edges. This holds for the
    piecewise-linear income and wealth tariffs, and also for the AHV non-worker step
    function, because floor division is scale invariant (`(f*a) // (f*b) == a // b`).
    Scaling the brackets by `f` is therefore exactly equivalent to deflating the base
    by `f`, applying the unscaled tariff, and re-inflating the result:

        T_indexed(x, f) == f * T_nominal(x / f)

    Note that only franc-denominated amounts scale; the bracket *rates* and the
    Steuerfuss multipliers are unaffected, which is correct since both are political
    parameters rather than inflation-linked ones.
    """
    return bracket_factors * tax_fn(taxable_amount / bracket_factors, *args)


def run_simulation(config: SimConfig, return_matrix: np.ndarray, inflation_matrix: np.ndarray = None) -> dict:
    """
    Executes the multi-asset monthly simulation loop.
    return_matrix shape: (num_runs, duration_months, 5)
    """
    num_runs = config.num_runs
    duration_months = config.duration_years * 12

    expected_ret_shape = (num_runs, duration_months, 5)
    if return_matrix.shape != expected_ret_shape:
        raise ValueError(
            f"return_matrix shape {return_matrix.shape} must match {expected_ret_shape}."
        )
    if inflation_matrix is not None:
        expected_inf_shape = (num_runs, config.duration_years)
        if inflation_matrix.shape != expected_inf_shape:
            raise ValueError(
                f"inflation_matrix shape {inflation_matrix.shape} must match {expected_inf_shape}."
            )
    
    # Target weights only change at year boundaries (Cash Tent glidepath), and each
    # evaluation re-runs the Year 0 tax estimate, so compute them once per year.
    target_weights_by_year = np.array([get_target_weights(config, y) for y in range(config.duration_years)])
    initial_target_weights = target_weights_by_year[0]
    
    # Assets (0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin)
    liquid_assets = np.zeros((num_runs, 5))
    liquid_assets[:] = config.initial_liquid_wealth * initial_target_weights
    
    pillar_2 = np.full(num_runs, config.initial_pillar_2, dtype=float)
    pillar_3a = [np.full(num_runs, acc, dtype=float) for acc in config.initial_pillar_3a_accounts]
    
    initial_net_worth = config.initial_liquid_wealth + config.initial_pillar_2 + sum(config.initial_pillar_3a_accounts)
    
    history_net_worth = np.zeros((duration_months // 12, num_runs))
    history_liquid = np.zeros((duration_months // 12, num_runs))
    history_liquid_by_class = np.zeros((duration_months // 12, num_runs, 5))
    history_pillar_2 = np.zeros((duration_months // 12, num_runs))
    history_pillar_3a = np.zeros((duration_months // 12, num_runs))
    history_taxes = np.zeros((duration_months // 12, num_runs))
    history_expenses = np.zeros((duration_months // 12, num_runs))
    history_income_divs = np.zeros((duration_months // 12, num_runs))
    history_income_ahv = np.zeros((duration_months // 12, num_runs))
    history_below_watermark = np.zeros((duration_months // 12, num_runs), dtype=bool)
    
    inflation_factors = np.ones(num_runs)
    bracket_factors = np.ones(num_runs)
    rng_inf = np.random.default_rng(config.seed + 10_000)
    vanguard_prev_expenses = np.full(num_runs, config.annual_base_expenses, dtype=float)
    vanguard_prev_inflation = np.ones(num_runs, dtype=float)
    taxable_liquidation_amount = np.zeros(num_runs)
    capital_withdrawal_tax_this_year = np.zeros(num_runs)

    # Hoist loop-invariant constants outside the monthly simulation loop
    total_stocks = config.alloc_us_stocks + config.alloc_non_us_stocks
    if total_stocks > 0:
        weight_us = config.alloc_us_stocks / total_stocks
        weight_non_us = config.alloc_non_us_stocks / total_stocks
    else:
        weight_us, weight_non_us = 0.5, 0.5
    penalty_monthly = (1.05) ** (1.0 / 12.0) - 1.0
    
    for m in range(duration_months):
        year = m // 12
        month_of_year = m % 12
        current_age = config.start_age + year
        
        current_target_weights = target_weights_by_year[year]
        # A year-end (month 11) rebalance sets the allocation held throughout the
        # *next* year, so it must target next year's glidepath weights. Otherwise the
        # Cash Tent lags by one year and base weights are only reached in year T + 1.
        if month_of_year == 11 and year + 1 < config.duration_years:
            rebalance_target_weights = target_weights_by_year[year + 1]
        else:
            rebalance_target_weights = current_target_weights

        # 1. Inflation (Update once a year at month 0)
        if month_of_year == 0:
            if inflation_matrix is not None:
                inflation = inflation_matrix[:, year]
            else:
                inflation = rng_inf.normal(config.inflation_mean, config.inflation_std, num_runs)
            inflation_factors *= np.maximum(0.01, 1.0 + inflation)
            bracket_factors *= np.maximum(0.01, 1.0 + config.bracket_indexation * inflation)
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
                    liquid_assets[mask] += np.outer(amount, current_target_weights)
                    p3a[mask] = 0
                    has_liquidated[mask] = True
            else:
                # Age >= 65: Liquidate ALL remaining accounts
                for p3a in pillar_3a:
                    mask = p3a > 0
                    amount = p3a[mask]
                    taxable_liquidation_amount[mask] += amount
                    liquid_assets[mask] += np.outer(amount, current_target_weights)
                    p3a[mask] = 0
                
            mask_p2 = (current_age >= 65) & (pillar_2 > 0)
            amount_p2 = pillar_2[mask_p2]
            taxable_liquidation_amount[mask_p2] += amount_p2
            liquid_assets[mask_p2] += np.outer(amount_p2, current_target_weights)
            pillar_2[mask_p2] = 0
            
            # Calculate Capital Withdrawal Tax immediately upon liquidation
            has_liquidation = taxable_liquidation_amount > 0
            if np.any(has_liquidation):
                cap_tax = np.zeros(num_runs)
                cap_tax[has_liquidation] = _indexed_tax(
                    calculate_capital_withdrawal_tax,
                    taxable_liquidation_amount[has_liquidation],
                    bracket_factors[has_liquidation],
                    config.cantonal_multiplier,
                    config.municipal_multiplier
                )
                
                # Deduct immediately at source (proportionally as it was added)
                liquid_assets -= np.outer(cap_tax, current_target_weights)
                
                # Track it for the annual history log
                capital_withdrawal_tax_this_year += cap_tax
        
        # 3. Market Returns
        monthly_returns = return_matrix[:, m, :] # (num_runs, 5)
        
        # Apply 100% Equity returns to Pillar 2 and 3a (proportional to US/Non-US target alloc)
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
        returns_to_apply = np.where(liquid_assets > 0, monthly_returns, penalty_monthly)
        returns_to_apply = np.maximum(-1.0, returns_to_apply)
        liquid_assets *= (1 + returns_to_apply)
        
        # 4. Rebalancing Check
        do_rebalance = np.zeros(num_runs, dtype=bool)
        
        # 4.5. Monthly AHV Pension Addition
        if current_age >= 65:
            monthly_ahv = config.monthly_ahv_pension * inflation_factors
            liquid_assets[:, 2] += monthly_ahv # Add directly to CHF Cash

        
        total_liquid_val = np.sum(liquid_assets, axis=1)
        is_solvent = total_liquid_val > 0
        
        if config.rebalance_strategy == 'Monthly':
            do_rebalance[:] = True
        elif config.rebalance_strategy == 'Quarterly' and month_of_year % 3 == 2:
            do_rebalance[:] = True
        elif config.rebalance_strategy in ('Yearly', 'Cash Tent') and month_of_year == 11:
            do_rebalance[:] = True
        elif config.rebalance_strategy == 'Threshold':
            current_total = np.sum(liquid_assets, axis=1, keepdims=True)
            current_total_safe = np.where(current_total > 0, current_total, 1.0)
            current_weights = liquid_assets / current_total_safe
            drift = np.max(np.abs(current_weights - current_target_weights), axis=1)
            do_rebalance = (drift > config.rebalance_threshold)

        # Do not rebalance bankrupt or depleted portfolios
        do_rebalance &= is_solvent
            
        if config.enable_smart_selling:
            current_p3a = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            current_nw = np.sum(liquid_assets, axis=1) + pillar_2 + current_p3a
            inflation_adjusted_initial_nw = initial_net_worth * inflation_factors
            is_downturn = current_nw < inflation_adjusted_initial_nw
            do_rebalance[is_downturn] = False
            
        if np.any(do_rebalance):
            total_to_rebalance = np.sum(liquid_assets[do_rebalance], axis=1, keepdims=True)
            liquid_assets[do_rebalance] = total_to_rebalance * rebalance_target_weights
            
        # 5. Annual Taxation and Expenses
        if month_of_year == 11:
            # Spending Model Adjustment
            total_p3a_current = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            current_nw_before_expenses = np.sum(liquid_assets, axis=1) + pillar_2 + total_p3a_current
            
            inflation_adjusted_initial_nw = initial_net_worth * inflation_factors
            is_below_watermark = current_nw_before_expenses < inflation_adjusted_initial_nw
            history_below_watermark[year, :] = is_below_watermark
            
            if config.spending_strategy == "Vanguard Dynamic":
                inf_step = inflation_factors / vanguard_prev_inflation
                prior_exp_inflated = vanguard_prev_expenses * inf_step
                
                target_exp = config.vanguard_target_rate * np.maximum(0, current_nw_before_expenses)
                floor_exp = prior_exp_inflated * (1.0 - config.vanguard_floor_pct)
                ceiling_exp = prior_exp_inflated * (1.0 + config.vanguard_ceiling_pct)
                
                current_expenses = np.clip(target_exp, floor_exp, ceiling_exp)
                
                vanguard_prev_expenses = current_expenses.copy()
                vanguard_prev_inflation = inflation_factors.copy()
            elif config.spending_strategy == "Dynamic (Floor & Ceiling)":
                base_exp = config.annual_base_expenses * inflation_factors
                is_above_watermark = current_nw_before_expenses > inflation_adjusted_initial_nw
                
                current_expenses = np.where(
                    is_below_watermark,
                    base_exp * config.dynamic_expense_floor_pct,
                    np.where(
                        is_above_watermark,
                        base_exp * config.dynamic_expense_ceiling_pct,
                        base_exp
                    )
                )
            else: # "Static"
                current_expenses = config.annual_base_expenses * inflation_factors
                                            
            annual_ahv_received = np.where(current_age >= 65, 12 * config.monthly_ahv_pension * inflation_factors, 0.0)
            
            # Taxable Income = Dividends (yield on Equity portion)
            # US Stocks (0) and Non-US Stocks (1) generate dividends
            equities = np.maximum(0, liquid_assets[:, 0] + liquid_assets[:, 1])
            dividends = equities * config.dividend_yield
            
            # Cash (index 2) generates taxable interest from the realized annual return on CHF Cash
            # in return_matrix[:, (m - 11):(m + 1), 2] (floored at 0.0 so negative rates never reduce taxable income).
            cash = np.maximum(0, liquid_assets[:, 2])
            realized_cash_ret = np.prod(1.0 + return_matrix[:, (m - 11) : (m + 1), 2], axis=1) - 1.0
            interest = cash * np.maximum(0.0, realized_cash_ret)
            
            taxable_income = dividends + interest + annual_ahv_received
            
            income_tax = _indexed_tax(calculate_income_tax, taxable_income, bracket_factors, config.cantonal_multiplier, config.municipal_multiplier)
            
            total_liquid_end = np.sum(liquid_assets, axis=1)
            # Wealth Tax and AHV non-worker tax are assessed on wealth *after* deducting living expenses
            taxable_wealth = np.maximum(0, total_liquid_end - current_expenses)
            
            wealth_tax = _indexed_tax(calculate_wealth_tax, taxable_wealth, bracket_factors, config.cantonal_multiplier, config.municipal_multiplier)
            
            ahv_mask = current_age < 65
            ahv_contrib = np.zeros(num_runs)
            ahv_contrib[ahv_mask] = _indexed_tax(calculate_ahv_non_worker, taxable_wealth[ahv_mask], bracket_factors[ahv_mask])
            
            # Capital withdrawal tax was already deducted at source in month 0
            total_taxes = income_tax + wealth_tax + ahv_contrib
            net_cash_flow = current_expenses + total_taxes # AHV already added to cash monthly
            
            deficit = np.maximum(0, net_cash_flow)
            
            # Amount we can actually pay from current positive assets
            payable = np.minimum(deficit, np.maximum(0, total_liquid_end))
            
            # Any remaining deficit goes directly to negative Cash
            remaining_deficit = deficit - payable
            
            if config.enable_smart_selling:
                # Pay from cash first for downturn runs (reuse is_below_watermark computed above)
                downturn_payable = payable[is_below_watermark]
                available_cash = np.maximum(0, liquid_assets[is_below_watermark, 2])
                sell_cash = np.minimum(downturn_payable, available_cash)
                
                liquid_assets[is_below_watermark, 2] -= sell_cash
                payable[is_below_watermark] -= sell_cash
            
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
            history_pillar_2[year, :] = pillar_2
            history_pillar_3a[year, :] = total_p3a
            history_taxes[year, :] = total_taxes + capital_withdrawal_tax_this_year
            history_expenses[year, :] = current_expenses
            history_income_divs[year, :] = dividends
            history_income_ahv[year, :] = annual_ahv_received

    return {
        'net_worth': history_net_worth,
        'liquid_assets': history_liquid,
        'liquid_assets_by_class': history_liquid_by_class,
        'pillar_2': history_pillar_2,
        'pillar_3a': history_pillar_3a,
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
    seed: int = 42,
    btc_equity_corr: float = 0.50,
    exus_equity_corr: float = 0.75,
    gold_equity_corr: float = 0.08,
) -> np.ndarray:
    """Generates monthly returns using a correlated lognormal model to prevent negative asset prices.

    Draws US Stocks standard normal shocks `z_us` first (preserving stream alignment),
    then couples Non-US Stocks (`exus_equity_corr=0.75`), Gold (`gold_equity_corr=0.08`),
    and Bitcoin (`btc_equity_corr=0.50` default) to `z_us` via Gaussian copula.
    """
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")
    if duration_years <= 0:
        raise ValueError(f"duration_years must be positive, got {duration_years}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if any(r <= -1.0 for r in (ret_us, ret_non_us, ret_cash, ret_gold, ret_btc)):
        raise ValueError("Expected annual returns must be greater than -100% (-1.0)")
    if any(v < 0.0 for v in (vol_eq, vol_gold, vol_btc)):
        raise ValueError("Volatilities cannot be negative")
    if any(not (-1.0 <= c <= 1.0) for c in (btc_equity_corr, exus_equity_corr, gold_equity_corr)):
        raise ValueError("Asset correlations must be between -1.0 and 1.0")

    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    shape = (num_runs, duration_months)

    def _from_z(ann_ret: float, ann_vol: float, z: np.ndarray) -> np.ndarray:
        drift = np.log(1.0 + ann_ret) - 0.5 * (ann_vol ** 2)
        log_ret = (drift / 12.0) + (ann_vol / np.sqrt(12.0)) * z
        return np.exp(log_ret) - 1.0

    def _coupled_z(z_base: np.ndarray, rho: float) -> np.ndarray:
        eps = rng.normal(0.0, 1.0, shape)
        return rho * z_base + np.sqrt(max(0.0, 1.0 - rho ** 2)) * eps

    z_us = rng.normal(0.0, 1.0, shape)
    z_non_us = _coupled_z(z_us, exus_equity_corr)
    z_gold = _coupled_z(z_us, gold_equity_corr)
    z_btc = _coupled_z(z_us, btc_equity_corr)

    matrix[:, :, 0] = _from_z(ret_us, vol_eq, z_us)
    matrix[:, :, 1] = _from_z(ret_non_us, vol_eq, z_non_us)
    # Cash is constant nominal return (geometric monthly rate)
    matrix[:, :, 2] = (1.0 + ret_cash) ** (1.0 / 12.0) - 1.0
    matrix[:, :, 3] = _from_z(ret_gold, vol_gold, z_gold)
    matrix[:, :, 4] = _from_z(ret_btc, vol_btc, z_btc)

    return matrix


def generate_monte_carlo_inflation(
    num_runs: int,
    duration_years: int,
    inflation_mean: float = 0.015,
    inflation_std: float = 0.01,
    seed: int = 42
) -> np.ndarray:
    """Generates independent annual inflation draws for Monte Carlo simulations.

    Uses a dedicated RNG stream offset (`seed + 10_000`) so inflation draws
    do not collide with the standard normal sequence used for US Stock returns
    in `generate_monte_carlo_returns`.
    """
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")
    if duration_years <= 0:
        raise ValueError(f"duration_years must be positive, got {duration_years}")
    if inflation_std < 0.0:
        raise ValueError(f"inflation_std cannot be negative, got {inflation_std}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    rng = np.random.default_rng(seed + 10_000)
    inflation_matrix = rng.normal(inflation_mean, inflation_std, (num_runs, duration_years))
    return np.maximum(-0.99, inflation_matrix)

