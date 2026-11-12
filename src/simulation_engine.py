import numpy as np
from dataclasses import dataclass, field
from .jurisdiction import (
    COUNTRY_DEFAULTS,
    STATE_PENSION_AGE,
    get_jurisdiction,
    indexed_tax,
    unsupported_country_message,
)
from .tax_engine import (
    ZURICH_CANTONAL_MULTIPLIER,
    ZURICH_CITY_MUNICIPAL_MULTIPLIER,
)
from .tax_engine_at import HEALTH_SELF_INSURANCE_MONTHLY, HEALTH_SELF_INSURANCE_REAL_GROWTH

VALID_SPENDING_STRATEGIES = {"Static", "Dynamic (Floor & Ceiling)", "Vanguard Dynamic"}
VALID_REBALANCE_STRATEGIES = {"Cash Tent", "Monthly", "Quarterly", "Yearly", "Threshold", "Never"}
# Strategies that always rebalance at year end, right after the annual outflow sale.
YEAR_END_REBALANCE_STRATEGIES = ("Monthly", "Quarterly", "Yearly", "Cash Tent")
# Number of bisection steps used to solve the year-end sale S = expenses + taxes(gains_0 + gain(S)).
# Because E + T(S) - S is continuous and strictly decreasing for both capital gains (0 <= g <= 1,
# slope(T) <= 0.55 < 1) and capital losses (g < 0, slope(T) <= 0, where Picard iteration oscillates
# whenever |g * marginal_rate| > 1), [E + min(T(0), T(cap)), E + max(T(0), T(cap))] brackets the
# unique root and 50 bisection steps shrink the interval by 2^-50 ~ 9e-16.
OUTFLOW_TAX_BISECTION_STEPS = 50


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
    # Monthly gross state pension from age 65 in today's money: AHV in CH (paid 12x/yr), the
    # ASVG state pension in AT (paid 14x/yr, see COUNTRY_DEFAULTS[country].pension_payments_per_year).
    monthly_ahv_pension: float = 0.0
    
    # Taxes (Steuerfuss). Defaults are Canton Zurich 2026 (0.95) + City of Zurich (1.19).
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER

    # Fraction of realized CPI applied annually to the tax bracket edges.
    # 1.0 = full indexation (Art. 39 DBG / § 48 StG ZH); 0.0 = frozen nominal
    # brackets, i.e. the retiree absorbs the full bracket creep.
    # None -> COUNTRY_DEFAULTS[country].bracket_indexation (CH 1.0, AT 2/3 per § 33a EStG).
    bracket_indexation: float | None = None

    # Cash Tent Strategy (Pfau Glidepath)
    tent_duration_years: int = 7

    # Expected real appreciation of the HOME currency (CHF for "CH", EUR for "AT") beyond Relative
    # Purchasing Power Parity. Not used by run_simulation itself; the app passes it to the matching
    # return module (real_chf_appreciation / real_eur_appreciation). 0.0 = PPP neutrality (default);
    # HISTORIC_REAL_{CHF,EUR}_APPRECIATION (derived from the data at build time) reproduces raw history.
    real_chf_appreciation: float = 0.0

    # Expected cash yield used for Year 0 tax estimation (and Cash Tent buffer sizing).
    # None -> COUNTRY_DEFAULTS[country].cash_rate (CH 1%, AT 2%).
    cash_rate: float | None = None

    # Correlation between synthetic Bitcoin monthly log-returns and US equity log-returns.
    btc_equity_corr: float = 0.50

    # Random Seed for reproducible stochastic simulation
    seed: int = 42

    # Tax jurisdiction (ISO 3166-1 alpha-2): selects the `jurisdiction.Jurisdiction` rules.
    # "CH" (Switzerland, Canton Zurich) or "AT" (Austria).
    country: str = "CH"

    # AT only: monthly § 16 ASVG health self-insurance contribution in today's EUR (2026 rate),
    # charged until the state pension starts. Ignored in CH.
    health_insurance_monthly: float = HEALTH_SELF_INSURANCE_MONTHLY
    # AT only: annual growth of that contribution beyond CPI. It follows the wage-based
    # Aufwertungszahl, so it is indexed with CPI x (1 + this)^year. Ignored in CH.
    health_insurance_real_growth: float = HEALTH_SELF_INSURANCE_REAL_GROWTH

    def __post_init__(self):
        if self.country not in COUNTRY_DEFAULTS:
            raise ValueError(unsupported_country_message(self.country))
        defaults = COUNTRY_DEFAULTS[self.country]
        if self.bracket_indexation is None:
            self.bracket_indexation = defaults.bracket_indexation
        if self.cash_rate is None:
            self.cash_rate = defaults.cash_rate

        for int_name in ("num_runs", "duration_years", "start_age", "tent_duration_years", "seed"):
            val = getattr(self, int_name)
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise ValueError(f"{int_name} ({val!r}) must be an integer.")

        if self.num_runs <= 0 or self.duration_years <= 0:
            raise ValueError(f"num_runs ({self.num_runs}) and duration_years ({self.duration_years}) must be positive integers.")
            
        if self.start_age < 0:
            raise ValueError(f"start_age ({self.start_age}) cannot be negative.")

        if self.tent_duration_years < 0:
            raise ValueError(f"tent_duration_years ({self.tent_duration_years}) cannot be negative.")

        if self.initial_pillar_3a_accounts is None:
            self.initial_pillar_3a_accounts = []

        float_fields = (
            self.inflation_mean,
            self.inflation_std,
            self.dividend_yield,
            self.dynamic_expense_floor_pct,
            self.dynamic_expense_ceiling_pct,
            self.vanguard_target_rate,
            self.vanguard_floor_pct,
            self.vanguard_ceiling_pct,
            self.initial_liquid_wealth,
            self.initial_pillar_2,
            self.alloc_us_stocks,
            self.alloc_non_us_stocks,
            self.alloc_chf_cash,
            self.alloc_gold,
            self.alloc_bitcoin,
            self.rebalance_threshold,
            self.annual_base_expenses,
            self.monthly_ahv_pension,
            self.cantonal_multiplier,
            self.municipal_multiplier,
            self.bracket_indexation,
            self.real_chf_appreciation,
            self.cash_rate,
            self.btc_equity_corr,
            self.health_insurance_monthly,
            self.health_insurance_real_growth,
            *self.initial_pillar_3a_accounts,
        )
        if not all(np.isfinite(x) for x in float_fields):
            raise ValueError("All numeric SimConfig parameters must be finite (no NaN or inf).")

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

        if self.spending_strategy == "Dynamic (Floor & Ceiling)" and self.dynamic_expense_floor_pct > self.dynamic_expense_ceiling_pct:
            raise ValueError(
                f"dynamic_expense_floor_pct ({self.dynamic_expense_floor_pct}) cannot exceed "
                f"dynamic_expense_ceiling_pct ({self.dynamic_expense_ceiling_pct})."
            )

        if self.rebalance_strategy not in VALID_REBALANCE_STRATEGIES:
            raise ValueError(f"Invalid rebalance_strategy '{self.rebalance_strategy}'. Must be one of {sorted(VALID_REBALANCE_STRATEGIES)}.")

        if self.health_insurance_monthly < 0.0:
            raise ValueError(f"health_insurance_monthly ({self.health_insurance_monthly}) cannot be negative.")

        if self.health_insurance_real_growth <= -1.0:
            raise ValueError(
                f"health_insurance_real_growth ({self.health_insurance_real_growth}) must be greater than -1.0."
            )

        if self.country == "AT" and (self.initial_pillar_2 > 0.0 or any(a > 0.0 for a in self.initial_pillar_3a_accounts)):
            raise ValueError("Pillar 2 / Pillar 3a balances are Swiss-only and must be 0 when country == 'AT'.")


def _get_year_0_liquid_wealth(config: SimConfig) -> float:
    """Year 0 liquid wealth incl. Day-1 pension lump sums net of tax (see `Jurisdiction.year_0_liquid_wealth`)."""
    return get_jurisdiction(config.country).year_0_liquid_wealth(config)


def _cash_tent_weights_for_year(
    base_weights: np.ndarray,
    tent_duration_years: int,
    year: int,
    init_wealth: float,
    annual_outflow: float,
) -> np.ndarray:
    """Computes Cash Tent weights for `year` given Year-0 liquid wealth and annual outflow."""
    T = tent_duration_years
    w_base_cash = float(base_weights[2])
    if T <= 0 or year >= T or init_wealth <= 0:
        return base_weights
    w_peak = min(1.0, (T * annual_outflow) / init_wealth)
    if w_peak <= w_base_cash:
        return base_weights
    cash_weight = w_peak - (year / T) * (w_peak - w_base_cash)
    non_cash_base_total = 1.0 - w_base_cash
    if non_cash_base_total <= 0:
        return base_weights
    non_cash_scale = (1.0 - cash_weight) / non_cash_base_total
    weights = base_weights * non_cash_scale
    weights[2] = cash_weight
    return weights


def _estimate_year_0_taxes_for_weights(
    config: SimConfig, init_wealth: float, weights: np.ndarray
) -> float:
    return get_jurisdiction(config.country).estimate_year_0_taxes_for_weights(config, init_wealth, weights)


def estimate_year_0_taxes(config: SimConfig) -> float:
    """Estimates Year 0 total taxes (income, wealth, AHV non-worker) for initial Cash Tent and TWR calculation.

    When `rebalance_strategy == 'Cash Tent'`, evaluates dividends and cash interest on Year 0's
    actual Cash Tent allocation rather than post-tent base weights (using a 2-step fixed point).
    """
    init_wealth = _get_year_0_liquid_wealth(config)
    base_weights = np.array([
        config.alloc_us_stocks,
        config.alloc_non_us_stocks,
        config.alloc_chf_cash,
        config.alloc_gold,
        config.alloc_bitcoin,
    ], dtype=float)

    est_tax = _estimate_year_0_taxes_for_weights(config, init_wealth, base_weights)
    if config.rebalance_strategy == "Cash Tent" and config.tent_duration_years > 0 and init_wealth > 0:
        for _ in range(2):
            w0 = _cash_tent_weights_for_year(
                base_weights, config.tent_duration_years, 0, init_wealth, config.annual_base_expenses + est_tax
            )
            est_tax = _estimate_year_0_taxes_for_weights(config, init_wealth, w0)
    return est_tax


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
    if T <= 0 or year >= T:
        return base_weights
        
    init_wealth = _get_year_0_liquid_wealth(config)
    if init_wealth <= 0:
        return base_weights
        
    est_taxes = estimate_year_0_taxes(config)
    annual_outflow = config.annual_base_expenses + est_taxes
    return _cash_tent_weights_for_year(base_weights, T, year, init_wealth, annual_outflow)


# Backwards-compatible alias; the implementation lives in `jurisdiction.indexed_tax`.
_indexed_tax = indexed_tax


def _deferred_withdrawal_tax(
    pillar_2: np.ndarray,
    pillar_3a: list,
    from_age: int,
    bracket_factors: np.ndarray,
    config: SimConfig,
) -> np.ndarray:
    """Tax still owed on the current pension balances, per run (see `Jurisdiction.latent_tax`)."""
    return get_jurisdiction(config.country).latent_tax(config, pillar_2, pillar_3a, from_age, bracket_factors)


# Asset sleeve indices of `liquid_assets` / `cost_basis`.
ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_CASH, ASSET_GOLD, ASSET_BITCOIN = range(5)
# Sleeves whose sales can realize capital gains (cash is always held at cost).
NON_CASH_ASSETS = np.array([ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_GOLD, ASSET_BITCOIN])


def _basis_ratio(values: np.ndarray, cost_basis: np.ndarray) -> np.ndarray:
    """Average-cost ratio `basis / value` per sleeve; 1.0 where the holding is not positive."""
    return np.divide(cost_basis, values, out=np.ones_like(values, dtype=float), where=values > 0)


def _sell(liquid_assets: np.ndarray, cost_basis: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    """Sells `amounts` (shape of `liquid_assets`, 0 <= amounts <= positive holdings) in place.

    Uses the average-cost method (the Austrian "gleitender Durchschnittspreis"): each sale
    removes `amounts * basis / value` of cost basis and realizes the rest as gain (negative = loss).
    Returns the realized gains per run and sleeve.
    """
    basis_sold = amounts * _basis_ratio(liquid_assets, cost_basis)
    cost_basis -= basis_sold
    liquid_assets -= amounts
    return amounts - basis_sold


def _buy(liquid_assets: np.ndarray, cost_basis: np.ndarray, amounts: np.ndarray) -> None:
    """Buys `amounts` at market value in place (cost basis increases by the amount paid)."""
    liquid_assets += amounts
    cost_basis += amounts


def _rebalance_to(
    liquid_assets: np.ndarray, cost_basis: np.ndarray, mask: np.ndarray, target_values: np.ndarray
) -> np.ndarray:
    """Sets `liquid_assets[mask] = target_values`, booking the implied sells and buys against the cost basis.

    Sleeves above target are sold (average cost, gains realized), sleeves below target are bought.
    The holdings are *assigned* rather than adjusted trade by trade so the resulting balances are
    bit-identical to a plain reallocation. Returns the realized gains per run and sleeve (zero
    outside `mask`).
    """
    realized = np.zeros_like(liquid_assets)
    old = liquid_assets[mask]
    basis = cost_basis[mask]
    delta = target_values - old
    sells = np.maximum(0.0, -delta)
    basis_sold = sells * _basis_ratio(old, basis)
    realized[mask] = sells - basis_sold
    cost_basis[mask] = basis - basis_sold + np.maximum(0.0, delta)
    liquid_assets[mask] = target_values
    return realized


def _unrealized_gains(liquid_assets: np.ndarray, cost_basis: np.ndarray) -> np.ndarray:
    """Net unrealized gains over the non-cash sleeves, per run (losses offset gains)."""
    return (liquid_assets[:, NON_CASH_ASSETS] - cost_basis[:, NON_CASH_ASSETS]).sum(axis=1)


def _unassessed_gains(
    liquid_assets: np.ndarray,
    cost_basis: np.ndarray,
    carried_taxable_gains: np.ndarray,
    realized_gains_this_year: np.ndarray,
) -> np.ndarray:
    """Net non-cash capital gains not yet taxed (realized gains awaiting assessment + unrealized gains)."""
    return (
        carried_taxable_gains
        + realized_gains_this_year[:, NON_CASH_ASSETS].sum(axis=1)
        + _unrealized_gains(liquid_assets, cost_basis)
    )


def _outflow_sale_gain_profile(
    liquid_assets: np.ndarray,
    cost_basis: np.ndarray,
    cash_first: np.ndarray,
    distributed_cash: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Describes the realized gain of the year-end outflow sale as a function of its size.

    Mirrors the outflow sale in `run_simulation` (Month 11): the payable amount is capped at total
    liquid wealth; runs in `cash_first` (Smart Cash Buffer downturn, or a year-end rebalance follows
    where sales are taxed) spend all positive cash first, the other runs only the year's cash
    distributions (`distributed_cash`, at most the positive cash); then all runs sell their remaining
    positive holdings proportionally. Returns `(gain_fraction, cash_spent_first, cap)` such that the
    realized non-cash gain of an outflow `S` is `gain_fraction * max(0, min(S, cap) - cash_spent_first)`.
    """
    positive = np.maximum(0.0, liquid_assets)
    unrealized = (positive[:, NON_CASH_ASSETS] * (1.0 - _basis_ratio(liquid_assets, cost_basis)[:, NON_CASH_ASSETS])).sum(axis=1)
    all_total = positive.sum(axis=1)
    cash_spent_first = _cash_spent_first(liquid_assets, cash_first, distributed_cash)
    denom = all_total - cash_spent_first
    gain_fraction = np.divide(unrealized, denom, out=np.zeros_like(unrealized), where=denom > 0)
    cap = np.maximum(0.0, liquid_assets.sum(axis=1))
    return gain_fraction, cash_spent_first, cap


def _cash_spent_first(
    liquid_assets: np.ndarray, cash_first: np.ndarray, distributed_cash: np.ndarray | None
) -> np.ndarray:
    """Cash used for the outflow before any holding is sold: all positive cash for `cash_first`
    runs, otherwise the year's cash distributions (capped at the positive cash balance)."""
    positive_cash = np.maximum(0.0, liquid_assets[:, ASSET_CASH])
    distributed = np.zeros_like(positive_cash) if distributed_cash is None else np.minimum(positive_cash, distributed_cash)
    return np.where(cash_first, positive_cash, distributed)


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
    
    jurisdiction = get_jurisdiction(config.country)
    pension_payments_per_year = jurisdiction.defaults.pension_payments_per_year
    # Monthly cash credit of the state pension (14 payments/yr are spread over 12 months).
    monthly_pension_credit = config.monthly_ahv_pension * (pension_payments_per_year / 12)

    # Target weights only change at year boundaries (Cash Tent glidepath), and each
    # evaluation re-runs the Year 0 tax estimate, so compute them once per year.
    target_weights_by_year = np.array([get_target_weights(config, y) for y in range(config.duration_years)])
    initial_target_weights = target_weights_by_year[0]
    
    # Assets (0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin)
    liquid_assets = np.zeros((num_runs, 5))
    liquid_assets[:] = config.initial_liquid_wealth * initial_target_weights
    # Average-cost basis per sleeve. The whole starting portfolio is treated as bought on day 0
    # (zero unrealized gains). Cash is always held at cost (`cost_basis[:, ASSET_CASH]` is synced
    # to the cash balance each month), so only the non-cash sleeves can realize gains.
    cost_basis = liquid_assets.copy()
    realized_gains_this_year = np.zeros((num_runs, 5))
    # Gains of the previous year-end rebalance: that rebalance runs after the year's tax assessment,
    # so (like a rebalance on the first trading day of January) its gains are taxed next year.
    carried_taxable_gains = np.zeros(num_runs)
    
    pillar_2 = np.full(num_runs, config.initial_pillar_2, dtype=float)
    pillar_3a = [np.full(num_runs, acc, dtype=float) for acc in config.initial_pillar_3a_accounts]
    
    initial_net_worth = config.initial_liquid_wealth + config.initial_pillar_2 + sum(config.initial_pillar_3a_accounts)
    # Watermark decisions (Smart Cash Buffer, Dynamic Floor & Ceiling, "Years Below") compare
    # *after-tax* net worth: pension balances are pre-tax claims that lose the capital withdrawal
    # tax on payout. Both sides of the comparison are net, so a pension-heavy portfolio is not
    # flagged as "below watermark" merely because the tax is realized at 60-65.
    initial_net_worth_after_tax = initial_net_worth - jurisdiction.latent_tax(
        config, pillar_2, pillar_3a, config.start_age, np.ones(num_runs),
        _unassessed_gains(liquid_assets, cost_basis, carried_taxable_gains, realized_gains_this_year)
        if jurisdiction.taxes_capital_gains
        else None,
    )
    
    history_net_worth = np.zeros((duration_months // 12, num_runs))
    history_liquid = np.zeros((duration_months // 12, num_runs))
    history_liquid_by_class = np.zeros((duration_months // 12, num_runs, 5))
    history_pillar_2 = np.zeros((duration_months // 12, num_runs))
    history_pillar_3a = np.zeros((duration_months // 12, num_runs))
    history_taxes = np.zeros((duration_months // 12, num_runs))
    history_expenses = np.zeros((duration_months // 12, num_runs))
    history_income_divs = np.zeros((duration_months // 12, num_runs))
    history_income_interest = np.zeros((duration_months // 12, num_runs))
    history_income_ahv = np.zeros((duration_months // 12, num_runs))
    history_unfunded_deficit = np.zeros((duration_months // 12, num_runs))
    history_below_watermark = np.zeros((duration_months // 12, num_runs), dtype=bool)
    history_cost_basis_by_class = np.zeros((duration_months // 12, num_runs, 5))
    history_realized_gains = np.zeros((duration_months // 12, num_runs))
    history_unrealized_gains = np.zeros((duration_months // 12, num_runs))
    history_tax_components: dict[str, np.ndarray] = {}
    
    inflation_factors = np.ones(num_runs)
    bracket_factors = np.ones(num_runs)
    indexed_cpi_level = np.ones(num_runs)
    rng_inf = np.random.default_rng(config.seed + 10_000)
    init_vanguard_exp = (
        config.annual_base_expenses
        if config.annual_base_expenses > 0.0
        else (config.vanguard_target_rate * initial_net_worth if config.spending_strategy == "Vanguard Dynamic" else 0.0)
    )
    vanguard_prev_expenses = np.full(num_runs, init_vanguard_exp, dtype=float)
    vanguard_prev_inflation = np.ones(num_runs, dtype=float)
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

        # 1. Start-of-Year Pension Liquidations (Month 0, evaluated at start-of-year bracket_factors)
        if month_of_year == 0:
            realized_gains_this_year.fill(0.0)
            liquidation = jurisdiction.liquidate_pensions(config, pillar_2, pillar_3a, current_age, bracket_factors)
            # Tax deducted at source, tracked for the annual history log
            capital_withdrawal_tax_this_year = liquidation.tax
            if np.any(liquidation.gross > 0):
                net_proceeds = np.maximum(0.0, liquidation.gross - liquidation.tax)
                # Pay off any negative CHF Cash debt from prior bridge deficits first so solvent
                # portfolios never carry involuntary 5% margin debt alongside leveraged equities.
                existing_debt = np.maximum(0.0, -liquid_assets[:, 2])
                debt_repaid = np.minimum(net_proceeds, existing_debt)
                liquid_assets[:, 2] += debt_repaid
                surplus_proceeds = net_proceeds - debt_repaid
                _buy(liquid_assets, cost_basis, np.outer(surplus_proceeds, current_target_weights))

        # 2. Annual Inflation & Bracket Indexation Update (Month 0, accruing over the year)
        if month_of_year == 0:
            if inflation_matrix is not None:
                inflation = inflation_matrix[:, year]
            else:
                inflation = rng_inf.normal(config.inflation_mean, config.inflation_std, num_runs)
            inflation_factors *= np.maximum(0.01, 1.0 + inflation)
            # Brackets are never lowered in deflation (Art. 39 Abs. 2 DBG / § 48 StG ZH: high-water
            # mark of the CPI level; § 33a EStG: negative years skipped). `indexed_cpi_level` is the
            # unfloored indexation-weighted CPI level the Swiss high-water mark is measured against.
            indexed_cpi_level *= np.maximum(0.01, 1.0 + config.bracket_indexation * inflation)
            bracket_factors = jurisdiction.next_bracket_factors(
                bracket_factors, indexed_cpi_level, inflation, config.bracket_indexation
            )
        
        # 3. Market Returns
        monthly_returns = return_matrix[:, m, :] # (num_runs, 5)
        
        # Apply 100% Equity returns to Pillar 2 and 3a (proportional to US/Non-US target alloc)
        equity_monthly_returns = np.maximum(
            -1.0, monthly_returns[:, 0] * weight_us + monthly_returns[:, 1] * weight_non_us
        )
        
        pillar_2 *= (1 + equity_monthly_returns)
        for p3a in pillar_3a:
            p3a *= (1 + equity_monthly_returns)
        
        # Handle Bankruptcy (Total liquid < 0)
        total_liquid = np.sum(liquid_assets, axis=1, keepdims=True)
        is_bankrupt = (total_liquid.flatten() < 0)
        
        if np.any(is_bankrupt):
            # Any positive non-cash holdings are used up against the debt: book them as sold.
            bankrupt_holdings = np.maximum(0.0, liquid_assets[is_bankrupt][:, NON_CASH_ASSETS])
            realized_gains_this_year[np.ix_(is_bankrupt, NON_CASH_ASSETS)] += (
                bankrupt_holdings - cost_basis[np.ix_(is_bankrupt, NON_CASH_ASSETS)]
                * (bankrupt_holdings > 0)
            )
            cost_basis[is_bankrupt, :] = 0.0
            # Move all debt to CHF Cash (index 2), zero out other assets
            liquid_assets[is_bankrupt, :] = 0
            liquid_assets[is_bankrupt, 2] = total_liquid[is_bankrupt].flatten()
            
        # Positive balances get market returns. Negative balances get a 5% APY penalty (converted to monthly)
        returns_to_apply = np.where(liquid_assets > 0, monthly_returns, penalty_monthly)
        returns_to_apply = np.maximum(-1.0, returns_to_apply)
        liquid_assets *= (1 + returns_to_apply)
        
        # 4. Monthly State Pension Addition (AHV in CH)
        if current_age >= STATE_PENSION_AGE:
            liquid_assets[:, 2] += monthly_pension_credit * inflation_factors  # Add directly to Cash
        # Cash is held at cost: interest, pension credits and debt never create capital gains.
        cost_basis[:, ASSET_CASH] = liquid_assets[:, ASSET_CASH]

        # 4.5. Intra-Year Rebalancing Check (Months 0..10; Month 11 rebalances after annual tax/expenses)
        if month_of_year < 11:
            do_rebalance = np.zeros(num_runs, dtype=bool)
            total_liquid_val = np.sum(liquid_assets, axis=1)
            is_solvent = total_liquid_val > 0

            if config.rebalance_strategy == 'Monthly':
                do_rebalance[:] = True
            elif config.rebalance_strategy == 'Quarterly' and month_of_year % 3 == 2:
                do_rebalance[:] = True
            elif config.rebalance_strategy == 'Threshold':
                current_total = np.sum(liquid_assets, axis=1, keepdims=True)
                current_total_safe = np.where(current_total > 0, current_total, 1.0)
                current_weights = liquid_assets / current_total_safe
                drift = np.max(np.abs(current_weights - current_target_weights), axis=1)
                do_rebalance = (drift > config.rebalance_threshold)

            do_rebalance &= is_solvent

            if config.enable_smart_selling and np.any(do_rebalance):
                current_p3a = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
                # This year's liquidation already ran at month 0, so the remaining schedule starts next year.
                current_nw_after_tax = (np.sum(liquid_assets, axis=1) + pillar_2 + current_p3a
                                        - jurisdiction.latent_tax(
                                            config, pillar_2, pillar_3a, current_age + 1, bracket_factors,
                                            _unassessed_gains(
                                                liquid_assets, cost_basis, carried_taxable_gains, realized_gains_this_year
                                            )
                                            if jurisdiction.taxes_capital_gains
                                            else None,
                                        ))
                is_downturn = current_nw_after_tax < initial_net_worth_after_tax * inflation_factors
                do_rebalance[is_downturn] = False

            if np.any(do_rebalance):
                total_to_rebalance = np.sum(liquid_assets[do_rebalance], axis=1, keepdims=True)
                realized_gains_this_year += _rebalance_to(
                    liquid_assets, cost_basis, do_rebalance, total_to_rebalance * current_target_weights
                )
            
        # 5. Annual Taxation, Expenses, and Year-End Rebalancing (Month 11)
        if month_of_year == 11:
            # Spending Model Adjustment
            total_p3a_current = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            current_nw_before_expenses = np.sum(liquid_assets, axis=1) + pillar_2 + total_p3a_current
            nw_after_tax_before_expenses = current_nw_before_expenses - jurisdiction.latent_tax(
                config, pillar_2, pillar_3a, current_age + 1, bracket_factors,
                _unassessed_gains(liquid_assets, cost_basis, carried_taxable_gains, realized_gains_this_year)
                if jurisdiction.taxes_capital_gains
                else None,
            )

            inflation_adjusted_watermark = initial_net_worth_after_tax * inflation_factors
            is_below_watermark = nw_after_tax_before_expenses < inflation_adjusted_watermark
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
                is_above_watermark = nw_after_tax_before_expenses > inflation_adjusted_watermark
                
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
                                            
            annual_ahv_received = np.where(
                current_age >= STATE_PENSION_AGE,
                pension_payments_per_year * config.monthly_ahv_pension * inflation_factors,
                0.0,
            )
            
            # Taxable Income = Dividends (yield on Equity portion held during this year, BEFORE year-end glidepath rebalance)
            # US Stocks (0) and Non-US Stocks (1) generate dividends
            equities = np.maximum(0, liquid_assets[:, 0] + liquid_assets[:, 1])
            dividends = equities * config.dividend_yield

            # Cash (index 2) generates taxable interest from the realized annual return on CHF Cash
            # in return_matrix[:, (m - 11):(m + 1), 2] (floored at 0.0 so negative rates never reduce taxable income).
            cash = np.maximum(0, liquid_assets[:, 2])
            realized_cash_ret = np.prod(1.0 + return_matrix[:, (m - 11) : (m + 1), 2], axis=1) - 1.0
            interest = cash * np.maximum(0.0, realized_cash_ret)

            if jurisdiction.taxes_capital_gains:
                # Distributing funds: the year's dividends are paid out to cash (the total-return
                # holdings drop by the distribution, their cost basis is unchanged) and fund the
                # year-end outflow before any holding is sold. Reinvesting them and then selling the
                # full outflow would realize extra taxable gains on an appreciated position.
                distributions = np.maximum(0.0, liquid_assets[:, :2]) * config.dividend_yield
                liquid_assets[:, :2] -= distributions
                distributed_cash = distributions.sum(axis=1)
                liquid_assets[:, ASSET_CASH] += distributed_cash
                cost_basis[:, ASSET_CASH] = liquid_assets[:, ASSET_CASH]
            else:
                # No capital gains tax: the dividends are treated as reinvested (the return series
                # are total-return), raising the cost basis of the sleeve that paid them.
                cost_basis[:, :2] += np.maximum(0.0, liquid_assets[:, :2]) * config.dividend_yield
                distributed_cash = None
            
            total_liquid_end = np.sum(liquid_assets, axis=1)
            taxable_gains_before_sale = carried_taxable_gains + realized_gains_this_year[:, NON_CASH_ASSETS].sum(axis=1)

            def assess(realized_gains):
                return jurisdiction.assess_annual_taxes(
                    config,
                    current_age=current_age,
                    total_liquid_end=total_liquid_end,
                    current_expenses=current_expenses,
                    dividends=dividends,
                    interest=interest,
                    annual_pension=annual_ahv_received,
                    bracket_factors=bracket_factors,
                    realized_gains=realized_gains,
                    inflation_factors=inflation_factors,
                )

            # Runs that pay the outflow from positive cash before selling other holdings: Smart Cash
            # Buffer runs in a downturn and, where sales are taxed, every run under a strategy whose
            # year-end rebalance follows right after the sale. Selling holdings proportionally and
            # then buying them back with the leftover cash in the same tick would realize taxable
            # gains for nothing; spending the cash first reaches the same post-rebalance allocation
            # with the smallest sale. (Without a capital gains tax the order makes no difference.)
            cash_first = is_below_watermark & config.enable_smart_selling
            if jurisdiction.taxes_capital_gains and config.rebalance_strategy in YEAR_END_REBALANCE_STRATEGIES:
                cash_first = np.ones(num_runs, dtype=bool)

            annual_taxes = assess(taxable_gains_before_sale)
            if jurisdiction.taxes_capital_gains:
                # The outflow sale realizes gains/losses that are taxed in the same year: solve
                # h(S) = expenses + taxes(gains_so_far + gain(S)) - S = 0 by monotone bisection.
                gain_fraction, cash_spent_first, sale_cap = _outflow_sale_gain_profile(
                    liquid_assets, cost_basis, cash_first, distributed_cash
                )
                cap_gain = gain_fraction * np.maximum(0.0, sale_cap - cash_spent_first)
                taxes_at_0 = annual_taxes.total
                taxes_at_cap = assess(taxable_gains_before_sale + cap_gain).total
                lo = np.maximum(0.0, current_expenses + np.minimum(taxes_at_0, taxes_at_cap))
                hi = np.maximum(0.0, current_expenses + np.maximum(taxes_at_0, taxes_at_cap))
                if not np.allclose(lo, hi, rtol=0.0, atol=1e-12):
                    for _ in range(OUTFLOW_TAX_BISECTION_STEPS):
                        mid = 0.5 * (lo + hi)
                        mid_gain = gain_fraction * np.maximum(0.0, np.minimum(mid, sale_cap) - cash_spent_first)
                        mid_taxes = assess(taxable_gains_before_sale + mid_gain).total
                        above_zero = current_expenses + mid_taxes >= mid
                        lo = np.where(above_zero, mid, lo)
                        hi = np.where(above_zero, hi, mid)
                    mid = 0.5 * (lo + hi)
                    mid_gain = gain_fraction * np.maximum(0.0, np.minimum(mid, sale_cap) - cash_spent_first)
                    annual_taxes = assess(taxable_gains_before_sale + mid_gain)
            # Pension lump-sum taxes were already deducted at source in month 0
            total_taxes = annual_taxes.total
            net_cash_flow = current_expenses + total_taxes # AHV already added to cash monthly
            
            deficit = np.maximum(0, net_cash_flow)
            
            # Amount we can actually pay from current positive assets
            payable = np.minimum(deficit, np.maximum(0, total_liquid_end))
            
            # Any remaining deficit goes directly to negative Cash
            remaining_deficit = deficit - payable
            
            # Pay from positive cash first: all of it for the cash-first runs (see `cash_first`
            # above), otherwise the year's cash distributions (Austria, see `distributed_cash`).
            spend_first = np.minimum(payable, _cash_spent_first(liquid_assets, cash_first, distributed_cash))
            if np.any(spend_first > 0):
                liquid_assets[:, 2] -= spend_first
                payable -= spend_first
            
            # Sell proportionally from remaining positive assets
            positive_assets = np.maximum(0, liquid_assets)
            total_positive = np.sum(positive_assets, axis=1)
            safe_total_pos = np.where(total_positive > 0, total_positive, 1.0)
            
            fractions = payable / safe_total_pos
            sell_amounts = positive_assets * fractions[:, None]
            sale_realized = _sell(liquid_assets, cost_basis, sell_amounts)
            realized_gains_this_year += sale_realized
            
            liquid_assets[:, 2] -= remaining_deficit

            if jurisdiction.taxes_capital_gains:
                # Settle exactly on the gains the sale actually realized; any tiny float rounding
                # residual is booked against cash.
                final_taxes = assess(taxable_gains_before_sale + sale_realized[:, NON_CASH_ASSETS].sum(axis=1))
                tax_diff = final_taxes.total - total_taxes
                liquid_assets[:, 2] -= tax_diff
                liquid_assets[:, 2] = np.where(
                    (np.abs(tax_diff) < 1e-7) & (np.abs(liquid_assets[:, 2]) < 1e-9),
                    0.0,
                    liquid_assets[:, 2],
                )
                remaining_deficit = np.maximum(
                    0.0, current_expenses + final_taxes.total - np.maximum(0.0, total_liquid_end)
                )
                annual_taxes = final_taxes
                total_taxes = final_taxes.total

            # 6. Year-End Rebalancing (sets the allocation held going into the next year)
            do_rebalance_eoy = np.zeros(num_runs, dtype=bool)
            post_sale_liquid = np.sum(liquid_assets, axis=1)
            if config.rebalance_strategy in YEAR_END_REBALANCE_STRATEGIES:
                do_rebalance_eoy[:] = True
            elif config.rebalance_strategy == 'Threshold':
                current_total = post_sale_liquid[:, None]
                current_total_safe = np.where(current_total > 0, current_total, 1.0)
                current_weights = liquid_assets / current_total_safe
                drift = np.max(np.abs(current_weights - current_target_weights), axis=1)
                do_rebalance_eoy = (drift > config.rebalance_threshold)

            do_rebalance_eoy &= (post_sale_liquid > 0)
            if config.enable_smart_selling:
                do_rebalance_eoy[is_below_watermark] = False

            if np.any(do_rebalance_eoy):
                eoy_realized = _rebalance_to(
                    liquid_assets,
                    cost_basis,
                    do_rebalance_eoy,
                    post_sale_liquid[do_rebalance_eoy, None] * rebalance_target_weights,
                )
                realized_gains_this_year += eoy_realized
                carried_taxable_gains = eoy_realized[:, NON_CASH_ASSETS].sum(axis=1)
            else:
                carried_taxable_gains = np.zeros(num_runs)
            cost_basis[:, ASSET_CASH] = liquid_assets[:, ASSET_CASH]
            
            # Record
            total_p3a = sum(pillar_3a) if pillar_3a else np.zeros(num_runs)
            net_worth = np.sum(liquid_assets, axis=1) + pillar_2 + total_p3a
            
            history_net_worth[year, :] = net_worth
            history_liquid[year, :] = np.sum(liquid_assets, axis=1)
            history_liquid_by_class[year, :, :] = liquid_assets
            history_pillar_2[year, :] = pillar_2
            history_pillar_3a[year, :] = total_p3a
            history_taxes[year, :] = total_taxes + capital_withdrawal_tax_this_year
            for component, values in annual_taxes.components.items():
                if component not in history_tax_components:
                    history_tax_components[component] = np.zeros((duration_months // 12, num_runs))
                history_tax_components[component][year, :] = values
            history_expenses[year, :] = current_expenses
            history_income_divs[year, :] = dividends
            history_income_interest[year, :] = interest
            history_income_ahv[year, :] = annual_ahv_received
            history_unfunded_deficit[year, :] = remaining_deficit
            history_cost_basis_by_class[year, :, :] = cost_basis
            history_realized_gains[year, :] = realized_gains_this_year[:, NON_CASH_ASSETS].sum(axis=1)
            history_unrealized_gains[year, :] = (
                liquid_assets[:, NON_CASH_ASSETS] - cost_basis[:, NON_CASH_ASSETS]
            ).sum(axis=1)

    return {
        'net_worth': history_net_worth,
        'liquid_assets': history_liquid,
        'liquid_assets_by_class': history_liquid_by_class,
        'pillar_2': history_pillar_2,
        'pillar_3a': history_pillar_3a,
        'taxes_paid': history_taxes,
        'expenses_paid': history_expenses,
        'income_dividends': history_income_divs,
        'income_interest': history_income_interest,
        'income_ahv': history_income_ahv,
        'unfunded_deficit': history_unfunded_deficit,
        'below_watermark': history_below_watermark,
        'cost_basis_by_class': history_cost_basis_by_class,
        'realized_gains': history_realized_gains,
        'unrealized_gains': history_unrealized_gains,
        'tax_components': history_tax_components,
        'initial_net_worth': initial_net_worth,
        'initial_net_worth_after_tax': float(initial_net_worth_after_tax[0]),
    }


def historic_lognormal_params(annual_returns: np.ndarray) -> tuple[float, float]:
    """Monte Carlo parameters `(ann_ret, ann_vol)` that reproduce a historical annual series.

    Inverse of the parameterisation in `generate_monte_carlo_returns`: there `ann_ret`
    is the *arithmetic* mean of annual simple returns (E[1 + R] = 1 + ann_ret) and
    `ann_vol` is the standard deviation of annual *log* returns. Feeding the simple-return
    standard deviation instead would overstate risk (e.g. US stocks in CHF 1922-2025:
    23.1% simple vs 22.2% log).
    """
    r = np.asarray(annual_returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("annual_returns must be a 1-D series with at least 2 observations")
    if np.any(r <= -1.0):
        raise ValueError("annual returns must be greater than -100%")
    return float(r.mean()), float(np.log1p(r).std(ddof=1))


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
    if any(not np.isfinite(r) or r <= -1.0 for r in (ret_us, ret_non_us, ret_cash, ret_gold, ret_btc)):
        raise ValueError("Expected annual returns must be greater than -100% (-1.0) and finite")
    if any(not np.isfinite(v) or v < 0.0 for v in (vol_eq, vol_gold, vol_btc)):
        raise ValueError("Volatilities cannot be negative and must be finite")
    if any(not np.isfinite(c) or not (-1.0 <= c <= 1.0) for c in (btc_equity_corr, exus_equity_corr, gold_equity_corr)):
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
    if not np.isfinite(inflation_mean) or inflation_mean <= -1.0:
        raise ValueError(f"inflation_mean must be finite and > -1.0, got {inflation_mean}")
    if not np.isfinite(inflation_std) or inflation_std < 0.0:
        raise ValueError(f"inflation_std cannot be negative, got {inflation_std}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    rng = np.random.default_rng(seed + 10_000)
    inflation_matrix = rng.normal(inflation_mean, inflation_std, (num_runs, duration_years))
    return np.maximum(-0.99, inflation_matrix)

