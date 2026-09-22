# Design Document: Zurich Early Retirement Simulator (Phase 1)

## 1. Overview
This document outlines the technical design for Phase 1 of the Zurich Early Retirement Simulator. Phase 1 focuses exclusively on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich. 

The system will ingest initial asset balances, apply historic or Monte Carlo investment returns in CHF, compute complex Swiss tax obligations (including non-worker AHV contributions and capital withdrawal taxes), and visualize the resulting net worth trajectories.

## 2. Architecture
To prioritize rapid development, rich interactive visualizations, and local data privacy, the application will be built using a **Python-based data science stack** and a **Streamlit** frontend.

*   **Frontend / UI**: Streamlit (Python). Runs locally, ensuring no sensitive financial data leaves the user's machine. Provides built-in interactive widgets and seamless integration with charting libraries.
*   **Core Engine**: Python 3.11+.
*   **Vectorization & Math**: `NumPy` and `Pandas` for running thousands of simulation iterations concurrently and efficiently.
*   **Visualization**: `Plotly` for interactive, performant web-based charts (crucial for "spaghetti" plots showing 10,000 runs).

## 3. Core Modules

The system is divided into four primary modules:

### 3.1. User State & Configuration (Input Layer)
A centralized data structure (`SimConfig`) representing the simulation parameters, enforcing strict mathematical validation in `__post_init__`:
*   **Validation Rules**: `num_runs > 0`, `duration_years > 0`, `start_age >= 0`, `tent_duration_years >= 0`, `inflation_std >= 0.0`, `seed >= 0`, `spending_strategy in VALID_SPENDING_STRATEGIES`, `rebalance_strategy in VALID_REBALANCE_STRATEGIES`, individual asset allocations $\ge 0.0$ and $\sum \text{allocations} = 1.0$, non-negative initial asset balances (`initial_liquid_wealth`, `initial_pillar_2`, `initial_pillar_3a_accounts`), non-negative expenses/income/yields (`annual_base_expenses`, `monthly_ahv_pension`, `dividend_yield`), non-negative tax multipliers (`cantonal_multiplier`, `municipal_multiplier`), non-negative rebalancing thresholds, and non-negative spending strategy parameters (`vanguard_target_rate`, `vanguard_floor_pct`, `vanguard_ceiling_pct`, `dynamic_expense_floor_pct`, `dynamic_expense_ceiling_pct`).
*   **Demographics**: Retirement age (`start_age`), simulation duration (`duration_years`).
*   **Economics & Taxes**: Inflation mean & volatility (for Monte Carlo), dividend yield, cantonal tax multiplier (`0.95` for Canton Zurich 2026), and municipal tax multiplier (Steuerfuss, e.g., `1.19` for Zurich City).
*   **Assets**: Taxable Liquid Wealth, Pillar 2 (`Freizügigkeitskonto`), Pillar 3a balances (stored as a list up to 5 accounts to support staggered withdrawals).
*   **Expenses & Income**: Annual base expenses, Spending Strategy (`Static`, `Dynamic (Floor & Ceiling)`, `Vanguard Dynamic`), and Expected **Monthly** Pillar 1 (AHV) Pension from age 65.
*   **Simulation Config**: Simulation Mode (`Historic Backtesting`, `Historic Bootstrapping`, `Monte Carlo`), Number of Runs (`mc_num_runs`, `boot_num_runs`), **Random Seed** (`random_seed`, default `42`), and Success Criteria (`success_pct` of inflation-adjusted starting net worth).

### 3.2. Tax & Pension Engine
This is a stateless utility module (`src/tax_engine.py`) responsible for all Swiss-specific tax and social security calculations for a given year. All functions are vectorized, support both scalar and N-dimensional array inputs while preserving return type and shape, and safely clamp negative inputs to zero.

*   **Income Tax (`calculate_income_tax(taxable_income, cantonal_multiplier=1.0, municipal_multiplier=1.19)`)**:
    *   Applies Federal progressive brackets (Single tariff).
    *   Applies Zurich Cantonal progressive base brackets (Single tariff) multiplied by `cantonal_multiplier + municipal_multiplier` (e.g., `0.95 + 1.19 = 2.14` in `app.py` for Zurich City 2026).
*   **Wealth Tax (`calculate_wealth_tax(taxable_wealth, cantonal_multiplier=1.0, municipal_multiplier=1.19)`)**:
    *   Applies Zurich Cantonal progressive base wealth tax brackets multiplied by `cantonal_multiplier + municipal_multiplier`. Assessed on year-end liquid wealth after deducting the current year's living expenses (`max(0, total_liquid_end - current_expenses)`).
*   **Capital Withdrawal Tax (`calculate_capital_withdrawal_tax(amount, cantonal_multiplier=1.0, municipal_multiplier=1.19)`)**:
    *   Applied immediately at source when Pillar 2 or Pillar 3a accounts are liquidated. Uses the separate progressive capital withdrawal tax approximation: Federal at `1/5` of the standard Federal tariff, plus Zurich Cantonal base at `1/10` of the standard Zurich tariff multiplied by `cantonal_multiplier + municipal_multiplier`.
*   **AHV Non-Worker Contributions (`calculate_ahv_non_worker(wealth, imputed_pension_income=0)`)**:
    *   Calculated for early retirees (`current_age < 65`) based on the official 2025 AHV tables: `determining_wealth = wealth + 20 * imputed_pension_income` (where `wealth` is `taxable_wealth`). Contribution is 530 CHF for wealth < 350k CHF. For wealth between 350k and 1.75M CHF, it adds 106 CHF for every 50k CHF step above 300k CHF. For wealth above 1.75M CHF, it adds 159 CHF for every 50k CHF step above 1.75M CHF. Capped at 26,500 CHF/year (2025 limits).
    *   *Note: The tax brackets and AHV contribution thresholds are modeled as nominal constants throughout the simulation, which is a conservative assumption.*
*   **Pillar 1 AHV Pension Payments**:
    *   Starting at age 65, the user receives a monthly AHV pension (`monthly_ahv_pension * inflation_factors`), added directly to CHF Cash each month. The input value (in today's CHF) compounds with annual CPI inflation from Year 0 (including all years prior to age 65).
    *   Annual AHV payouts (`12 * monthly_ahv_pension * inflation_factors`) are included in annual taxable income.

### 3.3. Simulation Engine (The Core Loop)
The engine (`run_simulation` in `src/simulation_engine.py`) executes a **monthly tick** (`m in range(duration_years * 12)`) for `num_runs` paths simultaneously using NumPy arrays across 5 liquid asset classes (`0: US Stocks`, `1: Non-US Stocks`, `2: CHF Cash`, `3: Gold`, `4: Bitcoin`).

**Monthly Tick Execution Order:**
1. **Annual Inflation Update (Month 0 of each year)**:
    * Updates cumulative `inflation_factors *= np.maximum(0.01, 1.0 + inflation)` using `inflation_matrix[:, year]` (or sampling from a deterministic `np.random.default_rng(config.seed + 10_000)` stream if `inflation_matrix` is `None`). Clamping `1.0 + inflation >= 0.01` prevents division-by-zero or sign inversion under extreme synthetic deflation draws.
2. **Pension Liquidations & Immediate Capital Withdrawal Tax (Month 0 of each year)**:
    * **Age < 65**: Staggered liquidation of Pillar 3a accounts at ages `60 + i` (`60, 61, 62, 63, 64` for up to 5 accounts, at most 1 account liquidated per year).
    * **Age $\ge 65$**: All remaining Pillar 3a accounts and the Pillar 2 (`Freizügigkeitskonto`) account are liquidated (if starting retirement at age $\ge 65$, this occurs immediately in Year 0 Month 0).
    * **At-Source Tax Deduction**: `calculate_capital_withdrawal_tax` is computed on the total pension liquidation for the year and deducted immediately at source; net proceeds are invested into `liquid_assets` proportionally to `current_target_weights`.
3. **Market Returns & Bankruptcy Handling (Every Month)**:
    * Remaining Pillar 2 and Pillar 3a balances grow by 100% equity monthly returns weighted by the user's US vs. Non-US stock target allocation (or 50/50 if target equity weight is zero).
    * **Bankruptcy Check**: If total liquid wealth is negative (`total_liquid < 0`), all negative debt is consolidated into CHF Cash (index 2) and other liquid asset classes are zeroed.
    * Positive liquid asset balances earn their respective monthly market returns (`return_matrix[:, m, :]`); negative balances incur a 5% APY debt interest penalty (`(1.05)**(1/12) - 1`).
4. **Monthly AHV Pension Addition (Every Month if `current_age >= 65`)**:
    * Adds `monthly_ahv_pension * inflation_factors` directly to `liquid_assets[:, 2]` (CHF Cash).
5. **Rebalancing Check (Monthly / Quarterly / Threshold / Month 11 for Yearly & Cash Tent)**:
    * Triggered every month (`Monthly`), at quarter-end (`Quarterly`, `month_of_year % 3 == 2`), at year-end (`Yearly` and `Cash Tent`, `month_of_year == 11`), or whenever max absolute weight drift exceeds `rebalance_threshold` (`Threshold`).
    * Only solvent portfolios (`total_liquid_val > 0`) are rebalanced.
    * **Cash Tent (`get_target_weights` & `_get_year_0_liquid_wealth`)**: Peak target cash weight at Year 0 is computed as `min(1.0, tent_duration_years * (annual_base_expenses + estimate_year_0_taxes(config)) / init_wealth)` (incorporating net Pillar 2/3a liquidations via `_get_year_0_liquid_wealth` for both `start_age >= 65` and `60 <= start_age < 65`), and glides linearly down to `alloc_chf_cash` over `tent_duration_years` (default: 7 years).
    * **Smart Cash Buffer**: If `enable_smart_selling` is enabled, rebalancing is skipped for any run currently in a market downturn (`current_nw < initial_net_worth * inflation_factors`).
6. **Annual Taxation, Spending & Outflow Deduction (Month 11 of each year)**:
    * **Spending Evaluation**:
        * **Static**: `current_expenses = annual_base_expenses * inflation_factors`.
        * **Dynamic (Floor & Ceiling)**: Scales inflation-adjusted base expenses by `dynamic_expense_floor_pct` when net worth is below the inflation-adjusted starting watermark, and by `dynamic_expense_ceiling_pct` when above.
        * **Vanguard Dynamic**: Recalculates target net living expenses as `vanguard_target_rate * max(0, current_nw_before_expenses)`, clamped between `(1 - vanguard_floor_pct)` and `(1 + vanguard_ceiling_pct)` of the prior year's inflation-adjusted living expenses. In `app.py`, the UI's tax-inclusive Target Withdrawal Rate ($\text{TWR}$) is bi-directionally synchronized with Year 0 Annual Base Expenses ($E_0$) via fixed-point iteration solving $E_0 + \text{Taxes}_0(E_0) = \text{TWR} \times \text{NetWorth}_0$ using live user tax/yield inputs, and converted to the net living expense rate `vanguard_target_rate = E_0 / NetWorth_0` for `SimConfig` so taxes are never double-counted.
    * **Tax Assessment**: Computes annual `income_tax` on `dividends` (`(US Stocks + Non-US Stocks) * dividend_yield`) + `interest` (`CHF Cash * 0.01`) + `annual_ahv_received`, plus `wealth_tax` and `ahv_contrib` (for `current_age < 65`) on `taxable_wealth = max(0, total_liquid_end - current_expenses)`.
    * **Asset Selling**: Deducts `deficit = current_expenses + total_taxes`. If `enable_smart_selling` is enabled and the run is in a downturn, positive CHF Cash is spent down first before selling other positive assets proportionally to their current holdings. Otherwise, all positive liquid assets are sold proportionally to their current holdings; any unpaid deficit beyond total liquid assets becomes negative CHF Cash.

### 3.4. Return Generators & Data Provenance
The system provides three complementary return simulation engines powered by empirical datasets from Baptiste Wicht (*The Poor Swiss*), hosted at [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator) (and documented at [The Poor Swiss](https://thepoorswiss.com)):
*   **Empirical Source Datasets (`data/`)**:
    *   `us_stocks.csv`: Robert Shiller monthly S&P 500 Total Returns (1871–2025).
    *   `ex_us_stocks.csv`: MSCI EAFE / World ex-US Total Returns proxy (1871–2025).
    *   `usd_chf.csv`: Historical monthly USD/CHF exchange rates (1913–2019 via *The Poor Swiss*, extended through 2025 via the Swiss National Bank).
    *   `ch_inflation.csv`: Historical Swiss Consumer Price Index (1921–2023 via *The Poor Swiss* / Swiss Federal Statistical Office, extended through 2025 via FSO).
*   **Historic Backtesting Mode (`get_historic_return_matrix`, `get_historic_inflation_matrix`)**: Ingests contiguous Swiss-adjusted historical equity returns and Swiss CPI inflation sequences (1922–2025 in CHF, 104 years). Produces $N = \text{total\_years} - \text{duration\_years} + 1$ overlapping cohorts. Uses empirical CHF returns for US and Non-US Stocks, exact 1% geometric nominal APY (`(1.01)**(1/12) - 1`) for CHF Cash, and Ito-corrected lognormal monthly returns (`seed=random_seed`) for Gold (`6%` mean, `15%` vol) and Bitcoin (`10%` mean, `60%` vol).
*   **Historic Bootstrapping Mode (`generate_bootstrapped_data`)**: Generates $N$ simulation runs of length `duration_years` seeded by `random_seed` by jointly sampling random historical calendar years with replacement for US Equities (CHF), Non-US Equities (CHF), and Swiss CPI inflation, paired with exact 1% geometric nominal APY for CHF Cash and vectorized Ito-corrected lognormal Gold/Bitcoin returns.
*   **Parametric Monte Carlo Mode (`generate_monte_carlo_returns`, `generate_monte_carlo_inflation`)**: Generates an asset return matrix of `shape=(num_runs, duration_months, 5)` seeded by `random_seed` using a lognormal model with Ito drift correction ($\text{drift} = \ln(1 + R) - 0.5\sigma^2$) for US Stocks, Non-US Stocks, Gold, and Bitcoin, constant geometric monthly return $(1 + R_{\text{cash}})^{1/12} - 1$ for CHF Cash, and independent normally distributed annual inflation `generate_monte_carlo_inflation` seeded at `random_seed + 10_000` (preventing RNG stream collisions with US Stock return draws).

## 4. Data Models

```python
@dataclass
class SimConfig:
    num_runs: int
    duration_years: int
    inflation_mean: float
    inflation_std: float
    start_age: int
    dividend_yield: float
    spending_strategy: str = "Static" # "Static", "Dynamic (Floor & Ceiling)", "Vanguard Dynamic" (UI default: "Vanguard Dynamic")
    enable_dynamic_expenses: bool = False
    dynamic_expense_floor_pct: float = 1.0
    dynamic_expense_ceiling_pct: float = 1.0
    vanguard_target_rate: float = 0.035
    vanguard_floor_pct: float = 0.05
    vanguard_ceiling_pct: float = 0.05
    initial_liquid_wealth: float = 0.0
    initial_pillar_2: float = 0.0
    initial_pillar_3a_accounts: list[float] = None
    alloc_us_stocks: float = 0.0
    alloc_non_us_stocks: float = 0.0
    alloc_chf_cash: float = 0.0
    alloc_gold: float = 0.0
    alloc_bitcoin: float = 0.0
    rebalance_strategy: str = "Yearly" # "Cash Tent", "Monthly", "Quarterly", "Yearly", "Threshold", "Never" (UI default: "Cash Tent")
    rebalance_threshold: float = 0.0
    enable_smart_selling: bool = True
    annual_base_expenses: float = 0.0
    monthly_ahv_pension: float = 0.0
    cantonal_multiplier: float = 0.0
    municipal_multiplier: float = 0.0
    tent_duration_years: int = 7
    seed: int = 42
```

**Simulation Output Dictionary:**
*   `net_worth`: Annual total net worth array `shape=(duration_years, num_runs)`
*   `liquid_assets`: Annual taxable liquid assets array `shape=(duration_years, num_runs)`
*   `liquid_assets_by_class`: Annual asset class breakdown `shape=(duration_years, num_runs, 5)`
*   `pillar_2`: Annual Pillar 2 balance `shape=(duration_years, num_runs)`
*   `pillar_3a`: Annual total Pillar 3a balance `shape=(duration_years, num_runs)`
*   `taxes_paid`: Annual taxes paid (income + wealth + AHV non-worker + capital withdrawal tax) `shape=(duration_years, num_runs)`
*   `expenses_paid`: Annual living expenses paid `shape=(duration_years, num_runs)`
*   `income_dividends`: Annual dividend income `shape=(duration_years, num_runs)`
*   `income_ahv`: Annual AHV pension received `shape=(duration_years, num_runs)`
*   `below_watermark`: Boolean mask indicating runs below starting watermark `shape=(duration_years, num_runs)`
*   `initial_net_worth`: Float scalar starting net worth

## 5. UI Layout (Streamlit)
*   **Sidebar**: All inputs categorized into 10 dedicated sections: **Demographics**, **Success Criteria**, **Initial Assets (CHF)**, **Target Asset Allocation (%)**, **Rebalancing Strategy**, **Withdrawal Strategy**, **Spending Strategy**, **Income & Yield**, **Monte Carlo Parameters** (including run counts and **Random Seed**), and **Zurich Tax Location**.
*   **Main Panel**: Displays results across three side-by-side columns for direct comparative analysis: **Historic Backtesting**, **Historic Bootstrapping**, and **Monte Carlo**. Each column contains mouse-over header tooltips and:
    *   **TL;DR Status**: Displays 'BROKE', 'RICH', or 'DEAD' based on median final net worth vs 3x inflation-adjusted initial net worth.
    *   **Metrics**: Probability of Success (based on selected success criteria), Avg Years Below Start NW, Median Ending Net Worth (Real & Nominal), Median Total Withdrawals (Real & Nominal), Pre-AHV Outflow (< Age 65), and Post-65 Outflow (Age 65+).
    *   **Net Worth Trajectory Chart** (Plotly): Faint lines for individual runs (all cohorts for Historic Backtesting; capped at 100 runs for Monte Carlo and Bootstrapping), bold lines for 5th, 25th, 50th, 75th, 95th percentiles, and dashed reference line for Inflation-Adj Start NW with horizontal bottom legend.
    *   **Income vs Required Cash Chart** (Plotly): Stacked bar chart showing median Dividends, AHV Pension, and Capital Sold, with a dashed reference line for Total Cash Needed (Expenses + Taxes) and bottom legend.
    *   **Annual Withdrawal Breakdown Chart** (Plotly): Stacked bar chart showing median living expenses and taxes paid over time with bottom legend.
    *   **Withdrawal Rate Chart** (Plotly): Percentile lines for the withdrawal rate over time (dynamically capped at 25% max) with bottom legend.
    *   **Asset Allocation Development Chart** (Plotly): Stacked area chart showing the median nominal balance of all asset categories (CHF Cash, US Stocks, Non-US Stocks, Gold, Bitcoin, Pillar 3a, and Pillar 2) over time to visualize portfolio glidepaths, rebalancing, and account liquidations with bottom legend.
    *   **Cohort / Run Analysis Tables**: Interactive tables of Top 10 Best and Worst cohorts/runs based on final net worth, displaying `Cohort`/`Run`, `Final NW (Real)`, `Final NW (Nom)`, `Min NW (Nom)`, and `Yrs Below` (years below the inflation-adjusted starting net worth watermark).

## 6. Implementation Plan & Milestones
1.  **Setup**: Initialize Git repo, project structure, and `requirements.txt` (Streamlit, Pandas, NumPy, Plotly, Pytest).
2.  **Tax Engine**: Implement and unit test the Zurich/Federal tax brackets and AHV non-worker logic.
3.  **Simulation Loop**: Build the vectorized monthly simulation engine with Cash Tent, Smart Cash Buffer, and dynamic spending rules.
4.  **UI Integration**: Build the 3-column comparative Streamlit frontend and connect the simulation engine.
5.  **Return Data**: Integrate the 1922–2025 historical Swiss dataset, bootstrapper, and lognormal Monte Carlo generator.

