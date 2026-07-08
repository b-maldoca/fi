# Design Document: Swiss Early Retirement Simulator (Phase 1)

## 1. Overview
This document outlines the technical design for Phase 1 of the Swiss Early Retirement Simulator. Phase 1 focuses exclusively on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich. 

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
A centralized data structure (e.g., a Pydantic model or Dataclass) representing the simulation parameters.
*   **Demographics**: Age, target duration.
*   **Economics**: Inflation rate, municipal tax multiplier (Steuerfuss).
*   **Assets**: Taxable Brokerage, Cash, Pillar 2 (Freizügigkeitskonto), Pillar 3a balances (stored as an array to support staggered withdrawals).
*   **Expenses & Income**: Annual base expenses, Expected **Monthly** Pillar 1 (AHV) Pension.
*   **Simulation Config**: Mode (Historic vs. Monte Carlo), Num Runs, Success Criteria (Survival vs. Capital Preservation).

### 3.2. Tax & Pension Engine
This is a stateless utility module responsible for all Swiss-specific calculations for a given year.

*   **Income Tax (`calculate_income_tax(taxable_income, steuerfuss)`)**: 
    *   Applies Federal progressive brackets (Single tariff).
    *   Applies Zurich Cantonal progressive brackets (Single tariff) multiplied by the Cantonal multiplier (100%) + Municipal `steuerfuss` (e.g., 119% for Zurich City).
*   **Wealth Tax (`calculate_wealth_tax(taxable_wealth, steuerfuss)`)**: 
    *   Applies Zurich Cantonal wealth tax brackets.
*   **Capital Withdrawal Tax (`calculate_capital_withdrawal_tax(amount, steuerfuss)`)**: 
    *   Applied when Pillar 2 or Pillar 3a accounts are liquidated. Uses the separate capital withdrawal tax rate (typically 1/5th or 1/10th of standard rate, depending on the canton's formula).
*   **AHV Non-Worker Contributions (`calculate_ahv_non_worker(wealth, imputed_pension_income=0)`)**: 
    *   Calculated based on the official 2025 AHV tables: `determining_wealth = wealth + 20 * imputed_pension_income`. Contribution is 530 CHF for wealth < 350k CHF. For wealth between 350k and 1.75M CHF, it adds 106 CHF for every 50k CHF step above 300k CHF. For wealth above 1.75M CHF, it adds 159 CHF for every 50k CHF step above 1.75M CHF. Capped at 26,500 CHF/year (2025 limits). In Phase 1, `imputed_pension_income` defaults to 0 in the simulation loop.
*   **Pillar 1 AHV Pension Payments**:
    *   Starting at age 65, the user receives an annual AHV pension. This is treated as taxable income (added to dividends) and reduces the required capital liquidation to meet annual expenses. If the pension exceeds expenses, the surplus is reinvested.

### 3.3. Simulation Engine (The Core Loop)
The engine executes a **monthly tick** for `N` runs simultaneously using NumPy arrays. It tracks multi-asset portfolios (US Stocks, Non-US Stocks, CHF Cash, Gold, Bitcoin) via N-dimensional arrays. It handles automatic rebalancing logic based on user configuration (Never, Monthly, Yearly, Threshold) and applies taxes at the end of each simulated year.

**Monthly Tick Logic:**
1. **Monthly Tick**: The simulation loop operates on a `duration_years * 12` timescale.
2. **Growth & Drift**: At the start of each month, the 5 asset classes (US Stocks, Non-US Stocks, Cash, Gold, Bitcoin) grow by their respective monthly returns. Pillar 2 and Pillar 3a accounts also grow monthly, assumed to be 100% invested in equities proportional to the user's US/Non-US target allocation.
3. **Pillar 1 (AHV) Pension (Monthly Check)**: Starting at age 65, the monthly AHV pension is added directly to the CHF Cash asset class each month, adjusted for inflation.
4. **Rebalancing (Monthly Check)**: If the rebalance strategy is 'Monthly', or if 'Threshold' is breached, or if it's the 12th month of the year and the strategy is 'Yearly', the asset weights are mathematically reset to the target allocation.
    * **Smart Cash Buffer**: If enabled, rebalancing is disabled during market downturns (net worth < inflation-adjusted starting net worth).
5. **Liquidation**: Age-triggered accounts (Pillar 3a at 60-64, Pillar 2 at 65) are liquidated in their respective months and moved into the taxable liquid wealth pool according to the target allocation.
6. **Annual Taxation**: Every 12 months, federal, cantonal, and municipal taxes are calculated based on end-of-year wealth and taxable income (dividends, AHV pension, capital withdrawals).
7. **Outflows**: Annual expenses and taxes are applied (divided monthly or lumped annually). Deductions are made by selling assets proportionally to their target allocation.
    * **Smart Cash Buffer**: If enabled and the portfolio is in a downturn, expenses are paid out of the CHF Cash allocation first, protecting equities from being sold at depressed prices.

### 3.4. Return Generator
*   **Historic Mode**: Loads a CSV of historic Swiss market returns (e.g., SPI, global equities hedged to CHF) and inflation. These returns are explicitly **nominal** (unadjusted for inflation). For a 50-year simulation, it selects contiguous 50-year blocks (e.g., 1900-1950, 1901-1951). If there are 100 years of data, it yields 50 distinct runs.
*   **Monte Carlo Mode**: Generates a matrix of `shape=(num_runs, simulation_years)` using `numpy.random.lognormal` based on user-provided **Nominal** Means ($\mu$) and Standard Deviations ($\sigma$) for individual asset classes, along with a constant or normally distributed inflation rate applied separately to expenses.

## 4. Data Models

```python
@dataclass
class SimulationState:
    age: np.ndarray             # shape: (num_runs,)
    taxable_brokerage: np.ndarray 
    cash: np.ndarray
    pillar_2: np.ndarray
    pillar_3a_accounts: list[np.ndarray] # List of accounts for staggering
    
@dataclass
class SimulationResult:
    year: int
    net_worth: np.ndarray
    income_tax_paid: np.ndarray
    wealth_tax_paid: np.ndarray
    capital_withdrawal_tax_paid: np.ndarray
    ahv_paid: np.ndarray
```

## 5. UI Layout (Streamlit)
*   **Sidebar**: All inputs (Initial Balances, Asset Allocation, Target Allocations, Economics, Monte Carlo Parameters, Success Criteria, Zurich Tax Multiplier).
*   **Main Panel**: Displays results in two sequential full-width sections: **Historic Returns** followed by **Monte Carlo**. Each section contains:
    *   **TL;DR Status**: Displays 'BROKE', 'RICH', or 'DEAD' based on median final net worth vs 3x inflation-adjusted initial net worth.
    *   **Metrics**: Probability of Success (based on selected success criteria), Median Ending Net Worth (Real), and Median Ending Net Worth (Nominal). Also displays average years below watermark.
    *   **Net Worth Trajectory Chart** (Plotly): Faint lines for individual runs (capped at 100 for Monte Carlo), bold lines for 5th, 25th, 50th, 75th, 95th percentiles.
    *   **Income vs Required Cash Chart** (Plotly): Stacked bar chart showing median Dividends, AHV Pension, and Capital Sold, with a reference line for Total Cash Needed (Expenses + Taxes).
    *   **Withdrawal Rate Chart** (Plotly): Percentile lines for the withdrawal rate over time.
    *   **Expenses & Taxes Chart** (Plotly): Stacked bar chart showing median Expenses and Taxes paid over time.
    *   **Run Analysis Tables**: Lists of top 5 best and worst runs/cohorts based on final net worth (showing Final NW (Real), Final NW (Nom), and Min NW (Nominal) values).

## 6. Implementation Plan & Milestones
1.  **Setup**: Initialize Git repo, basic project structure, and `requirements.txt` (Streamlit, Pandas, NumPy, Plotly).
2.  **Tax Engine**: Implement and unit test the Zurich/Federal tax brackets and AHV non-worker logic. *This is the most mathematically rigorous step.*
3.  **Simulation Loop**: Build the vectorized simulation engine.
4.  **UI Integration**: Build the Streamlit frontend and connect the engine.
5.  **Return Data**: Integrate a basic historic dataset and the Monte Carlo generator.

