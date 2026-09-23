# Product Requirement Document (PRD): Zurich Early Retirement Simulator

## 1. Executive Summary
The goal of this project is to build a financial simulation tool designed for individuals based in Switzerland who wish to plan for early retirement (FIRE - Financial Independence, Retire Early). 

The tool will simulate the user's financial life over a configurable period (e.g., 50 years), taking into account the unique aspects of Swiss taxation, the Swiss three-pillar pension system, and the cost of living in Canton Zurich. It will help users determine their readiness for retirement, optimize their withdrawal strategies, and visualize their net worth trajectory.

## 2. Target Audience & User Persona
*   **Role**: High-earning professional based in Zurich.
*   **Residency/Legal Status**: C Permit holder or Swiss Citizen (meaning they file standard tax returns and are not taxed at source).
*   **Aspiration**: Retire early (e.g., age 40–55) and remain in Canton Zurich (or nearby municipalities) for the next 50+ years.
*   **Financial Profile**: High income, significant portion of compensation in RSUs/GSUs, active contributor to Pillar 2 (Pensionskasse) and Pillar 3a, taxable brokerage accounts.

## 3. Scope & Phasing of the Emulation
To manage complexity, the project will be split into two development phases:
*   **Phase 1 (Current Focus): Post-Retirement (Decumulation Phase) Only**. The simulation starts on the day of retirement with a user-specified initial portfolio (including Pillar 2 cash and Pillar 3a cash ready for withdrawal, and taxable brokerage). The engine simulates decumulation, taxes, investment returns, and mandatory AHV contributions for non-workers until the end of the simulation.
*   **Phase 2: Pre-Retirement (Accumulation Phase) & Transition**. Adds compensation modeling, active saving, and optimization of transition (e.g., staggering Pillar 3a withdrawals, timing of retiring, voluntary Pillar 2 buy-ins).

### Phase 1 In-Scope
*   Modeling return scenarios using three complementary simulation methods, calculated in **CHF**:
    *   **Historic Backtesting**: Replaying contiguous historical market periods.
    *   **Historic Bootstrapping (Sampling with Replacement)**: Stochastic sampling of empirical annual returns with replacement across thousands of runs to test non-historical return sequences.
    *   **Parametric Monte Carlo**: Stochastic simulation using lognormal asset return distributions.
*   Zurich Cantonal, Municipal, and Federal Income & Wealth tax rules.
*   Mandatory AHV contributions for non-working early retirees.
*   Pillar 2 modeled via a **Freizügigkeitskonto (Vesting Account)**, restricted to **lump-sum withdrawal (Kapitalbezug)** at retirement (no annuities for now).
*   Pillar 3a lump-sum withdrawals.
*   Basic cost of living / expense modeling.
*   Configurable definition of "Success" for the simulation.
*   Advanced net worth trajectory chart showing all simulation paths, median, and key percentiles.

### Phase 1 Out-of-Scope
*   Pre-retirement phase (Salary, RSUs, active accumulation) -> deferred to Phase 2.
*   Pillar 2 Annuity (Rente) option -> deferred to Phase 2.
*   Real estate / mortgage calculations.
*   Relocating outside Canton Zurich.


---

## 4. Functional Requirements

### 4.1. Data Inputs

The application must allow the user to input the following parameters:

#### A. Demographics & Timeline
*   Current Age (e.g., 45 - representing retirement age in Phase 1)
*   Simulation Duration (e.g., 50 years)
*   Civil Status: Fixed to **Single** (for Phase 1 simplification of tax brackets).
*   Number of Dependents / Children: Fixed to **0**.


#### B. Income (Deferred to Phase 2)
*   *Note: In Phase 1, the user is assumed to be retired. No employment income is modeled.*


#### C. Current Assets & Pensions (At Start of Simulation)
*   **Initial Assets**:
    *   Taxable Liquid Wealth (will be allocated according to the target portfolio)
    *   Pillar 2 (Freizügigkeitskonto) balance.
    *   List of Pillar 3a account balances.
*   **Target Asset Allocation**:
    *   US Stocks (%)
    *   Non-US Stocks (%)
    *   CHF Cash (%)
    *   Gold (%)
    *   Bitcoin (%)
    *   *Constraint: Sum of allocation must exactly equal 100%.*
*   **Rebalancing Strategy**:
    *   **Cash Tent (Pfau Rising Equity Glide Path)**: A defensive sequence-of-returns risk strategy using cash instead of bonds. Assumes a cash buffer of [Tent Duration × (Annual Base Expenses + Estimated Taxes)] is already built up at retirement start (Year 0), and linearly glides down over the configured tent duration (default: 7 years) to the baseline target cash weight, allowing equity allocations to rise over time.
    *   **Monthly**: Rebalanced to target weights every month.
    *   **Quarterly**: Rebalanced to target weights once every 3 months.
    *   **Yearly**: Rebalanced to target weights once every 12 months.
    *   **Threshold**: Rebalanced when any asset drifts beyond a user-configured threshold (e.g. 1.0%).
    *   **Never**: Assets drift naturally.
*   **Smart Cash Buffer (Selling Strategy):** A defensive mechanism during market downturns. If the net worth drops below the inflation-adjusted starting net worth, the engine skips portfolio rebalancing and forces all expenses to be paid out of the CHF Cash allocation first, protecting equities from being sold at depressed prices. Normal proportional selling and rebalancing resumes once the portfolio recovers above the watermark.

#### D. Expenses, Income & Simulation Parameters (Post-Retirement)
*   Projected annual base retirement expenses (post-retirement) in CHF.
*   **Spending Strategy Selector**: Configurable spending behavior with three selectable models:
    *   **Static Spending**: Base expenses grow strictly with CPI inflation.
    *   **Dynamic Spending (Floor & Ceiling)**: Expenses drop to a configurable floor (e.g. 85% of base) when total net worth is below the starting watermark, and expand to a ceiling (e.g. 115% of base) when above.
    *   **Vanguard Dynamic Spending**: Annual net living expenses are recalculated each year as a target percentage of total portfolio net worth, bounded by a maximum annual cut (floor, e.g. -5.0%) and maximum annual raise (ceiling, e.g. +5.0%) relative to prior year's inflation-adjusted spending. The UI's Target Withdrawal Rate (TWR) represents total annual portfolio outflows (net living expenses + estimated taxes) and is bi-directionally synchronized with Annual Base Expenses using live user tax/yield inputs, while the underlying simulation engine scales net living expenses by `Annual Base Expenses / Starting Net Worth` so taxes are never double-counted.
*   **Income & Yield**:
    *   Expected **Monthly** Pillar 1 (AHV) Pension from age 65 (CHF).
    *   Expected Annual **Dividend Yield (%)** on equity holdings (US & Non-US Stocks).
*   **Monte Carlo & Simulation Parameters**:
    *   Nominal Mean Returns (%) and Volatilities (%) for asset classes, plus Inflation Mean (%) and Inflation Volatility (%) for Monte Carlo mode.
    *   Number of Monte Carlo Runs (`mc_num_runs`) and Number of Bootstrapping Runs (`boot_num_runs`).
    *   **Random Seed** (`random_seed`, default `42`) for reproducible Monte Carlo, Historic Bootstrapping, and synthetic Gold/Bitcoin simulation paths.


---

### 4.2. Emulation Engine & Calculation Logic

The core simulator executes monthly steps with annual tax and spending evaluations:

#### A. Income Tax (Zurich & Federal)
*   Calculate combined taxable income:
    *   Dividends from taxable equity investments (`(US Stocks + Non-US Stocks) * dividend_yield`).
    *   Interest from CHF Cash holdings (`CHF Cash * 1.0%` nominal APY).
    *   **Pillar 1 (AHV) Pension payments** received from age 65 onward (`12 * monthly_ahv_pension * inflation_factor`, fully taxable).
    *   *(Deferred to Phase 2: Pre-retirement Base + Bonus + Vesting GSUs and working-phase Pillar 2/3a contribution deductions).*
*   Apply the progressive Federal Income Tax rate (Single tariff).
*   Apply the progressive Zurich Cantonal and Municipal Income Tax rates (base rate multiplied by `cantonal_multiplier` (95% for 2026) + `municipal_multiplier` (e.g., 119% for Zurich City)).

#### B. Wealth Tax (Zurich)
*   Calculate taxable wealth:
    *   Year-end total value of taxable liquid investments (US Stocks, Non-US Stocks, CHF Cash, Gold, Bitcoin) *minus* the current year's living expenses (`max(0, total_liquid_end - current_expenses)`).
    *   *Note: Pillar 2 and Pillar 3a balances are exempt from wealth tax until withdrawal.*
*   Apply Zurich progressive wealth tax base rates multiplied by `cantonal_multiplier + municipal_multiplier`.

#### C. Swiss Pension System (The 3 Pillars)
*   **Pillar 1 (AHV)**:
    *   *(Deferred to Phase 2: Mandatory employee contributions during the pre-retirement working phase).*
    *   **Crucial (Phase 1)**: Calculate mandatory AHV contributions for *non-working* individuals post-retirement up to age 65 (`current_age < 65`). Based on `determining_wealth = taxable_wealth + 20 * imputed_pension_income` (min 530 CHF/year, max 26,500 CHF/year using official 2025 brackets).
    *   Model monthly payout starting at official retirement age (65). The user's input representing today's monthly pension value is adjusted for cumulative CPI inflation from Year 0 (including all years before age 65), added directly to CHF Cash each month, and taxed annually as income.
*   **Pillar 2**:
    *   Model monthly growth of the **Freizügigkeitskonto** (assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model **lump-sum withdrawal (Kapitalbezug)** at retirement age 65 (or immediately in Year 0 Month 0 if starting retirement at age $\ge 65$). Subject to immediate progressive capital withdrawal tax at source.
*   **Pillar 3a**:
    *   Model monthly growth (assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model staggered lump-sum withdrawals between age 60 and 64 (`60 + i` for up to 5 accounts, max 1 account per year). If starting at age $\ge 65$, all remaining accounts liquidate immediately in Year 0 Month 0. Subject to immediate progressive capital withdrawal tax at source.

#### D. Investment Growth & Returns (CHF-based)
The application supports three distinct return simulation engines utilizing empirical datasets curated by Baptiste Wicht (*The Poor Swiss*), available at [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator) (and [The Poor Swiss](https://thepoorswiss.com)):
*   **Empirical Datasets Ingested**:
    *   **US Stocks (USD)**: Robert Shiller monthly S&P 500 Total Returns (1871–2025).
    *   **Non-US Equities (USD)**: MSCI EAFE / World ex-US Total Returns proxy (1871–2025).
    *   **USD/CHF Exchange Rates**: Historical monthly FX rates (1913–2019 via *The Poor Swiss*, 2020–2025 via Swiss National Bank SNB).
    *   **Swiss Inflation (CPI)**: Historical Swiss Consumer Price Index (1921–2023 via *The Poor Swiss* / Swiss Federal Statistical Office FSO/BFS, 2024–2025 via FSO).
*   **Historic Backtesting Mode**: Replays contiguous historical **nominal** return and inflation sequences (1922–2025 in CHF, 104 years). Preserves historical sequence, macroeconomic autocorrelation, and historical Swiss inflation, generating $N = \text{total\_years} - \text{duration\_years} + 1$ overlapping cohorts, paired with exact 1% geometric APY for CHF Cash and Ito-corrected lognormal synthetic Gold/Bitcoin returns.
*   **Historic Bootstrapping (Sampling) Mode**: Jointly samples contiguous 5-year return and inflation blocks (`block_size_years = 5`) from the historical dataset with replacement across $N$ simulation iterations seeded by `random_seed`, paired with exact 1% geometric APY for CHF Cash and vectorized Ito-corrected lognormal synthetic Gold/Bitcoin returns. Using 5-year blocks preserves multi-year macroeconomic cycles (e.g., crashes and subsequent recoveries) and cross-asset/inflation correlations while stress-testing sequence-of-returns risk beyond contiguous records.
*   **Parametric Monte Carlo Mode**: Stochastic simulation seeded by `random_seed` using user-provided **nominal** asset class means ($\mu$) and standard deviations ($\sigma$) via lognormal returns (with Ito drift correction) and an independent normally distributed annual inflation stream (`seed + 10_000`).
*   Model inflation in CHF explicitly by increasing base retirement expenses and AHV pension payouts annually. This separates nominal asset growth from the rising cost of living.


---

### 4.3. Outputs & Visualizations

The tool presents results side-by-side across all three simulation modes (**Historic Backtesting**, **Historic Bootstrapping**, **Monte Carlo**):
*   **TL;DR Status Indicator**: A quick overarching assessment of the median outcome:
    *   **BROKE**: Median final net worth $\le 0$ (You run out of money).
    *   **RICH**: Median final net worth $\ge 3\times$ inflation-adjusted initial net worth (Real wealth grows massively).
    *   **DEAD**: Final net worth is positive but below $3\times$ the inflation-adjusted initial net worth (Safe, but real wealth depletes or grows modestly).
*   **Net Worth Trajectory Chart**: A chart showing the progression of assets over the simulation horizon.
    *   Plots individual runs (faint "spaghetti" lines for all contiguous cohorts in Historic Backtesting, capped at 100 runs for performant rendering in Bootstrapping and Monte Carlo) to show dispersion.
    *   Overlays clear percentile lines (**5th, 25th, 50th (median), 75th, and 95th** percentiles) and a dashed **Inflation-Adj Start NW** reference line.
*   **Income vs Required Cash Chart**: Dynamic annual stacked bar view of inflows (Dividends, AHV Pension post-65, Capital Sold) vs. a dashed reference line for Total Cash Needed (Expenses + Taxes).
*   **Annual Withdrawal Breakdown Chart**: Stacked bar chart showing median Living Expenses and median Taxes Paid over time.
*   **Withdrawal Rate Chart**: Percentile chart showing annual withdrawal rate over time (dynamically capped at 25% max for readability).
*   **Asset Allocation Development Chart**: Stacked area chart displaying the median nominal balance of all asset categories (CHF Cash, US Stocks, Non-US Stocks, Gold, Bitcoin, Pillar 3a, and Pillar 2) over time to visualize total net worth breakdown, glidepaths, and liquidations.
*   **Key Outflow KPI Metrics**:
    *   **Avg Years Below Start NW**: Average number and percentage of years where net worth falls below the inflation-adjusted starting net worth watermark.
    *   **Median Ending NW**: Real (inflation-adjusted) and Nominal ending net worth.
    *   **Median Total Withdrawals**: Total cumulative cash spent on living expenses and taxes over the entire retirement horizon (Real and Nominal).
    *   **Pre-AHV Outflow (< Age 65)**: Total median cash required to cover living expenses and taxes during the early retirement gap before age 65.
    *   **Post-65 Outflow (Age 65+)**: Total median cash required to cover living expenses and taxes from age 65 through end-of-life.
*   **Cohort / Run Analysis Tables**: Interactive data tables detailing the Top 10 Best and Worst individual simulation cohorts/runs, reporting `Cohort`/`Run`, `Final NW (Real)`, `Final NW (Nom)`, `Min NW (Nom)`, and `Yrs Below` (years below the starting watermark).
*   **Success Metric**:
    *   Allows configuring a target ending net worth as a percentage of inflation-adjusted starting net worth (default is 50.0%).
    *   Shows the calculated probability of success matching this definition across all simulation runs.


---

## 5. Key Swiss/Zurich Specific Rules to Model

| Rule | Description | Simulation Impact |
| :--- | :--- | :--- |
| **No Capital Gains Tax** | Capital gains on private assets (e.g., selling stocks) are 100% tax-free. | Only dividends/interest add to taxable income. |
| **Wealth Tax** | Canton Zurich taxes net wealth globally. | High net worth individuals pay significant wealth tax, which acts as a drag on portfolio growth during decumulation. |
| **AHV for Non-Workers** | Early retirees must pay AHV contributions based on their wealth and pension income. | Can cost up to ~26,500 CHF/year per person if assets are high. |
| **Pillar 2 Buy-ins** | Voluntary contributions to BVG are tax-deductible. | Excellent tax optimization strategy in high-earning years (Phase 2). |
| **Capital Withdrawal Tax** | Pillar 2 & 3a withdrawals are taxed at a separate, progressive rate, independent of normal income. | Staggering withdrawals over multiple years (different accounts) is required for optimization. |

---

## 6. Non-Functional Requirements

*   **Privacy & Data Security**: Financial data is highly sensitive. The tool runs locally without sending private financial data to an external server.
*   **Extensibility**: The simulation engine (`src/simulation_engine.py`, `src/tax_engine.py`, `src/historic_returns.py`) is decoupled from the Streamlit UI (`app.py`), allowing direct programmatic use and unit testing.
*   **Performance**: Vectorized NumPy operations allow Monte Carlo and Bootstrapping simulations (1,000–10,000 runs over 50 years) to execute in sub-second to a few seconds.

## 7. Architecture & Design Decisions Implemented
1.  **Architecture**: Built with Python, Streamlit, NumPy, Pandas, and Plotly (`app.py` + `src/`).
2.  **Tax Formula Accuracy**: Vectorized progressive bracket engine in `src/tax_engine.py` modeling Federal income tax, Zurich Cantonal/Municipal income & wealth taxes, capital withdrawal tax, and 2025 AHV non-worker contribution tables.
3.  **Simulation & Return Modeling**: Side-by-side execution of contiguous Historic Backtesting (`1922–2025`), joint Historic Bootstrapping, and Ito-corrected lognormal Monte Carlo simulation with configurable `Random Seed`.
