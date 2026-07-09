# Product Requirement Document (PRD): Swiss Early Retirement Financial Simulator (Zurich)

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
*   Modeling return scenarios using **both Monte Carlo and Historic Returns**, calculated in **CHF**.
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
    *   **Never**: Assets drift naturally.
    *   **Yearly**: Rebalanced to target weights once every 12 months.
    *   **Monthly**: Rebalanced to target weights every month.
    *   **Threshold-based**: Rebalanced only if any asset's weight drifts from its target by a user-specified percentage (e.g., ±5%).
*   **Smart Cash Buffer (Selling Strategy):** A defensive mechanism during market downturns. If the net worth drops below the inflation-adjusted starting net worth, the engine skips portfolio rebalancing and forces all expenses to be paid out of the CHF Cash allocation first, protecting equities from being sold at depressed prices. Normal proportional selling and rebalancing resumes once the portfolio recovers above the watermark.

#### D. Expenses (Post-Retirement)
*   Projected annual base retirement expenses (post-retirement) in CHF.
*   Expected **Monthly** Pillar 1 (AHV) Pension from age 65 (CHF).
*   One-off future expenses (e.g., buying a car, world trip) in CHF with specified target years.


---

### 4.2. Emulation Engine & Calculation Logic

The core simulator must run annual cycles (ticks) and compute the following:

#### A. Income Tax (Zurich & Federal)
*   Calculate combined taxable income:
    *   Base + Bonus + Vesting GSUs (treated as income upon vest in Switzerland).
    *   Dividends from taxable investments (assumed yield, e.g., 2%).
    *   **Pillar 1 (AHV) Pension payments** (fully taxable).
    *   Interest.
    *   *Minus* deductions: Pillar 3a contributions, Pillar 2 standard contributions, Pillar 2 voluntary buy-ins, professional expenses.
*   Apply the progressive Federal Income Tax rate.
*   Apply the progressive Zurich Cantonal and Municipal Income Tax rates (multiplied by the municipality's specific tax multiplier, e.g., 119% for Zurich city).

#### B. Wealth Tax (Zurich)
*   Calculate taxable wealth:
    *   Total value of taxable brokerage accounts.
    *   Pillar 3b and cash.
    *   *Note: Pillar 2 and Pillar 3a balances are exempt from wealth tax until withdrawal.*
*   Apply Zurich progressive wealth tax rates.

#### C. Swiss Pension System (The 3 Pillars)
*   **Pillar 1 (AHV)**:
    *   Calculate mandatory contributions during the working phase (approx. 5.3% employee share).
    *   **Crucial**: Calculate mandatory AHV contributions for *non-working* individuals post-retirement up to age 65. This is based on wealth and imputed pension income (can be significant for early retirees).
    *   Model payout starting at official retirement age (currently 65 for both men and women). The user's input representing today's pension value is adjusted for cumulative inflation from the start of the simulation until the payout starts at 65. This payout acts as an income stream that offsets living expenses and is fully subject to income tax (adjusted annually for inflation thereafter).
*   **Pillar 2**:
    *   Model monthly growth of the **Freizügigkeitskonto** (Assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model **lump-sum withdrawal (Kapitalbezug)** at retirement or staggered up to 5 years after AHV retirement age. Subject to separate capital withdrawal tax.
*   **Pillar 3a**:
    *   Model monthly growth (Assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model staggered lump-sum withdrawals between age 60 and 65 (up to 5 accounts can be held to stagger tax brackets). Apply capital withdrawal tax.

#### D. Investment Growth & Returns (CHF-based)
*   **Historic Returns Mode**: Simulates the portfolio using actual historical **nominal** returns of CHF-denominated or CHF-hedged asset classes.
*   **Monte Carlo Mode**: Stochastic simulation using historical averages, volatilities, and correlations. Inputs are strictly **nominal**.
*   Model inflation in CHF explicitly by increasing base retirement expenses and AHV pension payouts annually. This separates nominal asset growth from the rising cost of living.


---

### 4.3. Outputs & Visualizations

The tool must present the user with:
*   **TL;DR Status Indicator**: A quick overarching assessment of the median outcome:
    *   **BROKE**: Median final net worth <= 0 (You run out of money).
    *   **RICH**: Median final net worth >= 3x inflation-adjusted initial net worth (Real wealth grows massively).
    *   **DEAD**: Final net worth is positive but below 3x the inflation-adjusted initial net worth (Safe, but real wealth depletes or stagnates).
*   **Net Worth Trajectory Chart**: A chart showing the progression of assets over the 50-year horizon.
    *   For multi-run simulations (Monte Carlo/Historic roll period), it must plot **all individual runs** (faint "spaghetti" lines) to show dispersion (capped at 100 runs for Monte Carlo for performance).
    *   It must overlay clear percentile lines: **5th, 25th, 50th (median), 75th, and 95th** percentiles.
*   **Income vs Required Cash Chart**: Dynamic annual view of inflows (Dividends, AHV annuities post-65, Capital Sold) vs. outflows (Total Cash Needed: Expenses + taxes).
*   **Expenses & Taxes Chart**: Stacked bar chart showing median expenses and median taxes paid over time.
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
| **Pillar 2 Buy-ins** | Voluntary contributions to BVG are tax-deductible. | Excellent tax optimization strategy in high-earning years. |
| **Capital Withdrawal Tax** | Pillar 2 & 3a withdrawals are taxed at a separate, progressive rate, independent of normal income. | Staggering withdrawals over multiple years (different accounts) is required for optimization. |

---

## 6. Non-Functional Requirements

*   **Privacy & Data Security**: Financial data is highly sensitive. The tool should ideally run entirely in the browser (client-side) or local host, without sending private financial data to a backend server.
*   **Extensibility**: The simulation engine should be decoupled from the UI, allowing it to be run as a CLI or library for scripting.
*   **Performance**: Monte Carlo simulations (10,000 runs over 50 years) should complete in under 5 seconds.

## 7. Next Steps & Design Doc Focus Areas
1.  **Architecture**: Decide between a TypeScript React client-side app or a Python backend with a simple frontend (Streamlit/FastAPI).
2.  **Tax Formula Accuracy**: How to approximate the complex progressive tax curves of Zurich and Federal taxes without requiring a heavy external database.
3.  **Monte Carlo Implementation**: Choosing the right statistical modeling library.
