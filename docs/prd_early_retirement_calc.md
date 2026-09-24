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
    *   **Historic Backtesting**: Replaying contiguous historical market periods (1922–2025) using real monthly returns end-to-end, reporting **Effective Sample Size ($N_{\text{eff}} = T / D$)** and the **90% Wilson confidence interval**, and displaying **Worst Cohort (Min) / Best Cohort (Max)** bounds (`[0, 25, 50, 75, 100]`).
    *   **Historic Bootstrapping (Politis–Romano Stationary Bootstrap)**: Stochastic sampling of geometric-length blocks (`block_size_years = 5` default, restart probability $p = 1/b$) with **circular wrap-around** (`year 2025 -> year 1922`) across empirical returns, cash rates, and inflation with replacement across thousands of runs.
    *   **Parametric Monte Carlo**: Stochastic simulation using lognormal asset return distributions with cross-asset shock correlation (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$, `Bitcoin` $\rho=\text{btc\_equity\_corr}$, default `0.50`).
*   Zurich Cantonal, Municipal, and Federal Income & Wealth tax rules, including dynamic cash interest taxation based on realized CHF cash returns.
*   Mandatory AHV contributions for non-working early retirees.
*   Pillar 2 modeled via a **Freizügigkeitskonto (Vesting Account)**, restricted to **lump-sum withdrawal (Kapitalbezug)** at retirement (no annuities for now).
*   Pillar 3a lump-sum withdrawals.
*   Basic cost of living / expense modeling.
*   Configurable definition of "Success" for the simulation.
*   Advanced net worth trajectory chart showing all simulation paths, median, and key percentiles/extrema.

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
    *   Number of Pillar 3a Accounts (`0` to `5`) and a single **Balance per Pillar 3a Account (CHF)** input (all Pillar 3a accounts hold the same equal balance, constructing `[balance_per_account] * num_accounts` for staggered withdrawals).
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
*   **Smart Cash Buffer (Selling Strategy):** A defensive mechanism during market downturns. If the net worth drops below the inflation-adjusted starting net worth, the engine skips portfolio rebalancing and forces all expenses to be paid out of the CHF Cash allocation first, protecting equities from being sold at depressed prices. Normal proportional selling and rebalancing resumes once the portfolio recovers above the watermark. **Watermark definition (after tax):** all watermark decisions — Smart Cash Buffer, Dynamic Floor & Ceiling spending, and the "Years Below" metric — compare *after-tax* net worth against the inflation-adjusted *after-tax* starting net worth, where "after tax" deducts the capital withdrawal tax still owed on the Pillar 2 / 3a balances (estimated on current balances with the simulator's payout schedule: one 3a account per year from 60, the rest together with Pillar 2 at 65). Pension balances are pre-tax claims; comparing them gross made the realized withdrawal tax look like a market loss. Vanguard Dynamic spending still applies its rate to gross net worth, consistent with how the UI derives that rate from gross starting net worth. The success criterion is unchanged (gross starting net worth).

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
    *   Nominal Mean Returns (%) and Volatilities (%) for asset classes, plus Inflation Mean (%) and Inflation Volatility (%) for Monte Carlo mode. The section caption quotes the **data-derived** 1922–2025 CAGRs (US equities in USD and CHF, ex-US in CHF) computed from `src/historic_returns.py` at runtime rather than hardcoded figures. **Defaults (calibrated vs. assumed):** the equity volatility, gold mean and gold volatility defaults are **derived at runtime from the 1922–2025 CHF history** via `historic_lognormal_params` (arithmetic mean of annual simple returns and standard deviation of annual *log* returns — exactly the parameters `generate_monte_carlo_returns` consumes): equity vol ≈ 21.0% (average of US 22.2% and ex-US 19.9%), gold mean ≈ 4.7%, gold vol ≈ 15.3%. The stock means (US 7%, ex-US 6%), CHF cash (1%) and inflation (2.5% ± 1%) remain **deliberately conservative forward-looking assumptions** — far below the historic arithmetic means (US 11.3%, ex-US 8.6%, cash 2.5%, inflation 1.8% ± 3.6%). Every Monte Carlo input's help text shows its historic counterpart. Combined, realistic volatility plus conservative means make Monte Carlo materially more pessimistic than the historical engines (app default, Static spending: success 52.5% → 36.4% vs. the old 15% vol placeholders).
    *   **Bitcoin–Equity Correlation ($\rho$)** (`btc_equity_corr`, slider from `-0.50` to `0.90`, default `0.50`): Monthly log-return correlation between synthetic Bitcoin and US equities across all 3 simulation engines (Historic Backtesting, Stationary Bootstrapping, and Parametric Monte Carlo).
    *   **Stationary Bootstrap Mean Block Length (Years)** (`block_size_years`, slider from `1` to `15`, default `5`, Politis–White optimal length for annual macro returns).
    *   Number of Monte Carlo Runs (`mc_num_runs`) and Number of Bootstrapping Runs (`boot_num_runs`).
    *   **Random Seed** (`random_seed`, default `42`) for reproducible Monte Carlo, Historic Bootstrapping, and synthetic Bitcoin simulation paths.
*   **Currency / PPP Assumption**:
    *   **Real CHF Appreciation beyond PPP (%/yr)** (`real_chf_appreciation`, slider from `-1.00%` to `+2.00%`, step `0.05%`, default `0.00%`): Controls the expected long-run real appreciation of the Swiss Franc beyond Relative Purchasing Power Parity (inflation differentials) applied to foreign-priced sleeves (`US Stocks`, `Non-US Stocks`, `Gold`) in Historic Backtesting and Block Bootstrapping. `0.00%` (default) assumes Relative PPP holds going forward (removing the historical real FX drag); `+0.70%` (`HISTORIC_REAL_CHF_APPRECIATION`, derived from `usd_chf.csv`, `ch_inflation.csv` and `us_cpi.csv` at build time and quoted in the slider help) reproduces the raw unadjusted 1922–2025 historical CHF returns.
*   **Input Guards**: All monetary, percentage and rate inputs carry `min_value` bounds (e.g. balances/expenses ≥ 0, Vanguard floor ≤ 100%, MC means ≥ −99%) so that invalid values are rejected by the widget instead of crashing `SimConfig` validation.
*   **Tax Location & Assumptions**:
    *   Municipal Multiplier (Steuerfuss, %) — default `119.0` for Zurich City. The cantonal multiplier is fixed at `95%` (Canton Zurich 2026).
    *   **Tax Bracket Indexation (% of CPI)** (`bracket_indexation`, default `100`) — how much of each year's inflation flows through to the tax bracket edges and the AHV contribution table. See §4.2.B2.


---

### 4.2. Emulation Engine & Calculation Logic

The core simulator executes monthly steps with annual tax and spending evaluations:

#### A. Income Tax (Zurich & Federal)
*   Calculate combined taxable income:
    *   Dividends from taxable equity investments (`(US Stocks + Non-US Stocks) * dividend_yield`).
    *   Interest from CHF Cash holdings (`CHF Cash * max(0.0, realized_annual_cash_return)`, where `realized_annual_cash_return` is compounded from the actual 12 monthly returns of the CHF Cash sleeve in `return_matrix[:, (m - 11):(m + 1), 2]`, and `config.cash_rate` is used for Year 0 tax pre-estimates). This eliminates phantom taxable interest during `0%` ZIRP/NIRP years.
    *   **Pillar 1 (AHV) Pension payments** received from age 65 onward (`12 * monthly_ahv_pension * inflation_factor`, fully taxable).
    *   *(Deferred to Phase 2: Pre-retirement Base + Bonus + Vesting GSUs and working-phase Pillar 2/3a contribution deductions).*
*   Apply the progressive Federal Income Tax rate (Single tariff), capped at the constitutional maximum of **11.5% of taxable income** (Art. 128 BV): `min(tariff(x), 0.115 * x)`.
*   Apply the progressive Zurich Cantonal and Municipal Income Tax rates (base rate multiplied by `cantonal_multiplier` (95% for 2026) + `municipal_multiplier` (e.g., 119% for Zurich City)). These Zurich City values (`ZURICH_CANTONAL_MULTIPLIER = 0.95`, `ZURICH_CITY_MUNICIPAL_MULTIPLIER = 1.19`) are the shared defaults of the tax functions, `SimConfig`, and the UI. *Note: all base tariffs (federal Art. 36 DBG, Zurich § 35 income and § 47 wealth) are the **2026** tables, reverse-engineered from and unit-tested against the official ESTV tax calculator for Zurich City (income, wealth and capital-withdrawal taxes match to within CHF 1–2; the calculator's CHF 100/1,000 rounding, the CHF 25 federal minimum and the CHF 24 Zurich personal tax are deliberately not modelled).*

#### B. Wealth Tax (Zurich)
*   Calculate taxable wealth:
    *   Year-end total value of taxable liquid investments (US Stocks, Non-US Stocks, CHF Cash, Gold, Bitcoin) *minus* the current year's living expenses (`max(0, total_liquid_end - current_expenses)`).
    *   *Note: Pillar 2 and Pillar 3a balances are exempt from wealth tax until withdrawal.*
*   Apply Zurich progressive wealth tax base rates multiplied by `cantonal_multiplier + municipal_multiplier`.

#### B2. Tax Bracket Indexation ("Kalte Progression")
*   All franc-denominated bracket edges — Federal income, Zurich income, Zurich wealth, the derived capital withdrawal tariff, and the AHV non-worker contribution table — are indexed to realized CPI each year, matching Swiss law: **Art. 39 DBG** obliges the EFD to adjust the federal tariff annually to the LIK, and **§ 48 StG ZH** indexes the Canton Zurich income *and* wealth tariffs.
*   Controlled by **Tax Bracket Indexation (% of CPI)** (`bracket_indexation`, default **100%**). Setting it to 0% freezes brackets in nominal terms so inflation alone pushes the retiree into higher brackets; intermediate values model delayed or partially suspended cantonal compensation.
*   Tax *rates* and the Steuerfuss multipliers are never indexed — both are political parameters, not inflation-linked ones.
*   *Note: Indexation removes only the spurious, inflation-driven bracket creep. A portfolio compounding faster than inflation still climbs the wealth tax schedule in real terms, which is intended.*

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
    *   Model staggered lump-sum withdrawals between age 60 and 64 (`60 + i` for up to 5 equal-balance accounts configured via account count and balance per account, max 1 account per year). If starting at age $\ge 65$, all remaining accounts liquidate immediately in Year 0 Month 0. Subject to immediate progressive capital withdrawal tax at source.
*   **Capital Withdrawal Tax (Kapitalleistungen aus Vorsorge)**, levied separately from income tax on each lump sum $C$:
    *   **Federal** (Art. 38 DBG): 1/5 of the ordinary federal income tariff, $T_{\text{fed}}(C)/5$ (including the 11.5% cap).
    *   **Zurich** (§ 37 StG ZH, in force since 2022): the simple tax is computed at the rate that would apply to a notional annual pension of $C/20$, applied to the full amount, with a **2% minimum**: $\text{simple} = \max\left(0.02\,C,\; 20 \cdot T_{\text{ZH}}(C/20)\right)$, then multiplied by `cantonal_multiplier + municipal_multiplier`.
    *   Resulting all-in rate in Zurich City: ≈ 4.8% on CHF 100k, ≈ 10.9% on CHF 1M. Brackets are CPI-indexed like all other tariffs (§4.2.B2).

#### D. Investment Growth & Returns (CHF-based)
The application uses a two-stage reproducible data pipeline (`scripts/fetch_source_data.py` and `scripts/build_historic_returns.py`) generating `src/historic_returns.py` across 1922–2025 (`104` calendar years, `1,248` monthly steps):
*   **Empirical Datasets Ingested (`data/`)**:
    *   **US Stocks (`data/us_stocks.csv`)**: True **month-end** S&P 500 Total Return index in USD (dividends reinvested, 1871–2025), verified to $\pm 0.01\text{pp}$ against published month-end S&P 500 total returns (via [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator); previously misattributed to Shiller's monthly-averaged price series). Carried as **real monthly returns** throughout 1922–2025.
    *   **Non-US Equities (`data/jst_exus_usd.csv` + `data/ex_us_stocks.csv`)**: Spliced series. Because `ex_us_stocks.csv` is byte-identical to `us_stocks.csv` prior to MSCI EAFE's Dec 1969 inception, **1922–1969** uses a GDP-weighted 17-country ex-US equity total return index built from the **Jordà–Schularick–Taylor Macrohistory Database R6** (`eq_tr` converted to USD via `xrusd`, then to CHF; annual observations expanded into 12 geometric monthly steps, tracked in `MONTHLY_EXUS_IS_SMOOTHED`), spliced with **real monthly** MSCI EAFE / World ex-US returns from `data/ex_us_stocks.csv` for **1970–2025**.
    *   **USD/CHF Exchange Rates (`data/usd_chf.csv`)**: True **month-end** spot rates for **1971–present** built from daily **FRED `DEXSZUS`** observations (fixing a 13.4% 2024–2025 tail error in the legacy hand-extended series), spliced with **SNB `devkum`** monthly averages for **1914–1970** (the gold-standard / Bretton Woods peg era).
    *   **Swiss Inflation (`data/ch_inflation.csv`)**: Historical monthly Swiss Consumer Price Index (CPI / LIK) from the Swiss Federal Statistical Office (FSO/BFS, 1921–present).
    *   **Gold (`data/gold_usd.csv`)**: **LBMA London PM Fix** month-end USD/oz prices from **April 1968–present** multiplied by month-end `USD/CHF` to yield **real monthly CHF gold returns** (`MONTHLY_GOLD_CHF`). Pre-1968 statutory fixed gold prices (`$20.67/oz` then `$35.00/oz`) reflect official currency devaluations spread evenly across the year (`MONTHLY_GOLD_IS_SMOOTHED`).
    *   **Swiss Short-Term Cash Rate (`data/ch_cash_rate.csv`)**: Annual Swiss short-term interest rates (`1900–2025`) built from the **Jordà–Schularick–Taylor Macrohistory Database R6** (`bill_rate` for Switzerland, 1900–2020) spliced with **SNB policy / SARON rates** (2021–2025), floored at `0.0%` (`retail_rate`) for retail deposit accounts (`MONTHLY_CASH_CHF`, `HISTORIC_RETURNS_CASH_CHF`, 1922–2025: `2.46%` nominal CAGR, `1.88%` vol, `+0.71%` real after `1.74%` Swiss CPI). This wholesale-based series is an optimistic proxy for actual retail savings rates.
    *   **US Inflation (`data/us_cpi.csv`)**: Monthly US CPI-U (1871–2025, via swr-calculator). Build-time only: used to derive `HISTORIC_REAL_CHF_APPRECIATION` $= 1 - \frac{1 + g_{\text{FX}}}{1 + g_{\text{PPP}}}$ (geometric 1922–2025 drifts, currently `0.0070`) instead of a hand-maintained constant.
    *   **Data integrity**: The build script asserts that the Gold series has no gaps inside the simulation window rather than silently filling them with `0.0` returns.
*   **Historic Backtesting Mode**: Replays contiguous historical **nominal** monthly return (`MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`) and annual Swiss CPI inflation sequences (1922–2025 in CHF, 104 years), adjusted by the user's `real_chf_appreciation` PPP setting (`0.00%` default). Generates $N = \text{total\_years} - \text{duration\_years} + 1$ overlapping cohorts, paired with Ito-corrected lognormal synthetic Bitcoin returns (`btc_mean`, default `7%`, and `btc_vol`, default `50%`, shared with the sidebar inputs) coupled to per-path standardized US equity monthly log-returns via a Gaussian copula at correlation `btc_equity_corr` (default `0.50`).
*   **Historic Bootstrapping (Politis–Romano Stationary Bootstrap) Mode**: Jointly samples historical years (`MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`, and `HISTORIC_SWISS_INFLATION`) using the **Politis–Romano (1994) Stationary Bootstrap** (`stationary=True`, geometric block lengths of mean `block_size_years = 5` via restart probability $p = 1/b$) with **circular wrap-around** `(year_idx[:, t - 1] + 1) % total_years`. This eliminates endpoint under-sampling (every year from `1922` to `2025` has uniform $1/104$ selection probability) and fixed 5-year seam artifacts while preserving multi-year stagflation and crash/recovery clusters. Paired with Gaussian-copula Bitcoin returns (`btc_mean`, `btc_vol`, `btc_equity_corr`) coupled to the per-path standardized bootstrapped US equity path.
*   **Parametric Monte Carlo Mode**: Stochastic simulation seeded by `random_seed` using user-provided **nominal** asset class means ($\mu$) and standard deviations ($\sigma$) via lognormal returns (with Ito drift correction), cross-asset shock coupling (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$, `Bitcoin` $\rho=\text{btc\_equity\_corr}$), and an independent normally distributed annual inflation stream (`seed + 10_000`).
*   Model inflation in CHF explicitly by increasing base retirement expenses and AHV pension payouts annually. This separates nominal asset growth from the rising cost of living.


---

### 4.3. Outputs & Visualizations

The tool presents results side-by-side across all three simulation modes (**Historic Backtesting**, **Historic Bootstrapping**, **Monte Carlo**):
*   **TL;DR Status Indicator**: A quick overarching assessment of the median outcome, based on the **median of per-run real final net worth** (each run's nominal final net worth deflated by that run's own cumulative inflation, `src/metrics.py::classify_outcome`). Comparing the nominal median against a multiple of the *median price level* would pair values from different runs and can mislabel outcomes.
    *   **BROKE**: Median real final net worth $\le 0$ (You run out of money).
    *   **RICH**: Median real final net worth $\ge 3\times$ initial net worth (`RICH_MULTIPLE = 3.0`; real wealth grows massively).
    *   **DEAD**: Median real final net worth is positive but below $3\times$ the initial net worth (Safe, but real wealth depletes or grows modestly).
*   **Net Worth Trajectory Chart**: A chart showing the progression of assets over the simulation horizon.
    *   Plots individual runs (faint "spaghetti" lines for all contiguous cohorts in Historic Backtesting, capped at 100 runs for performant rendering in Bootstrapping and Monte Carlo) to show dispersion.
    *   For **Historic Backtesting**, overlays **Worst Cohort (Min), 25th, 50th (median), 75th, and Best Cohort (Max)** (`[0, 25, 50, 75, 100]`) rather than 5th/95th percentiles that cannot be non-parametrically estimated at $N_{\text{eff}} \approx 2.1\text{–}2.6$. For **Historic Bootstrapping** and **Monte Carlo**, overlays **5th, 25th, 50th (median), 75th, and 95th** percentiles (`[5, 25, 50, 75, 95]`). Both include a dashed **Inflation-Adj Start NW** reference line.
*   **Income vs Required Cash Chart**: Dynamic annual stacked bar view of inflows (Dividends, AHV Pension post-65, Capital Sold) vs. a dashed reference line for Total Cash Needed (Expenses + Taxes).
*   **Annual Withdrawal Breakdown Chart**: Stacked bar chart showing median Living Expenses and median Taxes Paid over time.
*   **Withdrawal Rate Chart**: Percentile/extrema chart showing the annual withdrawal rate over time, computed as `(Living Expenses + Taxes) / Beginning-of-Year Net Worth`, where the beginning-of-year net worth is reconstructed as end-of-year net worth plus that year's outflows (dynamically capped at 25% max for readability).
*   **Asset Allocation Development Chart**: Stacked area chart displaying the median nominal balance of all asset categories (CHF Cash, US Stocks, Non-US Stocks, Gold, Bitcoin, Pillar 3a, and Pillar 2) over time to visualize total net worth breakdown, glidepaths, and liquidations.
*   **Key Outflow KPI Metrics**:
    *   **Avg Years Below Start NW**: Average number and percentage of years where net worth falls below the inflation-adjusted starting net worth watermark.
    *   **Median Ending NW**: Real (inflation-adjusted) and Nominal ending net worth.
    *   **Median Total Withdrawals**: Total cumulative cash spent on living expenses and taxes over the entire retirement horizon (Real and Nominal).
    *   **Pre-AHV Outflow (< Age 65)**: Total median cash required to cover living expenses and taxes during the early retirement gap before age 65.
    *   **Post-65 Outflow (Age 65+)**: Total median cash required to cover living expenses and taxes from age 65 through end-of-life.
*   **Cohort / Run Analysis Tables**: Interactive data tables detailing the Top 10 Best and Worst individual simulation cohorts/runs, reporting `Cohort`/`Run`, `Final NW (Real)`, `Final NW (Nom)`, `Min NW (Nom)`, and `Yrs Below` (years below the starting watermark).
*   **Success Metric & Effective Sample Size ($N_{\text{eff}}$)**:
    *   Allows configuring a target ending net worth as a percentage of inflation-adjusted starting net worth (default is 50.0%).
    *   Shows the calculated probability of success matching this definition across all simulation runs.
    *   In **Historic Backtesting**, explicitly surfaces the number of overlapping cohorts ($N_{\text{cohorts}} = T - D + 1$), the **Effective Sample Size ($N_{\text{eff}} = T / D$)** (e.g. `2.6` for a 40-year horizon or `2.1` for a 50-year horizon across 104 years), and the **90% two-sided Wilson confidence interval** evaluated at $N_{\text{eff}}$ (with matching vertical space reserved in **Historic Bootstrapping** and **Monte Carlo** so all three columns remain vertically aligned).


---

## 5. Key Swiss/Zurich Specific Rules to Model

| Rule | Description | Simulation Impact |
| :--- | :--- | :--- |
| **No Capital Gains Tax** | Capital gains on private assets (e.g., selling stocks) are 100% tax-free. | Only dividends/interest add to taxable income. |
| **Wealth Tax** | Canton Zurich taxes net wealth globally. | High net worth individuals pay significant wealth tax, which acts as a drag on portfolio growth during decumulation. |
| **AHV for Non-Workers** | Early retirees must pay AHV contributions based on their wealth and pension income. | Can cost up to ~26,500 CHF/year per person if assets are high. |
| **Pillar 2 Buy-ins** | Voluntary contributions to BVG are tax-deductible. | Excellent tax optimization strategy in high-earning years (Phase 2). |
| **Capital Withdrawal Tax** | Pillar 2 & 3a withdrawals are taxed separately from normal income: federal at 1/5 of the tariff (Art. 38 DBG); Zurich at the rate for a notional pension of 1/20 of the lump sum, minimum 2% simple tax (§ 37 StG ZH). | ≈ 4.8% (CHF 100k) to ≈ 10.9% (CHF 1M) all-in in Zurich City; staggering withdrawals over multiple years (different accounts) lowers the rate. |
| **Bracket Indexation** | Art. 39 DBG and § 48 StG ZH require federal and Zurich income/wealth bracket edges to be indexed to the CPI. | Over a 50-year horizon this is material: freezing brackets instead overstates median lifetime real taxes by roughly a third. Modelled at 100% by default, adjustable. |

---

## 6. Non-Functional Requirements

*   **Privacy & Data Security**: Financial data is highly sensitive. The tool runs locally without sending private financial data to an external server.
*   **Extensibility**: The simulation engine is a self-contained importable package (`src/`) decoupled from the Streamlit UI (`app.py`). From the repository root it supports direct programmatic use and unit testing with no `sys.path` manipulation — `from src import SimConfig, run_simulation`, or per-module (`from src.tax_engine import calculate_income_tax`).
*   **Performance**: Vectorized NumPy operations allow Monte Carlo and Bootstrapping simulations (1,000–10,000 runs over 50 years) to execute in sub-second to a few seconds.

## 7. Architecture & Design Decisions Implemented
1.  **Architecture**: Built with Python, Streamlit, NumPy, Pandas, and Plotly (`app.py` + `src/`).
2.  **Tax Formula Accuracy**: Vectorized progressive bracket engine in `src/tax_engine.py` modeling Federal income tax, Zurich Cantonal/Municipal income & wealth taxes, capital withdrawal tax, and 2025 AHV non-worker contribution tables.
3.  **Simulation & Return Modeling**: Side-by-side execution of contiguous Historic Backtesting (`1922–2025`), joint Historic Bootstrapping, and Ito-corrected lognormal Monte Carlo simulation with configurable `Random Seed`.
4.  **Result Metrics**: Success criterion, real ending net worth, TL;DR classification, and withdrawal-rate reconstruction live in `src/metrics.py` (pure, unit-tested functions shared by all UI panels) instead of being re-implemented inline in `app.py`.
5.  **Testing**: `pytest` suite covering the tax engine, simulation engine, metrics, and Streamlit `AppTest` smoke tests of `app.py` (default run, spending strategies, allocation validation, PPP slider).
