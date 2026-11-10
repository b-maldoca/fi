# Product Requirement Document (PRD): Early Retirement Simulator (Zurich · Austria)

## 1. Executive Summary
The goal of this project is to build a financial simulation tool designed for individuals based in Switzerland who wish to plan for early retirement (FIRE - Financial Independence, Retire Early). 

The tool will simulate the user's financial life over a configurable period (e.g., 50 years), taking into account the unique aspects of Swiss taxation, the Swiss three-pillar pension system, and the cost of living in Canton Zurich. It will help users determine their readiness for retirement, optimize their withdrawal strategies, and visualize their net worth trajectory.

> **Jurisdiction status:** 🇨🇭 **Switzerland (Canton Zurich)** — ✅ implemented (§1–§7). 🇦🇹 **Austria** — ✅ implemented (§8: engine, EUR data and UI; select *Austria* in the sidebar **Country** box); pension taxation verified against the official BMF Brutto-Netto-Rechner 2026 (13 pensions from €1,100 to €95,000 a month: health contribution, income tax and net pension within €0.13). Both countries live in **one app with a country switch**.

## 2. Target Audience & User Persona
*   **Role**: High-earning professional based in Zurich.
*   **Residency/Legal Status**: C Permit holder or Swiss Citizen (meaning they file standard tax returns and are not taxed at source).
*   **Aspiration**: Retire early (e.g., age 40–55) and remain in Canton Zurich (or nearby municipalities) for the next 50+ years.
*   **Financial Profile**: High income, significant portion of compensation in RSUs/GSUs, active contributor to Pillar 2 (Pensionskasse) and Pillar 3a, taxable brokerage accounts.

## 3. Scope & Phasing of the Emulation
To manage complexity, the project will be split into two development phases:
*   **Phase 1 (Current Focus): Post-Retirement (Decumulation Phase) Only**. The simulation starts on the day of retirement with a user-specified initial portfolio (including tax-sheltered Pillar 2 and Pillar 3a equity accounts locked until their statutory withdrawal ages, and taxable brokerage). The engine simulates decumulation, taxes, investment returns, and mandatory AHV contributions for non-workers until the end of the simulation.
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
*   Other Swiss cantons and jurisdictions beyond Switzerland (Canton Zurich) and Austria (see §8).


---

## 4. Functional Requirements

### 4.1. Data Inputs

The application must allow the user to input the following parameters:

#### A. Demographics & Timeline
*   Retirement Age (default `40` — representing current/retirement age in Phase 1)
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
    *   **Dynamic Spending (Floor & Ceiling)**: Expenses drop to a configurable floor (e.g. 85% of base) when after-tax net worth is below the starting watermark, and expand to a ceiling (e.g. 115% of base, enforced $\ge$ floor) when above.
    *   **Vanguard Dynamic Spending**: Annual net living expenses are recalculated each year as a target percentage of total portfolio net worth, bounded by a maximum annual cut (floor, e.g. -5.0%) and maximum annual raise (ceiling, e.g. +5.0%) relative to prior year's inflation-adjusted spending. The UI's Target Withdrawal Rate (TWR) represents total annual portfolio outflows (net living expenses + estimated taxes) and is bi-directionally synchronized with Annual Base Expenses using live user tax/yield/Cash-Tent inputs, while the underlying simulation engine scales net living expenses by `Annual Base Expenses / Starting Net Worth` so taxes are never double-counted.
*   **Income & Yield**:
    *   Expected **Monthly** Pillar 1 (AHV) Pension from age 65 (CHF).
    *   Expected Annual **Dividend Yield (%)** on equity holdings (US & Non-US Stocks).
*   **Monte Carlo & Simulation Parameters**:
    *   Nominal Mean Returns (%) and Volatilities (%) for asset classes, plus Inflation Mean (%) and Inflation Volatility (%) for Monte Carlo mode. The section subheader tooltip (`help`) quotes the **data-derived** 1922–2025 CAGRs (US equities in USD and CHF, ex-US in CHF) computed from `src/historic_returns.py` at runtime rather than hardcoded figures. **Defaults (calibrated vs. assumed):** the US and ex-US stock means, equity volatility, gold mean, gold volatility, inflation mean and inflation volatility defaults are **derived at runtime from the 1922–2025 CHF history** (`historic_lognormal_params` for lognormal assets — arithmetic mean of annual simple returns and standard deviation of annual *log* returns — and sample mean/std for `HISTORIC_SWISS_INFLATION`). **The foreign-priced means (US, Non-US, Gold) are calibrated on the history rescaled to the sidebar's Currency / PPP Assumption** (`ppp_adjusted_annual_returns`), so that Monte Carlo embeds the same FX drift as the Historic Backtesting and Bootstrapping engines at any slider setting (while preserving any return mean manually customized by the user across subsequent PPP slider adjustments): at the default `0.00%` (PPP neutrality) US mean ≈ 12.1%, ex-US mean ≈ 9.3%, gold mean ≈ 5.5%; at `+0.71%` (raw history) US ≈ 11.3%, ex-US ≈ 8.6%, gold ≈ 4.7%. Log-volatility is invariant to that rescaling: equity vol ≈ 21.0% (average of US 22.2% and ex-US 19.9%), gold vol ≈ 15.3%, inflation mean ≈ 1.8%, inflation volatility ≈ 3.6%. Only CHF cash (1%) remains a **forward-looking assumption** (today's near-zero savings rates and Year 0 taxable interest; the historic 2.5% cash mean is dominated by earlier regimes). Every Monte Carlo input's help text shows its historic counterpart (and, for the foreign sleeves, the PPP assumption it was computed under). At app defaults (`1,000` runs, `seed=42`, `PPP=0.00%`), Monte Carlo success is `86.0%` under Static spending (`81.6%` at raw `+0.71%` PPP) and `95.3%` under Vanguard Dynamic (`92.9%` at raw `+0.71%` PPP).
    *   **Bitcoin–Equity Correlation ($\rho$)** (`btc_equity_corr`, slider from `-0.50` to `0.90`, default `0.50`): Monthly log-return correlation between synthetic Bitcoin and US equities across all 3 simulation engines (Historic Backtesting, Stationary Bootstrapping, and Parametric Monte Carlo).
    *   **Stationary Bootstrap Mean Block Length (Years)** (`block_size_years`, slider from `1` to `15`, default `5`, Politis–White optimal length for annual macro returns).
    *   Number of Monte Carlo Runs (`mc_num_runs`) and Number of Bootstrapping Runs (`boot_num_runs`).
    *   **Random Seed** (`random_seed`, default `42`) for reproducible Monte Carlo, Historic Bootstrapping, and synthetic Bitcoin simulation paths.
*   **Currency / PPP Assumption**:
    *   **Real CHF Appreciation beyond PPP (%/yr)** (`real_chf_appreciation`, slider from `-1.00%` to `+2.00%`, step `0.01%` (so the data-derived historic value is always selectable), default `0.00%`): Controls the expected long-run real appreciation of the Swiss Franc beyond Relative Purchasing Power Parity (inflation differentials) applied to foreign-priced sleeves (`US Stocks`, `Non-US Stocks`, `Gold`) in Historic Backtesting and Block Bootstrapping, **and used to calibrate the Monte Carlo default means for the same sleeves** (`ppp_adjusted_annual_returns`), which is why the section is rendered *before* **Monte Carlo Parameters** in the sidebar. `0.00%` (default) assumes Relative PPP holds going forward (removing the historical real FX drag); `+0.71%` (`HISTORIC_REAL_CHF_APPRECIATION`, derived from `usd_chf.csv`, `ch_inflation.csv` and `us_cpi.csv` at build time and quoted in the slider help) reproduces the raw unadjusted 1922–2025 historical CHF returns.
*   **Input Guards**: All monetary, percentage and rate inputs carry `min_value` bounds (e.g. balances/expenses ≥ 0, Vanguard floor ≤ 100%, MC means ≥ −99%) so that invalid values are rejected by the widget instead of crashing `SimConfig` validation.
*   **Tax Location & Assumptions**:
    *   Municipal Multiplier (Steuerfuss, %) — default `119.0` for Zurich City. The cantonal multiplier is fixed at `95%` (Canton Zurich 2026).
    *   **Tax Bracket Indexation (% of CPI)** (`bracket_indexation`, default `100`) — how much of each year's inflation flows through to the tax bracket edges and the AHV contribution table. See §4.2.B2.


---

### 4.2. Emulation Engine & Calculation Logic

The core simulator executes monthly steps with annual tax and spending evaluations:

#### A. Income Tax (Zurich & Federal)
*   Calculate combined taxable income (`max(0, dividends + interest + annual_ahv_received - ahv_contrib)`), assessed on the asset balances held during the year *prior* to the Month-11 year-end rebalance:
    *   Dividends from taxable equity investments (`(US Stocks + Non-US Stocks) * dividend_yield`).
    *   Interest from CHF Cash holdings (`CHF Cash * max(0.0, realized_annual_cash_return)`, where `realized_annual_cash_return` is compounded from the actual 12 monthly returns of the CHF Cash sleeve in `return_matrix[:, (m - 11):(m + 1), 2]`, and `config.cash_rate` — applied to Year-0 Cash Tent weights when `Cash Tent` is active — is used for Year 0 tax pre-estimates). This eliminates phantom taxable interest during `0%` ZIRP/NIRP years.
    *   **Pillar 1 (AHV) Pension payments** received from age 65 onward (`12 * monthly_ahv_pension * inflation_factor`, fully taxable).
    *   **Less mandatory non-worker AHV/IV/EO contributions (`ahv_contrib`)** paid before age 65, which are fully deductible from taxable income under **Art. 33 Abs. 1 lit. d DBG** (federal) and **§ 31 Abs. 1 lit. d StG ZH** (Canton Zurich).
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
    *   **Crucial (Phase 1)**: Calculate mandatory AHV contributions for *non-working* individuals post-retirement up to age 65 (`current_age < 65`). Based on `determining_wealth = taxable_wealth + 20 * imputed_pension_income` (min 530 CHF/year, max 26,500 CHF/year using official 2025/2026 brackets), and deducted from taxable income (Art. 33 Abs. 1 lit. d DBG / § 31 Abs. 1 lit. d StG ZH).
    *   Model monthly payout starting at official retirement age (65). The user's input representing today's monthly pension value is adjusted for cumulative CPI inflation from Year 0 (including all years before age 65), added directly to CHF Cash each month, and taxed annually as income.
*   **Pillar 2**:
    *   Model monthly growth of the **Freizügigkeitskonto** (assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model **lump-sum withdrawal (Kapitalbezug)** at retirement age 65 (or immediately in Year 0 Month 0 if starting retirement at age $\ge 65$). Subject to immediate progressive capital withdrawal tax at source (evaluated at start-of-year tax brackets, with net proceeds repaying any negative CHF Cash bridge debt first before allocating surplus across target weights).
*   **Pillar 3a**:
    *   Model monthly growth (assumed to be 100% invested in equities proportional to target US/Non-US allocation).
    *   Model staggered lump-sum withdrawals between age 60 and 64 (`60 + i` for up to 5 equal-balance accounts configured via account count and balance per account, max 1 account per year). If starting at age $\ge 65$, all remaining accounts liquidate immediately in Year 0 Month 0. Subject to immediate progressive capital withdrawal tax at source (repaying any negative CHF Cash debt first before allocating surplus).
*   **Capital Withdrawal Tax (Kapitalleistungen aus Vorsorge)**, levied separately from income tax on each lump sum $C$:
    *   **Federal** (Art. 38 DBG): 1/5 of the ordinary federal income tariff, $T_{\text{fed}}(C)/5$ (including the 11.5% cap).
    *   **Zurich** (§ 37 StG ZH, in force since 2022): the simple tax is computed at the rate that would apply to a notional annual pension of $C/20$, applied to the full amount, with a **2% minimum**: $\text{simple} = \max\left(0.02\,C,\; 20 \cdot T_{\text{ZH}}(C/20)\right)$, then multiplied by `cantonal_multiplier + municipal_multiplier`.
    *   Resulting all-in rate in Zurich City: ≈ 4.8% on CHF 100k, ≈ 10.9% on CHF 1M. Brackets are CPI-indexed like all other tariffs (§4.2.B2).

#### D. Investment Growth & Returns (CHF-based)
The application uses a two-stage reproducible data pipeline (`scripts/fetch_source_data.py` and `scripts/build_historic_returns.py`) generating `src/historic_returns.py` across 1922–2025 (`104` calendar years, `1,248` monthly steps). The EUR series for Austria are generated by the same build script with `--currency EUR` (§8.6.F):
*   **Empirical Datasets Ingested (`data/`)**:
    *   **US Stocks (`data/us_stocks.csv`)**: True **month-end** S&P 500 Total Return index in USD (dividends reinvested, 1871–2025), verified to $\pm 0.01\text{pp}$ against published month-end S&P 500 total returns (via [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator); previously misattributed to Shiller's monthly-averaged price series). Carried as **real monthly returns** throughout 1922–2025.
    *   **Non-US Equities (`data/jst_exus_usd.csv` + `data/ex_us_stocks.csv`)**: Spliced series. Because `ex_us_stocks.csv` is byte-identical to `us_stocks.csv` prior to MSCI EAFE's Dec 1969 inception, **1922–1969** uses a GDP-weighted 17-country ex-US equity total return index built from the **Jordà–Schularick–Taylor Macrohistory Database R6** (`eq_tr` converted to USD via `xrusd`, then to CHF; annual observations expanded into 12 geometric monthly steps, tracked in `MONTHLY_EXUS_IS_SMOOTHED`), spliced with **real monthly** MSCI EAFE / World ex-US returns from `data/ex_us_stocks.csv` for **1970–2025**.
    *   **USD/CHF Exchange Rates (`data/usd_chf.csv`)**: True **month-end** spot rates for **1971–present** built from daily **FRED `DEXSZUS`** observations (fixing a 13.4% 2024–2025 tail error in the legacy hand-extended series), spliced with **SNB `devkum`** monthly averages for **1914–1970** (the gold-standard / Bretton Woods peg era).
    *   **Swiss Inflation (`data/ch_inflation.csv`)**: Historical monthly Swiss Consumer Price Index (CPI / LIK) from the Swiss Federal Statistical Office (FSO/BFS, 1921–present).
    *   **Gold (`data/gold_usd.csv`)**: **LBMA London PM Fix** month-end USD/oz prices from **April 1968–present** multiplied by month-end `USD/CHF` to yield **real monthly CHF gold returns** (`MONTHLY_GOLD_CHF`). Pre-1968 statutory fixed gold prices (`$20.67/oz` then `$35.00/oz`) reflect official currency devaluations spread evenly across the year (`MONTHLY_GOLD_IS_SMOOTHED`).
    *   **Swiss Retail Cash Rate (`data/ch_cash_rate.csv`)**: Annual CHF cash yield (`1900–2025`, `retail_rate`) = **SNB interest rate on private-client savings deposits** (data.snb.ch cube `zikrepro`, `D1=S1`, annual mean of monthly values) from **1933**, and the **Jordà–Schularick–Taylor R6** Swiss `bill_rate` for 1900–1932 (identical to the savings rate on the 1933–1968 overlap, 0.007pp mean abs diff). This replaces the former wholesale proxy (JST money-market `bill_rate` from 1969 + SNB SARON, floored at 0%), which had the same average (2.30% vs 2.32% over 1933–2025) but far higher volatility (e.g. 1989: 9.7% vs 3.45%). `MONTHLY_CASH_CHF`, `HISTORIC_RETURNS_CASH_CHF`, 1922–2025: `2.49%` nominal CAGR, `1.42%` vol, `+0.74%` real after `1.74%` Swiss CPI. The wholesale series is retained in the CSV (`wholesale_rate`) for reference only.
    *   **US Inflation (`data/us_cpi.csv`)**: Monthly US CPI-U — **FRED `CPIAUCNS`** from 1913, legacy swr-calculator leg for 1871–1912 — fetched by `scripts/fetch_source_data.py`. Build-time only: used to derive `HISTORIC_REAL_CHF_APPRECIATION` $= 1 - \frac{1 + g_{\text{FX}}}{1 + g_{\text{PPP}}}$ (geometric 1922–2025 drifts, currently `0.0071`) instead of a hand-maintained constant.
    *   **Data integrity**: The build script asserts that the Gold series has no gaps inside the simulation window rather than silently filling them with `0.0` returns.
*   **Historic Backtesting Mode**: Replays contiguous historical **nominal** monthly return (`MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`) and annual Swiss CPI inflation sequences (1922–2025 in CHF, 104 years), adjusted by the user's `real_chf_appreciation` PPP setting (`0.00%` default). Generates $N = \text{total\_years} - \text{duration\_years} + 1$ overlapping cohorts, paired with Ito-corrected lognormal synthetic Bitcoin returns (`btc_mean`, default `7%`, and `btc_vol`, default `50%`, shared with the sidebar inputs) coupled to full-sample standardized US equity monthly log-returns via a Gaussian copula at correlation `btc_equity_corr` (default `0.50`).
*   **Historic Bootstrapping (Politis–Romano Stationary Bootstrap) Mode**: Jointly samples historical years (`MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`, and `HISTORIC_SWISS_INFLATION`) using the **Politis–Romano (1994) Stationary Bootstrap** (`stationary=True`, geometric block lengths of mean `block_size_years = 5` via restart probability $p = 1/b$) with **circular wrap-around** `(year_idx[:, t - 1] + 1) % total_years`. This eliminates endpoint under-sampling (every year from `1922` to `2025` has uniform $1/104$ selection probability) and fixed 5-year seam artifacts while preserving multi-year stagflation and crash/recovery clusters. Paired with Gaussian-copula Bitcoin returns (`btc_mean`, `btc_vol`, `btc_equity_corr`) coupled to full-sample standardized US equity monthly log-returns.
*   **Parametric Monte Carlo Mode**: Stochastic simulation seeded by `random_seed` using user-provided **nominal** asset class means ($\mu$) and standard deviations ($\sigma$) via lognormal returns (with Ito drift correction), cross-asset shock coupling (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$, `Bitcoin` $\rho=\text{btc\_equity\_corr}$), and an independent normally distributed annual inflation stream (`seed + 10_000`).
*   Model inflation in CHF explicitly by increasing base retirement expenses and AHV pension payouts annually. This separates nominal asset growth from the rising cost of living.


---

### 4.3. Outputs & Visualizations

The tool presents results side-by-side across all three simulation modes (**Historic Backtesting**, **Historic Bootstrapping**, **Monte Carlo**):
*   **TL;DR Status Indicator**: A quick overarching assessment of the median outcome (with a hover tooltip explaining all four statuses), based on the **median of per-run real final net worth** (each run's nominal final net worth deflated by that run's own cumulative inflation, `src/metrics.py::classify_outcome`). Comparing the nominal median against a multiple of the *median price level* would pair values from different runs and can mislabel outcomes.
    *   **BROKE**: Median real final net worth $\le 0$ (You run out of money).
    *   **SHORTFALL**: Median real final net worth is $> 0\text{ CHF}$ (solvent), but $\le$ the configured target ending wealth criterion (`success_pct`% of inflation-adjusted starting net worth).
    *   **DEAD**: Median real final net worth hits the configured success criterion ($> \text{success\_pct}\%$ of inflation-adjusted starting net worth) but is below $3\times$ the initial net worth.
    *   **RICH**: Median real final net worth hits the success criterion and is $\ge 3\times$ initial net worth (`RICH_MULTIPLE = 3.0`; real wealth grows massively).
*   **Net Worth Trajectory Chart**: A chart showing the progression of assets over the simulation horizon.
    *   Plots individual runs (faint "spaghetti" lines for all contiguous cohorts in Historic Backtesting, capped at 100 runs for performant rendering in Bootstrapping and Monte Carlo) to show dispersion.
    *   For **Historic Backtesting**, overlays **Worst Cohort (Min), 25th, 50th (median), 75th, and Best Cohort (Max)** (`[0, 25, 50, 75, 100]`) rather than 5th/95th percentiles that cannot be non-parametrically estimated at $N_{\text{eff}} \approx 2.1\text{–}2.6$. For **Historic Bootstrapping** and **Monte Carlo**, overlays **5th, 25th, 50th (median), 75th, and 95th** percentiles (`[5, 25, 50, 75, 95]`). Both include a dashed **Inflation-Adj Start NW** reference line.
*   **Portfolio Survival by Age Chart**: A Kaplan–Meier style survival chart across retirement ages `start_age .. start_age + duration_years - 1` showing *when* portfolio failures occur over the retirement horizon rather than only at the final year:
    *   **Solvent (`NW > 0`)**: Solid green curve showing the % of paths that have never depleted total net worth up to each age.
    *   **Liquid Solvent (`Liquid > 0`)**: Dotted orange curve showing the % of paths that have never exhausted accessible liquid wealth up to each age (highlighting Pre-AHV bridge insolvency before Age 65 while Pillar 2/3a are still locked).
    *   **Above Target (`{success_pct}% Start NW`)**: Dashed blue curve showing the % of never-depleted paths whose real net worth exceeds the target wealth preservation threshold at each age.
    *   Includes a vertical dashed **Age 65 (AHV + P2)** reference marker when the retirement span crosses age 65.
*   **Income vs Required Cash Chart**: Dynamic annual stacked bar view of inflows (Dividends, Cash Interest, AHV Pension from Age 65, Capital Sold) vs. a continuous reference line for **Inflation-Adj Start** (`(annual_base_expenses + estimate_year_0_taxes(config)) * median_cum_inflation`).
*   **Annual Withdrawal Breakdown Chart**: Stacked bar chart showing median (`50th Pct`) Living Expenses and Taxes Paid over time, overlaid with stacked lines (`stackgroup='bad'`, `fill='none'`) for the funded outflows of the bad-case outcome path (**Worst Cohort** in Historic Backtesting; **5th Pct** outcome path in Historic Bootstrapping and Monte Carlo, ranked via `worst_path_order` and re-expressed on the median CPI scale) and a continuous **Inflation-Adj Start** reference line (`(annual_base_expenses + estimate_year_0_taxes(config)) * median_cum_inflation`). Hovering the **Taxes Paid (50th Pct)** bar shows the median of each tax component (income tax, wealth tax, AHV contributions and the capital withdrawal tax; Austria: income tax, KESt, health insurance).
*   **Withdrawal Rate Chart**: Percentile/extrema chart showing the net annual portfolio withdrawal rate over time, computed as `max(0, Living Expenses + Taxes - AHV Pension) / Beginning-of-Year Net Worth`, using actual beginning-of-year net worth (`initial_net_worth` at Year 0 and `net_worth[t - 1]` for `t >= 1`, dynamically capped at 25% max on the y-axis for readability). In **Historic Backtesting**, the `0th` percentile (lowest withdrawal rate) is labeled **Best Cohort (Min)** and the `100th` percentile (highest withdrawal rate) is labeled **Worst Cohort (Max)**, with color coding inverted (`colors[::-1]`) relative to the Net Worth chart so `Best` (`purple`) and `Worst` (`crimson`) remain visually consistent.
*   **Asset Allocation Development Chart**: Stacked area chart displaying the median nominal balance of all asset categories (CHF Cash, US Stocks, Non-US Stocks, Gold, Bitcoin, and combined Pillar 2 & 3a) over time to visualize total net worth breakdown, glidepaths, and liquidations.
*   **Key Outflow & Failure Timing KPI Metrics**:
    *   **Avg Years Below Start NW**: Average number and percentage of years where after-tax net worth falls below the inflation-adjusted after-tax starting net worth watermark.
    *   **Depletion Rate (`NW ≤ 0`)**: Percentage of runs that completely deplete total net worth at any point during the horizon, with a tooltip decomposing failures into **Total Net Worth Depletion (`NW ≤ 0`)**, **Pre-AHV Liquid Depletion (`Liquid ≤ 0` before Age 65)**, and **Target Shortfall (`0 < Final Real NW ≤ Target` without ever depleting)**.
    *   **Depletion Age (`Min / Med`)**: Earliest and median age at which depleted runs first reach `Net Worth ≤ 0` (or `"None (0 broke)"` when no runs deplete), with a tooltip reporting simulation years and Pre-AHV liquid depletion rate.
    *   **Shortfall Rate (`Real NW ≤ Target`)**: Percentage of runs where real net worth drops to or below the target ending wealth criterion at any point during retirement.
    *   **Shortfall Age (`Min / Med`)**: Earliest and median age at which runs first drop to or below the target ending wealth criterion (e.g., $\le 50\%$ of inflation-adjusted starting wealth), or `"None (100% kept)"` if real net worth stays strictly above target across all simulation years.
    *   **Median Ending NW**: Real (inflation-adjusted) and Nominal ending net worth.
    *   **Median Total Withdrawals**: Total cumulative cash spent on living expenses and taxes over the entire retirement horizon (Real and Nominal).
    *   **Pre-AHV Outflow (< Age 65)**: Total median **real** cash required to cover living expenses and taxes during the early retirement gap before age 65 (with Nominal total in hover tooltip).
    *   **Post-65 Outflow (Age 65+)**: Total median **real** cash required to cover living expenses and taxes from age 65 through end-of-life (with Nominal total in hover tooltip).
*   **Cohort / Run Analysis Tables**: Interactive data tables detailing the Top 10 Best and Worst individual simulation cohorts/runs. **Best** paths are sorted by `Final NW (Real)` descending. **Worst** paths are ranked by **outcome severity** relative to the configurable success criterion (default 50% of inflation-adjusted starting net worth), most severe group first: (1) **Depleted** — net worth reached ≤ 0, earliest depletion first; (2) **Failed** — survived (`NW > 0` every year) but final real net worth ≤ criterion, lowest final real net worth first; (3) **Recovered** — real net worth dropped to ≤ criterion at some point during retirement (without ever hitting `NW ≤ 0`) and finished above it, most years in shortfall first; (4) **Never breached** — lowest final real net worth first. Remaining ties are broken by lowest final real net worth. The sort order is stated in a caption above each table (Streamlit `st.dataframe` offers no API to preset its interactive header sort arrow). Columns, each with a native `st.column_config` hover tooltip: `Cohort`/`Run`, `Outcome` [severity group], `Depletion Age` [`Survived`, `Age X (Yr Y)`, or `Liq Age X (Yr Y)`], `Yrs in Shortfall` [years at/below the criterion], `Shortfall Age` [`Never` or `Age X (Yr Y)`, first breach even if later recovered], `Final NW (Real)`, `Final NW (Nom)`, `Min NW (Real)` [deepest real trough], and `Yrs Below` [years below the starting watermark].
*   **Success Metric & Effective Sample Size ($N_{\text{eff}}$)**:
    *   Allows configuring a target ending net worth as a percentage of inflation-adjusted starting net worth (default is 50.0%).
    *   Shows the calculated probability of success matching this definition across all simulation runs (requiring both never depleting `NW ≤ 0` during retirement and ending above target, so `Success + Depletion + Final Shortfall = 100%`).
    *   In **Historic Backtesting**, explicitly surfaces the number of overlapping cohorts ($N_{\text{cohorts}} = T - D + 1$), the **Effective Sample Size ($N_{\text{eff}} = T / D$)** (e.g. `2.6` for a 40-year horizon or `2.1` for a 50-year horizon across 104 years), and the **90% two-sided Wilson confidence interval** evaluated at $N_{\text{eff}}$ (with matching vertical space reserved in **Historic Bootstrapping** and **Monte Carlo** so all three columns remain vertically aligned).


---

## 5. Key Swiss/Zurich Specific Rules to Model

| Rule | Description | Simulation Impact |
| :--- | :--- | :--- |
| **No Capital Gains Tax** | Capital gains on private assets (e.g., selling stocks) are 100% tax-free. | Only dividends/interest add to taxable income. |
| **Wealth Tax** | Canton Zurich taxes net wealth globally. | High net worth individuals pay significant wealth tax, which acts as a drag on portfolio growth during decumulation. |
| **AHV for Non-Workers** | Early retirees (`< 65`) must pay AHV/IV/EO contributions based on their wealth and 20× pension income, which are deductible from taxable income (Art. 33 Abs. 1 lit. d DBG / § 31 Abs. 1 lit. d StG ZH). | Costs 530 to 26,500 CHF/year per person, partially offset by reducing taxable dividend/interest income. |
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
2.  **Tax Formula Accuracy**: Vectorized progressive bracket engine in `src/tax_engine.py` modeling Federal income tax, Zurich Cantonal/Municipal income & wealth taxes, capital withdrawal tax, and 2025/2026 AHV non-worker contribution tables. The simulation engine reaches country-specific rules only through the country dispatch layer `src/jurisdiction.py` (`get_jurisdiction(config.country)`), which dispatches to `Switzerland` (`src/tax_engine.py`) or `Austria` (`src/tax_engine_at.py`: 2026 § 33 EStG tariff, KESt, Regelbesteuerungsoption, pension taxation, § 16 ASVG health self-insurance; §8).
3.  **Simulation & Return Modeling**: Side-by-side execution of contiguous Historic Backtesting (`1922–2025`), joint Historic Bootstrapping, and Ito-corrected lognormal Monte Carlo simulation with configurable `Random Seed`. The engine also tracks an average-cost basis per asset class and reports realized / unrealized capital gains (`realized_gains`, `unrealized_gains`, `cost_basis_by_class`); they are not taxed in Switzerland and are the base for the Austrian capital gains tax (§8).
4.  **Result Metrics**: Success criterion, survival curves by age, failure timing/depletion decomposition, shortfall duration, outcome severity classification and worst-path ranking, minimum real spending ratios, real ending net worth, TL;DR classification, and withdrawal-rate reconstruction live in `src/metrics.py` (pure, unit-tested functions shared by all UI panels) instead of being re-implemented inline in `app.py`.
5.  **Testing**: `pytest` suite covering both Swiss and Austrian tax engines (including ESTV and BMF 2026 golden tests), the jurisdiction layer, cost-basis and capital-gains accounting, CHF and EUR historical return series, the simulation engine, metrics, and Streamlit `AppTest` smoke tests of `app.py` (both countries, country-default resets on switching, spending strategies, allocation validation, PPP slider).

---

## 8. Austria Jurisdiction (Country Switch)

> [!IMPORTANT]
> **Status: ✅ implemented (M1–M6) and selectable in the app** (sidebar **Country** → *Austria*). Everything below is implemented; the pension taxation is verified against the official BMF Brutto-Netto-Rechner 2026 (13 pensions from €1,100 to €95,000 a month: health contribution, income tax and net pension within €0.13). What exists: (M1) the `SimConfig.country` field and the `src/jurisdiction.py` dispatch layer; (M2) average-cost basis tracking for every sleeve with `realized_gains`, `unrealized_gains` and `cost_basis_by_class` outputs (tracked but untaxed in Switzerland); (M3) the EUR return series `src/historic_returns_eur.py` built from BIS/ECB/Statistik Austria data with the documented Swiss-data stand-ins (§8.6.F); (M4/M4b) the Austrian tax rules in `src/tax_engine_at.py` and the `Austria` jurisdiction (KESt, Regelbesteuerungsoption, BMF-verified pension taxation, health insurance); (M5) the sidebar country selector, Austrian inputs/labels, Swiss-data warning banner and tax-component hover. The technical design is in [`design_doc_early_retirement.md` §7](design_doc_early_retirement.md#7-multi-jurisdiction-design-austria); a status summary is in the [README](../README.md#jurisdictions--implementation-status).

**Status legend** (shared by this section, design doc §7 and the README):
✅ implemented · 🔁 country-agnostic, reused unchanged (already implemented) · ❌ not implemented. All design questions are resolved (§8.9).

### 8.1. Agreed Decisions (2026-11-10)
1.  **One app with a country switch** (`Switzerland (Zurich)` / `Austria`), not a separate app: one simulation engine, one metrics module, one test suite.
2.  **Health self-insurance is mandatory** for Austrian early retirees (§ 16 ASVG Selbstversicherung in der Krankenversicherung); it is not a user toggle.
3.  **All securities are "Neuvermögen"**: all stocks/ETFs are assumed bought after 1 Jan 2011 and all Bitcoin after 28 Feb 2021, so every realized gain is taxable. There is no grandfathering (Altvermögen) input.
4.  **State pension is a single user input** (today's value, starting at age 65), exactly like the Swiss AHV input. No Pensionskonto projection, no early (Korridor) pension, no voluntary pension contributions.
5.  **Zero unrealized gains at start**: the whole starting portfolio is treated as bought (or gains realized and "cleared") on day 0, so the initial cost basis equals the market value of every sleeve. There is no unrealized-gain input.
6.  **Gold is held as a gold ETC**, taxed like other capital income at 27.5% (no one-year holding-period exemption).
7.  The remaining questions follow the recommendations in §8.9: 1922–2025 EUR history where **1922–1957 uses Swiss inflation and Swiss cash-rate data** (no Austrian data before 1958, §8.6.F), pension entered as a monthly gross amount paid 14×/yr, ⅔-of-CPI bracket indexation by default, after-tax watermark net of latent KESt, proportional selling, and a forward-looking EUR cash rate set from current Austrian savings rates.
8.  **EUR Cash 1958–1999 = Swiss savings rate translated to EUR under relative PPP** (Swiss real cash return re-inflated with Austrian CPI), documented as a second Swiss-data stand-in; 2000+ uses the ECB rate on Austrian household deposits with agreed maturity up to 1 year (§8.9 Q9).

### 8.2. Persona & Scope
*   **Persona**: Austrian tax resident, single, no dependents, no church contribution (Kirchenbeitrag not modelled), already retired at simulation start (same Phase 1 decumulation scope as Switzerland).
*   **No location input**: Austria levies a single nationwide income tax; there are no Länder or municipal tax multipliers.
*   **Out of scope**: Austrian occupational/private pension vehicles (Abfertigung Neu / Vorsorgekasse, Pensionskasse, prämienbegünstigte Zukunftsvorsorge), real estate, cross-border relocation effects (e.g. Swiss → Austrian move, exit taxation), means-tested benefits (Ausgleichszulage), reduced self-insurance contributions on application, non-reporting funds (Nicht-Meldefonds), foreign withholding-tax credit mechanics (assumed fully creditable), and the SV-Rückerstattung negative tax.

### 8.3. Reused Country-Agnostic Functionality
| Capability | Status |
| :--- | :--- |
| Three simulation engines (Historic Backtesting, Stationary Bootstrap, Parametric Monte Carlo), Bitcoin Gaussian copula | 🔁 (engines reused; ✅ EUR input data, M3, used by the app, M5; see §8.6.F) |
| Spending strategies (Static, Dynamic Floor & Ceiling, Vanguard Dynamic incl. TWR sync) | 🔁 (✅ Austrian Year-0 tax estimate for TWR sync and Cash Tent sizing, M4) |
| Rebalancing strategies (Cash Tent, Monthly, Quarterly, Yearly, Threshold, Never), Smart Cash Buffer | 🔁 (✅ sales realize taxable gains in Austria, M4, see §8.6.A) |
| Success criterion, TL;DR, survival curves, KPIs, cohort tables, all charts | 🔁 (✅ Austrian labels, M5, see §8.7) |
| Currency / PPP slider mechanism | 🔁 (✅ re-derived for EUR, M3; Real EUR Appreciation slider, M5) |
| Bracket indexation mechanism (`bracket_indexation`, homogeneity identity) | 🔁 (✅ Austrian tariff plugged in, M4) |

### 8.4. Data Inputs (Austria Mode)
| Input | Switzerland (✅) | Austria | Status |
| :--- | :--- | :--- | :--- |
| **Country selector** | — | Top of sidebar: `Switzerland (Zurich)` / `Austria`. Switching loads that country's defaults and hides inputs that do not apply. |✅ |
| Currency | CHF | EUR (all monetary inputs, outputs and labels) |✅ |
| Taxable Liquid Wealth | ✅ (`CHF 2,450,000` default) | Same input, in EUR (`€3,250,000` default). Initial cost basis = market value (zero unrealized gains on day 0, decision 5). | ✅ |
| Pillar 2 / Pillar 3a balances | ✅ | Must be 0 (`SimConfig` rejects them for `"AT"`); hidden in the UI | ✅ |
| Target Asset Allocation | 5 sleeves incl. `CHF Cash` | Same 5 sleeves; `CHF Cash` becomes `EUR Cash`, `Gold` is a gold ETC |✅ |
| **State pension** | Monthly AHV from 65 (paid 12×/yr) | **Expected monthly gross state pension** from 65, in today's EUR, **paid 14×/yr** (12 regular + 13th/14th special payments); same `SimConfig.monthly_ahv_pension` field | ✅ |
| **Health self-insurance** | n/a | `SimConfig.health_insurance_monthly`, default **€565.25/month** (2026 ÖGK rate, 12×/yr); mandatory (no toggle) until the state pension starts, for life if the pension is 0 (§8.6.D); shown read-only in the UI | ✅ |
| AHV non-worker contributions | ✅ (automatic) | Not applicable | ✅ (bypassed) |
| Municipal Multiplier | ✅ | Hidden |✅ |
| Tax Bracket Indexation (% of CPI) | Default 100% | Same number input, **default 66.7%** (statutory automatic ⅔ indexation, § 33a EStG); `SimConfig.bracket_indexation=None` resolves to the country default | ✅ |
| Currency / PPP Assumption | Real CHF appreciation beyond PPP | **Real EUR appreciation beyond PPP**, historic value re-derived from EUR data (`real_eur_appreciation`) | ✅ |
| Monte Carlo defaults | Calibrated to 1922–2025 CHF history | Calibrated to the EUR history (§8.6.F) |✅ |
| Cash rate (forward-looking) | 1% CHF | **2%** (ECB MIR, Austrian households, Sep 2026: agreed maturity ≤ 1y 2.18%, notice deposits 2.05%); `SimConfig.cash_rate=None` resolves to the country default | ✅ |

### 8.5. Key Differences vs. Switzerland
| Topic | Switzerland (✅) | Austria (✅) |
| :--- | :--- | :--- |
| Wealth tax | Canton + municipality | **None** (abolished 1994) |
| Realized capital gains (private) | Tax-free | **27.5% KESt** (§ 27a Abs. 1 Z 2 EStG) |
| Dividends / fund income | Progressive income tax | 27.5% flat; accumulating funds taxed annually on deemed distributions (ausschüttungsgleiche Erträge) |
| Bank interest | Progressive income tax | 25% flat (§ 27a Abs. 1 Z 1 EStG) |
| Gold | Wealth-taxed; gains tax-free | Gold ETC: 27.5% KESt on gains, part of the loss-offset pool |
| Bitcoin | Wealth-taxed; gains tax-free | 27.5% on gains (Neuvermögen) |
| Mandatory contributions before pension | AHV non-worker (CHF 530–26,500/yr, deductible) | Health self-insurance (€6,783/yr in 2026, not deductible) |
| Pension vehicles | Pillar 2 / 3a lump sums with capital withdrawal tax | Not modelled (out of scope) |
| Income tax on pension | Federal + cantonal + municipal tariff | National tariff; 6% health contribution withheld (capped at €6,930/payment); 13th/14th payment at 0% / 6% / 27% / 35.75% / tariff bands from €620 (tax-free up to the €2,615 Freigrenze); pensioner tax credit — verified against the BMF Brutto-Netto-Rechner |
| Bracket indexation | 100% CPI by law (Art. 39 DBG, § 48 StG ZH) | Brackets 1–5 automatically indexed by ⅔ of inflation (§ 33a EStG); remaining ⅓ discretionary |

### 8.6. Calculation Logic (Austria)

#### A. Capital Income Tax — KESt (✅ M4)
*   **Dividends / deemed distributions**: `(US Stocks + Non-US Stocks) × dividend_yield` on the holdings held during the year, taxed at **27.5%**. Distributing and accumulating reporting funds (Meldefonds) are treated identically. Because all return series are total-return (dividends reinvested), the taxed amount is treated as reinvested and **raises the cost basis** of the sleeve.
*   **Realized capital gains**: every sale of US Stocks, Non-US Stocks, Gold (ETC) or Bitcoin — to fund expenses and taxes, for rebalancing, or along the Cash Tent glide path — realizes `proceeds × (1 − basis / value)` of gain (average-cost method, matching the Austrian "gleitender Durchschnittspreis"), taxed at **27.5%**. On day 0 the basis equals the market value of every sleeve (decision 5), so only post-retirement gains are ever taxed.
*   **EUR Cash interest**: `EUR Cash × max(0, realized_annual_cash_return)` (same realized-return logic as the Swiss model), taxed at **25%**.
*   **Gold**: held as a **gold ETC** (decision 6) → gains are ordinary 27.5% capital income and part of the loss-offset pool; no distributions (no dividend yield).
*   **Loss offset** (§ 27 Abs. 8 EStG): realized losses offset realized gains and dividends / deemed distributions **within the same calendar year** (27.5% pool only; bank interest is excluded). **No loss carry-forward**: an unused loss expires at year-end.
*   **Settlement (simplified timing)**: all KESt of a calendar year is settled at the year-end assessment (Month 11) on that year's pool, rather than withheld at each sale. The amount is the same as with a domestic ("steuereinfach") broker that withholds per sale and nets losses in-year; only the intra-year timing differs. The year-end outflow sale realizes gains or losses that affect the same year's tax, so the engine solves `sale = expenses + taxes(gains so far + gain(sale))` by monotone bracketed bisection (globally convergent to machine precision for both capital gains and capital losses) and settles any tiny rounding residual after the sale. Gains of the year-end rebalance (which runs after the assessment) are taxed in the following year, as if the rebalance happened on the first trading day of January.
*   **Regelbesteuerungsoption** (§ 27a Abs. 5 EStG): every year the engine computes the tax both ways — (a) flat: progressive tax on the pension + 27.5% on the pool + 25% on interest; (b) all special-rate capital income taxed at the progressive tariff together with the pension — and applies the **lower** one. The option is all-or-nothing for all special-rate capital income. It is applied in the same year (in reality the KESt is withheld first and the difference refunded by the assessment the year after; timing only). This matters most for early retirees with little other income (e.g. €30k of gains and no other income: ≈ €4.1k at the 2026 progressive tariff vs. €8.25k flat).

#### B. Progressive Income Tax (§ 33 EStG, 2026 tariff) (✅ M4)
| Taxable income (EUR) | Marginal rate |
| :--- | :--- |
| 0 – 13,539 | 0% |
| 13,539 – 21,992 | 20% |
| 21,992 – 36,458 | 30% |
| 36,458 – 70,365 | 40% |
| 70,365 – 104,859 | 48% |
| 104,859 – 1,000,000 | 50% |
| > 1,000,000 | 55% (temporary, until 2029) |

*   **Taxable base**: the 12 regular monthly pension payments (`12 × monthly_state_pension`), after deducting the 6% health insurance contribution on the pension (capped at the 2026 Höchstbeitragsgrundlage of €6,930 per payment, i.e. at most €415.80); plus all capital income if the Regelbesteuerungsoption is cheaper (§8.6.A).
*   **Special payments** (13th and 14th pension payment, `2 × monthly_state_pension`, after the health contribution; § 67 Abs. 1 EStG within the Jahressechstel, which two monthly payments never exceed): **no tax if the Jahressechstel (two monthly gross payments) is at most €2,615** (2026 Freigrenze, i.e. pensions up to €1,307.50/month); otherwise 0% on the first **€620**, **6%** on the next €24,380, 27% on the next €25,000, 35.75% on the next €33,333.33, and the rest at the regular tariff.
*   **Pensionistenabsetzbetrag** (tax credit, 2026): **€1,020** for pension income up to **€21,614**, phasing out linearly to 0 at **€31,494**. Credits cannot make the tax negative.
*   **Indexation**: all euro amounts (bracket edges, €620 allowance, €2,615 Freigrenze, § 67 bands, €6,930 health-contribution ceiling, and the pensioner credit and its phase-out limits) are indexed each year with `bracket_indexation × CPI` via the same homogeneity identity as the Swiss tariffs. **Default `bracket_indexation = 2/3`** for Austria (the statutory automatic part, § 33a EStG); the discretionary third is not assumed. Simplification: the €1M edge, § 67 bands and contribution ceiling are indexed with the same bracket factor (by law they follow different or no indexation).

#### C. No Wealth Tax, No Capital Withdrawal Tax, No AHV (✅ M4)
*   Wealth tax, AHV non-worker contributions and the Pillar 2 / 3a capital withdrawal tax evaluate to 0 in Austria mode. With no locked pension balances, liquid net worth equals total net worth.

#### D. Health Self-Insurance — § 16 ASVG (✅ M4)
*   **Mandatory** from the start of the simulation until the state pension starts (age 65). If the state pension input is 0, it applies for the whole horizon.
*   **2026 contribution: €565.25/month (€6,783/yr)**, indexed annually with CPI (simplification: the law indexes it with the wage-based Aufwertungszahl).
*   **Not tax-deductible** (conservative: voluntary insurance is not a Werbungskosten deduction, and the capped "Topf-Sonderausgaben" deduction expired after 2020).
*   Reported together with taxes as a mandatory outflow, like the Swiss AHV contribution, and also as its own output series.
*   From age 65 it is replaced by the **6% health insurance contribution withheld from the gross pension**.

#### E. State Pension (✅ M4)
*   User input: **monthly gross pension in today's EUR, paid 14× per year** from age 65 (12 regular payments + 13th/14th special payments in April and October; modelled as `14/12 × monthly` credited every month), CPI-indexed from Year 0 (same convention as the Swiss AHV input; actual Austrian pension adjustments can lag CPI). Credited gross to EUR Cash; the 6% health contribution is part of the year-end tax assessment (reported in `health_insurance`), and the pension is taxed per §8.6.B.

#### F. Investment Returns in EUR (✅ data, M3; used by the app, M5)
*   All engines run on EUR-denominated series for US Stocks, Non-US Stocks, Gold and EUR Cash, with Austrian CPI as the inflation series. Austria's monetary history (1921–22 hyperinflation, 1938 annexation, 1945–47 currency reforms) means there is **no clean 1922–2025 local-currency series**, and Austria is not in the JST database.
*   **Implemented method (Q1, Q9)**: the full **1922–2025** window (same 104 years and cohort count as Switzerland), generated as `src/historic_returns_eur.py`:
    *   **Equities and gold**: the same USD total-return / gold series as the CHF module, converted to EUR at index level with month-end EUR per USD.
    *   **EUR/USD**: 1958–2025 market rate — BIS `WS_XRU` `M.AT.EUR.E` (Austrian schilling converted at the fixed 13.7603 ATS/EUR before 1999; identical to the ECB reference-rate month-ends from 1999). 1922–1957: relative-PPP-implied rate (Swiss vs US CPI), anchored to the Dec 1957 market rate.
    *   **Inflation**: 1958–2025 Austrian CPI — Statistik Austria VPI 1958 (the official long-chained national CPI, from Dec 1959) with the BIS long CPI (sourced from Statistik Austria) chained on for Dec 1957 / Dec 1958. 1922–1957: **Swiss CPI**.
    *   **EUR Cash**: 2000–2025 ECB MIR rate on Austrian household deposits with agreed maturity up to 1 year (new business; within ~0.16pp of the notice-deposit rate where both exist). 1958–1999: **Swiss savings rate translated to EUR under relative PPP**, $(1 + r_{CH})(1 + \pi_{AT})/(1 + \pi_{CH}) - 1$ (keeps the Swiss real cash return), because no open Austrian retail savings-rate series exists before 2000 (the regulated "Eckzinssatz" is only published in OeNB PDFs). 1922–1957: **Swiss savings rate** as-is.
    *   The Currency / PPP slider (Real EUR Appreciation beyond PPP) works exactly as for CHF; `HISTORIC_REAL_EUR_APPRECIATION` = 0.57%/yr, derived at build time over the whole 1922–2025 window (the 1922–1957 segment is PPP-neutral by construction, so the residual comes from the market-rate years).

> [!WARNING]
> **The Austrian simulation uses Swiss data for 1922–1957, and Swiss-based cash returns until 1999.** For every year before 1958, the "Austrian" inflation series *is* the Swiss CPI (`data/ch_inflation.csv`) and the "EUR Cash" return *is* the Swiss retail savings rate (`data/ch_cash_rate.csv`); equities and gold for those years are USD returns translated at a PPP-implied (not market) exchange rate. No Austrian data enters before 1958. This covers 36 of 104 years, so it affects every Historic Backtesting cohort that starts before 1958 and every bootstrap block that samples those years. Austria's own 1921–22 hyperinflation and 1945–47 currency reforms are therefore **not** in the Austrian history. In addition, the **1958–1999 EUR Cash return is the PPP-translated Swiss savings rate** (Swiss real return re-inflated with Austrian CPI), so its year-to-year nominal path follows the Austrian–Swiss inflation differential rather than the smoother regulated Austrian savings rate. The generated module labels both segments (`HISTORIC_YEAR_IS_SWISS_PROXY`, `HISTORIC_YEAR_CASH_IS_SWISS_PROXY`); the app states this in a warning banner above the results, in the Country selector help, the historic-data info box and the Historic Backtesting header tooltip (M5).
*   Resulting 1922–2025 real CAGRs (EUR, deflated by the composite Austrian inflation): US 7.1%, Non-US 4.9%, gold 1.8%, cash 0.6% (CHF module: 6.9%, 4.7%, 1.6%, 0.7%). Bitcoin stays synthetic.

### 8.7. Outputs (Austria Mode)
*   All §4.3 outputs are reused (🔁) with country-specific labels (✅ M5), a title **Austria Early Retirement Simulator** and a warning banner stating the Swiss-data stand-ins: `AHV Pension` → `State Pension`, `Pre-AHV` → `Pre-Pension`, `Age 65 (AHV + P2)` → `Age 65 (State Pension)`; the `Liquid Solvent` curve and `Pre-AHV Liquid Depletion` decomposition are hidden (identical to `Solvent` without locked pensions).
*   **New outputs** (✅): `tax_components` with annual `income_tax`, `kest` and `health_insurance` (self-insurance + 6% on the pension) per run, plus `realized_gains`, `unrealized_gains` and `cost_basis_by_class` (M2) — the component medians are shown when hovering the **Taxes Paid (50th Pct)** bar of the Annual Withdrawal Breakdown chart (both countries).
*   **After-tax watermark** (✅ M4): the Swiss watermark deducts the deferred capital withdrawal tax on Pillar 2 / 3a. The Austrian equivalent deducts the latent KESt on net unassessed non-cash gains (`0.275 × max(0, carried_gains + intra_year_realized_gains + Σ(value − basis))` over US Stocks, Non-US Stocks, Gold and Bitcoin, i.e. losses offset gains and rebalancing before year-end settlement does not artificially raise after-tax net worth). Because the basis equals the market value on day 0, the starting watermark equals the starting net worth.

### 8.8. Implementation Status Summary (Austria)
| Component | Status |
| :--- | :--- |
| `SimConfig.country` field, `COUNTRY_DEFAULTS`, jurisdiction dispatch (`src/jurisdiction.py`) (M1; `"AT"` accepted since M4) | ✅ |
| Country selector in the UI (M5) | ✅ |
| EUR / Austrian CPI data pipeline and generated return module (M3: `scripts/fetch_source_data_at.py`, `build_historic_returns.py --currency EUR`, `src/historic_returns_eur.py` with Swiss-proxy flags) | ✅ |
| Cost-basis tracking and realized-gain accounting in the engine (M2: `_sell` / `_buy` / `_rebalance_to`, dividend basis step-up, `realized_gains` / `unrealized_gains` / `cost_basis_by_class` outputs) | ✅ |
| KESt (27.5% / 25%), in-year loss offset, year-end outflow sale solved for the tax on its own gains (M4) | ✅ |
| Regelbesteuerungsoption (M4) | ✅ |
| Progressive tariff, pension taxation (6% health contribution, 13th/14th, pensioner credit) (M4) | ✅ |
| Health self-insurance (§ 16 ASVG), CPI-indexed (M4) | ✅ |
| Wealth tax / AHV / Pillar bypass (M4) | ✅ |
| Austrian Year-0 tax estimate (Cash Tent, TWR sync) (M4) | ✅ |
| New output series (`tax_components`, M4) | ✅ |
| Austrian labels, inputs, Swiss-data banner and tax-component hover in the UI (M5) | ✅ |
| Hand-computed statutory examples (`tests/test_tax_engine_at.py`, M4) | ✅ |
| Golden tests vs. the official BMF Brutto-Netto-Rechner 2026 for pensions (13 cases, ≤ €0.13; `test_pension_matches_bmf_2026`). The BMF has no online calculator for capital income, so KESt / Regelbesteuerung stay covered by the hand-computed examples. | ✅ |

### 8.9. Resolved Design Questions (2026-11-10)
| # | Question | Decision |
| :--- | :--- | :--- |
| Q1 | **EUR return history**: (a) real Austrian data from 1958 only, or (b) keep 1922–2025 with a synthetic PPP-translated pre-1958 segment? | **(b)**: 1922–2025; **1922–1957 uses Swiss CPI and the Swiss savings rate** (no Austrian data before 1958), flagged in the UI (§8.6.F) |
| Q2 | **State pension input format**: monthly ×14 or annual? | **Monthly gross, paid 14×/yr** |
| Q3 | **Gold**: physical gold (tax-free after 1 year) or gold ETC (27.5%)? | **Gold ETC**, 27.5% |
| Q4 | **Unrealized gain** in the starting portfolio. | **0%**: everything treated as bought / cleared on day 0; no input |
| Q5 | **Bracket indexation default**: ⅔ of CPI or 100%? | **⅔** (66.7%) |
| Q6 | **After-tax watermark**: deduct latent KESt on unrealized gains? | **Yes** |
| Q7 | **Tax-aware selling** vs. proportional selling? | **Proportional** for now; tax-aware selling as a possible later toggle |
| Q8 | **Forward-looking EUR cash rate** default. | Set from **current Austrian savings-account rates** at implementation time |
| Q9 | **EUR Cash 1958–1999**: no open Austrian retail savings-rate series exists (Eckzinssatz only in OeNB PDFs; IMF/ECB/BIS deposit rates start 2000/2003). | **Swiss savings rate translated to EUR under relative PPP**; 2000+ ECB MIR Austrian household deposits, agreed maturity ≤ 1 year |
