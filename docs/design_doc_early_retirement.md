# Design Document: Zurich Early Retirement Simulator (Phase 1)

## 1. Overview
This document outlines the technical design for Phase 1 of the Zurich Early Retirement Simulator. Phase 1 focuses exclusively on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich. 

The system will ingest initial asset balances, apply historic or Monte Carlo investment returns in CHF, compute complex Swiss tax obligations (including non-worker AHV contributions and capital withdrawal taxes), and visualize the resulting net worth trajectories.

## 2. Architecture
To prioritize rapid development, rich interactive visualizations, and local data privacy, the application will be built using a **Python-based data science stack** and a **Streamlit** frontend.

*   **Frontend / UI**: Streamlit (Python). Runs locally, ensuring no sensitive financial data leaves the user's machine. Provides built-in interactive widgets and seamless integration with charting libraries.
*   **Core Engine**: Python 3.11+.
*   **Vectorization & Math**: `NumPy` and `Pandas` for running thousands of simulation iterations concurrently and efficiently.
*   **Visualization**: `Plotly` for interactive, performant web-based charts ("spaghetti" plots of individual paths, rendered for all historic cohorts and capped at the first 100 runs for Monte Carlo and Bootstrapping).
*   **Module Layout**: The engine lives in a regular Python package, `src/`, with an `__init__.py` that re-exports the public API. Intra-package imports are relative (`from .tax_engine import ...`); `app.py` and the tests import absolutely from the repository root (`from src.simulation_engine import ...`). No `sys.path` manipulation is used anywhere: Streamlit prepends the script's directory for `app.py`, and a root-level `conftest.py` makes pytest prepend the repository root for the test suite. Hot-reload during development is handled by Streamlit's own `LocalSourcesWatcher`, which evicts changed modules from `sys.modules` between reruns — no manual `importlib.reload` is required.
    *   `src/tax_engine.py` (tariffs, shared Zurich multiplier constants), `src/simulation_engine.py` (`SimConfig`, `run_simulation`, return generators), `src/historic_returns.py` (generated data), and `src/metrics.py` (pure post-processing of simulation output: `cumulative_inflation`, `compute_success_mask`, `real_final_net_worth`, `classify_outcome`, `beginning_of_year_withdrawal_rate`, `RICH_MULTIPLE`). `app.py` only renders; it contains no success/TL;DR/withdrawal-rate math of its own.
    *   Tests: `tests/test_tax_engine.py`, `tests/test_simulation_engine.py`, `tests/test_metrics.py`, and `tests/test_app.py` (Streamlit `streamlit.testing.v1.AppTest` smoke tests running the full script headlessly; assertions use `at.main.*` because `at.header` also contains sidebar headers).

## 3. Core Modules

The system is divided into four primary modules:

### 3.1. User State & Configuration (Input Layer)
A centralized data structure (`SimConfig`) representing the simulation parameters, enforcing strict mathematical validation in `__post_init__`:
*   **Validation Rules**: `num_runs > 0`, `duration_years > 0`, `start_age >= 0`, `tent_duration_years >= 0`, `inflation_std >= 0.0`, `inflation_mean > -1.0`, `cash_rate > -1.0`, `seed >= 0`, `real_chf_appreciation < 1.0`, `-1.0 <= btc_equity_corr <= 1.0`, `bracket_indexation >= 0.0`, `spending_strategy in VALID_SPENDING_STRATEGIES`, `rebalance_strategy in VALID_REBALANCE_STRATEGIES`, individual asset allocations $\ge 0.0$ and $\sum \text{allocations} = 1.0$, non-negative initial asset balances (`initial_liquid_wealth`, `initial_pillar_2`, `initial_pillar_3a_accounts`), non-negative expenses/income/yields (`annual_base_expenses`, `monthly_ahv_pension`, `dividend_yield`), non-negative tax multipliers (`cantonal_multiplier`, `municipal_multiplier`), non-negative rebalancing thresholds, non-negative spending strategy parameters (`vanguard_target_rate`, `vanguard_floor_pct`, `vanguard_ceiling_pct`, `dynamic_expense_floor_pct`, `dynamic_expense_ceiling_pct`), and `vanguard_floor_pct <= 1.0` (a cut above 100% would permit negative spending). The Streamlit widgets carry matching `min_value`/`max_value` bounds so invalid UI input never reaches `__post_init__`.
*   **Normalization**: `initial_pillar_3a_accounts` defaults to a fresh empty list (`field(default_factory=list)`; an explicit `None` is also coerced to `[]`), and the legacy `enable_dynamic_expenses=True` flag coerces `spending_strategy` from `"Static"` to `"Dynamic (Floor & Ceiling)"` before strategy validation runs.
*   **Demographics**: Retirement age (`start_age`), simulation duration (`duration_years`).
*   **Economics & Taxes**: Inflation mean & volatility (for Monte Carlo), dividend yield, Year 0 cash interest rate (`cash_rate`, default `0.01`), Bitcoin–equity monthly log-return correlation (`btc_equity_corr`, default `0.50`), cantonal tax multiplier (default `ZURICH_CANTONAL_MULTIPLIER = 0.95`, Canton Zurich 2026), municipal tax multiplier (Steuerfuss, default `ZURICH_CITY_MUNICIPAL_MULTIPLIER = 1.19`, Zurich City), **Real CHF Appreciation beyond PPP** (`real_chf_appreciation`, default `0.0` = Relative PPP neutrality; `HISTORIC_REAL_CHF_APPRECIATION` ≈ `0.0070` = raw 1922–2025 historical real CHF appreciation), and **Tax Bracket Indexation** (`bracket_indexation`, default `1.0` = full CPI indexation per Art. 39 DBG / § 48 StG ZH).
*   **Assets**: Taxable Liquid Wealth, Pillar 2 (`Freizügigkeitskonto`), Pillar 3a balances (`initial_pillar_3a_accounts`, configured in the UI via `Number of Pillar 3a Accounts` (`0–5`) and a single `Balance per Pillar 3a Account (CHF)` input applied equally across all accounts: `[pillar_3a_balance] * int(num_pillar_3a)`).
*   **Expenses & Income**: Annual base expenses, Spending Strategy (`Static`, `Dynamic (Floor & Ceiling)`, `Vanguard Dynamic`), and Expected **Monthly** Pillar 1 (AHV) Pension from age 65.
*   **Simulation Config**: Simulation Mode (`Historic Backtesting`, `Historic Bootstrapping`, `Monte Carlo`), Number of Runs (`mc_num_runs`, `boot_num_runs`), Stationary Bootstrap Mean Block Length (`block_size_years`, default `5`), **Random Seed** (`random_seed`, default `42`), and Success Criteria (`success_pct` of inflation-adjusted starting net worth).

### 3.2. Tax & Pension Engine
This is a stateless utility module (`src/tax_engine.py`) responsible for all Swiss-specific tax and social security calculations for a given year. All functions are vectorized, support both scalar and N-dimensional array inputs while preserving return type and shape, and safely clamp negative inputs to zero.

*   **Bracket Table Vintages & Verification**: `FEDERAL_INCOME_BRACKETS` (2026, Art. 36 Abs. 1 DBG, single: 15,200 / 33,200 / 43,500 / 58,000 / 76,200 / 82,100 / 108,900 / 141,500 / 185,100), `ZURICH_INCOME_BRACKETS` (2026, § 35 Abs. 1 StG ZH Grundtarif, base rate: 7,000 / 12,000 / 16,800 / 24,800 / 34,500 / 45,700 / 58,800 / 76,400 / 110,400 / 144,100 / 197,400 / 266,700 at 2%…13%), `ZURICH_WEALTH_BRACKETS` (2026, § 47 StG ZH, base rate: 80,000 / 321,000 / 727,000 / 1,371,000 / 2,339,000 / 3,304,000 at 0.5‰…3‰), and the AHV non-worker table (530 CHF min / 26,500 CHF max, unchanged for 2026). The three tax tables were reverse-engineered from ~1,800 queries to the official ESTV calculator API (`swisstaxcalculator.estv.admin.ch`, Zurich City, single, no church) and are pinned by golden tests (`test_income_tax_matches_estv_2026`, `test_wealth_tax_matches_estv_2026`, `test_capital_withdrawal_tax_matches_estv_2026`); the model matches within CHF 1–2. Deliberately not modelled (each < CHF 25/yr and would break positive homogeneity): flooring of taxable amounts to CHF 100 / 1,000, the Art. 36 Abs. 3 DBG CHF 25 minimum, and the CHF 24 Zurich personal tax. These are the *base-year* tables; the simulation engine indexes them forward each year at runtime (see **Bracket Indexation** below) rather than storing pre-inflated copies. The 2026 Zurich City multipliers are defined once as `ZURICH_CANTONAL_MULTIPLIER = 0.95` and `ZURICH_CITY_MUNICIPAL_MULTIPLIER = 1.19` and used as the defaults of every tax function, `SimConfig`, and the UI.
*   **Scalar/array handling**: the private helpers `_is_scalar` / `_as_output` implement the "scalar in → float out, array in → array out" contract once for all public functions.

*   **Income Tax (`calculate_income_tax(taxable_income, cantonal_multiplier=0.95, municipal_multiplier=1.19)`)**:
    *   Applies Federal progressive brackets (Single tariff), capped at the constitutional maximum `min(tariff(x), 0.115 * x)` (Art. 128 BV) in `_federal_income_tariff`.
    *   Applies Zurich Cantonal progressive base brackets (Single tariff) multiplied by `cantonal_multiplier + municipal_multiplier` (e.g., `0.95 + 1.19 = 2.14` for Zurich City 2026).
*   **Wealth Tax (`calculate_wealth_tax(taxable_wealth, cantonal_multiplier=0.95, municipal_multiplier=1.19)`)**:
    *   Applies Zurich Cantonal progressive base wealth tax brackets multiplied by `cantonal_multiplier + municipal_multiplier`. Assessed on year-end liquid wealth after deducting the current year's living expenses (`max(0, total_liquid_end - current_expenses)`).
*   **Capital Withdrawal Tax (`calculate_capital_withdrawal_tax(amount, cantonal_multiplier=0.95, municipal_multiplier=1.19)`)**:
    *   Applied immediately at source when Pillar 2 or Pillar 3a accounts are liquidated.
    *   **Federal** (Art. 38 DBG): `federal_income_tariff(C) / 5` (inherits the 11.5% cap).
    *   **Zurich** (§ 37 StG ZH, since 2022): the rate for a notional annual pension of `C / 20` is applied to the full amount, with a 2% minimum simple tax: `simple = max(0.02 * C, 20 * T_ZH(C / 20))`, then `simple * (cantonal_multiplier + municipal_multiplier)`. (The previous `T_ZH(C) / 10` approximation understated the tax by ~2.6× at CHF 1M and ignored the 2% floor.)
    *   All-in Zurich City reference points: CHF 100k → 4.80% (floor binds), CHF 1M → ≈ 10.9%.
    *   Both parts remain positively homogeneous of degree 1 in the bracket edges, so the indexation identity below stays exact.
*   **AHV Non-Worker Contributions (`calculate_ahv_non_worker(wealth, imputed_pension_income=0)`)**:
    *   Calculated for early retirees (`current_age < 65`) based on the official 2025 AHV tables: `determining_wealth = wealth + 20 * imputed_pension_income` (where `wealth` is `taxable_wealth`). Contribution is 530 CHF for wealth < 350k CHF. For wealth between 350k and 1.75M CHF, it adds 106 CHF for every 50k CHF step above 300k CHF. For wealth above 1.75M CHF, it adds 159 CHF for every 50k CHF step above 1.75M CHF. Capped at 26,500 CHF/year (2025 limits).
*   **Bracket Indexation (`SimConfig.bracket_indexation`, `_indexed_tax` in `src/simulation_engine.py`)**:
    *   Swiss law indexes franc-denominated tax brackets to the CPI so that purely nominal growth does not push a taxpayer into higher brackets ("kalte Progression"). **Art. 39 DBG** requires the EFD to adjust the federal tariff steps and franc-denominated deductions **annually** to the LIK (index level at 30 June preceding the tax period, rounded to CHF 100), and **§ 48 StG ZH** indexes the Canton Zurich tariff steps and deductions for **both income and wealth tax**. The AHV non-worker scale tracks the Mischindex (mean of wage and price indices) roughly every two years, so CPI indexation of that table is a slight under-adjustment.
    *   `bracket_indexation` is the fraction of each year's realized inflation passed through to the bracket edges: `1.0` (default) reproduces the legal regime, `0.0` freezes the brackets in nominal terms so the retiree absorbs the full bracket creep, and intermediate values model delayed or partially suspended cantonal compensation. The engine compounds a dedicated `bracket_factors` array in lockstep with `inflation_factors`, using the same `np.maximum(0.01, ...)` deflation clamp.
    *   Indexation is applied via the homogeneity identity `T_indexed(x, f) == f * T_nominal(x / f)`, which is exact for every tariff in `tax_engine`: the income and wealth tariffs are piecewise linear, and the AHV step function scales correctly because floor division is scale invariant (`(f*a) // (f*b) == a // b`). This keeps `tax_engine` the single source of truth for the bracket tables rather than maintaining a second, inflated copy. Bracket *rates* and the Steuerfuss multipliers are deliberately **not** scaled, since both are political parameters.
    *   *Note: Indexation removes only the spurious, inflation-driven bracket creep. A portfolio compounding faster than the CPI still climbs the wealth tax schedule in real terms at any setting, which is economically correct and remains one of the dominant drags in the model.*
*   **Pillar 1 AHV Pension Payments**:
    *   Starting at age 65, the user receives a monthly AHV pension (`monthly_ahv_pension * inflation_factors`), added directly to CHF Cash each month. The input value (in today's CHF) compounds with annual CPI inflation from Year 0 (including all years prior to age 65).
    *   Annual AHV payouts (`12 * monthly_ahv_pension * inflation_factors`) are included in annual taxable income.

### 3.3. Simulation Engine (The Core Loop)
The engine (`run_simulation` in `src/simulation_engine.py`) validates that `return_matrix.shape == (num_runs, duration_years * 12, 5)` and (when provided) `inflation_matrix.shape == (num_runs, duration_years)`, and executes a **monthly tick** (`m in range(duration_years * 12)`) for `num_runs` paths simultaneously using NumPy arrays across 5 liquid asset classes (`0: US Stocks`, `1: Non-US Stocks`, `2: CHF Cash`, `3: Gold`, `4: Bitcoin`).

**Monthly Tick Execution Order:**
1. **Annual Inflation & Bracket Indexation Update (Month 0 of each year)**:
    * Updates cumulative `inflation_factors *= np.maximum(0.01, 1.0 + inflation)` using `inflation_matrix[:, year]` (or sampling from a deterministic `np.random.default_rng(config.seed + 10_000)` stream if `inflation_matrix` is `None`). Clamping `1.0 + inflation >= 0.01` prevents division-by-zero or sign inversion under extreme synthetic deflation draws.
    * Updates a parallel `bracket_factors *= np.maximum(0.01, 1.0 + config.bracket_indexation * inflation)`, which scales all franc-denominated tax bracket edges for the year (see §3.2). At `bracket_indexation = 1.0` this tracks `inflation_factors` exactly; at `0.0` it stays pinned at `1.0`.
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
    * **Cash Tent (`get_target_weights` & `_get_year_0_liquid_wealth`)**: Peak target cash weight at Year 0 is computed as `min(1.0, tent_duration_years * (annual_base_expenses + estimate_year_0_taxes(config)) / init_wealth)` (incorporating net Pillar 2/3a liquidations via `_get_year_0_liquid_wealth` for both `start_age >= 65` and `60 <= start_age < 65`), and glides linearly down to `alloc_chf_cash` over `tent_duration_years` (default: 7 years). Non-cash weights are rescaled by `(1 - cash_weight) / (1 - alloc_chf_cash)` so the vector always sums to 1.0. The glidepath is bypassed and flat `base_weights` are returned whenever `tent_duration_years <= 0`, `year >= tent_duration_years`, `init_wealth <= 0`, `alloc_chf_cash >= 1.0`, or the computed peak weight does not exceed the baseline cash weight (`w_peak <= alloc_chf_cash`).
    * **Per-year precompute & timing**: `run_simulation` evaluates `get_target_weights` (and thus the Year 0 tax estimate) **once per year** before the monthly loop (`target_weights_by_year`, one row per simulated year) instead of every month. `current_target_weights = target_weights_by_year[year]` drives pension-proceeds allocation and intra-year rebalancing, while the Month-11 (year-end) rebalance targets `target_weights_by_year[year + 1]` (the final year keeps its own weights), so the allocation *held during* year *y* is the glidepath weight for year *y* and the base weights are reached after exactly `tent_duration_years` (previously the glidepath lagged one year).
    * **Smart Cash Buffer**: If `enable_smart_selling` is enabled, rebalancing is skipped for any run currently in a market downturn (`current_nw < initial_net_worth * inflation_factors`).
6. **Annual Taxation, Spending & Outflow Deduction (Month 11 of each year)**:
    * **Spending Evaluation**:
        * **Static**: `current_expenses = annual_base_expenses * inflation_factors`.
        * **Dynamic (Floor & Ceiling)**: Scales inflation-adjusted base expenses by `dynamic_expense_floor_pct` when net worth is below the inflation-adjusted starting watermark, and by `dynamic_expense_ceiling_pct` when above.
        * **Vanguard Dynamic**: Recalculates target net living expenses as `vanguard_target_rate * max(0, current_nw_before_expenses)`, clamped between `(1 - vanguard_floor_pct)` and `(1 + vanguard_ceiling_pct)` of the prior year's inflation-adjusted living expenses. In `app.py`, the UI's tax-inclusive Target Withdrawal Rate ($\text{TWR}$) is bi-directionally synchronized with Year 0 Annual Base Expenses ($E_0$) via fixed-point iteration solving $E_0 + \text{Taxes}_0(E_0) = \text{TWR} \times \text{NetWorth}_0$ using live user tax/yield inputs, and converted to the net living expense rate `vanguard_target_rate = E_0 / NetWorth_0` for `SimConfig` so taxes are never double-counted.
    * **Tax Assessment**: Computes annual `income_tax` on `dividends` (`(US Stocks + Non-US Stocks) * dividend_yield`) + `interest` (`CHF Cash * np.maximum(0.0, realized_cash_ret)`, where `realized_cash_ret = np.prod(1.0 + return_matrix[:, (m - 11):(m + 1), 2], axis=1) - 1.0`) + `annual_ahv_received`, plus `wealth_tax` and `ahv_contrib` (for `current_age < 65`) on `taxable_wealth = max(0, total_liquid_end - current_expenses)`. In `estimate_year_0_taxes(config)`, Year 0 cash interest uses `max(0.0, config.cash_rate)`.
    * **Asset Selling**: Deducts `deficit = current_expenses + total_taxes`. If `enable_smart_selling` is enabled and the run is in a downturn, positive CHF Cash is spent down first before selling other positive assets proportionally to their current holdings. Otherwise, all positive liquid assets are sold proportionally to their current holdings; any unpaid deficit beyond total liquid assets becomes negative CHF Cash.

### 3.4. Return Generators & Data Provenance
The system provides a two-stage reproducible data pipeline (`scripts/fetch_source_data.py` $\to$ `data/*.csv` $\to$ `scripts/build_historic_returns.py` $\to$ `src/historic_returns.py`) covering 1922–2025 (`104` years, `1,248` monthly observations):
*   **Empirical Source Datasets (`data/`)**:
    *   `us_stocks.csv`: True **month-end** S&P 500 Total Return index in USD (1871–2025, via [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator); verified to $\pm 0.01\text{pp}$ against published month-end S&P 500 total returns). Carried as **real monthly returns** (`MONTHLY_US_CHF`) throughout 1922–2025.
    *   `jst_exus_usd.csv` + `ex_us_stocks.csv`: Spliced Non-US Equities total return series. Because `ex_us_stocks.csv` is byte-identical to `us_stocks.csv` prior to Dec 1969, **1922–1969** uses a GDP-weighted 17-country ex-US total return index constructed from the **Jordà–Schularick–Taylor Macrohistory Database R6** (`eq_tr` converted to USD via `xrusd`, then to CHF; annual returns expanded into 12 geometric monthly returns, flagged in `MONTHLY_EXUS_IS_SMOOTHED`), spliced with **real monthly** MSCI EAFE / World ex-US returns (`ex_us_stocks.csv`) for **1970–2025** (`MONTHLY_NON_US_CHF`).
    *   `usd_chf.csv`: True **month-end** USD/CHF exchange rate from **FRED `DEXSZUS`** daily observations for **1971–present** (replacing a legacy hand-extended 2024–2025 tail that overstated Dec 2025 USD/CHF by 13.4%), spliced with **SNB `devkum`** monthly averages for **1914–1970** (Bretton Woods / gold-standard peg era).
    *   `ch_inflation.csv`: Historical monthly Swiss Consumer Price Index (CPI / LIK) from the Swiss Federal Statistical Office (FSO/BFS, 1921–present).
    *   `gold_usd.csv`: **LBMA London PM Fix** month-end USD/oz prices from **April 1968–present** multiplied by `usd_chf.csv` to yield **real monthly CHF gold returns** (`MONTHLY_GOLD_CHF`), with pre-1968 statutory fixed USD prices (`$20.67` then `$35.00`) reflecting currency devaluations smoothed across the year (`MONTHLY_GOLD_IS_SMOOTHED`).
    *   `ch_cash_rate.csv`: Empirical Swiss short-term interest rate (`1900–2025`), splicing the **Jordà–Schularick–Taylor Macrohistory Database R6** (`bill_rate` for Switzerland, 1900–2020) with **SNB policy / SARON rates** (2021–2025), floored at `0.0%` (`retail_rate`) to reflect Swiss retail savings deposit pass-through during the 2015–2022 SNB negative-interest-rate era (`MONTHLY_CASH_CHF`, `HISTORIC_RETURNS_CASH_CHF`, 1922–2025: `2.46%` nominal CAGR, `1.88%` vol, `+0.71%` real after `1.74%` Swiss CPI). Being wholesale-based, it is an optimistic proxy for retail savings rates.
    *   `us_cpi.csv`: Monthly US CPI-U (1871–2025, via swr-calculator). **Build-time only** — consumed by `scripts/build_historic_returns.py` to derive the PPP constant; never loaded by the simulation.
*   **Build-time integrity checks**: the Gold series is asserted to be gap-free across the window (no silent `fillna(0.0)`), and `HISTORIC_REAL_CHF_APPRECIATION` is recomputed on every build from the geometric 1922–2025 drifts: $g_{\text{FX}}$ (USD/CHF) and $g_{\text{PPP}} = \frac{1 + \pi_{\text{CH}}}{1 + \pi_{\text{US}}} - 1$, giving $a_{\text{hist}} = \text{round}\left(1 - \frac{1 + g_{\text{FX}}}{1 + g_{\text{PPP}}}, 4\right)$ = `0.0070`. `tests/test_simulation_engine.py::test_historic_real_chf_appreciation_is_derived_from_data` recomputes it from the CSVs.
*   **Currency / PPP Adjustment (`_apply_fx_ppp_adjustment`, `HISTORIC_REAL_CHF_APPRECIATION = 0.0070`)**: The CHF has appreciated ≈ `0.70%/yr` in real terms beyond Relative PPP over 1922–2025, i.e. a real FX drag on foreign-priced assets. When `real_chf_appreciation` is passed (UI default `0.0`), foreign-priced sleeves (`ASSET_US_STOCKS`, `ASSET_NON_US_STOCKS`, `ASSET_GOLD`) are multiplicatively scaled each month by $\text{month\_adj} = \left(\frac{1 - \text{real\_chf\_appreciation}}{1 - a_{\text{hist}}}\right)^{1/12}$ via $r_m^{\text{adj}} = (1 + r_m)\cdot\text{month\_adj} - 1$.
*   **Gaussian Copula Bitcoin–Equity Coupling (`_generate_lognormal_monthly_returns`, `BITCOIN_NOMINAL_MEAN = 0.07`, `BITCOIN_VOL = 0.50`, `DEFAULT_BTC_EQUITY_CORR = 0.50`)**: Per-path standardized monthly log-returns of the US equity series ($Z_{\text{US}, t} = (\ln(1 + R_{\text{US}, t}) - \hat{\mu}_{\text{path}})/\hat{\sigma}_{\text{path}}$ along `axis=-1`, ensuring $\text{mean}(Z_{\text{US}}) = 0$ and $\text{std}(Z_{\text{US}}) = 1$ within each cohort/run so US cohort drift never leaks into Bitcoin's expected log-return) are combined with an independent standard normal draw $\varepsilon_t \sim \mathcal{N}(0, 1)$ via $Z_{\text{BTC}, t} = \rho Z_{\text{US}, t} + \sqrt{1 - \rho^2}\varepsilon_t$ before applying the Ito-corrected lognormal map (`btc_mean`, default `7%`, and `btc_vol`, default `50%`). This couples Bitcoin drawdowns to equity market crashes (`1929`, `1973–74`, `2008`, `2022`) across all 3 engines.
*   **Historic Backtesting Mode (`get_historic_return_matrix`, `get_historic_inflation_matrix`, `compute_effective_sample_size`)**: Replays contiguous historical monthly returns (`MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`, adjusted for `real_chf_appreciation`) and annual Swiss CPI inflation (`HISTORIC_SWISS_INFLATION`) across 1922–2025 (`104` years, $N = \text{total\_years} - \text{duration\_years} + 1$ cohorts), paired with Gaussian-copula Bitcoin returns (`btc_mean`, `btc_vol`, `btc_equity_corr`). `compute_effective_sample_size` calculates the non-overlapping sample size $N_{\text{eff}} = T / D$ and the two-sided 90% Wilson confidence interval ($z = 1.64485$).
*   **Historic Bootstrapping Mode (`generate_bootstrapped_data`, `POLITIS_WHITE_BLOCK_YEARS = 5`)**: Implements the **Politis–Romano (1994) Stationary Bootstrap** (`stationary=True` default) with **circular wrap-around** `(year_idx[:, t - 1] + 1) % total_years` and geometric block lengths of mean $b = \text{block\_size\_years}$ (at each year $t > 0$, starting a new random year with probability $p = 1/b$ and continuing the contiguous historical sequence with probability $1 - 1/b$). Setting `stationary=False` falls back to fixed non-circular `block_size_years` block sampling. Jointly resamples `MONTHLY_US_CHF`, `MONTHLY_NON_US_CHF`, `MONTHLY_CASH_CHF`, `MONTHLY_GOLD_CHF`, and `HISTORIC_SWISS_INFLATION` with Gaussian-copula Bitcoin returns (`btc_mean`, `btc_vol`, `btc_equity_corr`).
*   **Parametric Monte Carlo Mode (`generate_monte_carlo_returns`, `generate_monte_carlo_inflation`)**: Generates an asset return matrix of `shape=(num_runs, duration_months, 5)` seeded by `random_seed` using a lognormal model with Ito drift correction ($\text{drift} = \ln(1 + R) - 0.5\sigma^2$) and correlated monthly shocks (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$, `Bitcoin` $\rho=\text{btc\_equity\_corr}$) for US Stocks, Non-US Stocks, Gold, and Bitcoin, constant geometric monthly return $(1 + R_{\text{cash}})^{1/12} - 1$ for CHF Cash, and independent normally distributed annual inflation `generate_monte_carlo_inflation` seeded at `random_seed + 10_000`.

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
    initial_pillar_3a_accounts: list[float] = field(default_factory=list)
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
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER # 0.95 (Canton Zurich 2026)
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER # 1.19 (City of Zurich)
    bracket_indexation: float = 1.0 # Fraction of CPI applied to bracket edges (1.0 = full, per Art. 39 DBG / § 48 StG ZH)
    tent_duration_years: int = 7
    real_chf_appreciation: float = 0.0 # Expected real CHF appreciation beyond PPP (0.0 = PPP neutrality; HISTORIC_REAL_CHF_APPRECIATION ~0.0070 = raw 1922-2025 history)
    cash_rate: float = 0.01 # Expected CHF cash rate used for Year 0 tax pre-estimates (actual realized return used in simulation loop)
    btc_equity_corr: float = 0.50 # Monthly log-return correlation between synthetic Bitcoin and US equities (-1.0 to 1.0)
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
*   **Sidebar**: All inputs categorized into 11 dedicated sections: **Demographics**, **Success Criteria**, **Initial Assets (CHF)** (Taxable Liquid Wealth, Pillar 2, Number of Pillar 3a Accounts, and a single equal Balance per Pillar 3a Account), **Target Asset Allocation (%)**, **Rebalancing Strategy**, **Withdrawal Strategy**, **Spending Strategy**, **Income & Yield**, **Monte Carlo Parameters** (including **Bitcoin–Equity Correlation ($\rho$)** slider default `0.50`, **Stationary Bootstrap Mean Block Length (Years)** slider default `5`, run counts, and **Random Seed**), **Currency / PPP Assumption** (**Real CHF Appreciation beyond PPP (%/yr)** slider, default `0.00%`), and **Zurich Tax Location** (Municipal Multiplier and **Tax Bracket Indexation (% of CPI)**, default `100`).
*   **Main Panel**: Displays results across three side-by-side columns for direct comparative analysis: **Historic Backtesting**, **Historic Bootstrapping**, and **Monte Carlo**. Each column contains mouse-over header tooltips and:
    *   **TL;DR Status**: Displays 'BROKE', 'RICH', or 'DEAD' via `metrics.classify_outcome`: BROKE if the median final net worth is ≤ 0, RICH if the median of per-run *real* final net worth (each run deflated by its own cumulative inflation) is ≥ `RICH_MULTIPLE` (3) × initial net worth, DEAD otherwise.
    *   **Metrics**: Probability of Success (based on selected success criteria, with **Historic Backtesting** displaying a caption and tooltip with $N_{\text{cohorts}}$, **Effective Sample Size ($N_{\text{eff}} = T / D$)**, and the **90% Wilson confidence interval**, while **Historic Bootstrapping** and **Monte Carlo** reserve an identical `visibility: hidden` layout box so all 3 columns stay pixel-aligned vertically), Avg Years Below Start NW, Median Ending Net Worth (Real & Nominal), Median Total Withdrawals (Real & Nominal), Pre-AHV Outflow (< Age 65), and Post-65 Outflow (Age 65+).
    *   **Net Worth Trajectory Chart** (Plotly): Faint lines for individual runs (all cohorts for Historic Backtesting; capped at 100 runs for Monte Carlo and Bootstrapping), bold lines for **Worst Cohort (Min), 25th, 50th, 75th, Best Cohort (Max)** (`[0, 25, 50, 75, 100]`) in **Historic Backtesting** and **5th, 25th, 50th, 75th, 95th** percentiles (`[5, 25, 50, 75, 95]`) in **Historic Bootstrapping** and **Monte Carlo**, plus a dashed reference line for Inflation-Adj Start NW with horizontal bottom legend.
    *   **Income vs Required Cash Chart** (Plotly): Stacked bar chart showing median Dividends, AHV Pension, and Capital Sold, with a dashed reference line for Total Cash Needed (Expenses + Taxes) and bottom legend.
    *   **Annual Withdrawal Breakdown Chart** (Plotly): Stacked bar chart showing median living expenses and taxes paid over time with bottom legend.
    *   **Withdrawal Rate Chart** (Plotly): Percentile/extrema lines (`[0, 25, 50, 75, 100]` for Historic Backtesting; `[5, 25, 50, 75, 95]` for Bootstrapping and Monte Carlo) for `(expenses_paid + taxes_paid) / max(1.0, net_worth + expenses_paid + taxes_paid)` — i.e. the outflow as a share of beginning-of-year net worth — over time (dynamically capped at 25% max) with bottom legend.
    *   **Asset Allocation Development Chart** (Plotly): Stacked area chart showing the median nominal balance of all asset categories (CHF Cash, US Stocks, Non-US Stocks, Gold, Bitcoin, Pillar 3a, and Pillar 2) over time to visualize portfolio glidepaths, rebalancing, and account liquidations with bottom legend.
    *   **Cohort / Run Analysis Tables**: Interactive tables of Top 10 Best and Worst cohorts/runs based on final net worth, displaying `Cohort`/`Run`, `Final NW (Real)`, `Final NW (Nom)`, `Min NW (Nom)`, and `Yrs Below` (years below the inflation-adjusted starting net worth watermark).

## 6. Implementation Plan & Milestones
1.  **Setup**: Initialize Git repo, project structure, and `requirements.txt` (Streamlit, Pandas, NumPy, Plotly, Pytest).
2.  **Tax Engine**: Implement and unit test the Zurich/Federal tax brackets and AHV non-worker logic.
3.  **Simulation Loop**: Build the vectorized monthly simulation engine with Cash Tent, Smart Cash Buffer, and dynamic spending rules.
4.  **UI Integration**: Build the 3-column comparative Streamlit frontend and connect the simulation engine.
5.  **Return Data**: Integrate the 1922–2025 historical Swiss dataset, bootstrapper, and lognormal Monte Carlo generator.

