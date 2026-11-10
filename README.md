# Early Retirement Simulator (Zurich · Austria)

This application is a financial simulation tool designed for individuals planning for early retirement (FIRE) in **Canton Zurich, Switzerland** or in **Austria** (selected with the **Country** box at the top of the sidebar).

Phase 1 focuses on the **post-retirement decumulation phase** for a single individual with no dependents.

## Jurisdictions & Implementation Status

Both countries are implemented in one app (milestones M1–M6); pick the country in the sidebar. Programmatically: `run_simulation(SimConfig(country="AT", ...), ...)` with the EUR return module. The [Features](#features) section below describes the Swiss model; the Austrian differences are summarised here and specified in PRD §8. Austrian pension taxation is verified against the official BMF Brutto-Netto-Rechner (within €0.13 for pensions from €1,100 to €95,000 a month).

Legend: ✅ implemented · 🔁 country-agnostic, reused unchanged · ❌ not implemented

| Component | 🇨🇭 Switzerland (Zurich) | 🇦🇹 Austria |
| :--- | :--- | :--- |
| Country dispatch (`SimConfig.country`, `src/jurisdiction.py`, `COUNTRY_DEFAULTS`) | ✅ (default `"CH"`) | ✅ (`country="AT"`) |
| Country selector in the sidebar | ✅ | ✅ (title, labels, inputs and defaults switch; Swiss-data warning banner) |
| Return engines (Backtesting, Stationary Bootstrap, Monte Carlo), Bitcoin copula | ✅ | 🔁 engines |
| Historic return data 1922–2025 | ✅ `src/historic_returns.py` (CHF) | ✅ `src/historic_returns_eur.py` (EUR; **1922–1957 Swiss inflation and cash data, 1958–1999 cash PPP-translated from Swiss data**, see below) |
| Spending strategies, rebalancing, Cash Tent, Smart Cash Buffer | ✅ | 🔁 (✅ Austrian Year-0 tax estimate; sales and rebalancing realize taxable gains) |
| Metrics, charts, cohort tables, success criterion | ✅ (tax-component medians on hover) | 🔁 (✅ EUR / State Pension labels, no Pillar or Liquid Solvent series, KESt / income tax / health insurance on hover) |
| Income tax | ✅ Federal + Zurich 2026, ESTV-verified | ✅ § 33 EStG 2026 tariff, pension taxation (6% health contribution capped at the contribution ceiling, 13th/14th payment under § 67 incl. the €2,615 Freigrenze, pensioner credit), BMF-verified |
| Wealth tax | ✅ Zurich | n/a (no wealth tax) |
| Cost-basis tracking (average cost, basis = value on day 0; `realized_gains` / `unrealized_gains` outputs) | ✅ tracked, not taxed | ✅ (engine, shared) |
| Capital gains tax | n/a (tax-free) | ✅ 27.5% KESt on stocks, gold ETC and Bitcoin, in-year loss offset, Regelbesteuerungsoption (the year-end sale is solved for the tax on its own gains) |
| Dividend / interest tax | ✅ progressive | ✅ 27.5% (incl. deemed distributions) / 25% on bank interest |
| Mandatory contributions before 65 | ✅ AHV non-worker | ✅ Health self-insurance (§ 16 ASVG, €565.25/month in 2026, CPI-indexed; for life if no pension) |
| State pension (single input from 65) | ✅ AHV (monthly, 12×/yr) | ✅ monthly gross, 14×/yr |
| Pension lump sums | ✅ Pillar 2 / 3a with capital withdrawal tax | n/a (out of scope) |
| Tax tests | ✅ golden tests vs. the ESTV calculator | ✅ golden tests vs. the BMF Brutto-Netto-Rechner (pensions) + hand-computed statutory examples (capital income) |

Requirements, agreed decisions (all design questions resolved): [PRD §8](docs/prd_early_retirement_calc.md#8-austria-jurisdiction-country-switch). Technical design and milestones: [Design Doc §7](docs/design_doc_early_retirement.md#7-multi-jurisdiction-design-austria).

> [!WARNING]
> **Austrian history before 1958 is Swiss data, and Austrian cash returns before 2000 are derived from Swiss data.** Austria has no usable inflation or cash-rate series for 1922–1957 (1921–22 hyperinflation, 1938 annexation, 1945–47 currency reforms). For those 36 years `src/historic_returns_eur.py` uses the **Swiss CPI** ([`data/ch_inflation.csv`](data/ch_inflation.csv)) as Austrian inflation and the **Swiss retail savings rate** ([`data/ch_cash_rate.csv`](data/ch_cash_rate.csv)) as the EUR Cash return, and translates USD equity and gold returns at a PPP-implied rather than a market exchange rate. For **1958–1999** no open Austrian retail savings-rate series exists either, so EUR Cash is the **Swiss savings rate translated to EUR under relative PPP** (Swiss real cash return + Austrian inflation). Real Austrian data starts in 1958 for exchange rates and inflation (BIS, Statistik Austria) and in 2000 for cash (ECB). The module flags both segments (`HISTORIC_YEAR_IS_SWISS_PROXY`, `HISTORIC_YEAR_CASH_IS_SWISS_PROXY`).

## Features

*   **Three Comparative Simulation Methods**: Compare outcomes side-by-side using:
    1. **Historic Backtesting**: Replay contiguous historical periods (1922–2025, 104 years of Swiss-adjusted market, cash, gold, and CPI data, paired with Ito-corrected lognormal synthetic Bitcoin returns coupled to US equity shocks). Surfaces **Effective Sample Size ($N_{\text{eff}} = T / D$)** and the **90% Wilson confidence interval** on the cohort survival rate (reserving matching vertical space in the adjacent columns to keep all 3 columns aligned), and plots **Worst Cohort (Min) / Best Cohort (Max)** bounds (`[0, 25, 50, 75, 100]`; reversed to **Best Cohort (Min) / Worst Cohort (Max)** on the Withdrawal Rate chart where lower rates are favorable) rather than unsupported 5th/95th percentiles.
    2. **Historic Bootstrapping (Politis–Romano Stationary Bootstrap)**: Joint stationary block sampling (`stationary=True`, mean block length `block_size_years = 5` via geometric restart probability $p = 1/b$) with **circular wrap-around** (`year 2025 -> year 1922`) across historical equity, gold, cash, and Swiss CPI series, preserving multi-year volatility clusters while ensuring uniform sampling of every historical year (`1922` and `2025` have identical inclusion probability $1/104$).
    3. **Parametric Monte Carlo**: Stochastic lognormal return generator (with Ito drift correction) with cross-asset shock coupling (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$ — the 1922–2025 annual US/gold correlation — and `Bitcoin` $\rho=\text{btc\_equity\_corr}$, default `0.50`) and independent annual inflation stream (`seed + 10_000`). Stock means, volatilities, gold and inflation defaults are calibrated at runtime to the 1922–2025 CHF history (log-vols equity ≈ 21%, gold ≈ 15.3%; Swiss CPI inflation 1.8% ± 3.6%). The **US, Non-US and Gold means are calibrated on the history rescaled to the sidebar's Currency / PPP Assumption** (`ppp_adjusted_annual_returns`), so all three engines share one FX assumption: at the default `0.00%` (PPP neutrality) the arithmetic means are US ≈ 12.1%, ex-US ≈ 9.3%, gold ≈ 5.5%; at `+0.71%` (raw history) they are US ≈ 11.3%, ex-US ≈ 8.6%, gold ≈ 4.7%. Only CHF cash (1%) stays forward-looking. The **Monte Carlo Parameters** subheader tooltip summarizes the nominal/CHF arithmetic-mean parameterization and historic CAGRs, and every input's help text shows its historic value.
*   **Swiss Tax Modeling**: Models Federal, Cantonal (95% Steuerfuss), and Municipal (e.g., 119% Zurich City) income and wealth taxes for Canton Zurich using the **2026 tariffs, verified against the official ESTV tax calculator** (income, wealth and capital-withdrawal taxes match within CHF 1–2; pinned by golden tests) — including the constitutional **11.5% ceiling on federal income tax** (Art. 128 BV) — and **dynamic cash interest taxation** (taxable dividends and interest are computed on the holdings held during the year *before* year-end rebalancing, using the actual realized annual return on CHF Cash rather than a hardcoded `1%`, and `estimate_year_0_taxes` accounts for Year-0 Cash Tent weights). The same Zurich City multipliers are the defaults of both the tax functions and `SimConfig`.
*   **Tax Bracket Indexation**: Bracket edges and the AHV contribution table are indexed to realized CPI each year, as Swiss law requires (Art. 39 DBG federally, § 48 StG ZH for Zurich income *and* wealth tax). Configurable from 0–100% of CPI to stress-test delayed or suspended compensation — freezing brackets at 0% overstates median lifetime real taxes by roughly a third over a 50-year horizon.
*   **AHV for Non-Workers**: Models mandatory AHV/IV/EO contributions for early retirees before age 65 based on wealth (`530` to `26,500 CHF/yr`), and deducts them from annual taxable income as allowed under federal and Zurich tax law (Art. 33 Abs. 1 lit. d DBG / § 31 Abs. 1 lit. d StG ZH).
*   **Pillar 2 & 3a Liquidations**: Simulates tax-sheltered equity growth and staggered start-of-year lump-sum withdrawals of Pillar 3a accounts (up to 5 equal-balance accounts between ages 60–64, configured via account count and a single balance per account input) and Pillar 2 vesting accounts (at age 65, or Year 0 if retiring at $\ge 65$, displayed as a combined **Pillar 2 & 3a** sleeve in the Asset Allocation Development chart), with the separate **capital withdrawal tax** deducted at source (paying off any prior negative CHF Cash bridge debt first before investing net surplus across target weights so solvent portfolios never carry involuntary margin debt): federal at 1/5 of the ordinary tariff (Art. 38 DBG) plus Zurich at the rate for a notional annual pension of 1/20 of the lump sum with a **2% minimum simple tax** (§ 37 StG ZH, since 2022) — about 4.8% all-in on CHF 100k and 10.9% on CHF 1M in Zurich City.
*   **Rebalancing & Pfau Cash Tent Glidepath**: Supports Cash Tent (rising equity glidepath from a multi-year cash buffer of expenses + estimated taxes; the allocation held during year *y* is the glidepath weight for year *y*, reaching the base weights after exactly `tent_duration_years`), Monthly, Quarterly, Yearly, Threshold-based, and Never rebalancing strategies.
*   **Smart Cash Buffer**: Optional defensive strategy during market downturns (net worth below inflation-adjusted starting watermark) to pause rebalancing and spend CHF Cash first before selling equities. The watermark comparison (also used by Dynamic Floor & Ceiling spending and "Years Below") is **after tax**: both current and starting net worth deduct the capital withdrawal tax still owed on Pillar 2 / 3a balances under the simulator's own payout schedule, so paying that tax at 60–65 is not mistaken for a market downturn.
*   **Flexible Spending Strategies**: Choose between **Static** (inflation-adjusted base expenses), **Dynamic (Floor & Ceiling)**, and **Vanguard Dynamic** spending (tax-inclusive UI Target Withdrawal Rate bi-directionally synchronized with Year 0 base expenses and estimated taxes using live tax/yield/Cash-Tent inputs), with the **Income vs Required Cash** (Dividends, Cash Interest, AHV Pension, Capital Sold) and **Annual Withdrawal Breakdown** charts comparing median (`50th Pct` stacked bars) and bad-case (**Worst Cohort** in Historic Backtesting, **5th Pct** outcome path in Bootstrapping & Monte Carlo, funded outflows re-expressed on the median CPI scale) withdrawals against the **Inflation-Adj Start** withdrawal trajectory, and a **Withdrawal Rate** chart showing net portfolio withdrawals (`max(0, expenses + taxes - AHV)`) relative to actual beginning-of-year net worth.
*   **Configurable Success Criteria & Failure Timing Analysis**: Set target ending net worth (e.g., preserve 50% of inflation-adjusted starting wealth; success requires both never depleting `NW ≤ 0` during retirement and finishing above target, so `Success + Depletion + Final Shortfall = 100%`) and distinguish early-retirement ruin from late-life depletion or target shortfall across a unified retirement-age axis (`start_age .. start_age + duration_years - 1`) via:
    *   **Depletion Rate (`NW ≤ 0`) vs. Target Shortfall (`0 < Final Real NW ≤ Target`)** and **Pre-AHV Liquid Depletion (`Liquid ≤ 0` before Age 65)**.
    *   **Depletion Rate (`NW ≤ 0`) & Depletion Age (Min / Med)** paired with **Shortfall Rate (`Real NW ≤ Target`) & Shortfall Age (Min / Med)** KPIs tracking earliest and median timing when portfolios exhaust wealth or first breach target ending wealth.
    *   **Portfolio Survival by Age** (Kaplan–Meier style curve plotting `Solvent (NW > 0)`, `Liquid Solvent (Liquid > 0)`, and `Above Target` across ages, with an `Age 65 (AHV + P2)` regime marker).
    *   **Top 10 Best/Worst Cohort & Run tables** with `Outcome`, `Depletion Age`, `Yrs in Shortfall`, `Shortfall Age`, `Final NW (Real/Nom)`, `Min NW (Real)` and `Yrs Below` columns (interactive `column_config` tooltips on every header). Best paths are sorted by `Final NW (Real)` descending. Worst paths are ranked by **outcome severity** relative to the configurable success criterion: **Depleted** (earliest depletion first) → **Failed** (survived but ended at/below the criterion; lowest real ending NW first) → **Recovered** (breached the criterion during retirement but ended above it; most years in shortfall first) → **Never breached** (lowest real ending NW first). The sort order is stated in a caption above each table (Streamlit has no API to preset the interactive header sort arrow).

## Data Sources & Provenance

This project incorporates historical datasets from primary macroeconomic sources and [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator):

1. **US Stocks (USD)** — [`data/us_stocks.csv`](data/us_stocks.csv), 1871–2025, monthly.
   A **month-end S&P 500 total-return index** (dividends reinvested).
   *Verified* against published month-end S&P 500 total returns (Oct 1929 −19.71%, Oct 1987 −21.53%, Oct 2008 −16.80%, Mar 2020 −12.35%) and against published calendar-year total returns for 1974, 1995, 2000, 2008, 2011, 2013, 2017–19, 2021–22 (agreement to ±0.00pp).
2. **Non-US Equities (USD)** — [`data/jst_exus_usd.csv`](data/jst_exus_usd.csv) (1900–1969, annual) spliced with [`data/ex_us_stocks.csv`](data/ex_us_stocks.csv) (1871–2025 file; real monthly 1970–2025 leg ingested).
   * **1922–1969**: GDP-weighted 17-country total-return index built from the **Jordà–Schularick–Taylor Macrohistory Database R6** (`eq_tr`, nominal local currency, converted to USD via `xrusd`), replacing the pre-1970 US duplicate in `ex_us_stocks.csv`.
   * **1970–2025**: MSCI EAFE / World ex-US monthly total return index in USD.
3. **Currency Exchange (USD/CHF)** — [`data/usd_chf.csv`](data/usd_chf.csv), 1914–2025, monthly.
   * **1914–1970**: SNB historical exchange rate series (Bretton Woods fixed parity era).
   * **1971–2025**: True **month-end** rates rebuilt from the Federal Reserve H.10 daily series (`DEXSZUS`), capturing the 2024–2025 USD decline to `0.7936 CHF/USD` at year-end 2025.
4. **Swiss Inflation (CPI)** — [`data/ch_inflation.csv`](data/ch_inflation.csv), 1921–2025, monthly.
   Swiss Consumer Price Index via the [Swiss Federal Statistical Office (FSO/BFS)](https://www.bfs.admin.ch).
5. **Gold (USD -> CHF)** — [`data/gold_usd.csv`](data/gold_usd.csv), 1871–2025, monthly.
   Month-end [LBMA](https://prices.lbma.org.uk) London PM fix from Apr 1968; statutory fixed price (\$20.67, then \$35) before that, converted to CHF via `usd_chf.csv`.
6. **Swiss Retail Cash Rate (CHF)** — [`data/ch_cash_rate.csv`](data/ch_cash_rate.csv), 1900–2025, annual (`retail_rate` drives the simulation; `wholesale_rate` kept for reference).
   `retail_rate` is the **SNB interest rate on private-client savings deposits** ([data.snb.ch](https://data.snb.ch) cube `zikrepro`, `D1=S1`, annual mean of monthly observations) from 1933, and the **Jordà–Schularick–Taylor Macrohistory Database R6** Swiss `bill_rate` for 1900–1932 (on the 1933–1968 overlap the two agree to 0.007pp — JST itself uses the savings rate there). From 1969 JST switches to a volatile money-market rate (e.g. 1989: 9.7% vs 3.45% on savings) and the 2015–2022 SNB policy rate was negative while savings paid ≈ 0.0x%, so the retail series needs no artificial 0% floor (1922–2025: `2.49%` nominal CAGR, `1.42%` vol, `+0.74%` real return after `1.74%` Swiss CPI). `wholesale_rate` = JST `bill_rate` 1900–2020 + SNB policy / SARON 2021–2025. Over 1933–2025 the savings rate averaged 2.32% vs 2.30% for the previous wholesale-based proxy — the old proxy was not optimistic on average, but was far more volatile.
7. **US Inflation (CPI-U)** — [`data/us_cpi.csv`](data/us_cpi.csv), 1871–2025, monthly: **FRED `CPIAUCNS`** (BLS CPI-U, not seasonally adjusted) from 1913, legacy [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator) leg for 1871–1912 (the legacy series deviated from BLS by up to 0.74% on the overlap).
   Used **only** by `scripts/build_historic_returns.py` to derive the relative-PPP drift behind `HISTORIC_REAL_CHF_APPRECIATION`; it never enters the simulation directly.

### Return construction

**Real monthly returns are carried end-to-end.** Monthly data is used wherever it genuinely exists, preserving intra-year volatility and monthly sequence-of-returns risk.

CHF conversion is applied at the index level before differencing:
$$\text{Return}^{\text{CHF}}_t = \frac{I^{\text{USD}}_t \times \text{FX}_t}{I^{\text{USD}}_{t-1} \times \text{FX}_{t-1}} - 1$$

*   **Currency / PPP Assumption (`real_chf_appreciation`, default `0.00%/yr`)**: `HISTORIC_REAL_CHF_APPRECIATION` is **derived from the data at build time** (not hardcoded): over 1922–2025 the geometric `USD/CHF` drift and the relative-PPP drift implied by Swiss vs US CPI give $a_{\text{hist}} = 1 - \frac{1 + g_{\text{FX}}}{1 + g_{\text{PPP}}} \approx$ `+0.71%/yr` real CHF appreciation beyond PPP (a real FX drag on foreign-priced assets). The sidebar exposes **Real CHF Appreciation beyond PPP (%/yr)** defaulting to `0.00%` (PPP neutrality, scaling foreign-priced monthly returns by $\left(\frac{1 - a_{\text{target}}}{1 - a_{\text{hist}}}\right)^{1/12}$), while `+0.71%` reproduces the raw unadjusted 1922–2025 historical CHF returns. The same assumption drives the **Monte Carlo default means** for US, Non-US and Gold via the annual counterpart `ppp_adjusted_annual_returns(annual, a_target) = (1 + R)\,\frac{1 - a_{\text{target}}}{1 - a_{\text{hist}}} - 1` (log-volatility is invariant), so the slider is rendered *above* the Monte Carlo section and the three engines never embed different FX drifts.
*   **Bitcoin–Equity Correlation Coupling (`btc_equity_corr`, default `0.50`)**: Because Bitcoin has no 40-year history, its monthly log-returns ($\mu = 7.0\%$, $\sigma = 50.0\%$ default across all 3 engines, Ito-corrected and user-configurable via `btc_mean` / `btc_vol`) are coupled to **full-sample standardized** monthly US equity log-returns via a Gaussian copula:
    $$Z_{\text{BTC}, t} = \rho_{\text{BTC,EQ}} Z_{\text{US}, t} + \sqrt{1 - \rho_{\text{BTC,EQ}}^2}\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$
    Full-sample standardization ($\text{mean}(Z_{\text{US}}) = 0$, $\text{std}(Z_{\text{US}}) = 1$ over 1922–2025 `MONTHLY_US_CHF`) ensures secular US equity drift never leaks into Bitcoin's unconditional expected log-return while preserving multi-decade variance, horizon-level equity correlation, and crash co-movement (`2022`, `1929`, `1973–74`, `2008`) without per-path variance collapse or look-ahead bias across $\rho \in [-0.50, 0.90]$.

| Series | Real monthly | Annual, smoothed |
|---|---|---|
| US equities | 1922–2025 (all) | — |
| Ex-US equities | 1970–2025 | 1922–1969 (JST) |
| Gold | 1968–2025 | 1922–1967 (peg era) |
| CHF Cash | — | 1922–2025 (`data/ch_cash_rate.csv` retail rate: SNB savings deposits from 1933, annual) |
| Bitcoin | — | synthetic throughout (**coupled to US equity shocks at $\rho = 0.50$**) |

### Rebuilding the data

```bash
python scripts/fetch_source_data.py                     # network: FRED, SNB, LBMA, JST
python scripts/build_historic_returns.py                # offline: regenerates src/historic_returns.py (CHF)
python scripts/fetch_source_data_at.py                  # network: BIS, ECB, Statistik Austria (Austria / EUR)
python scripts/build_historic_returns.py --currency EUR # offline: regenerates src/historic_returns_eur.py
```

### Austria / EUR data (`src/historic_returns_eur.py`)

8. **EUR per USD** — [`data/usd_eur.csv`](data/usd_eur.csv), 1957–2025, month-end: BIS `WS_XRU` `M.AT.EUR.E` (Austrian schilling converted at the fixed 13.7603 ATS/EUR before 1999; identical to the ECB reference-rate month-ends from 1999). Before 1958: PPP-implied from Swiss vs US CPI (no market rate).
9. **Austrian CPI** — [`data/at_inflation.csv`](data/at_inflation.csv), 1948–2025, monthly: Statistik Austria VPI 1958 (official long-chained national CPI) from Dec 1959, BIS long CPI chained on before that. Before 1958 the EUR module uses the **Swiss CPI**.
10. **Austrian household deposit rates** — [`data/at_cash_rate.csv`](data/at_cash_rate.csv), 2000–2025, annual: ECB MIR, deposits with agreed maturity up to 1 year (`retail_rate`, used), overnight and notice deposits (reference). Before 2000 the EUR module uses the **Swiss savings rate** (as-is to 1957, PPP-translated 1958–1999).

Equities, gold and Bitcoin use the same sources as the CHF build, converted to EUR. 1922–2025 real CAGRs in EUR: US 7.1%, Non-US 4.9%, gold 1.8%, cash 0.6%. Details: [Design Doc §7.5](docs/design_doc_early_retirement.md#7-multi-jurisdiction-design-austria).

## Project Structure

```text
├── app.py                              # Streamlit frontend UI (country switch: Switzerland / Austria)
├── data/                               # Normalised source datasets (every file here is ingested)
│                                       #   us_stocks.csv      month-end S&P 500 TR, USD, monthly
│                                       #   ex_us_stocks.csv   ex-US equities, USD, monthly (1970+ only)
│                                       #   jst_exus_usd.csv   JST GDP-weighted ex-US, USD, annual (1900-1969)
│                                       #   usd_chf.csv        month-end USD/CHF (SNB pre-1971, FRED after)
│                                       #   ch_inflation.csv   Swiss CPI, monthly
│                                       #   gold_usd.csv       LBMA PM fix, USD, monthly (1968+ real)
│                                       #   ch_cash_rate.csv   CHF retail savings rate (JST <1933, SNB 1933+) + wholesale ref
│                                       #   us_cpi.csv         US CPI-U, monthly, FRED from 1913 (build-time PPP only)
│                                       #   usd_eur.csv        month-end EUR per USD, BIS (1957+)             [Austria]
│                                       #   at_inflation.csv   Austrian CPI, Statistik Austria VPI 1958 + BIS  [Austria]
│                                       #   at_cash_rate.csv   Austrian household deposit rates, ECB MIR (2000+) [Austria]
├── src/                                # Importable `src` package (no `sys.path` manipulation required)
│   ├── __init__.py                     # Re-exports the public API (`from src import SimConfig, run_simulation`)
│   ├── simulation_engine.py            # Core monthly decumulation simulation loop & Monte Carlo generator
│   ├── jurisdiction.py                 # Country dispatch: COUNTRY_DEFAULTS, Jurisdiction protocol, Switzerland and Austria rules
│   ├── tax_engine.py                   # Swiss Federal & Canton Zurich tax and AHV calculations
│   ├── tax_engine_at.py                # Austrian 2026 tariff, KESt, Regelbesteuerungsoption, pension taxation, health self-insurance
│   ├── metrics.py                      # Pure result metrics (success mask, survival curves, depletion/shortfall timing, outcome severity ranking, real wealth, TL;DR outcome, withdrawal rates)
│   ├── historic_returns.py             # GENERATED: monthly/annual return series & bootstrap generators (CHF)
│   └── historic_returns_eur.py         # GENERATED: EUR / Austria counterpart with Swiss-proxy flags
├── scripts/
│   ├── fetch_source_data.py            # Network: FRED, SNB, LBMA, JST -> normalised CSVs in data/
│   ├── fetch_source_data_at.py         # Network: BIS, ECB, Statistik Austria -> Austrian / EUR CSVs in data/
│   └── build_historic_returns.py       # Offline: data/ -> src/historic_returns.py (--currency EUR -> src/historic_returns_eur.py)
├── tests/
│   ├── test_simulation_engine.py       # Unit & integration tests for simulation engine and return generators
│   ├── test_tax_engine.py              # Unit tests for income, wealth, capital withdrawal, and AHV taxes
│   ├── test_jurisdiction.py            # Country registry/validation and Swiss rules behind the Jurisdiction interface
│   ├── test_tax_engine_at.py           # Austrian tax rules: hand-computed statutory examples
│   ├── test_austria_engine.py          # Austria inside run_simulation (KESt on realized gains, health insurance, pension)
│   ├── test_cost_basis.py              # Average-cost basis helpers and realized/unrealized gain tracking in the engine
│   ├── test_historic_returns_eur.py    # EUR series and their Swiss-data stand-in segments
│   ├── test_metrics.py                 # Unit tests for src/metrics.py
│   └── test_app.py                     # Streamlit AppTest smoke tests for app.py (both countries + switching)
├── docs/
│   ├── prd_early_retirement_calc.md    # Product Requirement Document (PRD)
│   └── design_doc_early_retirement.md  # Technical Design Document (DD)
├── conftest.py                         # Puts the repo root on `sys.path` so tests can `import src`
├── requirements.txt                    # Python dependencies
└── venv/                               # Python virtual environment (ignored by git)
```

Not covered by an official calculator: Austrian capital income (KESt / Regelbesteuerungsoption) — the BMF offers no online calculator for it, so it is tested with hand-computed statutory examples.

## Getting Started

### Prerequisites

*   Python 3.11+
*   Virtual environment tool (`venv`)

### Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:b-maldoca/fi.git
    cd fi
    ```

2.  Set up the virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Running the Application

Start the Streamlit app:
```bash
./venv/bin/streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Running Tests

Execute the test suite using `pytest`:
```bash
./venv/bin/pytest
```
