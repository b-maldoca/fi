# Zurich Early Retirement Simulator

This application is a financial simulation tool designed for individuals planning for early retirement (FIRE) in Canton Zurich, Switzerland.

Phase 1 focuses on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich.

## Features

*   **Three Comparative Simulation Methods**: Compare outcomes side-by-side using:
    1. **Historic Backtesting**: Replay contiguous historical periods (1922–2025, 104 years of Swiss-adjusted market, cash, gold, and CPI data, paired with Ito-corrected lognormal synthetic Bitcoin returns coupled to US equity shocks). Surfaces **Effective Sample Size ($N_{\text{eff}} = T / D$)** and the **90% Wilson confidence interval** on the cohort survival rate (reserving matching vertical space in the adjacent columns to keep all 3 columns aligned), and plots **Worst Cohort (Min) / Best Cohort (Max)** bounds (`[0, 25, 50, 75, 100]`) rather than unsupported 5th/95th percentiles.
    2. **Historic Bootstrapping (Politis–Romano Stationary Bootstrap)**: Joint stationary block sampling (`stationary=True`, mean block length `block_size_years = 5` via geometric restart probability $p = 1/b$) with **circular wrap-around** (`year 2025 -> year 1922`) across historical equity, gold, cash, and Swiss CPI series, preserving multi-year volatility clusters while ensuring uniform sampling of every historical year (`1922` and `2025` have identical inclusion probability $1/104$).
    3. **Parametric Monte Carlo**: Stochastic lognormal return generator (with Ito drift correction) with cross-asset shock coupling (`US`/`Non-US` $\rho=0.75$, `Gold` $\rho=0.08$ — the 1922–2025 annual US/gold correlation — and `Bitcoin` $\rho=\text{btc\_equity\_corr}$, default `0.50`) and independent annual inflation stream (`seed + 10_000`). Volatility and gold defaults are calibrated at runtime to the 1922–2025 CHF history (equity log-vol ≈ 21%, gold ≈ 4.7% mean / 15.3% vol, via `historic_lognormal_params`); stock, cash and inflation means are conservative forward-looking assumptions (US 7%, ex-US 6% vs. historic 11.3% / 8.6%), so Monte Carlo is deliberately more pessimistic than the historical engines. Every input's help text shows its historic value.
*   **Swiss Tax Modeling**: Models Federal, Cantonal (95% Steuerfuss), and Municipal (e.g., 119% Zurich City) income and wealth taxes for Canton Zurich using the **2026 tariffs, verified against the official ESTV tax calculator** (income, wealth and capital-withdrawal taxes match within CHF 1–2; pinned by golden tests) — including the constitutional **11.5% ceiling on federal income tax** (Art. 128 BV) — and **dynamic cash interest taxation** (taxable interest is computed from the actual realized annual return on CHF Cash rather than a hardcoded `1%`, eliminating phantom tax drag during `0%` ZIRP/NIRP years). The same Zurich City multipliers are the defaults of both the tax functions and `SimConfig`.
*   **Tax Bracket Indexation**: Bracket edges and the AHV contribution table are indexed to realized CPI each year, as Swiss law requires (Art. 39 DBG federally, § 48 StG ZH for Zurich income *and* wealth tax). Configurable from 0–100% of CPI to stress-test delayed or suspended compensation — freezing brackets at 0% overstates median lifetime real taxes by roughly a third over a 50-year horizon.
*   **AHV for Non-Workers**: Models mandatory AHV contributions for early retirees before age 65 based on wealth.
*   **Pillar 2 & 3a Liquidations**: Simulates tax-sheltered equity growth and staggered lump-sum withdrawals of Pillar 3a accounts (up to 5 equal-balance accounts between ages 60–64, configured via account count and a single balance per account input) and Pillar 2 vesting accounts (at age 65, or Year 0 if retiring at $\ge 65$), with the separate **capital withdrawal tax** deducted at source: federal at 1/5 of the ordinary tariff (Art. 38 DBG) plus Zurich at the rate for a notional annual pension of 1/20 of the lump sum with a **2% minimum simple tax** (§ 37 StG ZH, since 2022) — about 4.8% all-in on CHF 100k and 10.9% on CHF 1M in Zurich City.
*   **Rebalancing & Pfau Cash Tent Glidepath**: Supports Cash Tent (rising equity glidepath from a multi-year cash buffer of expenses + estimated taxes; the allocation held during year *y* is the glidepath weight for year *y*, reaching the base weights after exactly `tent_duration_years`), Monthly, Quarterly, Yearly, Threshold-based, and Never rebalancing strategies.
*   **Smart Cash Buffer**: Optional defensive strategy during market downturns (net worth below inflation-adjusted starting watermark) to pause rebalancing and spend CHF Cash first before selling equities. The watermark comparison (also used by Dynamic Floor & Ceiling spending and "Years Below") is **after tax**: both current and starting net worth deduct the capital withdrawal tax still owed on Pillar 2 / 3a balances under the simulator's own payout schedule, so paying that tax at 60–65 is not mistaken for a market downturn.
*   **Flexible Spending Strategies**: Choose between **Static** (inflation-adjusted base expenses), **Dynamic (Floor & Ceiling)**, and **Vanguard Dynamic** spending (tax-inclusive UI Target Withdrawal Rate bi-directionally synchronized with Year 0 base expenses and estimated taxes using live tax/yield inputs).
*   **Configurable Success Criteria**: Set target ending net worth (e.g., preserve 50% of inflation-adjusted starting wealth) and evaluate survival and wealth preservation metrics.

## Data Sources & Provenance

This project incorporates historical datasets from primary macroeconomic sources and [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator):

1. **US Stocks (USD)** — [`data/us_stocks.csv`](data/us_stocks.csv), 1871–2025, monthly.
   A **month-end S&P 500 total-return index** (dividends reinvested).
   *Verified* against published month-end S&P 500 total returns (Oct 1929 −19.71%, Oct 1987 −21.53%, Oct 2008 −16.80%, Mar 2020 −12.35%) and against published calendar-year total returns for 1974, 1995, 2000, 2008, 2011, 2013, 2017–19, 2021–22 (agreement to ±0.00pp).
2. **Non-US Equities (USD)** — [`data/jst_exus_usd.csv`](data/jst_exus_usd.csv) (1900–1969, annual) spliced with [`data/ex_us_stocks.csv`](data/ex_us_stocks.csv) (1970–2025, monthly).
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

*   **Currency / PPP Assumption (`real_chf_appreciation`, default `0.00%/yr`)**: `HISTORIC_REAL_CHF_APPRECIATION` is **derived from the data at build time** (not hardcoded): over 1922–2025 the geometric `USD/CHF` drift and the relative-PPP drift implied by Swiss vs US CPI give $a_{\text{hist}} = 1 - \frac{1 + g_{\text{FX}}}{1 + g_{\text{PPP}}} \approx$ `+0.71%/yr` real CHF appreciation beyond PPP (a real FX drag on foreign-priced assets). The sidebar exposes **Real CHF Appreciation beyond PPP (%/yr)** defaulting to `0.00%` (PPP neutrality, scaling foreign-priced monthly returns by $\left(\frac{1 - a_{\text{target}}}{1 - a_{\text{hist}}}\right)^{1/12}$), while `+0.71%` reproduces the raw unadjusted 1922–2025 historical CHF returns.
*   **Bitcoin–Equity Correlation Coupling (`btc_equity_corr`, default `0.50`)**: Because Bitcoin has no 40-year history, its monthly log-returns ($\mu = 7.0\%$, $\sigma = 50.0\%$ default across all 3 engines, Ito-corrected and user-configurable via `btc_mean` / `btc_vol`) are coupled to **per-path standardized** monthly US equity log-returns via a Gaussian copula:
    $$Z_{\text{BTC}, t} = \rho_{\text{BTC,EQ}} Z_{\text{US}, t} + \sqrt{1 - \rho_{\text{BTC,EQ}}^2}\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$
    Per-path standardization ($\text{mean}(Z_{\text{US}}) = 0$, $\text{std}(Z_{\text{US}}) = 1$ within each cohort/run) ensures US cohort drift never leaks into Bitcoin's expected log-return while eliminating the zero-correlation "free lunch" during equity market drawdowns (`2022`, `1929`, `1973–74`, `2008`) across $\rho \in [-0.50, 0.90]$.

| Series | Real monthly | Annual, smoothed |
|---|---|---|
| US equities | 1922–2025 (all) | — |
| Ex-US equities | 1970–2025 | 1922–1969 (JST) |
| Gold | 1968–2025 | 1922–1967 (peg era) |
| CHF Cash | — | 1922–2025 (`data/ch_cash_rate.csv` retail rate: SNB savings deposits from 1933, annual) |
| Bitcoin | — | synthetic throughout (**coupled to US equity shocks at $\rho = 0.50$**) |

### Rebuilding the data

```bash
python scripts/fetch_source_data.py      # network: FRED, SNB, LBMA, JST
python scripts/build_historic_returns.py # offline: regenerates src/historic_returns.py
```

## Project Structure

```text
├── app.py                              # Streamlit frontend UI
├── data/                               # Normalised source datasets (every file here is ingested)
│                                       #   us_stocks.csv      month-end S&P 500 TR, USD, monthly
│                                       #   ex_us_stocks.csv   ex-US equities, USD, monthly (1970+ only)
│                                       #   jst_exus_usd.csv   JST GDP-weighted ex-US, USD, annual (1900-1969)
│                                       #   usd_chf.csv        month-end USD/CHF (SNB pre-1971, FRED after)
│                                       #   ch_inflation.csv   Swiss CPI, monthly
│                                       #   gold_usd.csv       LBMA PM fix, USD, monthly (1968+ real)
│                                       #   ch_cash_rate.csv   CHF retail savings rate (JST <1933, SNB 1933+) + wholesale ref
│                                       #   us_cpi.csv         US CPI-U, monthly, FRED from 1913 (build-time PPP only)
├── src/                                # Importable `src` package (no `sys.path` manipulation required)
│   ├── __init__.py                     # Re-exports the public API (`from src import SimConfig, run_simulation`)
│   ├── simulation_engine.py            # Core monthly decumulation simulation loop & Monte Carlo generator
│   ├── tax_engine.py                   # Swiss Federal & Canton Zurich tax and AHV calculations
│   ├── metrics.py                      # Pure result metrics (success mask, real wealth, TL;DR outcome, withdrawal rates)
│   └── historic_returns.py             # GENERATED: monthly/annual return series & bootstrap generators
├── scripts/
│   ├── fetch_source_data.py            # Network: FRED, SNB, LBMA, JST -> normalised CSVs in data/
│   └── build_historic_returns.py       # Offline: data/ -> src/historic_returns.py
├── tests/
│   ├── test_simulation_engine.py       # Unit & integration tests for simulation engine and return generators
│   ├── test_tax_engine.py              # Unit tests for income, wealth, capital withdrawal, and AHV taxes
│   ├── test_metrics.py                 # Unit tests for src/metrics.py
│   └── test_app.py                     # Streamlit AppTest smoke tests for app.py
├── docs/
│   ├── prd_early_retirement_calc.md    # Product Requirement Document (PRD)
│   └── design_doc_early_retirement.md  # Technical Design Document (DD)
├── conftest.py                         # Puts the repo root on `sys.path` so tests can `import src`
├── requirements.txt                    # Python dependencies
└── venv/                               # Python virtual environment (ignored by git)
```

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
