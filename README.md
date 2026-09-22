# Zurich Early Retirement Simulator

This application is a financial simulation tool designed for individuals planning for early retirement (FIRE) in Canton Zurich, Switzerland.

Phase 1 focuses on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich.

## Features

*   **Three Comparative Simulation Methods**: Compare outcomes side-by-side using:
    1. **Historic Backtesting**: Replay contiguous historical periods (1922–2025, 104 years of Swiss-adjusted market and CPI data).
    2. **Historic Bootstrapping**: Joint random sampling with replacement across historical annual equity returns and Swiss CPI inflation (as described in FIRE literature like *The Poor Swiss*).
    3. **Parametric Monte Carlo**: Stochastic lognormal return generator (with Ito drift correction) and customizable asset class means, volatilities, inflation parameters, and reproducible **Random Seed**.
*   **Swiss Tax Modeling**: Accurately models Federal, Cantonal (95% Steuerfuss), and Municipal (e.g., 119% Zurich City) income and wealth taxes for Canton Zurich.
*   **AHV for Non-Workers**: Models mandatory AHV contributions for early retirees before age 65 based on wealth.
*   **Pillar 2 & 3a Liquidations**: Simulates tax-sheltered equity growth and staggered lump-sum withdrawals of Pillar 3a accounts (up to 5 accounts between ages 60–64) and Pillar 2 vesting accounts (at age 65, or Year 0 if retiring at $\ge 65$), including immediate progressive capital withdrawal taxes.
*   **Rebalancing & Pfau Cash Tent Glidepath**: Supports Cash Tent (rising equity glidepath from a multi-year cash buffer of expenses + estimated taxes), Monthly, Quarterly, Yearly, Threshold-based, and Never rebalancing strategies.
*   **Smart Cash Buffer**: Optional defensive strategy during market downturns (net worth below inflation-adjusted starting watermark) to pause rebalancing and spend CHF Cash first before selling equities.
*   **Flexible Spending Strategies**: Choose between **Static** (inflation-adjusted base expenses), **Dynamic (Floor & Ceiling)**, and **Vanguard Dynamic** spending (target withdrawal rate bi-directionally synchronized with Year 0 base expenses and estimated taxes).
*   **Configurable Success Criteria**: Set target ending net worth (e.g., preserve 50% of inflation-adjusted starting wealth) and evaluate survival and wealth preservation metrics.

## Data Sources & Provenance

This project incorporates historical datasets curated and maintained by Baptiste Wicht (*The Poor Swiss*), available in the open-source repository [wichtounet/swr-calculator](https://github.com/wichtounet/swr-calculator) and described at [The Poor Swiss](https://thepoorswiss.com):

1. **US Stocks (USD)**: Robert Shiller monthly S&P 500 Total Returns dataset (1871–2025) from [`data/us_stocks.csv`](https://github.com/wichtounet/swr-calculator/blob/master/stock-data/us_stocks.csv).
2. **Non-US Equities (USD)**: Empirical ex-US stocks total return dataset (MSCI EAFE / World ex-US proxy, 1871–2025) from [`data/ex_us_stocks.csv`](https://github.com/wichtounet/swr-calculator/blob/master/stock-data/ex_us_stocks.csv).
3. **Currency Exchange (USD/CHF)**: Historical monthly USD/CHF exchange rates (1913–2019 from [`data/usd_chf.csv`](https://github.com/wichtounet/swr-calculator/blob/master/stock-data/usd_chf.csv), extended through 2025 via the [Swiss National Bank (SNB)](https://www.snb.ch)).
4. **Swiss Inflation (CPI)**: Historical Swiss Consumer Price Index (1921–2023 from [`data/ch_inflation.csv`](https://github.com/wichtounet/swr-calculator/blob/master/stock-data/ch_inflation.csv), cleaned of 2022 entry error and extended through 2025 via the [Swiss Federal Statistical Office (FSO/BFS)](https://www.bfs.admin.ch)).

CHF-converted equity returns are computed annually as:
$$\text{Return}_{\text{CHF}} = (1 + \text{Return}_{\text{USD}}) \times \left(\frac{\text{FX}_{\text{End}}}{\text{FX}_{\text{Start}}}\right) - 1$$

## Project Structure

```text
├── app.py                              # Streamlit frontend UI
├── data/                               # Historical CSV datasets from wichtounet/swr-calculator & SNB/FSO
├── src/
│   ├── simulation_engine.py            # Core monthly decumulation simulation loop & Monte Carlo generator
│   ├── tax_engine.py                   # Swiss Federal & Canton Zurich tax and AHV calculations
│   └── historic_returns.py             # Historic return datasets and bootstrapping generators
├── scripts/
│   └── build_historic_returns.py       # Data pipeline building src/historic_returns.py from data/
├── tests/
│   ├── test_simulation_engine.py       # Unit & integration tests for simulation engine and return generators
│   └── test_tax_engine.py              # Unit tests for income, wealth, capital withdrawal, and AHV taxes
├── docs/
│   ├── prd_early_retirement_calc.md    # Product Requirement Document (PRD)
│   └── design_doc_early_retirement.md  # Technical Design Document (DD)
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
