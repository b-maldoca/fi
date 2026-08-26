# Zurich Early Retirement Simulator

This application is a financial simulation tool designed for individuals planning for early retirement (FIRE) in Canton Zurich, Switzerland.

Phase 1 focuses on the **post-retirement decumulation phase** for a single individual with no dependents living in Canton Zurich.

## Features

*   **Three Comparative Simulation Methods**: Compare outcomes side-by-side using:
    1. **Historic Backtesting**: Replay contiguous historical periods (~100 years of Swiss-adjusted data).
    2. **Historic Bootstrapping**: Random sampling with replacement across historical annual returns (as described in FIRE literature like *The Poor Swiss*).
    3. **Parametric Monte Carlo**: Stochastic lognormal return generator with customizable asset class means and volatilities.
*   **Swiss Tax Modeling**: Accurately models Federal, Cantonal, and Municipal income and wealth taxes for Canton Zurich.
*   **AHV for Non-Workers**: Models mandatory AHV contributions for early retirees before age 65.
*   **Pillar 2 & 3a Liquidations**: Simulates growth and staggered lump-sum withdrawals of Pillar 3a accounts (up to 5) and Pillar 2 vesting accounts, including capital withdrawal taxes.
*   **Smart Cash Buffer**: Optional defensive strategy to spend cash first during market downturns, protecting equities.
*   **Dynamic Expenses**: Optional adjustment to reduce expenses when net worth drops below the starting watermark, plus Vanguard Dynamic Spending rules.
*   **Configurable Success Criteria**: Set target ending net worth (e.g., preserve 50% of inflation-adjusted starting wealth) and calculate probability of success.

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
├── app.py                  # Streamlit frontend UI
├── data/                   # Historical CSV datasets from wichtounet/swr-calculator & SNB/FSO
├── src/
│   ├── simulation_engine.py # Core decumulation simulation loop
│   ├── tax_engine.py        # Swiss/Zurich tax calculations
│   └── historic_returns.py  # Historic return data and generators
├── scripts/
│   └── build_historic_returns.py # Pipeline building historic_returns.py from data/
├── docs/
│   ├── prd_early_retirement_calc.md # Product Requirement Document
│   └── design_doc_early_retirement.md # Technical Design Document
├── requirements.txt         # Python dependencies
└── venv/                    # Python virtual environment (ignored by git)
```

## Getting Started

### Prerequisites

*   Python 3.11+
*   Virtual environment tool (`venv`)

### Installation

1.  Clone the repository (once created on GitHub):
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
streamlit run app.py
```

The app will open in your default browser, typically at `http://localhost:8501`.
