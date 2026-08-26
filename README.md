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

## Project Structure

```text
├── app.py                  # Streamlit frontend UI
├── src/
│   ├── simulation_engine.py # Core decumulation simulation loop
│   ├── tax_engine.py        # Swiss/Zurich tax calculations
│   └── historic_returns.py  # Historic return data and generators
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
