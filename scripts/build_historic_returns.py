import os
import sys
import pandas as pd
import numpy as np

# Path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_FILE = os.path.join(BASE_DIR, 'src', 'historic_returns.py')

print(f"Ingesting datasets from {DATA_DIR}...")

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, header=None, names=['month', 'year', 'value'])
    return df.sort_values(['year', 'month']).reset_index(drop=True)

us_stocks = load_data("us_stocks.csv")
ex_us_stocks = load_data("ex_us_stocks.csv")
usd_chf = load_data("usd_chf.csv")
ch_inflation = load_data("ch_inflation.csv")

# Filter for Dec months to obtain year-end index levels
us_dec = us_stocks[us_stocks['month'] == 12].rename(columns={'value': 'us_stocks'})[['year', 'us_stocks']]
exus_dec = ex_us_stocks[ex_us_stocks['month'] == 12].rename(columns={'value': 'exus_stocks'})[['year', 'exus_stocks']]
fx_dec = usd_chf[usd_chf['month'] == 12].rename(columns={'value': 'usd_chf'})[['year', 'usd_chf']]
cpi_dec = ch_inflation[ch_inflation['month'] == 12].rename(columns={'value': 'ch_cpi'})[['year', 'ch_cpi']]

# Merge all series on Year
df = pd.merge(us_dec, exus_dec, on='year')
df = pd.merge(df, fx_dec, on='year')
df = pd.merge(df, cpi_dec, on='year')
df = df.sort_values('year').reset_index(drop=True)

# Calculate annual returns (1922 to 2025: 104 full calendar years)
df['us_usd_ret'] = df['us_stocks'].pct_change()
df['exus_usd_ret'] = df['exus_stocks'].pct_change()
df['fx_ret'] = df['usd_chf'].pct_change()
df['ch_cpi_ret'] = df['ch_cpi'].pct_change()

# CHF returns: (1 + R_USD) * (1 + FX_Change) - 1
df['us_chf_ret'] = (1.0 + df['us_usd_ret']) * (1.0 + df['fx_ret']) - 1.0
df['exus_chf_ret'] = (1.0 + df['exus_usd_ret']) * (1.0 + df['fx_ret']) - 1.0

clean_df = df.dropna().reset_index(drop=True)
years = clean_df['year'].astype(int).values
us_chf_returns = clean_df['us_chf_ret'].values
exus_chf_returns = clean_df['exus_chf_ret'].values
us_usd_returns = clean_df['us_usd_ret'].values
exus_usd_returns = clean_df['exus_usd_ret'].values
ch_inflation_rates = clean_df['ch_cpi_ret'].values
usd_chf_fx_changes = clean_df['fx_ret'].values

print(f"Processed {len(clean_df)} contiguous annual records from {years[0]} to {years[-1]}.")

# Format strings for python arrays
def fmt_arr(arr, decimals=4):
    return ",\n    ".join([f"{x:.{decimals}f}" for x in arr])

years_str = ", ".join([str(y) for y in years])
us_chf_str = fmt_arr(us_chf_returns)
exus_chf_str = fmt_arr(exus_chf_returns)
us_usd_str = fmt_arr(us_usd_returns)
exus_usd_str = fmt_arr(exus_usd_returns)
ch_inf_str = fmt_arr(ch_inflation_rates)
fx_str = fmt_arr(usd_chf_fx_changes)

code_content = f'''# Provenance & Data Sources:
# 1. US Stocks (USD): Robert Shiller monthly S&P 500 Total Returns dataset (1871-2025) via The Poor Swiss (wichtounet/swr-calculator).
# 2. Non-US Equities (USD): Empirical ex-US stocks total return dataset (MSCI EAFE / World ex-US proxy, 1871-2025) via The Poor Swiss (wichtounet/swr-calculator).
# 3. Currency Exchange (USD/CHF): Historical monthly USD/CHF exchange rates (1913-2019 via The Poor Swiss, 2020-2025 via Swiss National Bank SNB).
# 4. Swiss Inflation (CPI): Historical Swiss Consumer Price Index (1921-2023 via The Poor Swiss / Swiss Federal Statistical Office FSO/BFS, 2024-2025 via FSO).
#
# Returns in CHF are computed as: Return_CHF = ((1 + Return_USD) * (FX_End / FX_Start)) - 1

import numpy as np

HISTORIC_YEARS = np.array([
    {years_str}
])

# US Stocks Total Returns in CHF ({years[0]}-{years[-1]})
HISTORIC_RETURNS_US_CHF = np.array([
    {us_chf_str}
])

# Non-US Stocks Total Returns in CHF ({years[0]}-{years[-1]})
HISTORIC_RETURNS_NON_US_CHF = np.array([
    {exus_chf_str}
])

# US Stocks Total Returns in USD ({years[0]}-{years[-1]})
HISTORIC_RETURNS_US_USD = np.array([
    {us_usd_str}
])

# Non-US Stocks Total Returns in USD ({years[0]}-{years[-1]})
HISTORIC_RETURNS_NON_US_USD = np.array([
    {exus_usd_str}
])

# Swiss CPI Annual Inflation ({years[0]}-{years[-1]})
HISTORIC_SWISS_INFLATION = np.array([
    {ch_inf_str}
])

# USD/CHF Annual Exchange Rate Change ({years[0]}-{years[-1]})
HISTORIC_USD_CHF_FX = np.array([
    {fx_str}
])

# Backwards compatibility alias
HISTORIC_RETURNS = HISTORIC_RETURNS_US_CHF


def get_historic_return_matrix(duration_years: int) -> np.ndarray:
    """
    Returns a return matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks (Empirical S&P 500 in CHF)
    1: Non-US Stocks (Empirical MSCI ex-US in CHF)
    2: CHF Cash (1% nominal return)
    3: Gold (Synthetic uncorrelated, 6% mean, 15% vol)
    4: Bitcoin (Synthetic uncorrelated, 10% mean, 60% vol)
    """
    total_years = len(HISTORIC_RETURNS_US_CHF)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {{duration_years}}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {{duration_years}} exceeds available historic data ({{total_years}} years).")
        
    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    
    for i in range(num_runs):
        annual_us_chf = HISTORIC_RETURNS_US_CHF[i : i + duration_years]
        annual_non_us_chf = HISTORIC_RETURNS_NON_US_CHF[i : i + duration_years]
        
        # Approximate monthly returns by taking the 12th root of (1 + annual return)
        safe_base_us = np.maximum(0.0, 1.0 + annual_us_chf)
        monthly_us = safe_base_us**(1/12) - 1.0
        monthly_us_expanded = np.repeat(monthly_us, 12)
        
        safe_base_non_us = np.maximum(0.0, 1.0 + annual_non_us_chf)
        monthly_non_us = safe_base_non_us**(1/12) - 1.0
        monthly_non_us_expanded = np.repeat(monthly_non_us, 12)
        
        matrix[i, :, 0] = monthly_us_expanded        # US Stocks (CHF)
        matrix[i, :, 1] = monthly_non_us_expanded    # Non-US Stocks (CHF)
        matrix[i, :, 2] = 0.01 / 12                  # CHF Cash (1% nominal APY)
        
        # Synthetic Gold (Uncorrelated, 6% mean, 15% vol)
        rng = np.random.default_rng(i)
        matrix[i, :, 3] = rng.normal(0.06 / 12, 0.15 / np.sqrt(12), duration_months)
        
        # Synthetic Bitcoin (Uncorrelated, 10% mean, 60% vol)
        matrix[i, :, 4] = rng.normal(0.10 / 12, 0.60 / np.sqrt(12), duration_months)
        
    return matrix


def get_historic_inflation_matrix(duration_years: int) -> np.ndarray:
    """
    Returns an empirical inflation matrix of shape (num_runs, duration_years)
    containing the true contiguous Swiss CPI inflation sequence for each historical cohort.
    """
    total_years = len(HISTORIC_SWISS_INFLATION)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {{duration_years}}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {{duration_years}} exceeds available historic data ({{total_years}} years).")
        
    num_runs = total_years - duration_years + 1
    matrix = np.zeros((num_runs, duration_years))
    
    for i in range(num_runs):
        matrix[i, :] = HISTORIC_SWISS_INFLATION[i : i + duration_years]
        
    return matrix


def generate_bootstrapped_data(
    num_runs: int,
    duration_years: int,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jointly generates bootstrapped return matrix and inflation matrix from historical data.
    Ensures returns and inflation for each simulation run are sampled using the exact same random year sequence.
    
    Returns:
        tuple (return_matrix, inflation_matrix)
        return_matrix shape: (num_runs, duration_years * 12, 5)
        inflation_matrix shape: (num_runs, duration_years)
    """
    total_years = len(HISTORIC_RETURNS_US_CHF)
    if duration_years <= 0 or num_runs <= 0:
        raise ValueError(f"num_runs ({{num_runs}}) and duration_years ({{duration_years}}) must be positive integers.")
        
    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    return_matrix = np.zeros((num_runs, duration_months, 5))
    
    # Jointly sample random historical years with replacement
    random_indices = rng.integers(0, total_years, size=(num_runs, duration_years))
    sampled_annual_us = HISTORIC_RETURNS_US_CHF[random_indices]
    sampled_annual_non_us = HISTORIC_RETURNS_NON_US_CHF[random_indices]
    sampled_inflation = HISTORIC_SWISS_INFLATION[random_indices]
    
    # Convert annual returns to monthly returns
    safe_base_us = np.maximum(0.0, 1.0 + sampled_annual_us)
    sampled_monthly_us = safe_base_us**(1/12) - 1.0
    sampled_monthly_us_expanded = np.repeat(sampled_monthly_us, 12, axis=1)
    
    safe_base_non_us = np.maximum(0.0, 1.0 + sampled_annual_non_us)
    sampled_monthly_non_us = safe_base_non_us**(1/12) - 1.0
    sampled_monthly_non_us_expanded = np.repeat(sampled_monthly_non_us, 12, axis=1)
    
    return_matrix[:, :, 0] = sampled_monthly_us_expanded
    return_matrix[:, :, 1] = sampled_monthly_non_us_expanded
    return_matrix[:, :, 2] = 0.01 / 12 # 1% annual nominal
    
    # Synthetic Gold & Bitcoin monthly returns per run
    for i in range(num_runs):
        rng_asset = np.random.default_rng(seed + i + 1000)
        return_matrix[i, :, 3] = rng_asset.normal(0.06 / 12, 0.15 / np.sqrt(12), duration_months)
        return_matrix[i, :, 4] = rng_asset.normal(0.10 / 12, 0.60 / np.sqrt(12), duration_months)
        
    return return_matrix, sampled_inflation


def generate_bootstrapped_returns(
    num_runs: int,
    duration_years: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a matrix of shape (num_runs, duration_months, 5) using historical bootstrapping 
    (sampling annual return blocks with replacement from empirical Swiss-adjusted market history).
    """
    ret_matrix, _ = generate_bootstrapped_data(num_runs, duration_years, seed=seed)
    return ret_matrix


def generate_bootstrapped_inflation(
    num_runs: int,
    duration_years: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generates an inflation matrix of shape (num_runs, duration_years) by bootstrapping 
    empirical Swiss CPI inflation rates using the same random seed alignment as asset returns.
    """
    _, inf_matrix = generate_bootstrapped_data(num_runs, duration_years, seed=seed)
    return inf_matrix
'''

with open(SRC_FILE, 'w') as f:
    f.write(code_content)

print(f"Successfully generated {SRC_FILE}!")

