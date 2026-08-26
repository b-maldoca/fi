import numpy as np

# Historical Market Returns in CHF (1928-2025)
# Base Asset: Aswath Damodaran's S&P 500 Total Returns
# Currency Adjustment: Historical USD/CHF Exchange Rates
HISTORIC_RETURNS = np.array([
    0.4381,
    -0.0830,
    -0.2512,
    -0.4384,
    -0.0864,
    0.1060,
    -0.2085,
    0.4674,
    0.8843,
    -0.3534,
    0.2928,
    -0.0110,
    -0.1067,
    -0.1277,
    0.1917,
    0.2506,
    0.1903,
    0.3582,
    -0.0843,
    0.0520,
    0.0570,
    0.1830,
    0.3081,
    0.2368,
    0.1815,
    -0.0121,
    0.5256,
    0.3260,
    0.0744,
    -0.1046,
    0.4372,
    0.1206,
    0.0034,
    0.2664,
    -0.0881,
    0.2261,
    0.1642,
    0.1240,
    -0.0997,
    0.2380,
    0.1081,
    -0.0824,
    0.0356,
    0.0037,
    0.1659,
    -0.2635,
    -0.4214,
    0.4187,
    0.1532,
    -0.2414,
    -0.1329,
    0.1559,
    0.4674,
    -0.0308,
    0.3387,
    0.3402,
    0.2611,
    0.0438,
    -0.0681,
    -0.1705,
    0.3764,
    0.3498,
    -0.1880,
    0.3629,
    0.1546,
    0.1224,
    -0.1031,
    0.2044,
    0.4295,
    0.4403,
    0.2126,
    0.4030,
    -0.0674,
    -0.0969,
    -0.3552,
    0.1440,
    0.0264,
    0.2047,
    0.0679,
    -0.0236,
    -0.3995,
    0.2237,
    0.0367,
    0.0210,
    0.1340,
    0.2924,
    0.2628,
    0.0240,
    0.1289,
    0.1679,
    -0.0324,
    0.2853,
    0.0819,
    0.3285,
    -0.1714,
    0.1510,
    0.3083,
    0.2040
])

def get_historic_return_matrix(duration_years: int) -> np.ndarray:
    """
    Returns a matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin.
    Currently uses S&P 500 for US Stocks, and synthetic proxies for the others 
    since 100-year historical data is limited.
    """
    total_years = len(HISTORIC_RETURNS)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {duration_years}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {duration_years} exceeds available historic data ({total_years} years).")
        
    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    
    for i in range(num_runs):
        annual_sp500 = HISTORIC_RETURNS[i : i + duration_years]
        
        # Approximate monthly returns by taking the 12th root of the annual return
        # and repeating it 12 times per year
        safe_base = np.maximum(0.0, 1.0 + annual_sp500)
        monthly_sp500 = safe_base**(1/12) - 1.0
        monthly_sp500_expanded = np.repeat(monthly_sp500, 12)
        
        matrix[i, :, 0] = monthly_sp500_expanded # US Stocks
        matrix[i, :, 1] = monthly_sp500_expanded * 0.8 # Non-US Stocks (Synthetic proxy)
        matrix[i, :, 2] = 0.01/12 # CHF Cash (1% nominal return)
        
        # Synthetic Gold (Uncorrelated, 6% mean, 15% vol)
        rng = np.random.default_rng(i)
        matrix[i, :, 3] = rng.normal(0.06/12, 0.15/np.sqrt(12), duration_months)
        
        # Synthetic Bitcoin (Uncorrelated, 10% mean, 60% vol)
        matrix[i, :, 4] = rng.normal(0.10/12, 0.60/np.sqrt(12), duration_months)
        
    return matrix


def generate_bootstrapped_returns(
    num_runs: int,
    duration_years: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a matrix of shape (num_runs, duration_months, 5) using historical bootstrapping 
    (sampling annual returns with replacement from the historical dataset).
    
    Asset Classes:
    0: US Stocks (Sampled from Damodaran/S&P500 CHF returns)
    1: Non-US Stocks (0.8x proxy of sampled US stocks)
    2: CHF Cash (1% annual nominal return)
    3: Gold (Synthetic uncorrelated, 6% mean, 15% vol)
    4: Bitcoin (Synthetic uncorrelated, 10% mean, 60% vol)
    """
    total_years = len(HISTORIC_RETURNS)
    if duration_years <= 0 or num_runs <= 0:
        raise ValueError(f"num_runs ({num_runs}) and duration_years ({duration_years}) must be positive integers.")
        
    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    
    # Sample random historical years with replacement: shape (num_runs, duration_years)
    random_indices = rng.integers(0, total_years, size=(num_runs, duration_years))
    sampled_annual_returns = HISTORIC_RETURNS[random_indices] # shape: (num_runs, duration_years)
    
    # Convert annual returns to monthly returns
    safe_base = np.maximum(0.0, 1.0 + sampled_annual_returns)
    sampled_monthly_returns = safe_base**(1/12) - 1.0 # shape: (num_runs, duration_years)
    
    # Repeat 12 times along the time axis to match duration_months
    sampled_monthly_expanded = np.repeat(sampled_monthly_returns, 12, axis=1) # shape: (num_runs, duration_months)
    
    matrix[:, :, 0] = sampled_monthly_expanded
    matrix[:, :, 1] = sampled_monthly_expanded * 0.8
    matrix[:, :, 2] = 0.01 / 12 # 1% annual nominal
    
    # Synthetic Gold & Bitcoin monthly returns per run
    for i in range(num_runs):
        rng_asset = np.random.default_rng(seed + i + 1000)
        matrix[i, :, 3] = rng_asset.normal(0.06 / 12, 0.15 / np.sqrt(12), duration_months)
        matrix[i, :, 4] = rng_asset.normal(0.10 / 12, 0.60 / np.sqrt(12), duration_months)
        
    return matrix
