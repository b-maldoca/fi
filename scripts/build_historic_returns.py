import pandas as pd
import numpy as np

# 1. Download Damodaran Data
url = "https://www.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls"
print("Downloading Damodaran S&P 500 Historical Returns...")
df = pd.read_excel(url, sheet_name="Returns by year", skiprows=18)

# Extract Year and S&P 500 Return (including dividends)
df = df.iloc[:, [0, 1]].dropna()
df.columns = ["Year", "SP500_Return"]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)
df['SP500_Return'] = pd.to_numeric(df['SP500_Return'], errors='coerce')

# Filter 1928 to 2025
df = df[(df['Year'] >= 1928) & (df['Year'] <= 2025)].copy()

# 2. Historical USD/CHF Exchange Rates
fx_rates = []
for year in df['Year']:
    if 1928 <= year <= 1932:
        fx_rates.append(5.18)
    elif year == 1933:
        fx_rates.append(3.82)
    elif 1934 <= year <= 1935:
        fx_rates.append(3.06)
    elif 1936 <= year <= 1970:
        fx_rates.append(4.37)
    else:
        fx_rates.append(None) 

df['USDCHF_End'] = fx_rates

# Exact array of approximate end-of-year USD/CHF rates from 1971 to 2025
real_fx = [
    3.84, 3.77, 3.24, 2.53, 2.62, 2.44, 1.99, 1.62, 1.58, 1.76, 
    1.79, 1.99, 2.18, 2.59, 2.06, 1.62, 1.27, 1.50, 1.54, 1.29, 
    1.35, 1.45, 1.48, 1.31, 1.15, 1.34, 1.45, 1.37, 1.59, 1.63, 
    1.67, 1.38, 1.23, 1.14, 1.31, 1.21, 1.12, 1.06, 1.03, 0.93, 
    0.93, 0.91, 0.89, 0.99, 1.00, 1.01, 0.97, 0.98, 0.96, 0.88,
    0.91, 0.92, 0.84, 0.88, 0.90 # Added 2024, 2025
]

post_1970_mask = df['Year'] >= 1971
df.loc[post_1970_mask, 'USDCHF_End'] = real_fx

# Calculate start of year rates
df['USDCHF_Start'] = df['USDCHF_End'].shift(1)
df.loc[df['Year'] == 1928, 'USDCHF_Start'] = 5.18

# 3. Calculate True CHF Return
# Return_CHF = ((1 + Return_USD) * (FX_End / FX_Start)) - 1
df['Return_CHF'] = ((1 + df['SP500_Return']) * (df['USDCHF_End'] / df['USDCHF_Start'])) - 1

print(df[['Year', 'SP500_Return', 'USDCHF_End', 'Return_CHF']].head())

# 4. Generate new historic_returns.py
chf_returns_str = ",\n    ".join([f"{x:.4f}" for x in df['Return_CHF'].values])

code = f'''import numpy as np

# Historical Market Returns in CHF (1928-2025)
# Base Asset: Aswath Damodaran's S&P 500 Total Returns
# Currency Adjustment: Historical USD/CHF Exchange Rates
HISTORIC_RETURNS = np.array([
    {chf_returns_str}
])

def get_historic_return_matrix(duration_years: int) -> np.ndarray:
    """
    Returns a matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks, 1: Non-US Stocks, 2: CHF Cash, 3: Gold, 4: Bitcoin.
    Currently uses S&P 500 for US Stocks, and synthetic proxies for the others 
    since 100-year historical data is limited.
    """
    total_years = len(HISTORIC_RETURNS)
    if duration_years > total_years:
        raise ValueError(f"Duration {{duration_years}} exceeds available historic data ({{total_years}} years).")
        
    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    
    for i in range(num_runs):
        annual_sp500 = HISTORIC_RETURNS[i : i + duration_years]
        
        # Approximate monthly returns by taking the 12th root of the annual return
        # and repeating it 12 times per year
        monthly_sp500 = (1 + annual_sp500)**(1/12) - 1
        monthly_sp500_expanded = np.repeat(monthly_sp500, 12)
        
        matrix[i, :, 0] = monthly_sp500_expanded # US Stocks
        matrix[i, :, 1] = monthly_sp500_expanded * 0.8 # Non-US Stocks (Synthetic proxy)
        matrix[i, :, 2] = 0.01/12 # CHF Cash (1% nominal return)
        
        # Synthetic Gold (Uncorrelated, 6% mean, 15% vol)
        np.random.seed(i)
        matrix[i, :, 3] = np.random.normal(0.06/12, 0.15/np.sqrt(12), duration_months)
        
        # Synthetic Bitcoin (Uncorrelated, 10% mean, 60% vol)
        matrix[i, :, 4] = np.random.normal(0.10/12, 0.60/np.sqrt(12), duration_months)
        
    return matrix
'''

with open("src/historic_returns.py", "w") as f:
    f.write(code)

print("Successfully generated src/historic_returns.py with real Damodaran S&P 500 data adjusted for CHF!")
