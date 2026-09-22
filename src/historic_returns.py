# Provenance & Data Sources:
# 1. US Stocks (USD): Robert Shiller monthly S&P 500 Total Returns dataset (1871-2025) via The Poor Swiss (wichtounet/swr-calculator).
# 2. Non-US Equities (USD): Empirical ex-US stocks total return dataset (MSCI EAFE / World ex-US proxy, 1871-2025) via The Poor Swiss (wichtounet/swr-calculator).
# 3. Currency Exchange (USD/CHF): Historical monthly USD/CHF exchange rates (1913-2019 via The Poor Swiss, 2020-2025 via Swiss National Bank SNB).
# 4. Swiss Inflation (CPI): Historical Swiss Consumer Price Index (1921-2023 via The Poor Swiss / Swiss Federal Statistical Office FSO/BFS, 2024-2025 via FSO).
#
# Returns in CHF are computed as: Return_CHF = ((1 + Return_USD) * (FX_End / FX_Start)) - 1

import numpy as np

HISTORIC_YEARS = np.array([
    1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
])

# US Stocks Total Returns in CHF (1922-2025)
HISTORIC_RETURNS_US_CHF = np.array([
    0.1632,
    0.0997,
    0.2458,
    0.2215,
    0.1122,
    0.3749,
    0.4332,
    -0.0900,
    -0.2565,
    -0.4391,
    -0.0888,
    0.1946,
    -0.2506,
    0.4662,
    0.4296,
    -0.1479,
    0.3359,
    0.0062,
    -0.1068,
    -0.1380,
    0.2108,
    0.2576,
    0.1969,
    0.3646,
    -0.0878,
    0.0524,
    0.0510,
    0.1831,
    0.3158,
    0.2496,
    0.1806,
    -0.0182,
    0.5237,
    0.3138,
    0.0661,
    -0.1084,
    0.4336,
    0.1281,
    0.0044,
    0.2682,
    -0.0868,
    0.2261,
    0.1629,
    0.1258,
    -0.1014,
    0.2399,
    0.1069,
    -0.0853,
    0.0393,
    0.0383,
    0.1456,
    -0.2654,
    -0.4250,
    0.4135,
    0.1539,
    -0.2491,
    -0.1329,
    0.1696,
    0.4752,
    -0.0420,
    0.3579,
    0.3296,
    0.2671,
    0.0427,
    -0.0707,
    -0.1737,
    0.3791,
    0.3527,
    -0.2002,
    0.3954,
    0.1614,
    0.1165,
    -0.1084,
    0.2129,
    0.4286,
    0.4537,
    0.2110,
    0.3979,
    -0.0792,
    -0.0915,
    -0.3520,
    0.1538,
    0.0184,
    0.2108,
    0.0742,
    -0.0185,
    -0.4070,
    0.2272,
    0.0377,
    0.0255,
    0.1323,
    0.2914,
    0.2656,
    0.0219,
    0.1378,
    0.1657,
    -0.0368,
    0.2966,
    0.0825,
    0.3263,
    -0.1703,
    0.1494,
    0.3156,
    0.1988
])

# Non-US Stocks Total Returns in CHF (1922-2025)
HISTORIC_RETURNS_NON_US_CHF = np.array([
    0.1632,
    0.0997,
    0.2458,
    0.2215,
    0.1122,
    0.3749,
    0.4332,
    -0.0900,
    -0.2565,
    -0.4391,
    -0.0888,
    0.1946,
    -0.2506,
    0.4662,
    0.4296,
    -0.1479,
    0.3359,
    0.0062,
    -0.1068,
    -0.1380,
    0.2108,
    0.2576,
    0.1969,
    0.3646,
    -0.0878,
    0.0524,
    0.0510,
    0.1831,
    0.3158,
    0.2496,
    0.1806,
    -0.0182,
    0.5237,
    0.3138,
    0.0661,
    -0.1084,
    0.4336,
    0.1281,
    0.0044,
    0.2682,
    -0.0868,
    0.2261,
    0.1629,
    0.1258,
    -0.1014,
    0.2399,
    0.1069,
    -0.0853,
    -0.1346,
    0.2106,
    0.3520,
    -0.2298,
    -0.3623,
    0.3673,
    -0.0338,
    -0.0471,
    0.0843,
    0.0964,
    0.3990,
    -0.0158,
    0.1149,
    0.3545,
    0.2351,
    0.1991,
    0.2994,
    -0.0213,
    0.5114,
    0.1446,
    -0.3629,
    0.2025,
    -0.0497,
    0.3451,
    -0.0527,
    -0.0147,
    0.2454,
    0.1180,
    0.1218,
    0.4813,
    -0.1202,
    -0.1871,
    -0.2972,
    0.2554,
    0.1099,
    0.3267,
    0.1710,
    0.0506,
    -0.4657,
    0.3041,
    -0.0130,
    -0.1141,
    0.1422,
    0.1859,
    0.0700,
    -0.0183,
    0.0497,
    0.1942,
    -0.1300,
    0.2146,
    -0.0118,
    0.1662,
    -0.1269,
    0.0794,
    0.1077,
    0.3480
])

# US Stocks Total Returns in USD (1922-2025)
HISTORIC_RETURNS_US_USD = np.array([
    0.2779,
    0.0417,
    0.2570,
    0.2955,
    0.1114,
    0.3713,
    0.4331,
    -0.0891,
    -0.2526,
    -0.4386,
    -0.0886,
    0.5289,
    -0.0234,
    0.4721,
    0.3280,
    -0.3526,
    0.3320,
    -0.0091,
    -0.1008,
    -0.1177,
    0.2108,
    0.2576,
    0.1969,
    0.3646,
    -0.0818,
    0.0524,
    0.0510,
    0.1806,
    0.3058,
    0.2455,
    0.1850,
    -0.0110,
    0.5240,
    0.3143,
    0.0663,
    -0.1085,
    0.4334,
    0.1190,
    0.0048,
    0.2681,
    -0.0878,
    0.2269,
    0.1636,
    0.1236,
    -0.1010,
    0.2394,
    0.1100,
    -0.0847,
    0.0399,
    0.1433,
    0.1894,
    -0.1479,
    -0.2654,
    0.3725,
    0.2367,
    -0.0739,
    0.0644,
    0.1835,
    0.3227,
    -0.0505,
    0.2148,
    0.2250,
    0.0615,
    0.3165,
    0.1860,
    0.0517,
    0.1661,
    0.3168,
    -0.0310,
    0.3047,
    0.0762,
    0.1008,
    0.0132,
    0.3758,
    0.2296,
    0.3336,
    0.2858,
    0.2104,
    -0.0910,
    -0.1189,
    -0.2210,
    0.2868,
    0.1088,
    0.0491,
    0.1579,
    0.0549,
    -0.3700,
    0.2646,
    0.1506,
    0.0211,
    0.1600,
    0.3239,
    0.1369,
    0.0138,
    0.1196,
    0.2183,
    -0.0438,
    0.3149,
    0.1840,
    0.2871,
    -0.1811,
    0.2629,
    0.2502,
    0.1788
])

# Non-US Stocks Total Returns in USD (1922-2025)
HISTORIC_RETURNS_NON_US_USD = np.array([
    0.2779,
    0.0417,
    0.2570,
    0.2955,
    0.1114,
    0.3713,
    0.4331,
    -0.0891,
    -0.2526,
    -0.4386,
    -0.0886,
    0.5289,
    -0.0234,
    0.4721,
    0.3280,
    -0.3526,
    0.3320,
    -0.0091,
    -0.1008,
    -0.1177,
    0.2108,
    0.2576,
    0.1969,
    0.3646,
    -0.0818,
    0.0524,
    0.0510,
    0.1806,
    0.3058,
    0.2455,
    0.1850,
    -0.0110,
    0.5240,
    0.3143,
    0.0663,
    -0.1085,
    0.4334,
    0.1190,
    0.0048,
    0.2681,
    -0.0878,
    0.2269,
    0.1636,
    0.1236,
    -0.1010,
    0.2394,
    0.1100,
    -0.0847,
    -0.1341,
    0.3330,
    0.4036,
    -0.1067,
    -0.1853,
    0.3276,
    0.0355,
    0.1752,
    0.3310,
    0.1094,
    0.2544,
    -0.0245,
    -0.0026,
    0.2479,
    0.0347,
    0.5140,
    0.6584,
    0.2456,
    0.2780,
    0.1142,
    -0.2281,
    0.1244,
    -0.1193,
    0.3261,
    0.0764,
    0.1176,
    0.0720,
    0.0256,
    0.1911,
    0.2827,
    -0.1316,
    -0.2116,
    -0.1551,
    0.4001,
    0.2084,
    0.1496,
    0.2623,
    0.1292,
    -0.4323,
    0.3439,
    0.0943,
    -0.1178,
    0.1702,
    0.2157,
    -0.0388,
    -0.0260,
    0.0329,
    0.2481,
    -0.1364,
    0.2316,
    0.0809,
    0.1317,
    -0.1382,
    0.1860,
    0.0526,
    0.3255
])

# Swiss CPI Annual Inflation (1922-2025)
HISTORIC_SWISS_INFLATION = np.array([
    -0.1262,
    0.0479,
    0.0166,
    -0.0251,
    -0.0354,
    0.0068,
    0.0012,
    -0.0043,
    -0.0328,
    -0.0730,
    -0.0718,
    -0.0231,
    -0.0190,
    0.0093,
    0.0154,
    0.0439,
    -0.0065,
    0.0373,
    0.1261,
    0.1526,
    0.0830,
    0.0286,
    0.0141,
    -0.0072,
    0.0256,
    0.0533,
    0.0058,
    -0.0191,
    0.0000,
    0.0635,
    0.0013,
    -0.0055,
    0.0150,
    0.0059,
    0.0218,
    0.0201,
    0.0089,
    -0.0060,
    0.0177,
    0.0351,
    0.0324,
    0.0388,
    0.0231,
    0.0493,
    0.0457,
    0.0351,
    0.0220,
    0.0230,
    0.0544,
    0.0663,
    0.0684,
    0.1193,
    0.0755,
    0.0344,
    0.0127,
    0.0116,
    0.0073,
    0.0519,
    0.0440,
    0.0663,
    0.0545,
    0.0213,
    0.0290,
    0.0324,
    0.0004,
    0.0189,
    0.0194,
    0.0502,
    0.0528,
    0.0522,
    0.0343,
    0.0248,
    0.0042,
    0.0195,
    0.0079,
    0.0039,
    -0.0017,
    0.0167,
    0.0149,
    0.0033,
    0.0089,
    0.0059,
    0.0133,
    0.0101,
    0.0062,
    0.0200,
    0.0071,
    0.0028,
    0.0053,
    -0.0071,
    -0.0044,
    0.0007,
    -0.0033,
    -0.0130,
    -0.0001,
    0.0084,
    0.0069,
    0.0015,
    -0.0071,
    0.0153,
    0.0584,
    -0.0119,
    0.0060,
    0.0070
])

# USD/CHF Annual Exchange Rate Change (1922-2025)
HISTORIC_USD_CHF_FX = np.array([
    -0.0898,
    0.0557,
    -0.0089,
    -0.0571,
    0.0007,
    0.0026,
    0.0001,
    -0.0010,
    -0.0053,
    -0.0010,
    -0.0002,
    -0.2187,
    -0.2327,
    -0.0040,
    0.0765,
    0.3161,
    0.0029,
    0.0154,
    -0.0067,
    -0.0230,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.0066,
    0.0000,
    0.0000,
    0.0021,
    0.0077,
    0.0033,
    -0.0038,
    -0.0072,
    -0.0002,
    -0.0004,
    -0.0002,
    0.0002,
    0.0001,
    0.0080,
    -0.0004,
    0.0000,
    0.0012,
    -0.0006,
    -0.0006,
    0.0020,
    -0.0003,
    0.0004,
    -0.0028,
    -0.0007,
    -0.0006,
    -0.0918,
    -0.0368,
    -0.1378,
    -0.2173,
    0.0299,
    -0.0670,
    -0.1892,
    -0.1854,
    -0.0118,
    0.1153,
    0.0090,
    0.1178,
    0.0854,
    0.1937,
    -0.2080,
    -0.2165,
    -0.2143,
    0.1826,
    0.0272,
    -0.1746,
    0.0695,
    0.0791,
    0.0143,
    -0.1200,
    -0.1184,
    0.1618,
    0.0900,
    -0.0582,
    0.1549,
    0.0131,
    0.0310,
    -0.1682,
    -0.1034,
    -0.0815,
    0.1541,
    -0.0723,
    -0.0696,
    -0.0587,
    -0.0296,
    -0.0981,
    0.0043,
    -0.0239,
    -0.0246,
    0.1132,
    0.0079,
    0.0163,
    -0.0432,
    0.0074,
    -0.0139,
    -0.0857,
    0.0305,
    0.0132,
    -0.0898,
    0.0523,
    0.0169
])

# Backwards compatibility alias
HISTORIC_RETURNS = HISTORIC_RETURNS_US_CHF


def _generate_lognormal_monthly_returns(
    rng: np.random.Generator,
    ann_ret: float,
    ann_vol: float,
    shape: tuple
) -> np.ndarray:
    """Generates monthly returns using a lognormal model with Ito drift correction."""
    drift = np.log(1.0 + ann_ret) - 0.5 * (ann_vol ** 2)
    log_ret = rng.normal(drift / 12.0, ann_vol / np.sqrt(12.0), shape)
    return np.exp(log_ret) - 1.0


def get_historic_return_matrix(duration_years: int, seed: int = 42) -> np.ndarray:
    """
    Returns a return matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks (Empirical S&P 500 in CHF)
    1: Non-US Stocks (Empirical MSCI ex-US in CHF)
    2: CHF Cash (1% nominal return, geometric monthly rate)
    3: Gold (Synthetic lognormal uncorrelated, 6% mean, 15% vol)
    4: Bitcoin (Synthetic lognormal uncorrelated, 10% mean, 60% vol)
    """
    total_years = len(HISTORIC_RETURNS_US_CHF)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {duration_years}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {duration_years} exceeds available historic data ({total_years} years).")
    if seed < 0:
        raise ValueError(f"seed ({seed}) must be a non-negative integer.")
        
    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, 5))
    cash_monthly = (1.0 + 0.01)**(1.0 / 12.0) - 1.0
    
    for i in range(num_runs):
        annual_us_chf = HISTORIC_RETURNS_US_CHF[i : i + duration_years]
        annual_non_us_chf = HISTORIC_RETURNS_NON_US_CHF[i : i + duration_years]
        
        # Approximate monthly returns by taking the 12th root of (1 + annual return)
        safe_base_us = np.maximum(0.0, 1.0 + annual_us_chf)
        monthly_us = safe_base_us**(1.0 / 12.0) - 1.0
        monthly_us_expanded = np.repeat(monthly_us, 12)
        
        safe_base_non_us = np.maximum(0.0, 1.0 + annual_non_us_chf)
        monthly_non_us = safe_base_non_us**(1.0 / 12.0) - 1.0
        monthly_non_us_expanded = np.repeat(monthly_non_us, 12)
        
        matrix[i, :, 0] = monthly_us_expanded        # US Stocks (CHF)
        matrix[i, :, 1] = monthly_non_us_expanded    # Non-US Stocks (CHF)
        matrix[i, :, 2] = cash_monthly               # CHF Cash (1% nominal APY)
        
        # Synthetic Gold & Bitcoin (Lognormal with Ito drift correction)
        rng = np.random.default_rng(seed + i)
        matrix[i, :, 3] = _generate_lognormal_monthly_returns(rng, 0.06, 0.15, duration_months)
        matrix[i, :, 4] = _generate_lognormal_monthly_returns(rng, 0.10, 0.60, duration_months)
        
    return matrix


def get_historic_inflation_matrix(duration_years: int) -> np.ndarray:
    """
    Returns an empirical inflation matrix of shape (num_runs, duration_years)
    containing the true contiguous Swiss CPI inflation sequence for each historical cohort.
    """
    total_years = len(HISTORIC_SWISS_INFLATION)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {duration_years}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {duration_years} exceeds available historic data ({total_years} years).")
        
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
        raise ValueError(f"num_runs ({num_runs}) and duration_years ({duration_years}) must be positive integers.")
    if seed < 0:
        raise ValueError(f"seed ({seed}) must be a non-negative integer.")
        
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
    sampled_monthly_us = safe_base_us**(1.0 / 12.0) - 1.0
    sampled_monthly_us_expanded = np.repeat(sampled_monthly_us, 12, axis=1)
    
    safe_base_non_us = np.maximum(0.0, 1.0 + sampled_annual_non_us)
    sampled_monthly_non_us = safe_base_non_us**(1.0 / 12.0) - 1.0
    sampled_monthly_non_us_expanded = np.repeat(sampled_monthly_non_us, 12, axis=1)
    
    return_matrix[:, :, 0] = sampled_monthly_us_expanded
    return_matrix[:, :, 1] = sampled_monthly_non_us_expanded
    return_matrix[:, :, 2] = (1.0 + 0.01)**(1.0 / 12.0) - 1.0 # 1% annual nominal
    
    # Vectorized synthetic Gold & Bitcoin monthly returns (lognormal with Ito drift correction)
    return_matrix[:, :, 3] = _generate_lognormal_monthly_returns(rng, 0.06, 0.15, (num_runs, duration_months))
    return_matrix[:, :, 4] = _generate_lognormal_monthly_returns(rng, 0.10, 0.60, (num_runs, duration_months))
        
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
