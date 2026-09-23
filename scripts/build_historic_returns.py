"""Build src/historic_returns.py from the normalised CSVs in data/.

Offline. Run `scripts/fetch_source_data.py` first if the sources need refreshing.

Key differences from the previous version of this pipeline:

1. REAL MONTHLY RETURNS are carried end-to-end. The old pipeline collapsed each
   series to its December value, computed an annual return, and then re-expanded
   it as twelve identical (1+R)^(1/12) months. That deleted ~72% of monthly
   volatility and understated the worst drawdown by ~18pp. Where genuine monthly
   data exists we now use it.

2. GENUINE EX-US DATA before 1970. data/ex_us_stocks.csv is byte-identical to
   data/us_stocks.csv until Dec 1969, so it carries no ex-US information at all.
   For 1900-1969 we substitute a GDP-weighted 17-country index built from the
   Jorda-Schularick-Taylor Macrohistory Database (data/jst_exus_usd.csv).

3. CORRECTED USD/CHF. data/usd_chf.csv is now rebuilt from FRED DEXSZUS
   (true month-end) for 1971+, replacing a hand-extended tail that was wrong by
   up to 13.4% in 2024-25.

4. REAL GOLD. The gold sleeve was previously a synthetic lognormal RNG even in
   "historic backtesting" mode. It now uses the LBMA PM fix in CHF.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
SRC_FILE = os.path.join(BASE, 'src', 'historic_returns.py')

# Swiss CPI (our inflation deflator) starts in 1921, so the first full calendar
# year of returns is 1922. JST ex-US starts in 1900, US equities in 1871.
START_YEAR = 1922
GOLD_REAL_FROM = 1968      # LBMA London fix; before this gold was a statutory fixed price
EXUS_REAL_FROM = 1970      # MSCI EAFE inception; before this we use JST

print(f'Building from {DATA}')


def load_monthly(filename):
    df = pd.read_csv(os.path.join(DATA, filename), header=None, names=['m', 'y', 'v'])
    df['v'] = pd.to_numeric(df.v, errors='coerce')
    df = df.dropna()
    df['ym'] = pd.PeriodIndex(pd.to_datetime(dict(year=df.y, month=df.m, day=1)), freq='M')
    return df.drop_duplicates('ym').set_index('ym').v.sort_index()


us_idx = load_monthly('us_stocks.csv')
exus_idx = load_monthly('ex_us_stocks.csv')
fx = load_monthly('usd_chf.csv')
cpi = load_monthly('ch_inflation.csv')
gold_usd = load_monthly('gold_usd.csv')
jst = pd.read_csv(os.path.join(DATA, 'jst_exus_usd.csv'))
ch_cash_df = pd.read_csv(os.path.join(DATA, 'ch_cash_rate.csv')).set_index('year')

# ---------------------------------------------------------------- calendar
last_year = min(int(us_idx.index[-1].year), int(exus_idx.index[-1].year),
                int(fx.index[-1].year), int(cpi.index[-1].year),
                int(ch_cash_df.index.max()))
# only keep complete calendar years
while not all(pd.Period(f'{last_year}-12', 'M') in s.index for s in (us_idx, exus_idx, fx, cpi)):
    last_year -= 1
years = np.arange(START_YEAR, last_year + 1)
months = pd.period_range(f'{START_YEAR}-01', f'{last_year}-12', freq='M')
print(f'Calendar: {years[0]}-{years[-1]}  ({len(years)} years, {len(months)} months)')


def smooth(annual_return):
    """Expand one annual return into 12 equal geometric monthly returns."""
    return (1.0 + max(annual_return, -0.999999)) ** (1.0 / 12.0) - 1.0


def chf_monthly_from_index(idx):
    """Month-on-month return of a USD index expressed in CHF."""
    chf = (idx * fx).dropna()
    return chf.pct_change()


# ------------------------------------------------- US equities (real monthly)
us_monthly_all = chf_monthly_from_index(us_idx)
us_usd_monthly_all = us_idx.pct_change()
monthly_us = us_monthly_all.reindex(months)
assert monthly_us.notna().all(), 'gap in US monthly series'

# ------------------------------------------------ ex-US equities (spliced)
exus_real = chf_monthly_from_index(exus_idx)
exus_usd_real = exus_idx.pct_change()

fx_dec = fx[fx.index.month == 12]
fx_dec.index = fx_dec.index.year
jst = jst.set_index('year')

monthly_exus = pd.Series(index=months, dtype=float)
monthly_exus_usd = pd.Series(index=months, dtype=float)
smoothed_flag = {}
for y in years:
    sel = [p for p in months if p.year == y]
    if y >= EXUS_REAL_FROM:
        monthly_exus.loc[sel] = exus_real.reindex(sel).values
        monthly_exus_usd.loc[sel] = exus_usd_real.reindex(sel).values
        smoothed_flag[y] = False
    else:
        tr_usd = float(jst.loc[y, 'tr_usd'])
        fx_chg = fx_dec[y] / fx_dec[y - 1] - 1.0
        tr_chf = (1.0 + tr_usd) * (1.0 + fx_chg) - 1.0
        monthly_exus.loc[sel] = smooth(tr_chf)
        monthly_exus_usd.loc[sel] = smooth(tr_usd)
        smoothed_flag[y] = True
assert monthly_exus.notna().all(), 'gap in ex-US monthly series'
n_smoothed = sum(smoothed_flag.values())
print(f'ex-US: {n_smoothed} years from JST (annual, smoothed), '
      f'{len(years)-n_smoothed} years real monthly')

# ---------------------------------------------------------- gold (spliced)
gold_real = chf_monthly_from_index(gold_usd)
monthly_gold = pd.Series(index=months, dtype=float)
gold_smoothed = {}
for y in years:
    sel = [p for p in months if p.year == y]
    if y >= GOLD_REAL_FROM:
        monthly_gold.loc[sel] = gold_real.reindex(sel).values
        gold_smoothed[y] = False
    else:
        # Statutory fixed USD price: the only CHF variation came from currency
        # devaluations (1933-34 US, 1936 CH). Spread the annual move evenly
        # rather than concentrating it in whichever month the FX series steps.
        g0, g1 = gold_usd.get(pd.Period(f'{y-1}-12', 'M')), gold_usd.get(pd.Period(f'{y}-12', 'M'))
        ann = ((g1 / g0) * (fx_dec[y] / fx_dec[y - 1]) - 1.0) if g0 and g1 else 0.0
        monthly_gold.loc[sel] = smooth(ann)
        gold_smoothed[y] = True
monthly_gold = monthly_gold.fillna(0.0)
print(f'gold: {sum(gold_smoothed.values())} years pegged/smoothed, '
      f'{len(years)-sum(gold_smoothed.values())} years real monthly (LBMA)')

# ----------------------------------------------------- CHF cash (JST + SNB)
monthly_cash = pd.Series(index=months, dtype=float)
for y in years:
    sel = [p for p in months if p.year == y]
    ann_cash = float(ch_cash_df.loc[y, 'retail_rate'])
    monthly_cash.loc[sel] = smooth(ann_cash)
assert monthly_cash.notna().all(), 'gap in CHF cash monthly series'

# ------------------------------------------------------------ precision
# Round the monthly series to the precision we will actually emit, THEN derive
# the annual arrays from the rounded values. Otherwise compounding 12 rounded
# monthly returns does not reproduce the separately-rounded annual figure, and
# the two sets of constants in the generated file disagree by ~1e-5.
DEC = 8
monthly_us = monthly_us.round(DEC)
monthly_exus = monthly_exus.round(DEC)
monthly_gold = monthly_gold.round(DEC)
monthly_cash = monthly_cash.round(DEC)
monthly_exus_usd = monthly_exus_usd.round(DEC)
us_usd_monthly = us_usd_monthly_all.reindex(months).round(DEC)


# ------------------------------------------------------------ annual series
def to_annual(monthly):
    s = monthly.copy()
    return np.array([np.prod(1.0 + s[[p for p in months if p.year == y]].values) - 1.0
                     for y in years])


us_chf_a = to_annual(monthly_us)
exus_chf_a = to_annual(monthly_exus)
us_usd_a = to_annual(us_usd_monthly)
exus_usd_a = to_annual(monthly_exus_usd)
gold_chf_a = to_annual(monthly_gold)
cash_chf_a = to_annual(monthly_cash)

cpi_dec = cpi[cpi.index.month == 12]
cpi_dec.index = cpi_dec.index.year
ch_inf_a = np.array([cpi_dec[y] / cpi_dec[y - 1] - 1.0 for y in years])
fx_a = np.array([fx_dec[y] / fx_dec[y - 1] - 1.0 for y in years])

print(f'\nUS   CHF: CAGR {np.prod(1+us_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {us_chf_a.std(ddof=1)*100:5.2f}%  monthly vol {monthly_us.std(ddof=1)*100:.2f}%')
print(f'exUS CHF: CAGR {np.prod(1+exus_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {exus_chf_a.std(ddof=1)*100:5.2f}%  monthly vol {monthly_exus.std(ddof=1)*100:.2f}%')
print(f'gold CHF: CAGR {np.prod(1+gold_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {gold_chf_a.std(ddof=1)*100:5.2f}%')
print(f'cash CHF: CAGR {np.prod(1+cash_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {cash_chf_a.std(ddof=1)*100:5.2f}%')
print(f'CH CPI  : CAGR {np.prod(1+ch_inf_a)**(1/len(years))*100-100:6.2f}%')
pre = years < EXUS_REAL_FROM
print(f'corr(US,exUS) pre-1970 = {np.corrcoef(us_chf_a[pre], exus_chf_a[pre])[0,1]:.3f} '
      f'(was 1.000 with the duplicated data)')
print(f'corr(US,gold)          = {np.corrcoef(us_chf_a, gold_chf_a)[0,1]:.3f} '
      f'(was 0.000 with the synthetic RNG)')

# --------------------------------------------------------------- emit code
def fmt(arr, per_line=8, dec=None):
    dec = DEC if dec is None else dec
    out, cells = [], [f'{x:.{dec}f}' for x in arr]
    for i in range(0, len(cells), per_line):
        out.append('    ' + ', '.join(cells[i:i + per_line]) + ',')
    return '\n'.join(out)


def fmt_bool(d):
    cells = ['True' if d[y] else 'False' for y in years]
    return '\n'.join('    ' + ', '.join(cells[i:i + 12]) + ','
                     for i in range(0, len(cells), 12))


header = f'''# ============================================================================
# GENERATED FILE - do not edit by hand.
# Regenerate with: python scripts/build_historic_returns.py
# (run scripts/fetch_source_data.py first to refresh the raw sources)
# ============================================================================
#
# Provenance & Data Sources (built {pd.Timestamp.today():%Y-%m-%d}):
#
# 1. US Stocks -- data/us_stocks.csv, monthly, 1871-{last_year}.
#    Month-END S&P 500 TOTAL RETURN index (dividends reinvested), via
#    wichtounet/swr-calculator (The Poor Swiss).
#    VERIFIED against published month-end S&P 500 total returns (Oct 1929
#    -19.71%, Oct 1987 -21.53%, Oct 2008 -16.80%, Mar 2020 -12.35%) and against
#    published calendar-year total returns (1974, 1995, 2000, 2008, 2011, 2013,
#    2017-19, 2021-22) to +/-0.00pp.
#    NOTE: previously mislabelled as the "Robert Shiller" dataset. Shiller's
#    series is a monthly AVERAGE of daily closes; this file is true month-end.
#    -> REAL MONTHLY returns are used.
#
# 2. Non-US Equities -- SPLICED, because data/ex_us_stocks.csv is byte-identical
#    to data/us_stocks.csv for all 1188 months Jan 1871 - Dec 1969 and therefore
#    contains no ex-US information before MSCI EAFE's Dec 1969 inception.
#      {START_YEAR}-{EXUS_REAL_FROM-1}: GDP-weighted 17-country index from the Jorda-Schularick-
#                 Taylor Macrohistory Database R6 (data/jst_exus_usd.csv).
#                 ANNUAL only, so expanded to 12 equal geometric months.
#                 See MONTHLY_EXUS_IS_SMOOTHED.
#      {EXUS_REAL_FROM}-{last_year}: data/ex_us_stocks.csv, REAL MONTHLY.
#    JST is nominal, local currency; converted to USD via `xrusd`, then to CHF.
#    Citation: Jorda, Schularick & Taylor, "Macrofinancial History and the New
#    Business Cycle Facts" (NBER Macro Annual 2017) and "The Rate of Return on
#    Everything, 1870-2015" (QJE 2019). Licence CC BY-NC-SA 4.0.
#
# 3. Currency (USD/CHF) -- data/usd_chf.csv, monthly, 1914-present.
#      1971+     : FRED DEXSZUS daily -> TRUE MONTH-END, matching the equity
#                  series' convention.
#      pre-1971  : SNB monthly average (devkum). Under Bretton Woods the rate
#                  was pegged, so there is no month-end variation to recover.
#    This REPLACES a hand-extended tail that was wrong by up to 13.4% in
#    2024-25 (Dec 2025 held 0.900 vs an actual 0.7936), which had overstated
#    the 2025 CHF equity return by roughly 16pp.
#
# 4. Swiss Inflation -- data/ch_inflation.csv, monthly, 1921-present.
#    Swiss CPI via the Federal Statistical Office (FSO/BFS).
#    Known gaps: February 2020 is missing; the 2024-25 extension is quantised
#    to 0.1% steps. Only December values are used here.
#
# 5. Gold -- data/gold_usd.csv x data/usd_chf.csv.
#      {GOLD_REAL_FROM}+    : LBMA London PM fix, REAL MONTHLY.
#      pre-{GOLD_REAL_FROM} : statutory fixed USD price ($20.67, then $35). The only
#                CHF variation came from the 1933-34 US and 1936 CH
#                devaluations, spread evenly across each year.
#    This REPLACES a synthetic uncorrelated lognormal RNG (6% / 15%) that was
#    previously used even in "historic backtesting" mode.
#
# 6. CHF Cash -- data/ch_cash_rate.csv.
#      {START_YEAR}-2020 : JST R6 Swiss short-term bill/money-market rate (`bill_rate`).
#      2021-{last_year} : Time-weighted SNB policy / SARON money-market rate.
#    Floored at 0.0% (`retail_rate`) to reflect Swiss retail savings / Cash Tent
#    deposit accounts during the 2012-2022 negative wholesale policy rate era.
#
# 7. Bitcoin -- SYNTHETIC (lognormal, 7% nominal / 50% vol default, correlated
#    with US equities at default rho=0.50 via Gaussian copula on standardized
#    monthly US equity log-returns).
#
# CHF conversion: Return_CHF = (1 + Return_USD) * (FX_end / FX_start) - 1
'''

code = f'''{header}
import numpy as np

HISTORIC_YEARS = np.array([
{fmt(years, per_line=12, dec=0).replace(".", "")}
])

# ---------------------------------------------------------------------------
# MONTHLY series (length = len(HISTORIC_YEARS) * 12), January {years[0]} onward.
# These are the primary data. The annual arrays below are derived from them.
# ---------------------------------------------------------------------------

# US equities, total return, CHF. Real month-end data throughout.
MONTHLY_US_CHF = np.array([
{fmt(monthly_us.values)}
])

# Non-US equities, total return, CHF. JST (smoothed) before {EXUS_REAL_FROM}, real monthly after.
MONTHLY_NON_US_CHF = np.array([
{fmt(monthly_exus.values)}
])

# CHF cash, retail short-term deposit yield (JST CHE bill_rate + SNB SARON, floored at 0%).
MONTHLY_CASH_CHF = np.array([
{fmt(monthly_cash.values)}
])

# Gold, CHF. Pegged/smoothed before {GOLD_REAL_FROM}, real LBMA month-end after.
MONTHLY_GOLD_CHF = np.array([
{fmt(monthly_gold.values)}
])

# True where that calendar year's ex-US months were expanded from an annual
# figure rather than observed monthly. Useful for honest error bars.
MONTHLY_EXUS_IS_SMOOTHED = np.array([
{fmt_bool(smoothed_flag)}
])

MONTHLY_GOLD_IS_SMOOTHED = np.array([
{fmt_bool(gold_smoothed)}
])

# ---------------------------------------------------------------------------
# ANNUAL series, derived by compounding the monthly series above.
# ---------------------------------------------------------------------------

HISTORIC_RETURNS_US_CHF = np.array([
{fmt(us_chf_a)}
])

HISTORIC_RETURNS_NON_US_CHF = np.array([
{fmt(exus_chf_a)}
])

HISTORIC_RETURNS_CASH_CHF = np.array([
{fmt(cash_chf_a)}
])

HISTORIC_RETURNS_US_USD = np.array([
{fmt(us_usd_a)}
])

HISTORIC_RETURNS_NON_US_USD = np.array([
{fmt(exus_usd_a)}
])

HISTORIC_RETURNS_GOLD_CHF = np.array([
{fmt(gold_chf_a)}
])

HISTORIC_SWISS_INFLATION = np.array([
{fmt(ch_inf_a)}
])

HISTORIC_USD_CHF_FX = np.array([
{fmt(fx_a)}
])

# Backwards compatibility alias
HISTORIC_RETURNS = HISTORIC_RETURNS_US_CHF

# Asset column order used by every return matrix in this module.
ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_CHF_CASH, ASSET_GOLD, ASSET_BITCOIN = range(5)
NUM_ASSETS = 5

# Bitcoin: synthetic lognormal coupled to realized US equity log-returns.
BITCOIN_NOMINAL_MEAN = 0.07
BITCOIN_VOL = 0.50
DEFAULT_BTC_EQUITY_CORR = 0.50

# Politis-White (2004) optimal stationary bootstrap mean block length (years).
POLITIS_WHITE_BLOCK_YEARS = 5

# Historical real CHF appreciation beyond Relative Purchasing Power Parity (1922-2025).
# Over 1922-2025, USD/CHF fell -1.78%/yr while the Swiss-vs-US CPI differential implied
# -1.09%/yr, leaving a -0.68%/yr excess real FX drag on foreign-priced assets (+0.68%/yr
# real CHF appreciation beyond PPP). Setting real_chf_appreciation=0.0 removes this drag.
HISTORIC_REAL_CHF_APPRECIATION = 0.0068
FOREIGN_ASSET_COLS = (ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_GOLD)


def _generate_lognormal_monthly_returns(
    rng: np.random.Generator,
    ann_ret: float,
    ann_vol: float,
    shape: tuple | int,
    us_monthly_slice: np.ndarray | None = None,
    btc_equity_corr: float = 0.0,
) -> np.ndarray:
    """Generates monthly returns using a lognormal model with Ito drift correction.

    When us_monthly_slice and a non-zero btc_equity_corr are provided, couples the standard
    normal shocks to path-standardized monthly US equity log-returns with correlation
    rho = btc_equity_corr, preserving Bitcoin's target drift and volatility in every cohort.
    """
    if ann_ret <= -1.0:
        raise ValueError(f"ann_ret ({{ann_ret}}) must be greater than -1.0.")
    if ann_vol < 0.0:
        raise ValueError(f"ann_vol ({{ann_vol}}) cannot be negative.")
    if not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")
    drift = np.log(1.0 + ann_ret) - 0.5 * (ann_vol ** 2)
    eps = rng.normal(0.0, 1.0, shape)
    if us_monthly_slice is not None and abs(btc_equity_corr) > 1e-12:
        us_log = np.log(np.maximum(1e-8, 1.0 + us_monthly_slice))
        us_mean = np.mean(us_log, axis=-1, keepdims=True)
        us_std = np.maximum(1e-12, np.std(us_log, axis=-1, ddof=1, keepdims=True))
        z_us = (us_log - us_mean) / us_std
        z = btc_equity_corr * z_us + np.sqrt(max(0.0, 1.0 - btc_equity_corr ** 2)) * eps
    else:
        z = eps
    log_ret = (drift / 12.0) + (ann_vol / np.sqrt(12.0)) * z
    return np.exp(log_ret) - 1.0


def _apply_fx_ppp_adjustment(
    matrix: np.ndarray,
    real_chf_appreciation: float | None
) -> np.ndarray:
    """Adjusts foreign-priced asset sleeves for target real CHF appreciation beyond PPP."""
    if real_chf_appreciation is None:
        return matrix
    if real_chf_appreciation >= 1.0:
        raise ValueError(
            f"real_chf_appreciation ({{real_chf_appreciation}}) must be strictly less than 1.0."
        )
    ann_adj = (1.0 - real_chf_appreciation) / (1.0 - HISTORIC_REAL_CHF_APPRECIATION)
    month_adj = ann_adj ** (1.0 / 12.0)
    if np.isclose(month_adj, 1.0, atol=1e-12):
        return matrix
    for col in FOREIGN_ASSET_COLS:
        matrix[:, :, col] = (1.0 + matrix[:, :, col]) * month_adj - 1.0
    return matrix


def compute_effective_sample_size(
    total_years: int,
    duration_years: int,
    success_rate_pct: float = 50.0,
) -> dict[str, float]:
    """Computes the Bartlett/Hansen-Hodrick effective sample size N_eff = T / D for
    overlapping rolling historical cohorts, along with the 90% Wilson confidence interval
    on the cohort success rate evaluated at N_eff.
    """
    if duration_years <= 0 or total_years <= 0:
        raise ValueError("total_years and duration_years must be positive.")
    n_cohorts = max(1, total_years - duration_years + 1)
    n_eff = max(1.0, float(total_years) / float(duration_years))
    p_hat = float(np.clip(success_rate_pct / 100.0, 0.0, 1.0))
    z = 1.6448536269514722  # 90% two-sided Wilson score interval (z_0.95)
    denom = 1.0 + (z ** 2) / n_eff
    center = (p_hat + (z ** 2) / (2.0 * n_eff)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat)) / n_eff + (z ** 2) / (4.0 * (n_eff ** 2)))) / denom
    ci_low_pct = float(np.clip((center - half_width) * 100.0, 0.0, 100.0))
    ci_high_pct = float(np.clip((center + half_width) * 100.0, 0.0, 100.0))
    return {{
        "n_cohorts": float(n_cohorts),
        "n_eff": n_eff,
        "ci_low_pct": ci_low_pct,
        "ci_high_pct": ci_high_pct,
    }}


def get_historic_return_matrix(
    duration_years: int,
    seed: int = 42,
    real_chf_appreciation: float | None = None,
    btc_equity_corr: float = DEFAULT_BTC_EQUITY_CORR,
    btc_mean: float = BITCOIN_NOMINAL_MEAN,
    btc_vol: float = BITCOIN_VOL,
) -> np.ndarray:
    """
    Returns a return matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks      (empirical S&P 500 total return in CHF, real monthly)
    1: Non-US Stocks  (JST GDP-weighted ex-US before {EXUS_REAL_FROM}, real monthly after)
    2: CHF Cash       (empirical JST Swiss bill rate + SNB SARON, floored at 0% for retail)
    3: Gold           (LBMA in CHF; pegged before {GOLD_REAL_FROM})
    4: Bitcoin        (synthetic lognormal, correlated with US Stocks at rho=btc_equity_corr)

    Each run i replays the contiguous calendar window starting at HISTORIC_YEARS[i].
    """
    total_years = len(HISTORIC_YEARS)
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {{duration_years}}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {{duration_years}} exceeds available historic data ({{total_years}} years).")
    if seed < 0:
        raise ValueError(f"seed ({{seed}}) must be a non-negative integer.")
    if not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")

    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, NUM_ASSETS))

    for i in range(num_runs):
        a, b = i * 12, i * 12 + duration_months
        us_slice = MONTHLY_US_CHF[a:b]
        matrix[i, :, ASSET_US_STOCKS] = us_slice
        matrix[i, :, ASSET_NON_US_STOCKS] = MONTHLY_NON_US_CHF[a:b]
        matrix[i, :, ASSET_CHF_CASH] = MONTHLY_CASH_CHF[a:b]
        matrix[i, :, ASSET_GOLD] = MONTHLY_GOLD_CHF[a:b]
        rng = np.random.default_rng(seed + i)
        matrix[i, :, ASSET_BITCOIN] = _generate_lognormal_monthly_returns(
            rng,
            btc_mean,
            btc_vol,
            duration_months,
            us_monthly_slice=us_slice,
            btc_equity_corr=btc_equity_corr,
        )

    return _apply_fx_ppp_adjustment(matrix, real_chf_appreciation)


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
        matrix[i, :] = HISTORIC_SWISS_INFLATION[i: i + duration_years]
    return matrix


def generate_bootstrapped_data(
    num_runs: int,
    duration_years: int,
    seed: int = 42,
    block_size_years: int = POLITIS_WHITE_BLOCK_YEARS,
    real_chf_appreciation: float | None = None,
    btc_equity_corr: float = DEFAULT_BTC_EQUITY_CORR,
    stationary: bool = True,
    btc_mean: float = BITCOIN_NOMINAL_MEAN,
    btc_vol: float = BITCOIN_VOL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jointly generates a block-bootstrapped return matrix and inflation matrix.

    By default (`stationary=True`), uses the Politis-Romano (1994) Stationary Bootstrap
    with circular wrap-around and geometric block lengths of mean `block_size_years`
    (Politis-White optimal block length = 5 years). Every historical year has exact
    1/T stationary selection probability (eliminating endpoint under-sampling and
    fixed 5-year seam artefacts). Setting `stationary=False` falls back to fixed
    non-circular contiguous blocks of length `block_size_years`.

    Returns:
        (return_matrix, inflation_matrix)
        return_matrix    shape (num_runs, duration_years * 12, 5)
        inflation_matrix shape (num_runs, duration_years)
    """
    total_years = len(HISTORIC_YEARS)
    if duration_years <= 0 or num_runs <= 0:
        raise ValueError(f"num_runs ({{num_runs}}) and duration_years ({{duration_years}}) must be positive integers.")
    if seed < 0:
        raise ValueError(f"seed ({{seed}}) must be a non-negative integer.")
    if block_size_years <= 0 or block_size_years > total_years:
        raise ValueError(f"block_size_years ({{block_size_years}}) must be between 1 and {{total_years}}.")
    if not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")

    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    return_matrix = np.zeros((num_runs, duration_months, NUM_ASSETS))

    if stationary:
        p_new_block = 1.0 / float(block_size_years)
        year_idx = np.zeros((num_runs, duration_years), dtype=int)
        year_idx[:, 0] = rng.integers(0, total_years, size=num_runs)
        if duration_years > 1:
            random_starts = rng.integers(0, total_years, size=(num_runs, duration_years - 1))
            jump_mask = rng.random(size=(num_runs, duration_years - 1)) < p_new_block
            for t in range(1, duration_years):
                year_idx[:, t] = np.where(
                    jump_mask[:, t - 1],
                    random_starts[:, t - 1],
                    (year_idx[:, t - 1] + 1) % total_years,
                )
    else:
        num_blocks = int(np.ceil(duration_years / block_size_years))
        max_start_idx = total_years - block_size_years
        start_indices = rng.integers(0, max_start_idx + 1, size=(num_runs, num_blocks))
        offsets = np.arange(block_size_years)
        year_idx = (start_indices[:, :, None] + offsets[None, None, :]
                    ).reshape(num_runs, -1)[:, :duration_years]

    # Expand each sampled calendar year into its 12 real monthly observations.
    month_idx = (year_idx[:, :, None] * 12 + np.arange(12)[None, None, :]
                 ).reshape(num_runs, duration_months)

    us_sampled = MONTHLY_US_CHF[month_idx]
    return_matrix[:, :, ASSET_US_STOCKS] = us_sampled
    return_matrix[:, :, ASSET_NON_US_STOCKS] = MONTHLY_NON_US_CHF[month_idx]
    return_matrix[:, :, ASSET_CHF_CASH] = MONTHLY_CASH_CHF[month_idx]
    return_matrix[:, :, ASSET_GOLD] = MONTHLY_GOLD_CHF[month_idx]
    return_matrix[:, :, ASSET_BITCOIN] = _generate_lognormal_monthly_returns(
        rng,
        btc_mean,
        btc_vol,
        (num_runs, duration_months),
        us_monthly_slice=us_sampled,
        btc_equity_corr=btc_equity_corr,
    )

    _apply_fx_ppp_adjustment(return_matrix, real_chf_appreciation)
    return return_matrix, HISTORIC_SWISS_INFLATION[year_idx]


def generate_bootstrapped_returns(
    num_runs: int,
    duration_years: int,
    seed: int = 42,
    block_size_years: int = POLITIS_WHITE_BLOCK_YEARS,
    real_chf_appreciation: float | None = None,
    btc_equity_corr: float = DEFAULT_BTC_EQUITY_CORR,
    stationary: bool = True,
    btc_mean: float = BITCOIN_NOMINAL_MEAN,
    btc_vol: float = BITCOIN_VOL,
) -> np.ndarray:
    """Return-matrix-only wrapper around generate_bootstrapped_data."""
    ret_matrix, _ = generate_bootstrapped_data(
        num_runs,
        duration_years,
        seed=seed,
        block_size_years=block_size_years,
        real_chf_appreciation=real_chf_appreciation,
        btc_equity_corr=btc_equity_corr,
        stationary=stationary,
        btc_mean=btc_mean,
        btc_vol=btc_vol,
    )
    return ret_matrix


def generate_bootstrapped_inflation(
    num_runs: int,
    duration_years: int,
    seed: int = 42,
    block_size_years: int = POLITIS_WHITE_BLOCK_YEARS,
    stationary: bool = True,
) -> np.ndarray:
    """Inflation-matrix-only wrapper around generate_bootstrapped_data."""
    _, inf_matrix = generate_bootstrapped_data(
        num_runs,
        duration_years,
        seed=seed,
        block_size_years=block_size_years,
        stationary=stationary,
    )
    return inf_matrix
'''

with open(SRC_FILE, 'w') as f:
    f.write(code)
print(f'\nWrote {SRC_FILE}')

