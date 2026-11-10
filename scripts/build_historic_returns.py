"""Build src/historic_returns.py (CHF) or src/historic_returns_eur.py (EUR) from data/.

Offline. Run `scripts/fetch_source_data.py` first if the sources need refreshing (and
`scripts/fetch_source_data_at.py` for the EUR build).

    python scripts/build_historic_returns.py                  # CHF -> src/historic_returns.py
    python scripts/build_historic_returns.py --currency EUR   # EUR -> src/historic_returns_eur.py

EUR build (Austria, PRD §8.6.F): the USD index levels are converted at month-end EUR/USD and
deflated with Austrian CPI. SWISS DATA STANDS IN where no usable Austrian data exists:
  1922-1957 : inflation = Swiss CPI (chained onto Austrian CPI at Dec 1957), EUR Cash = Swiss
              savings rate, EUR/USD = relative-PPP-implied rate (Swiss vs US CPI), not a market rate.
  1958-1999 : EUR Cash = Swiss savings rate translated to EUR under relative PPP
              (1 + r_CH) * (1 + pi_AT) / (1 + pi_CH) - 1.
  2000+     : EUR Cash = ECB MIR rate on Austrian household deposits with agreed maturity <= 1 year.

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
import argparse
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
_parser = argparse.ArgumentParser(description='Build the generated historic return module.')
_parser.add_argument('--currency', choices=('CHF', 'EUR'), default='CHF',
                     help='CHF -> src/historic_returns.py (default), EUR -> src/historic_returns_eur.py')
_parser.add_argument('--output', default=None,
                     help='write the generated module here instead (used by the in-sync test)')
_args = _parser.parse_args()
CCY = _args.currency
SRC_FILE = _args.output or os.path.join(BASE, 'src', 'historic_returns.py' if CCY == 'CHF' else 'historic_returns_eur.py')

# Swiss CPI (our inflation deflator) starts in 1921, so the first full calendar
# year of returns is 1922. JST ex-US starts in 1900, US equities in 1871.
START_YEAR = 1922
GOLD_REAL_FROM = 1968      # LBMA London fix; before this gold was a statutory fixed price
EXUS_REAL_FROM = 1970      # MSCI EAFE inception; before this we use JST
# EUR build only: last year whose inflation, cash and EUR/USD are Swiss-data stand-ins, and
# last year whose EUR Cash return is the PPP-translated Swiss savings rate.
SWISS_PROXY_LAST_YEAR = 1957
CASH_SWISS_PROXY_LAST_YEAR = 1999

print(f'Building {CCY} from {DATA}')


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
us_cpi = load_monthly('us_cpi.csv')
gold_usd = load_monthly('gold_usd.csv')
jst = pd.read_csv(os.path.join(DATA, 'jst_exus_usd.csv'))
ch_cash_df = pd.read_csv(os.path.join(DATA, 'ch_cash_rate.csv')).set_index('year')


def dec_value(series, year):
    return series[pd.Period(f'{year}-12', 'M')]


if CCY == 'CHF':
    home_cash_rate = ch_cash_df['retail_rate']
else:
    # ------------------------------------------------- EUR (Austria) composites
    ch_cpi, ch_fx = cpi, fx
    at_fx = load_monthly('usd_eur.csv')          # EUR per USD, month-end, 1957+
    at_cpi = load_monthly('at_inflation.csv')    # Austrian CPI, monthly, 1948+
    at_cash_df = pd.read_csv(os.path.join(DATA, 'at_cash_rate.csv')).set_index('year')
    anchor = pd.Period(f'{SWISS_PROXY_LAST_YEAR}-12', 'M')
    assert anchor in at_fx.index and anchor in at_cpi.index, 'need Dec 1957 EUR/USD and Austrian CPI'
    assert int(at_cash_df.index.min()) == CASH_SWISS_PROXY_LAST_YEAR + 1, 'Austrian cash data must start in 2000'

    # Inflation: Swiss CPI (1921-1957) chained onto the Austrian CPI at Dec 1957, so every
    # pre-1958 inflation rate IS the Swiss rate and every later one is Austrian.
    swiss_leg = ch_cpi[ch_cpi.index <= anchor] * (at_cpi[anchor] / ch_cpi[anchor])
    cpi = pd.concat([swiss_leg, at_cpi[at_cpi.index > anchor]]).sort_index()

    # EUR/USD before 1958: relative-PPP-implied from Swiss vs US CPI, anchored to the market
    # rate at Dec 1957 and interpolated geometrically between Decembers (no market rate exists
    # for a currency that did not exist; the Austrian schilling history is unusable pre-1957).
    ppp_dec = {y: at_fx[anchor]
               * (dec_value(ch_cpi, y) / dec_value(ch_cpi, SWISS_PROXY_LAST_YEAR))
               / (dec_value(us_cpi, y) / dec_value(us_cpi, SWISS_PROXY_LAST_YEAR))
               for y in range(START_YEAR - 1, SWISS_PROXY_LAST_YEAR + 1)}
    synth_idx = [pd.Period(f'{START_YEAR - 1}-12', 'M')]
    synth_val = [ppp_dec[START_YEAR - 1]]
    for y in range(START_YEAR, SWISS_PROXY_LAST_YEAR + 1):
        for m in range(1, 13):
            synth_idx.append(pd.Period(f'{y}-{m:02d}', 'M'))
            synth_val.append(ppp_dec[y - 1] * (ppp_dec[y] / ppp_dec[y - 1]) ** (m / 12.0))
    fx = pd.concat([pd.Series(synth_val, index=pd.PeriodIndex(synth_idx)), at_fx[at_fx.index > anchor]]).sort_index()

    # EUR Cash (annual): Swiss savings rate as-is through 1957, PPP-translated Swiss savings
    # rate 1958-1999, ECB MIR Austrian household agreed-maturity (<= 1 year) rate from 2000.
    rates = {}
    for y in range(START_YEAR, int(at_cash_df.index.max()) + 1):
        if y > CASH_SWISS_PROXY_LAST_YEAR:
            rates[y] = float(at_cash_df.loc[y, 'retail_rate'])
        elif y <= SWISS_PROXY_LAST_YEAR:
            rates[y] = float(ch_cash_df.loc[y, 'retail_rate'])
        else:
            pi_home = dec_value(cpi, y) / dec_value(cpi, y - 1) - 1.0
            pi_ch = dec_value(ch_cpi, y) / dec_value(ch_cpi, y - 1) - 1.0
            rates[y] = (1.0 + float(ch_cash_df.loc[y, 'retail_rate'])) * (1.0 + pi_home) / (1.0 + pi_ch) - 1.0
    home_cash_rate = pd.Series(rates)
    print(f'EUR: inflation/cash/FX from Swiss data {START_YEAR}-{SWISS_PROXY_LAST_YEAR}; '
          f'PPP-translated Swiss cash to {CASH_SWISS_PROXY_LAST_YEAR}; Austrian data afterwards')

# ---------------------------------------------------------------- calendar
last_year = min(int(us_idx.index[-1].year), int(exus_idx.index[-1].year),
                int(fx.index[-1].year), int(cpi.index[-1].year),
                int(home_cash_rate.index.max()))
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
    """Month-on-month return of a USD index expressed in the home currency (CHF or EUR)."""
    chf = idx * fx
    return chf.pct_change(fill_method=None)


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
assert monthly_gold.notna().all(), 'gap in gold monthly series'
print(f'gold: {sum(gold_smoothed.values())} years pegged/smoothed, '
      f'{len(years)-sum(gold_smoothed.values())} years real monthly (LBMA)')

# --------------------------------- CHF cash (JST pre-1933, SNB savings deposits 1933+)
monthly_cash = pd.Series(index=months, dtype=float)
for y in years:
    sel = [p for p in months if p.year == y]
    ann_cash = float(home_cash_rate.loc[y])
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

# Real CHF appreciation beyond Relative PPP over the whole sample, in the (1 - a)
# convention used by `_apply_fx_ppp_adjustment`:
#   (1 + FX drift) = (1 + PPP-implied drift) * (1 - a)
# where the PPP-implied drift is the Swiss-vs-US CPI differential. Derived here
# rather than hard-coded so it stays consistent with the FX and CPI data.
us_cpi_dec = us_cpi[us_cpi.index.month == 12]
us_cpi_dec.index = us_cpi_dec.index.year
n_years = len(years)
fx_drift = (fx_dec[years[-1]] / fx_dec[years[0] - 1]) ** (1.0 / n_years) - 1.0
ppp_drift = ((cpi_dec[years[-1]] / cpi_dec[years[0] - 1])
             / (us_cpi_dec[years[-1]] / us_cpi_dec[years[0] - 1])) ** (1.0 / n_years) - 1.0
real_appreciation = round(1.0 - (1.0 + fx_drift) / (1.0 + ppp_drift), 4)
print(f'USD/{CCY} drift {fx_drift*100:.2f}%/yr, PPP-implied {ppp_drift*100:.2f}%/yr '
      f'-> real {CCY} appreciation beyond PPP {real_appreciation*100:.2f}%/yr')

print(f'\nUS   {CCY}: CAGR {np.prod(1+us_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {us_chf_a.std(ddof=1)*100:5.2f}%  monthly vol {monthly_us.std(ddof=1)*100:.2f}%')
print(f'exUS {CCY}: CAGR {np.prod(1+exus_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {exus_chf_a.std(ddof=1)*100:5.2f}%  monthly vol {monthly_exus.std(ddof=1)*100:.2f}%')
print(f'gold {CCY}: CAGR {np.prod(1+gold_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {gold_chf_a.std(ddof=1)*100:5.2f}%')
print(f'cash {CCY}: CAGR {np.prod(1+cash_chf_a)**(1/len(years))*100-100:6.2f}%  '
      f'ann.vol {cash_chf_a.std(ddof=1)*100:5.2f}%')
print(f'{"CH" if CCY == "CHF" else "AT"} CPI  : CAGR {np.prod(1+ch_inf_a)**(1/len(years))*100-100:6.2f}%')
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
# 6. CHF Cash -- data/ch_cash_rate.csv (`retail_rate`).
#      {START_YEAR}-1932 : JST R6 Swiss `bill_rate` (in this era identical to the savings rate).
#      1933-{last_year} : SNB savings-deposit rate for private clients (data.snb.ch cube
#                         `zikrepro`, D1=S1), annual mean of monthly observations -- the rate a
#                         Swiss retail saver actually earned. Never negative, so no 0% floor.
#    The wholesale JST bill / SNB SARON rate is kept in the CSV (`wholesale_rate`) for reference.
#
# 7. Bitcoin -- SYNTHETIC (lognormal, 7% nominal / 50% vol default, correlated
#    with US equities at default rho=0.50 via Gaussian copula on full-sample
#    standardized monthly US equity log-returns).
#
# 8. US CPI -- data/us_cpi.csv, monthly: FRED CPIAUCNS (BLS CPI-U, NSA) from 1913,
#    legacy wichtounet/swr-calculator leg for 1871-1912.
#    Used ONLY to derive HISTORIC_REAL_CHF_APPRECIATION (PPP residual); it does
#    not enter any return series.
#
# CHF conversion: Return_CHF = (1 + Return_USD) * (FX_end / FX_start) - 1
'''

if CCY == 'CHF':
    T_INFL_NAME = 'HISTORIC_SWISS_INFLATION'
    T_PPP_KW = 'real_chf_appreciation'
    T_CASH_COMMENT = '# CHF cash, retail savings-deposit yield (JST CHE bill_rate pre-1933, SNB savings deposits 1933+).'
    T_CASH_DOC = '    2: CHF Cash       (empirical retail savings-deposit rate: JST pre-1933, SNB zikrepro 1933+)'
    T_INFL_DOC = '    containing the true contiguous Swiss CPI inflation sequence for each historical cohort.'
    T_PPP_COMMENT = (
        f'# Historical real CHF appreciation beyond Relative Purchasing Power Parity\n'
        f'# ({years[0]}-{years[-1]}), DERIVED AT BUILD TIME from data/usd_chf.csv, data/ch_inflation.csv\n'
        f'# and data/us_cpi.csv. USD/CHF drifted {fx_drift*100:.2f}%/yr while the Swiss-vs-US CPI\n'
        f'# differential implied {ppp_drift*100:.2f}%/yr, leaving a -{real_appreciation*100:.2f}%/yr excess\n'
        f'# real FX drag on foreign-priced assets. Setting real_chf_appreciation=0.0 removes it.'
    )
    T_EXTRA_ARRAYS = ''
else:
    T_INFL_NAME = 'HISTORIC_AUSTRIAN_INFLATION'
    T_PPP_KW = 'real_eur_appreciation'
    T_CASH_COMMENT = (f'# EUR cash. {years[0]}-{SWISS_PROXY_LAST_YEAR}: Swiss savings rate (SWISS DATA); '
                      f'{SWISS_PROXY_LAST_YEAR + 1}-{CASH_SWISS_PROXY_LAST_YEAR}: Swiss savings rate\n'
                      f'# translated to EUR under relative PPP (SWISS DATA); {CASH_SWISS_PROXY_LAST_YEAR + 1}+: '
                      f'ECB MIR Austrian household deposits, agreed maturity <= 1y.')
    T_CASH_DOC = (f'    2: EUR Cash       (Swiss savings rate to {SWISS_PROXY_LAST_YEAR}, PPP-translated Swiss '
                  f'savings rate to {CASH_SWISS_PROXY_LAST_YEAR}, ECB MIR AT after)')
    T_INFL_DOC = (f'    containing the contiguous Austrian inflation sequence for each historical cohort\n'
                  f'    (SWISS CPI for {years[0]}-{SWISS_PROXY_LAST_YEAR}, Austrian CPI afterwards).')
    T_PPP_COMMENT = (
        f'# Historical real EUR appreciation beyond Relative Purchasing Power Parity\n'
        f'# ({years[0]}-{years[-1]}), DERIVED AT BUILD TIME from the composite EUR/USD and inflation series\n'
        f'# and data/us_cpi.csv. USD/EUR drifted {fx_drift*100:.2f}%/yr while the home-vs-US CPI\n'
        f'# differential implied {ppp_drift*100:.2f}%/yr, leaving a -{real_appreciation*100:.2f}%/yr excess\n'
        f'# real FX drag on foreign-priced assets. Setting real_eur_appreciation=0.0 removes it.\n'
        f'# The {years[0]}-{SWISS_PROXY_LAST_YEAR} segment is PPP-neutral by construction, so the whole residual\n'
        f'# comes from the {SWISS_PROXY_LAST_YEAR + 1}-{years[-1]} market-rate segment.'
    )
    swiss_flag = {y: y <= SWISS_PROXY_LAST_YEAR for y in years}
    cash_flag = {y: y <= CASH_SWISS_PROXY_LAST_YEAR for y in years}
    T_EXTRA_ARRAYS = f'''
# SWISS-DATA STAND-INS (PRD §8.6.F). True where that calendar year's inflation
# (HISTORIC_AUSTRIAN_INFLATION) is the Swiss CPI, its EUR Cash return is the Swiss savings
# rate and its EUR/USD rate is PPP-implied rather than observed ({years[0]}-{SWISS_PROXY_LAST_YEAR}).
SWISS_PROXY_LAST_YEAR = {SWISS_PROXY_LAST_YEAR}
HISTORIC_YEAR_IS_SWISS_PROXY = np.array([
{fmt_bool(swiss_flag)}
])

# True where that calendar year's EUR Cash return is derived from the Swiss savings rate
# (as-is to {SWISS_PROXY_LAST_YEAR}, translated to EUR under relative PPP to {CASH_SWISS_PROXY_LAST_YEAR}).
CASH_SWISS_PROXY_LAST_YEAR = {CASH_SWISS_PROXY_LAST_YEAR}
HISTORIC_YEAR_CASH_IS_SWISS_PROXY = np.array([
{fmt_bool(cash_flag)}
])
'''
    header = f'''# ============================================================================
# GENERATED FILE - do not edit by hand.
# Regenerate with: python scripts/build_historic_returns.py --currency EUR
# (run scripts/fetch_source_data.py and scripts/fetch_source_data_at.py first)
# ============================================================================
#
# EUR return series for the AUSTRIA jurisdiction (built {pd.Timestamp.today():%Y-%m-%d}).
# Same public API as src/historic_returns.py (CHF), with EUR names.
#
# !!! SWISS DATA STANDS IN FOR AUSTRIA BEFORE 1958 (and for cash before 2000) !!!
#   {years[0]}-{SWISS_PROXY_LAST_YEAR}: inflation = Swiss CPI (data/ch_inflation.csv); EUR Cash = Swiss
#              savings rate (data/ch_cash_rate.csv); EUR/USD = relative-PPP-implied
#              (Swiss vs US CPI), not a market rate. No Austrian data enters.
#              See HISTORIC_YEAR_IS_SWISS_PROXY.
#   {SWISS_PROXY_LAST_YEAR + 1}-{CASH_SWISS_PROXY_LAST_YEAR}: EUR Cash = Swiss savings rate translated to EUR under relative
#              PPP, (1 + r_CH) * (1 + pi_AT) / (1 + pi_CH) - 1, because no open Austrian retail
#              savings-rate series exists before 2000. See HISTORIC_YEAR_CASH_IS_SWISS_PROXY.
#   Austria's 1921-22 hyperinflation and 1945-47 currency reforms are therefore NOT in this history.
#
# 1. US Stocks / 2. Non-US Equities / 5. Gold -- the same USD sources as the CHF module
#    (data/us_stocks.csv, data/jst_exus_usd.csv + data/ex_us_stocks.csv, data/gold_usd.csv),
#    converted to EUR at index level with the composite EUR/USD rate below.
#
# 3. EUR/USD -- data/usd_eur.csv, month-end EUR per USD from {SWISS_PROXY_LAST_YEAR}-12: BIS WS_XRU M.AT.EUR.E
#    (Austrian schilling converted at the fixed 13.7603 ATS/EUR before 1999; identical to the
#    ECB reference rate month-ends from 1999). Before that: PPP-implied, see above.
#
# 4. Inflation -- data/at_inflation.csv (BIS long CPI M.AT.628, Statistik Austria) from
#    {SWISS_PROXY_LAST_YEAR + 1}; Swiss CPI chained on at Dec {SWISS_PROXY_LAST_YEAR} before that. Only December values are used.
#
# 6. EUR Cash -- see above; {CASH_SWISS_PROXY_LAST_YEAR + 1}+ from data/at_cash_rate.csv (`retail_rate`): ECB MIR
#    M.AT.B.L22.F.R.A.2250.EUR.N, Austrian household deposits with agreed maturity up to
#    1 year (new business), annual mean of monthly observations.
#
# 7. Bitcoin -- SYNTHETIC (lognormal, 7% nominal / 50% vol default, correlated
#    with US equities at default rho=0.50 via Gaussian copula on full-sample
#    standardized monthly US equity log-returns).
#
# EUR conversion: Return_EUR = (1 + Return_USD) * (FX_end / FX_start) - 1
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

# US equities, total return, {CCY}. Real month-end data throughout.
MONTHLY_US_{CCY} = np.array([
{fmt(monthly_us.values)}
])

# Non-US equities, total return, {CCY}. JST (smoothed) before {EXUS_REAL_FROM}, real monthly after.
MONTHLY_NON_US_{CCY} = np.array([
{fmt(monthly_exus.values)}
])

{T_CASH_COMMENT}
MONTHLY_CASH_{CCY} = np.array([
{fmt(monthly_cash.values)}
])

# Gold, {CCY}. Pegged/smoothed before {GOLD_REAL_FROM}, real LBMA month-end after.
MONTHLY_GOLD_{CCY} = np.array([
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

HISTORIC_RETURNS_US_{CCY} = np.array([
{fmt(us_chf_a)}
])

HISTORIC_RETURNS_NON_US_{CCY} = np.array([
{fmt(exus_chf_a)}
])

HISTORIC_RETURNS_CASH_{CCY} = np.array([
{fmt(cash_chf_a)}
])

HISTORIC_RETURNS_US_USD = np.array([
{fmt(us_usd_a)}
])

HISTORIC_RETURNS_NON_US_USD = np.array([
{fmt(exus_usd_a)}
])

HISTORIC_RETURNS_GOLD_{CCY} = np.array([
{fmt(gold_chf_a)}
])

{T_INFL_NAME} = np.array([
{fmt(ch_inf_a)}
])

HISTORIC_USD_{CCY}_FX = np.array([
{fmt(fx_a)}
])
{T_EXTRA_ARRAYS}
# Backwards compatibility alias
HISTORIC_RETURNS = HISTORIC_RETURNS_US_{CCY}

# Asset column order used by every return matrix in this module.
ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_{CCY}_CASH, ASSET_GOLD, ASSET_BITCOIN = range(5)
NUM_ASSETS = 5

# Bitcoin: synthetic lognormal coupled to realized US equity log-returns.
BITCOIN_NOMINAL_MEAN = 0.07
BITCOIN_VOL = 0.50
DEFAULT_BTC_EQUITY_CORR = 0.50

# Politis-White (2004) optimal stationary bootstrap mean block length (years).
POLITIS_WHITE_BLOCK_YEARS = 5

{T_PPP_COMMENT}
HISTORIC_REAL_{CCY}_APPRECIATION = {real_appreciation:.4f}
FOREIGN_ASSET_COLS = (ASSET_US_STOCKS, ASSET_NON_US_STOCKS, ASSET_GOLD)

# Full-sample (1922-{last_year}) monthly US equity log-return moments used to standardize
# US shocks in the Bitcoin Gaussian copula without per-path sum-to-zero variance collapse
# or look-ahead bias, while keeping unconditional E[z_us] == 0 (no secular drift leak).
_US_LOG_RETURNS = np.log1p(MONTHLY_US_{CCY})
_US_LOG_MEAN = float(np.mean(_US_LOG_RETURNS))
_US_LOG_STD = float(np.std(_US_LOG_RETURNS, ddof=1))


def _generate_lognormal_monthly_returns(
    rng: np.random.Generator,
    ann_ret: float,
    ann_vol: float,
    shape: tuple | int,
    us_monthly_slice: np.ndarray | None = None,
    btc_equity_corr: float = 0.0,
    ref_log_mean: float = _US_LOG_MEAN,
    ref_log_std: float = _US_LOG_STD,
) -> np.ndarray:
    """Generates monthly returns using a lognormal model with Ito drift correction.

    When us_monthly_slice and a non-zero btc_equity_corr are provided, couples the standard
    normal shocks to full-sample-standardized monthly US equity log-returns with correlation
    rho = btc_equity_corr, preserving Bitcoin's target multi-decade drift, volatility, and
    horizon-level equity correlation without secular US drift leak or look-ahead bias.
    """
    if not np.isfinite(ann_ret) or ann_ret <= -1.0:
        raise ValueError(f"ann_ret ({{ann_ret}}) must be greater than -1.0 and finite.")
    if not np.isfinite(ann_vol) or ann_vol < 0.0:
        raise ValueError(f"ann_vol ({{ann_vol}}) cannot be negative and must be finite.")
    if not np.isfinite(btc_equity_corr) or not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")
    drift = np.log(1.0 + ann_ret) - 0.5 * (ann_vol ** 2)
    eps = rng.normal(0.0, 1.0, shape)
    if us_monthly_slice is not None and abs(btc_equity_corr) > 1e-12:
        us_log = np.log(np.maximum(1e-8, 1.0 + us_monthly_slice))
        z_us = (us_log - ref_log_mean) / max(1e-12, float(ref_log_std))
        z = btc_equity_corr * z_us + np.sqrt(max(0.0, 1.0 - btc_equity_corr ** 2)) * eps
    else:
        z = eps
    log_ret = (drift / 12.0) + (ann_vol / np.sqrt(12.0)) * z
    return np.exp(log_ret) - 1.0


def _apply_fx_ppp_adjustment(
    matrix: np.ndarray,
    {T_PPP_KW}: float | None
) -> np.ndarray:
    """Adjusts foreign-priced asset sleeves for target real {CCY} appreciation beyond PPP."""
    if {T_PPP_KW} is None:
        return matrix
    if not np.isfinite({T_PPP_KW}) or {T_PPP_KW} >= 1.0:
        raise ValueError(
            f"{T_PPP_KW} ({{{T_PPP_KW}}}) must be finite and strictly less than 1.0."
        )
    ann_adj = (1.0 - {T_PPP_KW}) / (1.0 - HISTORIC_REAL_{CCY}_APPRECIATION)
    month_adj = ann_adj ** (1.0 / 12.0)
    if np.isclose(month_adj, 1.0, atol=1e-12):
        return matrix
    for col in FOREIGN_ASSET_COLS:
        matrix[:, :, col] = (1.0 + matrix[:, :, col]) * month_adj - 1.0
    return matrix


def ppp_adjusted_annual_returns(
    annual_returns: np.ndarray,
    {T_PPP_KW}: float,
) -> np.ndarray:
    """Annual {CCY} returns of a foreign-priced sleeve under a target real {CCY} appreciation beyond PPP.

    Annual counterpart of `_apply_fx_ppp_adjustment`: compounding the twelve adjusted monthly
    returns of a year gives exactly `(1 + R) * (1 - target) / (1 - HISTORIC_REAL_{CCY}_APPRECIATION) - 1`.
    Used to calibrate the Monte Carlo default means on the same FX assumption that the
    Historic Backtesting and Bootstrapping engines run under. Annual log-return volatility
    is invariant to this rescaling, so only the means move.
    """
    if not np.isfinite({T_PPP_KW}) or {T_PPP_KW} >= 1.0:
        raise ValueError(
            f"{T_PPP_KW} ({{{T_PPP_KW}}}) must be finite and strictly less than 1.0."
        )
    ann_adj = (1.0 - {T_PPP_KW}) / (1.0 - HISTORIC_REAL_{CCY}_APPRECIATION)
    return (1.0 + np.asarray(annual_returns, dtype=float)) * ann_adj - 1.0


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
    if duration_years > total_years:
        raise ValueError(f"duration_years ({{duration_years}}) cannot exceed total_years ({{total_years}}).")
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
    {T_PPP_KW}: float | None = None,
    btc_equity_corr: float = DEFAULT_BTC_EQUITY_CORR,
    btc_mean: float = BITCOIN_NOMINAL_MEAN,
    btc_vol: float = BITCOIN_VOL,
) -> np.ndarray:
    """
    Returns a return matrix of shape (num_runs, duration_months, 5) for 5 asset classes:
    0: US Stocks      (empirical S&P 500 total return in {CCY}, real monthly)
    1: Non-US Stocks  (JST GDP-weighted ex-US before {EXUS_REAL_FROM}, real monthly after)
{T_CASH_DOC}
    3: Gold           (LBMA in {CCY}; pegged before {GOLD_REAL_FROM})
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
    if not np.isfinite(btc_equity_corr) or not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")

    num_runs = total_years - duration_years + 1
    duration_months = duration_years * 12
    matrix = np.zeros((num_runs, duration_months, NUM_ASSETS))

    for i in range(num_runs):
        a, b = i * 12, i * 12 + duration_months
        us_slice = MONTHLY_US_{CCY}[a:b]
        matrix[i, :, ASSET_US_STOCKS] = us_slice
        matrix[i, :, ASSET_NON_US_STOCKS] = MONTHLY_NON_US_{CCY}[a:b]
        matrix[i, :, ASSET_{CCY}_CASH] = MONTHLY_CASH_{CCY}[a:b]
        matrix[i, :, ASSET_GOLD] = MONTHLY_GOLD_{CCY}[a:b]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, i]))
        matrix[i, :, ASSET_BITCOIN] = _generate_lognormal_monthly_returns(
            rng,
            btc_mean,
            btc_vol,
            duration_months,
            us_monthly_slice=us_slice,
            btc_equity_corr=btc_equity_corr,
        )

    return _apply_fx_ppp_adjustment(matrix, {T_PPP_KW})


def get_historic_inflation_matrix(duration_years: int) -> np.ndarray:
    """
    Returns an empirical inflation matrix of shape (num_runs, duration_years)
{T_INFL_DOC}
    """
    total_years = len({T_INFL_NAME})
    if duration_years <= 0:
        raise ValueError(f"Duration must be a positive integer, got {{duration_years}}.")
    if duration_years > total_years:
        raise ValueError(f"Duration {{duration_years}} exceeds available historic data ({{total_years}} years).")

    num_runs = total_years - duration_years + 1
    matrix = np.zeros((num_runs, duration_years))
    for i in range(num_runs):
        matrix[i, :] = {T_INFL_NAME}[i: i + duration_years]
    return matrix


def _sample_bootstrap_year_indices(
    rng: np.random.Generator,
    num_runs: int,
    duration_years: int,
    total_years: int,
    block_size_years: int,
    stationary: bool,
) -> np.ndarray:
    """Samples calendar-year indices (shape `(num_runs, duration_years)`) for block bootstrapping."""
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
        return year_idx
    num_blocks = int(np.ceil(duration_years / block_size_years))
    max_start_idx = total_years - block_size_years
    start_indices = rng.integers(0, max_start_idx + 1, size=(num_runs, num_blocks))
    offsets = np.arange(block_size_years)
    return (start_indices[:, :, None] + offsets[None, None, :]).reshape(num_runs, -1)[:, :duration_years]


def generate_bootstrapped_data(
    num_runs: int,
    duration_years: int,
    seed: int = 42,
    block_size_years: int = POLITIS_WHITE_BLOCK_YEARS,
    {T_PPP_KW}: float | None = None,
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
    if not np.isfinite(btc_equity_corr) or not (-1.0 <= btc_equity_corr <= 1.0):
        raise ValueError(f"btc_equity_corr ({{btc_equity_corr}}) must be between -1.0 and 1.0.")

    rng = np.random.default_rng(seed)
    duration_months = duration_years * 12
    return_matrix = np.zeros((num_runs, duration_months, NUM_ASSETS))
    year_idx = _sample_bootstrap_year_indices(
        rng, num_runs, duration_years, total_years, block_size_years, stationary
    )

    # Expand each sampled calendar year into its 12 real monthly observations.
    month_idx = (year_idx[:, :, None] * 12 + np.arange(12)[None, None, :]
                 ).reshape(num_runs, duration_months)

    us_sampled = MONTHLY_US_{CCY}[month_idx]
    return_matrix[:, :, ASSET_US_STOCKS] = us_sampled
    return_matrix[:, :, ASSET_NON_US_STOCKS] = MONTHLY_NON_US_{CCY}[month_idx]
    return_matrix[:, :, ASSET_{CCY}_CASH] = MONTHLY_CASH_{CCY}[month_idx]
    return_matrix[:, :, ASSET_GOLD] = MONTHLY_GOLD_{CCY}[month_idx]
    return_matrix[:, :, ASSET_BITCOIN] = _generate_lognormal_monthly_returns(
        rng,
        btc_mean,
        btc_vol,
        (num_runs, duration_months),
        us_monthly_slice=us_sampled,
        btc_equity_corr=btc_equity_corr,
    )

    _apply_fx_ppp_adjustment(return_matrix, {T_PPP_KW})
    return return_matrix, {T_INFL_NAME}[year_idx]


def generate_bootstrapped_returns(
    num_runs: int,
    duration_years: int,
    seed: int = 42,
    block_size_years: int = POLITIS_WHITE_BLOCK_YEARS,
    {T_PPP_KW}: float | None = None,
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
        {T_PPP_KW}={T_PPP_KW},
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
    """Inflation-matrix-only wrapper sharing the exact block-sampling RNG path of generate_bootstrapped_data."""
    total_years = len(HISTORIC_YEARS)
    if duration_years <= 0 or num_runs <= 0:
        raise ValueError(f"num_runs ({{num_runs}}) and duration_years ({{duration_years}}) must be positive integers.")
    if seed < 0:
        raise ValueError(f"seed ({{seed}}) must be a non-negative integer.")
    if block_size_years <= 0 or block_size_years > total_years:
        raise ValueError(f"block_size_years ({{block_size_years}}) must be between 1 and {{total_years}}.")
    rng = np.random.default_rng(seed)
    year_idx = _sample_bootstrap_year_indices(
        rng, num_runs, duration_years, total_years, block_size_years, stationary
    )
    return {T_INFL_NAME}[year_idx]
'''

with open(SRC_FILE, 'w') as f:
    f.write(code)
print(f'\nWrote {SRC_FILE}')

