"""Fetch and normalise all external source data for the simulator.

Network step. Run this before `build_historic_returns.py`.

Produces, in data/:
  usd_chf.csv       month-end USD/CHF        1914-present  (SNB pre-1971, FRED DEXSZUS after)
  gold_usd.csv      month-end gold USD/oz    1871-present  (fixed peg pre-1968, LBMA PM fix after)
  jst_exus_usd.csv  annual ex-US equity TR   1900-1969     (GDP-weighted, from JST R6)
  ch_cash_rate.csv  annual CHF cash rates    1900-present  (retail: JST CHE bill_rate pre-1933, SNB savings
                                                            deposit rate 1933+; wholesale: JST + SNB SARON)
  us_cpi.csv        monthly US CPI-U          1871-present  (legacy swr-calculator leg pre-1913, FRED CPIAUCNS after)

Sources (all free, verified 2026-09-23; SNB savings + FRED CPI verified 2026-09-24):
  FRED  DEXSZUS  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSZUS
  FRED  CPIAUCNS https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS
  SNB   devkum   https://data.snb.ch/api/cube/devkum/data/csv/en
  SNB   zikrepro https://data.snb.ch/api/cube/zikrepro/data/csv/en  (D1=S1: private-client savings deposits)
  LBMA           https://prices.lbma.org.uk/json/gold_pm.json
  JST R6         https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.xlsx
"""
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
CACHE = os.path.join(BASE, '.cache')
os.makedirs(CACHE, exist_ok=True)

FRED_DEXSZUS = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSZUS'
FRED_CPIAUCNS = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS'
SNB_DEVKUM = 'https://data.snb.ch/api/cube/devkum/data/csv/en'
SNB_SAVINGS = 'https://data.snb.ch/api/cube/zikrepro/data/csv/en?dimSel=D0(M),D1(S1)&fromDate=1900-01'
LBMA_GOLD = 'https://prices.lbma.org.uk/json/gold_pm.json'
JST_R6 = 'https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.xlsx'
LAST_YEAR = 2025  # last complete calendar year carried by the simulator


def fetch(url, name, binary=False):
    """Download with a local cache so reruns are offline-friendly."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f'  [cache] {name}')
    else:
        print(f'  [fetch] {name} <- {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, 'wb') as f:
            f.write(r.read())
    return open(path, 'rb').read() if binary else open(path, encoding='utf-8', errors='replace').read()


def month_end(df, date_col, val_col):
    df = df.dropna(subset=[val_col]).sort_values(date_col)
    return df.groupby(df[date_col].dt.to_period('M'))[val_col].last()


# --------------------------------------------------------------------------
print('1. USD/CHF')
# (a) FRED daily -> true month-end, 1971-01 onwards
fred = pd.read_csv(io.StringIO(fetch(FRED_DEXSZUS, 'dexszus.csv')), na_values=['.'])
fred.columns = ['date', 'rate']
fred['date'] = pd.to_datetime(fred.date)
fx_modern = month_end(fred, 'date', 'rate')
print(f'   FRED month-end: {fx_modern.index.min()} -> {fx_modern.index.max()} ({len(fx_modern)})')

# (b) SNB monthly average for the pre-1971 peg era (no month-end exists; the rate was fixed)
rows, started = [], False
for line in fetch(SNB_DEVKUM, 'snb_devkum.csv').splitlines():
    p = [x.strip('"').strip() for x in line.split(';')]
    if not started:
        started = p[:1] == ['Date']
        continue
    if len(p) >= 4 and p[1] == 'M0' and p[2] == 'USD1' and p[3]:
        rows.append((p[0], float(p[3])))
snb = pd.DataFrame(rows, columns=['ym', 'rate']).drop_duplicates('ym')
snb['ym'] = pd.PeriodIndex(snb.ym, freq='M')
fx_old = snb.set_index('ym').rate
fx_old = fx_old[fx_old.index < pd.Period('1971-01', 'M')]
print(f'   SNB monthly avg (peg era): {fx_old.index.min()} -> {fx_old.index.max()} ({len(fx_old)})')

fx = pd.concat([fx_old, fx_modern]).sort_index()
fx = fx[~fx.index.duplicated(keep='last')]
out = pd.DataFrame({'month': fx.index.month, 'year': fx.index.year, 'v': fx.values})
out.to_csv(os.path.join(DATA, 'usd_chf.csv'), index=False, header=False, float_format='%.4f')
print(f'   -> data/usd_chf.csv  {fx.index.min()} - {fx.index.max()}  ({len(fx)} months)')

# --------------------------------------------------------------------------
print('2. Gold (USD/oz)')
lb = json.loads(fetch(LBMA_GOLD, 'lbma_gold.json'))
g = pd.DataFrame([{'date': r['d'], 'usd': r['v'][0]} for r in lb if r.get('v') and r['v'][0]])
g['date'] = pd.to_datetime(g.date)
gold_modern = month_end(g, 'date', 'usd')
print(f'   LBMA PM fix: {gold_modern.index.min()} -> {gold_modern.index.max()} ({len(gold_modern)})')

# Pre-1968 the price was fixed by statute ($20.67 then $35). Carry the legacy
# series forward so the CHF result still reflects the 1933-36 devaluations,
# which a Swiss gold holder genuinely experienced.
# Prefer the original upstream file; fall back to our own previous output so
# this script stays runnable after data/gold.csv is pruned.
legacy_path = next((p for p in (os.path.join(DATA, 'gold.csv'),
                                os.path.join(DATA, 'gold_usd.csv'))
                    if os.path.exists(p)), None)
if legacy_path:
    lg = pd.read_csv(legacy_path, header=None, names=['m', 'y', 'v'])
    lg['ym'] = pd.PeriodIndex(pd.to_datetime(dict(year=lg.y, month=lg.m, day=1)), freq='M')
    gold_old = lg.set_index('ym').v
    gold_old = gold_old[gold_old.index < pd.Period('1968-04', 'M')]
    print(f'   legacy pre-1968 leg from {os.path.basename(legacy_path)} ({len(gold_old)} months)')
    gold = pd.concat([gold_old, gold_modern]).sort_index()
else:
    print('   WARNING: no legacy pre-1968 gold source found; series starts 1968-04')
    gold = gold_modern
gold = gold[~gold.index.duplicated(keep='last')]
out = pd.DataFrame({'month': gold.index.month, 'year': gold.index.year, 'v': gold.values})
out.to_csv(os.path.join(DATA, 'gold_usd.csv'), index=False, header=False, float_format='%.4f')
print(f'   -> data/gold_usd.csv  {gold.index.min()} - {gold.index.max()}  ({len(gold)} months)')

# --------------------------------------------------------------------------
print('3. JST ex-US equities (annual, USD)')
raw = fetch(JST_R6, 'JSTdatasetR6.dta', binary=True)
jst = pd.read_stata(io.BytesIO(raw))
print(f'   JST R6: {jst.shape[0]} rows, {jst.country.nunique()} countries, '
      f'{int(jst.year.min())}-{int(jst.year.max())}')

d = jst[['year', 'country', 'iso', 'eq_tr', 'xrusd', 'gdp', 'cpi']].copy()
# eq_tr is nominal, in LOCAL currency. xrusd = local currency units per USD.
# A USD investor's return = (1 + eq_tr_local) * (xrusd_{t-1} / xrusd_t) - 1
d['fx_vs_usd'] = d.groupby('country', observed=True).xrusd.transform(lambda s: s.shift(1) / s - 1)
d['tr_usd'] = (1.0 + d.eq_tr) * (1.0 + d.fx_vs_usd) - 1.0

exus = d[(d.iso != 'USA')].dropna(subset=['tr_usd', 'gdp'])
exus = exus[(exus.year >= 1900) & (exus.year <= 1969)]
agg = exus.groupby('year').apply(
    lambda x: pd.Series({
        'tr_usd': np.average(x.tr_usd, weights=x.gdp),
        'n_countries': len(x),
    }), include_groups=False)
agg = agg.reset_index()
agg['year'] = agg.year.astype(int)
agg['n_countries'] = agg.n_countries.astype(int)
agg.to_csv(os.path.join(DATA, 'jst_exus_usd.csv'), index=False, float_format='%.6f')
print(f'   -> data/jst_exus_usd.csv  {agg.year.min()}-{agg.year.max()} '
      f'({len(agg)} years, {agg.n_countries.min()}-{agg.n_countries.max()} countries/yr)')
cagr = np.prod(1 + agg.tr_usd) ** (1 / len(agg)) - 1
print(f'      nominal USD CAGR {cagr*100:.2f}%, vol {agg.tr_usd.std(ddof=1)*100:.2f}%')
print('      worst years: '
      + ', '.join(f'{int(r.year)} {r.tr_usd*100:.1f}%'
                  for r in agg.nsmallest(3, 'tr_usd').itertuples()))

# Cross-check: JST Swiss CPI vs the FSO series we already have, on the overlap.
ch = jst[jst.iso == 'CHE'][['year', 'cpi']].dropna()
fso = pd.read_csv(os.path.join(DATA, 'ch_inflation.csv'), header=None, names=['m', 'y', 'v'])
fso_dec = fso[fso.m == 12].set_index('y').v
j = ch.set_index('year').cpi
common = sorted(set(j.index.astype(int)) & set(fso_dec.index))
a = j.loc[common].pct_change().dropna()
b = fso_dec.loc[common].pct_change().dropna()
print(f'   JST vs FSO Swiss CPI on {len(common)} overlapping years: '
      f'corr={a.corr(b):.4f}, mean abs diff={np.abs(a-b).mean()*100:.3f}pp')

# --------------------------------------------------------------------------
print('4. Swiss short-term cash rate (annual, CHF)')
che_cash = jst[(jst.iso == 'CHE') & (jst.year >= 1900)][['year', 'bill_rate']].dropna().copy()
che_cash['year'] = che_cash['year'].astype(int)
che_cash = che_cash.rename(columns={'bill_rate': 'wholesale_rate'})

# Spliced 2021-2025 time-weighted SNB policy / SARON money-market rates (wholesale reference only)
snb_modern_cash = pd.DataFrame([
    {'year': 2021, 'wholesale_rate': -0.0075},
    {'year': 2022, 'wholesale_rate': -0.0024},
    {'year': 2023, 'wholesale_rate': 0.0154},
    {'year': 2024, 'wholesale_rate': 0.0135},
    {'year': 2025, 'wholesale_rate': 0.0018},
])
cash_df = pd.concat([che_cash, snb_modern_cash], ignore_index=True).drop_duplicates('year', keep='last').sort_values('year')

# Retail leg: what a Swiss private client actually earns on a savings account. SNB zikrepro
# D1=S1 is monthly from 1933. Pre-1969 it coincides with JST's CHE bill_rate to <0.05pp (JST
# uses it as its proxy); from 1969 JST switches to a volatile money-market rate (e.g. 1989:
# 9.7% vs 3.45% on savings), and during 2015-2022 wholesale was negative while savings paid
# ~0.0x%, so no artificial 0% floor is needed any more.
rows, started = [], False
for line in fetch(SNB_SAVINGS, 'snb_zikrepro_savings.csv').splitlines():
    p = [x.strip('"').strip() for x in line.split(';')]
    if not started:
        started = p[:1] == ['Date']
        continue
    if len(p) >= 4 and p[1] == 'M' and p[2] == 'S1' and p[3]:
        rows.append((int(p[0][:4]), float(p[3]) / 100.0))
sav = pd.DataFrame(rows, columns=['year', 'rate'])
sav = sav[sav.year <= LAST_YEAR]
months_per_year = sav.groupby('year').size()
full_years = months_per_year[months_per_year == 12].index
savings_annual = sav[sav.year.isin(full_years)].groupby('year').rate.mean()
print(f'   SNB savings deposits: {savings_annual.index.min()}-{savings_annual.index.max()} '
      f'({len(savings_annual)} full years), mean {savings_annual.mean()*100:.2f}%')
assert (savings_annual >= 0).all(), 'negative savings deposit rate'
cash_df = cash_df.set_index('year')
cash_df['retail_rate'] = cash_df['wholesale_rate']  # pre-1933: JST (== savings rate in that era)
common = cash_df.index.intersection(savings_annual.index)
cash_df.loc[common, 'retail_rate'] = savings_annual.loc[common]
pre = cash_df.index[(cash_df.index < 1969) & cash_df.index.isin(common)]
print(f'   JST bill_rate vs SNB savings, {pre.min()}-{pre.max()}: mean abs diff '
      f'{(cash_df.loc[pre, "wholesale_rate"] - savings_annual.loc[pre]).abs().mean()*100:.3f}pp')
cash_df = cash_df.reset_index()
cash_df.to_csv(os.path.join(DATA, 'ch_cash_rate.csv'), index=False, float_format='%.6f')
print(f'   -> data/ch_cash_rate.csv  {cash_df.year.min()}-{cash_df.year.max()} ({len(cash_df)} years)')

# --------------------------------------------------------------------------
print('5. US CPI-U (monthly, build-time PPP derivation only)')
cpi = pd.read_csv(io.StringIO(fetch(FRED_CPIAUCNS, 'cpiaucns.csv')), na_values=['.'])
cpi.columns = ['date', 'v']
cpi['date'] = pd.to_datetime(cpi.date)
cpi = cpi.dropna()
cpi = cpi[cpi.date.dt.year <= LAST_YEAR]
cpi_modern = pd.Series(cpi.v.values, index=pd.PeriodIndex(cpi.date, freq='M'))
print(f'   FRED CPIAUCNS: {cpi_modern.index.min()} -> {cpi_modern.index.max()} ({len(cpi_modern)})')
# 1871-1912 predates the BLS index; keep the legacy leg (swr-calculator / Shiller) already in data/.
legacy = pd.read_csv(os.path.join(DATA, 'us_cpi.csv'), header=None, names=['m', 'y', 'v'])
legacy.index = pd.PeriodIndex(pd.to_datetime(dict(year=legacy.y, month=legacy.m, day=1)), freq='M')
overlap = legacy.index.intersection(cpi_modern.index)
if len(overlap):
    rel = (legacy.v.loc[overlap] / cpi_modern.loc[overlap] - 1).abs()
    print(f'   legacy vs FRED on {len(overlap)} overlapping months: max rel diff {rel.max()*100:.3f}%')
us_cpi = pd.concat([legacy.v[legacy.index < cpi_modern.index.min()], cpi_modern]).sort_index()
out = pd.DataFrame({'month': us_cpi.index.month, 'year': us_cpi.index.year, 'v': us_cpi.values})
out.to_csv(os.path.join(DATA, 'us_cpi.csv'), index=False, header=False, float_format='%.3f')
print(f'   -> data/us_cpi.csv  {us_cpi.index.min()} - {us_cpi.index.max()}  ({len(us_cpi)} months)')
print('\nDone.')
