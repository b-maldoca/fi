"""Fetch and normalise the Austrian / EUR source data for the Austrian return series.

Network step for the Austria jurisdiction (PRD §8, Design Doc §7.5). Run this before
`build_historic_returns.py --currency EUR`. The Swiss/CHF sources are fetched by
`fetch_source_data.py`; the EUR build additionally reads data/ch_inflation.csv,
data/ch_cash_rate.csv and data/us_cpi.csv from that step, because the Austrian history
uses SWISS data as a stand-in where no usable Austrian data exists (see below).

Produces, in data/:
  usd_eur.csv        month-end EUR per USD    1957-present  (BIS WS_XRU M.AT.EUR.E: Austrian schilling
                                                             converted at the fixed 13.7603 ATS/EUR before 1999)
  at_inflation.csv   monthly Austrian CPI     1948-present  (Statistik Austria VPI 1958, the official long-chained
                                                             national CPI, from Dec 1959; BIS WS_LONG_CPI M.AT.628
                                                             chained on before that, for Dec 1957 / Dec 1958)
  at_cash_rate.csv   annual EUR deposit rates 2000-present  (ECB MIR, Austrian households, new business:
                                                             retail_rate = agreed maturity up to 1 year;
                                                             overnight_rate / notice_rate for reference only)

Cross-checks printed (not written): ECB daily reference rate month-ends vs the BIS series
(1999+), and the BIS CPI vs the Statistik Austria VPI 1958 (1960+).

Swiss-data stand-ins applied later in build_historic_returns.py --currency EUR (NOT here):
  1922-1957 : Austrian inflation = Swiss CPI; EUR Cash = Swiss savings rate; EUR/USD = PPP-implied.
  1958-1999 : EUR Cash = Swiss savings rate translated to EUR under relative PPP (no open Austrian
              retail savings-rate series exists before 2000; the regulated "Eckzinssatz" is only
              published in OeNB PDFs).

Sources (all free, verified 2026-11-10):
  BIS   WS_XRU       https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/M.AT.EUR.E?format=csv
  BIS   WS_LONG_CPI  https://stats.bis.org/api/v2/data/dataflow/BIS/WS_LONG_CPI/1.0/M.AT.628?format=csv
  ECB   EXR          https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata
  ECB   MIR          https://data-api.ecb.europa.eu/service/data/MIR/M.AT.B.L22.F.R.A.2250.EUR.N  (+ L21, L23)
  Statistik Austria  https://data.statistik.gv.at/data/OGD_vpi58_VPI_1958_i_1.csv
"""
import io
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
CACHE = os.path.join(BASE, '.cache')
os.makedirs(CACHE, exist_ok=True)

BIS_FX_EOP = 'https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/M.AT.EUR.E?format=csv'
BIS_CPI = 'https://stats.bis.org/api/v2/data/dataflow/BIS/WS_LONG_CPI/1.0/M.AT.628?format=csv'
ECB_EXR_DAILY = 'https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata'
ECB_MIR = 'https://data-api.ecb.europa.eu/service/data/MIR/{key}?format=csvdata'
MIR_AGREED_1Y = 'M.AT.B.L22.F.R.A.2250.EUR.N'   # households, agreed maturity <= 1 year, new business
MIR_OVERNIGHT = 'M.AT.B.L21.A.R.A.2250.EUR.N'   # households, overnight deposits
MIR_NOTICE = 'M.AT.B.L23.A.R.A.2250.EUR.N'      # households, redeemable at notice (gaps)
STAT_AT_VPI58 = 'https://data.statistik.gv.at/data/OGD_vpi58_VPI_1958_i_1.csv'
LAST_YEAR = 2025  # last complete calendar year carried by the simulator


def fetch(url, name, attempts=4):
    """Download with an atomic local cache write so reruns are offline-friendly.

    The ECB data API intermittently answers 504 Gateway Time-out, so retry with backoff.
    """
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f'  [cache] {name}')
    else:
        print(f'  [fetch] {name} <- {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_path = f'{path}.tmp.{os.getpid()}'
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp_path, 'wb') as f:
                    f.write(r.read())
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                if attempt == attempts:
                    raise
                print(f'    attempt {attempt} failed ({e}); retrying in {10 * attempt}s')
                time.sleep(10 * attempt)
        os.replace(tmp_path, path)
    return open(path, encoding='utf-8', errors='replace').read()


def sdmx_monthly(text):
    """Monthly SDMX CSV (BIS or ECB) -> Series indexed by pandas monthly Period."""
    df = pd.read_csv(io.StringIO(text))
    df = df.dropna(subset=['OBS_VALUE'])
    return pd.Series(df.OBS_VALUE.astype(float).values,
                     index=pd.PeriodIndex(df.TIME_PERIOD, freq='M')).sort_index()


def write_monthly(series, filename, float_format):
    series = series[series.index.year <= LAST_YEAR]
    out = pd.DataFrame({'month': series.index.month, 'year': series.index.year, 'v': series.values})
    out.to_csv(os.path.join(DATA, filename), index=False, header=False, float_format=float_format)
    print(f'   -> data/{filename}  {series.index.min()} - {series.index.max()}  ({len(series)} months)')


# --------------------------------------------------------------------------
print('1. EUR per USD (month-end)')
fx = sdmx_monthly(fetch(BIS_FX_EOP, 'bis_xru_at_eur_eop.csv'))
print(f'   BIS end-of-period: {fx.index.min()} -> {fx.index.max()} ({len(fx)})')
assert fx.index.min() <= pd.Period('1957-12', 'M'), 'need Dec 1957 to anchor the 1958 FX return'
months = pd.period_range(fx.index.min(), f'{LAST_YEAR}-12', freq='M')
assert fx.reindex(months).notna().all(), 'gap in BIS EUR/USD series'

# Cross-check 1999+ against ECB daily reference rates (USD per EUR -> EUR per USD, month-end).
ecb = pd.read_csv(io.StringIO(fetch(ECB_EXR_DAILY, 'ecb_exr_usd_eur_daily.csv'))).dropna(subset=['OBS_VALUE'])
ecb['date'] = pd.to_datetime(ecb.TIME_PERIOD)
ecb_me = ecb.sort_values('date').groupby(ecb.date.dt.to_period('M')).OBS_VALUE.last()
ecb_me = 1.0 / ecb_me.astype(float)
common = fx.index.intersection(ecb_me.index)
common = common[common.year <= LAST_YEAR]
rel = (fx.loc[common] / ecb_me.loc[common] - 1.0).abs()
print(f'   BIS vs ECB month-end on {len(common)} months (1999+): mean abs {rel.mean()*100:.3f}%, max {rel.max()*100:.3f}%')
assert rel.max() < 0.01, 'BIS and ECB month-end EUR/USD disagree by more than 1%'
write_monthly(fx, 'usd_eur.csv', '%.6f')

# --------------------------------------------------------------------------
print('2. Austrian CPI (monthly)')
# Primary: Statistik Austria's VPI 1958 -- the official long-chained national CPI (each newer
# VPI base linked on), published to 0.1 on a 1958 = 100 base, i.e. ~0.1% precision. It starts in
# March 1959, so Dec 1957 and Dec 1958 (needed for the 1958 and 1959 inflation rates) come from
# the BIS long CPI (sourced from Statistik Austria), chained on at Dec 1959. The BIS monthly levels
# are rounded to 2 decimals on a 2010 = 100 base (~0.9% steps in the 1950s-60s), too coarse to
# use for later years (e.g. 1960: BIS 0.0% vs VPI 0.8%).
bis_cpi = sdmx_monthly(fetch(BIS_CPI, 'bis_long_cpi_at.csv'))
print(f'   BIS long CPI: {bis_cpi.index.min()} -> {bis_cpi.index.max()} ({len(bis_cpi)})')
rows = []
for line in fetch(STAT_AT_VPI58, 'statat_vpi58.csv').splitlines()[1:]:
    p = line.strip().split(';')
    if len(p) == 3 and p[0].startswith('VPIZR-') and len(p[0]) == 12 and p[2]:
        rows.append((pd.Period(f'{p[0][6:10]}-{p[0][10:12]}', 'M'), float(p[2].replace(',', '.'))))
vpi = pd.Series([v for _, v in rows], index=pd.PeriodIndex([k for k, _ in rows])).sort_index()
print(f'   Statistik Austria VPI 1958: {vpi.index.min()} -> {vpi.index.max()} ({len(vpi)})')
link = pd.Period('1959-12', 'M')
assert link in vpi.index and pd.Period('1957-12', 'M') in bis_cpi.index
vpi_months = pd.period_range(link, f'{LAST_YEAR}-12', freq='M')
assert vpi.reindex(vpi_months).notna().all(), 'gap in Statistik Austria VPI 1958'
bis_leg = bis_cpi[bis_cpi.index < link] * (vpi[link] / bis_cpi[link])
cpi = pd.concat([bis_leg, vpi[vpi.index >= link]]).sort_index()

# Cross-check: Dec/Dec inflation of the two series where both exist.
dec_b, dec_v = bis_cpi[bis_cpi.index.month == 12], vpi[vpi.index.month == 12]
yrs = [y for y in sorted(set(dec_b.index.year) & set(dec_v.index.year)) if y <= LAST_YEAR]
inf_b = pd.Series([dec_b[pd.Period(f'{y}-12', 'M')] for y in yrs], index=yrs).pct_change().dropna()
inf_v = pd.Series([dec_v[pd.Period(f'{y}-12', 'M')] for y in yrs], index=yrs).pct_change().dropna()
diff = (inf_b - inf_v).abs()
print(f'   BIS vs VPI 1958 Dec/Dec inflation, {inf_b.index.min()}-{inf_b.index.max()}: mean abs diff '
      f'{diff.mean()*100:.3f}pp, max {diff.max()*100:.3f}pp ({diff.idxmax()})')
write_monthly(cpi, 'at_inflation.csv', '%.6f')

# --------------------------------------------------------------------------
print('3. Austrian household deposit rates (annual, EUR)')


def mir_annual(key, name):
    s = sdmx_monthly(fetch(ECB_MIR.format(key=key), name)) / 100.0
    s = s[s.index.year <= LAST_YEAR]
    counts = s.groupby(s.index.year).size()
    full = counts[counts == 12].index
    return s[s.index.year.isin(full)].groupby(s.index.year[s.index.year.isin(full)]).mean()


agreed = mir_annual(MIR_AGREED_1Y, 'ecb_mir_at_l22_agreed_1y.csv')
overnight = mir_annual(MIR_OVERNIGHT, 'ecb_mir_at_l21_overnight.csv')
notice = mir_annual(MIR_NOTICE, 'ecb_mir_at_l23_notice.csv')
assert agreed.index.min() == 2000 and agreed.index.max() == LAST_YEAR, 'agreed-maturity series must cover 2000-LAST_YEAR'
assert list(agreed.index) == list(range(2000, LAST_YEAR + 1)), 'gap in agreed-maturity deposit rate'
both = agreed.index.intersection(notice.index)
print(f'   agreed<=1y vs notice deposits on {len(both)} full years: '
      f'mean abs diff {(agreed.loc[both] - notice.loc[both]).abs().mean()*100:.2f}pp')
cash = pd.DataFrame({'retail_rate': agreed, 'overnight_rate': overnight, 'notice_rate': notice})
cash.index.name = 'year'
cash = cash.loc[2000:LAST_YEAR].reset_index()
cash.to_csv(os.path.join(DATA, 'at_cash_rate.csv'), index=False, float_format='%.6f')
print(f'   -> data/at_cash_rate.csv  {cash.year.min()}-{cash.year.max()} ({len(cash)} years), '
      f'retail mean {cash.retail_rate.mean()*100:.2f}%')
print('\nDone.')
