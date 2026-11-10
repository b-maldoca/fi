"""Tests for the generated EUR (Austria) return module and its Swiss-data stand-ins."""

import os

import numpy as np
import pandas as pd
import pytest

from src import historic_returns as chf
from src import historic_returns_eur as eur

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_YEARS = len(eur.HISTORIC_YEARS)
PROXY_YEARS = eur.HISTORIC_YEARS <= eur.SWISS_PROXY_LAST_YEAR


def _load_monthly(name):
    df = pd.read_csv(os.path.join(DATA, name), header=None, names=["m", "y", "v"])
    return pd.Series(df.v.values, index=pd.PeriodIndex(pd.to_datetime(dict(year=df.y, month=df.m, day=1)), freq="M"))


def _dec(series, year):
    return series[pd.Period(f"{year}-12", "M")]


def test_same_calendar_and_shapes_as_chf_module():
    np.testing.assert_array_equal(eur.HISTORIC_YEARS, chf.HISTORIC_YEARS)
    for name in ("MONTHLY_US_EUR", "MONTHLY_NON_US_EUR", "MONTHLY_CASH_EUR", "MONTHLY_GOLD_EUR"):
        assert getattr(eur, name).shape == (N_YEARS * 12,)
    for name in ("HISTORIC_RETURNS_US_EUR", "HISTORIC_RETURNS_NON_US_EUR", "HISTORIC_RETURNS_CASH_EUR",
                 "HISTORIC_RETURNS_GOLD_EUR", "HISTORIC_AUSTRIAN_INFLATION", "HISTORIC_USD_EUR_FX"):
        assert getattr(eur, name).shape == (N_YEARS,)


def test_swiss_proxy_flags():
    assert eur.SWISS_PROXY_LAST_YEAR == 1957
    assert eur.CASH_SWISS_PROXY_LAST_YEAR == 1999
    np.testing.assert_array_equal(eur.HISTORIC_YEAR_IS_SWISS_PROXY, eur.HISTORIC_YEARS <= 1957)
    np.testing.assert_array_equal(eur.HISTORIC_YEAR_CASH_IS_SWISS_PROXY, eur.HISTORIC_YEARS <= 1999)
    assert int(eur.HISTORIC_YEAR_IS_SWISS_PROXY.sum()) == 36


def test_pre_1958_inflation_is_swiss_cpi():
    np.testing.assert_allclose(
        eur.HISTORIC_AUSTRIAN_INFLATION[PROXY_YEARS], chf.HISTORIC_SWISS_INFLATION[PROXY_YEARS], atol=2e-8
    )


def test_post_1957_inflation_is_austrian_cpi():
    at_cpi = _load_monthly("at_inflation.csv")
    expected = np.array([_dec(at_cpi, y) / _dec(at_cpi, y - 1) - 1.0 for y in eur.HISTORIC_YEARS[~PROXY_YEARS]])
    np.testing.assert_allclose(eur.HISTORIC_AUSTRIAN_INFLATION[~PROXY_YEARS], expected, atol=1e-8)
    # And it genuinely differs from Swiss inflation (e.g. 1970s), i.e. Austrian data is used.
    assert np.max(np.abs(eur.HISTORIC_AUSTRIAN_INFLATION[~PROXY_YEARS] - chf.HISTORIC_SWISS_INFLATION[~PROXY_YEARS])) > 0.01


def test_pre_1958_cash_is_swiss_savings_rate():
    months = np.repeat(PROXY_YEARS, 12)
    np.testing.assert_array_equal(eur.MONTHLY_CASH_EUR[months], chf.MONTHLY_CASH_CHF[months])


def test_1958_1999_cash_is_ppp_translated_swiss_rate():
    sel = (eur.HISTORIC_YEARS > 1957) & (eur.HISTORIC_YEARS <= 1999)
    expected = (
        (1.0 + chf.HISTORIC_RETURNS_CASH_CHF[sel])
        * (1.0 + eur.HISTORIC_AUSTRIAN_INFLATION[sel])
        / (1.0 + chf.HISTORIC_SWISS_INFLATION[sel])
        - 1.0
    )
    np.testing.assert_allclose(eur.HISTORIC_RETURNS_CASH_EUR[sel], expected, atol=1e-6)


def test_2000_plus_cash_is_ecb_mir_rate():
    cash = pd.read_csv(os.path.join(DATA, "at_cash_rate.csv")).set_index("year")
    sel = eur.HISTORIC_YEARS >= 2000
    np.testing.assert_allclose(
        eur.HISTORIC_RETURNS_CASH_EUR[sel], cash.loc[eur.HISTORIC_YEARS[sel], "retail_rate"].values, atol=1e-6
    )


def test_pre_1958_equity_real_returns_follow_relative_ppp():
    # EUR return deflated by (Swiss) inflation == USD return deflated by US inflation.
    us_cpi = _load_monthly("us_cpi.csv")
    years = eur.HISTORIC_YEARS[PROXY_YEARS]
    us_inf = np.array([_dec(us_cpi, y) / _dec(us_cpi, y - 1) - 1.0 for y in years])
    real_eur = (1.0 + eur.HISTORIC_RETURNS_US_EUR[PROXY_YEARS]) / (1.0 + eur.HISTORIC_AUSTRIAN_INFLATION[PROXY_YEARS])
    real_usd = (1.0 + eur.HISTORIC_RETURNS_US_USD[PROXY_YEARS]) / (1.0 + us_inf)
    np.testing.assert_allclose(real_eur, real_usd, rtol=1e-6)


def test_post_1957_fx_is_market_rate():
    fx = _load_monthly("usd_eur.csv")
    years = eur.HISTORIC_YEARS[~PROXY_YEARS]
    expected = np.array([_dec(fx, y) / _dec(fx, y - 1) - 1.0 for y in years])
    np.testing.assert_allclose(eur.HISTORIC_USD_EUR_FX[~PROXY_YEARS], expected, atol=1e-8)


def test_historic_real_eur_appreciation_is_within_plausible_range():
    assert 0.0 < eur.HISTORIC_REAL_EUR_APPRECIATION < 0.02


def test_return_generators_api():
    duration = 30
    matrix = eur.get_historic_return_matrix(duration, real_eur_appreciation=0.0)
    assert matrix.shape == (N_YEARS - duration + 1, duration * 12, eur.NUM_ASSETS)
    assert eur.get_historic_inflation_matrix(duration).shape == (N_YEARS - duration + 1, duration)
    returns, inflation = eur.generate_bootstrapped_data(50, duration, seed=3, real_eur_appreciation=0.0)
    assert returns.shape == (50, duration * 12, 5) and inflation.shape == (50, duration)
    assert np.isin(inflation, eur.HISTORIC_AUSTRIAN_INFLATION).all()


def test_ppp_slider_at_historic_value_reproduces_raw_series():
    raw = eur.get_historic_return_matrix(20)
    at_hist = eur.get_historic_return_matrix(20, real_eur_appreciation=eur.HISTORIC_REAL_EUR_APPRECIATION)
    np.testing.assert_array_equal(raw, at_hist)
    neutral = eur.get_historic_return_matrix(20, real_eur_appreciation=0.0)
    assert np.all(neutral[:, :, eur.ASSET_US_STOCKS] > raw[:, :, eur.ASSET_US_STOCKS])


@pytest.mark.parametrize("name", ["MONTHLY_US_EUR", "MONTHLY_NON_US_EUR", "MONTHLY_CASH_EUR", "MONTHLY_GOLD_EUR"])
def test_annual_series_compound_monthly(name):
    monthly = getattr(eur, name).reshape(N_YEARS, 12)
    annual = getattr(eur, name.replace("MONTHLY_", "HISTORIC_RETURNS_"))
    np.testing.assert_allclose(np.prod(1.0 + monthly, axis=1) - 1.0, annual, atol=1e-7)
