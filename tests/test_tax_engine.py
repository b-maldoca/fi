import numpy as np

from src.tax_engine import (
    calculate_income_tax,
    calculate_wealth_tax,
    calculate_capital_withdrawal_tax,
    calculate_ahv_non_worker
)

def test_income_tax():
    incomes = np.array([0, 20_000, 100_000])
    # For 0 income, tax should be 0
    tax = calculate_income_tax(incomes, cantonal_multiplier=1.0, municipal_multiplier=1.19)
    assert tax[0] == 0
    # For 20k, it should be small
    assert tax[1] > 0
    # For 100k, it should be significantly larger
    assert tax[2] > tax[1]
    
def test_wealth_tax():
    wealth = np.array([0, 80_000, 80_001, 1_000_000])
    tax = calculate_wealth_tax(wealth, cantonal_multiplier=1.0, municipal_multiplier=1.19)
    assert tax[0] == 0
    assert tax[1] == 0  # Exactly at 80k threshold (tax-free)
    assert tax[2] > 0   # 80_001 should trigger tax
    assert tax[3] > 0   # 1M should trigger tax

def test_ahv():
    wealth = np.array([100_000, 1_000_000, 10_000_000])
    ahv = calculate_ahv_non_worker(wealth)
    assert ahv[0] == 530  # Min contribution
    assert ahv[1] > 530   # Mid contribution
    assert ahv[2] == 26_500 # Max contribution cap

def test_ahv_precise():
    # Test specific values from the 2025 table
    wealths = np.array([349_999, 350_000, 400_000, 500_000, 1_750_000, 8_950_000, 9_000_000])
    ahv = calculate_ahv_non_worker(wealths)
    
    assert ahv[0] == 530.0   # Under 350k
    assert ahv[1] == 636.0   # Exactly 350k
    assert ahv[2] == 742.0   # 400k
    assert ahv[3] == 954.0   # 500k
    assert ahv[4] == 3604.0  # 1.75M
    assert ahv[5] == 26500.0 # 8.95M
    assert ahv[6] == 26500.0 # Above 8.95M (capped)

def test_capital_withdrawal_tax():
    amounts = np.array([0, 50_000, 200_000, 1_000_000])
    tax = calculate_capital_withdrawal_tax(amounts, cantonal_multiplier=1.0, municipal_multiplier=1.19)
    assert tax[0] == 0.0
    assert tax[1] > 0.0
    assert tax[2] > tax[1]
    assert tax[3] > tax[2]

def test_negative_tax_inputs():
    # Negative incomes/wealth must not produce negative taxes
    tax_inc = calculate_income_tax(np.array([-50_000, -100]))
    assert np.all(tax_inc == 0.0)
    tax_wealth = calculate_wealth_tax(np.array([-1_000_000, -50]))
    assert np.all(tax_wealth == 0.0)
    tax_cap = calculate_capital_withdrawal_tax(np.array([-200_000]))
    assert np.all(tax_cap == 0.0)

def test_ahv_scalar_vs_array():
    scalar_res = calculate_ahv_non_worker(500_000.0)
    assert isinstance(scalar_res, float)
    assert scalar_res == 954.0
    
    arr_res = calculate_ahv_non_worker(np.array([500_000.0]))
    assert isinstance(arr_res, np.ndarray)
    assert arr_res.shape == (1,)
    assert arr_res[0] == 954.0

def test_tax_functions_scalar_return():
    inc_tax = calculate_income_tax(80_000.0)
    assert isinstance(inc_tax, float)
    assert inc_tax > 0.0

    wealth_tax = calculate_wealth_tax(1_000_000.0)
    assert isinstance(wealth_tax, float)
    assert wealth_tax > 0.0

    cap_tax = calculate_capital_withdrawal_tax(200_000.0)
    assert isinstance(cap_tax, float)
    assert cap_tax > 0.0

def test_ahv_mixed_scalar_array():
    # Wealth is scalar, imputed pension is array
    res = calculate_ahv_non_worker(0.0, np.array([20_000.0, 50_000.0]))
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert res[0] < res[1]
    assert res[0] == 742.0
    assert res[1] == 2014.0 or res[1] > res[0]


def test_zurich_capital_withdrawal_minimum_simple_tax_of_2pct():
    # § 37 StG ZH: rate of a pension of C/20. For C = 100k that pension is 5k, which
    # falls in the 0% bracket, so the 2% minimum simple state tax binds.
    # Federal (Art. 38 DBG): 1/5 of the Art. 36 tariff on the full 100k.
    from src.tax_engine import _federal_income_tariff
    expected = _federal_income_tariff(100_000.0) / 5.0 + 0.02 * 100_000 * (0.95 + 1.19)
    assert np.isclose(calculate_capital_withdrawal_tax(100_000.0, 0.95, 1.19), expected)
    # ~4.8% all-in, matching the official ESTV calculator (CHF 4,817 for Zurich City 2026).
    assert 0.047 < expected / 100_000 < 0.049


def test_zurich_capital_withdrawal_rate_rule_above_minimum():
    # C = 1M -> notional pension 50k. Zurich 2026 base tariff on 50k:
    zh_tariff_50k = 5_000 * 0.02 + 4_800 * 0.03 + 8_000 * 0.04 + 9_700 * 0.05 + 11_200 * 0.06 + 4_300 * 0.07
    rate = zh_tariff_50k / 50_000  # 4.04% > 2% minimum
    simple_tax = rate * 1_000_000
    # Federal tariff at 1M exceeds the 11.5% ceiling, so it is 11.5% / 5.
    fed = 0.115 * 1_000_000 / 5.0
    expected = fed + simple_tax * 2.14
    assert np.isclose(calculate_capital_withdrawal_tax(1_000_000.0, 0.95, 1.19), expected)


def test_capital_withdrawal_tax_is_progressive_in_effective_rate():
    amounts = np.array([50_000.0, 200_000.0, 500_000.0, 1_000_000.0, 3_000_000.0])
    eff = calculate_capital_withdrawal_tax(amounts) / amounts
    assert np.all(np.diff(eff) >= 0)


def test_federal_income_tax_capped_at_11_5_pct():
    # With zero cantonal/municipal multipliers only the federal tax remains.
    assert np.isclose(calculate_income_tax(2_000_000.0, 0.0, 0.0), 0.115 * 2_000_000)
    # Below the crossover the ordinary marginal tariff applies (average < 11.5%).
    assert calculate_income_tax(300_000.0, 0.0, 0.0) < 0.115 * 300_000
    # The cap is continuous: no jump at the crossover.
    x = np.linspace(700_000, 900_000, 2001)
    tax = calculate_income_tax(x, 0.0, 0.0)
    assert np.all(np.diff(tax) >= 0)
    assert np.max(np.diff(tax)) <= 0.132 * (x[1] - x[0]) + 1e-9


def test_capital_withdrawal_tax_homogeneous_under_bracket_scaling(monkeypatch):
    # The engine indexes brackets via T_indexed(x, f) == f * T(x / f). This must stay
    # exact for the § 37 rate rule, the 2% minimum, and the federal 11.5% cap.
    import src.tax_engine as te
    f = 1.37
    amounts = np.array([30_000.0, 150_000.0, 700_000.0, 5_000_000.0])
    via_identity = f * calculate_capital_withdrawal_tax(amounts / f)
    monkeypatch.setattr(te, "FEDERAL_INCOME_BRACKETS", [(t * f, r) for t, r in te.FEDERAL_INCOME_BRACKETS])
    monkeypatch.setattr(te, "ZURICH_INCOME_BRACKETS", [(t * f, r) for t, r in te.ZURICH_INCOME_BRACKETS])
    with_scaled_brackets = te.calculate_capital_withdrawal_tax(amounts)
    assert np.allclose(via_identity, with_scaled_brackets)


def test_default_multipliers_are_zurich_city_2026():
    from src.tax_engine import ZURICH_CANTONAL_MULTIPLIER, ZURICH_CITY_MUNICIPAL_MULTIPLIER
    assert (ZURICH_CANTONAL_MULTIPLIER, ZURICH_CITY_MUNICIPAL_MULTIPLIER) == (0.95, 1.19)
    for fn, x in ((calculate_income_tax, 90_000.0), (calculate_wealth_tax, 2_000_000.0),
                  (calculate_capital_withdrawal_tax, 400_000.0)):
        assert fn(x) == fn(x, 0.95, 1.19)


# Reference values from the official ESTV tax calculator
# (swisstaxcalculator.estv.admin.ch, API_calculateSimpleTaxes /
# API_calculateManyCapitalTaxes), tax year 2026, Zurich City (BFS 261), single,
# no children, no church tax, queried 2026-09-24. Each tuple is
# (amount, federal, cantonal, municipal) in CHF. The calculator floors taxable
# amounts to CHF 100 (income) / CHF 1,000 (wealth) and rounds taxes to whole CHF;
# the model does not, hence the small absolute tolerance. The CHF 24 Zurich
# personal tax (Personalsteuer) is excluded because it is not modelled.
ESTV_2026_INCOME = [
    (30_000, 114, 783, 981),
    (60_000, 671, 2_597, 3_253),
    (100_000, 2_684, 5_862, 7_342),
    (150_000, 7_076, 10_569, 13_239),
    (250_000, 19_503, 21_518, 26_955),
    (500_000, 52_503, 52_235, 65_431),
    (1_000_000, 115_000, 113_985, 142_781),
]
ESTV_2026_WEALTH = [
    (500_000, 0, 284, 355),
    (1_000_000, 0, 889, 1_113),
    (2_000_000, 0, 2_613, 3_273),
    (3_000_000, 0, 4_826, 6_046),
    (4_000_000, 0, 7_532, 9_435),
]
ESTV_2026_CAPITAL = [
    (20_000, 7, 380, 476),
    (50_000, 80, 950, 1_190),
    (100_000, 537, 1_900, 2_380),
    (250_000, 3_901, 4_750, 5_950),
    (500_000, 10_501, 10_906, 13_661),
    (750_000, 17_101, 23_351, 29_250),
    (1_000_000, 23_000, 38_418, 48_124),
    (2_000_000, 46_000, 117_230, 146_846),
    (5_000_000, 115_000, 430_369, 539_094),
]


def test_income_tax_matches_estv_2026():
    import pytest
    from src.tax_engine import _federal_income_tariff
    for amount, fed, canton, city in ESTV_2026_INCOME:
        assert _federal_income_tariff(float(amount)) == pytest.approx(fed, abs=2)
        assert calculate_income_tax(float(amount)) == pytest.approx(fed + canton + city, abs=4)


def test_wealth_tax_matches_estv_2026():
    import pytest
    for amount, _, canton, city in ESTV_2026_WEALTH:
        assert calculate_wealth_tax(float(amount)) == pytest.approx(canton + city, abs=3)


def test_capital_withdrawal_tax_matches_estv_2026():
    import pytest
    for amount, fed, canton, city in ESTV_2026_CAPITAL:
        assert calculate_capital_withdrawal_tax(float(amount)) == pytest.approx(fed + canton + city, abs=3)


