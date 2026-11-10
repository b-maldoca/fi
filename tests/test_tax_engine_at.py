"""Austrian tax rules (2026): hand-computed examples from the statutory formulas.

Golden tests against the official BMF calculators are still open (they need manual queries of the
BMF web calculators); every expected value below is derived by hand from the § 33 EStG tariff,
§ 27a EStG rates, the pensioner credit and the § 67 EStG special-payment rule.
"""

import numpy as np
import pytest

from src.tax_engine_at import (
    HEALTH_SELF_INSURANCE_MONTHLY,
    KEST_RATE,
    KEST_RATE_BANK_INTEREST,
    calculate_annual_taxes_at,
    calculate_income_tax_at,
    pensioner_credit_at,
    special_payments_tax_at,
)


@pytest.mark.parametrize(
    "income, expected",
    [
        (0.0, 0.0),
        (13_539.0, 0.0),
        (21_992.0, 1_690.60),  # 8,453 x 20%
        (30_000.0, 4_093.00),  # + 8,008 x 30%
        (36_458.0, 6_030.40),
        (70_365.0, 19_593.20),  # + 33,907 x 40%
        (104_859.0, 36_150.32),  # + 34,494 x 48%
        (1_000_000.0, 483_720.82),  # + 895,141 x 50%
        (1_100_000.0, 538_720.82),  # + 100,000 x 55%
    ],
)
def test_income_tariff_2026(income, expected):
    assert calculate_income_tax_at(income) == pytest.approx(expected, abs=0.01)


def test_income_tariff_vectorized_and_clamped():
    out = calculate_income_tax_at(np.array([-5.0, 30_000.0]))
    np.testing.assert_allclose(out, [0.0, 4_093.0])


@pytest.mark.parametrize(
    "pension, expected",
    [(0.0, 0.0), (10_000.0, 1_020.0), (21_614.0, 1_020.0), (26_554.0, 510.0), (31_494.0, 0.0), (40_000.0, 0.0)],
)
def test_pensioner_credit_phase_out(pension, expected):
    assert pensioner_credit_at(pension) == pytest.approx(expected)


def test_special_payments_tax():
    assert special_payments_tax_at(500.0) == 0.0
    assert special_payments_tax_at(2_820.0) == pytest.approx(0.06 * 2_200.0)


def test_rates_and_health_insurance_constants():
    assert (KEST_RATE, KEST_RATE_BANK_INTEREST, HEALTH_SELF_INSURANCE_MONTHLY) == (0.275, 0.25, 565.25)


def test_pension_only():
    # EUR 2,000 gross x 14 = 28,000. Regular taxable: 24,000 x 0.94 = 22,560; special: 3,760.
    t = calculate_annual_taxes_at(28_000.0, 0.0, 0.0, 0.0)
    credit = 1_020.0 * (31_494.0 - 22_560.0) / (31_494.0 - 21_614.0)
    tariff = 1_690.60 + 0.30 * (22_560.0 - 21_992.0)
    special = 0.06 * (3_760.0 - 620.0)
    assert t["pension_health_insurance"] == pytest.approx(1_680.0)
    assert t["income_tax"] == pytest.approx(tariff - credit + special)
    assert t["kest"] == 0.0
    assert t["total"] == pytest.approx(tariff - credit + special + 1_680.0)


def test_regelbesteuerung_wins_for_small_capital_income():
    # No pension; 10k dividends + 20k gains + 2k interest: flat 8,750 vs tariff(32,000) = 4,693.
    t = calculate_annual_taxes_at(0.0, 10_000.0, 20_000.0, 2_000.0)
    assert t["regelbesteuerung"] == 1.0
    assert t["kest"] == 0.0
    assert t["income_tax"] == pytest.approx(4_693.0)
    assert t["total"] == pytest.approx(4_693.0)


def test_flat_kest_wins_for_large_capital_income():
    t = calculate_annual_taxes_at(0.0, 0.0, 300_000.0, 0.0)
    assert t["regelbesteuerung"] == 0.0
    assert t["kest"] == pytest.approx(0.275 * 300_000.0)
    assert t["income_tax"] == 0.0


def test_loss_offset_within_year_excludes_bank_interest():
    # High pension -> progressive option never pays; losses wipe out dividends but not interest.
    t = calculate_annual_taxes_at(14 * 8_000.0, 5_000.0, -8_000.0, 1_000.0)
    assert t["regelbesteuerung"] == 0.0
    assert t["kest"] == pytest.approx(0.25 * 1_000.0)


def test_no_loss_carry_forward_or_negative_tax():
    t = calculate_annual_taxes_at(0.0, 0.0, -50_000.0, 0.0)
    assert t["total"] == 0.0 and t["kest"] == 0.0


def test_indexed_brackets_equal_explicitly_scaled_thresholds():
    # Bracket indexation evaluates f * T(x / f). For the pure tariff this must equal the tariff
    # with every threshold multiplied by f.
    import src.tax_engine_at as at

    f, income = 1.37, 55_000.0
    scaled_brackets = [(threshold * f, rate) for threshold, rate in at.AT_INCOME_BRACKETS]
    explicit = at._calculate_bracket_tax(income, scaled_brackets)
    assert f * calculate_income_tax_at(income / f) == pytest.approx(float(explicit))
