"""Austrian tax rules (2026): hand-computed statutory examples and golden tests against the BMF.

`test_pension_matches_bmf_2026` pins the pension taxation to the official BMF Brutto-Netto-Rechner
(https://onlinerechner.haude.at/BMF-Brutto-Netto-Rechner/, year 2026, Beitragsgruppe "Pensionist",
Wien, no children, queried 2026-11-10). The BMF has no online calculator for capital income, so
KESt and the Regelbesteuerungsoption are covered by the hand-computed examples below.
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


# BMF Brutto-Netto-Rechner 2026, pensioner: monthly gross -> (health insurance / yr, Lohnsteuer / yr,
# net pension / yr). Covers the Freigrenze for the 13th/14th payment (EUR 1,300 vs 1,350), the
# pensioner credit phase-out (1,500-3,000), every tariff bracket up to 55%, the health insurance
# Hoechstbeitragsgrundlage (8,000+) and the § 67 bands above EUR 83,333 (95,000).
BMF_PENSION_2026 = [
    (1_100, 924.00, 0.00, 14_476.00),
    (1_200, 1_008.00, 0.00, 15_792.00),
    (1_300, 1_092.00, 0.00, 17_108.00),
    (1_350, 1_134.00, 115.08, 17_650.92),
    (1_500, 1_260.00, 132.00, 19_608.00),
    (2_000, 1_680.00, 1_127.04, 25_192.96),
    (2_500, 2_100.00, 3_457.68, 29_442.32),
    (3_000, 2_520.00, 5_546.16, 33_933.84),
    (4_000, 3_360.00, 9_909.24, 42_730.76),
    (6_000, 5_040.00, 19_158.84, 59_801.16),
    (8_000, 5_821.20, 30_375.86, 75_802.94),
    (12_000, 5_821.20, 54_578.90, 107_599.90),
    (95_000, 5_821.20, 636_315.42, 687_863.38),
]


@pytest.mark.parametrize("monthly_gross, bmf_health, bmf_tax, bmf_net", BMF_PENSION_2026)
def test_pension_matches_bmf_2026(monthly_gross, bmf_health, bmf_tax, bmf_net):
    t = calculate_annual_taxes_at(14 * monthly_gross, 0.0, 0.0, 0.0)
    # The BMF rounds each monthly payment to cents; the model computes the year at once.
    assert t["pension_health_insurance"] == pytest.approx(bmf_health, abs=0.01)
    assert t["income_tax"] == pytest.approx(bmf_tax, abs=0.25)
    assert 14 * monthly_gross - t["total"] == pytest.approx(bmf_net, abs=0.25)


def test_special_payments_freigrenze_and_bands():
    # Jahressechstel (two monthly gross payments) <= EUR 2,615 -> no tax on the 13th/14th payment.
    assert special_payments_tax_at(2_444.0, 2_600.0) == 0.0
    assert special_payments_tax_at(2_538.0, 2_700.0) == pytest.approx(0.06 * (2_538.0 - 620.0))
    # Bands: 6% up to 25,000, 27% to 50,000, 35.75% to 83,333.33; the rest goes to the tariff.
    expected = 0.06 * 24_380 + 0.27 * 25_000 + 0.3575 * 33_333.33
    assert special_payments_tax_at(100_000.0) == pytest.approx(expected, abs=0.01)
