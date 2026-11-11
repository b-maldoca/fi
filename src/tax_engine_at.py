"""Austrian income tax, capital income tax (KESt), pension taxation and health self-insurance.

Tax year 2026, single, no children, no church contribution. Sources (verified 2026-11-10):
  * § 33 Abs. 1 EStG tariff 2026 (brackets 1-5 indexed by 2/3 of the 2.6% July 2024 - June 2025
    inflation under § 33a EStG; the EUR 1M bracket and its temporary 55% rate are not indexed).
  * § 27a Abs. 1 EStG special rates: 27.5% on dividends, fund distributions / deemed distributions,
    realized gains on securities and crypto assets; 25% on bank-deposit interest.
  * § 27a Abs. 5 EStG Regelbesteuerungsoption: on application, ALL special-rate capital income is
    taxed at the progressive tariff instead (all-or-nothing).
  * § 27 Abs. 8 EStG: realized losses offset gains, dividends and deemed distributions of the same
    calendar year (not bank interest); no loss carry-forward.
  * Pensions: 6% health insurance contribution on the gross pension (since mid-2025), capped at the
    2026 Hoechstbeitragsgrundlage of EUR 6,930 per payment; 13th/14th payment (sonstige Bezuege,
    § 67 Abs. 1 EStG, within the Jahressechstel, which two monthly payments never exceed): tax-free if
    the Jahressechstel is at most EUR 2,615 (Freigrenze 2026), otherwise 0% on the first EUR 620,
    6% on the next EUR 24,380, 27% on the next EUR 25,000, 35.75% on the next EUR 33,333.33 and the
    tariff above; Pensionistenabsetzbetrag (§ 33 Abs. 6 EStG) EUR 1,020, phasing out linearly between
    EUR 21,614 and EUR 31,494 of taxable regular pension income.
  * § 16 ASVG Selbstversicherung in der Krankenversicherung: EUR 565.25 per month (2026, OeGK).

Verified against the official BMF Brutto-Netto-Rechner (pensioner, 2026) for monthly gross pensions
from EUR 1,100 to EUR 95,000: health contribution, income tax and net pension match to within the
calculator's monthly cent rounding (tests/test_tax_engine_at.py::test_pension_matches_bmf_2026).

Deliberate simplifications (each small, and all keep the tax positively homogeneous of degree 1 in
the money amounts, which `jurisdiction.indexed_tax`-style bracket indexation relies on): the
SV-Rueckerstattung (negative tax) is not modelled; every euro amount (incl. the EUR 1M bracket edge,
the § 67 bands and the Hoechstbeitragsgrundlage, which by law follow different or no indexation) is
indexed with the same bracket factor.

All functions are vectorized and follow the `tax_engine` contract: scalar in -> float out, array in
-> array out, negative amounts clamp to zero.
"""

import numpy as np

from .tax_engine import _as_output, _calculate_bracket_tax, _is_scalar

# § 33 Abs. 1 EStG, tax year 2026. Format: (threshold, marginal rate), as in tax_engine.
AT_INCOME_BRACKETS = [
    (0, 0.00),
    (13_539, 0.20),
    (21_992, 0.30),
    (36_458, 0.40),
    (70_365, 0.48),
    (104_859, 0.50),
    (1_000_000, 0.55),
]

# § 27a Abs. 1 EStG special tax rates (Kapitalertragsteuer, KESt).
KEST_RATE = 0.275
KEST_RATE_BANK_INTEREST = 0.25

# Pensions.
PENSION_PAYMENTS_PER_YEAR = 14          # 12 regular + 13th/14th special payment
PENSION_HEALTH_INSURANCE_RATE = 0.06    # Krankenversicherungsbeitrag on the gross pension
PENSION_HEALTH_INSURANCE_MAX_BASE = 6_930.0  # Hoechstbeitragsgrundlage 2026 per payment (14x)
# § 67 Abs. 1 EStG bands for sonstige Bezuege within the Jahressechstel. Format as tax brackets;
# amounts above SPECIAL_PAYMENT_TARIFF_FROM are taxed at the regular tariff.
SPECIAL_PAYMENT_BRACKETS = [
    (0, 0.00),
    (620, 0.06),
    (25_000, 0.27),
    (50_000, 0.3575),
]
SPECIAL_PAYMENT_TARIFF_FROM = 83_333.33
SPECIAL_PAYMENT_ALLOWANCE = 620.0       # tax-free first band (kept for reference)
SPECIAL_PAYMENT_RATE = 0.06             # rate of the second band
SPECIAL_PAYMENT_FREIGRENZE = 2_615.0    # 2026: no tax on sonstige Bezuege if the Jahressechstel <= this
PENSIONER_CREDIT = 1_020.0              # Pensionistenabsetzbetrag 2026
PENSIONER_CREDIT_FULL_UP_TO = 21_614.0
PENSIONER_CREDIT_ZERO_FROM = 31_494.0

# § 16 ASVG voluntary health self-insurance (always charged before the pension in the simulator,
# PRD §8.1). The contribution is a fixed share (~8.17%) of the monthly Hoechstbeitragsgrundlage
# (2024: 495.58, 2025: 526.79, 2026: 565.25), which moves with the wage-based Aufwertungszahl
# (§ 108a ASVG), not with CPI.
HEALTH_SELF_INSURANCE_MONTHLY = 565.25
# Default real (above-CPI) growth of that contribution: the Hoechstbeitragsgrundlage rose 3.3%/yr
# from 2010 (EUR 4,110) to 2026 (EUR 6,930) while Austrian CPI rose 2.6%/yr over 2008-2024 (the
# Aufwertungszahl lags two years), i.e. ~0.7%/yr real wage growth.
HEALTH_SELF_INSURANCE_REAL_GROWTH = 0.007


def calculate_income_tax_at(taxable_income):
    """§ 33 Abs. 1 EStG progressive tariff (before tax credits)."""
    result = _calculate_bracket_tax(taxable_income, AT_INCOME_BRACKETS)
    return _as_output(result, _is_scalar(taxable_income))


def pensioner_credit_at(taxable_regular_pension):
    """Pensionistenabsetzbetrag: EUR 1,020 up to EUR 21,614, linearly reduced to 0 at EUR 31,494.

    Zero when there is no pension income.
    """
    x = np.maximum(0.0, np.asarray(taxable_regular_pension, dtype=float))
    phase = np.clip(
        (PENSIONER_CREDIT_ZERO_FROM - x) / (PENSIONER_CREDIT_ZERO_FROM - PENSIONER_CREDIT_FULL_UP_TO), 0.0, 1.0
    )
    result = np.where(x > 0.0, PENSIONER_CREDIT * phase, 0.0)
    return _as_output(result, _is_scalar(taxable_regular_pension))


def special_payments_tax_at(taxable_special_payments, jahressechstel=None):
    """§ 67 Abs. 1 EStG tax on the 13th/14th payment (after health insurance), up to EUR 83,333.33.

    0% on the first EUR 620, 6% on the next 24,380, 27% on the next 25,000, 35.75% on the next
    33,333.33; amounts above are taxed at the regular tariff (see `special_payments_overflow_at`).
    No tax at all when `jahressechstel` (one sixth of the year's regular gross pay, i.e. two monthly
    payments) is at most the EUR 2,615 Freigrenze; `None` skips that check.
    """
    scalar = _is_scalar(taxable_special_payments) and (jahressechstel is None or _is_scalar(jahressechstel))
    x = np.maximum(0.0, np.asarray(taxable_special_payments, dtype=float))
    banded = _calculate_bracket_tax(np.minimum(x, SPECIAL_PAYMENT_TARIFF_FROM), SPECIAL_PAYMENT_BRACKETS)
    if jahressechstel is not None:
        banded = np.where(np.asarray(jahressechstel, dtype=float) <= SPECIAL_PAYMENT_FREIGRENZE, 0.0, banded)
    return _as_output(banded, scalar)


def special_payments_overflow_at(taxable_special_payments):
    """Part of the 13th/14th payment above EUR 83,333.33, which is taxed at the regular tariff."""
    x = np.maximum(0.0, np.asarray(taxable_special_payments, dtype=float))
    result = np.maximum(0.0, x - SPECIAL_PAYMENT_TARIFF_FROM)
    return _as_output(result, _is_scalar(taxable_special_payments))


def calculate_annual_taxes_at(
    annual_gross_pension,
    dividends,
    realized_gains,
    interest,
):
    """Annual Austrian taxes on a retiree's state pension and capital income (2026 base year).

    Args:
        annual_gross_pension: gross state pension of the year, all 14 payments.
        dividends: dividends and (deemed) fund distributions, taxed at 27.5%.
        realized_gains: net realized gains of the year (negative = net loss), 27.5%.
        interest: bank-deposit interest, 25%; not available for loss offset.

    Returns a dict of arrays (or floats for scalar input):
        pension_health_insurance: 6% health insurance contribution on the gross pension (capped).
        income_tax: progressive tax on the regular pension (after the pensioner credit) plus the
            § 67 tax on the 13th/14th payment, and -- if the Regelbesteuerungsoption is cheaper --
            the progressive tax on all capital income.
        kest: special-rate tax on capital income (0 when the Regelbesteuerungsoption applies).
        regelbesteuerung: 1.0 where the progressive option was chosen, else 0.0.
        total: income_tax + kest + pension_health_insurance.

    The Regelbesteuerungsoption (§ 27a Abs. 5 EStG) is evaluated every year and the cheaper of
    (a) progressive tax on the pension + KESt and (b) progressive tax on pension + all capital
    income is applied. Losses offset dividends and gains of the same year only (§ 27 Abs. 8 EStG).

    Every component is positively homogeneous of degree 1 in the money inputs, so bracket
    indexation can be applied as f * calculate_annual_taxes_at(x / f).
    """
    scalar = all(_is_scalar(a) for a in (annual_gross_pension, dividends, realized_gains, interest))
    gross, divs, gains, interest_arr = np.broadcast_arrays(
        np.asarray(annual_gross_pension, dtype=float),
        np.asarray(dividends, dtype=float),
        np.asarray(realized_gains, dtype=float),
        np.asarray(interest, dtype=float),
    )
    gross = np.maximum(0.0, gross)
    divs = np.maximum(0.0, divs)
    interest_arr = np.maximum(0.0, interest_arr)

    # Pension: 12 regular payments at the tariff, 2 special payments under § 67 Abs. 1; the 6%
    # health insurance contribution (capped at the Hoechstbeitragsgrundlage per payment) is deducted
    # before tax on both.
    pension_health = PENSION_HEALTH_INSURANCE_RATE * np.minimum(
        gross, PENSION_PAYMENTS_PER_YEAR * PENSION_HEALTH_INSURANCE_MAX_BASE
    )
    net_of_health = gross - pension_health
    taxable_regular = (12.0 / PENSION_PAYMENTS_PER_YEAR) * net_of_health
    taxable_special = (2.0 / PENSION_PAYMENTS_PER_YEAR) * net_of_health
    jahressechstel = (2.0 / PENSION_PAYMENTS_PER_YEAR) * gross
    special_tax = special_payments_tax_at(taxable_special, jahressechstel)
    # Special payments above EUR 83,333.33 join the regular income at the tariff (not the credit base).
    tariff_base = taxable_regular + special_payments_overflow_at(taxable_special)
    credit = pensioner_credit_at(taxable_regular)

    # Capital income pool at 27.5% after the in-year loss offset (unused losses expire).
    pool = np.maximum(0.0, divs + gains)
    kest_flat = KEST_RATE * pool + KEST_RATE_BANK_INTEREST * interest_arr

    tax_flat = np.maximum(0.0, _calculate_bracket_tax(tariff_base, AT_INCOME_BRACKETS) - credit)
    tax_regel = np.maximum(
        0.0, _calculate_bracket_tax(tariff_base + pool + interest_arr, AT_INCOME_BRACKETS) - credit
    )
    use_regel = tax_regel < tax_flat + kest_flat

    income_tax = np.where(use_regel, tax_regel, tax_flat) + special_tax
    kest = np.where(use_regel, 0.0, kest_flat)
    out = {
        "pension_health_insurance": pension_health,
        "income_tax": income_tax,
        "kest": kest,
        "regelbesteuerung": use_regel.astype(float),
        "total": income_tax + kest + pension_health,
    }
    return {k: _as_output(v, scalar) for k, v in out.items()}
