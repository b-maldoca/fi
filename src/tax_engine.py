import numpy as np

# Canton Zurich cantonal multiplier (Staatssteuerfuss) for 2026 and the City of
# Zurich municipal multiplier (Gemeindesteuerfuss). These are political, not
# inflation-linked, parameters. They are the defaults for every tariff below and
# for `SimConfig`, so programmatic callers get realistic Zurich City taxes unless
# they deliberately override them.
ZURICH_CANTONAL_MULTIPLIER = 0.95
ZURICH_CITY_MUNICIPAL_MULTIPLIER = 1.19

# Federal Income Tax Brackets (approx 2024, Single)
# Format: (Threshold, Rate)
FEDERAL_INCOME_BRACKETS = [
    (0, 0.0000),
    (18_800, 0.0077),
    (33_300, 0.0088),
    (43_800, 0.0264),
    (58_400, 0.0297),
    (77_400, 0.0594),
    (84_000, 0.0660),
    (109_300, 0.0880),
    (147_000, 0.1100),
    (191_000, 0.1320),
]

# Art. 128 BV / Art. 36 DBG: the federal income tax never exceeds 11.5% of
# taxable income. Above the top threshold the tariff switches from marginal 13.2%
# to a flat 11.5% of the *whole* income, i.e. tax = min(tariff(x), 0.115 * x).
FEDERAL_MAX_AVERAGE_RATE = 0.115

# Art. 38 DBG: lump-sum pension withdrawals are taxed at 1/5 of the Art. 36 tariff.
FEDERAL_CAPITAL_WITHDRAWAL_FRACTION = 1.0 / 5.0

# § 37 StG ZH (since tax period 2022): the rate is the one that would apply if an
# annual pension of 1/20 of the lump sum were paid instead, and the simple state
# tax is at least 2% of the lump sum.
ZURICH_CAPITAL_WITHDRAWAL_RATE_DIVISOR = 20.0
ZURICH_CAPITAL_WITHDRAWAL_MIN_SIMPLE_RATE = 0.02

# Zurich Cantonal Income Tax Brackets (approx 2024, Single, BASE RATE)
# Note: Base rate must be multiplied by the combined multiplier (Steuerfuss).
ZURICH_INCOME_BRACKETS = [
    (0, 0.00),
    (7_300, 0.02),
    (12_500, 0.03),
    (18_000, 0.04),
    (25_000, 0.05),
    (34_000, 0.06),
    (45_000, 0.07),
    (59_000, 0.08),
    (78_000, 0.09),
    (104_000, 0.10),
    (139_000, 0.11),
    (186_000, 0.12),
    (250_000, 0.13),
]

# Zurich Cantonal Wealth Tax Brackets (2025, Single, BASE RATE)
ZURICH_WEALTH_BRACKETS = [
    (0, 0.0000),
    (80_000, 0.0005),
    (318_000, 0.0010),
    (717_000, 0.0015),
    (1_353_000, 0.0020),
    (2_309_000, 0.0025),
    (3_262_000, 0.0030),
]


def _is_scalar(x) -> bool:
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def _as_output(result: np.ndarray, is_scalar: bool):
    """Returns a Python float for scalar inputs and the array otherwise."""
    return float(result) if is_scalar else result


def _calculate_bracket_tax(taxable_amount: np.ndarray, brackets: list) -> np.ndarray:
    """Vectorized calculation of progressive tax brackets (negative inputs clamp to 0)."""
    arr = np.maximum(0.0, np.asarray(taxable_amount, dtype=float))
    tax = np.zeros_like(arr, dtype=float)

    for i, (threshold, rate) in enumerate(brackets):
        next_threshold = brackets[i + 1][0] if i + 1 < len(brackets) else np.inf
        # Portion of the taxable amount that falls within this specific bracket
        tax += np.clip(arr - threshold, 0, next_threshold - threshold) * rate

    return tax


def _federal_income_tariff(taxable_income: np.ndarray) -> np.ndarray:
    """Art. 36 DBG tariff for singles, including the 11.5% flat-rate ceiling."""
    arr = np.maximum(0.0, np.asarray(taxable_income, dtype=float))
    return np.minimum(_calculate_bracket_tax(arr, FEDERAL_INCOME_BRACKETS), FEDERAL_MAX_AVERAGE_RATE * arr)


def calculate_income_tax(
    taxable_income: np.ndarray,
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER,
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER,
) -> np.ndarray:
    """
    Calculate combined Federal, Cantonal, and Municipal income tax.

    The Zurich base tariff is scaled by the sum of the cantonal and municipal
    multipliers (Steuerfuss). The defaults are the Canton Zurich 2026 cantonal
    multiplier of 0.95 plus the City of Zurich municipal multiplier of 1.19,
    i.e. a total multiplier of 2.14.
    """
    fed_tax = _federal_income_tariff(taxable_income)
    cantonal_base_tax = _calculate_bracket_tax(taxable_income, ZURICH_INCOME_BRACKETS)
    result = fed_tax + cantonal_base_tax * (cantonal_multiplier + municipal_multiplier)
    return _as_output(result, _is_scalar(taxable_income))


def calculate_wealth_tax(
    taxable_wealth: np.ndarray,
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER,
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER,
) -> np.ndarray:
    """
    Calculate Cantonal and Municipal wealth tax. (Federal level does not have a wealth tax).
    """
    cantonal_base_tax = _calculate_bracket_tax(taxable_wealth, ZURICH_WEALTH_BRACKETS)
    result = cantonal_base_tax * (cantonal_multiplier + municipal_multiplier)
    return _as_output(result, _is_scalar(taxable_wealth))


def calculate_capital_withdrawal_tax(
    amount: np.ndarray,
    cantonal_multiplier: float = ZURICH_CANTONAL_MULTIPLIER,
    municipal_multiplier: float = ZURICH_CITY_MUNICIPAL_MULTIPLIER,
) -> np.ndarray:
    """
    Calculate the separate tax on lump-sum withdrawals from Pillar 2 and Pillar 3a.

    Federal (Art. 38 DBG): 1/5 of the ordinary Art. 36 tariff applied to the
    whole amount (so the effective federal rate is capped at 11.5% / 5 = 2.3%).

    Zurich (§ 37 StG ZH, since 2022): the simple state tax is levied at the rate
    that the ordinary income tariff would apply to an annual pension of 1/20 of
    the lump sum, applied to the whole lump sum, and is at least 2% of it:

        simple_tax = max(0.02 * C, 20 * T_ZH(C / 20))

    It is then multiplied by the cantonal + municipal Steuerfuss. Both parts are
    positively homogeneous of degree 1 in (amount, bracket edges), so the
    bracket-indexation identity used by the simulation engine still holds.
    """
    arr = np.maximum(0.0, np.asarray(amount, dtype=float))
    fed_tax = _federal_income_tariff(arr) * FEDERAL_CAPITAL_WITHDRAWAL_FRACTION

    divisor = ZURICH_CAPITAL_WITHDRAWAL_RATE_DIVISOR
    rate_based_simple_tax = divisor * _calculate_bracket_tax(arr / divisor, ZURICH_INCOME_BRACKETS)
    simple_tax = np.maximum(ZURICH_CAPITAL_WITHDRAWAL_MIN_SIMPLE_RATE * arr, rate_based_simple_tax)

    result = fed_tax + simple_tax * (cantonal_multiplier + municipal_multiplier)
    return _as_output(result, _is_scalar(amount))


def calculate_ahv_non_worker(wealth: np.ndarray, imputed_pension_income: np.ndarray = 0) -> np.ndarray:
    """
    Calculate AHV mandatory contributions for non-working early retirees.
    Based on wealth and 20x imputed pension income (e.g., from an annuity).
    Min 530 CHF, Max 26,500 CHF (2025 values).
    """
    is_scalar = _is_scalar(wealth) and _is_scalar(imputed_pension_income)

    wealth_arr = np.maximum(0.0, np.asarray(wealth, dtype=float))
    imputed_arr = np.maximum(0.0, np.asarray(imputed_pension_income, dtype=float))

    determining_wealth = wealth_arr + (20 * imputed_arr)

    # 106 CHF per 50k step above 300k (up to 1.75M)
    steps_mid = np.maximum(0.0, (determining_wealth - 300_000) // 50_000)
    contrib_mid = 530.0 + steps_mid * 106.0

    # 159 CHF per 50k step above 1.75M
    steps_high = np.maximum(0.0, (determining_wealth - 1_750_000) // 50_000)
    contrib_high = 3604.0 + steps_high * 159.0

    # Combine based on thresholds
    contribution = np.where(determining_wealth < 350_000, 530.0,
                            np.where(determining_wealth <= 1_750_000, contrib_mid, contrib_high))

    # Cap at max limit
    return _as_output(np.minimum(contribution, 26_500.0), is_scalar)
