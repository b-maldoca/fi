import numpy as np

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


def _calculate_bracket_tax(taxable_amount: np.ndarray, brackets: list) -> np.ndarray:
    """Vectorized calculation of progressive tax brackets."""
    taxable_amount = np.asarray(taxable_amount, dtype=float)
    tax = np.zeros_like(taxable_amount, dtype=float)
    
    for i in range(len(brackets)):
        threshold, rate = brackets[i]
        next_threshold = brackets[i+1][0] if i + 1 < len(brackets) else np.inf
        
        # Calculate how much of the taxable amount falls within this specific bracket
        amount_in_bracket = np.clip(taxable_amount - threshold, 0, next_threshold - threshold)
        tax += amount_in_bracket * rate
        
    return tax

def calculate_income_tax(taxable_income: np.ndarray, cantonal_multiplier: float = 1.0, municipal_multiplier: float = 1.19) -> np.ndarray:
    """
    Calculate combined Federal, Cantonal, and Municipal income tax.
    multiplier is typically sum of cantonal (e.g., 1.0) + municipal (e.g., 1.19 in Zurich city).
    Total multiplier = 2.19
    """
    total_multiplier = cantonal_multiplier + municipal_multiplier
    
    fed_tax = _calculate_bracket_tax(taxable_income, FEDERAL_INCOME_BRACKETS)
    cantonal_base_tax = _calculate_bracket_tax(taxable_income, ZURICH_INCOME_BRACKETS)
    
    total_cantonal_municipal_tax = cantonal_base_tax * total_multiplier
    
    return fed_tax + total_cantonal_municipal_tax

def calculate_wealth_tax(taxable_wealth: np.ndarray, cantonal_multiplier: float = 1.0, municipal_multiplier: float = 1.19) -> np.ndarray:
    """
    Calculate Cantonal and Municipal wealth tax. (Federal level does not have a wealth tax).
    """
    total_multiplier = cantonal_multiplier + municipal_multiplier
    
    cantonal_base_tax = _calculate_bracket_tax(taxable_wealth, ZURICH_WEALTH_BRACKETS)
    
    return cantonal_base_tax * total_multiplier

def calculate_capital_withdrawal_tax(amount: np.ndarray, cantonal_multiplier: float = 1.0, municipal_multiplier: float = 1.19) -> np.ndarray:
    """
    Calculate the separate tax on lump-sum withdrawals from Pillar 2 and Pillar 3a.
    Federal: Roughly 1/5 of the standard tariff.
    Zurich: Roughly 1/10 of the standard tariff on the base rate.
    """
    # Federal capital withdrawal tax approximation (1/5 of regular)
    fed_tax = _calculate_bracket_tax(amount, FEDERAL_INCOME_BRACKETS) / 5.0
    
    # Zurich capital withdrawal tax approximation (1/10 of regular base)
    cantonal_base_tax = _calculate_bracket_tax(amount, ZURICH_INCOME_BRACKETS) / 10.0
    total_cantonal_tax = cantonal_base_tax * (cantonal_multiplier + municipal_multiplier)
    
    return fed_tax + total_cantonal_tax

def calculate_ahv_non_worker(wealth: np.ndarray, imputed_pension_income: np.ndarray = 0) -> np.ndarray:
    """
    Calculate AHV mandatory contributions for non-working early retirees.
    Based on wealth and 20x imputed pension income (e.g., from an annuity).
    Min 530 CHF, Max 26,500 CHF (2025 values).
    """
    wealth = np.atleast_1d(wealth).astype(float)
    imputed_pension_income = np.atleast_1d(imputed_pension_income).astype(float)
    
    determining_wealth = wealth + (20 * imputed_pension_income)
    
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
    result = np.minimum(contribution, 26_500.0)
    
    if result.size == 1:
        return result[0]
    return result

