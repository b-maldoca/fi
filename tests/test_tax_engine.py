import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax_engine import (
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

if __name__ == "__main__":
    test_income_tax()
    test_wealth_tax()
    test_ahv()
    test_ahv_precise()
    print("All basic tax engine tests passed!")
