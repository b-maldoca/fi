import numpy as np

from src.metrics import (
    RICH_MULTIPLE,
    beginning_of_year_withdrawal_rate,
    classify_outcome,
    compute_success_mask,
    cumulative_inflation,
    real_final_net_worth,
)


def test_cumulative_inflation_clamps_extreme_deflation_like_engine():
    inf = np.array([[0.10, -2.0, 0.0]])
    # -200% is clamped to a 0.01 step, matching run_simulation's np.maximum(0.01, 1 + inf)
    assert np.allclose(cumulative_inflation(inf), [[1.10, 0.011, 0.011]])


def test_success_mask_uses_each_runs_own_price_level():
    # Two runs end with the same nominal wealth, but run 1 had 2x cumulative inflation,
    # so it needs twice the nominal balance to meet the same real target.
    final = np.array([150.0, 150.0])
    inf = np.array([[0.0, 0.0], [1.0, 0.0]])  # price levels 1.0 and 2.0
    assert compute_success_mask(final, 100.0, inf, 100.0).tolist() == [True, False]


def test_success_mask_zero_pct_means_survival():
    final = np.array([0.0, 1.0, -5.0])
    inf = np.zeros((3, 4))
    assert compute_success_mask(final, 1_000.0, inf, 0.0).tolist() == [False, True, False]


def test_real_final_net_worth_deflates_per_run():
    final = np.array([200.0, 300.0])
    inf = np.array([[1.0], [0.5]])
    assert np.allclose(real_final_net_worth(final, inf), [100.0, 200.0])


def test_classify_outcome_uses_median_of_real_values():
    initial = 100.0
    final = np.array([100.0, 400.0, 1_000.0])
    # Run 1 had 4x inflation, so its real value is 100. Real values: [100, 100, 1000].
    inf = np.array([[0.0], [3.0], [0.0]])
    # The old logic compared the nominal median (400) with 3x the *median* price level
    # (1.0) and reported RICH, although the median run is only 1x real.
    assert classify_outcome(final, initial, inf) == "DEAD"
    assert classify_outcome(final * 10, initial, inf) == "RICH"
    assert classify_outcome(np.array([-1.0, 0.0, 5.0]), initial, np.zeros((3, 1))) == "BROKE"
    # Boundary: exactly RICH_MULTIPLE x counts as RICH
    assert classify_outcome(np.full(3, RICH_MULTIPLE * initial), initial, np.zeros((3, 1))) == "RICH"


def test_beginning_of_year_withdrawal_rate():
    history = {
        'net_worth': np.array([[900.0, -50.0]]),
        'expenses_paid': np.array([[80.0, 40.0]]),
        'taxes_paid': np.array([[20.0, 10.0]]),
    }
    # Run 0: 100 / (900 + 100) = 10%. Run 1: start NW = 0 -> floored at 1 CHF.
    assert np.allclose(beginning_of_year_withdrawal_rate(history), [[10.0, 5_000.0]])
