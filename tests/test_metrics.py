import numpy as np

from src.metrics import (
    OUTCOME_DEPLETED,
    OUTCOME_FAILED,
    OUTCOME_NEVER_BREACHED,
    OUTCOME_RECOVERED,
    RICH_MULTIPLE,
    beginning_of_year_withdrawal_rate,
    classify_outcome,
    classify_path_outcomes,
    compute_success_mask,
    compute_survival_curves,
    cumulative_inflation,
    first_depletion_year,
    first_shortfall_year,
    format_depletion_label,
    format_shortfall_label,
    min_real_net_worth,
    min_real_spending_pct,
    real_final_net_worth,
    summarize_failure_timing,
    worst_path_order,
    years_in_shortfall,
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


def test_first_depletion_year_and_format_label():
    # 3 years, 4 runs:
    # Run 0: never depletes
    # Run 1: depletes in year 1 (index 0)
    # Run 2: depletes in year 3 (index 2)
    # Run 3: liquid depletes in year 2, total NW never depletes
    nw = np.array([
        [100.0, 0.0, 80.0, 50.0],
        [90.0, -10.0, 30.0, 40.0],
        [80.0, -20.0, -5.0, 30.0],
    ])
    liq = np.array([
        [100.0, 0.0, 80.0, 10.0],
        [90.0, -10.0, 30.0, -2.0],
        [80.0, -20.0, -5.0, 30.0],
    ])
    dep_yr = first_depletion_year(nw)
    liq_yr = first_depletion_year(liq)
    assert np.isnan(dep_yr[0]) and np.isnan(dep_yr[3])
    assert dep_yr[1] == 1.0 and dep_yr[2] == 3.0
    assert liq_yr[3] == 2.0

    assert format_depletion_label(0, dep_yr, liq_yr, 40) == "Survived"
    assert format_depletion_label(1, dep_yr, liq_yr, 40) == "Age 41 (Yr 1)"
    assert format_depletion_label(2, dep_yr, liq_yr, 40) == "Age 43 (Yr 3)"
    assert format_depletion_label(3, dep_yr, liq_yr, 40) == "Liq Age 42 (Yr 2)"


def test_summarize_failure_timing_and_survival_curves():
    import pytest

    # 3 years, 4 runs, start_age = 60, initial_net_worth = 100, target = 50% (50 CHF real)
    # Run 0: ends at 80 (Success)
    # Run 1: ends at 30, never <= 0 (Target Shortfall)
    # Run 2: depletes in year 1 (Age 61, pre-65), ends at -20 (Depleted)
    # Run 3: depletes in year 3 (Age 63), liquid depletes in year 2 (Age 62)
    history = {
        "net_worth": np.array([
            [90.0, 70.0, 0.0, 60.0],
            [85.0, 45.0, -10.0, 20.0],
            [80.0, 30.0, -20.0, -5.0],
        ]),
        "liquid_assets": np.array([
            [90.0, 70.0, 0.0, 20.0],
            [85.0, 45.0, -10.0, 0.0],
            [80.0, 30.0, -20.0, -5.0],
        ]),
        "initial_net_worth": 100.0,
    }
    inf = np.zeros((4, 3))

    stats = summarize_failure_timing(history, inf, success_pct=50.0, start_age=60)
    assert stats["depletion_rate_pct"] == pytest.approx(50.0)
    assert stats["shortfall_rate_pct"] == pytest.approx(25.0)
    assert stats["pre_65_depletion_rate_pct"] == pytest.approx(50.0)
    assert stats["earliest_depletion_age"] == pytest.approx(61.0)
    assert stats["earliest_depletion_year"] == 1
    assert stats["median_depletion_age"] == pytest.approx(62.0)
    assert stats["median_depletion_year"] == pytest.approx(2.0)

    curves = compute_survival_curves(history, inf, success_pct=50.0)
    # Year 1: runs 0, 1, 3 solvent (75%); Year 2: 0, 1, 3 NW-solvent (75%) but only 0, 1 liquid-solvent (50%); Year 3: 0, 1 solvent (50%)
    assert np.allclose(curves["solvent_pct"], [75.0, 75.0, 50.0])
    assert np.allclose(curves["liquid_solvent_pct"], [75.0, 50.0, 50.0])
    # Above 50 CHF target: Year 1: runs 0, 1, 3 (75%); Year 2: only run 0 (25%); Year 3: only run 0 (25%)
    assert np.allclose(curves["above_target_pct"], [75.0, 25.0, 25.0])


def test_min_real_spending_pct():
    history = {
        "expenses_paid": np.array([
            [100.0, 80.0],
            [110.0, 66.0],
        ]),
    }
    # 10% inflation in year 2 -> cumulative price levels [1.0, 1.1]
    inf = np.array([[0.0, 0.10], [0.0, 0.10]])
    # Run 0 real expenses: [100, 100] -> min 100% of 100 base
    # Run 1 real expenses: [80, 60] -> min 60% of 100 base
    assert np.allclose(min_real_spending_pct(history, inf, 100.0), [100.0, 60.0])
    assert np.allclose(min_real_spending_pct(history, inf, 0.0), [100.0, 100.0])


def test_first_shortfall_year_and_format_label():
    # 3 years, 3 runs. NW_0 = 100, success_pct = 50%
    # No inflation -> target is 50.0 every year.
    # Run 0: stays above 50 (80, 70, 60) -> Never
    # Run 1: breaches target in year 2 (60, 45, 30) -> Year 2 (Age 42)
    # Run 2: breaches target in year 1 (40, 30, 20) -> Year 1 (Age 41)
    nw = np.array([
        [80.0, 60.0, 40.0],
        [70.0, 45.0, 30.0],
        [60.0, 30.0, 20.0],
    ])
    inf = np.zeros((3, 3))
    sf_yr = first_shortfall_year(nw, 100.0, inf, 50.0)
    assert np.isnan(sf_yr[0])
    assert sf_yr[1] == 2.0
    assert sf_yr[2] == 1.0

    assert format_shortfall_label(0, sf_yr, 40) == "Never"
    assert format_shortfall_label(1, sf_yr, 40) == "Age 42 (Yr 2)"
    assert format_shortfall_label(2, sf_yr, 40) == "Age 41 (Yr 1)"


def _severity_history():
    # 4 years, 7 runs, NW_0 = 100, no inflation, 50% criterion -> target 50 every year.
    # Run 0: depletes in year 3                          -> Depleted (dep yr 3)
    # Run 1: depletes in year 2                          -> Depleted (dep yr 2)
    # Run 2: never depletes, ends at 40                  -> Failed (final 40)
    # Run 3: never depletes, ends at 20                  -> Failed (final 20)
    # Run 4: 1 year below target, ends at 90             -> Recovered (1 yr)
    # Run 5: 3 years below target, ends at 80            -> Recovered (3 yrs)
    # Run 6: always above target, ends at 60             -> Never breached
    nw = np.array([
        [60.0, 40.0, 45.0, 30.0, 45.0, 40.0, 70.0],
        [30.0, -5.0, 45.0, 30.0, 60.0, 30.0, 65.0],
        [-1.0, -9.0, 45.0, 25.0, 70.0, 40.0, 62.0],
        [-3.0, 5.0, 40.0, 20.0, 90.0, 80.0, 60.0],
    ])
    return {"net_worth": nw, "initial_net_worth": 100.0}, np.zeros((7, 4))


def test_years_in_shortfall_and_min_real_net_worth():
    history, inf = _severity_history()
    assert list(years_in_shortfall(history["net_worth"], 100.0, inf, 50.0)) == [3, 4, 4, 4, 1, 3, 0]
    assert np.allclose(min_real_net_worth(history["net_worth"], inf), [-3.0, -9.0, 40.0, 20.0, 45.0, 30.0, 60.0])
    # Real values are deflated by each run's own price level.
    inf2 = np.array([[0.0, 1.0]])
    assert np.allclose(min_real_net_worth(np.array([[100.0], [150.0]]), inf2), [75.0])


def test_classify_path_outcomes_and_worst_path_order():
    history, inf = _severity_history()
    outcomes = classify_path_outcomes(history, inf, 50.0)
    assert list(outcomes) == [
        OUTCOME_DEPLETED, OUTCOME_DEPLETED, OUTCOME_FAILED, OUTCOME_FAILED,
        OUTCOME_RECOVERED, OUTCOME_RECOVERED, OUTCOME_NEVER_BREACHED,
    ]
    # Depleted by earliest depletion year, Failed by lowest final real NW,
    # Recovered by most years in shortfall, then Never breached.
    assert list(worst_path_order(history, inf, 50.0)) == [1, 0, 3, 2, 5, 4, 6]


def test_worst_path_order_respects_configurable_criterion():
    history, inf = _severity_history()
    # At a 0% criterion only depletion counts as a breach; everything else is "Never breached".
    outcomes = classify_path_outcomes(history, inf, 0.0)
    assert list(outcomes[2:]) == [OUTCOME_NEVER_BREACHED] * 5
    assert list(worst_path_order(history, inf, 0.0)) == [1, 0, 3, 2, 6, 5, 4]
