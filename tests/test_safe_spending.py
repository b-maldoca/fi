import numpy as np
import pytest

from src.metrics import effective_tax_rate, lifetime_real_totals
from src.safe_spending import (
    config_with_expenses,
    find_highest_safe_spending,
    initial_net_worth,
    success_rate_pct,
)
from src.simulation_engine import SimConfig, estimate_year_0_taxes


def _config(**overrides) -> SimConfig:
    params = dict(
        num_runs=40,
        duration_years=30,
        inflation_mean=0.02,
        inflation_std=0.01,
        start_age=50,
        dividend_yield=0.02,
        initial_liquid_wealth=2_000_000.0,
        alloc_us_stocks=0.6,
        alloc_non_us_stocks=0.2,
        alloc_chf_cash=0.2,
        annual_base_expenses=80_000.0,
        monthly_ahv_pension=2_000.0,
        rebalance_strategy="Yearly",
    )
    params.update(overrides)
    return SimConfig(**params)


def _paths(cfg: SimConfig, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.004, 0.035, size=(cfg.num_runs, cfg.duration_years * 12, 5))
    returns[:, :, 2] = 0.001  # cash
    inflation = rng.normal(0.02, 0.01, size=(cfg.num_runs, cfg.duration_years))
    return returns, inflation


def test_config_with_expenses_scales_the_vanguard_rate():
    cfg = _config(spending_strategy="Vanguard Dynamic", vanguard_target_rate=0.04)
    scaled = config_with_expenses(cfg, 120_000.0)
    assert scaled.annual_base_expenses == 120_000.0
    assert scaled.vanguard_target_rate == pytest.approx(0.06)
    zero_base = _config(spending_strategy="Vanguard Dynamic", annual_base_expenses=0.0)
    assert config_with_expenses(zero_base, 50_000.0).vanguard_target_rate == pytest.approx(50_000.0 / 2_000_000.0)
    # Other strategies keep the rate untouched.
    assert config_with_expenses(_config(), 120_000.0).vanguard_target_rate == _config().vanguard_target_rate


@pytest.mark.parametrize("strategy", ["Static", "Dynamic (Floor & Ceiling)", "Vanguard Dynamic"])
def test_highest_safe_spending_meets_target_and_next_step_does_not(strategy):
    extra = {"dynamic_expense_floor_pct": 0.85, "dynamic_expense_ceiling_pct": 1.1} if strategy.startswith("Dynamic") else {}
    cfg = _config(spending_strategy=strategy, vanguard_target_rate=80_000.0 / 2_000_000.0, **extra)
    returns, inflation = _paths(cfg)
    result = find_highest_safe_spending(cfg, returns, inflation, success_pct=50.0, target_success_pct=90.0)

    assert result.reachable and not result.capped
    assert result.expenses > 0.0 and result.expenses % 1_000.0 == 0.0
    assert result.success_rate_pct >= 90.0
    assert result.success_rate_pct == success_rate_pct(config_with_expenses(cfg, result.expenses), returns, inflation, 50.0)
    assert success_rate_pct(config_with_expenses(cfg, result.expenses + 1_000.0), returns, inflation, 50.0) < 90.0
    assert result.year_0_taxes == pytest.approx(estimate_year_0_taxes(config_with_expenses(cfg, result.expenses)))
    assert result.evaluations < 20


def test_highest_safe_spending_falls_with_a_stricter_target():
    cfg = _config()
    returns, inflation = _paths(cfg)
    loose = find_highest_safe_spending(cfg, returns, inflation, success_pct=0.0, target_success_pct=50.0)
    strict = find_highest_safe_spending(cfg, returns, inflation, success_pct=0.0, target_success_pct=100.0)
    assert strict.expenses <= loose.expenses


def test_highest_safe_spending_unreachable_target():
    # Ending with 10x the inflation-adjusted start is out of reach even without spending.
    cfg = _config()
    returns, inflation = _paths(cfg)
    result = find_highest_safe_spending(cfg, returns, inflation, success_pct=1_000.0, target_success_pct=50.0)
    assert not result.reachable
    assert result.expenses == 0.0
    assert result.success_rate_pct < 50.0


def test_highest_safe_spending_validates_inputs():
    cfg = _config(num_runs=2, duration_years=2)
    returns, inflation = _paths(cfg)
    with pytest.raises(ValueError, match="target_success_pct"):
        find_highest_safe_spending(cfg, returns, inflation, 0.0, target_success_pct=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        find_highest_safe_spending(cfg, returns, inflation, 0.0, tolerance=0.0)


def test_initial_net_worth_includes_pensions():
    cfg = _config(initial_pillar_2=300_000.0, initial_pillar_3a_accounts=[50_000.0, 25_000.0])
    assert initial_net_worth(cfg) == 2_375_000.0


def test_effective_tax_rate_is_real_taxes_over_real_withdrawals():
    history = {
        "expenses_paid": np.array([[80.0, 0.0], [90.0, 0.0]]),
        "taxes_paid": np.array([[20.0, 0.0], [30.0, 0.0]]),
        "unfunded_deficit": np.zeros((2, 2)),
    }
    inflation = np.array([[0.0, 1.0], [0.0, 0.0]])  # (runs, years); run 0: price level 1 then 2
    rate = effective_tax_rate(history, inflation)
    # Real: taxes 20 + 15 = 35, withdrawals 100 + 60 = 160.
    assert rate[0] == pytest.approx(35.0 / 160.0)
    assert rate[1] == 0.0  # nothing withdrawn
    np.testing.assert_allclose(lifetime_real_totals(history["taxes_paid"], inflation), [35.0, 0.0])
