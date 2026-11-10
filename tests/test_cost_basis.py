import numpy as np
import pytest

from src.simulation_engine import (
    ASSET_CASH,
    NON_CASH_ASSETS,
    SimConfig,
    _buy,
    _rebalance_to,
    _sell,
    run_simulation,
)


def _config(**overrides) -> SimConfig:
    params = dict(
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=66,
        dividend_yield=0.0,
        initial_liquid_wealth=1_000_000.0,
        alloc_us_stocks=1.0,
        annual_base_expenses=50_000.0,
        rebalance_strategy="Never",
        enable_smart_selling=False,
    )
    params.update(overrides)
    return SimConfig(**params)


# --- helpers -----------------------------------------------------------------


def test_sell_uses_average_cost():
    assets = np.array([[200.0, 50.0, 10.0, 0.0, 0.0]])
    basis = np.array([[100.0, 50.0, 10.0, 0.0, 0.0]])
    realized = _sell(assets, basis, np.array([[50.0, 10.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(realized, [[25.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(assets, [[150.0, 40.0, 10.0, 0.0, 0.0]])
    np.testing.assert_allclose(basis, [[75.0, 40.0, 10.0, 0.0, 0.0]])
    # Basis / value ratio is unchanged by a sale.
    assert basis[0, 0] / assets[0, 0] == pytest.approx(0.5)


def test_sell_realizes_losses():
    assets = np.array([[80.0, 0.0, 0.0, 0.0, 0.0]])
    basis = np.array([[100.0, 0.0, 0.0, 0.0, 0.0]])
    realized = _sell(assets, basis, np.array([[40.0, 0.0, 0.0, 0.0, 0.0]]))
    assert realized[0, 0] == pytest.approx(-10.0)
    assert basis[0, 0] == pytest.approx(50.0)


def test_buy_adds_purchase_price_to_basis():
    assets = np.array([[100.0, 0.0, 0.0, 0.0, 0.0]])
    basis = np.array([[60.0, 0.0, 0.0, 0.0, 0.0]])
    _buy(assets, basis, np.array([[40.0, 10.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(assets, [[140.0, 10.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(basis, [[100.0, 10.0, 0.0, 0.0, 0.0]])


def test_rebalance_books_sells_and_buys_only_for_masked_runs():
    assets = np.array([[150.0, 0.0, 50.0, 0.0, 0.0], [150.0, 0.0, 50.0, 0.0, 0.0]])
    basis = np.array([[75.0, 0.0, 50.0, 0.0, 0.0], [75.0, 0.0, 50.0, 0.0, 0.0]])
    mask = np.array([True, False])
    target = np.array([[100.0, 0.0, 100.0, 0.0, 0.0]])

    realized = _rebalance_to(assets, basis, mask, target)

    np.testing.assert_allclose(realized, [[25.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 5])
    np.testing.assert_allclose(assets[0], target[0])
    np.testing.assert_allclose(basis[0], [50.0, 0.0, 100.0, 0.0, 0.0])
    # Unmasked run untouched.
    np.testing.assert_array_equal(assets[1], [150.0, 0.0, 50.0, 0.0, 0.0])
    np.testing.assert_array_equal(basis[1], [75.0, 0.0, 50.0, 0.0, 0.0])


def test_non_cash_assets_constant():
    assert ASSET_CASH not in NON_CASH_ASSETS and len(NON_CASH_ASSETS) == 4


# --- engine -----------------------------------------------------------------


def test_zero_returns_never_create_gains():
    cfg = _config(
        num_runs=2,
        duration_years=12,
        start_age=55,
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.2,
        alloc_chf_cash=0.2,
        alloc_gold=0.05,
        alloc_bitcoin=0.05,
        rebalance_strategy="Cash Tent",
        spending_strategy="Vanguard Dynamic",
        initial_pillar_2=300_000.0,
        initial_pillar_3a_accounts=[40_000.0] * 5,
        monthly_ahv_pension=2_000.0,
        enable_smart_selling=True,
    )
    history = run_simulation(cfg, np.zeros((2, 12 * 12, 5)))
    np.testing.assert_allclose(history["realized_gains"], 0.0, atol=1e-6)
    np.testing.assert_allclose(history["unrealized_gains"], 0.0, atol=1e-6)
    np.testing.assert_allclose(history["cost_basis_by_class"], history["liquid_assets_by_class"], atol=1e-6)


def test_day_0_basis_equals_market_value():
    history = run_simulation(_config(annual_base_expenses=0.0), np.zeros((1, 12, 5)))
    # No sale was needed except the taxes; with zero returns basis tracks value exactly.
    np.testing.assert_allclose(history["cost_basis_by_class"][0], history["liquid_assets_by_class"][0])


def test_year_end_sale_realizes_gain_at_average_cost():
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0  # US stocks double in month 0: basis / value = 0.5
    history = run_simulation(_config(), returns)

    sale = history["expenses_paid"][0, 0] + history["taxes_paid"][0, 0]
    assert history["realized_gains"][0, 0] == pytest.approx(0.5 * sale)
    assert history["cost_basis_by_class"][0, 0, 0] == pytest.approx(1_000_000.0 - 0.5 * sale)
    assert history["unrealized_gains"][0, 0] == pytest.approx(1_000_000.0 - 0.5 * sale)


def test_rebalancing_realizes_gains():
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0
    cfg = _config(alloc_us_stocks=0.5, alloc_chf_cash=0.5, rebalance_strategy="Monthly")
    history = run_simulation(cfg, returns)

    # Month 0: US 1.0M (basis 0.5M) vs cash 0.5M -> rebalance sells 0.25M of US, realizing 0.125M.
    # Year end: the outflow is sold 50/50 from US (basis ratio 0.5) and cash.
    sale = history["expenses_paid"][0, 0] + history["taxes_paid"][0, 0]
    assert history["realized_gains"][0, 0] == pytest.approx(125_000.0 + 0.25 * sale)


def test_never_rebalance_without_outflows_keeps_basis():
    returns = np.full((1, 24, 5), 0.01)
    cfg = _config(duration_years=2, annual_base_expenses=0.0, alloc_us_stocks=0.6, alloc_gold=0.4)
    history = run_simulation(cfg, returns)
    # Only the (small) taxes are sold; basis never exceeds value and gains are positive.
    assert np.all(history["realized_gains"] > 0.0)
    assert np.all(history["unrealized_gains"] > 0.0)
    basis = history["cost_basis_by_class"][-1, 0, NON_CASH_ASSETS]
    value = history["liquid_assets_by_class"][-1, 0, NON_CASH_ASSETS]
    assert np.all(basis <= value + 1e-9)


def test_dividends_raise_cost_basis():
    cfg = _config(annual_base_expenses=0.0, dividend_yield=0.03)
    history = run_simulation(cfg, np.zeros((1, 12, 5)))
    # With zero total return, the reinvested dividend is added to basis but not to value, so the
    # holding shows an unrealized loss of (dividend - basis share of the tax sale).
    dividends = history["income_dividends"][0, 0]
    assert dividends == pytest.approx(30_000.0)
    assert history["unrealized_gains"][0, 0] < 0.0
    assert history["realized_gains"][0, 0] < 0.0  # the tax sale is at a loss vs. the raised basis


def test_bankruptcy_zeroes_non_cash_basis():
    cfg = _config(duration_years=3, annual_base_expenses=900_000.0, alloc_us_stocks=0.5, alloc_gold=0.5)
    returns = np.full((1, 36, 5), 0.005)
    history = run_simulation(cfg, returns)
    assert history["liquid_assets"][-1, 0] < 0.0
    np.testing.assert_allclose(history["cost_basis_by_class"][-1, 0, NON_CASH_ASSETS], 0.0)
    assert history["unrealized_gains"][-1, 0] == pytest.approx(0.0)


def test_output_shapes():
    cfg = _config(num_runs=3, duration_years=4)
    history = run_simulation(cfg, np.zeros((3, 48, 5)))
    assert history["cost_basis_by_class"].shape == (4, 3, 5)
    assert history["realized_gains"].shape == (4, 3)
    assert history["unrealized_gains"].shape == (4, 3)
