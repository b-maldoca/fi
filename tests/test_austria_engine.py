"""Austria jurisdiction inside the simulation engine (KESt on realized gains, health insurance, pension)."""

import numpy as np
import pytest

from src.jurisdiction import COUNTRY_DEFAULTS, Austria, get_jurisdiction
from src.simulation_engine import SimConfig, estimate_year_0_taxes, run_simulation
from src.tax_engine_at import HEALTH_SELF_INSURANCE_MONTHLY, HEALTH_SELF_INSURANCE_REAL_GROWTH, calculate_annual_taxes_at

HEALTH = 12 * HEALTH_SELF_INSURANCE_MONTHLY
# Real wage growth of the self-insurance contribution from Year 1 on (Year 0 pays today's rate).
WAGE = 1.0 + HEALTH_SELF_INSURANCE_REAL_GROWTH


def _config(**overrides) -> SimConfig:
    params = dict(
        country="AT",
        num_runs=1,
        duration_years=1,
        inflation_mean=0.0,
        inflation_std=0.0,
        start_age=40,
        dividend_yield=0.0,
        initial_liquid_wealth=1_000_000.0,
        alloc_us_stocks=1.0,
        annual_base_expenses=40_000.0,
        rebalance_strategy="Never",
        enable_smart_selling=False,
    )
    params.update(overrides)
    return SimConfig(**params)


def _zero_inflation(cfg):
    return np.zeros((cfg.num_runs, cfg.duration_years))


def test_country_defaults_resolved_for_austria():
    cfg = _config()
    assert cfg.bracket_indexation == pytest.approx(2.0 / 3.0)
    assert cfg.cash_rate == COUNTRY_DEFAULTS["AT"].cash_rate == 0.02
    assert isinstance(get_jurisdiction("AT"), Austria)
    assert COUNTRY_DEFAULTS["AT"].pension_payments_per_year == 14
    # Explicit values still win.
    assert _config(bracket_indexation=1.0, cash_rate=0.0).bracket_indexation == 1.0


def test_swiss_pillars_rejected_in_austria():
    with pytest.raises(ValueError, match="Swiss-only"):
        _config(initial_pillar_2=100_000.0)
    with pytest.raises(ValueError, match="Swiss-only"):
        _config(initial_pillar_3a_accounts=[10_000.0])


def test_health_insurance_must_be_non_negative():
    with pytest.raises(ValueError, match="health_insurance_monthly"):
        _config(health_insurance_monthly=-1.0)


def test_zero_returns_only_health_self_insurance():
    cfg = _config(duration_years=3)
    history = run_simulation(cfg, np.zeros((1, 36, 5)), _zero_inflation(cfg))
    expected = HEALTH * WAGE ** np.arange(3)
    np.testing.assert_allclose(history["taxes_paid"][:, 0], expected)
    np.testing.assert_allclose(history["tax_components"]["health_insurance"][:, 0], expected)
    np.testing.assert_allclose(history["tax_components"]["kest"][:, 0], 0.0)
    np.testing.assert_allclose(history["realized_gains"], 0.0, atol=1e-9)
    assert history["initial_net_worth_after_tax"] == history["initial_net_worth"]


def test_outflow_sale_pays_tax_on_its_own_gains():
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0  # US doubles: every euro sold carries 50 cents of gain
    cfg = _config(annual_base_expenses=80_000.0)
    history = run_simulation(cfg, returns, _zero_inflation(cfg))

    sale = history["expenses_paid"][0, 0] + history["taxes_paid"][0, 0]
    realized = history["realized_gains"][0, 0]
    assert realized == pytest.approx(0.5 * sale)
    expected_tax = calculate_annual_taxes_at(0.0, 0.0, realized, 0.0)["total"] + HEALTH
    assert history["taxes_paid"][0, 0] == pytest.approx(expected_tax, rel=1e-9)
    # Cash never goes negative: the sale exactly covered expenses + taxes.
    assert history["liquid_assets_by_class"][0, 0, 2] == pytest.approx(0.0, abs=1e-6)


def test_large_sale_falls_back_to_flat_kest():
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0
    cfg = _config(annual_base_expenses=600_000.0)
    history = run_simulation(cfg, returns, _zero_inflation(cfg))
    realized = history["realized_gains"][0, 0]
    assert history["tax_components"]["kest"][0, 0] == pytest.approx(0.275 * realized, rel=1e-9)
    assert history["tax_components"]["income_tax"][0, 0] == pytest.approx(0.0)


def test_year_end_rebalance_gains_are_taxed_next_year():
    returns = np.zeros((1, 24, 5))
    returns[0, 0, 0] = 1.0  # year 0, month 0: US 500k -> 1M (basis 500k); cash stays 500k
    cfg = _config(duration_years=2, alloc_us_stocks=0.5, alloc_chf_cash=0.5, rebalance_strategy="Yearly")
    history = run_simulation(cfg, returns, _zero_inflation(cfg))

    s0 = history["expenses_paid"][0, 0] + history["taxes_paid"][0, 0]
    # A year-end rebalance follows, so the outflow is paid from cash first and realizes no gain.
    # The rebalance back to 50/50 of (1.5M - s0) then sells 250k + s0/2 of US at basis ratio 0.5.
    eoy_gain = 125_000.0 + s0 / 4.0

    assert history["realized_gains"][0, 0] == pytest.approx(eoy_gain)
    assert history["taxes_paid"][0, 0] == pytest.approx(HEALTH, rel=1e-9)
    # Year 1's outflow is again paid from cash; only the carried rebalance gain is taxed.
    assert history["taxes_paid"][1, 0] == pytest.approx(
        calculate_annual_taxes_at(0.0, 0.0, eoy_gain, 0.0)["total"] + HEALTH * WAGE, rel=1e-9
    )


def test_outflow_paid_from_cash_before_year_end_rebalance():
    """With a year-end rebalance following, the outflow must be paid from cash (e.g. the pension
    credited during the year) instead of selling appreciated holdings and buying them back."""
    returns = np.zeros((1, 24, 5))
    returns[0, 0, 0] = 1.0  # US doubles: basis / value = 0.5
    cfg = _config(
        duration_years=2,
        start_age=65,
        monthly_ahv_pension=2_000.0,
        annual_base_expenses=10_000.0,
        alloc_us_stocks=1.0,
        rebalance_strategy="Yearly",
    )
    history = run_simulation(cfg, returns, _zero_inflation(cfg))
    pension_tax = calculate_annual_taxes_at(28_000.0, 0.0, 0.0, 0.0)["total"]
    # 28k of pension covers 10k expenses + pension tax: nothing is sold, no KESt.
    np.testing.assert_allclose(history["tax_components"]["kest"][:, 0], 0.0, atol=1e-9)
    assert history["realized_gains"][0, 0] == pytest.approx(0.0, abs=1e-6)
    assert history["taxes_paid"][0, 0] == pytest.approx(pension_tax)
    # The leftover cash is invested at year end (target 100% US).
    assert history["liquid_assets_by_class"][0, 0, 2] == pytest.approx(0.0, abs=1e-6)
    assert history["liquid_assets"][0, 0] == pytest.approx(2_000_000.0 + 28_000.0 - 10_000.0 - pension_tax)

    # Never rebalancing keeps the proportional sale (cash stays where it is).
    never = run_simulation(_config(**{**cfg.__dict__, "rebalance_strategy": "Never"}), returns, _zero_inflation(cfg))
    assert never["liquid_assets_by_class"][0, 0, 2] > 0.0


def test_cash_first_does_not_change_swiss_results():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.006, 0.04, size=(5, 10 * 12, 5))
    common = dict(
        num_runs=5, duration_years=10, inflation_mean=0.0, inflation_std=0.0, start_age=60, dividend_yield=0.02,
        initial_liquid_wealth=1_000_000.0, alloc_us_stocks=0.6, alloc_chf_cash=0.4, annual_base_expenses=60_000.0,
        monthly_ahv_pension=2_000.0, rebalance_strategy="Yearly", enable_smart_selling=False,
    )
    h = run_simulation(SimConfig(**common), returns, np.zeros((5, 10)))
    # Without a capital gains tax the composition of the sale is irrelevant after the rebalance:
    # every year ends exactly at the target weights.
    weights = h["liquid_assets_by_class"] / h["liquid_assets"][:, :, None]
    np.testing.assert_allclose(weights[:, :, 0], 0.6)
    np.testing.assert_allclose(weights[:, :, 2], 0.4)


def test_rebalancing_costs_tax_in_austria_but_not_in_switzerland():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.008, 0.04, size=(20, 10 * 12, 5))
    returns[:, :, 2] = 0.0
    common = dict(num_runs=20, duration_years=10, alloc_us_stocks=0.6, alloc_gold=0.2, alloc_chf_cash=0.2)
    at_monthly = run_simulation(_config(**common, rebalance_strategy="Monthly"), returns, np.zeros((20, 10)))
    at_never = run_simulation(_config(**common, rebalance_strategy="Never"), returns, np.zeros((20, 10)))
    assert at_monthly["tax_components"]["kest"].sum() + at_monthly["tax_components"]["income_tax"].sum() > (
        at_never["tax_components"]["kest"].sum() + at_never["tax_components"]["income_tax"].sum()
    )


def test_state_pension_taxed_and_replaces_self_insurance():
    cfg = _config(start_age=65, monthly_ahv_pension=2_000.0, annual_base_expenses=0.0)
    history = run_simulation(cfg, np.zeros((1, 12, 5)), _zero_inflation(cfg))
    assert history["income_ahv"][0, 0] == pytest.approx(28_000.0)  # 14 payments
    expected = calculate_annual_taxes_at(28_000.0, 0.0, 0.0, 0.0)
    assert history["taxes_paid"][0, 0] == pytest.approx(expected["total"])
    assert history["tax_components"]["health_insurance"][0, 0] == pytest.approx(0.06 * 28_000.0)
    # Pension credited monthly as 14/12 of the monthly amount; the year-end outflow is the tax only.
    assert history["liquid_assets"][0, 0] == pytest.approx(1_000_000.0 + 28_000.0 - expected["total"], abs=1e-6)


def test_self_insurance_continues_without_pension():
    cfg = _config(start_age=70, monthly_ahv_pension=0.0, annual_base_expenses=0.0)
    history = run_simulation(cfg, np.zeros((1, 12, 5)), _zero_inflation(cfg))
    assert history["taxes_paid"][0, 0] == pytest.approx(HEALTH)


def test_health_insurance_follows_cpi_times_real_wage_growth():
    cfg = _config(duration_years=3, annual_base_expenses=0.0, health_insurance_real_growth=0.01)
    inflation = np.full((1, 3), 0.10)
    history = run_simulation(cfg, np.zeros((1, 36, 5)), inflation)
    # Year 0 pays today's rate in real terms; the real wage premium compounds from Year 1.
    np.testing.assert_allclose(
        history["tax_components"]["health_insurance"][:, 0],
        [HEALTH * 1.1, HEALTH * 1.21 * 1.01, HEALTH * 1.331 * 1.01**2],
    )


def test_health_insurance_plain_cpi_with_zero_real_growth():
    cfg = _config(duration_years=2, annual_base_expenses=0.0, health_insurance_real_growth=0.0)
    inflation = np.full((1, 2), 0.10)
    history = run_simulation(cfg, np.zeros((1, 24, 5)), inflation)
    np.testing.assert_allclose(history["tax_components"]["health_insurance"][:, 0], [HEALTH * 1.1, HEALTH * 1.21])


def test_health_insurance_default_real_growth_and_validation():
    from src.tax_engine_at import HEALTH_SELF_INSURANCE_REAL_GROWTH

    assert _config().health_insurance_real_growth == HEALTH_SELF_INSURANCE_REAL_GROWTH == pytest.approx(0.007)
    with pytest.raises(ValueError, match="health_insurance_real_growth"):
        _config(health_insurance_real_growth=-1.0)


def test_year_0_estimate_austria():
    cfg = _config(dividend_yield=0.02, alloc_us_stocks=0.8, alloc_chf_cash=0.2, cash_rate=0.02)
    dividends = 800_000.0 * 0.02
    interest = 200_000.0 * 0.02
    expected = calculate_annual_taxes_at(0.0, dividends, 0.0, interest)["total"] + HEALTH
    assert estimate_year_0_taxes(cfg) == pytest.approx(expected)


def test_watermark_deducts_latent_kest():
    jur = Austria()
    cfg = _config()
    latent = jur.latent_tax(cfg, np.zeros(2), [], 41, np.ones(2), np.array([100_000.0, -50_000.0]))
    np.testing.assert_allclose(latent, [27_500.0, 0.0])


def test_bracket_indexation_applied_to_austrian_taxes():
    jur = Austria()
    cfg = _config(monthly_ahv_pension=2_500.0)
    f = np.array([1.0, 1.25])
    pension = np.array([30_000.0, 37_500.0])
    taxes = jur.assess_annual_taxes(
        cfg,
        current_age=70,
        total_liquid_end=np.zeros(2),
        current_expenses=np.zeros(2),
        dividends=np.zeros(2),
        interest=np.zeros(2),
        annual_pension=pension,
        bracket_factors=f,
        realized_gains=np.zeros(2),
        inflation_factors=np.ones(2),
    )
    # Same real pension, brackets indexed by the same factor -> tax scales exactly.
    assert taxes.total[1] == pytest.approx(1.25 * taxes.total[0])


def test_full_horizon_austria_runs_with_eur_data():
    from src.historic_returns_eur import generate_bootstrapped_data

    cfg = _config(
        num_runs=200,
        duration_years=40,
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.3,
        alloc_chf_cash=0.1,
        alloc_gold=0.05,
        alloc_bitcoin=0.05,
        dividend_yield=0.02,
        annual_base_expenses=45_000.0,
        monthly_ahv_pension=2_000.0,
        rebalance_strategy="Cash Tent",
        spending_strategy="Vanguard Dynamic",
        enable_smart_selling=True,
    )
    returns, inflation = generate_bootstrapped_data(200, 40, seed=1, real_eur_appreciation=0.0)
    history = run_simulation(cfg, returns, inflation)
    assert np.all(np.isfinite(history["net_worth"]))
    assert set(history["tax_components"]) == {"income_tax", "kest", "health_insurance"}
    np.testing.assert_allclose(sum(history["tax_components"].values()), history["taxes_paid"], rtol=1e-12)
    assert np.all(history["taxes_paid"] >= 0.0)


def test_outflow_sale_loss_offset_converges_without_oscillation():
    """When holdings sit far below cost basis (gain_fraction << -1) and carried gains are positive,
    Picard fixed-point iteration oscillates (|g * 0.275| > 1) and panic-sells equities into cash.
    Monotone bisection must find the exact equilibrium sale with zero unrequested cash residual."""
    returns = np.zeros((1, 24, 5))
    returns[0, 0, 0] = 4.0     # Year 0 Month 0: US stocks 5x -> Yearly rebalance carries large gain
    returns[0, 12, :] = -0.92  # Year 1 Month 0: 92% crash -> basis / value >> 1 (gain_fraction << -1)

    cfg = _config(
        duration_years=2,
        alloc_us_stocks=0.5,
        alloc_non_us_stocks=0.5,
        annual_base_expenses=5_000.0,
        rebalance_strategy="Yearly",
        enable_smart_selling=True,
    )
    history = run_simulation(cfg, returns, _zero_inflation(cfg))

    # Smart Cash Buffer skips the Year-1 EOY rebalance in a downturn, exposing any sale/tax mismatch
    # directly in the cash sleeve (index 2), which has 0% target weight and 0% returns.
    assert history["liquid_assets_by_class"][1, 0, 2] == pytest.approx(0.0, abs=1e-8)

    # Pre-sale Year 1 portfolio value was 0.08 * Year-0 ending net worth; exact sale = expenses + taxes.
    pre_sale_y1 = 0.08 * history["net_worth"][0, 0]
    sale_y1 = history["expenses_paid"][1, 0] + history["taxes_paid"][1, 0]
    assert history["net_worth"][1, 0] == pytest.approx(pre_sale_y1 - sale_y1, rel=1e-10)


def test_watermark_includes_unassessed_realized_gains_from_rebalancing():
    """Rebalancing before year-end KESt settlement converts unrealized gains into realized gains
    without paying the tax yet; after-tax watermark checks must still deduct that accrued KESt."""
    # Year 0: US stocks gain +20% (500k -> 600k; total 1.1M). Expenses + health ins ~ 96.8k -> end ~1,003k.
    # Yearly rebalance realizes gains carried into Year 1, where gross NW (~1,003k) > 1,000k start
    # only because the accrued KESt on carried + unrealized gains (> 20k * 27.5% = 5.5k) is unpaid.
    returns = np.zeros((1, 24, 5))
    returns[0, 0, 0] = 0.20
    common = dict(
        duration_years=2,
        alloc_us_stocks=0.5,
        alloc_chf_cash=0.5,
        annual_base_expenses=85_000.0,
        enable_smart_selling=False,
    )
    h_yearly = run_simulation(_config(**common, rebalance_strategy="Yearly"), returns, _zero_inflation(_config(duration_years=2)))
    h_never = run_simulation(_config(**common, rebalance_strategy="Never"), returns, _zero_inflation(_config(duration_years=2)))

    # At Year 1 Month 11 (before Year 1 expenses/taxes), gross NW is slightly > 1,000k, but after
    # deducting accrued + latent KESt it is < 1,000k under BOTH Yearly and Never rebalancing.
    assert h_never["below_watermark"][1, 0]
    assert h_yearly["below_watermark"][1, 0]


def test_dividends_are_distributed_and_fund_the_outflow():
    """Austria: dividends are paid out to cash and fund the year-end outflow before shares are sold,
    so the distribution itself does not trigger a sale of appreciated holdings."""
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0  # US doubles: 2M value, 1M basis
    cfg = _config(dividend_yield=0.05, annual_base_expenses=50_000.0, rebalance_strategy="Never")
    history = run_simulation(cfg, returns, _zero_inflation(cfg))

    dividends = history["income_dividends"][0, 0]
    assert dividends == pytest.approx(100_000.0)
    # Expenses + taxes (< 100k) are paid entirely from the distribution: nothing is sold.
    assert history["realized_gains"][0, 0] == pytest.approx(0.0, abs=1e-6)
    expected_tax = calculate_annual_taxes_at(0.0, dividends, 0.0, 0.0)["total"] + HEALTH
    assert history["taxes_paid"][0, 0] == pytest.approx(expected_tax)
    # The stock position drops by the distribution with its basis unchanged; the rest stays in cash.
    assert history["liquid_assets_by_class"][0, 0, 0] == pytest.approx(1_900_000.0)
    assert history["cost_basis_by_class"][0, 0, 0] == pytest.approx(1_000_000.0)
    assert history["liquid_assets_by_class"][0, 0, 2] == pytest.approx(100_000.0 - 50_000.0 - expected_tax)
    np.testing.assert_allclose(sum(history["tax_components"].values()), history["taxes_paid"])


def test_dividends_partly_cover_the_outflow_and_the_rest_is_sold():
    returns = np.zeros((1, 12, 5))
    returns[0, 0, 0] = 1.0
    cfg = _config(dividend_yield=0.01, annual_base_expenses=80_000.0, rebalance_strategy="Never")
    history = run_simulation(cfg, returns, _zero_inflation(cfg))
    sale = history["expenses_paid"][0, 0] + history["taxes_paid"][0, 0] - 20_000.0  # 20k distributed
    # After the distribution the position is 1.98M with a 1M basis.
    assert history["realized_gains"][0, 0] == pytest.approx((1.0 - 1_000_000.0 / 1_980_000.0) * sale)
    # Money is conserved: 2M - outflow (the distribution only moved value from stocks to cash).
    assert history["liquid_assets"][0, 0] == pytest.approx(
        2_000_000.0 - history["expenses_paid"][0, 0] - history["taxes_paid"][0, 0]
    )
