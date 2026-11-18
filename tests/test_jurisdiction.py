import numpy as np
import pytest

from src.jurisdiction import (
    COUNTRY_DEFAULTS,
    PLANNED_COUNTRIES,
    STATE_PENSION_AGE,
    SUPPORTED_COUNTRIES,
    Switzerland,
    get_jurisdiction,
    indexed_tax,
)
from src.simulation_engine import (
    SimConfig,
    _indexed_tax,
    estimate_year_0_taxes,
    run_simulation,
)
from src.tax_engine import (
    ZURICH_CANTONAL_MULTIPLIER,
    ZURICH_CITY_MUNICIPAL_MULTIPLIER,
    calculate_ahv_non_worker,
    calculate_capital_withdrawal_tax,
    calculate_income_tax,
    calculate_wealth_tax,
)


def _config(**overrides) -> SimConfig:
    params = dict(
        num_runs=3,
        duration_years=8,
        inflation_mean=0.02,
        inflation_std=0.01,
        start_age=60,
        dividend_yield=0.02,
        initial_liquid_wealth=1_500_000.0,
        initial_pillar_2=300_000.0,
        initial_pillar_3a_accounts=[40_000.0, 40_000.0],
        alloc_us_stocks=0.6,
        alloc_non_us_stocks=0.2,
        alloc_chf_cash=0.2,
        annual_base_expenses=70_000.0,
        monthly_ahv_pension=2_000.0,
    )
    params.update(overrides)
    return SimConfig(**params)


def test_country_defaults_and_registry():
    assert SUPPORTED_COUNTRIES == ("CH", "AT")
    assert PLANNED_COUNTRIES == ()
    ch = COUNTRY_DEFAULTS["CH"]
    assert (ch.currency, ch.pension_payments_per_year, ch.bracket_indexation, ch.cash_rate) == ("CHF", 12, 1.0, 0.01)
    assert isinstance(get_jurisdiction("CH"), Switzerland)
    assert get_jurisdiction("CH").defaults is ch


def test_simconfig_defaults_come_from_country_defaults():
    cfg = _config()
    assert cfg.country == "CH"
    assert cfg.bracket_indexation == COUNTRY_DEFAULTS["CH"].bracket_indexation
    assert cfg.cash_rate == COUNTRY_DEFAULTS["CH"].cash_rate


def test_austria_is_supported():
    cfg = _config(country="AT", initial_pillar_2=0.0, initial_pillar_3a_accounts=[])
    assert cfg.country == "AT"
    assert get_jurisdiction("AT").taxes_capital_gains
    assert not get_jurisdiction("CH").taxes_capital_gains


def test_unknown_country_is_rejected():
    with pytest.raises(ValueError, match="Unsupported country 'XX'") as excinfo:
        _config(country="XX")
    assert "not implemented yet" not in str(excinfo.value)


def test_indexed_tax_alias_is_kept_for_backwards_compatibility():
    assert _indexed_tax is indexed_tax


def test_switzerland_annual_taxes_match_tax_engine():
    ch = Switzerland()
    cfg = _config(municipal_multiplier=1.3)
    liquid = np.array([900_000.0, 2_000_000.0, 5_000_000.0])
    expenses = np.full(3, 80_000.0)
    dividends = np.array([10_000.0, 25_000.0, 60_000.0])
    interest = np.array([500.0, 1_000.0, 0.0])
    pension = np.zeros(3)
    factors = np.array([1.0, 1.1, 1.25])

    taxes = ch.assess_annual_taxes(
        cfg,
        current_age=50,
        total_liquid_end=liquid,
        current_expenses=expenses,
        dividends=dividends,
        interest=interest,
        annual_pension=pension,
        bracket_factors=factors,
    )

    taxable_wealth = liquid - expenses
    wealth = factors * calculate_wealth_tax(taxable_wealth / factors, ZURICH_CANTONAL_MULTIPLIER, 1.3)
    ahv = factors * calculate_ahv_non_worker(taxable_wealth / factors)
    taxable_income = dividends + interest - ahv
    income = factors * calculate_income_tax(taxable_income / factors, ZURICH_CANTONAL_MULTIPLIER, 1.3)

    np.testing.assert_allclose(taxes.components["wealth_tax"], wealth)
    np.testing.assert_allclose(taxes.components["ahv_contributions"], ahv)
    np.testing.assert_allclose(taxes.components["income_tax"], income)
    np.testing.assert_allclose(taxes.total, sum(taxes.components.values()))


def test_switzerland_no_ahv_contributions_from_pension_age():
    taxes = Switzerland().assess_annual_taxes(
        _config(),
        current_age=STATE_PENSION_AGE,
        total_liquid_end=np.array([2_000_000.0]),
        current_expenses=np.array([80_000.0]),
        dividends=np.array([20_000.0]),
        interest=np.array([0.0]),
        annual_pension=np.array([24_000.0]),
        bracket_factors=np.ones(1),
    )
    assert taxes.components["ahv_contributions"][0] == 0.0
    expected_income = calculate_income_tax(44_000.0, ZURICH_CANTONAL_MULTIPLIER, ZURICH_CITY_MUNICIPAL_MULTIPLIER)
    assert np.isclose(taxes.components["income_tax"][0], expected_income)


def test_switzerland_liquidate_pensions_schedule_and_tax():
    ch = Switzerland()
    cfg = _config()
    pillar_2 = np.full(2, 300_000.0)
    pillar_3a = [np.full(2, 40_000.0), np.full(2, 40_000.0)]
    ones = np.ones(2)

    # Age 59: nothing due.
    liq = ch.liquidate_pensions(cfg, pillar_2, pillar_3a, 59, ones)
    assert np.all(liq.gross == 0.0) and np.all(liq.tax == 0.0)

    # Age 60: only the first 3a account.
    liq = ch.liquidate_pensions(cfg, pillar_2, pillar_3a, 60, ones)
    np.testing.assert_allclose(liq.gross, 40_000.0)
    np.testing.assert_allclose(liq.tax, calculate_capital_withdrawal_tax(40_000.0))
    assert np.all(pillar_3a[0] == 0.0) and np.all(pillar_3a[1] == 40_000.0)

    # Age 65: remaining 3a account plus Pillar 2 in one taxable event.
    liq = ch.liquidate_pensions(cfg, pillar_2, pillar_3a, 65, ones)
    np.testing.assert_allclose(liq.gross, 340_000.0)
    np.testing.assert_allclose(liq.tax, calculate_capital_withdrawal_tax(340_000.0))
    assert np.all(pillar_2 == 0.0) and np.all(pillar_3a[1] == 0.0)


def test_switzerland_year_0_estimate_matches_public_estimate():
    cfg = _config(rebalance_strategy="Yearly", start_age=45)
    weights = np.array([cfg.alloc_us_stocks, cfg.alloc_non_us_stocks, cfg.alloc_chf_cash, cfg.alloc_gold, cfg.alloc_bitcoin])
    ch = Switzerland()
    init_wealth = ch.year_0_liquid_wealth(cfg)
    assert init_wealth == cfg.initial_liquid_wealth  # no Day-1 pension payout before 60
    assert ch.estimate_year_0_taxes_for_weights(cfg, init_wealth, weights) == estimate_year_0_taxes(cfg)


def test_explicit_country_ch_matches_default():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.005, 0.04, size=(3, 8 * 12, 5))
    inflation = rng.normal(0.02, 0.01, size=(3, 8))
    default = run_simulation(_config(), returns, inflation)
    explicit = run_simulation(_config(country="CH"), returns, inflation)
    for key, value in default.items():
        if isinstance(value, dict):
            assert value.keys() == explicit[key].keys()
            for sub, arr in value.items():
                np.testing.assert_array_equal(arr, explicit[key][sub])
        else:
            np.testing.assert_array_equal(np.asarray(value), np.asarray(explicit[key]))


# --------------------------------------------------------------------------- bracket indexation in deflation


def _capture_bracket_factors(monkeypatch, cls, config, inflation):
    """Runs the engine and returns the `bracket_factors` passed to each year's tax assessment."""
    seen = {}
    original = cls.assess_annual_taxes

    def spy(self, cfg, **kwargs):
        # The Austrian engine re-assesses during the year-end sale; keep the first call per year.
        seen.setdefault(kwargs["current_age"], kwargs["bracket_factors"].copy())
        return original(self, cfg, **kwargs)

    monkeypatch.setattr(cls, "assess_annual_taxes", spy)
    months = config.duration_years * 12
    run_simulation(config, np.zeros((config.num_runs, months, 5)), inflation)
    return np.array([seen[age] for age in sorted(seen)])


def test_swiss_brackets_follow_cpi_high_water_mark(monkeypatch):
    # Art. 39 Abs. 2 DBG: no downward adjustment; the next one is measured from the last adjustment.
    cfg = _config(num_runs=1, duration_years=4, initial_pillar_2=0.0, initial_pillar_3a_accounts=[])
    inflation = np.array([[0.10, -0.10, 0.05, 0.10]])
    factors = _capture_bracket_factors(monkeypatch, Switzerland, cfg, inflation)
    cpi = np.cumprod(1.0 + inflation[0])  # 1.10, 0.99, 1.0395, 1.14345
    np.testing.assert_allclose(factors[:, 0], [1.10, 1.10, 1.10, cpi[3]])


def test_swiss_partial_indexation_high_water_mark(monkeypatch):
    cfg = _config(num_runs=1, duration_years=3, initial_pillar_2=0.0, initial_pillar_3a_accounts=[], bracket_indexation=0.5)
    inflation = np.array([[0.10, -0.10, 0.20]])
    factors = _capture_bracket_factors(monkeypatch, Switzerland, cfg, inflation)
    # Half-indexed level: 1.05, 0.9975, 1.09725 -> high-water mark 1.05, 1.05, 1.09725.
    np.testing.assert_allclose(factors[:, 0], [1.05, 1.05, 1.05 * 0.95 * 1.10])


def test_austrian_brackets_skip_deflation_years(monkeypatch):
    # § 33a EStG: only positive inflation is passed on (2/3 by default); deflation years are skipped.
    from src.jurisdiction import Austria

    cfg = SimConfig(
        country="AT", num_runs=1, duration_years=3, inflation_mean=0.0, inflation_std=0.0, start_age=40,
        dividend_yield=0.0, initial_liquid_wealth=1_000_000.0, alloc_us_stocks=1.0, annual_base_expenses=40_000.0,
    )
    inflation = np.array([[0.06, -0.10, 0.03]])
    factors = _capture_bracket_factors(monkeypatch, Austria, cfg, inflation)
    np.testing.assert_allclose(factors[:, 0], [1.04, 1.04, 1.04 * 1.02])


@pytest.mark.parametrize("country", ["CH", "AT"])
def test_brackets_never_decrease_under_random_inflation(monkeypatch, country):
    from src.jurisdiction import Austria

    extra = {} if country == "CH" else dict(initial_pillar_2=0.0, initial_pillar_3a_accounts=[])
    cfg = _config(country=country, num_runs=50, duration_years=20, **extra)
    inflation = np.random.default_rng(0).normal(0.01, 0.04, (50, 20))
    cls = Switzerland if country == "CH" else Austria
    factors = _capture_bracket_factors(monkeypatch, cls, cfg, inflation)
    assert np.all(np.diff(factors, axis=0) >= 0.0)
    assert np.all(factors >= 1.0)


# --------------------------------------------------------------------------- US withholding tax / DA-1 credit


def _assess_us(cfg, us_dividends, other_dividends=0.0, interest=0.0, pension=0.0, age=50, factors=None):
    us = np.atleast_1d(np.asarray(us_dividends, dtype=float))
    n = us.shape[0]
    return Switzerland().assess_annual_taxes(
        cfg,
        current_age=age,
        total_liquid_end=np.full(n, 2_000_000.0),
        current_expenses=np.full(n, 80_000.0),
        dividends=us + other_dividends,
        interest=np.full(n, interest),
        annual_pension=np.full(n, pension),
        bracket_factors=np.ones(n) if factors is None else factors,
        us_dividends=us,
    )


def test_da1_credit_is_capped_at_swiss_tax_on_the_us_dividends():
    from src.tax_engine import calculate_foreign_tax_credit

    cfg = _config()
    us = np.array([20_000.0, 200_000.0])
    taxes = _assess_us(cfg, us, age=STATE_PENSION_AGE, pension=24_000.0)
    taxable = us + 24_000.0  # gross dividends + pension (no AHV contributions from 65)
    gross_tax = calculate_income_tax(taxable, cfg.cantonal_multiplier, cfg.municipal_multiplier)
    withheld = 0.15 * us
    credit = np.minimum(withheld, us * gross_tax / taxable)

    np.testing.assert_allclose(taxes.components["us_withholding_tax"], withheld)
    np.testing.assert_allclose(taxes.details["foreign_tax_credit"], credit)
    np.testing.assert_allclose(taxes.components["income_tax"], gross_tax - credit)
    np.testing.assert_allclose(taxes.details["taxable_income"], taxable)
    np.testing.assert_allclose(taxes.total, sum(taxes.components.values()))
    np.testing.assert_allclose(calculate_foreign_tax_credit(withheld, us, gross_tax, taxable), credit)
    # Low income: the average Swiss rate is below 15%, so part of the withholding is lost. High
    # income: the average rate exceeds 15% and the withholding is credited in full.
    assert credit[0] < withheld[0]
    assert credit[1] == pytest.approx(withheld[1])


def test_da1_credit_needs_chf_100_of_withholding_else_dividends_are_taxed_net():
    cfg = _config()
    # 600 * 15% = 90 < CHF 100: no credit, dividends declared net of the withholding.
    taxes = _assess_us(cfg, 600.0, other_dividends=40_000.0, age=STATE_PENSION_AGE)
    expected = calculate_income_tax(40_000.0 + 510.0, cfg.cantonal_multiplier, cfg.municipal_multiplier)
    assert taxes.details["foreign_tax_credit"][0] == 0.0
    assert taxes.components["us_withholding_tax"][0] == pytest.approx(90.0)
    assert taxes.components["income_tax"][0] == pytest.approx(expected)


def test_ireland_domiciled_etf_loses_withholding_and_is_taxed_net():
    cfg = _config(us_withholding_credit=False)
    taxes = _assess_us(cfg, 50_000.0, age=STATE_PENSION_AGE)
    expected = calculate_income_tax(42_500.0, cfg.cantonal_multiplier, cfg.municipal_multiplier)
    assert taxes.details["foreign_tax_credit"][0] == 0.0
    assert taxes.components["us_withholding_tax"][0] == pytest.approx(7_500.0)
    assert taxes.components["income_tax"][0] == pytest.approx(expected)


def test_zero_withholding_rate_reproduces_the_model_without_withholding():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.005, 0.04, size=(3, 8 * 12, 5))
    inflation = rng.normal(0.02, 0.01, size=(3, 8))
    off = run_simulation(_config(us_dividend_withholding_rate=0.0), returns, inflation)
    on = run_simulation(_config(), returns, inflation)
    assert "us_withholding_tax" not in off["tax_components"]
    np.testing.assert_array_equal(off["tax_details"]["foreign_tax_credit"], 0.0)
    # The withholding net of the credit is an extra cost.
    lost = on["tax_components"]["us_withholding_tax"] - on["tax_details"]["foreign_tax_credit"]
    assert np.all(lost >= -1e-9) and np.any(lost > 0.0)
    assert on["net_worth"][-1].sum() < off["net_worth"][-1].sum()


def test_us_withholding_rate_is_validated():
    with pytest.raises(ValueError, match="us_dividend_withholding_rate"):
        _config(us_dividend_withholding_rate=1.0)
    with pytest.raises(ValueError, match="us_withholding_credit"):
        _config(us_withholding_credit="yes")


def test_year_0_breakdown_matches_estimate_and_components():
    from src.simulation_engine import estimate_year_0_tax_breakdown

    for cfg in (
        _config(rebalance_strategy="Cash Tent", start_age=45),
        _config(rebalance_strategy="Yearly", start_age=66),
        _config(country="AT", initial_pillar_2=0.0, initial_pillar_3a_accounts=[], start_age=45),
    ):
        y0 = estimate_year_0_tax_breakdown(cfg)
        assert float(y0.total[0]) == estimate_year_0_taxes(cfg)
        assert float(y0.total[0]) == pytest.approx(sum(float(v[0]) for v in y0.components.values()))
    ch = estimate_year_0_tax_breakdown(_config(rebalance_strategy="Yearly", start_age=45))
    us_dividends = 1_500_000.0 * 0.6 * 0.02
    assert ch.components["us_withholding_tax"][0] == pytest.approx(0.15 * us_dividends)
    assert 0.0 < ch.details["foreign_tax_credit"][0] <= 0.15 * us_dividends


def test_da1_credit_never_makes_income_tax_negative():
    # Large wealth -> AHV non-worker contribution near the maximum, deducted from income: the
    # taxable income falls below the US dividends, so the credit is capped at the whole Swiss tax
    # (never exceeding `income_tax` even by 1 ULP of `a * (b / a)` rounding).
    cfg = _config(initial_liquid_wealth=8_000_000.0)
    us = np.linspace(35_000.0, 65_000.0, 200)
    taxes = Switzerland().assess_annual_taxes(
        cfg,
        current_age=50,
        total_liquid_end=np.full(200, 8_000_000.0),
        current_expenses=np.full(200, 80_000.0),
        dividends=us,
        interest=np.zeros(200),
        annual_pension=np.zeros(200),
        bracket_factors=np.ones(200),
        us_dividends=us,
    )
    assert np.all(taxes.details["taxable_income"] < us)
    assert np.all(taxes.components["income_tax"] >= 0.0)
    np.testing.assert_allclose(taxes.components["income_tax"], 0.0, atol=1e-9)
    assert np.all(taxes.details["foreign_tax_credit"] >= 0.0)
