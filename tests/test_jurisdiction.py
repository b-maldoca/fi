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
