"""Country-specific tax, contribution and pension rules behind one interface.

`simulation_engine.run_simulation` is country-agnostic: everything that depends on
the tax jurisdiction (income/wealth taxes, mandatory social contributions, pension
lump-sum liquidations and the tax still owed on them) is reached through the
`Jurisdiction` returned by `get_jurisdiction(config.country)`.

Implemented: Switzerland (Canton Zurich), `"CH"`, and Austria, `"AT"` (PRD §8 /
Design Doc §7; tax rules in `tax_engine_at`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .tax_engine import (
    FOREIGN_TAX_CREDIT_MIN_CLAIM,
    calculate_ahv_non_worker,
    calculate_capital_withdrawal_tax,
    calculate_foreign_tax_credit,
    calculate_income_tax,
    calculate_wealth_tax,
)
from .tax_engine_at import (
    KEST_RATE,
    PENSION_PAYMENTS_PER_YEAR,
    calculate_annual_taxes_at,
)

if TYPE_CHECKING:
    from .simulation_engine import SimConfig


# Statutory age at which state pensions start and Swiss Pillar 2 is paid out.
STATE_PENSION_AGE = 65


@dataclass(frozen=True)
class CountryDefaults:
    """Country-level defaults shared by `SimConfig` and the UI."""

    code: str
    name: str
    currency: str
    # State pension payments per calendar year (12 in CH; 14 in AT incl. 13th/14th payment).
    pension_payments_per_year: int
    # Default fraction of CPI applied to tax bracket edges.
    bracket_indexation: float
    # Default forward-looking cash rate (Year 0 tax estimate, Monte Carlo cash mean).
    cash_rate: float


COUNTRY_DEFAULTS: dict[str, CountryDefaults] = {
    "CH": CountryDefaults(
        code="CH",
        name="Switzerland (Zurich)",
        currency="CHF",
        pension_payments_per_year=12,
        bracket_indexation=1.0,  # Art. 39 DBG / § 48 StG ZH: full CPI indexation
        cash_rate=0.01,
    ),
    "AT": CountryDefaults(
        code="AT",
        name="Austria",
        currency="EUR",
        pension_payments_per_year=PENSION_PAYMENTS_PER_YEAR,  # 12 regular + 13th/14th payment
        bracket_indexation=2.0 / 3.0,  # § 33a EStG: automatic indexation by 2/3 of inflation
        # ECB MIR, Austrian households, Sep 2026: agreed maturity <= 1y 2.18%, notice deposits 2.05%.
        cash_rate=0.02,
    ),
}
SUPPORTED_COUNTRIES: tuple[str, ...] = tuple(COUNTRY_DEFAULTS)
# Specified in the docs but not implemented yet; rejected by `SimConfig` validation.
PLANNED_COUNTRIES: tuple[str, ...] = ()


def indexed_tax(tax_fn, taxable_amount: np.ndarray, bracket_factors: np.ndarray, *args) -> np.ndarray:
    """
    Applies a progressive tariff whose bracket edges are scaled by `bracket_factors`.

    Swiss law indexes tax bracket edges to the CPI so that purely nominal growth
    does not push a taxpayer into higher brackets: Art. 39 DBG requires the EFD to
    adjust the federal tariff annually, and § 48 StG ZH indexes the Canton Zurich
    income *and* wealth tariffs.

    Every tariff in `tax_engine` is positively homogeneous of degree 1 under a
    simultaneous scaling of the tax base and the bracket edges. This holds for the
    piecewise-linear income and wealth tariffs, and also for the AHV non-worker step
    function, because floor division is scale invariant (`(f*a) // (f*b) == a // b`).
    Scaling the brackets by `f` is therefore exactly equivalent to deflating the base
    by `f`, applying the unscaled tariff, and re-inflating the result:

        T_indexed(x, f) == f * T_nominal(x / f)

    Note that only currency-denominated amounts scale; the bracket *rates* and the
    Steuerfuss multipliers are unaffected, which is correct since both are political
    parameters rather than inflation-linked ones.
    """
    return bracket_factors * tax_fn(taxable_amount / bracket_factors, *args)


def next_bracket_factors_skip_deflation(
    bracket_factors: np.ndarray, indexed_cpi_level: np.ndarray, inflation: np.ndarray, indexation: float
) -> np.ndarray:
    """Austrian rule (§ 33a EStG): only a positive inflation rate is passed to the brackets.

    A deflation year leaves the brackets unchanged and is never clawed back: the next positive
    inflation is applied to the unchanged brackets.
    """
    return bracket_factors * (1.0 + indexation * np.maximum(0.0, inflation))


def next_bracket_factors_high_water_mark(
    bracket_factors: np.ndarray, indexed_cpi_level: np.ndarray, inflation: np.ndarray, indexation: float
) -> np.ndarray:
    """Swiss rule (Art. 39 Abs. 2 DBG, § 48 StG ZH): no downward adjustment after deflation, and the
    next adjustment is measured from the index level of the last adjustment.

    The brackets therefore follow the high-water mark of the (indexation-weighted) CPI level.
    """
    return np.maximum(bracket_factors, indexed_cpi_level)


@dataclass
class PensionLiquidation:
    """Start-of-year pension lump sums paid out, per run."""

    gross: np.ndarray  # amount liquidated before tax
    tax: np.ndarray  # tax deducted at source


@dataclass
class AnnualTaxes:
    """Year-end tax assessment, per run."""

    total: np.ndarray  # all taxes and mandatory contributions due at year end
    components: dict[str, np.ndarray]  # country-specific breakdown (sums to `total`)
    # Informational amounts that are not part of the sum, e.g. the DA-1 foreign tax credit
    # (already deducted from `income_tax`) and the taxable income the income tax was computed on.
    details: dict[str, np.ndarray] = field(default_factory=dict)


class Jurisdiction(Protocol):
    """Country-specific rules used by `run_simulation` and the Year 0 estimates."""

    defaults: CountryDefaults
    # True where realized capital gains are taxed: the engine then solves the year-end sale for
    # the tax on its own gains and passes the year's realized gains to `assess_annual_taxes`.
    taxes_capital_gains: bool

    def year_0_liquid_wealth(self, config: SimConfig) -> float:
        """Initial liquid wealth incl. pension lump sums paid out on Day 1, net of tax."""
        ...

    def next_bracket_factors(
        self, bracket_factors: np.ndarray, indexed_cpi_level: np.ndarray, inflation: np.ndarray, indexation: float
    ) -> np.ndarray:
        """Bracket factors after this year's inflation (no jurisdiction lowers its brackets in deflation).

        `indexed_cpi_level` is the cumulative CPI level weighted by `indexation` (incl. this year),
        `inflation` this year's CPI change.
        """
        ...

    def estimate_year_0_taxes_for_weights(self, config: SimConfig, init_wealth: float, weights: np.ndarray) -> float:
        """Year 0 total taxes and mandatory contributions for a given allocation."""
        ...

    def liquidate_pensions(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        current_age: int,
        bracket_factors: np.ndarray,
    ) -> PensionLiquidation:
        """Pays out the pension lump sums due at the start of the year (zeroing the balances in place)."""
        ...

    def latent_tax(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        from_age: int,
        bracket_factors: np.ndarray,
        unrealized_gains: np.ndarray | None = None,
    ) -> np.ndarray:
        """Tax still owed on current balances; subtracted for after-tax watermark comparisons."""
        ...

    def assess_annual_taxes(
        self,
        config: SimConfig,
        *,
        current_age: int,
        total_liquid_end: np.ndarray,
        current_expenses: np.ndarray,
        dividends: np.ndarray,
        interest: np.ndarray,
        annual_pension: np.ndarray,
        bracket_factors: np.ndarray,
        realized_gains: np.ndarray | None = None,
        inflation_factors: np.ndarray | None = None,
        us_dividends: np.ndarray | None = None,
    ) -> AnnualTaxes:
        """Year-end income/wealth taxes and mandatory contributions.

        `dividends` are all equity dividends; `us_dividends` is the part paid by US Stocks (subject
        to US withholding tax in CH, defaults to 0).
        """
        ...


def year_0_tax_assessment(
    jurisdiction: Jurisdiction, config: SimConfig, init_wealth: float, weights: np.ndarray
) -> AnnualTaxes:
    """Year 0 taxes for a given allocation, as a one-run `assess_annual_taxes` call.

    Assumes Year-0 values: dividends on the Year-0 equity holdings, interest at `config.cash_rate`,
    taxable wealth = liquid wealth minus the year's base expenses, no inflation or bracket indexation,
    no realized gains (the cost basis equals the market value on day 0), and the state pension only if
    retiring at or after the pension age.
    """
    one = np.ones(1)
    us_dividends = max(0.0, init_wealth * float(weights[0])) * config.dividend_yield
    dividends = max(0.0, init_wealth * float(weights[0] + weights[1])) * config.dividend_yield
    interest = max(0.0, init_wealth * float(weights[2])) * max(0.0, config.cash_rate)
    annual_pension = (
        jurisdiction.defaults.pension_payments_per_year * config.monthly_ahv_pension
        if config.start_age >= STATE_PENSION_AGE
        else 0.0
    )
    return jurisdiction.assess_annual_taxes(
        config,
        current_age=config.start_age,
        total_liquid_end=np.array([init_wealth], dtype=float),
        current_expenses=np.array([config.annual_base_expenses], dtype=float),
        dividends=np.array([dividends]),
        interest=np.array([interest]),
        annual_pension=np.array([annual_pension], dtype=float),
        bracket_factors=one,
        realized_gains=np.zeros(1),
        inflation_factors=one,
        us_dividends=np.array([us_dividends]),
    )


class Switzerland:
    """Swiss Federal + Canton Zurich rules (2026 tariffs, see `tax_engine`)."""

    defaults = COUNTRY_DEFAULTS["CH"]
    taxes_capital_gains = False  # private capital gains are tax-free in Switzerland

    def next_bracket_factors(
        self, bracket_factors: np.ndarray, indexed_cpi_level: np.ndarray, inflation: np.ndarray, indexation: float
    ) -> np.ndarray:
        return next_bracket_factors_high_water_mark(bracket_factors, indexed_cpi_level, inflation, indexation)

    def year_0_liquid_wealth(self, config: SimConfig) -> float:
        """
        Computes Year 0 initial liquid wealth including any immediate Month 0 pension liquidations
        (all Pillar 2 + Pillar 3a at age >= 65, or the first eligible Pillar 3a account at ages 60-64)
        net of immediate Capital Withdrawal Tax.
        """
        init_wealth = config.initial_liquid_wealth
        total_pension_liq = 0.0

        if config.start_age >= STATE_PENSION_AGE:
            p2 = config.initial_pillar_2
            p3a = sum(config.initial_pillar_3a_accounts) if config.initial_pillar_3a_accounts else 0.0
            total_pension_liq = p2 + p3a
        elif 60 <= config.start_age < STATE_PENSION_AGE and config.initial_pillar_3a_accounts:
            for i, acc in enumerate(config.initial_pillar_3a_accounts):
                if config.start_age >= 60 + i and acc > 0.0:
                    total_pension_liq = acc
                    break

        if total_pension_liq > 0.0:
            cap_tax = float(
                calculate_capital_withdrawal_tax(total_pension_liq, config.cantonal_multiplier, config.municipal_multiplier)
            )
            init_wealth += max(0.0, total_pension_liq - cap_tax)

        return init_wealth

    def estimate_year_0_taxes_for_weights(self, config: SimConfig, init_wealth: float, weights: np.ndarray) -> float:
        return float(year_0_tax_assessment(self, config, init_wealth, weights).total[0])

    def liquidate_pensions(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        current_age: int,
        bracket_factors: np.ndarray,
    ) -> PensionLiquidation:
        """
        Before 65, liquidates at most one Pillar 3a account per year (account `i` at age `60 + i`);
        from 65, all remaining Pillar 3a accounts plus Pillar 2. The capital withdrawal tax on the
        year's total is computed at start-of-year `bracket_factors` and deducted at source.
        """
        num_runs = pillar_2.shape[0]
        gross = np.zeros(num_runs)
        tax = np.zeros(num_runs)

        if current_age < STATE_PENSION_AGE:
            has_liquidated = np.zeros(num_runs, dtype=bool)
            for i, p3a in enumerate(pillar_3a):
                liquidation_age = 60 + i
                mask = (current_age >= liquidation_age) & (p3a > 0) & (~has_liquidated)
                gross[mask] += p3a[mask]
                p3a[mask] = 0
                has_liquidated[mask] = True
        else:
            # Age >= 65: Liquidate ALL remaining accounts
            for p3a in pillar_3a:
                mask = p3a > 0
                gross[mask] += p3a[mask]
                p3a[mask] = 0

        mask_p2 = (current_age >= STATE_PENSION_AGE) & (pillar_2 > 0)
        gross[mask_p2] += pillar_2[mask_p2]
        pillar_2[mask_p2] = 0

        has_liquidation = gross > 0
        if np.any(has_liquidation):
            tax[has_liquidation] = indexed_tax(
                calculate_capital_withdrawal_tax,
                gross[has_liquidation],
                bracket_factors[has_liquidation],
                config.cantonal_multiplier,
                config.municipal_multiplier,
            )
        return PensionLiquidation(gross=gross, tax=tax)

    def latent_tax(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        from_age: int,
        bracket_factors: np.ndarray,
        unrealized_gains: np.ndarray | None = None,
    ) -> np.ndarray:
        """Capital withdrawal tax still owed on the current pension balances, per run.

        Mirrors the liquidation schedule of `liquidate_pensions` from `from_age` onwards: before
        65, at most one Pillar 3a account per year (the first non-empty account whose
        liquidation age `60 + i` has been reached); at 65, all remaining 3a accounts together
        with Pillar 2 in one taxable event. Balances are taken as they are today (no growth)
        and taxed at today's bracket indexation. Pension balances are pre-tax claims, so this
        is subtracted wherever net worth drives a *decision* (watermark comparisons).
        """
        tax = np.zeros_like(pillar_2, dtype=float)
        remaining = [i for i, acc in enumerate(pillar_3a) if np.any(acc > 0)]
        if not remaining and not np.any(pillar_2 > 0):
            return tax
        events = []
        for age in range(from_age, STATE_PENSION_AGE):
            idx = next((i for i in remaining if 60 + i <= age), None)
            if idx is not None:
                events.append(pillar_3a[idx])
                remaining.remove(idx)
        events.append(pillar_2 + sum((pillar_3a[i] for i in remaining), np.zeros_like(pillar_2)))
        for amount in events:
            mask = amount > 0
            if np.any(mask):
                tax[mask] += indexed_tax(
                    calculate_capital_withdrawal_tax,
                    amount[mask],
                    bracket_factors[mask],
                    config.cantonal_multiplier,
                    config.municipal_multiplier,
                )
        return tax

    def assess_annual_taxes(
        self,
        config: SimConfig,
        *,
        current_age: int,
        total_liquid_end: np.ndarray,
        current_expenses: np.ndarray,
        dividends: np.ndarray,
        interest: np.ndarray,
        annual_pension: np.ndarray,
        bracket_factors: np.ndarray,
        realized_gains: np.ndarray | None = None,
        inflation_factors: np.ndarray | None = None,
        us_dividends: np.ndarray | None = None,
    ) -> AnnualTaxes:
        """Zurich wealth tax, AHV non-worker contributions (< 65), Federal + Zurich income tax and the
        US withholding tax on `us_dividends` net of the DA-1 foreign tax credit.

        US withholding (`config.us_dividend_withholding_rate`, 15% under the treaty with a W-8BEN):
        - `config.us_withholding_credit=True` (US-domiciled holdings): the dividends are declared gross
          and the withholding is credited against the Swiss income tax up to the Swiss tax on those
          dividends (`calculate_foreign_tax_credit`), but only if it is at least CHF 100; below that
          the dividends are declared net of the withholding and nothing is credited.
        - `False` (e.g. an Ireland-domiciled ETF): the fund loses the withholding, the investor is
          taxed on the net distribution and has nothing to credit.

        Components: `income_tax` (after the credit), `wealth_tax`, `ahv_contributions` and
        `us_withholding_tax` (the gross amount withheld); details: `foreign_tax_credit` and
        `taxable_income`. `realized_gains` and `inflation_factors` are ignored (no capital gains tax).
        """
        num_runs = total_liquid_end.shape[0]
        # Wealth Tax and AHV non-worker tax are assessed on wealth *after* deducting living expenses
        taxable_wealth = np.maximum(0, total_liquid_end - current_expenses)

        wealth_tax = indexed_tax(
            calculate_wealth_tax, taxable_wealth, bracket_factors, config.cantonal_multiplier, config.municipal_multiplier
        )

        ahv_mask = current_age < STATE_PENSION_AGE
        ahv_contrib = np.zeros(num_runs)
        ahv_contrib[ahv_mask] = indexed_tax(calculate_ahv_non_worker, taxable_wealth[ahv_mask], bracket_factors[ahv_mask])

        us_divs = np.zeros(num_runs) if us_dividends is None else np.maximum(0.0, us_dividends)
        withholding = config.us_dividend_withholding_rate * us_divs
        # The CHF 100 minimum is a fixed nominal amount (not indexed).
        claims_credit = config.us_withholding_credit & (withholding >= FOREIGN_TAX_CREDIT_MIN_CLAIM)
        declared_dividends = np.where(claims_credit, dividends, dividends - withholding)

        # Mandatory AHV/IV/EO non-worker contributions are deductible from taxable income
        # (Art. 33 Abs. 1 lit. d DBG / § 31 Abs. 1 lit. d StG ZH).
        taxable_income = np.maximum(0.0, declared_dividends + interest + annual_pension - ahv_contrib)
        income_tax_before_credit = indexed_tax(
            calculate_income_tax, taxable_income, bracket_factors, config.cantonal_multiplier, config.municipal_multiplier
        )
        credit = np.where(
            claims_credit,
            calculate_foreign_tax_credit(withholding, us_divs, income_tax_before_credit, taxable_income),
            0.0,
        )
        income_tax = income_tax_before_credit - credit

        components = {"income_tax": income_tax, "wealth_tax": wealth_tax, "ahv_contributions": ahv_contrib}
        total = income_tax + wealth_tax + ahv_contrib
        if config.us_dividend_withholding_rate > 0.0:
            components["us_withholding_tax"] = withholding
            total = total + withholding
        return AnnualTaxes(
            total=total,
            components=components,
            details={"foreign_tax_credit": credit, "taxable_income": taxable_income},
        )


class Austria:
    """Austrian rules (2026, see `tax_engine_at`): KESt with Regelbesteuerungsoption, pension
    taxation, mandatory § 16 ASVG health self-insurance; no wealth tax, no pension lump sums."""

    defaults = COUNTRY_DEFAULTS["AT"]
    taxes_capital_gains = True

    def next_bracket_factors(
        self, bracket_factors: np.ndarray, indexed_cpi_level: np.ndarray, inflation: np.ndarray, indexation: float
    ) -> np.ndarray:
        return next_bracket_factors_skip_deflation(bracket_factors, indexed_cpi_level, inflation, indexation)

    @staticmethod
    def _health_self_insurance_due(config: SimConfig, age: int) -> bool:
        # Required until the state pension starts (pensioners are insured through the pension,
        # which carries the 6% contribution); for the whole horizon if there is no pension.
        return age < STATE_PENSION_AGE or config.monthly_ahv_pension == 0.0

    @staticmethod
    def health_self_insurance_annual(config: SimConfig, age: int, inflation_factors: np.ndarray) -> np.ndarray:
        """Annual § 16 ASVG contribution in the simulation year lived at `age` (0 once the pension starts).

        Follows wages, not prices: CPI (`inflation_factors`, the engine's usual convention) times the
        real growth `config.health_insurance_real_growth` compounded per year since the start, so
        Year 0 pays today's rate in real terms.
        """
        if not Austria._health_self_insurance_due(config, age):
            return np.zeros_like(inflation_factors, dtype=float)
        real_growth = (1.0 + config.health_insurance_real_growth) ** (age - config.start_age)
        return 12.0 * config.health_insurance_monthly * inflation_factors * real_growth

    def year_0_liquid_wealth(self, config: SimConfig) -> float:
        return config.initial_liquid_wealth

    def estimate_year_0_taxes_for_weights(self, config: SimConfig, init_wealth: float, weights: np.ndarray) -> float:
        """KESt on Year-0 dividends and interest, pension taxes and health insurance.

        Assumes the Year-0 outflow sale realizes no gain: the cost basis equals the market value on
        day 0 (PRD decision 5) and the Year-0 price moves before the year-end sale are not known at
        estimation time. The simulation itself taxes whatever gain that sale actually realizes.
        """
        return float(year_0_tax_assessment(self, config, init_wealth, weights).total[0])

    def liquidate_pensions(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        current_age: int,
        bracket_factors: np.ndarray,
    ) -> PensionLiquidation:
        """No pension lump sums in the Austrian model (occupational pensions are out of scope)."""
        num_runs = pillar_2.shape[0]
        return PensionLiquidation(gross=np.zeros(num_runs), tax=np.zeros(num_runs))

    def latent_tax(
        self,
        config: SimConfig,
        pillar_2: np.ndarray,
        pillar_3a: list[np.ndarray],
        from_age: int,
        bracket_factors: np.ndarray,
        unrealized_gains: np.ndarray | None = None,
    ) -> np.ndarray:
        """KESt owed on full liquidation: 27.5% of net unassessed gains (already-realized + unrealized; losses offset)."""
        if unrealized_gains is None:
            return np.zeros_like(pillar_2, dtype=float)
        return KEST_RATE * np.maximum(0.0, unrealized_gains)

    def assess_annual_taxes(
        self,
        config: SimConfig,
        *,
        current_age: int,
        total_liquid_end: np.ndarray,
        current_expenses: np.ndarray,
        dividends: np.ndarray,
        interest: np.ndarray,
        annual_pension: np.ndarray,
        bracket_factors: np.ndarray,
        realized_gains: np.ndarray | None = None,
        inflation_factors: np.ndarray | None = None,
        us_dividends: np.ndarray | None = None,
    ) -> AnnualTaxes:
        """Income tax (pension; all capital income if the Regelbesteuerungsoption is cheaper),
        KESt and health insurance (6% on the pension, plus § 16 ASVG self-insurance before the
        pension starts). Bracket edges, the pensioner credit and the EUR 620 allowance are indexed
        with `bracket_factors` via positive homogeneity; the self-insurance contribution with CPI
        plus real wage growth (see `health_self_insurance_annual`).

        `us_dividends` is ignored: the 15% US withholding tax is assumed fully credited against the
        27.5% KESt (PRD §8.2), so it does not change the total.
        """
        gains = np.zeros(total_liquid_end.shape[0]) if realized_gains is None else realized_gains
        inflation = np.ones(total_liquid_end.shape[0]) if inflation_factors is None else inflation_factors
        f = bracket_factors
        taxes = calculate_annual_taxes_at(annual_pension / f, dividends / f, gains / f, interest / f)
        income_tax = f * taxes["income_tax"]
        kest = f * taxes["kest"]
        health = f * taxes["pension_health_insurance"] + self.health_self_insurance_annual(config, current_age, inflation)
        return AnnualTaxes(
            total=income_tax + kest + health,
            components={"income_tax": income_tax, "kest": kest, "health_insurance": health},
        )


_JURISDICTIONS: dict[str, Jurisdiction] = {"CH": Switzerland(), "AT": Austria()}


def get_jurisdiction(country: str) -> Jurisdiction:
    """Returns the rules for `country` (an ISO 3166-1 alpha-2 code in `SUPPORTED_COUNTRIES`)."""
    if country not in _JURISDICTIONS:
        raise ValueError(unsupported_country_message(country))
    return _JURISDICTIONS[country]


def unsupported_country_message(country: str) -> str:
    planned = " It is specified in the docs (PRD §8) but not implemented yet." if country in PLANNED_COUNTRIES else ""
    return f"Unsupported country {country!r}. Supported: {list(SUPPORTED_COUNTRIES)}.{planned}"
