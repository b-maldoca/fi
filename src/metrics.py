"""Pure post-processing of `run_simulation` output used by the Streamlit UI.

Kept separate from `app.py` so the success criterion, the TL;DR classification,
and the withdrawal-rate reconstruction are defined once and unit-tested, rather
than re-implemented inline in several UI helpers.
"""
import numpy as np

# TL;DR "RICH" threshold: median real ending net worth >= 3x the starting net worth.
RICH_MULTIPLE = 3.0

# Same deflation clamp as `run_simulation`, so reported real values are consistent
# with the inflation factors the engine actually applied.
_MIN_INFLATION_STEP = 0.01


def cumulative_inflation(inflation_matrix: np.ndarray) -> np.ndarray:
    """Cumulative price level per run and year, shape (num_runs, duration_years)."""
    return np.cumprod(np.maximum(_MIN_INFLATION_STEP, 1.0 + np.asarray(inflation_matrix, dtype=float)), axis=1)


def compute_success_mask(
    final_net_worth: np.ndarray,
    initial_net_worth: float,
    inflation_matrix: np.ndarray,
    success_pct: float,
    net_worth_history: np.ndarray | None = None,
) -> np.ndarray:
    """Runs that never depleted (`NW > 0` every year) and whose final NW exceeds `success_pct`% of inflation-adjusted start.

    Each run is compared against its *own* realized price level, so a run with high
    inflation needs a higher nominal ending balance. `success_pct == 0` means
    "never hit NW <= 0 and ending net worth > 0", i.e. plain survival without ruin.
    `final_net_worth` may be passed as 1-D `(num_runs,)` (with optional 2-D `net_worth_history`)
    or directly as 2-D `(duration_years, num_runs)`.
    """
    arr = np.asarray(final_net_worth, dtype=float)
    if arr.ndim == 2:
        nw_hist = arr
        final_nw = arr[-1, :]
    else:
        final_nw = arr
        nw_hist = np.asarray(net_worth_history, dtype=float) if net_worth_history is not None else None

    never_depleted = np.all(nw_hist > 0.0, axis=0) if nw_hist is not None else (final_nw > 0.0)
    final_price_level = cumulative_inflation(inflation_matrix)[:, -1]
    target = (success_pct / 100.0) * initial_net_worth * final_price_level
    return never_depleted & (final_nw > target)


def real_final_net_worth(final_net_worth: np.ndarray, inflation_matrix: np.ndarray) -> np.ndarray:
    """Final net worth of each run deflated by that run's own cumulative inflation."""
    return np.asarray(final_net_worth) / cumulative_inflation(inflation_matrix)[:, -1]


def classify_outcome(
    final_net_worth: np.ndarray,
    initial_net_worth: float,
    inflation_matrix: np.ndarray,
    success_pct: float = 0.0,
) -> str:
    """TL;DR label for the median run: 'BROKE', 'SHORTFALL', 'DEAD', or 'RICH'.

    - BROKE: median real ending net worth <= 0.
    - SHORTFALL: median real ending net worth is > 0 but <= (success_pct / 100) * initial_net_worth
      (does not hit the target ending wealth criterion).
    - RICH: median real ending net worth > target and >= RICH_MULTIPLE * initial_net_worth.
    - DEAD: median real ending net worth > target and < RICH_MULTIPLE * initial_net_worth.
    Uses per-run real values rather than mixing a nominal median with a median
    price level, which are generally not from the same run.
    """
    med_real = float(np.median(real_final_net_worth(final_net_worth, inflation_matrix)))
    if med_real <= 0:
        return "BROKE"
    target_real = (success_pct / 100.0) * initial_net_worth
    if med_real <= target_real:
        return "SHORTFALL"
    if med_real >= RICH_MULTIPLE * initial_net_worth:
        return "RICH"
    return "DEAD"


def beginning_of_year_withdrawal_rate(history: dict) -> np.ndarray:
    """Net annual portfolio withdrawal (`max(0, expenses + taxes - AHV)`) as % of beginning-of-year net worth.

    Uses `initial_net_worth` for Year 0 (falling back to `net_worth[0] + net_outflows[0]`
    if omitted) and `net_worth[t - 1]` for `t >= 1`. Denominators are floored at 1.0 so
    depleted portfolios do not divide by zero.
    """
    nw = np.asarray(history['net_worth'], dtype=float)
    ahv = np.asarray(history.get('income_ahv', 0.0), dtype=float)
    net_outflows = np.maximum(0.0, np.asarray(history['expenses_paid'], dtype=float) + np.asarray(history['taxes_paid'], dtype=float) - ahv)

    if 'initial_net_worth' in history:
        y0_nw = np.full((1, nw.shape[1]), float(history['initial_net_worth']), dtype=float)
    else:
        y0_nw = nw[0:1, :] + net_outflows[0:1, :]

    if nw.shape[0] > 1:
        boy_nw = np.vstack([y0_nw, nw[:-1, :]])
    else:
        boy_nw = y0_nw

    start_nw = np.maximum(boy_nw, 1.0)
    return net_outflows / start_nw * 100.0


def first_depletion_year(wealth_history: np.ndarray) -> np.ndarray:
    """First 1-based simulation year in which each run's wealth drops to <= 0, or NaN if never.

    Expects `wealth_history` of shape `(duration_years, num_runs)` (e.g. `history['net_worth']`
    or `history['liquid_assets']`).
    """
    w = np.asarray(wealth_history, dtype=float)
    depleted = w <= 0.0
    ever_depleted = np.any(depleted, axis=0)
    first_idx = np.argmax(depleted, axis=0)
    return np.where(ever_depleted, first_idx + 1.0, np.nan)


def first_shortfall_year(
    net_worth: np.ndarray,
    initial_net_worth: float,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> np.ndarray:
    """First 1-based simulation year in which each run's net worth drops to <= target, or NaN if never.

    The target at year `t` is `(success_pct / 100) * initial_net_worth * PriceLevel_{t}`.
    """
    nw = np.asarray(net_worth, dtype=float)
    cum_inf = cumulative_inflation(inflation_matrix).T
    target_trajectory = (success_pct / 100.0) * float(initial_net_worth) * cum_inf
    is_shortfall = nw <= target_trajectory
    ever_shortfall = np.any(is_shortfall, axis=0)
    first_idx = np.argmax(is_shortfall, axis=0)
    return np.where(ever_shortfall, first_idx + 1.0, np.nan)


def funded_outflows(history: dict) -> tuple[np.ndarray, np.ndarray]:
    """Living expenses and taxes actually paid each year, shape `(duration_years, num_runs)` each.

    `expenses_paid` / `taxes_paid` hold what was *due*; once a portfolio is depleted the unpaid part
    is recorded in `unfunded_deficit` (and becomes debt). The unpaid amount is attributed to living
    expenses first, then to taxes.
    """
    expenses = np.asarray(history['expenses_paid'], dtype=float)
    taxes = np.asarray(history.get('taxes_paid', np.zeros_like(expenses)), dtype=float)
    deficit = np.asarray(history.get('unfunded_deficit', np.zeros_like(expenses)), dtype=float)
    funded_expenses = np.maximum(0.0, expenses - deficit)
    funded_taxes = np.maximum(0.0, taxes - np.maximum(0.0, deficit - expenses))
    return funded_expenses, funded_taxes


def min_real_spending_pct(
    history: dict,
    inflation_matrix: np.ndarray,
    annual_base_expenses: float,
) -> np.ndarray:
    """Minimum annual real living expenses actually paid per run, as % of `annual_base_expenses`.

    Uses funded expenses (see `funded_outflows`), so a depleted run reports the spending it could
    really afford rather than its unpaid target.
    """
    expenses, _ = funded_outflows(history)
    num_runs = expenses.shape[1]
    if annual_base_expenses <= 0.0:
        return np.full(num_runs, 100.0)
    real_expenses = expenses / cumulative_inflation(inflation_matrix).T
    return (np.min(real_expenses, axis=0) / float(annual_base_expenses)) * 100.0


def compute_survival_curves(
    history: dict,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> dict:
    """Year-by-year survival and target-preservation rates across all runs, in % (0-100).

    - `solvent_pct`: % of runs that have never depleted total net worth (`min_{s <= t} NW_s > 0`).
    - `liquid_solvent_pct`: % of runs that have never depleted liquid assets (`min_{s <= t} Liquid_s > 0`).
    - `above_target_pct`: % of runs with `NW_{t, i} > (success_pct / 100) * NW_0 * PriceLevel_{t, i}` at year `t`.
    """
    net_worth = np.asarray(history['net_worth'], dtype=float)
    liquid_assets = np.asarray(history.get('liquid_assets', net_worth), dtype=float)
    initial_nw = float(history['initial_net_worth'])
    cum_inf = cumulative_inflation(inflation_matrix).T

    ever_solvent_nw = np.minimum.accumulate(net_worth, axis=0) > 0.0
    ever_solvent_liq = np.minimum.accumulate(liquid_assets, axis=0) > 0.0
    target_trajectory = (success_pct / 100.0) * initial_nw * cum_inf
    above_target = ever_solvent_nw & (net_worth > target_trajectory)

    return {
        "solvent_pct": np.mean(ever_solvent_nw, axis=1) * 100.0,
        "liquid_solvent_pct": np.mean(ever_solvent_liq, axis=1) * 100.0,
        "above_target_pct": np.mean(above_target, axis=1) * 100.0,
    }


def summarize_failure_timing(
    history: dict,
    inflation_matrix: np.ndarray,
    success_pct: float,
    start_age: int,
) -> dict:
    """Decomposes simulation failures by severity (depletion vs target shortfall) and timing."""
    net_worth = np.asarray(history['net_worth'], dtype=float)
    liquid_assets = np.asarray(history.get('liquid_assets', net_worth), dtype=float)
    final_nw = net_worth[-1, :]
    initial_nw = float(history['initial_net_worth'])

    success_mask = compute_success_mask(final_nw, initial_nw, inflation_matrix, success_pct, net_worth_history=net_worth)
    depletion_years = first_depletion_year(net_worth)
    liquid_depletion_years = first_depletion_year(liquid_assets)
    shortfall_years = first_shortfall_year(net_worth, initial_nw, inflation_matrix, success_pct)

    # 1-based year `yr` corresponds to retirement year `idx = yr - 1` lived at `start_age + yr - 1`.
    depletion_ages = start_age + depletion_years - 1.0
    liquid_depletion_ages = start_age + liquid_depletion_years - 1.0
    shortfall_ages = start_age + shortfall_years - 1.0

    depleted_mask = ~np.isnan(depletion_years)
    liquid_depleted_mask = ~np.isnan(liquid_depletion_years)
    ever_shortfall_mask = ~np.isnan(shortfall_years)
    pre_65_liquid_mask = liquid_depleted_mask & (liquid_depletion_ages < 65)
    shortfall_mask = (~success_mask) & (~depleted_mask)

    if np.any(depleted_mask):
        earliest_age = float(np.nanmin(depletion_ages))
        earliest_year = int(np.nanmin(depletion_years))
        median_age = float(np.nanmedian(depletion_ages))
        median_year = float(np.nanmedian(depletion_years))
    else:
        earliest_age = None
        earliest_year = None
        median_age = None
        median_year = None

    if np.any(ever_shortfall_mask):
        earliest_sf_age = float(np.nanmin(shortfall_ages))
        earliest_sf_year = int(np.nanmin(shortfall_years))
        median_sf_age = float(np.nanmedian(shortfall_ages))
        median_sf_year = float(np.nanmedian(shortfall_years))
    else:
        earliest_sf_age = None
        earliest_sf_year = None
        median_sf_age = None
        median_sf_year = None

    return {
        "depletion_years": depletion_years,
        "depletion_ages": depletion_ages,
        "liquid_depletion_years": liquid_depletion_years,
        "liquid_depletion_ages": liquid_depletion_ages,
        "shortfall_years": shortfall_years,
        "shortfall_ages": shortfall_ages,
        "depleted_mask": depleted_mask,
        "shortfall_mask": shortfall_mask,
        "ever_shortfall_mask": ever_shortfall_mask,
        "depletion_rate_pct": float(np.mean(depleted_mask) * 100.0),
        "liquid_depletion_rate_pct": float(np.mean(liquid_depleted_mask) * 100.0),
        "pre_65_depletion_rate_pct": float(np.mean(pre_65_liquid_mask) * 100.0),
        "shortfall_rate_pct": float(np.mean(shortfall_mask) * 100.0),
        "shortfall_occurrence_rate_pct": float(np.mean(ever_shortfall_mask) * 100.0),
        "earliest_depletion_age": earliest_age,
        "earliest_depletion_year": earliest_year,
        "median_depletion_age": median_age,
        "median_depletion_year": median_year,
        "earliest_shortfall_age": earliest_sf_age,
        "earliest_shortfall_year": earliest_sf_year,
        "median_shortfall_age": median_sf_age,
        "median_shortfall_year": median_sf_year,
    }


def format_depletion_label(
    run_idx: int,
    depletion_years: np.ndarray,
    liquid_depletion_years: np.ndarray,
    start_age: int,
) -> str:
    """Human-readable depletion age/year label for a single cohort or run."""
    dep_yr = depletion_years[run_idx]
    if not np.isnan(dep_yr):
        yr = int(dep_yr)
        return f"Age {start_age + yr - 1} (Yr {yr})"
    liq_yr = liquid_depletion_years[run_idx]
    if not np.isnan(liq_yr):
        yr = int(liq_yr)
        return f"Liq Age {start_age + yr - 1} (Yr {yr})"
    return "Survived"


def format_shortfall_label(
    run_idx: int,
    shortfall_years: np.ndarray,
    start_age: int,
) -> str:
    """Human-readable shortfall age/year label ('Age X (Yr Y)' or 'Never')."""
    sf_yr = shortfall_years[run_idx]
    if not np.isnan(sf_yr):
        yr = int(sf_yr)
        return f"Age {start_age + yr - 1} (Yr {yr})"
    return "Never"


def _shortfall_matrix(
    net_worth: np.ndarray,
    initial_net_worth: float,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> np.ndarray:
    """Boolean `(duration_years, num_runs)` mask of years with NW <= success target."""
    nw = np.asarray(net_worth, dtype=float)
    target = (success_pct / 100.0) * float(initial_net_worth) * cumulative_inflation(inflation_matrix).T
    return nw <= target


def years_in_shortfall(
    net_worth: np.ndarray,
    initial_net_worth: float,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> np.ndarray:
    """Number of simulation years each run spent at or below the success target (not necessarily contiguous)."""
    return np.sum(_shortfall_matrix(net_worth, initial_net_worth, inflation_matrix, success_pct), axis=0)


def min_real_net_worth(net_worth: np.ndarray, inflation_matrix: np.ndarray) -> np.ndarray:
    """Lowest real (deflated by each run's own price level) year-end net worth reached per run."""
    return np.min(np.asarray(net_worth, dtype=float) / cumulative_inflation(inflation_matrix).T, axis=0)


# Severity groups used by `worst_path_order`, most severe first.
OUTCOME_DEPLETED = "Depleted"
OUTCOME_FAILED = "Failed"
OUTCOME_RECOVERED = "Recovered"
OUTCOME_NEVER_BREACHED = "Never breached"


def classify_path_outcomes(
    history: dict,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> np.ndarray:
    """Severity label per run.

    - `Depleted`: total net worth reached <= 0 at some point.
    - `Failed`: never depleted, but final real net worth is at or below the success target.
    - `Recovered`: breached the success target during retirement but finished above it.
    - `Never breached`: stayed strictly above the success target every year.
    """
    net_worth = np.asarray(history['net_worth'], dtype=float)
    initial_nw = float(history['initial_net_worth'])
    depleted = ~np.isnan(first_depletion_year(net_worth))
    success = compute_success_mask(net_worth[-1, :], initial_nw, inflation_matrix, success_pct, net_worth_history=net_worth)
    ever_shortfall = np.any(_shortfall_matrix(net_worth, initial_nw, inflation_matrix, success_pct), axis=0)
    return np.select(
        [depleted, ~success, ever_shortfall],
        [OUTCOME_DEPLETED, OUTCOME_FAILED, OUTCOME_RECOVERED],
        default=OUTCOME_NEVER_BREACHED,
    )


def worst_path_order(
    history: dict,
    inflation_matrix: np.ndarray,
    success_pct: float,
) -> np.ndarray:
    """Run indices ordered from worst to best by outcome severity.

    1. Depleted: earliest depletion year first.
    2. Failed: lowest final real net worth first.
    3. Recovered: most years spent in shortfall first.
    4. Never breached: lowest final real net worth first.

    Remaining ties are broken by lowest final real net worth.
    """
    net_worth = np.asarray(history['net_worth'], dtype=float)
    initial_nw = float(history['initial_net_worth'])
    outcomes = classify_path_outcomes(history, inflation_matrix, success_pct)
    group = np.select(
        [outcomes == OUTCOME_DEPLETED, outcomes == OUTCOME_FAILED, outcomes == OUTCOME_RECOVERED],
        [0, 1, 2],
        default=3,
    )
    final_real = real_final_net_worth(net_worth[-1, :], inflation_matrix)
    depletion_year = first_depletion_year(net_worth)
    yrs_sf = years_in_shortfall(net_worth, initial_nw, inflation_matrix, success_pct)
    within_group = np.select(
        [group == 0, group == 1, group == 2],
        [np.nan_to_num(depletion_year, nan=np.inf), final_real, -yrs_sf.astype(float)],
        default=final_real,
    )
    return np.lexsort((final_real, within_group, group))
