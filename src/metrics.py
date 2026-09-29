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
) -> np.ndarray:
    """Runs whose final net worth exceeds `success_pct`% of the inflation-adjusted start.

    Each run is compared against its *own* realized price level, so a run with high
    inflation needs a higher nominal ending balance. `success_pct == 0` means
    "ending net worth > 0", i.e. plain survival.
    """
    final_price_level = cumulative_inflation(inflation_matrix)[:, -1]
    target = (success_pct / 100.0) * initial_net_worth * final_price_level
    return np.asarray(final_net_worth) > target


def real_final_net_worth(final_net_worth: np.ndarray, inflation_matrix: np.ndarray) -> np.ndarray:
    """Final net worth of each run deflated by that run's own cumulative inflation."""
    return np.asarray(final_net_worth) / cumulative_inflation(inflation_matrix)[:, -1]


def classify_outcome(final_net_worth: np.ndarray, initial_net_worth: float, inflation_matrix: np.ndarray) -> str:
    """TL;DR label for the median run: 'BROKE', 'RICH', or 'DEAD'.

    BROKE: median ending net worth <= 0. RICH: median *real* ending net worth
    >= RICH_MULTIPLE x starting net worth. DEAD: anything in between (you die
    before the money runs out, without growing real wealth massively).
    Uses per-run real values rather than mixing a nominal median with a median
    price level, which are generally not from the same run.
    """
    if np.median(final_net_worth) <= 0:
        return "BROKE"
    if np.median(real_final_net_worth(final_net_worth, inflation_matrix)) >= RICH_MULTIPLE * initial_net_worth:
        return "RICH"
    return "DEAD"


def beginning_of_year_withdrawal_rate(history: dict) -> np.ndarray:
    """Annual outflow (expenses + taxes) as % of the reconstructed beginning-of-year net worth.

    Beginning-of-year net worth is approximated as end-of-year net worth plus that
    year's outflows (ignoring the year's market return). Denominators are floored
    at 1 CHF so depleted portfolios do not divide by zero.
    """
    outflows = history['expenses_paid'] + history['taxes_paid']
    start_nw = np.maximum(history['net_worth'] + outflows, 1.0)
    return outflows / start_nw * 100.0


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


def min_real_spending_pct(
    history: dict,
    inflation_matrix: np.ndarray,
    annual_base_expenses: float,
) -> np.ndarray:
    """Minimum annual real living expenses per run as % of `annual_base_expenses`."""
    expenses = np.asarray(history['expenses_paid'], dtype=float)
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
    above_target = net_worth > target_trajectory

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

    success_mask = compute_success_mask(final_nw, initial_nw, inflation_matrix, success_pct)
    depletion_years = first_depletion_year(net_worth)
    liquid_depletion_years = first_depletion_year(liquid_assets)
    shortfall_years = first_shortfall_year(net_worth, initial_nw, inflation_matrix, success_pct)

    depletion_ages = start_age + depletion_years
    liquid_depletion_ages = start_age + liquid_depletion_years
    shortfall_ages = start_age + shortfall_years

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
        return f"Age {start_age + yr} (Yr {yr})"
    liq_yr = liquid_depletion_years[run_idx]
    if not np.isnan(liq_yr):
        yr = int(liq_yr)
        return f"Liq Age {start_age + yr} (Yr {yr})"
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
        return f"Age {start_age + yr} (Yr {yr})"
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
    success = compute_success_mask(net_worth[-1, :], initial_nw, inflation_matrix, success_pct)
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
