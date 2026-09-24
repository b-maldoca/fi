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
