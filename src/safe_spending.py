"""Highest safe spending: the largest Annual Base Expenses that still meets a target success rate.

The search reruns `run_simulation` on the *same* return and inflation paths for each candidate
spending level (common random numbers), so the success rate is a deterministic function of the
spending level. It is assumed to be non-increasing in spending, which holds for all spending
strategies in practice (more spending never leaves more money): an exponential search brackets the
answer around the current expenses, then a bisection narrows it to a multiple of `tolerance`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .metrics import compute_success_mask
from .simulation_engine import SimConfig, estimate_year_0_taxes, run_simulation

# Default success rate the highest safe spending must reach.
DEFAULT_SAFE_SPENDING_SUCCESS_PCT = 95.0
# Resolution of the search: the result is a multiple of this amount (today's money per year).
SAFE_SPENDING_TOLERANCE = 1_000.0
# Growth factor of the exponential search that brackets the answer around the current expenses.
_BRACKET_GROWTH = 1.5


@dataclass(frozen=True)
class SafeSpendingResult:
    """Outcome of `find_highest_safe_spending`."""

    # Highest Annual Base Expenses (today's money, excluding taxes) whose success rate is at least
    # the target; 0 if even zero spending misses it (`reachable` is then False).
    expenses: float
    # Success rate (%) at `expenses`.
    success_rate_pct: float
    # Estimated Year-0 taxes at `expenses` (expenses + these = the Year-0 total outflow).
    year_0_taxes: float
    # False if the target is missed even with zero spending (taxes alone deplete the portfolio or
    # the ending-wealth target is out of reach).
    reachable: bool
    # True if the target is still met at the upper search bound (spending the whole starting net
    # worth in one year), i.e. the true answer is at least `expenses`.
    capped: bool
    # Number of simulations run.
    evaluations: int


def initial_net_worth(config: SimConfig) -> float:
    """Starting net worth: taxable liquid wealth plus Pillar 2 and all Pillar 3a balances."""
    return config.initial_liquid_wealth + config.initial_pillar_2 + sum(config.initial_pillar_3a_accounts)


def config_with_expenses(config: SimConfig, expenses: float) -> SimConfig:
    """`config` with Annual Base Expenses set to `expenses`.

    Under Vanguard Dynamic spending the target rate moves with the expenses: it is scaled in
    proportion to them (or set to `expenses / starting net worth` when the base expenses are 0),
    matching the app, which derives the rate from the expenses.
    """
    rate = config.vanguard_target_rate
    if config.spending_strategy == "Vanguard Dynamic":
        if config.annual_base_expenses > 0.0:
            rate = config.vanguard_target_rate * expenses / config.annual_base_expenses
        else:
            start_nw = initial_net_worth(config)
            rate = expenses / start_nw if start_nw > 0.0 else 0.0
    return replace(config, annual_base_expenses=float(expenses), vanguard_target_rate=rate)


def success_rate_pct(
    config: SimConfig, return_matrix: np.ndarray, inflation_matrix: np.ndarray, success_pct: float
) -> float:
    """Share of runs (%) meeting the success criterion (`compute_success_mask`) under `config`."""
    history = run_simulation(config, return_matrix, inflation_matrix)
    mask = compute_success_mask(
        history["net_worth"][-1, :],
        history["initial_net_worth"],
        inflation_matrix,
        success_pct,
        net_worth_history=history["net_worth"],
    )
    return float(np.mean(mask) * 100.0)


def find_highest_safe_spending(
    config: SimConfig,
    return_matrix: np.ndarray,
    inflation_matrix: np.ndarray,
    success_pct: float,
    target_success_pct: float = DEFAULT_SAFE_SPENDING_SUCCESS_PCT,
    tolerance: float = SAFE_SPENDING_TOLERANCE,
) -> SafeSpendingResult:
    """Highest Annual Base Expenses (a multiple of `tolerance`) with success rate >= `target_success_pct`.

    `success_pct` is the ending-wealth criterion of `compute_success_mask` (% of inflation-adjusted
    starting net worth). Every other setting of `config` (allocation, spending strategy, pensions,
    taxes) is kept; taxes are on top of the expenses, as everywhere in the simulator.
    """
    if not 0.0 < target_success_pct <= 100.0:
        raise ValueError(f"target_success_pct ({target_success_pct}) must be in (0, 100].")
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f"tolerance ({tolerance}) must be positive and finite.")

    rates: dict[int, float] = {}

    def rate_at(k: int) -> float:
        if k not in rates:
            rates[k] = success_rate_pct(
                config_with_expenses(config, k * tolerance), return_matrix, inflation_matrix, success_pct
            )
        return rates[k]

    def meets(k: int) -> bool:
        return rate_at(k) >= target_success_pct

    # Spending the whole starting net worth in a single year can never be safe; it bounds the search.
    k_cap = max(1, math.ceil(initial_net_worth(config) / tolerance))
    k_start = min(k_cap, max(1, round(config.annual_base_expenses / tolerance)))
    # Bracket the answer: meets(lo) (or lo == 0, checked last) and not meets(hi).
    if meets(k_start):
        lo, hi = k_start, min(k_cap, math.ceil(k_start * _BRACKET_GROWTH) + 1)
        while hi > lo and meets(hi):
            lo = hi
            hi = min(k_cap, math.ceil(hi * _BRACKET_GROWTH) + 1)
    else:
        hi, lo = k_start, math.floor(k_start / _BRACKET_GROWTH)
        while lo > 0 and not meets(lo):
            hi, lo = lo, math.floor(lo / _BRACKET_GROWTH)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if meets(mid):
            lo = mid
        else:
            hi = mid

    # The target still holds at the upper search bound: the true answer is at least `lo`.
    capped = lo == k_cap and meets(lo)
    reachable = meets(lo) if lo == 0 else True
    best = config_with_expenses(config, lo * tolerance)
    return SafeSpendingResult(
        expenses=lo * tolerance,
        success_rate_pct=rate_at(lo),
        year_0_taxes=estimate_year_0_taxes(best),
        reachable=reachable,
        capped=capped,
        evaluations=len(rates),
    )
