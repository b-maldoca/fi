"""End-to-end smoke tests for the Streamlit UI (`app.py`) via Streamlit's AppTest.

These catch wiring errors (bad imports, invalid SimConfig construction, widget
key collisions) that the engine unit tests cannot see.
"""
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_TIMEOUT_S = 120


def _run(setup=None):
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    if setup is not None:
        setup(at)
        at.run()
    return at


def test_app_runs_with_defaults():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    headers = [h.value for h in at.main.header]
    assert headers == ["Historic Backtesting", "Historic Bootstrapping", "Monte Carlo"]
    tldr_els = [m for m in at.markdown if m.value.startswith("### ")]
    tldr = [m.value for m in tldr_els]
    assert len(tldr) == 3 and all(any(k in t for k in ("BROKE", "SHORTFALL", "RICH", "DEAD")) for t in tldr)
    assert all(all(k in m.proto.help for k in ("RICH", "DEAD", "SHORTFALL", "BROKE")) for m in tldr_els)
    assert sum(1 for m in at.metric if m.label == "Probability of Success") == 3
    assert sum(1 for m in at.metric if m.label == "Depletion Rate (NW ≤ 0)") == 3
    assert sum(1 for m in at.metric if m.label == "Depletion Age (Min / Med)") == 3
    assert sum(1 for m in at.metric if m.label == "Shortfall Rate (Real NW ≤ Target)") == 3
    assert sum(1 for m in at.metric if m.label == "Shortfall Age (Min / Med)") == 3
    # Historic Backtesting column (column 0):
    # chart 0 = Net Worth Trajectory, chart 1 = Portfolio Survival by Age,
    # chart 2 = Income vs Required Cash, chart 3 = Annual Withdrawal Breakdown,
    # chart 4 = Withdrawal Rate, chart 5 = Asset Allocation Development.
    import json
    hist_charts = [json.loads(c.proto.spec) for c in at.columns[0].get("plotly_chart")]
    assert len(hist_charts) == 6
    nw_names = [t["name"] for t in hist_charts[0]["data"] if t.get("showlegend") is not False]
    surv_names = [t["name"] for t in hist_charts[1]["data"]]
    inc_hist_names = [t["name"] for t in hist_charts[2]["data"]]
    wb_hist_names = [t["name"] for t in hist_charts[3]["data"]]
    all_charts = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    assert len(all_charts) == 18
    wb_boot_names = [t["name"] for t in all_charts[9]["data"]]
    wb_mc_names = [t["name"] for t in all_charts[15]["data"]]
    wr_traces = [t for t in hist_charts[4]["data"]]
    assert nw_names[:5] == ["Worst Cohort (Min)", "25th Pct", "50th Pct", "75th Pct", "Best Cohort (Max)"]
    assert surv_names == ["Solvent (NW > 0)", "Liquid Solvent (Liquid > 0)", "Above Target (50% Start NW)"]
    assert inc_hist_names == ["Dividends", "AHV Pension", "Capital Sold", "Inflation-Adj Start"]
    assert wb_hist_names == [
        "Living Expenses (50th Pct)",
        "Taxes Paid (50th Pct)",
        "Living Expenses (Worst Cohort (Min))",
        "Taxes Paid (Worst Cohort (Min))",
        "Inflation-Adj Start",
    ]
    assert wb_boot_names == wb_mc_names == [
        "Living Expenses (50th Pct)",
        "Taxes Paid (50th Pct)",
        "Living Expenses (5th Pct)",
        "Taxes Paid (5th Pct)",
        "Inflation-Adj Start",
    ]
    assert [t["name"] for t in wr_traces] == ["Best Cohort (Min)", "25th Pct", "50th Pct", "75th Pct", "Worst Cohort (Max)"]
    assert wr_traces[0]["line"]["color"] == "purple"
    assert wr_traces[-1]["line"]["color"] == "crimson"
    alloc_names = [t["name"] for t in hist_charts[5]["data"]]
    assert alloc_names == ["CHF Cash", "US Stocks", "Non-US Stocks", "Gold", "Bitcoin", "Pillar 2 & 3a"]
    for df in at.dataframe:
        assert "Depletion Age" in df.value.columns
        assert "Shortfall Age" in df.value.columns
        assert "Min Real Exp" not in df.value.columns
        assert bool(df.proto.columns)
    # Tables alternate Best / Worst per column. Best: Final NW (Real) descending.
    # Worst: Outcome severity groups in order, each group sorted by its own key.
    severity = ["Depleted", "Failed", "Recovered", "Never breached"]
    for i, df in enumerate(at.dataframe):
        v = df.value
        assert {"Outcome", "Yrs in Shortfall", "Min NW (Real)"} <= set(v.columns)
        assert "Min NW (Nom)" not in v.columns
        if i % 2 == 0:
            assert v["Final NW (Real)"].is_monotonic_decreasing
            continue
        groups = [severity.index(o) for o in v["Outcome"]]
        assert groups == sorted(groups)
        recovered = v[v["Outcome"] == "Recovered"]
        assert recovered["Yrs in Shortfall"].is_monotonic_decreasing



@pytest.mark.parametrize("spending", ["Static", "Dynamic (Floor & Ceiling)"])
def test_app_runs_with_other_spending_strategies(spending):
    def setup(at):
        next(s for s in at.selectbox if s.label == "Spending Model").set_value(spending)
    at = _run(setup)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_rejects_allocation_not_summing_to_100():
    def setup(at):
        next(n for n in at.number_input if n.label == "US Stocks").set_value(10.0)
    at = _run(setup)
    assert not at.exception
    assert any("must sum to exactly 100%" in e.value for e in at.error)
    assert not at.main.header  # st.stop() before any results are rendered


def test_ppp_slider_help_matches_data_constant():
    from src.historic_returns import HISTORIC_REAL_CHF_APPRECIATION
    at = _run()
    slider = next(s for s in at.slider if s.label.startswith("Real CHF Appreciation"))
    assert f"+{HISTORIC_REAL_CHF_APPRECIATION * 100:.2f}%" in slider.help
    # The advertised historic value must be selectable on the slider grid.
    steps = (HISTORIC_REAL_CHF_APPRECIATION * 100 - slider.min) / slider.step
    assert abs(steps - round(steps)) < 1e-9


def test_monte_carlo_defaults_are_calibrated_to_history():
    import numpy as np
    from src.historic_returns import (
        HISTORIC_RETURNS_GOLD_CHF,
        HISTORIC_RETURNS_NON_US_CHF,
        HISTORIC_RETURNS_US_CHF,
        HISTORIC_SWISS_INFLATION,
        ppp_adjusted_annual_returns,
    )
    from src.simulation_engine import historic_lognormal_params
    at = _run()
    inputs = {n.label: n for n in at.number_input}
    # The PPP slider defaults to 0.00%, so the MC means are calibrated on the PPP-neutral history,
    # i.e. the same series the Historic and Bootstrapping engines run under by default.
    slider = next(s for s in at.slider if s.label.startswith("Real CHF Appreciation"))
    assert slider.value == 0.0
    us_mean, us_vol = historic_lognormal_params(ppp_adjusted_annual_returns(HISTORIC_RETURNS_US_CHF, 0.0))
    exus_mean, exus_vol = historic_lognormal_params(ppp_adjusted_annual_returns(HISTORIC_RETURNS_NON_US_CHF, 0.0))
    gold_mean, gold_vol = historic_lognormal_params(ppp_adjusted_annual_returns(HISTORIC_RETURNS_GOLD_CHF, 0.0))
    infl_mean = float(np.mean(HISTORIC_SWISS_INFLATION))
    infl_std = float(np.std(HISTORIC_SWISS_INFLATION, ddof=1))
    assert inputs["Inflation Mean (%)"].value == pytest.approx(round(infl_mean * 100, 1))
    assert inputs["Inflation Volatility (%)"].value == pytest.approx(round(infl_std * 100, 1))
    assert inputs["US Stocks Nominal Mean (%)"].value == pytest.approx(round(us_mean * 100, 1))
    assert inputs["Non-US Stocks Nominal Mean (%)"].value == pytest.approx(round(exus_mean * 100, 1))
    assert inputs["Equities Volatility (%)"].value == pytest.approx(round((us_vol + exus_vol) / 2 * 100, 1))
    assert inputs["Gold Volatility (%)"].value == pytest.approx(round(gold_vol * 100, 1))
    assert inputs["Gold Nominal Mean (%)"].value == pytest.approx(round(gold_mean * 100, 1))
    # PPP-neutral means sit above the raw history, which still embeds the historic FX drag.
    raw_us_mean, _ = historic_lognormal_params(HISTORIC_RETURNS_US_CHF)
    assert us_mean > raw_us_mean
    # Sanity: the 1922-2025 CHF history is far more volatile than the old 15% placeholder.
    assert inputs["Equities Volatility (%)"].value > 18.0
    # Every MC return/vol input advertises its historic counterpart.
    for label in ("Inflation Mean (%)", "Inflation Volatility (%)", "US Stocks Nominal Mean (%)",
                  "Non-US Stocks Nominal Mean (%)", "CHF Cash Nominal Mean (%)", "Gold Nominal Mean (%)",
                  "Equities Volatility (%)", "Gold Volatility (%)"):
        assert "Historic" in inputs[label].help, label
    for label in ("US Stocks Nominal Mean (%)", "Non-US Stocks Nominal Mean (%)", "Gold Nominal Mean (%)"):
        assert "beyond PPP" in inputs[label].help, label
    mc_sub = next(s for s in at.subheader if s.value == "Monte Carlo Parameters")
    assert "Note: Returns and inflation must be Nominal" in mc_sub.proto.help
    assert not at.sidebar.caption


def test_monte_carlo_default_means_follow_ppp_slider():
    """Setting the slider to the historic value must reproduce the raw-history means (vols unchanged)."""
    from src.historic_returns import (
        HISTORIC_REAL_CHF_APPRECIATION,
        HISTORIC_RETURNS_GOLD_CHF,
        HISTORIC_RETURNS_NON_US_CHF,
        HISTORIC_RETURNS_US_CHF,
    )
    from src.simulation_engine import historic_lognormal_params

    def setup(at):
        slider = next(s for s in at.slider if s.label.startswith("Real CHF Appreciation"))
        slider.set_value(round(HISTORIC_REAL_CHF_APPRECIATION * 100, 2))

    at = _run(setup)
    assert not at.exception, [e.value for e in at.exception]
    inputs = {n.label: n for n in at.number_input}
    us_mean, us_vol = historic_lognormal_params(HISTORIC_RETURNS_US_CHF)
    exus_mean, exus_vol = historic_lognormal_params(HISTORIC_RETURNS_NON_US_CHF)
    gold_mean, gold_vol = historic_lognormal_params(HISTORIC_RETURNS_GOLD_CHF)
    assert inputs["US Stocks Nominal Mean (%)"].value == pytest.approx(round(us_mean * 100, 1))
    assert inputs["Non-US Stocks Nominal Mean (%)"].value == pytest.approx(round(exus_mean * 100, 1))
    assert inputs["Gold Nominal Mean (%)"].value == pytest.approx(round(gold_mean * 100, 1))
    assert inputs["Equities Volatility (%)"].value == pytest.approx(round((us_vol + exus_vol) / 2 * 100, 1))
    assert inputs["Gold Volatility (%)"].value == pytest.approx(round(gold_vol * 100, 1))
    # The PPP slider must be rendered before the Monte Carlo section so its value can drive the defaults.
    labels = [s.value for s in at.sidebar.subheader]
    assert labels.index("Currency / PPP Assumption") < labels.index("Monte Carlo Parameters")
