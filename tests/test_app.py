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
    assert inc_hist_names == ["Dividends", "Cash Interest", "AHV Pension", "Capital Sold", "Inflation-Adj Start"]
    assert wb_hist_names == [
        "Living Expenses (50th Pct)",
        "Taxes Paid (50th Pct)",
        "Living Expenses (Worst Cohort)",
        "Taxes Paid (Worst Cohort)",
        "Inflation-Adj Start",
    ]
    assert wb_boot_names == wb_mc_names == [
        "Living Expenses (50th Pct)",
        "Taxes Paid (50th Pct)",
        "Living Expenses (5th Pct)",
        "Taxes Paid (5th Pct)",
        "Inflation-Adj Start",
    ]
    # Chart age axis uses the retirement age lived during each simulation year (40 .. 89).
    x_vals = hist_charts[0]["data"][0]["x"]
    if isinstance(x_vals, dict):
        # Plotly binary array or list depending on version
        import base64
        import numpy as np
        arr = np.frombuffer(base64.b64decode(x_vals["bdata"]), dtype=x_vals["dtype"])
        assert (int(arr[0]), int(arr[-1])) == (40, 89)
    else:
        assert (x_vals[0], x_vals[-1]) == (40, 89)
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


def test_user_edited_mc_return_preserved_when_ppp_slider_moves():
    """Manually overriding US Stocks Nominal Mean (%) must survive subsequent PPP slider moves,
    while unedited Non-US and Gold means continue to follow the PPP slider."""
    from src.historic_returns import (
        HISTORIC_REAL_CHF_APPRECIATION,
        HISTORIC_RETURNS_GOLD_CHF,
        HISTORIC_RETURNS_NON_US_CHF,
    )
    from src.simulation_engine import historic_lognormal_params

    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    next(n for n in at.number_input if n.label == "US Stocks Nominal Mean (%)").set_value(7.0)
    at.run()
    slider = next(s for s in at.slider if s.label.startswith("Real CHF Appreciation"))
    slider.set_value(round(HISTORIC_REAL_CHF_APPRECIATION * 100, 2))
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    inputs = {n.label: n for n in at.number_input}
    exus_mean, _ = historic_lognormal_params(HISTORIC_RETURNS_NON_US_CHF)
    gold_mean, _ = historic_lognormal_params(HISTORIC_RETURNS_GOLD_CHF)
    assert inputs["US Stocks Nominal Mean (%)"].value == pytest.approx(7.0)
    assert inputs["Non-US Stocks Nominal Mean (%)"].value == pytest.approx(round(exus_mean * 100, 1))
    assert inputs["Gold Nominal Mean (%)"].value == pytest.approx(round(gold_mean * 100, 1))


def test_vanguard_bidirectional_sync_and_sidebar_modes():
    """Verify Annual Base Expenses <-> Target Withdrawal Rate sync, Threshold rebalancing,
    zero Pillar 3a accounts, and Retirement Age == 65."""
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    initial_twr = next(n for n in at.number_input if n.label == "Target Withdrawal Rate (%)").value

    # Increasing Annual Base Expenses increases Target Withdrawal Rate (%)
    next(n for n in at.number_input if n.label == "Annual Base Expenses (CHF)").set_value(120_000)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    new_twr = next(n for n in at.number_input if n.label == "Target Withdrawal Rate (%)").value
    assert new_twr > initial_twr

    # Lowering Target Withdrawal Rate (%) back down updates Annual Base Expenses (CHF)
    next(n for n in at.number_input if n.label == "Target Withdrawal Rate (%)").set_value(2.50)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    synced_exp = next(n for n in at.number_input if n.label == "Annual Base Expenses (CHF)").value
    assert synced_exp < 120_000

    # Exercise Threshold rebalancing, 0 Pillar 3a accounts, and Retirement Age = 65
    next(s for s in at.selectbox if s.label == "Frequency").set_value("Threshold")
    next(n for n in at.number_input if n.label == "Number of Pillar 3a Accounts").set_value(0)
    next(n for n in at.number_input if n.label == "Retirement Age").set_value(65)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    p3a_bal = next(n for n in at.number_input if n.label == "Balance per Pillar 3a Account (CHF)")
    assert p3a_bal.disabled



# --------------------------------------------------------------------------- Austria (country switch)


def _select_austria(at):
    next(s for s in at.selectbox if s.label == "Country").set_value("Austria")


def test_app_runs_in_austria_mode():
    import json

    import numpy as np

    from src.historic_returns_eur import HISTORIC_AUSTRIAN_INFLATION, HISTORIC_REAL_EUR_APPRECIATION
    from src.tax_engine_at import HEALTH_SELF_INSURANCE_MONTHLY

    at = _run(_select_austria)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "Austria Early Retirement Simulator"
    assert [h.value for h in at.main.header] == ["Historic Backtesting", "Historic Bootstrapping", "Monte Carlo"]
    # The Swiss-data stand-ins are disclosed prominently.
    assert any("1922–1957" in w.value and "Swiss" in w.value for w in at.warning)

    inputs = {n.label: n for n in at.number_input}
    assert "Pillar 2 (Freizügigkeitskonto)" not in inputs
    assert "Number of Pillar 3a Accounts" not in inputs
    assert "Municipal Multiplier (Steuerfuss, %)" not in inputs
    assert "Taxable Liquid Wealth (EUR)" in inputs and "EUR Cash" in inputs
    assert inputs["Taxable Liquid Wealth (EUR)"].value == 3_250_000
    assert "Expected Monthly Gross State Pension from age 65 (EUR, paid 14×/yr)" in inputs
    assert inputs["EUR Cash Nominal Mean (%)"].value == pytest.approx(2.0)
    assert inputs["Tax Bracket Indexation (% of CPI)"].value == pytest.approx(66.7)
    health = inputs["Health Self-Insurance (EUR/month, § 16 ASVG)"]
    assert not health.disabled and health.value == pytest.approx(HEALTH_SELF_INSURANCE_MONTHLY)
    assert "§ 19a" in health.help
    assert inputs["Health Insurance Real Growth (%/yr above CPI)"].value == pytest.approx(0.7)
    assert inputs["Inflation Mean (%)"].value == pytest.approx(round(float(np.mean(HISTORIC_AUSTRIAN_INFLATION)) * 100, 1))
    sidebar_subheaders = [s.value for s in at.sidebar.subheader]
    assert "Austrian Tax Assumptions" in sidebar_subheaders and "Zurich Tax Location" not in sidebar_subheaders
    slider = next(s for s in at.slider if s.label.startswith("Real EUR Appreciation"))
    assert f"+{HISTORIC_REAL_EUR_APPRECIATION * 100:.2f}%" in slider.help

    hist_charts = [json.loads(c.proto.spec) for c in at.columns[0].get("plotly_chart")]
    assert [t["name"] for t in hist_charts[1]["data"]] == ["Solvent (NW > 0)", "Above Target (50% Start NW)"]
    assert [t["name"] for t in hist_charts[2]["data"]] == ["Dividends", "Cash Interest", "State Pension", "Capital Sold", "Inflation-Adj Start"]
    taxes_bar = hist_charts[3]["data"][1]
    assert taxes_bar["name"] == "Taxes Paid (50th Pct)"
    assert "KESt" in taxes_bar["customdata"][0] and "Health insurance" in taxes_bar["customdata"][0]
    alloc_names = [t["name"] for t in hist_charts[5]["data"]]
    assert alloc_names[0] == "EUR Cash" and "Pillar 2 & 3a" not in alloc_names
    assert sum(1 for m in at.metric if m.label.startswith("Pre-Pension Outflow")) == 3
    assert all("EUR" in m.value for m in at.metric if m.label == "Median Ending NW (Real)")
    cohort_sub = next(s for s in at.subheader if s.value == "Cohort Analysis")
    assert "Liq Age" not in cohort_sub.proto.help and "Pillar" not in cohort_sub.proto.help


def test_switching_country_resets_country_defaults():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    _select_austria(at)
    at.run()
    assert next(n for n in at.number_input if n.label == "Taxable Liquid Wealth (EUR)").value == 3_250_000
    assert next(n for n in at.number_input if n.label == "EUR Cash Nominal Mean (%)").value == pytest.approx(2.0)
    next(s for s in at.selectbox if s.label == "Country").set_value("Switzerland (Zurich)")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    inputs = {n.label: n for n in at.number_input}
    assert inputs["Taxable Liquid Wealth (CHF)"].value == 2_450_000
    assert inputs["CHF Cash Nominal Mean (%)"].value == pytest.approx(1.0)
    assert inputs["Tax Bracket Indexation (% of CPI)"].value == pytest.approx(100.0)
    assert "Pillar 2 (Freizügigkeitskonto)" in inputs
    assert at.title[0].value == "Zurich Early Retirement Simulator"
    assert not at.warning


def test_switching_country_keeps_a_custom_allocation_valid():
    # Regression: the cash allocation input's label contains the currency; without a stable key the
    # switch reset it to 10% while the other allocation inputs kept their values (sum != 100%).
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    next(n for n in at.number_input if n.label == "US Stocks").set_value(45.0)
    next(n for n in at.number_input if n.label == "CHF Cash").set_value(15.0)
    at.run()
    assert not at.error, [e.value for e in at.error]
    _select_austria(at)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    inputs = {n.label: n for n in at.number_input}
    assert inputs["EUR Cash"].value == pytest.approx(15.0)
    assert inputs["US Stocks"].value == pytest.approx(45.0)
    # Liquid wealth is reset to the country default on purpose (CH excludes Pillar 2 / 3a).
    assert inputs["Taxable Liquid Wealth (EUR)"].value == 3_250_000


def test_austrian_health_insurance_input_reaches_the_simulation():
    def pre_pension_outflows(at):
        return [m.value for m in at.metric if m.label.startswith("Pre-Pension Outflow")]

    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT_S)
    at.run()
    _select_austria(at)
    at.run()
    before = pre_pension_outflows(at)
    next(n for n in at.number_input if n.label.startswith("Health Self-Insurance")).set_value(83.49)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert next(n for n in at.number_input if n.label.startswith("Health Self-Insurance")).value == pytest.approx(83.49)
    after = pre_pension_outflows(at)
    assert len(before) == len(after) == 3 and all(b != a for b, a in zip(before, after))
