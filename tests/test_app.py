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
    tldr = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert len(tldr) == 3 and all(any(k in t for k in ("BROKE", "RICH", "DEAD")) for t in tldr)
    assert sum(1 for m in at.metric if m.label == "Probability of Success") == 3


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
    )
    from src.simulation_engine import historic_lognormal_params
    at = _run()
    inputs = {n.label: n for n in at.number_input}
    us_mean, us_vol = historic_lognormal_params(HISTORIC_RETURNS_US_CHF)
    exus_mean, exus_vol = historic_lognormal_params(HISTORIC_RETURNS_NON_US_CHF)
    gold_mean, gold_vol = historic_lognormal_params(HISTORIC_RETURNS_GOLD_CHF)
    infl_mean = float(np.mean(HISTORIC_SWISS_INFLATION))
    infl_std = float(np.std(HISTORIC_SWISS_INFLATION, ddof=1))
    assert inputs["Inflation Mean (%)"].value == pytest.approx(round(infl_mean * 100, 1))
    assert inputs["Inflation Volatility (%)"].value == pytest.approx(round(infl_std * 100, 1))
    assert inputs["US Stocks Nominal Mean (%)"].value == pytest.approx(round(us_mean * 100, 1))
    assert inputs["Non-US Stocks Nominal Mean (%)"].value == pytest.approx(round(exus_mean * 100, 1))
    assert inputs["Equities Volatility (%)"].value == pytest.approx(round((us_vol + exus_vol) / 2 * 100, 1))
    assert inputs["Gold Volatility (%)"].value == pytest.approx(round(gold_vol * 100, 1))
    assert inputs["Gold Nominal Mean (%)"].value == pytest.approx(round(gold_mean * 100, 1))
    # Sanity: the 1922-2025 CHF history is far more volatile than the old 15% placeholder.
    assert inputs["Equities Volatility (%)"].value > 18.0
    # Every MC return/vol input advertises its historic counterpart.
    for label in ("Inflation Mean (%)", "Inflation Volatility (%)", "US Stocks Nominal Mean (%)",
                  "Non-US Stocks Nominal Mean (%)", "CHF Cash Nominal Mean (%)", "Gold Nominal Mean (%)",
                  "Equities Volatility (%)", "Gold Volatility (%)"):
        assert "Historic" in inputs[label].help, label
