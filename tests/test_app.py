"""Milestone 5 tests: every dashboard page runs without exceptions (Streamlit AppTest), the default
scenario reuses the saved GA/MILP artifacts, and a changed scenario recomputes."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pacopt import config as cfg
from pacopt import pipeline as pl
from pacopt import scenario as sc

APP = Path(__file__).resolve().parents[1] / "app" / "Home.py"
PAGES = ["views/0_Command_Centre.py", "views/1_Customer_Intelligence.py", "views/2_Propensity_and_Opportunities.py",
         "views/3_Campaign_Optimization.py", "views/4_What_If_Simulation.py"]


def _run(page: str | None = None, timeout: int = 180) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    if page:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Prototype results" in c.value or "illustrative" in c.value for c in at.caption)   # disclaimer (5.7)


def test_default_scenario_round_trips_and_reuses_saved_optimisation():
    assert sc.Scenario.from_json(sc.DEFAULT.to_json()) == sc.DEFAULT
    t = sc.build_table(sc.DEFAULT)
    saved = pl.load_optimisation(t, sc.DEFAULT.constraints)
    assert saved is not None, "run `uv run python -m pacopt.pipeline --no-retrain` to refresh artifacts"
    ga, milp = saved
    assert ga.x.sum() > 0 and milp is not None and milp.optimal


def test_default_scenario_table_matches_pipeline():
    a = sc.build_table(sc.DEFAULT)
    b = pl.campaign_table()
    assert (a["expected_profit"].round(9) == b["expected_profit"].round(9)).all()
    assert (a["segment"] == b["segment"]).all()


def _metric(at: AppTest, label: str) -> str:
    return next(mtr for mtr in at.metric if mtr.label == label).value


def test_sidebar_change_updates_results_and_persists_across_pages():
    at = _run()
    home_tile = _metric(at, "Expected contribution")
    at.number_input(key="s_budget").set_value(30_000.0).run()
    assert not at.exception
    assert _metric(at, "Campaign budget") == "£30.0K"
    assert _metric(at, "Expected contribution") != home_tile
    at.switch_page("views/1_Customer_Intelligence.py").run()          # the value follows the user
    assert at.number_input(key="s_budget").value == 30_000.0
    at.switch_page("views/0_Command_Centre.py").run()
    assert _metric(at, "Campaign budget") == "£30.0K"
    at.sidebar.button[0].click().run()                                  # reset to defaults
    assert _metric(at, "Campaign budget") == "£50.0K"


def test_changed_scenario_does_not_reuse_saved_optimisation():
    from dataclasses import replace
    s = replace(sc.DEFAULT, constraints=replace(sc.DEFAULT.constraints, budget=30_000.0))
    assert pl.load_optimisation(sc.build_table(s), s.constraints) is None


def test_model_switch_changes_propensity():
    a = sc.build_table(sc.DEFAULT)
    b = sc.build_table(sc.Scenario(model="rf"))
    assert not (a["propensity"] == b["propensity"]).all()
    assert b["propensity"].max() <= cfg.MODEL.propensity_cap
