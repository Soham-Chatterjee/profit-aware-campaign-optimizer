"""Milestone 2 tests: metrics helpers, leakage guard in training, acceptance criteria, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from pacopt import config as cfg
from pacopt import features as ft
from pacopt import model as m


# ----------------------------------------------------------------------------- metric helpers (fast)
def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert m.precision_at_k(y, p, 1) == 1.0
    assert m.precision_at_k(y, p, 2) == 0.5
    assert m.precision_at_k(y, p, 3) == pytest.approx(2 / 3)
    assert m.precision_at_k(y, p, 99) == pytest.approx(0.4)  # k capped at n


def test_budget_implied_k():
    p = np.array([0.9, 0.5, 0.1])
    cost = np.array([12.0, 2.0, 10.0])
    assert m.budget_implied_k(p, cost, budget=14) == 2
    assert m.budget_implied_k(p, cost, budget=11) == 0


def test_contact_cost_mapping():
    df = pd.DataFrame({"contact": ["cellular", "telephone", "unknown"]})
    assert m.contact_cost(df).tolist() == [12.0, 10.0, 2.0]


def test_gains_and_calibration_tables():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2_000)
    y = (rng.uniform(size=2_000) < p).astype(int)
    g = m.gains_table(y, p)
    assert g["n"].sum() == 2_000 and g["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert g["rate"].iloc[0] > g["rate"].iloc[-1]                   # decile 1 = highest propensity
    c = m.calibration_table(y, p)
    assert c["gap_pp"].abs().max() < 6                               # perfectly calibrated by construction


def test_choose_model_rule():
    lr = m.Evaluation("lr", 0.75, 0.38, 0.09, 0.3, 0.117, 1.0)
    rf_small = m.Evaluation("rf", 0.76, 0.39, 0.09, 0.3, 0.117, 0.9)
    rf_big_badcal = m.Evaluation("rf", 0.78, 0.42, 0.09, 0.3, 0.117, 3.0)
    rf_big_goodcal = m.Evaluation("rf", 0.78, 0.42, 0.09, 0.3, 0.117, 0.8)
    assert m.choose_model(lr, rf_small)[0] == "lr"
    assert m.choose_model(lr, rf_big_badcal)[0] == "lr"
    assert m.choose_model(lr, rf_big_goodcal)[0] == "rf"


@pytest.mark.parametrize("leaky", ["campaign", "duration", "month", "day"])
def test_training_rejects_leaky_features(leaky):
    df = pd.DataFrame({"age": [30, 40], leaky: [1, 2], "y": [0, 1]})
    with pytest.raises(ft.LeakageError):
        m.fit_lr(df, numeric=["age", leaky], categorical=[])


# ----------------------------------------------------------------------------- trained model (slow, ~20 s)
@pytest.fixture(scope="module")
def trained():
    train, test, universe = m.load_processed()
    return m.train_and_evaluate(train, test, universe), universe


def test_model_uses_only_allowed_features(trained):
    res, _ = trained
    cols = m.model_columns(res.lr)
    assert cols == ft.MODEL_FEATURES
    ft.assert_no_leakage(cols)


def test_acceptance_roc_auc(trained):
    res, _ = trained
    assert res.eval_lr.roc_auc >= 0.72


def test_acceptance_calibration(trained):
    res, _ = trained
    assert res.eval_lr.max_calibration_gap_pp <= 1.5


def test_acceptance_precision_lift(trained):
    res, _ = trained
    lift = res.eval_lr.precision_at_k["K=1000"] / res.eval_lr.base_rate
    assert lift >= 2.0


def test_budget_precision_key_is_stable(trained):
    res, _ = trained
    for ev in (res.eval_lr, res.eval_rf):
        assert "K=budget" in ev.precision_at_k
        assert 0 < ev.budget_k <= cfg.UNIVERSE_SIZE


def test_scored_universe(trained):
    res, universe = trained
    s = res.scored
    assert len(s) == cfg.UNIVERSE_SIZE
    assert s["customer_id"].tolist() == universe["customer_id"].tolist()
    assert s["propensity"].between(0, cfg.MODEL.propensity_cap).all()
    assert (s["propensity"] <= s["p_lr"] + 1e-12).all()             # cap only lowers, never raises
    assert set(s["propensity_decile"].unique()) == set(range(1, 11))
    assert s["propensity_rank"].is_unique


def test_reproducible(trained):
    res, universe = trained
    train, test, _ = m.load_processed()
    again = m.train_and_evaluate(train, test, universe)
    np.testing.assert_allclose(again.scored["propensity"], res.scored["propensity"])
    # RF trees are fitted in parallel; summation order can differ at ~1e-7 between runs.
    assert again.eval_rf.pr_auc == pytest.approx(res.eval_rf.pr_auc, abs=1e-5)
