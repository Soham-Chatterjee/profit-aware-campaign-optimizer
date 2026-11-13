"""Milestone 6 tests: end-to-end outputs, leakage checks, optimisation property tests, backtest,
robustness and fairness monitor."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pacopt import config as cfg
from pacopt import evaluation as ev
from pacopt import features as ft
from pacopt import model as m
from pacopt import optimize as op
from pacopt import pipeline as pl

SEGS = list(cfg.SEGMENTS)


@pytest.fixture(scope="module")
def table():
    return pl.load_campaign_table()


@pytest.fixture(scope="module")
def comparison(table):
    saved = pl.load_optimisation(table)
    assert saved is not None, "artifacts are stale: run `uv run python -m pacopt.pipeline --no-retrain`"
    return op.with_optimisers(table, op.compare_strategies(table, run_genetic=False, run_milp=False), *saved)


# ----------------------------------------------------------------------------- 6.1 end-to-end outputs
def test_pipeline_outputs_exist_with_expected_shapes(table, comparison):
    for p in (pl.CAMPAIGN_TABLE_PATH, m.MODEL_PATH, m.METRICS_PATH, m.SCORED_PATH, pl.STRATEGY_PATH,
              pl.TARGET_LIST_PATH, pl.GA_HISTORY_PATH, pl.SELECTIONS_PATH, pl.OPTIM_META_PATH):
        assert p.exists(), p
    assert table.shape[0] == cfg.UNIVERSE_SIZE and table["customer_id"].is_unique
    tl = pd.read_csv(pl.TARGET_LIST_PATH)
    assert op.is_feasible(table, tl["customer_id"])                       # saved GA list is feasible
    sc = pd.read_csv(pl.STRATEGY_PATH, index_col=0)
    assert list(sc.index) == list(op.STRATEGY_ORDER) and sc["feasible"].all()
    assert json.loads(pl.OPTIM_META_PATH.read_text())["fingerprint"] == op.fingerprint(table)


# ----------------------------------------------------------------------------- 6.2 leakage
def test_allow_list_has_no_forbidden_columns():
    ft.assert_no_leakage(ft.MODEL_FEATURES)
    assert not set(ft.MODEL_FEATURES) & cfg.FORBIDDEN_FEATURES


def test_saved_model_uses_allow_list_and_transforms_inside_pipeline():
    b = m.load_model()
    assert b["features"] == ft.MODEL_FEATURES
    for name in ("lr", "rf"):
        chk = ev.transforms_inside_pipeline(b[name])
        assert chk["preprocessing_inside_pipeline"] and chk["n_fitted_pipelines"] == cfg.MODEL.cv_folds, chk
        assert chk["steps"] == ["prep", "clf"]


def test_universe_never_used_in_training():
    train, test, universe = m.load_processed()
    assert not set(universe["customer_id"]) & set(train["customer_id"])
    assert set(universe["customer_id"]) <= set(test["customer_id"])


def test_leakage_ablation_shows_leaky_columns_inflate_auc():
    train, test, _ = m.load_processed()
    ab = ev.leakage_ablation(train, test)
    assert list(ab["leaky_columns_used"])[0] == "none"
    chosen, partial, full = ab["roc_auc"].tolist()
    assert chosen < partial < full and full - chosen > 0.10            # duration is the big leak
    assert chosen >= 0.72


# ----------------------------------------------------------------------------- 6.3 optimisation property tests
def _random_table(n: int, rng: np.random.Generator) -> pd.DataFrame:
    p_calls = rng.uniform(0.4, 0.9)
    group = rng.choice(["calls", "digital"], n, p=[p_calls, 1 - p_calls])
    cost = np.where(group == "digital", rng.uniform(1, 4), rng.choice([8.0, 10.0, 12.0, 15.0], n))
    ep = rng.normal(rng.uniform(-2, 12), rng.uniform(5, 20), n)
    return pd.DataFrame({
        "customer_id": [f"R{i:05d}" for i in range(n)], "eligible": (rng.random(n) > rng.uniform(0, 0.3)).astype(int),
        "expected_profit": ep, "contact_cost": cost, "propensity": rng.uniform(0.01, 0.7, n),
        "balance": rng.lognormal(7, 1.5, n), "channel_group": group,
        "segment": pd.Categorical(rng.choice(SEGS, n), categories=SEGS, ordered=True),
        "flag_age_75plus": (rng.random(n) < rng.uniform(0, 0.08)).astype(int),
    })


def _random_constraints(n_elig: int, rng: np.random.Generator) -> cfg.CampaignConstraints:
    return cfg.CampaignConstraints(
        budget=float(rng.uniform(200, 3_000)), max_contacts=int(rng.uniform(0.1, 0.6) * n_elig),
        channel_capacity={"calls": int(rng.uniform(0.05, 0.4) * n_elig), "digital": int(rng.uniform(0.05, 0.4) * n_elig)},
        segment_max_share={s: float(rng.uniform(0.35, 0.8)) for s in SEGS},
        segment_min_share={"Growing": float(rng.uniform(0, 0.25)), "Dormant": float(rng.uniform(0, 0.25))},
        protected_max_share=float(rng.uniform(0.0, 0.05)))


@pytest.mark.parametrize("seed", range(25))
def test_property_every_strategy_respects_every_limit(seed):
    rng = np.random.default_rng(1_000 + seed)
    t = _random_table(int(rng.integers(150, 600)), rng)
    cc = _random_constraints(int(t["eligible"].sum()), rng)
    pr = op.build_problem(t, cc)
    lists = {"Greedy": op.greedy(pr), "Propensity": op.propensity_only(pr), "Rule-based": op.rule_based(pr),
             "Random": op.random_selection(pr, seed),
             "GA": op.run_ga(pr, replace(cfg.GA, population=30, generations=25, patience=10, random_state=seed)).x}
    for name, x in lists.items():
        ck = op.check_constraints(t, pr.customer_id[x], cc)
        assert ck["ok"].all(), (seed, name, ck[~ck["ok"]].to_dict("records"))
        assert t.set_index("customer_id").loc[pr.customer_id[x], "eligible"].eq(1).all()


@pytest.mark.parametrize("seed", range(10))
def test_property_repair_always_feasible(seed):
    rng = np.random.default_rng(2_000 + seed)
    t = _random_table(int(rng.integers(150, 500)), rng)
    pr = op.build_problem(t, _random_constraints(int(t["eligible"].sum()), rng))
    o = op._orders(pr)
    eng = op._Engine(pr, o["ep_per_cost"], o["ep"], fill_pool=pr.ep > 0)
    X = rng.random((40, pr.n)) < rng.random((40, 1))               # densities from empty to full
    assert (eng.violation(eng.repair(X)) == 0).all()
    assert (eng.violation(eng.repair(X, fill=False)) == 0).all()


# ----------------------------------------------------------------------------- 6.4 backtest
def test_backtest_on_actual_outcomes(table, comparison):
    bt = ev.backtest(table, comparison)
    assert list(bt.index) == list(op.STRATEGY_ORDER)
    ga = bt.loc["Genetic Algorithm"]
    assert ga["actual_conversions"] == int(table.set_index("customer_id").loc[comparison.selected_ids("Genetic Algorithm"), "y"].sum())
    # calibration on the chosen list: actual conversions within 5% of expected
    assert abs(ga["actual_conversions"] / ga["expected_conversions"] - 1) < 0.05
    # the targeted lists beat Random on realised (not just expected) contribution
    for s in ("Propensity", "Greedy", "Genetic Algorithm"):
        assert bt.loc[s, "realised_contribution"] > bt.loc["Random", "realised_contribution"]
    ci = ev.bootstrap_backtest(table, comparison, ("Genetic Algorithm",), n_boot=300)
    assert ci.loc["Genetic Algorithm", "ci_low"] > bt.loc["Random", "realised_contribution"]


# ----------------------------------------------------------------------------- 6.5 robustness (fast form)
def test_ga_reaches_optimum_across_seeds(table, comparison):
    # one seed per mode here; notebook 05 runs the full 5-seed study (6.5)
    r = ev.ga_seed_runs(table, seeds=[7], optimum=comparison.milp.expected_profit,
                        ga=replace(cfg.GA, time_limit_s=45))
    assert r["feasible"].all()
    assert r["pct_of_optimum"].min() >= 0.995


# ----------------------------------------------------------------------------- 6.6 fairness monitor
def test_fairness_monitor(table, comparison):
    fm = ev.fairness_monitor(table, comparison.selected_ids("Genetic Algorithm"))
    elig = int(table["eligible"].sum())
    for dim, g in fm.groupby("dimension"):
        assert g["eligible"].sum() == elig
        assert g["selected"].sum() == len(comparison.selected_ids("Genetic Algorithm"))
    assert set(fm["flag"]) <= {"", "under-targeted", "over-targeted"}
    small = fm[fm["eligible"] < ev.FAIRNESS.min_group_size]
    assert (small["flag"] == "").all()                                   # tiny groups never flagged
    protected = fm[(fm["dimension"] == "age_band") & (fm["group"] == "75+")]
    assert protected["share_of_selected"].iloc[0] <= cfg.CONSTRAINTS.protected_max_share + 1e-9
