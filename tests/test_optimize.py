"""Milestone 4 tests: constraint checker, strategies, GA operators, MILP optimum and the comparison."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pacopt import config as cfg
from pacopt import optimize as op
from pacopt import pipeline as pl

SEGS = list(cfg.SEGMENTS)


# ----------------------------------------------------------------------------- synthetic problems
def _toy_table(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A random campaign table with the columns the optimiser needs."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["calls", "digital"], n, p=[0.7, 0.3])
    cost = np.where(group == "digital", 2.0, rng.choice([10.0, 12.0], n))
    ep = rng.normal(8, 12, n)
    return pd.DataFrame({
        "customer_id": [f"T{i:05d}" for i in range(n)],
        "eligible": (rng.random(n) > 0.1).astype(int),
        "expected_profit": ep,
        "contact_cost": cost,
        "propensity": rng.uniform(0.01, 0.6, n),
        "balance": rng.lognormal(7, 1.2, n),
        "channel_group": group,
        "channel": np.where(group == "digital", "digital", "mobile_call"),
        "segment": pd.Categorical(rng.choice(SEGS, n, p=[0.3, 0.15, 0.25, 0.3]), categories=SEGS, ordered=True),
        "flag_age_75plus": (rng.random(n) < 0.03).astype(int),
        "value": rng.uniform(50, 500, n),
        "break_even_cost": rng.uniform(0, 50, n),
        "ep_per_cost": ep / cost,
        "age": rng.integers(18, 90, n),
    })


def _toy_constraints(n_elig: int) -> cfg.CampaignConstraints:
    return cfg.CampaignConstraints(budget=1_200, max_contacts=int(0.35 * n_elig),
                                   channel_capacity={"calls": int(0.2 * n_elig), "digital": int(0.12 * n_elig)},
                                   segment_max_share={s: 0.5 for s in SEGS},
                                   segment_min_share={"Growing": 0.15, "Dormant": 0.10},
                                   protected_max_share=0.02)


FAST_GA = replace(cfg.GA, population=40, generations=60, patience=20)


# ----------------------------------------------------------------------------- constraint checker
def test_check_constraints_flags_each_violation():
    t = _toy_table()
    cc = _toy_constraints(int(t.eligible.sum()))
    elig = t[t.eligible == 1]
    assert op.is_feasible(t, [], cc)                                      # empty list is feasible
    bad = op.check_constraints(t, t[t.eligible == 0].customer_id[:1], cc)
    assert not bad.set_index("constraint").loc["Eligible customers only", "ok"]
    over = op.check_constraints(t, elig.customer_id, cc).set_index("constraint")["ok"]
    assert not over["Budget (£)"] and not over["Contacts"] and not over["Channel capacity: calls"]
    dup = op.check_constraints(t, list(elig.customer_id[:3]) * 2, cc).set_index("constraint")["ok"]
    assert not dup["Unique customers"]
    only_hv = elig[elig.segment == "High value"].customer_id[:5]
    seg = op.check_constraints(t, only_hv, cc).set_index("constraint")["ok"]
    assert not seg["Max share: High value"] and not seg["Min share: Growing"]


# ----------------------------------------------------------------------------- strategies are feasible
@pytest.mark.parametrize("seed", range(6))
def test_every_strategy_feasible_on_random_scenarios(seed):
    t = _toy_table(seed=seed)
    cc = _toy_constraints(int(t.eligible.sum()))
    pr = op.build_problem(t, cc)
    strategies = {"greedy": op.greedy(pr), "propensity": op.propensity_only(pr), "rule": op.rule_based(pr),
                  "random": op.random_selection(pr, seed),
                  "ga": op.run_ga(pr, replace(FAST_GA, random_state=seed)).x}
    for name, x in strategies.items():
        ck = op.check_constraints(t, pr.customer_id[x], cc)
        assert ck["ok"].all(), (name, ck[~ck["ok"]])
        assert x.sum() > 0, name


@pytest.mark.parametrize("seed", range(4))
def test_repair_returns_feasible_from_any_chromosome(seed):
    t = _toy_table(seed=seed)
    pr = op.build_problem(t, _toy_constraints(int(t.eligible.sum())))
    o = op._orders(pr)
    eng = op._Engine(pr, o["ep_per_cost"], o["ep"], fill_pool=pr.ep > 0)
    X = np.random.default_rng(seed).random((30, pr.n)) < np.linspace(0, 1, 30)[:, None]   # empty .. full
    for fill in (True, False):
        R = eng.repair(X, fill=fill)
        assert (eng.violation(R) == 0).all()
        for row in R:
            assert op.is_feasible(t, pr.customer_id[row], pr.cc)


def test_greedy_skips_customers_who_would_break_the_budget():
    t = pd.DataFrame({"customer_id": ["a", "b", "c"], "eligible": 1, "expected_profit": [100.0, 50.0, 4.0],
                      "contact_cost": [10.0, 10.0, 2.0], "propensity": [.5, .4, .3], "balance": [1, 2, 3],
                      "channel_group": "calls", "segment": "High value", "flag_age_75plus": 0})
    cc = cfg.CampaignConstraints(budget=12, max_contacts=10, channel_capacity={"calls": 10},
                                 segment_max_share={}, segment_min_share={}, protected_max_share=1.0)
    pr = op.build_problem(t, cc)
    x = op.greedy(pr)
    assert pr.customer_id[x].tolist() == ["a", "c"]          # b does not fit after a; c still does


def test_negative_ep_customers_used_only_to_meet_a_minimum():
    t = _toy_table(seed=3)
    cc = _toy_constraints(int(t.eligible.sum()))
    pr = op.build_problem(t, cc)
    x = op.greedy(pr)
    neg = x & (pr.ep <= 0)
    min_segs = [k for k, s in enumerate(pr.segments) if s in cc.segment_min_share]
    assert np.isin(pr.segment[neg], min_segs).all()


# ----------------------------------------------------------------------------- GA vs exact optimum
def test_ga_matches_bruteforce_on_tiny_problem():
    t = _toy_table(n=14, seed=7).assign(eligible=1)
    cc = cfg.CampaignConstraints(budget=40, max_contacts=6, channel_capacity={"calls": 3, "digital": 4},
                                 segment_max_share={s: 0.6 for s in SEGS}, segment_min_share={},
                                 protected_max_share=1.0)
    pr = op.build_problem(t, cc)
    X = ((np.arange(1 << pr.n)[:, None] >> np.arange(pr.n)) & 1).astype(bool)   # all 2^14 subsets
    feas = op._Engine(pr, np.arange(pr.n)).violation(X) == 0
    best = float((X[feas] * pr.ep).sum(1).max())
    x_best = X[feas][np.argmax((X[feas] * pr.ep).sum(1))]
    assert op.is_feasible(t, pr.customer_id[x_best], cc)                  # independent checker agrees
    m = op.milp_optimum(pr)
    ga = op.run_ga(pr, replace(FAST_GA, population=30))
    assert m.expected_profit == pytest.approx(best)
    assert ga.expected_profit == pytest.approx(best)


def test_ga_is_reproducible_and_monotone():
    t = _toy_table(seed=1)
    pr = op.build_problem(t, _toy_constraints(int(t.eligible.sum())))
    a, b = op.run_ga(pr, FAST_GA), op.run_ga(pr, FAST_GA)
    assert np.array_equal(a.x, b.x) and a.expected_profit == b.expected_profit
    best = a.history["best"].to_numpy()
    assert (np.diff(best) >= -1e-9).all()                                # elitism: best never falls


def test_milp_is_an_upper_bound_for_every_strategy():
    t = _toy_table(seed=2)
    cc = _toy_constraints(int(t.eligible.sum()))
    c = op.compare_strategies(t, cc, FAST_GA, replace(cfg.OPTIM, random_runs=5))
    assert c.milp.optimal
    assert (c.table["expected_contribution"] <= c.milp.expected_profit + 1e-6).all()
    assert c.table["feasible"].all()


# ----------------------------------------------------------------------------- default scenario (integration)
@pytest.fixture(scope="module")
def comparison():
    t = pl.campaign_table()
    return t, op.compare_strategies(t)


def test_default_scenario_acceptance(comparison):
    t, c = comparison
    tb = c.table
    assert tb["feasible"].all()                                          # every strategy passes check_constraints
    ga = tb.loc["Genetic Algorithm", "expected_contribution"]
    assert ga > tb.loc["Random", "expected_contribution"]
    assert ga > tb.loc["Propensity", "expected_contribution"]
    assert ga >= tb.loc["Greedy", "expected_contribution"] - 1e-6
    assert c.milp.optimal
    assert tb.loc["Genetic Algorithm", "pct_of_optimum"] >= 0.98
    assert c.ga.runtime_s < 60
    assert c.random_runs["feasible"].all() and len(c.random_runs) == cfg.OPTIM.random_runs


def test_target_list_matches_ga_selection(comparison):
    t, c = comparison
    tl = op.target_list(t, c)
    assert len(tl) == c.table.loc["Genetic Algorithm", "contacts"]
    assert tl["customer_id"].is_unique
    assert tl["expected_profit"].sum() == pytest.approx(c.ga.expected_profit)
    mix = op.selection_mix(t, c)
    assert np.allclose(mix.sum(), 1.0)
