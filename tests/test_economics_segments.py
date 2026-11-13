"""Milestone 3 tests: hand-calculated economics, no principal in revenue, segments, RFM, budget binding."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pacopt import config as cfg
from pacopt import economics as econ
from pacopt import pipeline as pl
from pacopt import segmentation as seg

EC = cfg.EconomicsConfig(deposit_min=10_000, deposit_alpha=1.0, deposit_max=100_000, ftp_rate=0.0475,
                         customer_rate=0.03, term_years=1, incentive=40, onboarding_cost=5, organic_rate=0.0)


def _toy():
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "balance": [2_000, -500, 500_000, 0],
        "contact": ["cellular", "telephone", "unknown", "unknown"],
        "propensity": [0.20, 0.05, 0.50, 0.0],
        "eligible": [1, 1, 1, 0],
    })


# ----------------------------------------------------------------------------- hand-calculated examples
def test_hand_calculated_ep():
    t = econ.add_economics(_toy(), EC)
    # A: D = 10,000 + 2,000 = 12,000; V = 12,000 x 1.75% - 40 - 5 = 165; BE = 0.2 x 165 = 33; EP = 33 - 12 = 21
    a = t.iloc[0]
    assert a["deposit"] == pytest.approx(12_000)
    assert a["value"] == pytest.approx(165.0)
    assert a["break_even_cost"] == pytest.approx(33.0)
    assert a["expected_profit"] == pytest.approx(21.0)
    assert a["ep_per_cost"] == pytest.approx(21 / 12)
    # B: negative balance -> minimum ticket; V = 175 - 45 = 130; EP = 0.05 x 130 - 10 = -3.5
    assert t.iloc[1]["deposit"] == pytest.approx(10_000)
    assert t.iloc[1]["expected_profit"] == pytest.approx(-3.5)
    # C: capped at 100,000; V = 1,750 - 45 = 1,705; EP = 0.5 x 1,705 - 2 = 850.5
    assert t.iloc[2]["deposit"] == pytest.approx(100_000)
    assert t.iloc[2]["expected_profit"] == pytest.approx(850.5)
    # D: zero propensity -> EP = -cost
    assert t.iloc[3]["expected_profit"] == pytest.approx(-2.0)


def test_organic_rate_reduces_uplift_and_floors_at_zero():
    ec = replace(EC, organic_rate=0.10)
    t = econ.add_economics(_toy(), ec)
    assert t.iloc[0]["break_even_cost"] == pytest.approx(0.10 * 165)
    assert t.iloc[1]["break_even_cost"] == 0.0          # P below organic rate -> no incremental value


def test_principal_is_never_revenue():
    """V_i rises by only spread x term per extra £ deposited, and is always far below the deposit."""
    d = np.array([10_000.0, 20_000.0])
    v = econ.contribution_value(d, EC)
    assert (v[1] - v[0]) == pytest.approx(10_000 * EC.spread * EC.term_years)
    assert (v < 0.05 * d).all()
    t = econ.add_economics(_toy(), EC)
    assert (t["value"] < t["deposit"] * EC.spread * EC.term_years + 1e-9).all()


def test_channel_costs_and_unknown_contact_rejected():
    assert econ.contact_cost(["cellular", "telephone", "unknown"], EC).tolist() == [12.0, 10.0, 2.0]
    with pytest.raises(ValueError):
        econ.contact_cost(["pigeon"], EC)


def test_rankings():
    t = econ.add_economics(_toy(), EC)
    assert t.loc[t["ep_rank"] == 1, "customer_id"].item() == "C"
    assert sorted(t["ep_rank"]) == [1, 2, 3, 4]
    assert t.loc[t["ep_per_cost_rank"] == 1, "customer_id"].item() == "C"


def test_scaled_sensitivity_parameters():
    assert econ.scaled(EC, "spread", 1.5).spread == pytest.approx(EC.spread * 1.5)
    assert econ.scaled(EC, "spread", 1.5).customer_rate == EC.customer_rate
    assert econ.scaled(EC, "incentive", 0.5).incentive == pytest.approx(20)
    assert econ.scaled(EC, "contact_cost", 2).channel_cost["mobile_call"] == pytest.approx(24)
    with pytest.raises(ValueError):
        econ.scaled(EC, "nonsense", 1.0)


# ----------------------------------------------------------------------------- segmentation (unit)
def test_segment_grid():
    df = pd.DataFrame({"propensity": [0.9, 0.9, 0.1, 0.1], "value": [200, 100, 200, 100]})
    s = seg.assign_segments(df, cut={"propensity": 0.5, "value": 150})
    assert s.tolist() == ["High value", "Growing", "Dormant", "Low engagement"]


def test_rfm_scores_bounds():
    df = pd.DataFrame({"recency_bucket": ["never", ">365d", "181-365d", "91-180d", "<=90d"],
                       "previous": [0, 1, 2, 4, 30], "balance_quintile": [1, 2, 3, 4, 5],
                       "prior_success": [0, 0, 0, 0, 1]})
    r = seg.rfm_scores(df)
    assert r["r_score"].tolist() == [1, 2, 3, 4, 5]
    assert r["f_score"].tolist() == [1, 2, 3, 4, 5]
    assert r["engagement_score"].iloc[0] == 0 and r["engagement_score"].iloc[-1] == 100
    assert r["rfm_code"].iloc[-1] == "555"


# ----------------------------------------------------------------------------- campaign table (integration)
@pytest.fixture(scope="module")
def table():
    return pl.campaign_table()


def test_every_customer_has_ep_and_one_segment(table):
    assert len(table) == cfg.UNIVERSE_SIZE
    assert table["expected_profit"].notna().all()
    assert table["segment"].notna().all()
    counts = table["segment"].value_counts()
    assert set(counts.index) == set(cfg.SEGMENTS)
    assert counts.sum() == cfg.UNIVERSE_SIZE


def test_no_principal_in_campaign_value(table):
    max_v = table["deposit"] * cfg.ECONOMICS.spread * cfg.ECONOMICS.term_years
    assert (table["value"] <= max_v + 1e-9).all()
    assert table["value"].max() < table["deposit"].min()


def test_budget_binds_by_default(table):
    b = econ.budget_check(table)
    assert b["budget_binds"], b
    assert b["cost_after_channel_caps"] > cfg.CAMPAIGN_BUDGET


def test_rfm_and_engagement_ranges(table):
    for c in ("r_score", "f_score", "m_score"):
        assert table[c].between(1, 5).all()
    assert table["engagement_score"].between(0, 100).all()


def test_top_decile_and_sensitivity(table):
    top = econ.top_decile_share(table)
    assert 0 < top["share"] <= 1
    base = table.drop(columns=econ.ECON_COLUMNS)
    s = econ.sensitivity(base, factors=(0.5, 1.0, 1.5)).set_index(["parameter", "factor"])["total_positive_ep"]
    assert s["spread", 0.5] < s["spread", 1.0] < s["spread", 1.5]
    assert s["incentive", 0.5] > s["incentive", 1.5]
    assert s["contact_cost", 0.5] > s["contact_cost", 1.5]
