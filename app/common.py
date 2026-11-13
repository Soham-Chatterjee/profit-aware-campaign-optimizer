"""Shared dashboard plumbing: global sidebar (scenario inputs), cached data, shared widgets.

Every number on every page comes from these cached calls into `pacopt`; nothing is hard-coded.
Fast work (campaign table, Random/Propensity/Greedy/Rule-based) recomputes on every change.
Slow work (GA, MILP) runs only on a button press, or is loaded from the pipeline artifacts when
the scenario is identical to the one they were computed for (fingerprint match).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import model as m
from pacopt import optimize as op
from pacopt import pipeline as pl
from pacopt import scenario as sc

GA_TIME_LIMIT_S = 30.0          # dashboard re-runs (D-038): ~99.9% of the optimum is reached by then
MILP_TIME_LIMIT_S = 120.0
CHANNEL_LABEL = {"mobile_call": "Mobile call", "landline_call": "Landline call", "digital": "Digital"}
FMT_GBP = "£%,.0f"
FMT_GBP2 = "£%,.2f"


# ----------------------------------------------------------------------------- page chrome
def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def disclaimer() -> None:
    st.caption(f"⚠️ {ch.PROTOTYPE_NOTE} Balances are euros in the UCI source data, shown as £ for "
               "illustration. Source: UCI Bank Marketing (CC BY 4.0), a 2008-2010 Portuguese campaign.")


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displayModeBar": False})
    st.caption(ch.PROTOTYPE_NOTE)


# ----------------------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner=False)
def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pl.load_universe(), pl.load_scored()


@st.cache_data(show_spinner="Recomputing customer economics…")
def campaign_table(scenario_json: str) -> pd.DataFrame:
    universe, scored = _inputs()
    return sc.build_table(sc.Scenario.from_json(scenario_json), universe, scored)


@st.cache_data(show_spinner="Running the baseline strategies…")
def baselines(scenario_json: str) -> op.Comparison:
    s = sc.Scenario.from_json(scenario_json)
    return op.compare_strategies(campaign_table(scenario_json), s.constraints, run_genetic=False, run_milp=False)


@st.cache_data(show_spinner=False)
def saved_optimisers(scenario_json: str):
    """GA/MILP results from the pipeline artifacts, only when computed for this exact problem."""
    s = sc.Scenario.from_json(scenario_json)
    return pl.load_optimisation(campaign_table(scenario_json), s.constraints)


@st.cache_resource(show_spinner=False)
def model_bundle() -> dict:
    return m.load_model()


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    import json
    return json.loads(m.METRICS_PATH.read_text())


@st.cache_data(show_spinner="Scoring the held-out test set…")
def test_predictions() -> pd.DataFrame:
    _, test, _ = m.load_processed()
    b = model_bundle()
    return pd.DataFrame({"y": test[cfg.TARGET].to_numpy(), "p_lr": m.predict(b["lr"], test),
                         "p_rf": m.predict(b["rf"], test)})


# ----------------------------------------------------------------------------- scenario sidebar
_DEFAULTS = {
    "budget": cfg.CONSTRAINTS.budget, "max_contacts": cfg.CONSTRAINTS.max_contacts,
    "cap_calls": cfg.CONSTRAINTS.channel_capacity["calls"], "cap_digital": cfg.CONSTRAINTS.channel_capacity["digital"],
    "cost_mobile": cfg.ECONOMICS.channel_cost["mobile_call"], "cost_landline": cfg.ECONOMICS.channel_cost["landline_call"],
    "cost_digital": cfg.ECONOMICS.channel_cost["digital"],
    "ftp_rate": cfg.ECONOMICS.ftp_rate * 100, "customer_rate": cfg.ECONOMICS.customer_rate * 100,
    "deposit_min": cfg.ECONOMICS.deposit_min, "deposit_alpha": cfg.ECONOMICS.deposit_alpha,
    "deposit_max": cfg.ECONOMICS.deposit_max, "term_years": cfg.ECONOMICS.term_years,
    "incentive": cfg.ECONOMICS.incentive, "onboarding": cfg.ECONOMICS.onboarding_cost,
    "organic_rate": cfg.ECONOMICS.organic_rate * 100,
    "excl_default": cfg.ELIGIBILITY.exclude_default, "excl_negative": cfg.ELIGIBILITY.exclude_negative_balance,
    "excl_fatigue": cfg.ELIGIBILITY.exclude_fatigue, "fatigue_max": cfg.ELIGIBILITY.fatigue_max_previous,
    "protected_cap": cfg.CONSTRAINTS.protected_max_share * 100,
    "seg_max": cfg.CONSTRAINTS.segment_max_share["High value"] * 100,
    "min_growing": cfg.CONSTRAINTS.segment_min_share["Growing"] * 100,
    "min_dormant": cfg.CONSTRAINTS.segment_min_share["Dormant"] * 100,
    "model": cfg.MODEL.selected,
}


# Widget keys ("s_*") belong to the page that renders them, and Streamlit drops a widget's state when
# you leave that page. The scenario therefore lives in persistent keys ("p_*"): each page copies them
# into its widgets before rendering, and every widget writes back on change (D-038).
def _init_state() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(f"p_{k}", v)
        st.session_state[f"s_{k}"] = st.session_state[f"p_{k}"]


def _sync(k: str) -> None:
    st.session_state[f"p_{k}"] = st.session_state[f"s_{k}"]


def _reset() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[f"p_{k}"] = v
        st.session_state[f"s_{k}"] = v


def _w(widget, label: str, *args, key: str, **kw):
    """A sidebar widget bound to persistent scenario key `key`."""
    return widget(label, *args, key=f"s_{key}", on_change=_sync, args=(key,), **kw)


def scenario_from_state() -> sc.Scenario:
    s = {k: st.session_state[f"p_{k}"] for k in _DEFAULTS}
    econ = replace(cfg.ECONOMICS, ftp_rate=s["ftp_rate"] / 100, customer_rate=s["customer_rate"] / 100,
                   deposit_min=float(s["deposit_min"]), deposit_alpha=float(s["deposit_alpha"]),
                   deposit_max=float(s["deposit_max"]), term_years=float(s["term_years"]),
                   incentive=float(s["incentive"]), onboarding_cost=float(s["onboarding"]),
                   organic_rate=s["organic_rate"] / 100,
                   channel_cost={"mobile_call": float(s["cost_mobile"]), "landline_call": float(s["cost_landline"]),
                                 "digital": float(s["cost_digital"])})
    elig = replace(cfg.ELIGIBILITY, exclude_default=bool(s["excl_default"]),
                   exclude_negative_balance=bool(s["excl_negative"]), exclude_fatigue=bool(s["excl_fatigue"]),
                   fatigue_max_previous=int(s["fatigue_max"]))
    cons = replace(cfg.CONSTRAINTS, budget=float(s["budget"]), max_contacts=int(s["max_contacts"]),
                   channel_capacity={"calls": int(s["cap_calls"]), "digital": int(s["cap_digital"])},
                   segment_max_share={k: s["seg_max"] / 100 for k in cfg.SEGMENTS},
                   segment_min_share={"Growing": s["min_growing"] / 100, "Dormant": s["min_dormant"] / 100},
                   protected_max_share=s["protected_cap"] / 100)
    return sc.Scenario(economics=econ, eligibility=elig, constraints=cons, model=s["model"])


def sidebar() -> sc.Scenario:
    """Global assumptions, rendered by every page; the values persist across pages."""
    st.set_page_config(page_title="Profit-Aware Campaign Optimizer", page_icon="📈", layout="wide")
    _init_state()
    sb = st.sidebar
    sb.header("Scenario assumptions")
    sb.caption("Illustrative defaults. Every page recomputes from these.")
    ni, sl, tg = st.number_input, st.slider, st.toggle
    with sb.expander("Campaign budget and capacity", expanded=True):
        _w(ni, "Campaign budget (£)", 1_000.0, 500_000.0, step=5_000.0, format="%.0f", key="budget")
        _w(ni, "Maximum contacts", 100, 20_000, step=250, key="max_contacts")
        c1, c2 = st.columns(2)
        with c1:
            _w(ni, "Call capacity", 0, 20_000, step=250, key="cap_calls", help="Mobile + landline calls")
        with c2:
            _w(ni, "Digital capacity", 0, 20_000, step=250, key="cap_digital")
    with sb.expander("Channel costs (fully loaded, £ per contact)"):
        c1, c2, c3 = st.columns(3)
        for col, lab, k in ((c1, "Mobile", "cost_mobile"), (c2, "Landline", "cost_landline"), (c3, "Digital", "cost_digital")):
            with col:
                _w(ni, lab, 0.0, 100.0, step=0.5, key=k)
    with sb.expander("Product economics"):
        _w(sl, "FTP rate (%)", 0.0, 10.0, step=0.05, key="ftp_rate",
           help="Transfer-pricing rate credited to the deposit (Treasury input)")
        _w(sl, "Customer rate (%)", 0.0, 10.0, step=0.05, key="customer_rate")
        spread = st.session_state["p_ftp_rate"] - st.session_state["p_customer_rate"]
        (st.caption if spread > 0 else st.warning)(f"Funding spread: {spread:.2f}%")
        _w(ni, "Minimum deposit D_min (£)", 0.0, 1e6, step=1_000.0, format="%.0f", key="deposit_min")
        _w(ni, "Deposit multiple of balance (alpha)", 0.0, 10.0, step=0.1, key="deposit_alpha")
        _w(ni, "Maximum deposit D_max (£)", 0.0, 1e7, step=10_000.0, format="%.0f", key="deposit_max")
        _w(ni, "Term (years)", 0.25, 10.0, step=0.25, key="term_years")
        c1, c2 = st.columns(2)
        with c1:
            _w(ni, "Incentive (£)", 0.0, 1_000.0, step=5.0, key="incentive")
        with c2:
            _w(ni, "Onboarding (£)", 0.0, 1_000.0, step=1.0, key="onboarding")
        _w(sl, "Organic rate P0 (%)", 0.0, 20.0, step=0.5, key="organic_rate",
           help="Share who would subscribe without contact. 0 = no control group.")
    with sb.expander("Eligibility and responsible targeting"):
        _w(tg, "Exclude credit default", key="excl_default")
        _w(tg, "Exclude negative balance", key="excl_negative")
        _w(tg, "Exclude contact fatigue", key="excl_fatigue")
        _w(ni, "Fatigue: max previous contacts", 0, 100, key="fatigue_max",
           disabled=not st.session_state["p_excl_fatigue"])
        _w(sl, "Max share aged 75+ (%)", 0.0, 10.0, step=0.5, key="protected_cap")
    with sb.expander("Segment limits (share of the list)"):
        _w(sl, "Max share of any segment (%)", 25.0, 100.0, step=5.0, key="seg_max")
        _w(sl, "Min share: Growing (%)", 0.0, 40.0, step=1.0, key="min_growing")
        _w(sl, "Min share: Dormant (%)", 0.0, 40.0, step=1.0, key="min_dormant")
    with sb.expander("Propensity model"):
        _w(st.radio, "Model driving P_i", ["lr", "rf"], key="model", horizontal=True,
           format_func=lambda k: {"lr": "Logistic Regression (primary)", "rf": "Random Forest (challenger)"}[k])
    sb.button("Reset to defaults", on_click=_reset, width="stretch")
    s = scenario_from_state()
    if s != sc.DEFAULT:
        sb.info("Scenario differs from the default.")
    return s


# ----------------------------------------------------------------------------- optimisation state
def optimisers(scenario: sc.Scenario) -> op.Comparison:
    """Baselines plus whatever GA/MILP results exist for this scenario (session run, or saved)."""
    key = scenario.to_json()
    base = baselines(key)
    runs = st.session_state.setdefault("opt_runs", {})
    ga_res, mres = runs.get(key, (None, None))
    if ga_res is None:
        saved = saved_optimisers(key)
        if saved is not None:
            ga_res, mres = saved
    return op.with_optimisers(campaign_table(key), base, ga_res, mres)


def run_optimisers(scenario: sc.Scenario, run_ga: bool = True, run_milp: bool = False,
                   seeded: bool = True) -> None:
    key = scenario.to_json()
    t = campaign_table(key)
    pr = op.build_problem(t, scenario.constraints)
    runs = st.session_state.setdefault("opt_runs", {})
    prev_ga, prev_m = runs.get(key, (None, None))
    if prev_ga is None and (saved := saved_optimisers(key)) is not None:
        prev_ga, prev_m = saved
    ga_res, mres = prev_ga, prev_m
    if run_ga:
        bar = st.progress(0.0, text="Genetic algorithm: generation 0")
        gcfg = replace(cfg.GA, time_limit_s=GA_TIME_LIMIT_S, greedy_seeds=cfg.GA.greedy_seeds if seeded else 0)

        def progress(g, best):
            bar.progress(min(g / gcfg.generations, 1.0),
                         text=f"Genetic algorithm: generation {g}, best £{best:,.0f}" if np.isfinite(best) else
                         f"Genetic algorithm: generation {g}")
        ga_res = op.run_ga(pr, gcfg, progress)
        bar.empty()
    if run_milp:
        with st.spinner("Solving the exact optimum (MILP)…"):
            mres = op.milp_optimum(pr, replace(cfg.OPTIM, milp_time_limit=MILP_TIME_LIMIT_S))
    runs[key] = (ga_res, mres)


# ----------------------------------------------------------------------------- shared formatting
def gbp(v: float, k: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"£{v / 1e3:,.1f}K" if k else f"£{v:,.0f}"


def channel_label(c: str) -> str:
    return CHANNEL_LABEL.get(c, c)
