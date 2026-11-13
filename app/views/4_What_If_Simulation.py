"""Screen 4: What-if Simulation (plan 5.6)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import optimize as op
from pacopt import scenario as sc

base_s = cm.sidebar()

cm.page_header("What-if Simulation",
               "Change the budget, channel costs, spread and incentive here without touching the global "
               "scenario, and compare the result with it.")

# ----------------------------------------------------------------------------- local inputs
with st.container(border=True):
    st.markdown("**What-if inputs** (start from the sidebar scenario)")
    c = st.columns(6)
    budget = c[0].number_input("Marketing budget (£)", 1_000.0, 500_000.0, float(base_s.constraints.budget), 5_000.0, format="%.0f")
    cm_ = c[1].number_input("Mobile call £", 0.0, 100.0, float(base_s.economics.channel_cost["mobile_call"]), 0.5)
    cl_ = c[2].number_input("Landline call £", 0.0, 100.0, float(base_s.economics.channel_cost["landline_call"]), 0.5)
    cd_ = c[3].number_input("Digital £", 0.0, 100.0, float(base_s.economics.channel_cost["digital"]), 0.5)
    spread = c[4].slider("Funding spread (%)", 0.0, 5.0, float(round(base_s.economics.spread * 100, 2)), 0.05,
                         help="Changed through the FTP rate; the customer rate stays fixed")
    incentive = c[5].number_input("Incentive (£)", 0.0, 1_000.0, float(base_s.economics.incentive), 5.0)

econ = replace(base_s.economics, ftp_rate=base_s.economics.customer_rate + spread / 100, incentive=incentive,
               channel_cost={"mobile_call": cm_, "landline_call": cl_, "digital": cd_})
w = replace(base_s, economics=econ, constraints=replace(base_s.constraints, budget=budget))
wt = cm.campaign_table(w.to_json())
wb = cm.baselines(w.to_json())
bt = cm.campaign_table(base_s.to_json())
bb = cm.baselines(base_s.to_json())

# ----------------------------------------------------------------------------- headline comparison
g_w, g_b = wb.table.loc["Greedy"], bb.table.loc["Greedy"]
k = st.columns(4)
k[0].metric("Expected contribution (Greedy)", cm.gbp(g_w["expected_contribution"]), border=True,
            delta=f"{cm.gbp(g_w['expected_contribution'] - g_b['expected_contribution'])} vs scenario")
k[1].metric("Contacts", f"{g_w['contacts']:,}", border=True, delta=f"{g_w['contacts'] - g_b['contacts']:+,}")
k[2].metric("Spend", cm.gbp(g_w["spend"]), border=True, delta=f"{cm.gbp(budget - g_w['spend'])} unspent", delta_color="off")
k[3].metric("Customers with positive EP", f"{int((wt[wt.eligible == 1].expected_profit > 0).sum()):,}", border=True,
            delta=f"{int((wt[wt.eligible == 1].expected_profit > 0).sum() - (bt[bt.eligible == 1].expected_profit > 0).sum()):+,}")
st.caption("Greedy (highest EP per £) gives an instant answer and is within a fraction of a percent of the optimum "
           "in the default scenario. Use the GA re-run below to optimise this what-if under every constraint.")

# ----------------------------------------------------------------------------- affordable vs actual
st.markdown("##### Maximum affordable contacts vs actual allocation")
aff = sc.affordable_contacts(wt, w.constraints, wb.selections["Greedy"])
aff = aff.rename(index=lambda i: cm.channel_label(i))
st.dataframe(aff.reset_index(), hide_index=True, column_config={
    "channel": "Channel", "channel_group": "Group",
    "cost_per_contact": st.column_config.NumberColumn("Cost per contact", format=cm.FMT_GBP2),
    "max_affordable_budget_only": st.column_config.NumberColumn("Max affordable (budget ÷ cost)", format="localized",
                                                                help="Budget-only upper bound if the whole budget went to this channel"),
    "eligible_customers": st.column_config.NumberColumn("Eligible", format="localized"),
    "positive_ep_customers": st.column_config.NumberColumn("Positive EP", format="localized"),
    "actual_allocation": st.column_config.NumberColumn("Actual allocation", format="localized")})
st.caption(f"The actual allocation also respects the {w.constraints.max_contacts:,}-contact cap, channel capacity "
           f"(calls {w.constraints.channel_capacity['calls']:,}, digital {w.constraints.channel_capacity['digital']:,}), "
           "segment limits and eligibility. The 'All channels' row uses the cheapest channel for the upper bound.")

# ----------------------------------------------------------------------------- profit vs budget
st.markdown("##### Profit vs budget")


@st.cache_data(show_spinner="Computing the profit curve…")
def curve(scn_json: str, upto: float):
    s_ = sc.Scenario.from_json(scn_json)
    t_ = cm.campaign_table(scn_json)
    grid = np.unique(np.r_[np.linspace(0, upto, 41)[1:], s_.constraints.budget])
    return sc.budget_curve(t_, s_.constraints, grid), sc.saturation_point(t_, s_.constraints)


sat_probe = sc.saturation_point(wt, w.constraints)
upto = max(budget * 2, sat_probe["spend"] * 1.25, 10_000)
cv, sat = curve(w.to_json(), float(upto))
fig = go.Figure()
fig.add_scatter(x=cv["budget"], y=cv["expected_contribution"], mode="lines", name="Expected contribution",
                line=dict(color=ch.BLUE, width=2), customdata=cv[["contacts", "spend"]],
                hovertemplate="budget £%{x:,.0f}<br>contribution £%{y:,.0f}<br>%{customdata[0]:,} contacts, spend £%{customdata[1]:,.0f}<extra></extra>")
cur = cv.loc[(cv["budget"] - budget).abs().idxmin()]
fig.add_scatter(x=[budget], y=[cur["expected_contribution"]], mode="markers", name="This budget",
                marker=dict(color=ch.ORANGE, size=11, line=dict(color=ch.SURFACE, width=2)),
                hovertemplate="this budget £%{x:,.0f}<br>£%{y:,.0f}<extra></extra>")
fig.add_vline(x=sat["spend"], line=dict(color=ch.MUTED, dash="dash", width=1.5))
fig.add_annotation(x=sat["spend"], y=sat["expected_contribution"] * 0.5, text=f"Saturation £{sat['spend']:,.0f}",
                   showarrow=False, xanchor="left", xshift=6, font=dict(color=ch.MUTED, size=11))
ch.style(fig, "Greedy expected contribution as the budget grows (other limits unchanged)", height=420)
xt = np.arange(0, upto + 1, ch._nice_step(upto, 8))
fig.update_xaxes(title="Budget", tickvals=xt, ticktext=["£0" if v == 0 else f"£{v / 1e3:g}K" for v in xt], range=[0, upto])
ch.set_value_axis(fig, cv["expected_contribution"].max() * 1.12, money=True)
fig.update_yaxes(title="Expected contribution")
cm.show(fig, key="budget_curve")
marg = cv.set_index("budget")["expected_contribution"].diff() / cv.set_index("budget").index.to_series().diff()
st.markdown(
    f"Beyond **£{sat['spend']:,.0f}** extra budget adds nothing: limited by **{sat['limited_by']}** "
    f"({sat['contacts']:,} contacts, {cm.gbp(sat['expected_contribution'])}). "
    f"At £{budget:,.0f} the marginal return is about **£{np.interp(budget, marg.index[1:], marg.values[1:]):.2f} per extra £**.")

# ----------------------------------------------------------------------------- GA re-run
st.markdown("##### Optimise this what-if with the GA")
runs = st.session_state.setdefault("opt_runs", {})
if st.button("Run GA on this what-if", type="primary", icon="🧬"):
    cm.run_optimisers(w, run_ga=True, run_milp=False)
res = cm.optimisers(w)
if res.has_ga:
    tbl = res.table.loc[[n for n in ("Random", "Propensity", "Greedy", "Genetic Algorithm") if n in res.table.index]]
    st.dataframe(tbl[["contacts", "spend", "expected_contribution", "roi", "incremental_vs_random", "feasible"]].reset_index(),
                 hide_index=True, column_config={
                     "strategy": "Strategy", "contacts": st.column_config.NumberColumn("Contacts", format="localized"),
                     "spend": st.column_config.NumberColumn("Spend", format=cm.FMT_GBP),
                     "expected_contribution": st.column_config.NumberColumn("Exp. contribution", format=cm.FMT_GBP),
                     "roi": st.column_config.NumberColumn("ROI", format="%.2f"),
                     "incremental_vs_random": st.column_config.NumberColumn("vs Random", format="£%+,.0f"),
                     "feasible": st.column_config.CheckboxColumn("All constraints met")})
    st.caption(f"GA: {res.ga.generations} generations in {res.ga.runtime_s:.1f}s "
               f"(capped at {cm.GA_TIME_LIMIT_S:.0f}s).")
else:
    st.caption("Not run yet for these what-if inputs.")
cm.disclaimer()
