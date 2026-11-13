"""Screen 3: Campaign Optimization (plan 5.5)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import evaluation as ev
from pacopt import optimize as op

s = cm.sidebar()
t = cm.campaign_table(s.to_json())

cm.page_header("Campaign Optimization",
               "Q4: the best combination of customers. Q5: how much more profit than the existing strategies.")

# ----------------------------------------------------------------------------- run controls
comp = cm.optimisers(s)
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.4, 3])
run_ga = c1.button("Run GA", type="primary", icon="🧬", width="stretch")
run_milp = c2.button("Exact optimum", icon="🎯", width="stretch",
                     help=f"MILP with scipy/HiGHS; up to {cm.MILP_TIME_LIMIT_S:.0f}s")
seeded = c3.toggle("Seed GA with Greedy", value=True, help="4 of 100 starting lists = the Greedy list")
if run_ga or run_milp:
    cm.run_optimisers(s, run_ga=run_ga, run_milp=run_milp, seeded=seeded)
    comp = cm.optimisers(s)
with c4:
    if comp.has_ga:
        g = comp.ga
        st.success(f"GA: {g.generations} generations in {g.runtime_s:.1f}s"
                   + (f" | exact optimum solved in {comp.milp.runtime_s:.0f}s" if comp.has_milp else
                      " | exact optimum not solved for this scenario"), icon="✅")
    else:
        st.info("This scenario differs from the saved default run. Press **Run GA** to optimise it "
                f"(capped at {cm.GA_TIME_LIMIT_S:.0f}s). Baselines below are already up to date.", icon="ℹ️")
tb = comp.table

# ----------------------------------------------------------------------------- headline chart
st.markdown("##### Expected contribution by strategy")
names = [n for n in op.CHART_STRATEGIES if n in tb.index]
fig = ch.reference_bars(names, tb.loc[names, "expected_contribution"], money=True)
if comp.has_milp:
    opt_ep = comp.milp.expected_profit
    ch.reference_line(fig, -0.5, opt_ep, len(names) - 0.5, opt_ep,
                      f"Theoretical max (MILP) £{opt_ep / 1e3:,.1f}K", label_at="start")
    ch.set_value_axis(fig, max(opt_ep, tb.loc[names, "expected_contribution"].max()) * 1.15, money=True)
cm.show(fig, key="strategy_bars")
if "Genetic Algorithm" in tb.index:
    gai = tb.loc["Genetic Algorithm"]
    msg = (f"The GA list earns **{cm.gbp(gai['incremental_vs_random'])} ({gai['uplift_vs_random']:+.0%})** more than "
           f"Random and **{cm.gbp(gai['expected_contribution'] - tb.loc['Propensity', 'expected_contribution'])}** more "
           f"than Propensity-only targeting.")
    if comp.has_milp:
        msg += (f" It reaches **{gai['pct_of_optimum']:.2%}** of the proven optimum; Greedy (profit per £) "
                f"reaches {tb.loc['Greedy', 'pct_of_optimum']:.2%}.")
    st.markdown(msg)

# ----------------------------------------------------------------------------- comparison table
st.markdown("##### Strategy comparison")
show_cols = ["contacts", "contacts_calls", "contacts_digital", "spend", "expected_conversions",
             "expected_contribution", "roi", "incremental_vs_random", "uplift_vs_random"]
if "pct_of_optimum" in tb:
    show_cols.append("pct_of_optimum")
show_cols.append("feasible")
st.dataframe(tb[show_cols].assign(**({"pct_of_optimum": tb["pct_of_optimum"] * 100} if "pct_of_optimum" in tb else {}))
             .reset_index(), hide_index=True, column_config={
    "strategy": "Strategy", "contacts": st.column_config.NumberColumn("Contacts", format="localized"),
    "contacts_calls": st.column_config.NumberColumn("Calls", format="%,.0f"),
    "contacts_digital": st.column_config.NumberColumn("Digital", format="%,.0f"),
    "spend": st.column_config.NumberColumn("Spend", format=cm.FMT_GBP),
    "expected_conversions": st.column_config.NumberColumn("Exp. conversions", format="%.0f"),
    "expected_contribution": st.column_config.NumberColumn("Exp. contribution", format=cm.FMT_GBP),
    "roi": st.column_config.NumberColumn("ROI", format="%.2f", help="Expected contribution / spend"),
    "incremental_vs_random": st.column_config.NumberColumn("vs Random", format="£%+,.0f"),
    "uplift_vs_random": st.column_config.NumberColumn("Uplift", format="percent"),
    "pct_of_optimum": st.column_config.NumberColumn("% of optimum", format="%.2f%%"),
    "feasible": st.column_config.CheckboxColumn("All constraints met"),
})
r = comp.random_runs["expected_contribution"]
st.caption(f"Random = mean of {len(r)} seeds (std {cm.gbp(r.std())}). Rule-based = highest balance first. "
           "Greedy = highest EP per £ first. All lists pass the same independent constraint check.")

# ----------------------------------------------------------------------------- selected list
strategies = [n for n in op.STRATEGY_ORDER if n in comp.selections]
default_ix = strategies.index("Genetic Algorithm") if "Genetic Algorithm" in strategies else strategies.index("Greedy")
pick = st.segmented_control("Inspect the list of", strategies, default=strategies[default_ix], key="pick_strategy")
pick = pick or strategies[default_ix]
ids = comp.selected_ids(pick)
if pick == "Random":
    st.caption("Random shows the seed-0 example list; the table above averages 50 seeds.")

tabs = st.tabs(["Constraint checklist", "Channel & segment mix", "Convergence", "Fairness monitor", "Target list"])
with tabs[0]:
    ck = op.check_constraints(t, ids, s.constraints)
    share = ck["constraint"].str.contains("share", case=False)
    ck["value_txt"] = np.where(share, (ck["value"] * 100).round(1).astype(str) + "%", ck["value"].map("{:,.0f}".format))
    ck["limit_txt"] = np.where(share, (ck["limit"] * 100).round(1).astype(str) + "%", ck["limit"].map("{:,.0f}".format))
    ck["status"] = np.where(ck["ok"], "✅ met", "❌ violated")
    st.dataframe(ck[["constraint", "value_txt", "limit_txt", "status"]], hide_index=True, column_config={
        "constraint": "Constraint", "value_txt": "This list", "limit_txt": "Limit", "status": "Status"})
with tabs[1]:
    seg_mix = op.selection_mix(t, comp, "segment")
    ch_mix = op.selection_mix(t, comp, "channel_group")
    l, r_ = st.columns(2)
    for col, mix, title in ((l, seg_mix, "Segment share of each list"), (r_, ch_mix, "Channel share of each list")):
        col.markdown(f"**{title}**")
        col.dataframe(mix, column_config={c: st.column_config.NumberColumn(c, format="percent") for c in mix.columns})
    sel = t[t["customer_id"].isin(ids)]
    fig = go.Figure()
    for name, color in zip(cfg.SEGMENTS, ch.SEGMENT_COLORS):
        d = sel[sel["segment"] == name].groupby("channel_group").size()
        fig.add_bar(y=d.index, x=d.values, name=name, orientation="h", marker=dict(color=color, line=dict(color=ch.SURFACE, width=2)),
                    hovertemplate=f"{name}<br>%{{y}}: %{{x:,}}<extra></extra>")
    ch.style(fig, f"{pick}: contacts by channel and segment", height=260)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Contacts", tickformat=",")
    cm.show(fig, key="mix")
with tabs[2]:
    if comp.has_ga:
        h = comp.ga.history
        fig = go.Figure()
        fig.add_scatter(x=h["generation"], y=h["best"], name="Best list", mode="lines", line=dict(color=ch.BLUE, width=2),
                        hovertemplate="generation %{x}<br>best £%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=h["generation"], y=h["mean"], name="Population mean", mode="lines",
                        line=dict(color=ch.ORANGE, width=1.5, dash="dot"),
                        hovertemplate="generation %{x}<br>mean £%{y:,.0f}<extra></extra>")
        lines = [(tb.loc["Greedy", "expected_contribution"], "Greedy")]
        if comp.has_milp:
            lines.append((comp.milp.expected_profit, "MILP optimum"))
        for yv, lab in lines:
            ch.reference_line(fig, 0, yv, int(h["generation"].max()), yv, lab, label_at="start" if lab == "Greedy" else "end")
        ch.style(fig, "GA convergence: best and mean fitness per generation", height=400)
        fig.update_xaxes(title="Generation")
        lo = np.nanmin(h["mean"].to_numpy()) if h["mean"].notna().any() else 0
        top = max(h["best"].max(), comp.milp.expected_profit if comp.has_milp else 0)
        fig.update_yaxes(title="Expected contribution", tickprefix="£", tickformat=",.0f",
                         range=[lo - 0.005 * abs(lo), top * 1.003])
        fig.update_layout(margin=dict(l=90))
        cm.show(fig, key="conv")
        st.caption(f"{'Greedy-seeded' if comp.ga.seeded else 'Unseeded'} run. With seeding, generation 0 already "
                   "contains the Greedy list, so the curve starts near the optimum. Switch seeding off to see the search.")
    else:
        st.info("Run the GA to see its convergence.")
with tabs[3]:
    fm = ev.fairness_monitor(t, ids)
    flagged = fm[fm["flag"] != ""]
    if len(flagged):
        st.warning(f"{len(flagged)} group(s) are targeted at under half or over twice the overall rate. "
                   "Reporting only: review before launch.", icon="⚠️")
    else:
        st.success("No group is targeted at under half or over twice the overall rate.", icon="✅")
    dim = st.radio("Group by", ["age_band", "job"], horizontal=True,
                   format_func={"age_band": "Age band", "job": "Job"}.get)
    f = fm[fm["dimension"] == dim]
    st.dataframe(f.drop(columns="dimension"), hide_index=True, column_config={
        "group": "Group", "eligible": st.column_config.NumberColumn("Eligible", format="localized"),
        "selected": st.column_config.NumberColumn("Selected", format="localized"),
        "share_of_eligible": st.column_config.NumberColumn("Share of eligible", format="percent"),
        "share_of_selected": st.column_config.NumberColumn("Share of list", format="percent"),
        "targeting_rate": st.column_config.NumberColumn("Targeting rate", format="percent"),
        "rate_ratio": st.column_config.NumberColumn("Rate vs overall", format="%.2f×"),
        "flag": "Flag"})
    st.caption("Targeting rate = selected / eligible in the group. Groups under 30 eligible customers are never "
               "flagged. Skews mostly follow propensity and balance; age is a protected characteristic, so the "
               "75+ group is capped, not excluded.")
with tabs[4]:
    if pick in comp.selections:
        tl = op.target_list(t, comp, pick).assign(channel_name=lambda x: x["channel"].map(cm.channel_label))
        st.dataframe(tl, hide_index=True, height=360,
                     column_order=["customer_id", "segment", "channel_name", "propensity", "value", "contact_cost",
                                   "break_even_cost", "expected_profit", "ep_per_cost", "flag_age_75plus"],
                     column_config={"customer_id": "Customer", "segment": "Segment", "channel_name": "Channel",
                                    "propensity": st.column_config.NumberColumn("Propensity", format="percent"),
                                    "value": st.column_config.NumberColumn("V_i", format=cm.FMT_GBP2),
                                    "contact_cost": st.column_config.NumberColumn("C_i", format=cm.FMT_GBP2),
                                    "break_even_cost": st.column_config.NumberColumn("Break-even", format=cm.FMT_GBP2),
                                    "expected_profit": st.column_config.NumberColumn("EP_i", format=cm.FMT_GBP2),
                                    "ep_per_cost": st.column_config.NumberColumn("EP per £", format="%.2f"),
                                    "flag_age_75plus": st.column_config.CheckboxColumn("75+ (review)")})
        st.download_button(f"Download the {pick} target list (CSV)", tl.drop(columns="channel_name").to_csv(index=False).encode(),
                           file_name=f"target_list_{pick.lower().replace(' ', '_')}.csv", mime="text/csv", icon="⬇️")
        st.caption("Customers aged 75+ are flagged for human review before contact.")
cm.disclaimer()
