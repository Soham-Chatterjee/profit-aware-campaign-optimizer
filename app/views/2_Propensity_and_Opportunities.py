"""Screen 2: Propensity and Revenue Opportunities (plan 5.4)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import model as m

s = cm.sidebar()
t = cm.campaign_table(s.to_json())

cm.page_header("Propensity & Revenue Opportunities",
               "Q1: who is likely to subscribe. Q2/Q3: what each customer is worth, and the most worth "
               "spending to reach them.")

# ----------------------------------------------------------------------------- customer table
st.markdown("##### Customer ranking")
c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
q = c1.text_input("Search customer ID", placeholder="e.g. C100123")
segs = c2.multiselect("Segments", list(cfg.SEGMENTS), default=list(cfg.SEGMENTS))
sort_by = c3.selectbox("Sort by", ["Propensity", "Expected profit", "EP per £", "Break-even cost"])
detail = c4.toggle("Economics", value=False, help="Show channel, V_i, C_i, break-even cost and EP_i")

d = t[t["segment"].astype(str).isin(segs)]
if q:
    d = d[d["customer_id"].str.contains(q.strip(), case=False, regex=False)]
sort_col = {"Propensity": "propensity", "Expected profit": "expected_profit", "EP per £": "ep_per_cost",
            "Break-even cost": "break_even_cost"}[sort_by]
d = d.sort_values(sort_col, ascending=False).assign(channel_name=lambda x: x["channel"].map(cm.channel_label),
                                                    eligible_flag=lambda x: x["eligible"].astype(bool),
                                                    propensity_pct=lambda x: x["propensity"] * 100)
base_cols = ["customer_id", "propensity_pct", "segment"]
extra = ["channel_name", "value", "contact_cost", "break_even_cost", "expected_profit", "ep_per_cost", "eligible_flag"]
colcfg = {
    "customer_id": "Customer",
    "propensity_pct": st.column_config.ProgressColumn("Propensity", min_value=0.0, max_value=100.0, format="%.1f%%"),
    "segment": "Segment", "channel_name": "Channel",
    "value": st.column_config.NumberColumn("Value V_i", format=cm.FMT_GBP2),
    "contact_cost": st.column_config.NumberColumn("Cost C_i", format=cm.FMT_GBP2),
    "break_even_cost": st.column_config.NumberColumn("Break-even P×V", format=cm.FMT_GBP2,
                                                     help="Most worth spending to contact this customer (Q3)"),
    "expected_profit": st.column_config.NumberColumn("Expected profit EP_i", format=cm.FMT_GBP2),
    "ep_per_cost": st.column_config.NumberColumn("EP per £", format="%.2f"),
    "eligible_flag": st.column_config.CheckboxColumn("Eligible"),
}
st.dataframe(d, hide_index=True, height=380, column_order=base_cols + (extra if detail else []), column_config=colcfg)
st.caption(f"{len(d):,} of {len(t):,} customers shown. Click a column header to re-sort.")
st.download_button("Download this table (CSV)",
                   d[["customer_id", "propensity", "segment"] + extra].to_csv(index=False).encode(),
                   file_name="customer_ranking.csv", mime="text/csv", icon="⬇️")

# ----------------------------------------------------------------------------- top-N opportunities
st.markdown("##### Top opportunities (eligible customers)")
c1, c2 = st.columns([1, 3])
n = c1.number_input("Top N", 10, 2_000, 50, step=10)
rank_by = c1.radio("Rank by", ["Expected profit", "EP per £"], horizontal=False)
e = t[t["eligible"] == 1]
top = e.nlargest(int(n), "expected_profit" if rank_by == "Expected profit" else "ep_per_cost")
total_pos = e["expected_profit"].clip(lower=0).sum()
with c1:
    st.metric(f"Expected profit of the top {int(n):,}", cm.gbp(top["expected_profit"].sum()),
              delta=f"{top['expected_profit'].sum() / total_pos:.0%} of all positive EP" if total_pos else None,
              delta_color="off")
    st.metric("Contact cost", cm.gbp(top["contact_cost"].sum()))
with c2:
    st.dataframe(top.assign(channel_name=lambda x: x["channel"].map(cm.channel_label),
                            propensity_pct=lambda x: x["propensity"] * 100), hide_index=True, height=330,
                 column_order=["customer_id", "segment", "channel_name", "propensity_pct", "value", "contact_cost",
                               "break_even_cost", "expected_profit", "ep_per_cost"],
                 column_config=colcfg)

# ----------------------------------------------------------------------------- model performance
st.markdown("##### Model performance (held-out test set)")
mt = cm.metrics()
ev = mt[s.model]
k = st.columns(5)
k[0].metric("ROC-AUC", f"{ev['roc_auc']:.3f}", border=True)
k[1].metric("PR-AUC", f"{ev['pr_auc']:.3f}", border=True, help=f"Base rate {ev['base_rate']:.1%}")
k[2].metric("Precision@1,000", f"{ev['precision_at_k']['K=1000']:.1%}", border=True,
            delta=f"{ev['precision_at_k']['K=1000'] / ev['base_rate']:.1f}× lift", delta_color="off")
k[3].metric("Brier score", f"{ev['brier']:.4f}", border=True)
k[4].metric("Max calibration gap", f"{ev['max_calibration_gap_pp']:.2f} pp", border=True)
st.caption(f"Showing **{ev['name']}** (the sidebar model). Recommended: {mt['recommended'].upper()}: {mt['reason']}.")

preds = cm.test_predictions()
y = preds["y"].to_numpy()
tabs = st.tabs(["ROC & PR curves", "Calibration", "Precision@K", "Top drivers"])
with tabs[0]:
    l, r = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    for key, name, color in (("lr", "Logistic Regression", ch.BLUE), ("rf", "Random Forest", ch.ORANGE)):
        c = m.curves(y, preds[f"p_{key}"].to_numpy())
        roc.add_scatter(x=c["roc"].fpr, y=c["roc"].tpr, name=f"{name} (AUC {mt[key]['roc_auc']:.3f})",
                        line=dict(color=color, width=2), hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")
        pr.add_scatter(x=c["pr"].recall, y=c["pr"].precision, name=f"{name} (AP {mt[key]['pr_auc']:.3f})",
                       line=dict(color=color, width=2), hovertemplate="recall %{x:.2f}<br>precision %{y:.1%}<extra></extra>")
    ch.reference_line(roc, 0, 0, 1, 1, "random", label_at="start")
    ch.style(roc, "ROC curve", height=400)
    roc.update_xaxes(title="False positive rate", range=[0, 1]); roc.update_yaxes(title="True positive rate", range=[0, 1])
    ch.reference_line(pr, 0, ev["base_rate"], 1, ev["base_rate"], f"base rate {ev['base_rate']:.1%}", label_at="start")
    ch.style(pr, "Precision-recall curve", height=400)
    pr.update_xaxes(title="Recall", range=[0, 1]); pr.update_yaxes(title="Precision", range=[0, 1], tickformat=".0%")
    with l:
        cm.show(roc, key="roc")
    with r:
        cm.show(pr, key="pr")
with tabs[1]:
    fig = go.Figure()
    rows = {}
    for key, name, color in (("lr", "Logistic Regression", ch.BLUE), ("rf", "Random Forest", ch.ORANGE)):
        cal = m.calibration_table(y, preds[f"p_{key}"].to_numpy())
        rows[name] = cal["gap_pp"]
        fig.add_scatter(x=cal["predicted"], y=cal["actual"], name=name, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=ch.SURFACE, width=2)),
                        hovertemplate="predicted %{x:.1%}<br>actual %{y:.1%}<extra>" + name + "</extra>")
    ch.reference_line(fig, 0, 0, 0.5, 0.5, "perfect calibration")
    ch.style(fig, "Calibration by decile", height=420)
    fig.update_xaxes(title="Mean predicted probability", tickformat=".0%", range=[0, 0.5])
    fig.update_yaxes(title="Observed subscription rate", tickformat=".0%", range=[0, 0.5])
    cm.show(fig, key="cal")
    st.dataframe(pd.DataFrame(rows).T.round(2), column_config={c: st.column_config.NumberColumn(f"Decile {c}", format="%.2f pp")
                                                             for c in range(1, 11)})
with tabs[2]:
    pk = pd.DataFrame({mt[k_]["name"]: mt[k_]["precision_at_k"] for k_ in ("lr", "rf")})
    pk.index = [i if i != "K=budget" else f"K=budget ({mt[s.model]['budget_k']:,} contacts)" for i in pk.index]
    pk["base rate"] = ev["base_rate"]
    st.dataframe(pk, column_config={c: st.column_config.NumberColumn(c, format="percent") for c in pk.columns})
    st.caption("Precision@K on the 10,000-customer campaign universe: share of actual subscribers among the K "
               "highest-scored customers. K=budget is the number of contacts £50K buys in propensity order.")
with tabs[3]:
    b = cm.model_bundle()
    coef = m.lr_coefficients(b["lr_explain"]).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=coef["coef"], y=coef["feature"], orientation="h",
                           marker=dict(color=[ch.BLUE if c > 0 else ch.RED for c in coef["coef"]], cornerradius=3),
                           customdata=coef["odds_ratio"],
                           hovertemplate="%{y}<br>coefficient %{x:.3f}<br>odds ratio %{customdata:.2f}<extra></extra>"))
    ch.style(fig, "Logistic Regression: 15 largest coefficients (blue raises, red lowers propensity)",
             height=480, showlegend=False)
    fig.update_xaxes(title="Coefficient (log-odds; numeric features per 1 SD)")
    cm.show(fig, key="coef")
    st.caption("The LR is isotonic-calibrated. Calibration is monotone, so these directions hold for "
               "the scores. Random Forest permutation importance is in notebook 02.")
cm.disclaimer()
