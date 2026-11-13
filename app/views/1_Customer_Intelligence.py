"""Screen 1: Customer Intelligence (plan 5.3)."""
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import segmentation as seg

s = cm.sidebar()
t = cm.campaign_table(s.to_json())

cm.page_header("Customer Intelligence",
               "Engine 1: four business segments from a transparent propensity × value grid.")

counts = t["segment"].value_counts().reindex(list(cfg.SEGMENTS))
st.markdown("##### Number of customers by segment")
cm.show(ch.reference_bars(counts.index, counts.values, hover_label="customers"), key="seg_bars")
cut = seg.thresholds(t)
st.caption(f"High propensity = top 40% of P_i (P ≥ {cut['propensity']:.1%}); high value = V_i at or above "
           f"the median (£{cut['value']:,.0f}). Counts sum to {counts.sum():,}.")

st.markdown("##### Segment profile")
prof = seg.segment_profile(t).reset_index()
st.dataframe(
    prof, hide_index=True,
    column_order=["segment", "customers", "share", "avg_age", "avg_balance", "avg_propensity", "avg_value",
                  "avg_expected_profit", "pct_eligible", "pct_positive_ep", "pct_never_contacted", "avg_engagement"],
    column_config={
        "segment": "Segment", "customers": st.column_config.NumberColumn("Customers", format="localized"),
        "share": st.column_config.NumberColumn("Share", format="percent"),
        "avg_age": st.column_config.NumberColumn("Avg age", format="%.1f"),
        "avg_balance": st.column_config.NumberColumn("Avg balance", format=cm.FMT_GBP),
        "avg_propensity": st.column_config.NumberColumn("Avg P_i", format="percent"),
        "avg_value": st.column_config.NumberColumn("Avg V_i", format=cm.FMT_GBP),
        "avg_expected_profit": st.column_config.NumberColumn("Avg EP_i", format=cm.FMT_GBP2),
        "pct_eligible": st.column_config.NumberColumn("% eligible", format="percent"),
        "pct_positive_ep": st.column_config.NumberColumn("% positive EP", format="percent"),
        "pct_never_contacted": st.column_config.NumberColumn("% never contacted", format="percent"),
        "avg_engagement": st.column_config.ProgressColumn("Avg engagement", min_value=0, max_value=100, format="%.0f"),
    })

left, right = st.columns([3, 2])
with left:
    st.markdown("##### Value by propensity: the four quadrants")
    fig = go.Figure()
    for name, color in zip(cfg.SEGMENTS, ch.SEGMENT_COLORS):
        d = t[t["segment"] == name]
        fig.add_scattergl(x=d["propensity"], y=d["value"], mode="markers", name=f"{name} ({len(d):,})",
                          marker=dict(color=color, size=5, opacity=0.55),
                          customdata=d[["customer_id", "expected_profit"]],
                          hovertemplate="%{customdata[0]}<br>P %{x:.1%}<br>V £%{y:,.0f}<br>EP £%{customdata[1]:,.2f}"
                                        f"<extra>{name}</extra>")
    fig.add_vline(x=cut["propensity"], line=dict(color=ch.MUTED, dash="dash", width=1.5))
    fig.add_hline(y=cut["value"], line=dict(color=ch.MUTED, dash="dash", width=1.5))
    ch.style(fig, height=440)
    fig.update_xaxes(title="Propensity P_i", tickformat=".0%")
    ticks = [v for v in (50, 100, 150, 200, 300, 500, 1_000, 2_000, 5_000) if t["value"].min() * 0.8 <= v <= t["value"].max() * 1.2]
    fig.update_yaxes(title="Contribution value V_i (£, log scale)", type="log", tickvals=ticks,
                     ticktext=[f"£{v:,}" for v in ticks])
    fig.update_layout(legend=dict(y=1.02, yanchor="bottom"), margin=dict(t=70))
    cm.show(fig, key="quadrants")
with right:
    st.markdown("##### Eligibility (responsible targeting)")
    rules = {"Credit default": "elig_default", "Negative balance": "elig_negative_balance",
             "Contact fatigue": "elig_fatigue"}
    rows = [{"rule": k, "customers": int(t[c].sum()), "treatment": "Excluded"} for k, c in rules.items()]
    rows.append({"rule": "Aged 75+", "customers": int(t["flag_age_75plus"].sum()),
                 "treatment": f"Flagged; max {s.constraints.protected_max_share:.0%} of the list"})
    rows.append({"rule": "Eligible for the campaign", "customers": int(t["eligible"].sum()), "treatment": ""})
    st.dataframe(rows, hide_index=True, column_config={
        "rule": "Rule", "customers": st.column_config.NumberColumn("Customers", format="localized"),
        "treatment": "Treatment"})
    st.caption("A customer can meet more than one exclusion rule. Toggle the rules in the sidebar.")
    elig_seg = t.groupby("segment", observed=False)["eligible"].agg(["sum", "size"])
    elig_seg["ineligible"] = elig_seg["size"] - elig_seg["sum"]
    st.dataframe(elig_seg.rename(columns={"sum": "eligible", "size": "customers"})[["customers", "eligible", "ineligible"]]
                 .reset_index(), hide_index=True,
                 column_config={"segment": "Segment",
                                **{c: st.column_config.NumberColumn(c.capitalize(), format="localized")
                                   for c in ("customers", "eligible", "ineligible")}})

st.markdown("##### RFM and engagement overlays")
ov = t.groupby("segment", observed=False)[["r_score", "f_score", "m_score", "engagement_score"]].mean()
st.dataframe(ov.reset_index(), hide_index=True, column_config={
    "segment": "Segment", "r_score": st.column_config.NumberColumn("Avg R (1-5)", format="%.2f"),
    "f_score": st.column_config.NumberColumn("Avg F (1-5)", format="%.2f"),
    "m_score": st.column_config.NumberColumn("Avg M (1-5)", format="%.2f"),
    "engagement_score": st.column_config.ProgressColumn("Avg engagement", min_value=0, max_value=100, format="%.1f")})
st.caption("R from recency of the last campaign contact, F from previous contacts, M from the balance quintile. "
           "Overlays only: they don't enter EP or the optimiser.")
cm.disclaimer()
