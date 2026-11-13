"""Home: Revenue AI Command Centre (plan 5.2)."""
import streamlit as st

import common as cm
from pacopt import config as cfg

s = cm.sidebar()
t = cm.campaign_table(s.to_json())
comp = cm.optimisers(s)
tb = comp.table

cm.page_header("PACOPT Command Centre",
               "Profit-aware targeting for a retail term-deposit campaign: who to contact, through which "
               "channel, and how much more profit that earns than the existing strategies.")

best = "Genetic Algorithm" if comp.has_ga else "Greedy"
sel = t.loc[comp.problem.table_index[comp.selections[best]]]
rnd = tb.loc["Random", "expected_contribution"]
ep = tb.loc[best, "expected_contribution"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers analysed", f"{len(t):,}", border=True,
          help="Campaign universe: held-out customers scored out-of-sample.")
k2.metric("Average conversion propensity", f"{sel['propensity'].mean():.1%}", border=True,
          delta=f"{sel['propensity'].mean() - t['propensity'].mean():+.1%} vs all customers", delta_color="off",
          help=f"Mean P_i of the {best} target list.")
k3.metric("Campaign budget", cm.gbp(s.constraints.budget, k=True), border=True,
          delta=f"{cm.gbp(tb.loc[best, 'spend'], k=True)} used", delta_color="off")
k4.metric("Expected contribution", cm.gbp(ep, k=True), border=True,
          delta=f"{cm.gbp(ep - rnd, k=True)} vs Random ({ep / rnd - 1:+.0%})" if rnd else None,
          help=f"{best} list. Incremental = vs the Random baseline (proxy for an untargeted campaign).")
if not comp.has_ga:
    st.info("The genetic algorithm has not been run for this scenario, so the tiles show the Greedy list. "
            "Run it on **Campaign Optimization**.", icon="ℹ️")

st.subheader("The five business questions")
elig = t[t["eligible"] == 1]
top_seg = sel["segment"].value_counts().idxmax()
answers = [
    ("Q1 · Who is likely to subscribe?",
     f"Calibrated propensity P_i for every customer. The top 1,000 convert at "
     f"{cm.metrics()[s.model]['precision_at_k']['K=1000']:.0%} vs a {cm.metrics()[s.model]['base_rate']:.1%} base rate.",
     "views/2_Propensity_and_Opportunities.py"),
    ("Q2 · Who has the greatest value?",
     f"{(t['segment'] == 'High value').sum():,} High value customers. The top 10% by expected profit hold "
     f"{elig['expected_profit'].clip(lower=0).nlargest(max(len(elig) // 10, 1)).sum() / max(elig['expected_profit'].clip(lower=0).sum(), 1e-9):.0%} "
     "of all positive expected profit.",
     "views/1_Customer_Intelligence.py"),
    ("Q3 · How much should we spend per customer?",
     f"At most the break-even cost P_i × V_i. The median eligible customer is worth "
     f"{cm.gbp(elig['break_even_cost'].median())} of contact, and "
     f"{(elig['expected_profit'] > 0).mean():.0%} are worth more than their channel cost.",
     "views/4_What_If_Simulation.py"),
    ("Q4 · Which combination to target?",
     f"{best}: {len(sel):,} customers ({(sel['channel_group'] == 'calls').sum():,} calls, "
     f"{(sel['channel_group'] == 'digital').sum():,} digital), mostly {top_seg}, within every constraint.",
     "views/3_Campaign_Optimization.py"),
    ("Q5 · How much extra profit?",
     f"{cm.gbp(ep - rnd)} ({ep / rnd - 1:+.0%}) more than Random and "
     f"{cm.gbp(ep - tb.loc['Propensity', 'expected_contribution'])} more than Propensity-only targeting"
     + (f"; {tb.loc[best, 'pct_of_optimum']:.2%} of the proven optimum." if comp.has_milp else "."),
     "views/3_Campaign_Optimization.py"),
]
for (q, a, page), col in zip(answers, st.columns(5)):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{q}**")
        st.write(a)
        st.page_link(page, label="Open", icon="➡️")

st.subheader("How it works")
st.markdown(
    "1. **Customer Intelligence**: RFM-inspired features, a calibrated propensity model and four "
    "segments.\n"
    "2. **Profit Intelligence**: the deposit is a funding product. Value = deposit × funding spread "
    "− incentive − onboarding, and expected profit = P_i × V_i − channel cost. The principal is never "
    "counted as revenue.\n"
    "3. **Optimization Intelligence**: a genetic algorithm picks the list under budget, contact, channel, "
    "segment and responsible-targeting limits, and is benchmarked against the exact optimum.")
with st.expander("Current scenario"):
    st.json({"budget": s.constraints.budget, "max_contacts": s.constraints.max_contacts,
             "channel_capacity": s.constraints.channel_capacity, "channel_cost": s.economics.channel_cost,
             "funding_spread": f"{s.economics.spread:.2%}", "incentive": s.economics.incentive,
             "deposit_min": s.economics.deposit_min, "model": s.model,
             "eligible_customers": int(t['eligible'].sum())}, expanded=False)
cm.disclaimer()
