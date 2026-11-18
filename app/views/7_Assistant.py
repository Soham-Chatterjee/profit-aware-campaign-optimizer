"""Screen 8: Campaign Assistant. Plain-language help for the people who make the contacts."""
import pandas as pd
import streamlit as st

import common as cm
from pacopt import assistant as asst

s = cm.sidebar()
t = cm.campaign_table(s.to_json())
comp = cm.optimisers(s)

cm.page_header("Campaign Assistant", "Ask about your contact list in plain English: who to call first, and why.")

strategy = "Genetic Algorithm" if comp.has_ga else "Greedy"
if not comp.has_ga:
    st.warning("The Genetic Algorithm has not been run for these settings, so the assistant is using the Greedy list. "
               "Press **Run GA** on the Campaign Optimization page to use the optimised list.", icon="⚠️")
cl = asst.CampaignList.build(t, comp, strategy, s.constraints, cm.metrics(), cm.driver_table)

prov = asst.provider()   # configured in the environment or .env; keys are never entered or shown in the UI
st.caption((f"AI mode ({prov.model}): ask anything about the campaign, its limits, the model or how the list was chosen."
            if prov else "Basic mode: answers come from fixed rules over the same list.")
           + f" Working from the {strategy} list of {len(cl.rows):,} customers.")

EXAMPLES = ["Give me the top 10 customers I should contact", "Top 5 customers for phone calls",
            "Who needs manager approval before I contact them?", "Which campaign limits are tight?",
            "Why is this list better than Greedy?", "How accurate is the prediction model?"]
msgs = st.session_state.setdefault("assistant_msgs", [])

picked = None
for row in (EXAMPLES[:3], EXAMPLES[3:]):
    for col, ex in zip(st.columns(len(row)), row):
        if col.button(ex, key=f"ex_{ex}", width="stretch"):
            picked = ex
if msgs and st.button("Clear chat", icon="🧹"):
    msgs.clear()
    st.rerun()


def show_rows(rows: list[dict], key: str) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch", height=min(60 + 35 * len(df), 420),
                 column_order=["customer_id", "channel", "segment", "chance_of_yes", "expected_profit", "contact_cost",
                               "needs_manager_review", "why"],
                 column_config={"customer_id": "Customer", "channel": "Channel", "segment": "Segment",
                                "chance_of_yes": st.column_config.NumberColumn("Chance of yes", format="percent"),
                                "expected_profit": st.column_config.NumberColumn("Expected profit", format=cm.FMT_GBP2),
                                "contact_cost": st.column_config.NumberColumn("Contact cost", format=cm.FMT_GBP2),
                                "needs_manager_review": st.column_config.CheckboxColumn("Manager approval"),
                                "why": "Why"})
    st.download_button("Download this list (CSV)", df.to_csv(index=False).encode(), file_name="my_contact_list.csv",
                       mime="text/csv", icon="⬇️", key=f"dl_{key}")


for i, m in enumerate(msgs):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("rows"):
            show_rows(m["rows"], str(i))

question = st.chat_input("Ask about your contact list…") or picked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    with st.chat_message("assistant"):
        with st.spinner("Looking at your list…"):
            ans = asst.ask(cl, question, history)
        st.markdown(ans.text)
        if ans.rows:
            show_rows(ans.rows, str(len(msgs) + 1))
    msgs += [{"role": "user", "content": question}, {"role": "assistant", "content": ans.text, "rows": ans.rows}]

st.divider()
st.caption("AI-generated from the campaign list above. It can read the list but cannot change it. Customers aged 75+ "
           "need a manager's approval before contact.")
cm.disclaimer()
