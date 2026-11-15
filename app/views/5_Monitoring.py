"""Screen 6: Monitoring (Evidently). Data drift on SIMULATED data: there is no production feed."""
from dataclasses import astuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import monitoring as mo
from pacopt import tracking

cm.page_header("Monitoring", "Evidently checks: has the data the model scores moved away from the data it was trained on?")
st.warning(f"**{mo.SIMULATION_LABEL}.** There is no production pipeline, so the drift below is dialled in with the sliders: "
           "the reference is the training data, the current data is the campaign universe with your shifts applied. "
           "With every slider at zero the two samples are the same population (the baseline).", icon="🧪")

DEFAULTS = {"drift_age": 0, "drift_balance": 0, "drift_contact": 0, "drift_prior": 0, "drift_fatigue": 0,
            "drift_mix": 0, "drift_loan": 0}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------- controls
st.markdown("**Customer profile**")
c1, c2, c3, c4 = st.columns([3, 3, 3, 1.2], vertical_alignment="bottom")
age = c1.slider("Customers older by (years)", -10, 25, key="drift_age")
bal = c2.slider("Balances change by (%)", -80, 300, step=10, key="drift_balance")
mix = c3.slider("Customer mix shift (%)", 0, 60, step=5, key="drift_mix",
                help="Share of working customers (blue-collar, admin, services, technician) who become students or retirees")
c4.button("Reset", on_click=_reset, width="stretch")
st.markdown("**Contact and campaign history**")
d1, d2, d3, d4 = st.columns(4)
unk = d1.slider("Cellular contacts lost (%)", 0, 80, step=5, key="drift_contact",
                help="Share of cellular customers whose contact channel becomes 'unknown'")
pri = d2.slider("Previous-campaign successes change (%)", -100, 200, step=10, key="drift_prior",
                help="Relative change in customers whose previous campaign succeeded (down: they become failures)")
fat = d3.slider("Never-contacted now contacted before (%)", 0, 80, step=5, key="drift_fatigue",
                help="Contact fatigue: share of never-contacted customers who have now been contacted 1-3 times")
loan = d4.slider("Loan stress (%)", 0, 60, step=5, key="drift_loan",
                 help="More customers with a housing or personal loan, and a few more credit defaults")
settings = mo.DriftSettings(float(age), float(bal), float(unk), float(pri), float(fat), float(mix), float(loan))


@st.cache_data(show_spinner="Running Evidently…")
def drift(values: tuple) -> mo.DriftResult:
    return mo.run_drift(mo.DriftSettings(*values))


r = drift(astuple(settings))

# ----------------------------------------------------------------------------- verdict
if r.dataset_drift:
    st.error(f"**Dataset drift detected:** {r.n_drifted} of {r.n_columns} features drifted "
             f"({r.share_drifted:.0%}). Evidently flags the dataset when at least half the features drift.", icon="🚨")
elif r.n_drifted:
    st.warning(f"**Partial drift:** {r.n_drifted} of {r.n_columns} features drifted ({r.share_drifted:.0%}), "
               "below the dataset-drift threshold of 50%.", icon="⚠️")
else:
    st.success(f"**No drift:** 0 of {r.n_columns} features drifted." + (" This is the baseline." if settings.is_baseline else ""),
               icon="✅")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Features drifted", f"{r.n_drifted} / {r.n_columns}")
k2.metric("Prediction drift", "Yes" if r.prediction_drift else "No", help=f"{r.prediction_test}: {r.prediction_score:.3f}")
k3.metric("Mean propensity (reference)", f"{r.ref_scores.mean():.1%}")
k4.metric("Mean propensity (current)", f"{r.cur_scores.mean():.1%}", f"{(r.cur_scores.mean() - r.ref_scores.mean()) * 100:+.1f} pp")

# ----------------------------------------------------------------------------- per-feature table + chart
left, right = st.columns([1, 1.4])
with left:
    st.markdown("##### Drift by feature")
    st.dataframe(r.summary.rename(columns={"drift_score": "score"}), hide_index=True, width="stretch", height=420,
                 column_config={"score": st.column_config.NumberColumn(format="%.3f"),
                                "drifted": st.column_config.CheckboxColumn("drifted")})
    st.caption("Score is the test statistic (distance: higher means more drift). Test per column is chosen by Evidently.")
with right:
    feat = st.selectbox("Compare reference and current for", r.summary["feature"],
                        help="Features are sorted with the drifted ones first")
    fig = go.Figure()
    if r.reference[feat].dtype == object:
        d = pd.DataFrame({"Reference": r.reference[feat].value_counts(normalize=True),
                          "Current": r.current[feat].value_counts(normalize=True)}).fillna(0).sort_values("Reference", ascending=False)
        fig.add_bar(x=d.index, y=d["Reference"], name="Reference", marker_color=ch.BLUE)
        fig.add_bar(x=d.index, y=d["Current"], name="Current", marker_color=ch.ORANGE)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Share of customers", tickformat=".0%")
    else:
        for name, s_, col in (("Reference", r.reference[feat], ch.BLUE), ("Current", r.current[feat], ch.ORANGE)):
            fig.add_histogram(x=s_, name=name, marker_color=col, opacity=0.65, histnorm="probability",
                              xbins=dict(size=None) if s_.nunique() > 12 else None)
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(title="Share of customers", tickformat=".0%")
    cm.show(ch.style(fig, f"{feat}: reference vs current", height=420, showlegend=True), key="drift_feature", note=False)

# ----------------------------------------------------------------------------- prediction drift
st.markdown("##### Prediction drift: the propensity scores the model produces")
fig = go.Figure()
for name, vals, col in (("Reference", r.ref_scores, ch.BLUE), ("Current", r.cur_scores, ch.ORANGE)):
    fig.add_histogram(x=vals, name=name, marker_color=col, opacity=0.65, histnorm="probability", nbinsx=40)
fig.update_layout(barmode="overlay")
fig.update_xaxes(title="Propensity P_i", tickformat=".0%")
fig.update_yaxes(title="Share of customers", tickformat=".0%")
cm.show(ch.style(fig, "Distribution of propensity scores", height=340, showlegend=True), key="drift_pred", note=False)
st.caption("If the inputs drift, the scores feed straight into expected profit (P_i x V_i), so a score shift is a "
           "shift in the campaign's expected return.")

# ----------------------------------------------------------------------------- full reports + MLflow
with st.expander("Full Evidently data drift report"):
    st.iframe(r.html, height=800)
with st.expander("Evidently data quality report"):
    qkey = astuple(settings)
    if st.button("Generate data quality report", key="quality_btn"):
        with st.spinner("Running Evidently data quality checks…"):
            st.session_state["quality_report"] = (qkey, mo.quality_html(r))
    saved = st.session_state.get("quality_report")
    if saved and saved[0] == qkey:
        st.iframe(saved[1], height=800)
    elif saved:
        st.caption("The sliders changed since the last report. Generate it again for the current settings.")

st.divider()
if st.button("Log this check to MLflow", icon="📝", help="Saves the settings, drift metrics and the Evidently report as a run"):
    run_id = tracking.log_drift(mo.settings_params(settings), r.metrics(), r.html, r.summary)
    if run_id:
        st.success(f"Logged to MLflow experiment `{tracking.EXP_DRIFT}` (run `{run_id[:8]}`). Open the Experiments page to browse runs.")
    else:
        st.error("MLflow logging failed; see the terminal for the warning.")
cm.disclaimer()
