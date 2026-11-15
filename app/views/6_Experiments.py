"""Screen 7: Experiments (MLflow). Reads the local tracking store written by the pipeline and the dashboard."""
import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from pacopt import charts as ch
from pacopt import tracking

cm.page_header("Experiments", "MLflow: every model and optimiser run, with the parameters and metrics behind each number.")
c = tracking.client()


@st.cache_data(ttl=10, show_spinner=False)
def runs(experiment: str) -> pd.DataFrame:
    if c.get_experiment_by_name(experiment) is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time DESC"])


@st.cache_data(show_spinner=False)
def artifact_csv(run_id: str, path: str) -> pd.DataFrame:
    return pd.read_csv(c.download_artifacts(run_id, path))


@st.cache_data(show_spinner=False)
def history(run_id: str, metric: str) -> pd.DataFrame:
    h = c.get_metric_history(run_id, metric)
    return pd.DataFrame({"step": [x.step for x in h], "value": [x.value for x in h]}).sort_values("step")


model_runs, optim_runs, drift_runs = runs(tracking.EXP_MODEL), runs(tracking.EXP_OPTIM), runs(tracking.EXP_DRIFT)
if model_runs.empty and optim_runs.empty:
    st.info("No runs yet. Run `uv run python -m pacopt.pipeline` to train the models and log them to MLflow.", icon="ℹ️")
    st.stop()

st.link_button("Open MLflow UI", tracking.UI_URL, icon="🔗")


CATEGORICAL_PREFIXES = ("age_band", "recency_bucket", "job", "marital", "education", "contact", "poutcome")


def pretty(name: str) -> str:
    """'age_band_45-59' -> 'age band: 45-59'; numeric features just lose their underscores."""
    for p in CATEGORICAL_PREFIXES:
        if name.startswith(p + "_"):
            return f"{p.replace('_', ' ')}: {name[len(p) + 1:]}"
    return name.replace("_", " ")


def short(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    keep = {k: v for k, v in cols.items() if k in df}
    out = df[list(keep)].rename(columns=keep)
    if "Started" in out:
        out["Started"] = pd.to_datetime(out["Started"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    return out


# ----------------------------------------------------------------------------- champion card
st.markdown("##### Registered propensity model")
versions = c.search_model_versions(f"name='{tracking.CHAMPION}'")
if versions:
    v = max(versions, key=lambda x: int(x.version))
    run = c.get_run(v.run_id)
    mt = run.data.metrics
    k = st.columns(5)
    k[0].metric("Model", tracking.CHAMPION)
    k[1].metric("Version", f"v{v.version}")
    k[2].metric("ROC-AUC", f"{mt.get('roc_auc', float('nan')):.3f}")
    k[3].metric("PR-AUC", f"{mt.get('pr_auc', float('nan')):.3f}")
    k[4].metric("Max calibration gap", f"{mt.get('max_calibration_gap_pp', float('nan')):.2f} pp")
    st.caption(f"Run: {run.data.tags.get('mlflow.runName', '')} (`{v.run_id[:8]}`). Registered each time the pipeline retrains.")
else:
    st.caption("No model registered yet.")

tab_models, tab_ga, tab_runs, tab_drift = st.tabs(["Model drivers", "GA convergence", "All runs", "Drift checks"])

# ----------------------------------------------------------------------------- model drivers
with tab_models:
    kids = model_runs[model_runs["tags.model_key"].notna()] if "tags.model_key" in model_runs else pd.DataFrame()
    if kids.empty:
        st.info("No model runs logged yet.")
    else:
        latest = kids.sort_values("start_time", ascending=False).drop_duplicates("tags.model_key")
        cmp_ = latest.set_index("tags.mlflow.runName")[[c_ for c_ in ("metrics.roc_auc", "metrics.pr_auc", "metrics.brier",
                                                                      "metrics.max_calibration_gap_pp") if c_ in latest]]
        st.dataframe(cmp_.rename(columns=lambda x: x.replace("metrics.", "")), width="stretch")
        pick = st.radio("Model", list(latest["tags.mlflow.runName"]), horizontal=True)
        rid = latest.set_index("tags.mlflow.runName").loc[pick, "run_id"]
        perm = artifact_csv(rid, "feature_importance/permutation_importance.csv").head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=perm["importance"], y=perm["feature"].map(pretty), orientation="h", marker_color=ch.BLUE,
                               error_x=dict(array=perm["std"], color=ch.MUTED)))
        fig.update_xaxes(title="Drop in PR-AUC when the feature is shuffled")
        fig.update_yaxes(automargin=True)
        cm.show(ch.style(fig, f"{pick}: top drivers (permutation importance)", height=460), key="perm", note=False)
        if pick == "Logistic Regression":
            coef = artifact_csv(rid, "feature_importance/lr_coefficients.csv").head(15).sort_values("coef")
            fig = go.Figure(go.Bar(x=coef["coef"], y=coef["feature"].map(pretty), orientation="h",
                                   marker_color=[ch.BLUE if x > 0 else ch.RED for x in coef["coef"]]))
            fig.update_xaxes(title="Coefficient (positive raises the chance of subscribing)")
            fig.update_yaxes(automargin=True)
            cm.show(ch.style(fig, "Logistic Regression: 15 strongest coefficients", height=500), key="coef", note=False)

# ----------------------------------------------------------------------------- GA convergence
with tab_ga:
    ga = optim_runs[optim_runs["tags.is_ga"] == "True"] if "tags.is_ga" in optim_runs else pd.DataFrame()
    if ga.empty:
        st.info("No GA runs yet. Press **Run GA** on the Campaign Optimization or What-if page, or run the pipeline.")
    else:
        ga = ga.assign(label=ga["tags.source"] + " | " + pd.to_datetime(ga["start_time"]).dt.tz_localize(None).dt.strftime("%m-%d %H:%M")
                       + " | £" + (ga["metrics.expected_contribution"] / 1e3).round(1).astype(str) + "K")
        chosen = st.multiselect("GA runs to overlay", ga["label"], default=list(ga["label"].head(1)))
        metric = st.radio("Curve", ["best_fitness", "mean_fitness", "feasible_share"], horizontal=True,
                          format_func=lambda x: {"best_fitness": "Best expected contribution", "mean_fitness": "Population mean",
                                                 "feasible_share": "Share of feasible lists"}[x])
        fig = go.Figure()
        palette = [ch.BLUE, ch.ORANGE, ch.RED, ch.MUTED]
        for i, lab in enumerate(chosen):
            rid = ga.loc[ga["label"] == lab, "run_id"].iloc[0]
            h = history(rid, metric)
            fig.add_scatter(x=h["step"], y=h["value"], name=lab, line=dict(color=palette[i % 4], width=2.5))
        fig.update_xaxes(title="Generation")
        fig.update_yaxes(automargin=True, title_standoff=12)
        if metric == "feasible_share":
            fig.update_yaxes(title="Share of lists that meet every constraint", tickformat=".0%")
        else:
            fig.update_yaxes(title="Expected profit of the list (£)", tickprefix="£", tickformat=",.0f")
        cm.show(ch.style(fig, "Genetic algorithm: performance by generation", height=400, showlegend=True), key="ga_curve", note=False)
        st.caption("Each point is the best (or average) list in that generation. The curve flattens when the GA has converged.")

# ----------------------------------------------------------------------------- all runs
with tab_runs:
    exp = st.radio("Experiment", [tracking.EXP_MODEL, tracking.EXP_OPTIM, tracking.EXP_DRIFT], horizontal=True)
    df = {tracking.EXP_MODEL: model_runs, tracking.EXP_OPTIM: optim_runs, tracking.EXP_DRIFT: drift_runs}[exp]
    if df.empty:
        st.info("No runs in this experiment yet.")
    else:
        base = {"tags.mlflow.runName": "Run", "tags.source": "Source", "start_time": "Started"}
        metric_cols = [x for x in df.columns if x.startswith("metrics.") and "fitness" not in x][:8]
        st.dataframe(short(df, {**base, **{x: x.replace("metrics.", "") for x in metric_cols}}),
                     hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- drift checks
with tab_drift:
    if drift_runs.empty:
        st.info("No drift checks logged. Use **Log this check to MLflow** on the Monitoring page.")
    else:
        cols = {"start_time": "Started", "params.age_shift_years": "Age shift", "params.balance_change_pct": "Balance %",
                "params.unknown_contact_pct": "Contact lost %", "params.prior_success_change_pct": "Prior success %",
                "params.contacted_before_pct": "Contacted before %", "params.mix_shift_pct": "Mix shift %",
                "params.loan_stress_pct": "Loan stress %", "metrics.n_drifted_columns": "Features drifted",
                "metrics.share_drifted_columns": "Share drifted", "metrics.prediction_drift": "Prediction drift"}
        st.dataframe(short(drift_runs, cols), hide_index=True, width="stretch")
cm.disclaimer()
