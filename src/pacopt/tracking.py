"""MLflow experiment tracking (local only: sqlite store + local artifact folder).

Three experiments:
- `propensity-model`: one parent run per training, with a child run for Logistic Regression and one for
  Random Forest (params, metrics, feature importance, model; the selected model is registered as
  `propensity-champion`).
- `campaign-optimization`: one parent run per optimisation, with a child run per strategy (Random,
  Rule-based, Propensity, Greedy, GA, MILP). The GA run also carries its per-generation convergence.
- `data-drift`: one run per explicit "log this check" click on the Monitoring page.

Tracking never breaks the pipeline: every public function catches errors and warns instead.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from functools import wraps

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mlflow.entities import Metric
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from pacopt import charts as ch
from pacopt import config as cfg
from pacopt import model as m

TRACKING_URI = f"sqlite:///{(cfg.PROJECT_ROOT / 'mlruns.db').as_posix()}"
ARTIFACT_ROOT = cfg.PROJECT_ROOT / "mlartifacts"
EXP_MODEL = "propensity-model"
EXP_OPTIM = "campaign-optimization"
EXP_DRIFT = "data-drift"
CHAMPION = "propensity-champion"
UI_URL = "http://localhost:5000"
UI_COMMAND = "uv run mlflow ui --backend-store-uri sqlite:///mlruns.db"   # run from the repo root


def _safe(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - tracking must never break the pipeline or the dashboard
            warnings.warn(f"MLflow tracking skipped ({fn.__name__}): {e}")
            return None
    return wrapper


def client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(TRACKING_URI)


def experiment(name: str) -> str:
    """Get or create an experiment whose artifacts live under ./mlartifacts/<name> (stable path)."""
    c = client()
    exp = c.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return c.create_experiment(name, artifact_location=(ARTIFACT_ROOT / name).as_uri())


def _flat(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(f"{prefix}{k}.", v))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[f"{prefix}{k}"] = v
    return out


def _num(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool) and np.isfinite(v)}


def _bar(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    d = df.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=ch.BLUE))
    return ch.style(fig, title, height=420)


# ----------------------------------------------------------------------------- propensity models
def _model_metrics(ev: m.Evaluation) -> dict[str, float]:
    d = ev.as_dict()
    out = _num({k: v for k, v in d.items() if k != "precision_at_k"})
    out.update(_num({f"precision_at_{k.replace('=', '_')}": v for k, v in d["precision_at_k"].items()}))
    return out


@_safe
def log_training(res: m.ModelResult, test: pd.DataFrame, mc: cfg.ModelConfig = cfg.MODEL) -> str | None:
    """Log both propensity models. Returns the parent run id (linked from the optimisation runs)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = experiment(EXP_MODEL)
    cols = m.model_columns(res.lr)
    sample = test[cols].head(200)
    entries = [("lr", "Logistic Regression", res.lr, res.eval_lr),
               ("rf", "Random Forest (calibrated)", res.rf, res.eval_rf)]
    summary: dict[str, float] = {}
    with mlflow.start_run(experiment_id=exp, run_name="training") as parent:
        mlflow.set_tags({"selected_model": mc.selected, "recommended_model": res.recommended})
        mlflow.log_param("recommendation_reason", res.reason[:480])
        for key, label, model, ev in entries:
            with mlflow.start_run(experiment_id=exp, run_name=label, nested=True):
                mlflow.set_tags({"model_key": key, "selected": str(key == mc.selected)})
                mlflow.log_params({"model": label, "calibration": mc.lr_calibration, "n_features": len(cols),
                                   "cv_folds": mc.cv_folds, "random_state": cfg.RANDOM_STATE,
                                   "features": ", ".join(cols)[:480]})
                if key == "lr":
                    mlflow.log_param("best_C", res.lr_search.best_params_["clf__C"])
                    mlflow.log_metric("cv_pr_auc", float(res.lr_search.best_score_))
                model_metrics = _model_metrics(ev)
                mlflow.log_metrics(model_metrics)
                summary.update({f"{key}_{k}": v for k, v in model_metrics.items()})
                if key == mc.selected:
                    summary.update({f"selected_{k}": v for k, v in model_metrics.items()})
                # feature importance: permutation (both models) + coefficients (LR only)
                perm = m.permutation_drivers(model, test, mc=mc)
                mlflow.log_table(perm, "feature_importance/permutation_importance.json")
                mlflow.log_text(perm.to_csv(index=False), "feature_importance/permutation_importance.csv")
                mlflow.log_figure(_bar(perm, "importance", "feature", "Permutation importance (drop in PR-AUC)"),
                                  "feature_importance/permutation_importance.html")
                if key == "lr":
                    coef = m.lr_coefficients(res.lr_search)
                    mlflow.log_text(coef.to_csv(index=False), "feature_importance/lr_coefficients.csv")
                    mlflow.log_figure(_bar(coef.assign(abs_coef=coef["abs_coef"]), "abs_coef", "feature",
                                           "Logistic Regression: |coefficient| (per 1 SD / one-hot level)"),
                                      "feature_importance/lr_coefficients.html")
                sig = infer_signature(sample, model.predict_proba(sample)[:, 1])
                mlflow.sklearn.log_model(model, name="model", signature=sig, serialization_format="cloudpickle",
                                         registered_model_name=CHAMPION if key == mc.selected else None)
        mlflow.log_metrics(summary)   # both models side by side on the parent run
        mlflow.log_text(json.dumps({"selected": mc.selected, "reason": res.reason}, indent=2), "decision.json")
        return parent.info.run_id


def latest_training_run_id() -> str | None:
    c = client()
    exp = c.get_experiment_by_name(EXP_MODEL)
    if exp is None:
        return None
    runs = c.search_runs([exp.experiment_id], "tags.mlflow.runName = 'training'", max_results=1,
                         order_by=["attributes.start_time DESC"])
    return runs[0].info.run_id if runs else None


# ----------------------------------------------------------------------------- optimisation
def _scenario_params(scenario, extra: dict | None = None) -> dict:
    cc, ec = scenario.constraints, scenario.economics
    p = {"model": scenario.model, "budget": cc.budget, "max_contacts": cc.max_contacts,
         "capacity_calls": cc.channel_capacity["calls"], "capacity_digital": cc.channel_capacity["digital"],
         "protected_max_share": cc.protected_max_share, "ftp_rate": ec.ftp_rate, "customer_rate": ec.customer_rate,
         "incentive": ec.incentive, "onboarding": ec.onboarding_cost, "organic_rate": ec.organic_rate,
         **{f"cost_{k}": v for k, v in ec.channel_cost.items()}}
    p.update(extra or {})
    return p


def _ga_params(g: cfg.GAConfig, seeded: bool | None) -> dict:
    d = asdict(g)
    d["seeded"] = seeded
    return d


def _log_ga_history(history: pd.DataFrame) -> None:
    """Per-generation curves as stepped metrics (MLflow draws them and overlays runs)."""
    run_id = mlflow.active_run().info.run_id
    ts = int(pd.Timestamp.now().timestamp() * 1000)
    batch = []
    for col, name in (("best", "best_fitness"), ("mean", "mean_fitness"), ("feasible_share", "feasible_share")):
        if col in history:
            batch += [Metric(name, float(v), ts, int(g)) for g, v in zip(history["generation"], history[col])
                      if np.isfinite(v)]
    if batch:
        client().log_batch(run_id, metrics=batch)
    mlflow.log_text(history.to_csv(index=False), "ga_history.csv")


def convergence_figure(history: pd.DataFrame, optimum: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=history["generation"], y=history["best"], name="Best", line=dict(color=ch.BLUE, width=2.5))
    if "mean" in history:
        fig.add_scatter(x=history["generation"], y=history["mean"], name="Population mean (feasible)",
                        line=dict(color=ch.ORANGE, width=1.5))
    if optimum is not None and np.isfinite(optimum):
        fig.add_hline(y=optimum, line=dict(color=ch.MUTED, dash="dash"),
                      annotation_text=f"Exact optimum £{optimum / 1e3:,.1f}K", annotation_position="bottom right")
    fig.update_xaxes(title="Generation")
    fig.update_yaxes(title="Expected contribution (£)", tickprefix="£", separatethousands=True,
                     automargin=True, title_standoff=12)
    return ch.style(fig, "GA convergence", height=380, showlegend=True)


def _log_strategy(name: str, row: pd.Series, params: dict, parent_run_id: str | None, comp, table: pd.DataFrame,
                  exp: str, scenario, train_run_id: str | None, source: str) -> None:
    from pacopt import optimize as op  # local import: keeps module import light
    with mlflow.start_run(experiment_id=exp, run_name=name, nested=True):
        mlflow.set_tags({"strategy": name, "source": source, "is_ga": str(name == "Genetic Algorithm"),
                         **({"propensity_training_run": train_run_id} if train_run_id else {})})
        mlflow.log_params(params)
        mlflow.log_metrics(_num({k: (float(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in row.items()}))
        if name in comp.selections:
            ids = comp.selected_ids(name)
            chk = op.check_constraints(table, ids, scenario.constraints)
            mlflow.log_text(chk.to_csv(index=False), "constraint_checklist.csv")
        if name == "Genetic Algorithm" and comp.ga is not None:
            g = comp.ga
            mlflow.log_params(_ga_params(cfg.GA, g.seeded))
            mlflow.log_metrics({"generations": g.generations, "runtime_s": g.runtime_s})
            _log_ga_history(g.history)
            opt_ep = comp.milp.expected_profit if comp.milp is not None else None
            mlflow.log_figure(convergence_figure(g.history, opt_ep), "ga_convergence.html")
            mlflow.log_text(op.target_list(table, comp).to_csv(index=False), "ga_target_list.csv")


@_safe
def log_comparison(comp, table: pd.DataFrame, scenario, source: str = "pipeline",
                   train_run_id: str | None = None) -> str | None:
    """One parent run + one child run per strategy in `comp.table` (pipeline, full comparison)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = experiment(EXP_OPTIM)
    train_run_id = train_run_id or latest_training_run_id()
    params = _scenario_params(scenario)
    with mlflow.start_run(experiment_id=exp, run_name="comparison") as parent:
        mlflow.set_tags({"source": source, **({"propensity_training_run": train_run_id} if train_run_id else {})})
        mlflow.log_params(params)
        if "Genetic Algorithm" in comp.table.index:
            mlflow.log_metric("ga_expected_contribution", float(comp.table.loc["Genetic Algorithm",
                                                                               "expected_contribution"]))
        for name, row in comp.table.iterrows():
            _log_strategy(name, row, params, parent.info.run_id, comp, table, exp, scenario, train_run_id, source)
        return parent.info.run_id


@_safe
def log_ga_run(comp, table: pd.DataFrame, scenario, source: str = "dashboard") -> str | None:
    """Dashboard 'Run GA' click: one run for the GA only, tagged with its scenario."""
    if comp.ga is None or "Genetic Algorithm" not in comp.table.index:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = experiment(EXP_OPTIM)
    train_run_id = latest_training_run_id()
    params = _scenario_params(scenario)
    with mlflow.start_run(experiment_id=exp, run_name="Genetic Algorithm (dashboard)") as run:
        mlflow.set_tags({"strategy": "Genetic Algorithm", "source": source, "is_ga": "True",
                         **({"propensity_training_run": train_run_id} if train_run_id else {})})
        mlflow.log_params(params)
        mlflow.log_params(_ga_params(cfg.GA, comp.ga.seeded))
        mlflow.log_metrics(_num(comp.table.loc["Genetic Algorithm"].to_dict()))
        mlflow.log_metrics({"generations": comp.ga.generations, "runtime_s": comp.ga.runtime_s})
        _log_ga_history(comp.ga.history)
        opt_ep = comp.milp.expected_profit if comp.milp is not None else None
        mlflow.log_figure(convergence_figure(comp.ga.history, opt_ep), "ga_convergence.html")
        return run.info.run_id


# ----------------------------------------------------------------------------- drift
@_safe
def log_drift(params: dict, metrics: dict, report_html: str, summary: pd.DataFrame) -> str | None:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = experiment(EXP_DRIFT)
    with mlflow.start_run(experiment_id=exp, run_name="simulated-drift") as run:
        mlflow.set_tag("simulated", "True")
        mlflow.log_params(params)
        mlflow.log_metrics(_num(metrics))
        mlflow.log_text(report_html, "evidently_drift_report.html")
        mlflow.log_text(summary.to_csv(index=False), "drift_by_column.csv")
        return run.info.run_id
