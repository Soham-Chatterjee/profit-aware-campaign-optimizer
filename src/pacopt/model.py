"""Engine 1 (part): propensity models, evaluation metrics and explainability.

- Logistic Regression (primary, D-006): one-hot + scaling, C tuned on PR-AUC, NO class weighting
  (class weights distort probabilities, and P_i is multiplied by money in EP_i). Wrapped in a
  cross-validated isotonic calibration (D-027): monotone, so the ranking and coefficient
  explanations are preserved, while the top-decile under-prediction is corrected.
- Random Forest (challenger): isotonic-calibrated, used as a benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from joblib import parallel_config
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pacopt import config as cfg
from pacopt import features as ft

MODEL_PATH = cfg.ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = cfg.ARTIFACTS_DIR / "metrics.json"
SCORED_PATH = cfg.PROCESSED_DIR / "universe_scored.parquet"


# ----------------------------------------------------------------------------- pipelines
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def _check_features(numeric: list[str], categorical: list[str], allow_leaky: bool) -> list[str]:
    cols = list(numeric) + list(categorical)
    if not allow_leaky:  # allow_leaky is ONLY for the M6 leakage ablation
        ft.assert_no_leakage(cols)
    return cols


def _cv(mc: cfg.ModelConfig) -> StratifiedKFold:
    return StratifiedKFold(mc.cv_folds, shuffle=True, random_state=cfg.RANDOM_STATE)


def tune_lr(train: pd.DataFrame, mc: cfg.ModelConfig = cfg.MODEL,
            numeric: list[str] = ft.MODEL_NUMERIC, categorical: list[str] = ft.MODEL_CATEGORICAL,
            allow_leaky: bool = False) -> GridSearchCV:
    """Tune C on PR-AUC; the refit best pipeline is the explainable (uncalibrated) LR."""
    cols = _check_features(numeric, categorical, allow_leaky)
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(max_iter=5_000)),
    ])
    search = GridSearchCV(
        pipe,
        param_grid={"clf__C": list(mc.lr_c_grid)},
        scoring="average_precision",
        cv=_cv(mc),
        n_jobs=mc.n_jobs,
        refit=True,
    )
    search.fit(train[cols], train[cfg.TARGET])
    return search


def fit_lr(train: pd.DataFrame, mc: cfg.ModelConfig = cfg.MODEL,
           numeric: list[str] = ft.MODEL_NUMERIC, categorical: list[str] = ft.MODEL_CATEGORICAL,
           allow_leaky: bool = False) -> tuple[GridSearchCV, object]:
    """Returns (search, scoring_model). scoring_model is the calibrated LR when
    mc.lr_calibration is set, otherwise the tuned pipeline itself."""
    search = tune_lr(train, mc, numeric, categorical, allow_leaky)
    if not mc.lr_calibration:
        return search, search.best_estimator_
    cols = list(numeric) + list(categorical)
    cal = CalibratedClassifierCV(sklearn.base.clone(search.best_estimator_),
                                 method=mc.lr_calibration, cv=_cv(mc))
    cal.fit(train[cols], train[cfg.TARGET])
    return search, cal


def fit_rf(train: pd.DataFrame, mc: cfg.ModelConfig = cfg.MODEL,
           numeric: list[str] = ft.MODEL_NUMERIC, categorical: list[str] = ft.MODEL_CATEGORICAL,
           allow_leaky: bool = False) -> CalibratedClassifierCV:
    cols = _check_features(numeric, categorical, allow_leaky)
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", RandomForestClassifier(
            n_estimators=mc.rf_n_estimators, min_samples_leaf=mc.rf_min_samples_leaf,
            max_features=mc.rf_max_features, n_jobs=mc.n_jobs, random_state=cfg.RANDOM_STATE)),
    ])
    cal = CalibratedClassifierCV(pipe, method=mc.rf_calibration, cv=_cv(mc))
    cal.fit(train[cols], train[cfg.TARGET])
    return cal


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[model_columns(model)])[:, 1]


def model_columns(model) -> list[str]:
    return list(model.feature_names_in_)


# ----------------------------------------------------------------------------- metrics
def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Share of positives among the k highest-scored customers (stable tie-break)."""
    k = int(min(k, len(p)))
    if k <= 0:
        return float("nan")
    top = np.argsort(-p, kind="stable")[:k]
    return float(np.asarray(y)[top].mean())


def budget_implied_k(p: np.ndarray, cost: np.ndarray, budget: float = cfg.CAMPAIGN_BUDGET) -> int:
    """Contacts affordable when contacting customers in descending propensity until the budget runs out."""
    order = np.argsort(-p, kind="stable")
    return int((np.cumsum(np.asarray(cost)[order]) <= budget).sum())


def contact_cost(df: pd.DataFrame) -> np.ndarray:
    from pacopt import economics  # single source for channel costs (Engine 2)
    return economics.contact_cost(df["contact"])


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count (decile) bins: mean predicted vs actual rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    t = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    t["gap_pp"] = (t["predicted"] - t["actual"]) * 100
    return t


def gains_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 = highest propensity. Lift = decile rate / base rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": p})
    d["decile"] = n_bins - pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    t = d.groupby("decile").agg(n=("y", "size"), positives=("y", "sum"),
                                avg_p=("p", "mean"), rate=("y", "mean"))
    base = d["y"].mean()
    t["lift"] = t["rate"] / base
    t["cum_capture"] = t["positives"].cumsum() / t["positives"].sum()
    t["cum_lift"] = (t["positives"].cumsum() / t["n"].cumsum()) / base
    return t


@dataclass
class Evaluation:
    name: str
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    base_rate: float
    max_calibration_gap_pp: float
    precision_at_k: dict = field(default_factory=dict)   # keys "K=500", ..., "K=budget"
    budget_k: int = 0                                    # contacts the budget buys in propensity order

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate(name: str, y_test, p_test, y_universe, p_universe, cost_universe,
             mc: cfg.ModelConfig = cfg.MODEL) -> Evaluation:
    """Discrimination + calibration on the full test set; precision@K on the 10,000 campaign universe."""
    cal = calibration_table(y_test, p_test)
    ks = {f"K={k}": k for k in mc.precision_k}
    kb = budget_implied_k(p_universe, cost_universe)
    ks["K=budget"] = kb   # stable key: K itself differs slightly per model, stored in budget_k
    return Evaluation(
        name=name,
        roc_auc=float(roc_auc_score(y_test, p_test)),
        pr_auc=float(average_precision_score(y_test, p_test)),
        brier=float(brier_score_loss(y_test, p_test)),
        log_loss=float(log_loss(y_test, p_test)),
        base_rate=float(np.mean(y_test)),
        max_calibration_gap_pp=float(cal["gap_pp"].abs().max()),
        precision_at_k={label: precision_at_k(y_universe, p_universe, k) for label, k in ks.items()},
        budget_k=kb,
    )


def curves(y, p) -> dict[str, pd.DataFrame]:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {"roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
            "pr": pd.DataFrame({"recall": rec, "precision": prec})}


# ----------------------------------------------------------------------------- explainability
def lr_coefficients(search_or_pipe) -> pd.DataFrame:
    """Coefficients of the tuned (uncalibrated) LR on the transformed features.
    Numeric = per 1 standard deviation; one-hot = vs the regularised average (no level dropped).
    Isotonic calibration is monotone, so these directions/rankings hold for the scoring model."""
    pipe = getattr(search_or_pipe, "best_estimator_", search_or_pipe)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def permutation_drivers(model, df: pd.DataFrame, n_repeats: int = 5, sample: int = 6_000,
                        mc: cfg.ModelConfig = cfg.MODEL) -> pd.DataFrame:
    """Drop in PR-AUC when each ORIGINAL feature is shuffled (comparable across LR and RF)."""
    cols = model_columns(model)
    d = df.sample(min(sample, len(df)), random_state=cfg.RANDOM_STATE)
    with parallel_config(backend=mc.parallel_backend):
        r = permutation_importance(model, d[cols], d[cfg.TARGET], scoring="average_precision",
                                   n_repeats=n_repeats, random_state=cfg.RANDOM_STATE, n_jobs=mc.n_jobs)
    return (pd.DataFrame({"feature": cols, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ----------------------------------------------------------------------------- model choice
def choose_model(lr: Evaluation, rf: Evaluation, mc: cfg.ModelConfig = cfg.MODEL) -> tuple[str, str]:
    """Return (recommended, reason). RF is preferred only if it beats LR on PR-AUC by the
    configured margin AND is at least as well calibrated."""
    gain = rf.pr_auc - lr.pr_auc
    better_cal = rf.max_calibration_gap_pp <= lr.max_calibration_gap_pp
    if gain > mc.switch_threshold_pr_auc and better_cal:
        return "rf", f"RF PR-AUC +{gain:.3f} (> {mc.switch_threshold_pr_auc}) with calibration at least as good"
    why = (f"RF PR-AUC gain {gain:+.3f} is below the {mc.switch_threshold_pr_auc} switch threshold"
           if gain <= mc.switch_threshold_pr_auc else
           f"RF is calibrated worse ({rf.max_calibration_gap_pp:.2f}pp vs {lr.max_calibration_gap_pp:.2f}pp)")
    return "lr", f"{why}; LR kept for calibration and explainability"


# ----------------------------------------------------------------------------- orchestration
@dataclass
class ModelResult:
    lr_search: GridSearchCV      # tuned, uncalibrated LR: used for coefficients / explanation
    lr: object                   # scoring LR (calibrated)
    rf: CalibratedClassifierCV
    eval_lr: Evaluation
    eval_rf: Evaluation
    recommended: str
    reason: str
    scored: pd.DataFrame


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, universe: pd.DataFrame,
                       mc: cfg.ModelConfig = cfg.MODEL) -> ModelResult:
    with parallel_config(backend=mc.parallel_backend):
        lr_search, lr = fit_lr(train, mc)
        rf = fit_rf(train, mc)
    cost_u = contact_cost(universe)
    p = {name: (predict(m, test), predict(m, universe)) for name, m in (("lr", lr), ("rf", rf))}
    ev_lr = evaluate("Logistic Regression", test[cfg.TARGET], p["lr"][0], universe[cfg.TARGET], p["lr"][1], cost_u, mc)
    ev_rf = evaluate("Random Forest (calibrated)", test[cfg.TARGET], p["rf"][0], universe[cfg.TARGET], p["rf"][1], cost_u, mc)
    recommended, reason = choose_model(ev_lr, ev_rf, mc)

    sel = mc.selected
    scored = universe[["customer_id"]].copy()
    scored["p_lr"] = p["lr"][1]
    scored["p_rf"] = p["rf"][1]
    scored["propensity"] = scored[f"p_{sel}"].clip(upper=mc.propensity_cap)  # D-028
    scored["propensity_rank"] = (scored[f"p_{sel}"].rank(ascending=False, method="first")).astype(int)
    scored["propensity_decile"] = (10 - pd.qcut(scored["propensity"].rank(method="first"), 10, labels=False)).astype(int)
    return ModelResult(lr_search, lr, rf, ev_lr, ev_rf, recommended, reason, scored)


def save(result: ModelResult, mc: cfg.ModelConfig = cfg.MODEL) -> None:
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"lr": result.lr, "lr_explain": result.lr_search.best_estimator_, "rf": result.rf,
                 "selected": mc.selected, "features": model_columns(result.lr),
                 "sklearn_version": sklearn.__version__},
                MODEL_PATH)
    metrics = {
        "selected": mc.selected, "recommended": result.recommended, "reason": result.reason,
        "lr_best_C": result.lr_search.best_params_["clf__C"],
        "lr_cv_pr_auc": float(result.lr_search.best_score_),
        "lr_calibration": mc.lr_calibration,
        "lr": result.eval_lr.as_dict(), "rf": result.eval_rf.as_dict(),
        "features": model_columns(result.lr),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    result.scored.to_parquet(SCORED_PATH, index=False)


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rd = lambda n: pd.read_parquet(cfg.PROCESSED_DIR / f"{n}.parquet")  # noqa: E731
    return rd("train"), rd("test"), rd("universe")


if __name__ == "__main__":
    tr, te, un = load_processed()
    res = train_and_evaluate(tr, te, un)
    save(res)
    for ev in (res.eval_lr, res.eval_rf):
        print(f"{ev.name:28s} ROC-AUC {ev.roc_auc:.4f}  PR-AUC {ev.pr_auc:.4f}  Brier {ev.brier:.4f}  "
              f"max cal gap {ev.max_calibration_gap_pp:.2f}pp  P@K {ev.precision_at_k}")
    print("LR best C:", res.lr_search.best_params_, "| recommended:", res.recommended, "-", res.reason)
