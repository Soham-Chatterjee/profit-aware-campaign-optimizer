"""Validation (Milestone 6): fairness monitor, leakage ablation, backtest and robustness checks.

Everything here *audits* the engines; nothing feeds back into targeting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.metrics import average_precision_score, roc_auc_score

from pacopt import config as cfg
from pacopt import economics as econ
from pacopt import features as ft
from pacopt import model as m
from pacopt import optimize as op


# ============================================================================= fairness monitor (6.6)
@dataclass(frozen=True)
class FairnessConfig:
    groups: tuple = ("age_band", "job")
    low_ratio: float = 0.5        # targeting rate below half the overall rate -> flagged
    high_ratio: float = 2.0       # ... or above twice the overall rate
    min_group_size: int = 30      # smaller groups are shown but never flagged (too noisy)


FAIRNESS = FairnessConfig()


def fairness_monitor(table: pd.DataFrame, selected_ids, fc: FairnessConfig = FAIRNESS) -> pd.DataFrame:
    """Targeting rate by group vs the eligible base. Reporting only: never a constraint.

    rate_ratio = (selected in group / eligible in group) / (selected / eligible overall)."""
    e = table[table["eligible"] == 1]
    sel = e["customer_id"].isin(pd.Index(list(selected_ids)))
    overall = sel.mean() if len(e) else np.nan
    rows = []
    for g in fc.groups:
        key = e[g].astype(str)
        for level, idx in key.groupby(key, sort=True).groups.items():
            n = len(idx)
            k = int(sel.loc[idx].sum())
            rate = k / n if n else np.nan
            ratio = rate / overall if overall else np.nan
            flag = (n >= fc.min_group_size) and (ratio < fc.low_ratio or ratio > fc.high_ratio)
            rows.append({"dimension": g, "group": level, "eligible": n, "selected": k,
                         "share_of_eligible": n / len(e), "share_of_selected": k / sel.sum() if sel.sum() else 0.0,
                         "targeting_rate": rate, "rate_ratio": ratio,
                         "flag": ("under-targeted" if ratio < fc.low_ratio else "over-targeted") if flag else ""})
    return pd.DataFrame(rows)


def fairness_drivers(table: pd.DataFrame, dimension: str = "job") -> pd.DataFrame:
    """Per group: average propensity, value and EP of eligible customers. Explains *why* a group is
    targeted more or less (the optimiser only sees EP, never the group itself)."""
    e = table[table["eligible"] == 1]
    return (e.groupby(e[dimension].astype(str))
            .agg(eligible=("customer_id", "size"), avg_propensity=("propensity", "mean"),
                 avg_value=("value", "mean"), avg_expected_profit=("expected_profit", "mean"),
                 pct_positive_ep=("positive_ep", "mean"), pct_digital=("channel_group", lambda s: (s == "digital").mean())))


# ============================================================================= leakage ablation (6.2)
ABLATION_SETS = {
    # name: (extra numeric, extra categorical) added to the chosen allow-list
    "Chosen model: duration, campaign, day, month removed": ([], []),
    "Only duration removed (campaign, day, month added)": (["campaign", "day"], ["month"]),
    "Everything included (duration, campaign, day, month)": (["duration", "campaign", "day"], ["month"]),
}


def leakage_ablation(train: pd.DataFrame, test: pd.DataFrame, mc: cfg.ModelConfig = cfg.MODEL) -> pd.DataFrame:
    """Test-set ROC-AUC / PR-AUC of the tuned LR with and without the post-campaign columns.
    The leaky variants are fitted with `allow_leaky=True`, which exists ONLY for this ablation."""
    rows = []
    for name, (num_extra, cat_extra) in ABLATION_SETS.items():
        num = ft.MODEL_NUMERIC + num_extra
        cat = ft.MODEL_CATEGORICAL + cat_extra
        leaky = sorted(set(num_extra + cat_extra) & cfg.LEAKAGE_COLUMNS)
        with parallel_config(backend=mc.parallel_backend):
            search = m.tune_lr(train, mc, num, cat, allow_leaky=bool(leaky))
        p = search.best_estimator_.predict_proba(test[num + cat])[:, 1]
        y = test[cfg.TARGET]
        rows.append({"variant": name, "leaky_columns_used": ", ".join(leaky) or "none",
                     "n_features": len(num + cat), "best_C": search.best_params_["clf__C"],
                     "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                     "precision_at_1000": m.precision_at_k(y.to_numpy(), p, 1_000)})
    out = pd.DataFrame(rows).set_index("variant")
    base = out.iloc[0]
    out["roc_auc_vs_chosen"] = out["roc_auc"] - base["roc_auc"]
    out["pr_auc_vs_chosen"] = out["pr_auc"] - base["pr_auc"]
    return out


def transforms_inside_pipeline(model) -> dict:
    """Confirm every fitted transform lives inside the sklearn Pipeline (so CV and calibration
    re-fit it on training folds only, after the split). Returns the step names found."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    pipes = []
    if isinstance(model, CalibratedClassifierCV):
        pipes = [c.estimator for c in model.calibrated_classifiers_]
    elif isinstance(model, Pipeline):
        pipes = [model]
    ok = bool(pipes) and all(isinstance(p, Pipeline) and isinstance(p.steps[0][1], ColumnTransformer) for p in pipes)
    return {"n_fitted_pipelines": len(pipes), "steps": [s for s, _ in pipes[0].steps] if pipes else [],
            "preprocessing_inside_pipeline": ok}


# ============================================================================= backtest (6.4)
def backtest(table: pd.DataFrame, comp: op.Comparison, ec: cfg.EconomicsConfig = cfg.ECONOMICS) -> pd.DataFrame:
    """Realised outcome of each strategy's list using the ACTUAL `y` of the campaign universe.

    Realised contribution = sum over converters of V_i, minus every contacted customer's C_i.
    Expected figures use P_i; Random is averaged over the same seeds as the comparison.
    Caveat: `y` was recorded under the historical campaign (customers were contacted then), so
    this checks the calibration of P_i on the chosen lists, not a causal campaign effect (D-015)."""
    pr = comp.problem
    y = table.loc[pr.table_index, cfg.TARGET].to_numpy()
    v = table.loc[pr.table_index, "value"].to_numpy()
    uplift = econ.uplift(pr.propensity, ec)

    def one(x):
        x = np.asarray(x, bool)
        conv = y[x].sum()
        return {"contacts": int(x.sum()), "spend": float(pr.cost[x].sum()),
                "expected_conversions": float(pr.propensity[x].sum()), "actual_conversions": int(conv),
                "expected_contribution": float(pr.ep[x].sum()),
                "realised_contribution": float((y[x] * v[x]).sum() - pr.cost[x].sum()),
                "expected_rate": float(pr.propensity[x].mean()) if x.any() else np.nan,
                "actual_rate": float(y[x].mean()) if x.any() else np.nan}

    rows = []
    for name in op.STRATEGY_ORDER:
        if name not in comp.selections:
            continue
        if name == "Random":
            runs = pd.DataFrame([one(op.random_selection(pr, s)) for s in comp.random_runs["seed"]])
            r = runs.mean().to_dict()
            r["realised_contribution_std"] = float(runs["realised_contribution"].std())
        else:
            r = one(comp.selections[name])
        rows.append({"strategy": name, **r})
    out = pd.DataFrame(rows).set_index("strategy")
    out["conversion_gap"] = out["actual_conversions"] - out["expected_conversions"]
    out["realised_vs_expected"] = out["realised_contribution"] / out["expected_contribution"] - 1
    rnd = out.loc["Random", "realised_contribution"]
    out["realised_incremental_vs_random"] = out["realised_contribution"] - rnd
    return out


def bootstrap_backtest(table: pd.DataFrame, comp: op.Comparison, strategies=("Genetic Algorithm", "Propensity"),
                       n_boot: int = 2_000, seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Uncertainty of the realised contribution: resample each list's customers with replacement.
    Also the realised gap between each strategy and Random (Random's 50-seed mean is held fixed)."""
    pr = comp.problem
    y = table.loc[pr.table_index, cfg.TARGET].to_numpy()
    v = table.loc[pr.table_index, "value"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for name in strategies:
        idx = np.flatnonzero(comp.selections[name])
        gain = y[idx] * v[idx] - pr.cost[idx]
        boots = np.array([gain[rng.integers(0, len(idx), len(idx))].sum() for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"strategy": name, "realised": gain.sum(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("strategy")


# ============================================================================= robustness (6.5)
def ga_seed_runs(table: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS, seeds=range(5),
                 ga: cfg.GAConfig = cfg.GA, optimum: float | None = None) -> pd.DataFrame:
    """GA result per seed, seeded and unseeded (mean +/- std reported by the caller)."""
    pr = op.build_problem(table, cc)
    rows = []
    for seeded in (True, False):
        for s in seeds:
            r = op.run_ga(pr, replace(ga, random_state=s, greedy_seeds=ga.greedy_seeds if seeded else 0))
            rows.append({"seeded": seeded, "seed": s, "expected_contribution": r.expected_profit,
                         "generations": r.generations, "runtime_s": r.runtime_s,
                         "feasible": op.is_feasible(table, pr.customer_id[r.x], cc),
                         "pct_of_optimum": r.expected_profit / optimum if optimum else np.nan})
    return pd.DataFrame(rows)


def ranking_sensitivity(base: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS,
                        ec: cfg.EconomicsConfig = cfg.ECONOMICS, factors=(0.5, 0.75, 1.25, 1.5),
                        parameters=("spread", "incentive", "contact_cost"),
                        ga: cfg.GAConfig = replace(cfg.GA, time_limit_s=30.0)) -> pd.DataFrame:
    """Strategy ranking when one economic driver changes and everything else stays at the default.
    `base` = the campaign table without economics columns (universe + scores + eligibility).
    Segments are re-derived because the value cut-off moves with the economics."""
    from pacopt import segmentation as seg

    rows = []
    cases = [("default", 1.0)] + [(p, f) for p in parameters for f in factors]
    for p, f in cases:
        e2 = ec if p == "default" else econ.scaled(ec, p, f)
        t = seg.add_segments(econ.add_economics(base, e2))
        c = op.compare_strategies(t, cc, ga, replace(cfg.OPTIM, random_runs=20), run_milp=False)
        tb = c.table["expected_contribution"]
        order = list(tb.sort_values(ascending=False).index)
        rows.append({"parameter": p, "factor": f, "change": "default" if p == "default" else f"{f - 1:+.0%}",
                     **{f"ep_{k}": v for k, v in tb.items()},
                     "ranking": " > ".join(order), "ga_rank": order.index("Genetic Algorithm") + 1,
                     "ga_vs_greedy": tb["Genetic Algorithm"] - tb["Greedy"],
                     "ga_vs_random_pct": tb["Genetic Algorithm"] / tb["Random"] - 1 if tb["Random"] else np.nan,
                     "all_feasible": bool(c.table["feasible"].all())})
    return pd.DataFrame(rows)
