"""One command for the whole flow: `uv run python -m pacopt.pipeline`.

Stages: data + features (M1) -> propensity model (M2) -> economics + segments (M3)
-> strategy comparison incl. GA and MILP optimum (M4). `campaign_table()` is cheap (no
training), so Streamlit can call it again with edited economics.
"""
from __future__ import annotations

import json

import pandas as pd

from pacopt import config as cfg
from pacopt import economics as econ
from pacopt import features as ft
from pacopt import model as m
from pacopt import optimize as opt
from pacopt import segmentation as seg

CAMPAIGN_TABLE_PATH = cfg.PROCESSED_DIR / "campaign_table.parquet"
STRATEGY_PATH = cfg.ARTIFACTS_DIR / "strategy_comparison.csv"
TARGET_LIST_PATH = cfg.ARTIFACTS_DIR / "ga_target_list.csv"
GA_HISTORY_PATH = cfg.ARTIFACTS_DIR / "ga_history.csv"
SELECTIONS_PATH = cfg.ARTIFACTS_DIR / "strategy_selections.parquet"
OPTIM_META_PATH = cfg.ARTIFACTS_DIR / "optimisation_meta.json"


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(cfg.PROCESSED_DIR / "universe.parquet")


def load_scored() -> pd.DataFrame:
    return pd.read_parquet(m.SCORED_PATH)


def campaign_table(universe: pd.DataFrame | None = None, scored: pd.DataFrame | None = None,
                   ec: cfg.EconomicsConfig = cfg.ECONOMICS,
                   sc: cfg.SegmentConfig = cfg.SEGMENTATION) -> pd.DataFrame:
    """One row per universe customer: features, eligibility, propensity, economics, segment, RFM."""
    universe = load_universe() if universe is None else universe
    scored = load_scored() if scored is None else scored
    t = universe.merge(scored, on="customer_id", how="left", validate="one_to_one")
    if len(t) != len(universe) or t["propensity"].isna().any():
        raise ValueError("Every universe customer needs a propensity score; re-run the model stage.")
    return seg.add_segments(econ.add_economics(t, ec), sc)


def save_campaign_table(table: pd.DataFrame) -> None:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(CAMPAIGN_TABLE_PATH, index=False)


def load_campaign_table() -> pd.DataFrame:
    return pd.read_parquet(CAMPAIGN_TABLE_PATH)


def save_optimisation(table: pd.DataFrame, comp: "opt.Comparison") -> None:
    """Strategy comparison, GA target list, GA convergence history, every selection and a metadata
    file whose fingerprint lets the dashboard reuse these results for an identical problem (M4/M5)."""
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comp.table.to_csv(STRATEGY_PATH)
    opt.target_list(table, comp).to_csv(TARGET_LIST_PATH, index=False)
    comp.ga.history.to_csv(GA_HISTORY_PATH, index=False)
    sel = pd.DataFrame({"customer_id": comp.problem.customer_id,
                        **{name: x for name, x in comp.selections.items()}})
    sel.to_parquet(SELECTIONS_PATH, index=False)
    g, mm = comp.ga, comp.milp
    meta = {"fingerprint": opt.fingerprint(table, comp.problem.cc),
            "ga": {"generations": g.generations, "runtime_s": g.runtime_s, "seeded": g.seeded,
                   "feasible": g.feasible},
            "milp": None if mm is None else {"status": mm.status, "message": mm.message, "mip_gap": mm.mip_gap,
                                             "dual_bound": mm.dual_bound, "runtime_s": mm.runtime_s}}
    OPTIM_META_PATH.write_text(json.dumps(meta, indent=2))


def load_optimisation(table: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS
                      ) -> tuple["opt.GAResult", "opt.MILPResult | None"] | None:
    """Saved GA (and MILP) results, only if they were computed for exactly this problem."""
    if not (OPTIM_META_PATH.exists() and SELECTIONS_PATH.exists() and GA_HISTORY_PATH.exists()):
        return None
    meta = json.loads(OPTIM_META_PATH.read_text())
    if meta.get("fingerprint") != opt.fingerprint(table, cc):
        return None
    pr = opt.build_problem(table, cc)
    sel = pd.read_parquet(SELECTIONS_PATH).set_index("customer_id").reindex(pr.customer_id)
    x_ga = sel["Genetic Algorithm"].fillna(False).to_numpy(bool)
    g = meta["ga"]
    ga_res = opt.GAResult(x=x_ga, expected_profit=float(pr.ep[x_ga].sum()), feasible=g["feasible"],
                          generations=g["generations"], runtime_s=g["runtime_s"],
                          history=pd.read_csv(GA_HISTORY_PATH), seeded=g["seeded"])
    mres = None
    if meta.get("milp") and "MILP optimum" in sel:
        x_m = sel["MILP optimum"].fillna(False).to_numpy(bool)
        mm = meta["milp"]
        mres = opt.MILPResult(x=x_m, expected_profit=float(pr.ep[x_m].sum()), status=mm["status"],
                              message=mm["message"], mip_gap=mm["mip_gap"], dual_bound=mm["dual_bound"],
                              runtime_s=mm["runtime_s"])
    return ga_res, mres


def run(retrain: bool = True, optimise: bool = True) -> tuple[pd.DataFrame, "opt.Comparison | None"]:
    feats, parts = ft.prepare()
    ft.save_processed(feats, parts)
    if retrain:
        res = m.train_and_evaluate(parts.train, parts.test, parts.universe)
        m.save(res)
        scored = res.scored
    else:
        scored = load_scored()
    table = campaign_table(parts.universe, scored)
    save_campaign_table(table)
    comp = None
    if optimise:
        comp = opt.compare_strategies(table)
        save_optimisation(table, comp)
    return table, comp


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the campaign pipeline.")
    ap.add_argument("--no-retrain", action="store_true", help="reuse the saved propensity scores")
    ap.add_argument("--no-optimise", action="store_true", help="stop after the campaign table (skip M4)")
    args = ap.parse_args()
    t, comp = run(retrain=not args.no_retrain, optimise=not args.no_optimise)
    b = econ.budget_check(t)
    print(f"campaign table {t.shape} -> {CAMPAIGN_TABLE_PATH}")
    print(t["segment"].value_counts().reindex(list(cfg.SEGMENTS)).to_string())
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
    if comp is not None:
        cols = ["contacts", "spend", "expected_contribution", "roi", "pct_of_optimum", "feasible"]
        print(comp.table[[c for c in cols if c in comp.table]].round(3).to_string())
        print(f"GA: {comp.ga.generations} generations in {comp.ga.runtime_s:.1f}s | "
              f"MILP: {comp.milp.message} in {comp.milp.runtime_s:.1f}s")
