"""Scenarios: one bundle of editable assumptions -> one campaign table and one optimisation problem.

The Streamlit dashboard (M5) edits a `Scenario`; everything downstream is recomputed from it.
No training happens here: the saved propensity scores (both models) are reused.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from pacopt import config as cfg
from pacopt import features as ft
from pacopt import optimize as op
from pacopt import pipeline as pl


@dataclass(frozen=True)
class Scenario:
    economics: cfg.EconomicsConfig = field(default_factory=cfg.EconomicsConfig)
    eligibility: cfg.EligibilityConfig = field(default_factory=cfg.EligibilityConfig)
    constraints: cfg.CampaignConstraints = field(default_factory=cfg.CampaignConstraints)
    model: str = cfg.MODEL.selected            # "lr" or "rf"

    def to_json(self) -> str:
        """Stable text form (used as a cache key by the dashboard)."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        d = json.loads(text)
        return cls(economics=cfg.EconomicsConfig(**d["economics"]),
                   eligibility=cfg.EligibilityConfig(**d["eligibility"]),
                   constraints=cfg.CampaignConstraints(**d["constraints"]),
                   model=d["model"])


DEFAULT = Scenario()


def scored_for(scored: pd.DataFrame, model: str, cap: float = cfg.MODEL.propensity_cap) -> pd.DataFrame:
    """Propensity column from the chosen model (both are saved by the M2 stage), capped (D-028)."""
    if model not in ("lr", "rf"):
        raise ValueError(f"Unknown model: {model}")
    out = scored.copy()
    out["propensity"] = out[f"p_{model}"].clip(upper=cap)
    out["propensity_rank"] = out["propensity"].rank(ascending=False, method="first").astype(int)
    out["propensity_decile"] = (10 - pd.qcut(out["propensity"].rank(method="first"), 10, labels=False)).astype(int)
    return out


def build_table(scenario: Scenario = DEFAULT, universe: pd.DataFrame | None = None,
                scored: pd.DataFrame | None = None) -> pd.DataFrame:
    """Campaign table for a scenario: eligibility rules, model choice and economics applied."""
    universe = pl.load_universe() if universe is None else universe
    scored = pl.load_scored() if scored is None else scored
    u = ft.add_eligibility(universe, scenario.eligibility)
    return pl.campaign_table(u, scored_for(scored, scenario.model), scenario.economics)


fingerprint = op.fingerprint


# ----------------------------------------------------------------------------- what-if helpers (Screen 4)
def affordable_contacts(table: pd.DataFrame, cc: cfg.CampaignConstraints, x: np.ndarray | None = None) -> pd.DataFrame:
    """Budget-only upper bound (budget / channel cost) next to the actual allocation of a list."""
    pr = op.build_problem(table, cc)
    e = table[table["eligible"] == 1]
    x = op.greedy(pr) if x is None else np.asarray(x, bool)
    sel = e.loc[pr.table_index[x]]
    rows = []
    for ch_name, g in e.groupby("channel", observed=True):
        cost = float(g["contact_cost"].iloc[0])
        rows.append({"channel": ch_name, "channel_group": cfg.CHANNEL_GROUP[ch_name], "cost_per_contact": cost,
                     "max_affordable_budget_only": int(cc.budget // cost) if cost else np.inf,
                     "eligible_customers": len(g),
                     "positive_ep_customers": int((g["expected_profit"] > 0).sum()),
                     "actual_allocation": int((sel["channel"] == ch_name).sum())})
    out = pd.DataFrame(rows).set_index("channel")
    out.loc["All channels"] = {"channel_group": "", "cost_per_contact": float(sel["contact_cost"].mean()) if len(sel) else np.nan,
                               "max_affordable_budget_only": int(cc.budget // e["contact_cost"].min()),
                               "eligible_customers": len(e), "positive_ep_customers": int((e["expected_profit"] > 0).sum()),
                               "actual_allocation": len(sel)}
    return out


def budget_curve(table: pd.DataFrame, cc: cfg.CampaignConstraints, budgets) -> pd.DataFrame:
    """Greedy expected contribution over a grid of budgets (all other limits unchanged)."""
    rows = []
    for b in budgets:
        pr = op.build_problem(table, replace(cc, budget=float(b)))
        x = op.greedy(pr)
        rows.append({"budget": float(b), **op.selection_metrics(pr, x)})
    return pd.DataFrame(rows)


def saturation_point(table: pd.DataFrame, cc: cfg.CampaignConstraints) -> dict:
    """Spend at which the next contact no longer adds profit: Greedy with an unlimited budget.
    `limited_by` says whether positive-EP customers ran out or another limit (contacts, channel) bound."""
    pr = op.build_problem(table, replace(cc, budget=1e12))
    x = op.greedy(pr)
    m = op.selection_metrics(pr, x)
    ck = op.check_constraints(table, pr.customer_id[x], replace(cc, budget=1e12)).set_index("constraint")
    caps = ck[ck.index.str.startswith(("Contacts", "Channel capacity"))]
    binding = [c for c, r in caps.iterrows() if r["value"] >= r["limit"]]
    return {"spend": m["spend"], "expected_contribution": m["expected_contribution"], "contacts": m["contacts"],
            "limited_by": ", ".join(binding) if binding else "no positive-EP customers left"}
