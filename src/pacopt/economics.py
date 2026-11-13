"""Engine 2: Profit Intelligence. The term deposit is valued as a funding product (D-008).

The bank does NOT earn the deposit principal. It earns the funding benefit: the transfer-pricing
(FTP) rate credited to the deposit minus the rate paid to the customer, over the term.

    D_i  = clip(deposit_min + alpha * max(balance_i, 0), deposit_min, deposit_max)   deposit size
    V_i  = D_i * spread * term_years - incentive - onboarding_cost                   contribution if converted
    C_i  = channel_cost[channel_i]                                                    fully-loaded contact cost
    BE_i = max(P_i - P0, 0) * V_i                                                     break-even contact cost
    EP_i = BE_i - C_i                                                                 expected incremental profit

All parameters come from `config.EconomicsConfig`, so Streamlit can pass edited copies.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pacopt import config as cfg

ECON_COLUMNS = ["channel", "channel_group", "deposit", "value", "contact_cost", "break_even_cost",
                "expected_profit", "ep_per_cost", "positive_ep", "ep_rank", "ep_per_cost_rank"]


# ----------------------------------------------------------------------------- per-customer formulas
def deposit_amount(balance, ec: cfg.EconomicsConfig = cfg.ECONOMICS) -> np.ndarray:
    b = np.maximum(np.asarray(balance, dtype=float), 0.0)
    return np.clip(ec.deposit_min + ec.deposit_alpha * b, ec.deposit_min, ec.deposit_max)


def contribution_value(deposit, ec: cfg.EconomicsConfig = cfg.ECONOMICS) -> np.ndarray:
    """Funding-spread contribution on conversion. The principal itself never enters this figure."""
    return np.asarray(deposit, dtype=float) * ec.spread * ec.term_years - ec.incentive - ec.onboarding_cost


def channel_of(contact) -> pd.Series:
    return pd.Series(contact).astype(str).map(cfg.CHANNEL_MAP)


def contact_cost(contact, ec: cfg.EconomicsConfig = cfg.ECONOMICS) -> np.ndarray:
    cost = channel_of(contact).map(ec.channel_cost)
    if cost.isna().any():
        raise ValueError(f"No channel cost for contact values: {sorted(pd.Series(contact)[cost.isna().values].unique())}")
    return cost.to_numpy(dtype=float)


def uplift(propensity, ec: cfg.EconomicsConfig = cfg.ECONOMICS) -> np.ndarray:
    """Incremental conversion probability vs the organic rate P0 (D-015). Floored at 0."""
    return np.maximum(np.asarray(propensity, dtype=float) - ec.organic_rate, 0.0)


def add_economics(df: pd.DataFrame, ec: cfg.EconomicsConfig = cfg.ECONOMICS,
                  propensity_col: str = "propensity") -> pd.DataFrame:
    """Add D, V, C, BE, EP, EP per £ and both rankings. Needs `balance`, `contact` and the propensity."""
    out = df.copy()
    out["channel"] = channel_of(out["contact"]).to_numpy()
    out["channel_group"] = out["channel"].map(cfg.CHANNEL_GROUP)
    out["deposit"] = deposit_amount(out["balance"], ec)
    out["value"] = contribution_value(out["deposit"], ec)
    out["contact_cost"] = contact_cost(out["contact"], ec)
    out["break_even_cost"] = uplift(out[propensity_col], ec) * out["value"]
    out["expected_profit"] = out["break_even_cost"] - out["contact_cost"]
    out["ep_per_cost"] = out["expected_profit"] / out["contact_cost"]
    out["positive_ep"] = (out["expected_profit"] > 0).astype("int8")
    out["ep_rank"] = out["expected_profit"].rank(ascending=False, method="first").astype(int)
    out["ep_per_cost_rank"] = out["ep_per_cost"].rank(ascending=False, method="first").astype(int)
    return out


# ----------------------------------------------------------------------------- campaign-level checks
def budget_check(table: pd.DataFrame, budget: float = cfg.CAMPAIGN_BUDGET,
                 capacity: dict = cfg.CHANNEL_CAPACITY) -> dict:
    """Does the budget bind? Take eligible positive-EP customers, keep the best by EP within each
    channel group's capacity, and compare their total contact cost with the budget (plan 3.1)."""
    pos = table[(table["eligible"] == 1) & (table["expected_profit"] > 0)]
    kept = pd.concat([g.nlargest(capacity.get(name, len(g)), "expected_profit")
                      for name, g in pos.groupby("channel_group", observed=True)])
    cost_after_caps = float(kept["contact_cost"].sum())
    eligible = int((table["eligible"] == 1).sum())
    return {
        "eligible": eligible,
        "positive_ep": len(pos),
        "positive_ep_share_of_eligible": len(pos) / eligible if eligible else float("nan"),
        "cost_all_positive_ep": float(pos["contact_cost"].sum()),
        "after_channel_caps": len(kept),
        "cost_after_channel_caps": cost_after_caps,
        "sum_positive_ep": float(pos["expected_profit"].sum()),
        "budget": budget,
        "budget_binds": cost_after_caps > budget,
    }


def top_decile_share(table: pd.DataFrame, eligible_only: bool = True) -> dict:
    """Share of total positive EP held by the top 10% of customers by EP."""
    t = table[table["eligible"] == 1] if eligible_only else table
    ep = t["expected_profit"].clip(lower=0).sort_values(ascending=False)
    top = ep.head(int(np.ceil(len(ep) / 10)))
    return {"customers": len(top), "top_decile_ep": float(top.sum()), "total_positive_ep": float(ep.sum()),
            "share": float(top.sum() / ep.sum()) if ep.sum() else float("nan"),
            "min_ep_in_top_decile": float(top.min())}


def ep_decile_table(table: pd.DataFrame, eligible_only: bool = True) -> pd.DataFrame:
    """EP deciles (1 = highest EP): customers, mean EP, share of total positive EP, mean P and V."""
    t = (table[table["eligible"] == 1] if eligible_only else table).copy()
    t["ep_decile"] = 10 - pd.qcut(t["expected_profit"].rank(method="first"), 10, labels=False)
    total = t["expected_profit"].clip(lower=0).sum()
    g = t.groupby("ep_decile").agg(customers=("expected_profit", "size"),
                                   mean_ep=("expected_profit", "mean"),
                                   positive_ep=("expected_profit", lambda s: s.clip(lower=0).sum()),
                                   mean_propensity=("propensity", "mean"),
                                   mean_value=("value", "mean"),
                                   mean_cost=("contact_cost", "mean"))
    g["share_of_positive_ep"] = g["positive_ep"] / total
    g["cum_share"] = g["share_of_positive_ep"].cumsum()
    return g


# ----------------------------------------------------------------------------- sensitivity
def scaled(ec: cfg.EconomicsConfig, parameter: str, factor: float) -> cfg.EconomicsConfig:
    """Copy of `ec` with one driver scaled: 'spread' (customer rate fixed), 'incentive' or 'contact_cost'."""
    if parameter == "spread":
        return replace(ec, ftp_rate=ec.customer_rate + ec.spread * factor)
    if parameter == "incentive":
        return replace(ec, incentive=ec.incentive * factor)
    if parameter == "contact_cost":
        return replace(ec, channel_cost={k: v * factor for k, v in ec.channel_cost.items()})
    raise ValueError(f"Unknown sensitivity parameter: {parameter}")


def sensitivity(df: pd.DataFrame, ec: cfg.EconomicsConfig = cfg.ECONOMICS,
                parameters: tuple = ("spread", "incentive", "contact_cost"),
                factors: tuple = (0.5, 0.75, 1.0, 1.25, 1.5),
                budget: float = cfg.CAMPAIGN_BUDGET) -> pd.DataFrame:
    """Total positive EP (eligible customers) and budget binding under ±50% changes to one driver."""
    rows = []
    for p in parameters:
        for f in factors:
            t = add_economics(df, scaled(ec, p, f))
            b = budget_check(t, budget)
            rows.append({"parameter": p, "factor": f, "change": f"{f - 1:+.0%}",
                         "total_positive_ep": b["sum_positive_ep"], "positive_ep_customers": b["positive_ep"],
                         "positive_ep_share": b["positive_ep_share_of_eligible"],
                         "cost_after_caps": b["cost_after_channel_caps"], "budget_binds": b["budget_binds"]})
    return pd.DataFrame(rows)
