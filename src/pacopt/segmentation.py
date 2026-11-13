"""Engine 1 (part): the four business segments, RFM scores and the engagement score.

Segments come from a transparent propensity x value grid (D-013, D-032):

                     value >= median      value < median
    P in top 40%     High value           Growing
    P in bottom 60%  Dormant              Low engagement

RFM scores (1-5) and the engagement score (0-100) are overlays, not segments (D-033).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pacopt import config as cfg

SEGMENT_DTYPE = pd.CategoricalDtype(list(cfg.SEGMENTS), ordered=True)


def thresholds(df: pd.DataFrame, sc: cfg.SegmentConfig = cfg.SEGMENTATION,
               propensity_col: str = "propensity", value_col: str = "value") -> dict:
    return {"propensity": float(df[propensity_col].quantile(sc.high_propensity_quantile)),
            "value": float(df[value_col].quantile(sc.high_value_quantile))}


def assign_segments(df: pd.DataFrame, sc: cfg.SegmentConfig = cfg.SEGMENTATION,
                    propensity_col: str = "propensity", value_col: str = "value",
                    cut: dict | None = None) -> pd.Series:
    """Exactly one segment per customer. `cut` overrides the quantile thresholds (for what-ifs)."""
    cut = cut or thresholds(df, sc, propensity_col, value_col)
    hp = df[propensity_col].to_numpy() >= cut["propensity"]
    hv = df[value_col].to_numpy() >= cut["value"]
    seg = np.select([hp & hv, hp & ~hv, ~hp & hv], list(cfg.SEGMENTS[:3]), default=cfg.SEGMENTS[3])
    return pd.Series(pd.Categorical(seg, dtype=SEGMENT_DTYPE), index=df.index, name="segment")


def rfm_scores(df: pd.DataFrame, sc: cfg.SegmentConfig = cfg.SEGMENTATION) -> pd.DataFrame:
    """R from the recency bucket, F from `previous`, M from the balance quintile; all 1-5 (5 = best).
    Engagement = weighted recency, frequency and prior success, scaled to 0-100."""
    r = df["recency_bucket"].astype(str).map(dict(sc.recency_score))
    if r.isna().any():
        raise ValueError(f"Unmapped recency buckets: {sorted(df.loc[r.isna(), 'recency_bucket'].unique())}")
    f = pd.cut(df["previous"], bins=list(sc.frequency_bins), labels=[1, 2, 3, 4, 5]).astype(int)
    m = df["balance_quintile"].astype(int)
    w_r, w_f, w_s = sc.engagement_weights
    engagement = 100 * (w_r * (r - 1) / 4 + w_f * (f - 1) / 4 + w_s * df["prior_success"]) / (w_r + w_f + w_s)
    out = pd.DataFrame({"r_score": r.astype("int8"), "f_score": f.astype("int8"), "m_score": m.astype("int8")},
                       index=df.index)
    out["rfm_code"] = out["r_score"].astype(str) + out["f_score"].astype(str) + out["m_score"].astype(str)
    out["engagement_score"] = engagement.round(1)
    return out


def add_segments(df: pd.DataFrame, sc: cfg.SegmentConfig = cfg.SEGMENTATION) -> pd.DataFrame:
    out = df.copy()
    out["segment"] = assign_segments(out, sc)
    return pd.concat([out, rfm_scores(out, sc)], axis=1)


def segment_profile(table: pd.DataFrame) -> pd.DataFrame:
    """Screen 1 profile table: one row per segment, in the fixed segment order."""
    g = table.groupby("segment", observed=False)
    prof = g.agg(customers=("customer_id", "size"), avg_age=("age", "mean"), avg_balance=("balance", "mean"),
                 avg_propensity=("propensity", "mean"), avg_value=("value", "mean"),
                 avg_expected_profit=("expected_profit", "mean"), pct_eligible=("eligible", "mean"),
                 pct_positive_ep=("positive_ep", "mean"), pct_never_contacted=("never_contacted", "mean"),
                 avg_engagement=("engagement_score", "mean"))
    prof["share"] = prof["customers"] / prof["customers"].sum()
    return prof
