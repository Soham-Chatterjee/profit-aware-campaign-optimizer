"""RFM-inspired features, eligibility flags, leakage guard and the train/test/universe split.

RFM here is *inspired*, not true transaction RFM (the dataset has no transactions):
- Recency   = days since the previous campaign contact (`pdays`, -1 = never contacted)
- Frequency = contacts in previous campaigns (`previous`) - NOT `campaign` (leakage, D-004)
- Monetary  = account `balance`, a value indicator only, never revenue (D-005)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pacopt import config as cfg

# ----------------------------------------------------------------------------- model feature allow-list
MODEL_NUMERIC = [
    "age",
    "balance_slog",
    "frequency",
    "never_contacted",
    "prior_success",
    "housing",
    "loan",
    "default",
]
MODEL_CATEGORICAL = [
    "job",
    "marital",
    "education",
    "contact",
    "poutcome",
    "age_band",
    "recency_bucket",
]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL


class LeakageError(ValueError):
    """Raised when a post-campaign (leaky) column is used as a model feature."""


def assert_no_leakage(features: list[str]) -> None:
    """Fail loudly if any forbidden column is present in a feature list."""
    leaked = sorted(set(features) & cfg.FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"Forbidden (post-campaign or target) columns in features: {leaked}")


assert_no_leakage(MODEL_FEATURES)  # guard at import time


# ----------------------------------------------------------------------------- feature engineering
def signed_log(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def add_rfm_features(df: pd.DataFrame, fc: cfg.FeatureConfig = cfg.FEATURES) -> pd.DataFrame:
    out = df.copy()

    # Recency
    out["never_contacted"] = (out["pdays"] == -1).astype("int8")
    out["recency_days"] = out["pdays"].where(out["pdays"] >= 0).astype("float")
    bucket = pd.cut(out["recency_days"], bins=list(fc.recency_bins), labels=list(fc.recency_labels))
    out["recency_bucket"] = bucket.cat.add_categories(["never"]).fillna("never")

    # Frequency
    out["frequency"] = out["previous"].clip(upper=fc.frequency_cap).astype("int16")
    out["prior_success"] = (out["poutcome"] == "success").astype("int8")

    # Monetary proxy (value indicator, not revenue)
    out["balance_slog"] = signed_log(out["balance"])
    out["balance_quintile"] = (
        pd.qcut(out["balance"].rank(method="first"), fc.balance_quintiles, labels=False) + 1
    ).astype("int8")

    # Profile
    out["age_band"] = pd.cut(out["age"], bins=list(fc.age_bins), labels=list(fc.age_labels))
    return out


def add_eligibility(df: pd.DataFrame, ec: cfg.EligibilityConfig = cfg.ELIGIBILITY) -> pd.DataFrame:
    """Flag responsible-outcome exclusions (D-007). 1 in an `elig_*` column = excluded by that rule."""
    out = df.copy()
    out["elig_default"] = ((out["default"] == 1) & ec.exclude_default).astype("int8")
    out["elig_negative_balance"] = ((out["balance"] < 0) & ec.exclude_negative_balance).astype("int8")
    out["elig_fatigue"] = ((out["previous"] > ec.fatigue_max_previous) & ec.exclude_fatigue).astype("int8")
    out["flag_age_75plus"] = (out["age"] >= ec.protected_age).astype("int8")
    excluded = out[["elig_default", "elig_negative_balance", "elig_fatigue"]].any(axis=1)
    out["eligible"] = (~excluded).astype("int8")
    return out


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    return add_eligibility(add_rfm_features(clean_df))


# ----------------------------------------------------------------------------- split and universe
@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    universe: pd.DataFrame  # 10,000-customer campaign universe, sampled from test (out-of-sample)


def assign_customer_ids(df: pd.DataFrame, start: int = cfg.CUSTOMER_ID_START) -> pd.DataFrame:
    """Stable IDs based on the original row order of bank-full.csv (C100001 = first row)."""
    out = df.copy()
    out.insert(0, "customer_id", [f"C{start + i}" for i in range(len(out))])
    return out


def split(
    df: pd.DataFrame,
    test_size: float = cfg.TEST_SIZE,
    universe_size: int = cfg.UNIVERSE_SIZE,
    random_state: int = cfg.RANDOM_STATE,
) -> SplitResult:
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[cfg.TARGET], random_state=random_state
    )
    universe, _ = train_test_split(
        test, train_size=universe_size, stratify=test[cfg.TARGET], random_state=random_state
    )
    return SplitResult(
        train=train.sort_index(), test=test.sort_index(), universe=universe.sort_index()
    )


def prepare(raw_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, SplitResult]:
    """Full Milestone-1 pipeline: raw -> clean -> features -> IDs -> split."""
    from pacopt import data

    raw_df = data.load_raw() if raw_df is None else raw_df
    feats = assign_customer_ids(build_features(data.clean(raw_df)))
    return feats, split(feats)


def save_processed(feats: pd.DataFrame, parts: SplitResult) -> None:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cfg.PROCESSED_DIR / "customers_features.parquet", index=False)
    parts.train.to_parquet(cfg.PROCESSED_DIR / "train.parquet", index=False)
    parts.test.to_parquet(cfg.PROCESSED_DIR / "test.parquet", index=False)
    parts.universe.to_parquet(cfg.PROCESSED_DIR / "universe.parquet", index=False)


if __name__ == "__main__":
    f, p = prepare()
    save_processed(f, p)
    print(f"features {f.shape}, train {p.train.shape}, test {p.test.shape}, universe {p.universe.shape}")
