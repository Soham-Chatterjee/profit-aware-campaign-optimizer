"""Milestone 1 tests: data contract, cleaning, RFM features, eligibility, leakage guard, split."""
import pandas as pd
import pytest

from pacopt import config as cfg
from pacopt import data, features


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def prepared(raw):
    return features.prepare(raw)


# ----------------------------------------------------------------------------- data contract
def test_raw_file_present_and_checksum():
    assert cfg.RAW_CSV.exists()
    assert data.sha256(cfg.RAW_CSV) == cfg.RAW_CSV_SHA256


def test_raw_schema(raw):
    assert raw.shape == (cfg.EXPECTED_ROWS, len(cfg.EXPECTED_COLUMNS))
    assert abs((raw["y"] == "yes").mean() - 0.117) < 0.002


def test_clean_maps_binaries_and_keeps_unknown(raw):
    c = data.clean(raw)
    for col in cfg.BINARY_RAW:
        assert set(c[col].unique()) <= {0, 1}
    assert "unknown" in c["contact"].cat.categories
    assert "admin" in c["job"].cat.categories  # "admin." normalised


def test_clean_rejects_unexpected_binary_values(raw):
    bad = raw.head(5).copy()
    bad.loc[bad.index[0], "loan"] = "maybe"
    with pytest.raises(data.DataValidationError):
        data.clean(bad)


# ----------------------------------------------------------------------------- RFM features
def test_rfm_recency():
    df = pd.DataFrame({"pdays": [-1, 30, 200, 400], "previous": [0, 1, 2, 3],
                       "poutcome": ["unknown", "success", "failure", "other"],
                       "balance": [-50, 0, 100, 5000], "age": [25, 40, 50, 80]})
    out = features.add_rfm_features(df)
    assert out["never_contacted"].tolist() == [1, 0, 0, 0]
    assert pd.isna(out["recency_days"].iloc[0]) and out["recency_days"].iloc[1] == 30
    assert out["recency_bucket"].astype(str).tolist() == ["never", "<=90d", "181-365d", ">365d"]
    assert out["prior_success"].tolist() == [0, 1, 0, 0]
    assert out["balance_slog"].iloc[0] < 0 < out["balance_slog"].iloc[3]
    assert out["age_band"].astype(str).tolist() == ["18-29", "30-44", "45-59", "75+"]


def test_frequency_uses_previous_not_campaign(prepared):
    feats, _ = prepared
    assert (feats["frequency"] == feats["previous"].clip(upper=cfg.FEATURES.frequency_cap)).all()


# ----------------------------------------------------------------------------- eligibility
def test_eligibility_rules():
    df = pd.DataFrame({"default": [1, 0, 0, 0, 0], "balance": [100, -1, 100, 100, 100],
                       "previous": [0, 0, 11, 0, 0], "age": [40, 40, 40, 80, 40]})
    out = features.add_eligibility(df)
    assert out["eligible"].tolist() == [0, 0, 0, 1, 1]          # 75+ is flagged, not excluded
    assert out["flag_age_75plus"].tolist() == [0, 0, 0, 1, 0]


def test_eligibility_toggles_off():
    df = pd.DataFrame({"default": [1], "balance": [-5], "previous": [20], "age": [30]})
    off = cfg.EligibilityConfig(exclude_default=False, exclude_negative_balance=False,
                                exclude_fatigue=False)
    assert features.add_eligibility(df, off)["eligible"].tolist() == [1]


# ----------------------------------------------------------------------------- leakage guard
def test_model_features_have_no_leakage():
    features.assert_no_leakage(features.MODEL_FEATURES)
    assert not set(features.MODEL_FEATURES) & cfg.FORBIDDEN_FEATURES


@pytest.mark.parametrize("col", sorted(cfg.FORBIDDEN_FEATURES))
def test_leakage_guard_rejects_forbidden(col):
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(features.MODEL_FEATURES + [col])


def test_model_features_exist(prepared):
    feats, _ = prepared
    assert set(features.MODEL_FEATURES) <= set(feats.columns)


# ----------------------------------------------------------------------------- split / universe
def test_split_sizes_and_stratification(prepared):
    feats, parts = prepared
    assert len(parts.train) + len(parts.test) == len(feats)
    assert abs(parts.train["y"].mean() - parts.test["y"].mean()) < 0.005
    assert len(parts.universe) == cfg.UNIVERSE_SIZE


def test_universe_is_out_of_sample(prepared):
    _, parts = prepared
    assert set(parts.universe["customer_id"]) <= set(parts.test["customer_id"])
    assert not set(parts.universe["customer_id"]) & set(parts.train["customer_id"])


def test_customer_ids_unique_and_stable(prepared):
    feats, _ = prepared
    assert feats["customer_id"].is_unique
    assert feats["customer_id"].iloc[0] == "C100001"


def test_no_unexpected_nulls(prepared):
    feats, _ = prepared
    nulls = feats.isna().sum()
    assert set(nulls[nulls > 0].index) <= {"recency_days"}           # NaN = never contacted, by design
    assert (feats["recency_days"].isna() == (feats["never_contacted"] == 1)).all()


def test_split_reproducible(raw):
    _, a = features.prepare(raw)
    _, b = features.prepare(raw)
    assert a.universe["customer_id"].tolist() == b.universe["customer_id"].tolist()
