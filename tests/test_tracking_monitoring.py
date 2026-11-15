"""MLflow tracking and Evidently drift monitoring (simulated drift)."""
import mlflow
import pandas as pd
import pytest

from pacopt import monitoring as mo
from pacopt import tracking


def test_baseline_has_no_drift_and_shifts_are_detected():
    base = mo.run_drift(mo.DriftSettings())
    assert base.settings.is_baseline and base.n_drifted == 0 and not base.dataset_drift
    shifted = mo.run_drift(mo.DriftSettings(age_shift_years=15, unknown_contact_pct=50))
    flagged = set(shifted.summary.loc[shifted.summary["drifted"], "feature"])
    assert {"age", "contact"} <= flagged
    assert shifted.n_drifted > base.n_drifted


def test_apply_drift_recomputes_derived_features():
    _, _, universe = __import__("pacopt.model", fromlist=["x"]).load_processed()
    out = mo.apply_drift(universe, mo.DriftSettings(age_shift_years=10, balance_change_pct=100))
    assert (out["age"] >= universe["age"]).all() and out["age"].max() <= 95
    assert out["balance_slog"].mean() > universe["balance_slog"].mean()
    assert out["age_band"].astype(str).ne(universe["age_band"].astype(str)).any()


def test_log_drift_writes_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(tracking, "ARTIFACT_ROOT", tmp_path / "artifacts")
    run_id = tracking.log_drift({"age_shift_years": 5}, {"n_drifted_columns": 2},
                                "<html></html>", pd.DataFrame({"feature": ["age"], "drifted": [True]}))
    assert run_id
    run = tracking.client().get_run(run_id)
    assert run.data.metrics["n_drifted_columns"] == 2 and run.data.tags["simulated"] == "True"
    mlflow.set_tracking_uri(None)


def test_tracking_failures_do_not_raise(monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", "sqlite:////nonexistent_dir/x/y.db")
    with pytest.warns(UserWarning, match="MLflow tracking skipped"):
        assert tracking.log_drift({}, {}, "", pd.DataFrame()) is None


def test_new_shifts_keep_derived_columns_consistent_and_combine_into_dataset_drift():
    from pacopt import model as m
    _, _, universe = m.load_processed()
    st = mo.DriftSettings(prior_success_change_pct=-60, contacted_before_pct=50, mix_shift_pct=30, loan_stress_pct=30)
    out = mo.apply_drift(universe.sample(3000, random_state=1), st)
    never = out["never_contacted"] == 1
    assert (never == (out["recency_bucket"].astype(str) == "never")).all()
    assert (out["prior_success"] == (out["poutcome"].astype(str) == "success")).all()
    assert (out.loc[never, "frequency"] == 0).all() and (out.loc[~never, "poutcome"].astype(str) != "unknown").all()
    big = mo.DriftSettings(age_shift_years=10, balance_change_pct=-40, unknown_contact_pct=40, prior_success_change_pct=-60,
                           contacted_before_pct=50, mix_shift_pct=30, loan_stress_pct=30)
    r = mo.run_drift(big)
    assert r.dataset_drift and r.n_drifted >= 8
