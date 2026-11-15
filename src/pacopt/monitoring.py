"""Data drift monitoring with Evidently, driven by SIMULATED drift (demo only).

There is no production pipeline, so there is no real "current" data. Instead the reference is the
training data and the current data is the campaign universe with a drift you dial in
(`DriftSettings`). With all sliders at zero the two samples come from the same population, which is
the baseline: Evidently should report little or no drift. Move a slider to see how the checks react.

Checks: data drift (per feature), prediction drift (propensity score) and data quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pandas.core.dtypes.common as _pdc
from pandas.api.types import pandas_dtype

from pacopt import config as cfg
from pacopt import features as ft
from pacopt import model as m

# Evidently 0.4.x imports a pandas helper that pandas 3 removed. Restore it before importing Evidently.
if not hasattr(_pdc, "infer_dtype_from_object"):
    _pdc.infer_dtype_from_object = lambda o: (np.dtype(o).type if isinstance(o, str) else pandas_dtype(o).type)

from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # noqa: E402
from evidently.metrics import ColumnDriftMetric  # noqa: E402
from evidently.report import Report  # noqa: E402

SAMPLE_SIZE = 3_000          # rows per side: keeps the embedded HTML report light
SIMULATION_LABEL = "Simulated drift (demo)"


@dataclass(frozen=True)
class DriftSettings:
    age_shift_years: float = 0.0           # every customer this many years older (or younger)
    balance_change_pct: float = 0.0        # every balance scaled by (1 + pct/100)
    unknown_contact_pct: float = 0.0       # % of cellular customers whose contact channel is lost ("unknown")
    prior_success_change_pct: float = 0.0  # relative change in customers whose previous campaign succeeded
    contacted_before_pct: float = 0.0      # % of never-contacted customers who have now been contacted before
    mix_shift_pct: float = 0.0             # % of working customers who become students / retirees
    loan_stress_pct: float = 0.0           # % more customers with a housing / personal loan (and a few defaults)

    @property
    def is_baseline(self) -> bool:
        return not any(asdict(self).values())


@dataclass
class DriftResult:
    settings: DriftSettings
    features: list[str]
    summary: pd.DataFrame                  # one row per feature: type, score, test, drifted
    n_columns: int
    n_drifted: int
    share_drifted: float
    dataset_drift: bool
    prediction_drift: bool
    prediction_score: float
    prediction_test: str
    reference: pd.DataFrame
    current: pd.DataFrame
    ref_scores: np.ndarray
    cur_scores: np.ndarray
    html: str

    def metrics(self) -> dict:
        return {"n_columns": self.n_columns, "n_drifted_columns": self.n_drifted,
                "share_drifted_columns": self.share_drifted, "dataset_drift": float(self.dataset_drift),
                "prediction_drift": float(self.prediction_drift), "prediction_drift_score": self.prediction_score,
                "mean_score_reference": float(self.ref_scores.mean()),
                "mean_score_current": float(self.cur_scores.mean())}


# ----------------------------------------------------------------------------- data
def _recat(out: pd.DataFrame, col: str, idx, values) -> None:
    """Set a categorical column to `values` at `idx`, keeping the original categories."""
    text = out[col].astype(str)
    text.loc[idx] = values
    out[col] = pd.Categorical(text, categories=list(out[col].cat.categories))


def apply_drift(df: pd.DataFrame, s: DriftSettings, fc: cfg.FeatureConfig = cfg.FEATURES) -> pd.DataFrame:
    """Apply the simulated shifts, recomputing every derived feature the model uses so the columns stay
    consistent (age -> age_band, balance -> balance_slog, poutcome -> prior_success,
    never_contacted <-> frequency / recency_bucket / poutcome)."""
    out = df.copy()
    rng = np.random.default_rng(cfg.RANDOM_STATE)

    def pick(idx: pd.Index, n: int) -> np.ndarray:
        return rng.choice(idx.to_numpy(), size=int(min(max(n, 0), len(idx))), replace=False)

    # --- customer profile
    if s.age_shift_years:
        out["age"] = (out["age"] + s.age_shift_years).round().clip(18, 95).astype(int)
        out["age_band"] = pd.cut(out["age"], bins=list(fc.age_bins), labels=list(fc.age_labels))
    if s.balance_change_pct:
        out["balance"] = (out["balance"] * (1 + s.balance_change_pct / 100)).round().astype(int)
        out["balance_slog"] = ft.signed_log(out["balance"])
    if s.mix_shift_pct:
        working = out.index[out["job"].astype(str).isin(["blue-collar", "admin", "services", "technician"])]
        chosen = pick(working, round(len(working) * s.mix_shift_pct / 100))
        students, retired = chosen[: len(chosen) // 2], chosen[len(chosen) // 2:]
        _recat(out, "job", students, "student")
        _recat(out, "education", students, "tertiary")
        _recat(out, "marital", students, "single")
        _recat(out, "job", retired, "retired")
        _recat(out, "marital", retired, "married")
    if s.loan_stress_pct:
        f = s.loan_stress_pct / 100
        for col, frac in (("housing", f), ("loan", f), ("default", f / 5)):   # defaults are rare: a fifth as many
            idx = out.index[out[col] == 0]
            out.loc[pick(idx, round(len(idx) * frac)), col] = 1

    # --- contact channel and campaign history
    if s.unknown_contact_pct:
        cellular = out.index[out["contact"].astype(str) == "cellular"]
        _recat(out, "contact", pick(cellular, round(len(cellular) * s.unknown_contact_pct / 100)), "unknown")
    if s.prior_success_change_pct:
        won = out.index[out["poutcome"].astype(str) == "success"]
        n = round(len(won) * abs(s.prior_success_change_pct) / 100)
        if s.prior_success_change_pct < 0:        # fewer successes: they become failures
            _recat(out, "poutcome", pick(won, n), "failure")
        else:                                     # more successes: converted from earlier failures
            lost = out.index[out["poutcome"].astype(str).isin(["failure", "other"])]
            _recat(out, "poutcome", pick(lost, n), "success")
    if s.contacted_before_pct:
        fresh = out.index[out["never_contacted"] == 1]
        chosen = pick(fresh, round(len(fresh) * s.contacted_before_pct / 100))
        seen = out.loc[out["never_contacted"] == 0, "recency_bucket"].astype(str).value_counts()
        labels = [x for x in fc.recency_labels]
        w = np.array([seen.get(x, 0) for x in labels], float)
        w = w / w.sum() if w.sum() else np.full(len(labels), 1 / len(labels))
        out.loc[chosen, "never_contacted"] = 0
        out.loc[chosen, "frequency"] = rng.integers(1, 4, size=len(chosen)).astype("int16")
        _recat(out, "recency_bucket", chosen, rng.choice(labels, size=len(chosen), p=w))
        _recat(out, "poutcome", chosen, "failure")   # contacted earlier without a recorded success
    out["prior_success"] = (out["poutcome"].astype(str) == "success").astype("int8")
    return out


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Evidently-friendly: categoricals become strings."""
    out = df[cols].copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
    return out


def model_features() -> list[str]:
    import json
    return json.loads(m.METRICS_PATH.read_text())["features"]


# ----------------------------------------------------------------------------- reports
def run_drift(s: DriftSettings, n: int = SAMPLE_SIZE) -> DriftResult:
    train, _, universe = m.load_processed()
    cols = model_features()
    bundle = m.load_model()
    model = bundle[bundle["selected"]]

    ref_raw = train.sample(min(n, len(train)), random_state=cfg.RANDOM_STATE)
    cur_raw = apply_drift(universe.sample(min(n, len(universe)), random_state=cfg.RANDOM_STATE), s)
    ref, cur = _clean(ref_raw, cols), _clean(cur_raw, cols)
    ref_scores, cur_scores = m.predict(model, ref_raw), m.predict(model, cur_raw)

    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    d = rep.as_dict()
    overall = next(x["result"] for x in d["metrics"] if "number_of_drifted_columns" in x["result"])
    table = next(x["result"] for x in d["metrics"] if "drift_by_columns" in x["result"])["drift_by_columns"]
    summary = pd.DataFrame([{"feature": c, "type": v["column_type"], "drift_score": v["drift_score"],
                             "test": v["stattest_name"], "drifted": bool(v["drift_detected"])}
                            for c, v in table.items()]).sort_values(["drifted", "drift_score"],
                                                                    ascending=[False, True]).reset_index(drop=True)

    pred = Report(metrics=[ColumnDriftMetric(column_name="propensity")])
    pred.run(reference_data=pd.DataFrame({"propensity": ref_scores}),
             current_data=pd.DataFrame({"propensity": cur_scores}))
    pr = pred.as_dict()["metrics"][0]["result"]

    return DriftResult(
        settings=s, features=cols, summary=summary, n_columns=int(overall["number_of_columns"]),
        n_drifted=int(overall["number_of_drifted_columns"]), share_drifted=float(overall["share_of_drifted_columns"]),
        dataset_drift=bool(overall["dataset_drift"]), prediction_drift=bool(pr["drift_detected"]),
        prediction_score=float(pr["drift_score"]), prediction_test=str(pr["stattest_name"]),
        reference=ref, current=cur, ref_scores=np.asarray(ref_scores), cur_scores=np.asarray(cur_scores),
        html=rep.get_html())


def quality_html(result: DriftResult) -> str:
    """Evidently data-quality report for the current (drifted) data against the reference."""
    rep = Report(metrics=[DataQualityPreset()])
    rep.run(reference_data=result.reference, current_data=result.current)
    return rep.get_html()


def settings_params(s: DriftSettings) -> dict:
    return {**asdict(s), "simulated": True, "sample_size": SAMPLE_SIZE}
