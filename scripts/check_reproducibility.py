"""Clean-clone reproducibility check (plan 6.1).

Copies only the files git would commit (no data/processed, no artifacts, no .venv) into a fresh
folder, runs the full pipeline there from the raw CSV, and compares its outputs with this
checkout's. Uses the current interpreter (the project .venv) instead of a second `uv sync`, so it
needs ~50 MB of disk rather than ~700 MB; `uv lock --check` separately confirms the lockfile.

    uv run python scripts/check_reproducibility.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("metrics.json", "strategy_comparison.csv", "ga_target_list.csv", "ga_history.csv",
             "optimisation_meta.json", "strategy_selections.parquet")


def tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git:
        out = subprocess.run([git, "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        return [ROOT / f for f in out if f]
    # fallback without git: the folders a clone contains (processed data and artifacts are git-ignored)
    keep = ["src", "data/raw", "app", "tests", "scripts", "pyproject.toml", "uv.lock", ".python-version"]
    files = []
    for k in keep:
        p = ROOT / k
        files += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts] if p.is_dir() else [p]
    return files


def compare(clone: Path) -> pd.DataFrame:
    rows = []
    a, b = (json.loads((r / "artifacts" / "metrics.json").read_text()) for r in (ROOT, clone))
    for k in ("roc_auc", "pr_auc", "brier", "max_calibration_gap_pp"):
        rows.append(("metrics: LR " + k, a["lr"][k], b["lr"][k], abs(a["lr"][k] - b["lr"][k]) < 1e-9))
        rows.append(("metrics: RF " + k, a["rf"][k], b["rf"][k], abs(a["rf"][k] - b["rf"][k]) < 1e-5))
    ta, tb = (pd.read_parquet(r / "data" / "processed" / "campaign_table.parquet") for r in (ROOT, clone))
    rows.append(("campaign table shape", str(ta.shape), str(tb.shape), ta.shape == tb.shape))
    rows.append(("campaign table EP (max abs diff)", 0.0, float((ta.expected_profit - tb.expected_profit).abs().max()),
                 np.allclose(ta.expected_profit, tb.expected_profit)))
    rows.append(("segments identical", "", "", bool((ta.segment == tb.segment).all())))
    sa, sb = (pd.read_csv(r / "artifacts" / "strategy_comparison.csv", index_col=0) for r in (ROOT, clone))
    for s in sa.index:
        x, y = sa.loc[s, "expected_contribution"], sb.loc[s, "expected_contribution"]
        rows.append((f"strategy: {s}", round(x, 2), round(y, 2), abs(x - y) < 1e-6))
    ma, mb = (json.loads((r / "artifacts" / "optimisation_meta.json").read_text())["fingerprint"] for r in (ROOT, clone))
    rows.append(("optimisation fingerprint", ma, mb, ma == mb))
    return pd.DataFrame(rows, columns=["check", "this checkout", "clean clone", "match"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temporary clone")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="pacopt_clone_"))
    try:
        files = tracked_files()
        for f in files:
            dst = tmp / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        assert not (tmp / "artifacts").exists() and not (tmp / "data" / "processed").exists()
        print(f"Clean clone: {len(files)} files -> {tmp}")
        t0 = time.time()
        boot = (f"import sys, runpy; sys.path.insert(0, {str(tmp / 'src')!r}); "
                "import pacopt; assert pacopt.__file__.startswith(" + repr(str(tmp)) + "), pacopt.__file__; "
                "runpy.run_module('pacopt.pipeline', run_name='__main__')")
        r = subprocess.run([sys.executable, "-c", boot], cwd=tmp, capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode:
            print(r.stderr[-3000:])
            return r.returncode
        print(f"Pipeline from raw CSV in the clone: {time.time() - t0:.0f}s")
        report = compare(tmp)
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(report.to_string(index=False))
        ok = bool(report["match"].all())
        print("REPRODUCED" if ok else "MISMATCH")
        return 0 if ok else 1
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
