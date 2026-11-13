"""Download, load, validate and clean the UCI Bank Marketing dataset."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from pacopt import config as cfg


class DataValidationError(ValueError):
    """Raised when the raw data does not match the expected contract."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def download_raw(dest: Path = cfg.RAW_CSV, url: str = cfg.UCI_ZIP_URL, force: bool = False) -> Path:
    """Download the UCI archive and extract `bank-full.csv` (nested inside `bank.zip`).

    Only the CSV is read from the archive; nothing in it is executed.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed, trusted URL)
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    dest.write_bytes(inner.read("bank-full.csv"))
    digest = sha256(dest)
    if digest != cfg.RAW_CSV_SHA256:
        raise DataValidationError(f"Checksum mismatch for {dest.name}: {digest}")
    return dest


def load_raw(path: Path = cfg.RAW_CSV) -> pd.DataFrame:
    """Load the raw semicolon-separated CSV and validate its schema."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run pacopt.data.download_raw() first.")
    df = pd.read_csv(path, sep=";")
    validate_raw(df)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    if list(df.columns) != cfg.EXPECTED_COLUMNS:
        raise DataValidationError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != cfg.EXPECTED_ROWS:
        raise DataValidationError(f"Expected {cfg.EXPECTED_ROWS} rows, got {len(df)}")
    if df.isna().any().any():
        raise DataValidationError("Raw data contains nulls")
    pos_rate = (df[cfg.TARGET].str.strip().str.lower() == "yes").mean()
    if not 0.10 <= pos_rate <= 0.13:
        raise DataValidationError(f"Unexpected positive rate {pos_rate:.3f}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise categoricals and map binary yes/no fields to 0/1.

    `"unknown"` is kept as its own category: missingness is informative here
    (e.g. contact == "unknown" marks customers without a recorded phone channel).
    """
    out = df.copy()
    for col in cfg.CATEGORICAL_RAW + cfg.BINARY_RAW:
        out[col] = out[col].astype(str).str.strip().str.lower()
    out["job"] = out["job"].str.replace(".", "", regex=False)  # "admin." -> "admin"
    for col in cfg.BINARY_RAW:
        bad = set(out[col].unique()) - {"yes", "no"}
        if bad:
            raise DataValidationError(f"Unexpected values in {col}: {bad}")
        out[col] = (out[col] == "yes").astype("int8")
    for col in cfg.CATEGORICAL_RAW:
        out[col] = out[col].astype("category")
    return out
