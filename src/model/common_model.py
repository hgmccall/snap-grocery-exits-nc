"""Shared setup for the D3 model scripts.

Run the scripts from the repo root, e.g.
    uv run python src/model/01_evaluate.py
    uv run python src/model/02_error_analysis.py   (needs 01's output)
Both need data/processed/store_year_trainval.parquet from src/prep/05_features.py.

Project rules this file enforces:
- Only columns with role == "feature" in docs/column_inventory.csv are model inputs.
  `year`, qa_*, impact_*, exit_* and bias_* columns never are.
- Baseline and models are compared on identical rows: train 2012-2018 (ACS starts with
  the 2012 cohort), validation 2019-2021. The reserved test file is never read.
- Anything fitted (imputation, scaling, tuning) is fitted on training years only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"
TRAINVAL = PROCESSED / "store_year_trainval.parquet"

TRAIN_YEARS = range(2012, 2019)  # first cohort with ACS features through the last train year
VAL_YEARS = range(2019, 2022)
LABELS = ("exit_12m", "exit_12m_strict")
TOP_K_STATE = 50   # proposal: top 50 statewide per year
TOP_K_MECK = 10    # proposal: top 10 in Mecklenburg per year
SEED = 2026

CATEGORICAL = ["store_type"]


def feature_columns() -> list[str]:
    inv = pd.read_csv(DOCS / "column_inventory.csv")
    cols = inv.loc[inv["role"] == "feature", "column"].tolist()
    # Guard: nothing outcome-, year- or bias-related may slip in.
    bad = [c for c in cols if c == "year" or c.startswith(("qa_", "impact_", "exit_", "bias_"))]
    assert not bad, f"non-feature columns in feature list: {bad}"
    return cols


def load_rows() -> pd.DataFrame:
    """Train 2012-2018 and validation 2019-2021 rows. Never touches the test file."""
    df = pd.read_parquet(TRAINVAL)
    assert df["year"].max() <= 2021, "test-period rows found in the train/validation file"
    df = df[df["year"].isin(list(TRAIN_YEARS) + list(VAL_YEARS))].copy()
    df["part"] = np.where(df["year"].isin(TRAIN_YEARS), "train", "validation")

    # Nominal ACS income drifts upward across vintages, so a dollar value means something
    # different in 2012 than in 2021. Replace it with the store's income percentile among
    # stores in the same cohort year. This uses only Jan-1 information (no outcomes).
    df["median_hh_income"] = df.groupby("year")["median_hh_income"].rank(pct=True)

    for c in feature_columns():
        if c in CATEGORICAL:
            df[c] = df[c].astype("category")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def rows_for_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """exit_12m_strict is missing for batch exits; those rows are left out of strict analyses."""
    out = df[df[label].notna()].copy()
    out[label] = out[label].astype(int)
    return out


def top_k_flags(scores: pd.Series, groups: pd.Series, k: int, seed: int = SEED) -> pd.Series:
    """Flag the k highest scores within each group (cohort year). Ties broken at random
    with a fixed seed, so the baseline's tied store-type scores do not depend on row order."""
    rng = np.random.default_rng(seed)
    jitter = pd.Series(rng.random(len(scores)) * 1e-9, index=scores.index)
    s = scores + jitter
    rank = s.groupby(groups).rank(ascending=False, method="first")
    return rank <= k
