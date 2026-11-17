"""Scoring code shared by the rebuild script (09_final_refit.py) and the D4 API: load the saved bundle, score store-years, self-check.

Bundle (models/, rebuilt by 09_final_refit.py, gitignored):
  trees_exit12m_2012_2021.joblib   bare HistGradientBoostingClassifier (no custom classes, so it unpickles with sklearn alone)
  bundle.json                      sklearn version, feature order, store_type categories, unemployment median, tier cutoffs, reference contributions
  impact_reference.json            sorted pooled reference values and weights for the impact score
  examples.json                    fixed example inputs with expected outputs (same file as docs/model_examples_d4.json)
Only load a joblib file you built yourself (pickle).

The caller supplies one row per store-year with the 20 feature columns (median_hh_income already converted to its within-cohort-year
percentile, see `income_percentile_within_year`) plus `dist_other_grocery_jan1_mi` (role `context`, not a model input).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from explain import contributions, display_reasons, raw_score
from impact import Ranker, impact_score

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SUM_TOL = 1e-9          # contributions must add up to the raw score this closely
MATCH_TOL = 1e-9        # local-vs-deployed: outputs must match the stored expected values this closely


def income_percentile_within_year(df: pd.DataFrame) -> pd.Series:
    """median_hh_income (dollars) -> percentile rank among all grocery store-years of the same cohort year (pandas rank(pct=True), 'average' ties).
    Same transformation as common_model.load_rows; computed over every store-year of that cohort, not over the rows being scored."""
    return df.groupby("year")["median_hh_income"].rank(pct=True)


class Bundle:
    def __init__(self, directory: Path = MODELS):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "bundle.json").read_text())
        self.model = joblib.load(self.dir / "trees_exit12m_2012_2021.joblib")
        ref = json.loads((self.dir / "impact_reference.json").read_text())
        self.rankers = {k: Ranker(ref[k]) for k in ("dist", "snap", "veh")}
        self.reference = pd.Series(self.meta["reference_contributions"])
        self.features = self.meta["features"]
        self.lo, self.hi = self.meta["low_cutoff"], self.meta["high_cutoff"]

    # ------------------------------------------------------------------ inputs
    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features].copy()
        cats = self.meta["store_type_categories"]
        bad = sorted(set(X["store_type"].dropna().astype(str)) - set(cats))
        if bad or X["store_type"].isna().any():
            raise ValueError(f"store_type must be one of {cats}; got {bad or 'missing'}")
        X["store_type"] = pd.Categorical(X["store_type"].astype(str), categories=cats)
        for c in self.features:
            if c != "store_type":
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        # same imputation as training: the training median, so a missing value cannot mark cohort 2012 (see docs/changes_since_d3.md)
        X["unemployment_rate"] = X["unemployment_rate"].fillna(self.meta["unemployment_median"])
        return X

    def tier(self, p) -> np.ndarray:
        p = np.asarray(p)
        return np.where(p < self.lo, "low", np.where(p < self.hi, "medium", "high"))

    # ------------------------------------------------------------------ scoring
    def score(self, X: pd.DataFrame, dist_other_grocery_jan1_mi, k: int = 3) -> pd.DataFrame:
        Xp = self.prepare(X)
        raw = raw_score(self.model, Xp)
        base, C = contributions(self.model, Xp)
        p = self.model.predict_proba(Xp)[:, 1]
        tiers = self.tier(p)
        imp = impact_score(np.asarray(dist_other_grocery_jan1_mi, dtype=float), Xp["pct_snap_hh"].to_numpy(), Xp["pct_no_vehicle"].to_numpy(), self.rankers)
        out = pd.DataFrame({"raw_score": raw, "probability": p, "tier": tiers, "impact": imp, "priority": p * imp,
                            "reasons": display_reasons(self.model, Xp, self.reference, tiers, k)}, index=X.index)
        out.attrs["baseline"], out.attrs["contributions"] = base, C
        return out

    # ------------------------------------------------------------------ startup self-check
    def self_check(self, examples: list[dict] | None = None) -> tuple[bool, list[str]]:
        """Run at API startup; the API must answer 503 until this returns ok. Fails if (1) the installed scikit-learn is not the pinned version,
        (2) on any fixed example baseline + sum(contributions) differs from the raw score by more than 1e-9, or (3) any example's raw score,
        probability, tier, impact, priority or reasons differ from the stored expected outputs (local-vs-deployed match)."""
        problems: list[str] = []
        if sklearn.__version__ != self.meta["sklearn_version"]:
            problems.append(f"scikit-learn {sklearn.__version__} installed, bundle pinned to {self.meta['sklearn_version']}")
        examples = examples if examples is not None else json.loads((self.dir / "examples.json").read_text())["examples"]
        try:
            X = pd.DataFrame([{**e["inputs"]} for e in examples], columns=self.features)
            dist = [e["context"]["dist_other_grocery_jan1_mi"] for e in examples]
            out = self.score(X, dist)
            base, C = out.attrs["baseline"], out.attrs["contributions"]
            gap = np.abs(base + C.sum(axis=1).to_numpy() - out["raw_score"].to_numpy()).max()
            if not gap <= SUM_TOL:
                problems.append(f"contributions do not sum to the raw score (max abs difference {gap:.3e} > {SUM_TOL})")
            for i, e in enumerate(examples):
                ex, r = e["expected"], out.iloc[i]
                for key in ("raw_score", "probability", "impact", "priority"):
                    if not abs(ex[key] - r[key]) <= MATCH_TOL:
                        problems.append(f"example {e['id']}: {key} {r[key]!r} != expected {ex[key]!r}")
                if ex["tier"] != r["tier"]:
                    problems.append(f"example {e['id']}: tier {r['tier']} != expected {ex['tier']}")
                if [x["feature"] for x in ex["reasons"]] != [x["feature"] for x in r["reasons"]] or [x["text"] for x in ex["reasons"]] != [x["text"] for x in r["reasons"]]:
                    problems.append(f"example {e['id']}: reasons differ from expected")
        except Exception as exc:  # any failure to run the check is a failed check
            problems.append(f"self-check raised {type(exc).__name__}: {exc}")
        return (not problems), problems
