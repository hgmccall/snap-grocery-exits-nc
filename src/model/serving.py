"""Scoring code shared by the rebuild script (09_final_refit.py) and the D4 API: load the bundle, score store-years, self-check.

models/pipeline.joblib is one joblib dict (the HW4 layout):
  "pipeline"             fitted sklearn Pipeline: prep = store_prep.StoreFeaturePrep -> hgb = the frozen HistGradientBoostingClassifier
  "metadata"             steps, created (UTC), sklearn_version, features, training_years, tier_cutoffs, ... (also written to models/metadata.json)
  "reference_contributions"  mean per-feature log-odds contribution over the refit rows (centres the explanations)
  "impact_reference"     weights and sorted pooled reference values for the impact score
  "examples"             fixed example inputs with expected outputs (same content as docs/model_examples_d4.json)
Unpickling needs `store_prep` importable (src/model/store_prep.py) and the pinned scikit-learn. Only load a file you built yourself (pickle).

Callers pass RAW rows: `year` + the 20 model inputs (median_hh_income in dollars) + `dist_other_grocery_jan1_mi` (role `context`, used only for
the impact score). The Pipeline does all input preparation; nothing is re-implemented here.
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
BUNDLE_FILE = "pipeline.joblib"
SUM_TOL = 1e-9          # contributions must add up to the raw score this closely
MATCH_TOL = 1e-9        # local-vs-deployed: outputs must match the stored expected values this closely
CONTEXT = "dist_other_grocery_jan1_mi"


class Bundle:
    def __init__(self, directory: Path = MODELS):
        self.dir = Path(directory)
        b = joblib.load(self.dir / BUNDLE_FILE)
        self.pipeline = b["pipeline"]
        self.meta = b["metadata"]
        self.examples = b["examples"]
        self.reference = pd.Series(b["reference_contributions"])
        ref = b["impact_reference"]
        self.rankers = {k: Ranker(ref[k]) for k in ("dist", "snap", "veh")}
        self.lo, self.hi = self.meta["tier_cutoffs"]["low"], self.meta["tier_cutoffs"]["high"]

    def tier(self, p) -> np.ndarray:
        p = np.asarray(p)
        return np.where(p < self.lo, "low", np.where(p < self.hi, "medium", "high"))

    def score(self, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """X: raw rows with `year`, the 20 inputs and `dist_other_grocery_jan1_mi`. Returns probability, tier, impact, priority, reasons
        (medium and high tier only). `.attrs` carries the baseline and the per-feature contributions."""
        p = self.pipeline.predict_proba(X)[:, 1]
        raw = raw_score(self.pipeline, X)
        base, C = contributions(self.pipeline, X)
        tiers = self.tier(p)
        Xp = self.pipeline[:-1].transform(X)    # the transformer's output: percentile income, imputed unemployment
        imp = impact_score(X[CONTEXT].to_numpy(dtype=float), Xp["pct_snap_hh"].to_numpy(), Xp["pct_no_vehicle"].to_numpy(), self.rankers)
        out = pd.DataFrame({"raw_score": raw, "probability": p, "tier": tiers, "impact": imp, "priority": p * imp,
                            "reasons": display_reasons(self.pipeline, X, self.reference, tiers, k)}, index=X.index)
        out.attrs["baseline"], out.attrs["contributions"] = base, C
        return out

    def self_check(self, examples: list[dict] | None = None) -> tuple[bool, list[str]]:
        """Run at API startup; the API must answer 503 until this returns ok. Fails if (1) the installed scikit-learn is not the pinned version,
        (2) the pipeline is not prep -> trees, (3) on any fixed example baseline + sum(contributions) differs from the raw score by more than
        1e-9, or (4) any example's raw score, probability, impact or priority differs from the stored expected value by more than 1e-9, or its
        tier or reasons differ (local-vs-deployed match)."""
        problems: list[str] = []
        if sklearn.__version__ != self.meta["sklearn_version"]:
            problems.append(f"scikit-learn {sklearn.__version__} installed, bundle pinned to {self.meta['sklearn_version']}")
        examples = self.examples if examples is None else examples
        try:
            if [n for n, _ in self.pipeline.steps] != ["prep", "hgb"]:
                problems.append(f"unexpected pipeline steps {[n for n, _ in self.pipeline.steps]}")
            X = pd.DataFrame([{"year": e["cohort_year"], **e["inputs"], **e["context"]} for e in examples])
            out = self.score(X)
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
                if [(x["feature"], x["text"]) for x in ex["reasons"]] != [(x["feature"], x["text"]) for x in r["reasons"]]:
                    problems.append(f"example {e['id']}: reasons differ from expected")
        except Exception as exc:  # any failure to run the check is a failed check
            problems.append(f"self-check raised {type(exc).__name__}: {exc}")
        return (not problems), problems
