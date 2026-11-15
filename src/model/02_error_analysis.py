"""D3 error analysis and feature associations for the selected model (logistic regression),
with the store-type baseline alongside, on validation 2019-2021 (exit_12m).

Reads data/processed/val_predictions.parquet from 01_evaluate.py. Flags follow the proposal's
use case: the 50 highest-scoring stores statewide in each cohort year are "flagged".
For each group: store-years, exits, flagged, exits caught (recall), precision among flagged,
share of non-exits flagged (false-positive rate), and mean predicted vs observed exit rate
(calibration). Groups are small; n is always reported.

bias_* columns are used here only to define groups, never as model inputs (project rules).

Also writes logistic regression coefficients (standardized inputs) and permutation importance
(drop in validation PR-AUC) for both models: these are associations, not causes.

Writes docs/model_error_analysis.md and docs/model_error_analysis.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from common_model import DOCS, PROCESSED, TOP_K_STATE, feature_columns, load_rows, rows_for_label, top_k_flags  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).parent / "01_evaluate.py")
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)

LABEL = "exit_12m"
MODELS = {"p_baseline": "Baseline", "p_logreg": "Logistic regression"}


def groups(v: pd.DataFrame) -> dict[str, pd.Series]:
    race_cols = {"bias_pct_white_nh": "White majority", "bias_pct_black_nh": "Black majority",
                 "bias_pct_hispanic": "Hispanic majority"}
    r = v[list(race_cols)]
    maj = pd.Series("No majority group", index=v.index)
    for c, name in race_cols.items():
        maj[r[c] > 0.5] = name
    maj[r.isna().all(axis=1)] = "Missing ACS"
    inc = pd.qcut(v["median_hh_income"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    age = pd.cut(v["store_age_years"], [-0.01, 5, 15, np.inf], labels=["<5 years", "5-15 years", ">15 years"])
    return {
        "Store type": v["store_type"].astype(str),
        "Urban-rural (RUCA)": v["ruca_group"].fillna("Missing").astype(str),
        "County": np.where(v["county"] == "MECKLENBURG", "Mecklenburg", "Rest of NC"),
        "Tract income quintile (within year)": inc.astype(str),
        "Tract majority race/ethnicity": maj,
        "Store age": age.astype(str),
    }


def group_table(v):
    rows = []
    flags = {m: top_k_flags(v[m], v["year"], TOP_K_STATE) for m in MODELS}
    y = v["y"].astype(bool)
    for gname, g in groups(v).items():
        g = pd.Series(g, index=v.index).astype(object).where(pd.notna(g), "Missing").astype(str)
        for level in sorted(pd.unique(g)):
            idx = g == level
            for m, mname in MODELS.items():
                f = flags[m][idx]; yy = y[idx]
                rows.append({
                    "grouping": gname, "group": level, "model": mname,
                    "store_years": int(idx.sum()), "exits": int(yy.sum()),
                    "flagged": int(f.sum()), "caught": int((f & yy).sum()),
                    "recall": (f & yy).sum() / yy.sum() if yy.sum() else np.nan,
                    "precision": (f & yy).sum() / f.sum() if f.sum() else np.nan,
                    "fpr": (f & ~yy).sum() / (~yy).sum() if (~yy).sum() else np.nan,
                    "mean_pred": float(v.loc[idx, m].mean()), "observed": float(yy.mean()),
                })
    # batch vs non-batch: recall only (exit_is_batch is defined for exits)
    ex = v[y]
    for level, sub in ex.groupby(ex["exit_is_batch"].map({True: "Batch exit", False: "Non-batch exit"})):
        for m, mname in MODELS.items():
            f = flags[m][sub.index]
            rows.append({"grouping": "Exit kind", "group": level, "model": mname, "store_years": len(sub),
                         "exits": len(sub), "flagged": int(f.sum()), "caught": int(f.sum()),
                         "recall": f.mean(), "precision": np.nan, "fpr": np.nan,
                         "mean_pred": float(sub[m].mean()), "observed": 1.0})
    return pd.DataFrame(rows)


def importance():
    df = load_rows()
    d = rows_for_label(df, LABEL)
    tr, va = d[d["part"] == "train"], d[d["part"] == "validation"]
    F = feature_columns()
    best = {}
    for name in ("Logistic regression", "Gradient-boosted trees"):
        b, _ = ev.tune(name, tr, LABEL)
        best[name] = ev.BUILDERS[name](b).fit(tr[F], tr[LABEL])
    lr = best["Logistic regression"]
    names = lr.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["lr"].coef_[0], index=names)
    coef.index = [i.split("__", 1)[1] for i in coef.index]
    perm = {}
    for name, m in best.items():
        p = permutation_importance(m, va[F], va[LABEL], scoring="average_precision", n_repeats=20,
                                   random_state=2026)
        perm[name] = pd.Series(p.importances_mean, index=F)
    return coef, pd.DataFrame(perm)


def main():
    pr = pd.read_parquet(PROCESSED / "val_predictions.parquet")
    v = pr[pr["label_name"] == LABEL].copy()
    t = group_table(v)
    t.to_csv(DOCS / "model_error_analysis.csv", index=False)
    coef, perm = importance()

    L = ["# D3 error analysis (validation 2019-2021, exit_12m)", "",
         "Generated by `src/model/02_error_analysis.py`; do not edit by hand. \"Flagged\" = among the 50 "
         "highest-scoring stores statewide in its cohort year (150 flags over 3 years). Recall = exits flagged / "
         "exits; FPR = non-exits flagged / non-exits; calibration compares the mean predicted probability with "
         "the observed exit rate. Race/ethnicity groups use the tract's bias_* shares (> 50% = majority); these "
         "columns are never model inputs. Small groups: read n before rates.", ""]
    for gname, sub in t.groupby("grouping", sort=False):
        L += [f"## {gname}", "",
              "| Group | Model | Store-years | Exits | Flagged | Caught (recall) | Precision | FPR | Mean predicted vs observed |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, x in sub.iterrows():
            prec = "" if np.isnan(x["precision"]) else f"{x['precision']:.0%}"
            fpr = "" if np.isnan(x["fpr"]) else f"{x['fpr']:.1%}"
            rec = "" if np.isnan(x["recall"]) else f"{x['caught']} ({x['recall']:.0%})"
            L.append(f"| {x['group']} | {x['model']} | {x['store_years']:,} | {x['exits']} | {x['flagged']} | {rec} | "
                     f"{prec} | {fpr} | {x['mean_pred']:.1%} vs {x['observed']:.1%} |")
        L.append("")
    L += ["## Logistic regression coefficients (standardized inputs; log1p for counts)", "",
          "Positive = higher exit risk holding the other inputs fixed. Associations, not causes.", "",
          "| Input | Coefficient |", "|---|---|"]
    for k, val in coef.sort_values(key=abs, ascending=False).items():
        L.append(f"| {k} | {val:+.3f} |")
    L += ["", "## Permutation importance (drop in validation PR-AUC when the input is shuffled; 20 repeats)", "",
          "| Input | Logistic regression | Gradient-boosted trees |", "|---|---|---|"]
    for k, x in perm.sort_values("Logistic regression", ascending=False).iterrows():
        L.append(f"| {k} | {x['Logistic regression']:+.4f} | {x['Gradient-boosted trees']:+.4f} |")
    (DOCS / "model_error_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print((DOCS / "model_error_analysis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
