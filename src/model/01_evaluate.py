"""D3 model comparison: store-type baseline vs logistic regression vs gradient-boosted trees.

Same rows, same validation method, same metrics for all three:
- Tuning: rolling-origin folds inside 2012-2018 (fit on 2012..k-1, score year k, k = 2015-2018).
- Final: fit on 2012-2018, score validation 2019-2021. The test file is never read.
- Metrics: PR-AUC (average precision, primary), ROC-AUC, Brier score, precision/recall in the
  top 50 statewide per year and top 10 in Mecklenburg per year (pooled over the 3 years),
  and how many exits caught in the top 50 left no other SNAP grocery within a mile
  (impact_dist_nearest_other_grocery_mi, used only for scoring, never as a feature).
- 95% intervals: bootstrap resampling stores (record_id), so repeated years of one store
  stay together.

Writes docs/model_results.md, docs/model_results.csv, figures/model_pr_curves.png and
data/processed/val_predictions.parquet (gitignored).
"""

import sys
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common_model import (  # noqa: E402
    CATEGORICAL, DOCS, FIGURES, LABELS, PROCESSED, SEED, TOP_K_MECK, TOP_K_STATE,
    feature_columns, load_rows, rows_for_label, top_k_flags,
)

N_BOOT = 1000
FEATURES = feature_columns()
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# Counts and population are right-skewed; log1p keeps a few large values from dominating
# the logistic regression. Trees do not need this.
SKEWED = [c for c in NUMERIC if c.startswith("n_") or c in ("chain_size", "acs_population",
          "pop_density_sqmi", "convenience_to_grocery_ratio_1mi", "store_age_years")]


# ---------------------------------------------------------------- models
class StoreTypeRate:
    """Baseline: each store type's exit rate in the training rows."""

    def fit(self, X, y):
        self.rates_ = pd.Series(np.asarray(y), index=X.index).groupby(X["store_type"], observed=True).mean()
        self.overall_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = X["store_type"].map(self.rates_).astype(float).fillna(self.overall_).to_numpy()
        return np.column_stack([1 - p, p])


def make_logreg(C):
    log_cols = [c for c in NUMERIC if c in SKEWED]
    lin_cols = [c for c in NUMERIC if c not in SKEWED]
    num = lambda: Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler())])
    pre = ColumnTransformer([
        ("log", Pipeline([("log1p", FunctionTransformer(lambda a: np.log1p(np.clip(a, 0, None)),
                                                        feature_names_out="one-to-one")),
                          ("rest", num())]), log_cols),
        ("lin", num(), lin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=5000))])


def make_hgb(lr, leaves, min_leaf, l2):
    return HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf, l2_regularization=l2,
        max_iter=300, early_stopping=False, categorical_features="from_dtype", random_state=SEED)


GRIDS = {
    "Logistic regression": [dict(C=c) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
    "Gradient-boosted trees": [dict(lr=a, leaves=b, min_leaf=c, l2=d) for a, b, c, d in
                               product((0.03, 0.1), (8, 16), (40, 100), (0.0, 1.0))],
}
BUILDERS = {"Logistic regression": lambda p: make_logreg(**p),
            "Gradient-boosted trees": lambda p: make_hgb(**p)}
SHORT = {"Store-type baseline": "baseline", "Logistic regression": "logreg",
         "Gradient-boosted trees": "hgb"}


def tune(name, train, label):
    """Rolling-origin folds inside the training years: fit 2012..k-1, score k."""
    rows = []
    for params in GRIDS[name]:
        aps = []
        for k in (2015, 2016, 2017, 2018):
            fit, ev = train[train["year"] < k], train[train["year"] == k]
            m = BUILDERS[name](params).fit(fit[FEATURES], fit[label])
            aps.append(average_precision_score(ev[label], m.predict_proba(ev[FEATURES])[:, 1]))
        rows.append({**params, "mean_ap": np.mean(aps)})
    res = pd.DataFrame(rows).sort_values("mean_ap", ascending=False)
    best = {k: v for k, v in res.iloc[0].items() if k != "mean_ap"}
    if "leaves" in best:
        best["leaves"] = int(best["leaves"]); best["min_leaf"] = int(best["min_leaf"])
    return best, res


def baseline_folds(train, label):
    aps = []
    for k in (2015, 2016, 2017, 2018):
        fit, ev = train[train["year"] < k], train[train["year"] == k]
        m = StoreTypeRate().fit(fit[FEATURES], fit[label])
        aps.append(average_precision_score(ev[label], m.predict_proba(ev[FEATURES])[:, 1]))
    return float(np.mean(aps))


# ---------------------------------------------------------------- metrics
def topk_stats(val, score_col, label):
    st = top_k_flags(val[score_col], val["year"], TOP_K_STATE)
    meck = val["county"] == "MECKLENBURG"
    mk = pd.Series(False, index=val.index)
    mk[meck] = top_k_flags(val.loc[meck, score_col], val.loc[meck, "year"], TOP_K_MECK)
    y = val[label].astype(bool)
    gap = val["impact_dist_nearest_other_grocery_mi"] > 1
    return {
        "top50_precision": (y & st).sum() / st.sum(),
        "top50_recall": (y & st).sum() / y.sum(),
        "top50_exits": int((y & st).sum()),
        "top50_gap_exits": int((y & st & gap).sum()),
        "gap_exits_total": int((y & gap).sum()),
        "meck_top10_precision": (y & mk).sum() / mk.sum(),
        "meck_top10_recall": (y & mk).sum() / (y & meck).sum(),
        "meck_top10_exits": int((y & mk).sum()),
    }


def bootstrap(val, cols, label, n=N_BOOT):
    """Store-level bootstrap of average precision for each model and each model minus baseline."""
    rng = np.random.default_rng(SEED)
    ids = val["record_id"].unique()
    idx_by_id = val.groupby("record_id").indices
    y = val[label].to_numpy()
    S = {c: val[c].to_numpy() for c in cols}
    out = {c: [] for c in cols}
    for _ in range(n):
        take = np.concatenate([idx_by_id[i] for i in rng.choice(ids, len(ids), replace=True)])
        if y[take].sum() == 0:
            continue
        for c in cols:
            out[c].append(average_precision_score(y[take], S[c][take]))
    return {c: np.array(v) for c, v in out.items()}


# ---------------------------------------------------------------- main
def main():
    df = load_rows()
    all_rows, tuning_notes, curves, preds_out, pair_notes = [], [], {}, [], []
    for label in LABELS:
        d = rows_for_label(df, label)
        train, val = d[d["part"] == "train"], d[d["part"] == "validation"].copy()
        print(f"\n== {label}: train {len(train):,} rows / {train[label].sum()} exits; "
              f"validation {len(val):,} / {val[label].sum()}")

        models = {"Store-type baseline": StoreTypeRate()}
        fold_ap = {"Store-type baseline": baseline_folds(train, label)}
        for name in GRIDS:
            best, res = tune(name, train, label)
            fold_ap[name] = float(res["mean_ap"].iloc[0])
            tuning_notes.append((label, name, best, res))
            models[name] = BUILDERS[name](best)
            print(f"  tuned {name}: {best} (fold PR-AUC {fold_ap[name]:.3f})")

        score_cols = []
        for name, m in models.items():
            m.fit(train[FEATURES], train[label])
            col = "p_" + SHORT[name]
            val[col] = m.predict_proba(val[FEATURES])[:, 1]
            score_cols.append(col)

        boot = bootstrap(val, score_cols, label)
        base = "p_baseline"
        hl = boot["p_hgb"] - boot["p_logreg"]
        pair_notes.append((label, np.percentile(hl, 2.5), np.percentile(hl, 97.5), float((hl > 0).mean())))
        for name in models:
            col = "p_" + SHORT[name]
            ap = average_precision_score(val[label], val[col])
            b = boot[col]
            diff = b - boot[base]
            row = {
                "label": label, "model": name,
                "fold_pr_auc_2015_2018": fold_ap[name],
                "val_pr_auc": ap, "val_pr_auc_lo": np.percentile(b, 2.5), "val_pr_auc_hi": np.percentile(b, 97.5),
                "diff_vs_baseline_lo": np.percentile(diff, 2.5) if col != base else np.nan,
                "diff_vs_baseline_hi": np.percentile(diff, 97.5) if col != base else np.nan,
                "share_boot_beats_baseline": float((diff > 0).mean()) if col != base else np.nan,
                "val_roc_auc": roc_auc_score(val[label], val[col]),
                "val_brier": brier_score_loss(val[label], val[col]),
                "mean_pred": float(val[col].mean()), "observed_rate": float(val[label].mean()),
                **topk_stats(val, col, label),
            }
            all_rows.append(row)
            p, r, _ = precision_recall_curve(val[label], val[col])
            curves[(label, name)] = (r, p, ap)
        keep = ["record_id", "year", "county", "store_type", "ruca_group", "median_hh_income",
                "store_age_years", "exit_is_batch", "impact_dist_nearest_other_grocery_mi",
                *[c for c in val.columns if c.startswith("bias_")], label, *score_cols]
        preds_out.append(val[keep].assign(label_name=label).rename(columns={label: "y"}))

    res = pd.DataFrame(all_rows)
    res.to_csv(DOCS / "model_results.csv", index=False)
    pd.concat(preds_out).to_parquet(PROCESSED / "val_predictions.parquet", index=False)

    # PR curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, label in zip(axes, LABELS):
        for name in ("Store-type baseline", "Logistic regression", "Gradient-boosted trees"):
            r, p, ap = curves[(label, name)]
            ax.plot(r, p, label=f"{name} (PR-AUC {ap:.3f})")
        rate = res.loc[res["label"] == label, "observed_rate"].iloc[0]
        ax.axhline(rate, ls=":", c="grey", lw=1, label=f"exit rate ({rate:.1%})")
        ax.set_title(f"{label}, validation 2019-2021")
        ax.set_xlabel("Recall"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Precision")
    fig.tight_layout(); fig.savefig(FIGURES / "model_pr_curves.png", dpi=150)

    write_markdown(res, tuning_notes, pair_notes)
    for label, lo, hi, share in pair_notes:
        print(f"HGB - LR {label}: {lo:.3f} to {hi:.3f}, HGB ahead {share:.0%}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res.round(3).to_string())


def write_markdown(res, tuning_notes, pair_notes):
    f = lambda x: f"{x:.3f}"
    L = ["# D3 model results (validation 2019-2021)", "",
         "Generated by `src/model/01_evaluate.py`; do not edit by hand. All models are fit on train cohorts "
         "2012-2018 and scored on validation cohorts 2019-2021, the same rows for every model. Tuning used "
         "rolling folds inside 2012-2018 (fit 2012..k-1, score k, k = 2015-2018). The reserved test file "
         f"is not read. Intervals: {N_BOOT} bootstrap resamples of stores (record_id).", ""]
    for label in LABELS:
        r = res[res["label"] == label]
        L += [f"## {label}", "",
              "| Model | Fold PR-AUC (2015-18) | Validation PR-AUC (95% CI) | Diff vs baseline (95% CI) | ROC-AUC | Brier | Top-50/yr precision | Top-50/yr recall | Meck top-10/yr precision | Meck top-10/yr recall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, x in r.iterrows():
            diff = "" if np.isnan(x["diff_vs_baseline_lo"]) else f"{f(x['diff_vs_baseline_lo'])} to {f(x['diff_vs_baseline_hi'])}"
            L.append(f"| {x['model']} | {f(x['fold_pr_auc_2015_2018'])} | {f(x['val_pr_auc'])} ({f(x['val_pr_auc_lo'])}-{f(x['val_pr_auc_hi'])}) | {diff} | "
                     f"{f(x['val_roc_auc'])} | {x['val_brier']:.4f} | {x['top50_precision']:.1%} ({x['top50_exits']}) | {x['top50_recall']:.1%} | "
                     f"{x['meck_top10_precision']:.1%} ({x['meck_top10_exits']}) | {x['meck_top10_recall']:.1%} |")
        L += ["", "Exits caught in the top 50 that left no other SNAP grocery within 1 mile: " +
              "; ".join(f"{x['model']} {x['top50_gap_exits']} of {x['gap_exits_total']}" for _, x in r.iterrows()), ""]
    L += ["## Gradient-boosted trees minus logistic regression (validation PR-AUC)", ""]
    for label, lo, hi, share in pair_notes:
        L.append(f"- {label}: 95% CI {lo:.3f} to {hi:.3f}; trees ahead in {share:.0%} of resamples")
    L += ["", "## Tuning (best setting per model and label)", ""]
    for label, name, best, tab in tuning_notes:
        L.append(f"- {label}, {name}: {best}, mean fold PR-AUC {tab['mean_ap'].iloc[0]:.3f} "
                 f"(range over grid {tab['mean_ap'].min():.3f}-{tab['mean_ap'].max():.3f})")
    L += ["", "![PR curves](../figures/model_pr_curves.png)", ""]
    (DOCS / "model_results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
