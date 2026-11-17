"""D4 step 6: final refit of the frozen trees model on cohorts 2012-2021 and the bundle the D4 API loads.

Rebuild: `uv run python src/model/09_final_refit.py` (needs data/processed/store_year_trainval.parquet from the prep scripts, through 05d).
Frozen settings, nothing tuned or re-selected: learning_rate 0.1, max_leaf_nodes 2, min_samples_leaf 100, l2 0, 300 iterations, no early stopping,
random_state 2026, exit_12m, the 20 `feature` columns, unemployment_rate median-imputed with the median of the refit rows.
Rows: train 2012-2018 + validation 2019-2021 (the 2022-2024 test file is never read). Because validation is now inside the training data,
THIS MODEL CANNOT BE EVALUATED ON VALIDATION; its only evaluation is the one D6 test run.

Writes (gitignored) models/: trees_exit12m_2012_2021.joblib, bundle.json, impact_reference.json, examples.json
Writes (committed)  docs/model_examples_d4.json (fixed examples with expected outputs), docs/refit_tier_shares.md, and docs/model_spec_d4.md
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common_model import DOCS, PROCESSED, ROOT, SEED, feature_columns, load_rows, rows_for_label  # noqa: E402
from explain import contributions, raw_score, reference_contributions  # noqa: E402
from serving import MODELS, Bundle  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("fv", Path(__file__).parent / "04_fairness_variant.py")
fv = importlib.util.module_from_spec(_s); _s.loader.exec_module(fv)

LABEL = "exit_12m"
FROZEN = dict(learning_rate=0.1, max_leaf_nodes=2, min_samples_leaf=100, l2_regularization=0.0, max_iter=300, early_stopping=False,
              categorical_features="from_dtype", random_state=SEED)
SHARE_TOL_PP = 5.0   # rule written before the refit: each tier's share must move by less than 5 percentage points


def shares(p, lo, hi):
    t = np.where(p < lo, "low", np.where(p < hi, "medium", "high"))
    return {k: float((t == k).mean()) for k in ("low", "medium", "high")}


def main():
    tiers = json.loads((DOCS / "risk_tiers.json").read_text())
    lo, hi, base_rate_2012_18 = tiers["low_cutoff"], tiers["high_cutoff"], tiers["base_rate"]
    assert (lo, hi) == (0.021, 0.084)
    F = feature_columns()
    d = rows_for_label(load_rows(), LABEL)
    tr, va = d[d["part"] == "train"], d[d["part"] == "validation"]
    assert d["year"].min() == 2012 and d["year"].max() == 2021
    print(f"refit rows: {len(d):,} store-years, {int(d[LABEL].sum())} exits (2012-2018: {len(tr):,}/{int(tr[LABEL].sum())}; 2019-2021: {len(va):,}/{int(va[LABEL].sum())})")

    # ---- fit: impute unemployment with the refit median, then the bare HGB (no custom classes in the pickle)
    X = d[F].copy()
    med = float(X["unemployment_rate"].median())
    X["unemployment_rate"] = X["unemployment_rate"].fillna(med)
    hgb = HistGradientBoostingClassifier(**FROZEN).fit(X, d[LABEL])
    assert hgb.n_iter_ == 300
    # equivalence with the pipeline used in 04-08 (same settings): identical predictions
    pipe = fv.make_model(F).fit(d[F], d[LABEL])
    assert np.abs(pipe.predict_proba(d[F])[:, 1] - hgb.predict_proba(X)[:, 1]).max() < 1e-12

    ref = reference_contributions(hgb, X)
    base, C = contributions(hgb, X)
    assert np.abs(base + C.sum(axis=1).to_numpy() - raw_score(hgb, X)).max() < 1e-9
    cats = [str(c) for c in X["store_type"].cat.categories]

    MODELS.mkdir(exist_ok=True)
    joblib.dump(hgb, MODELS / "trees_exit12m_2012_2021.joblib")
    shutil.copy(DOCS / "impact_reference.json", MODELS / "impact_reference.json")
    meta = {"sklearn_version": sklearn.__version__, "python_version": sys.version.split()[0], "label": LABEL, "training_years": [2012, 2021],
            "n_train_rows": int(len(d)), "n_train_exits": int(d[LABEL].sum()), "features": F, "store_type_categories": cats,
            "unemployment_median": med, "settings": {k: v for k, v in FROZEN.items()}, "low_cutoff": lo, "high_cutoff": hi,
            "calibration": "none", "reference_contributions": {k: float(v) for k, v in ref.items()}, "baseline_log_odds": base}
    (MODELS / "bundle.json").write_text(json.dumps(meta, indent=1))

    # ---- tier shares: refit vs the 2012-2018 model (rule: each tier within 5 percentage points)
    old = fv.make_model(F).fit(tr[F], tr[LABEL])
    p_old_tr, p_old_va = old.predict_proba(tr[F])[:, 1], old.predict_proba(va[F])[:, 1]
    p_new_tr, p_new_va = hgb.predict_proba(X.loc[tr.index])[:, 1], hgb.predict_proba(X.loc[va.index])[:, 1]
    p_new_all = hgb.predict_proba(X)[:, 1]
    rows = {"2012-2018 model on validation 2019-2021 (out-of-sample; the tier table in risk_tiers.md)": (p_old_va, len(va)),
            "refit model on validation 2019-2021 (now in-sample)": (p_new_va, len(va)),
            "2012-2018 model on training 2012-2018 (in-sample)": (p_old_tr, len(tr)),
            "refit model on training 2012-2018 (in-sample)": (p_new_tr, len(tr)),
            "refit model on all 2012-2021 (in-sample)": (p_new_all, len(d))}
    sh = {k: shares(p, lo, hi) for k, (p, _) in rows.items()}
    pairs = [("refit model on validation 2019-2021 (now in-sample)", "2012-2018 model on validation 2019-2021 (out-of-sample; the tier table in risk_tiers.md)"),
             ("refit model on training 2012-2018 (in-sample)", "2012-2018 model on training 2012-2018 (in-sample)")]
    worst = max(abs(sh[a][t] - sh[b][t]) * 100 for a, b in pairs for t in ("low", "medium", "high"))
    ok_shares = worst < SHARE_TOL_PP
    from scipy.stats import spearmanr
    rho_va = float(spearmanr(p_old_va, p_new_va).statistic)
    L = ["# Tier shares after the refit", "", "Generated by `src/model/09_final_refit.py`. Cutoffs fixed at 0.021 / 0.084 (`docs/risk_tiers.md`). Rule written before the refit: "
         f"each tier's share of store-years must be within {SHARE_TOL_PP:.0f} percentage points of the 2012-2018 model's on the same rows. Shares only; no outcome rates are shown "
         "(the refit model has seen validation, so its validation outcomes would be in-sample).", "",
         "| Rows / model | Low | Medium | High | Mean predicted |", "|---|---|---|---|---|"]
    for k, (p, _) in rows.items():
        L.append(f"| {k} | {sh[k]['low']:.1%} | {sh[k]['medium']:.1%} | {sh[k]['high']:.1%} | {p.mean():.2%} |")
    L += ["", f"Largest share difference on matched rows: {worst:.1f} percentage points -> **{'similar (rule met)' if ok_shares else 'NOT similar (rule failed)'}**. "
          f"Spearman correlation of the two models' scores on validation rows: {rho_va:.3f}. Base rate: 2012-2018 {base_rate_2012_18:.2%}, 2012-2021 {d[LABEL].mean():.2%}.", ""]
    (DOCS / "refit_tier_shares.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if not ok_shares:
        raise SystemExit("tier shares moved too much after the refit; stop and review before writing the spec")

    # ---- fixed examples (inputs are model inputs only; no outcomes)
    full = pd.read_parquet(PROCESSED / "store_year_trainval.parquet")
    ctx = full.set_index(["record_id", "year"])
    sc = pd.Series(p_new_all, index=d.index)
    d2 = d.assign(p=sc, tier=np.where(sc < lo, "low", np.where(sc < hi, "medium", "high")))
    va2 = d2[d2["part"] == "validation"].sort_values("p", ascending=False)
    pick = [("top-scoring validation store-year", va2.index[0]),
            ("10th highest validation score", va2.index[9]),
            ("high tier, median score", va2[va2["tier"] == "high"].index[len(va2[va2["tier"] == "high"]) // 2]),
            ("medium tier, median score", va2[va2["tier"] == "medium"].index[len(va2[va2["tier"] == "medium"]) // 2]),
            ("low tier, median score", va2[va2["tier"] == "low"].index[len(va2[va2["tier"] == "low"]) // 2]),
            ("lowest validation score", va2.index[-1])]
    miss_unemp = d2[(d2["year"] == 2012) & d2["unemployment_rate"].isna()].sort_values("p", ascending=False)
    miss_age = d2[d2["store_age_years"].isna() & (d2["part"] == "validation")].sort_values("p", ascending=False)
    if len(miss_unemp):
        pick.append(("2012 cohort, unemployment_rate missing (imputed)", miss_unemp.index[len(miss_unemp) // 2]))
    if len(miss_age):
        pick.append(("store_age_years missing (placeholder start date)", miss_age.index[0]))
    pick.append(("no other grocery within 3 miles on Jan 1 (largest impact inputs)", d2[(d2["part"] == "validation")].assign(dd=lambda t: [ctx.loc[(r, y), "dist_other_grocery_jan1_mi"] for r, y in zip(t["record_id"], t["year"])]).query("dd > 3").sort_values("p", ascending=False).index[0]))
    idx = [i for _, i in pick]
    Xe = d2.loc[idx, F]
    dist = [float(ctx.loc[(d2.loc[i, "record_id"], d2.loc[i, "year"]), "dist_other_grocery_jan1_mi"]) for i in idx]

    bundle = Bundle(MODELS)
    out = bundle.score(Xe, dist)
    _, Ce = out.attrs["baseline"], out.attrs["contributions"]
    examples = []
    for n, ((label, i), (_, r)) in enumerate(zip(pick, out.iterrows()), 1):
        row = Xe.loc[i]
        inputs = {c: (str(row[c]) if c == "store_type" else (None if pd.isna(row[c]) else float(row[c]))) for c in F}
        examples.append({"id": f"ex{n}", "description": label, "record_id": int(d2.loc[i, "record_id"]), "cohort_year": int(d2.loc[i, "year"]),
                         "inputs": inputs, "context": {"dist_other_grocery_jan1_mi": dist[n - 1]},
                         "expected": {"raw_score": float(r["raw_score"]), "probability": float(r["probability"]), "tier": r["tier"],
                                      "impact": float(r["impact"]), "priority": float(r["priority"]),
                                      "reasons": [{"feature": x["feature"], "text": x["text"], "effect": x["effect"]} for x in r["reasons"]],
                                      "contributions": {c: float(v) for c, v in Ce.loc[i].items()}}})
    payload = {"note": "median_hh_income is the within-cohort-year percentile, not dollars. Tolerance 1e-9. Generated by src/model/09_final_refit.py from the refit bundle.",
               "sklearn_version": sklearn.__version__, "baseline_log_odds": base, "examples": examples}
    (DOCS / "model_examples_d4.json").write_text(json.dumps(payload, indent=1))
    shutil.copy(DOCS / "model_examples_d4.json", MODELS / "examples.json")

    ok, problems = Bundle(MODELS).self_check()
    assert ok, problems
    print("self-check on freshly written bundle: OK;", len(examples), "examples")
    # failure demos used in the spec: a tampered expected value and a wrong pinned version must fail the check
    bad = json.loads(json.dumps(payload)); bad["examples"][0]["expected"]["probability"] += 1e-6
    assert not Bundle(MODELS).self_check(bad["examples"])[0]
    b2 = Bundle(MODELS); b2.meta["sklearn_version"] = "0.0.0"
    assert not b2.self_check()[0]
    print("self-check correctly fails on a perturbed expected value and on a wrong sklearn version")

    import spec_writer  # noqa: E402
    spec_writer.write(meta, examples, sh, worst, d, tr, va)


if __name__ == "__main__":
    main()
