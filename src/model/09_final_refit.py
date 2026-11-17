"""D4 step 6: final refit of the frozen trees model on cohorts 2012-2021, saved as ONE fitted sklearn Pipeline.

Rebuild: `uv run python src/model/09_final_refit.py` (needs data/processed/store_year_*.parquet from the prep scripts, through 05d).
Pipeline = store_prep.StoreFeaturePrep (learns per-cohort-year income reference distributions for 2012-2024, the unemployment_rate median and the
store_type categories) -> HistGradientBoostingClassifier with the frozen settings (learning_rate 0.1, max_leaf_nodes 2, min_samples_leaf 100, l2 0,
300 iterations, no early stopping, random_state 2026; nothing tuned or re-selected). Label exit_12m, rows 2012-2018 train + 2019-2021 validation.
THE REFIT MODEL CANNOT BE EVALUATED ON VALIDATION (validation is now training data); its only evaluation is the one D6 test run.

Test-file access: the 2022-2024 file is opened with columns=["year", "median_hh_income"] ONLY, to build the income reference distributions the API
needs for those cohorts (a Jan-1 feature; no label, outcome or QA column is loaded). Logged in docs/test_reservation_log.md.

Writes (gitignored) models/: pipeline.joblib (HW4 layout: pipeline + metadata dict), metadata.json
Writes (committed)  docs/model_examples_d4.json, docs/refit_tier_shares.md, docs/model_spec_d4.md
Checks (each asserted): the Pipeline's predictions equal a bare model's EXACTLY on all refit rows and on the nine examples; a fresh `python -I`
process loads the bundle and passes the startup self-check; the self-check fails on a perturbed expected value and on a wrong version.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from common_model import DOCS, PROCESSED, SEED, TRAINVAL, feature_columns, load_rows, rows_for_label  # noqa: E402
from explain import contributions, raw_score, reference_contributions  # noqa: E402
from serving import BUNDLE_FILE, MODELS, Bundle  # noqa: E402
from store_prep import FEATURES, INPUT_COLUMNS, StoreFeaturePrep  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("fv", Path(__file__).parent / "04_fairness_variant.py")
fv = importlib.util.module_from_spec(_s); _s.loader.exec_module(fv)

LABEL = "exit_12m"
FROZEN = dict(learning_rate=0.1, max_leaf_nodes=2, min_samples_leaf=100, l2_regularization=0.0, max_iter=300, early_stopping=False,
              categorical_features="from_dtype", random_state=SEED)
SHARE_TOL_PP = 5.0   # rule written before the first refit: each tier's share must move by less than 5 percentage points
REF_YEARS = range(2012, 2025)
TEST_FILE = PROCESSED / "store_year_test.parquet"


def shares(p, lo, hi):
    t = np.where(p < lo, "low", np.where(p < hi, "medium", "high"))
    return {k: float((t == k).mean()) for k in ("low", "medium", "high")}


def raw_inputs(frame: pd.DataFrame, index) -> pd.DataFrame:
    """RAW store-year inputs as the API receives them: year + the 20 features, income in dollars."""
    X = frame.loc[index, list(INPUT_COLUMNS)].copy()
    for c in FEATURES:
        X[c] = X[c].astype("category") if c == "store_type" else pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def main():
    tiers = json.loads((DOCS / "risk_tiers.json").read_text())
    lo, hi, base_rate_2012_18 = tiers["low_cutoff"], tiers["high_cutoff"], tiers["base_rate"]
    assert (lo, hi) == (0.021, 0.084)
    F = feature_columns()
    assert tuple(F) == FEATURES, "store_prep.FEATURES differs from the column inventory"
    d = rows_for_label(load_rows(), LABEL)            # prepared exactly as in 01-08 (income already a within-year rank)
    tr, va = d[d["part"] == "train"], d[d["part"] == "validation"]
    assert d["year"].min() == 2012 and d["year"].max() == 2021
    full = pd.read_parquet(TRAINVAL)
    X_raw, y = raw_inputs(full, d.index), d[LABEL]
    assert (X_raw["year"].to_numpy() == d["year"].to_numpy()).all()
    print(f"refit rows: {len(d):,} store-years, {int(y.sum())} exits (2012-2018: {len(tr):,}/{int(tr[LABEL].sum())}; 2019-2021: {len(va):,}/{int(va[LABEL].sum())})")

    # ---- income reference rows for every cohort 2012-2024: Jan-1 feature data only
    ref_tv = full.loc[full["year"].isin(range(2012, 2022)), ["year", "median_hh_income"]]
    ref_te = pd.read_parquet(TEST_FILE, columns=["year", "median_hh_income"])      # nothing else is read from the test file
    assert list(ref_te.columns) == ["year", "median_hh_income"] and set(ref_te["year"]) <= {2022, 2023, 2024}
    reference = pd.concat([ref_tv, ref_te], ignore_index=True)
    assert set(reference["year"]) == set(REF_YEARS)

    # ---- the Pipeline
    pipe = Pipeline([("prep", StoreFeaturePrep()), ("hgb", HistGradientBoostingClassifier(**FROZEN))])
    pipe.fit(X_raw, y, prep__reference=reference)
    assert pipe[-1].n_iter_ == 300
    prep = pipe.named_steps["prep"]

    # ---- exactness vs the bare model (built the way 04-08 and the first refit built it: prepared frame, imputed, bare HGB)
    Xb = d[F].copy()
    med = float(Xb["unemployment_rate"].median())
    Xb["unemployment_rate"] = Xb["unemployment_rate"].fillna(med)
    bare = HistGradientBoostingClassifier(**FROZEN).fit(Xb, y)
    pd.testing.assert_frame_equal(prep.transform(X_raw), Xb, check_exact=True)       # transformer output == the previously prepared inputs
    p_pipe, p_bare = pipe.predict_proba(X_raw)[:, 1], bare.predict_proba(Xb)[:, 1]
    assert np.array_equal(p_pipe, p_bare), f"pipeline != bare model (max abs diff {np.abs(p_pipe - p_bare).max():.3e})"
    assert np.array_equal(raw_score(pipe, X_raw), np.ravel(bare._raw_predict(Xb)))
    print(f"Pipeline == bare model exactly on all {len(d):,} refit rows (predict_proba and raw score); transformer output identical to the prepared frame")

    ref_c = reference_contributions(pipe, X_raw)
    base, C = contributions(pipe, X_raw)
    assert np.abs(base + C.sum(axis=1).to_numpy() - raw_score(pipe, X_raw)).max() < 1e-9

    # ---- tier shares: refit vs the 2012-2018 model (rule: each tier within 5 percentage points)
    old = fv.make_model(F).fit(tr[F], tr[LABEL])
    p_old_tr, p_old_va = old.predict_proba(tr[F])[:, 1], old.predict_proba(va[F])[:, 1]
    p_new = pd.Series(p_pipe, index=d.index)
    p_new_tr, p_new_va = p_new.loc[tr.index].to_numpy(), p_new.loc[va.index].to_numpy()
    rows = {"2012-2018 model on validation 2019-2021 (out-of-sample; the tier table in risk_tiers.md)": p_old_va,
            "refit model on validation 2019-2021 (now in-sample)": p_new_va,
            "2012-2018 model on training 2012-2018 (in-sample)": p_old_tr,
            "refit model on training 2012-2018 (in-sample)": p_new_tr,
            "refit model on all 2012-2021 (in-sample)": p_pipe}
    sh = {k: shares(p, lo, hi) for k, p in rows.items()}
    ks = list(rows)
    pairs = [(ks[1], ks[0]), (ks[3], ks[2])]
    worst = max(abs(sh[a][t] - sh[b][t]) * 100 for a, b in pairs for t in ("low", "medium", "high"))
    ok_shares = worst < SHARE_TOL_PP
    rho_va = float(spearmanr(p_old_va, p_new_va).statistic)
    L = ["# Tier shares after the refit", "", "Generated by `src/model/09_final_refit.py`. Cutoffs fixed at 0.021 / 0.084 (`docs/risk_tiers.md`). Rule written before the refit: "
         f"each tier's share of store-years must be within {SHARE_TOL_PP:.0f} percentage points of the 2012-2018 model's on the same rows. Shares only; no outcome rates are shown "
         "(the refit model has seen validation, so its validation outcomes would be in-sample).", "",
         "| Rows / model | Low | Medium | High | Mean predicted |", "|---|---|---|---|---|"]
    for k, p in rows.items():
        L.append(f"| {k} | {sh[k]['low']:.1%} | {sh[k]['medium']:.1%} | {sh[k]['high']:.1%} | {p.mean():.2%} |")
    L += ["", f"Largest share difference on matched rows: {worst:.1f} percentage points -> **{'similar (rule met)' if ok_shares else 'NOT similar (rule failed)'}**. "
          f"Spearman correlation of the two models' scores on validation rows: {rho_va:.3f}. Base rate: 2012-2018 {base_rate_2012_18:.2%}, 2012-2021 {d[LABEL].mean():.2%}.", ""]
    (DOCS / "refit_tier_shares.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if not ok_shares:
        raise SystemExit("tier shares moved too much after the refit; stop and review before writing the spec")

    # ---- fixed examples (RAW inputs; no outcomes). Expected outputs from the previous bare-model build, if present, must be reproduced exactly.
    prev_path = DOCS / "model_examples_d4.json"
    prev = {e["id"]: e for e in json.loads(prev_path.read_text())["examples"]} if prev_path.exists() else {}
    ctx = full.set_index(["record_id", "year"])
    d2 = d.assign(p=p_new, tier=np.where(p_new < lo, "low", np.where(p_new < hi, "medium", "high")))
    d2["dist"] = [float(ctx.loc[(r, yr), "dist_other_grocery_jan1_mi"]) for r, yr in zip(d2["record_id"], d2["year"])]
    va2 = d2[d2["part"] == "validation"].sort_values("p", ascending=False)
    mid = lambda t: va2[va2["tier"] == t].index[len(va2[va2["tier"] == t]) // 2]
    miss_unemp = d2[(d2["year"] == 2012) & d2["unemployment_rate"].isna()].sort_values("p", ascending=False)
    miss_age = va2[va2["store_age_years"].isna()]
    pick = [("top-scoring validation store-year", va2.index[0]), ("10th highest validation score", va2.index[9]),
            ("high tier, median score", mid("high")), ("medium tier, median score", mid("medium")), ("low tier, median score", mid("low")),
            ("lowest validation score", va2.index[-1])]
    if len(miss_unemp):
        pick.append(("2012 cohort, unemployment_rate missing (imputed)", miss_unemp.index[len(miss_unemp) // 2]))
    if len(miss_age):
        pick.append(("store_age_years missing (placeholder start date)", miss_age.index[0]))
    pick.append(("no other grocery within 3 miles on Jan 1 (largest impact inputs)", va2[va2["dist"] > 3].index[0]))
    idx = [i for _, i in pick]

    X_ex = X_raw.loc[idx].copy()
    X_ex["dist_other_grocery_jan1_mi"] = d2.loc[idx, "dist"].to_numpy()
    meta = {"steps": [n for n, _ in pipe.steps], "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sklearn_version": sklearn.__version__,
            "python_version": sys.version.split()[0], "numpy_version": np.__version__, "pandas_version": pd.__version__,
            "label": LABEL, "training_years": [2012, 2021], "n_train_rows": int(len(d)), "n_train_exits": int(y.sum()),
            "features": list(FEATURES), "input_columns": list(INPUT_COLUMNS), "context_columns": ["dist_other_grocery_jan1_mi"],
            "tier_cutoffs": {"low": lo, "high": hi, "base_rate_2012_2018": base_rate_2012_18},
            "calibration": "none", "settings": {k: v for k, v in FROZEN.items()}, "baseline_log_odds": base,
            "unemployment_median": prep.unemployment_median_, "store_type_categories": prep.store_type_categories_,
            "income_reference_years": sorted(prep.income_reference_), "income_reference_n": {str(k): int(len(v)) for k, v in sorted(prep.income_reference_.items())},
            "transformer": "store_prep.StoreFeaturePrep (src/model/store_prep.py)"}
    impact_ref = json.loads((DOCS / "impact_reference.json").read_text())
    bundle = {"pipeline": pipe, "metadata": meta, "reference_contributions": {k: float(v) for k, v in ref_c.items()}, "impact_reference": impact_ref, "examples": []}
    out = Bundle.__new__(Bundle)   # score the examples with the in-memory pipeline before writing anything
    out.pipeline, out.meta, out.reference = pipe, meta, pd.Series(bundle["reference_contributions"])
    from impact import Ranker
    out.rankers = {k: Ranker(impact_ref[k]) for k in ("dist", "snap", "veh")}
    out.lo, out.hi = lo, hi
    sc = out.score(X_ex)
    Ce = sc.attrs["contributions"]
    examples = []
    for n, ((label, i), (_, r)) in enumerate(zip(pick, sc.iterrows()), 1):
        row = X_raw.loc[i]
        inputs = {c: (str(row[c]) if c == "store_type" else (None if pd.isna(row[c]) else float(row[c]))) for c in FEATURES}
        e = {"id": f"ex{n}", "description": label, "record_id": int(d2.loc[i, "record_id"]), "cohort_year": int(d2.loc[i, "year"]),
             "inputs": inputs, "context": {"dist_other_grocery_jan1_mi": float(d2.loc[i, "dist"])},
             "expected": {"raw_score": float(r["raw_score"]), "probability": float(r["probability"]), "tier": r["tier"],
                          "impact": float(r["impact"]), "priority": float(r["priority"]),
                          "reasons": [{"feature": x["feature"], "text": x["text"], "effect": x["effect"]} for x in r["reasons"]],
                          "contributions": {c: float(v) for c, v in Ce.loc[i].items()}}}
        if e["id"] in prev:   # regression: the Pipeline must reproduce the previous bare-model build's expected outputs exactly
            for key in ("raw_score", "probability", "impact", "priority", "tier"):
                assert prev[e["id"]]["expected"][key] == e["expected"][key], (e["id"], key, prev[e["id"]]["expected"][key], e["expected"][key])
            assert [x["text"] for x in prev[e["id"]]["expected"]["reasons"]] == [x["text"] for x in e["expected"]["reasons"]]
        examples.append(e)
    # and exact match with the bare model on the nine examples
    Xb_ex = Xb.loc[idx]
    assert np.array_equal(pipe.predict_proba(X_ex)[:, 1], bare.predict_proba(Xb_ex)[:, 1])
    assert np.array_equal(sc["probability"].to_numpy(), bare.predict_proba(Xb_ex)[:, 1])
    print(f"Pipeline == bare model exactly on the {len(examples)} examples" + (f"; also identical to the previous build's expected outputs for {len(set(prev) & {e['id'] for e in examples})} of them" if prev else ""))

    payload = {"note": "Inputs are RAW store-year values as the API receives them (median_hh_income in dollars, `year` = cohort year); the Pipeline converts income to a percentile. Tolerance 1e-9. Generated by src/model/09_final_refit.py.",
               "sklearn_version": sklearn.__version__, "baseline_log_odds": base, "examples": examples}
    (DOCS / "model_examples_d4.json").write_text(json.dumps(payload, indent=1))
    bundle["examples"] = examples
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS / BUNDLE_FILE)
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=1))

    # ---- fresh-process load (HW4 check): clean interpreter, only src/model on the path, then the startup self-check
    code = ("import sys; sys.path.insert(0, r'%s'); from serving import Bundle; b = Bundle(r'%s'); ok, p = b.self_check(); print('FRESH_PROCESS_SELF_CHECK', ok, p); "
            "print('metadata', {k: b.meta[k] for k in ('steps', 'created', 'sklearn_version', 'training_years', 'tier_cutoffs')}); raise SystemExit(0 if ok else 1)"
            % (Path(__file__).parent, MODELS))
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(MODELS))
    print(res.stdout.strip() or res.stderr.strip())
    assert res.returncode == 0, res.stderr
    b = Bundle(MODELS)
    assert b.self_check()[0]
    bad = json.loads(json.dumps(examples)); bad[0]["expected"]["probability"] += 1e-6
    assert not b.self_check(bad)[0]
    b.meta["sklearn_version"] = "0.0.0"
    assert not b.self_check()[0]
    print("self-check passes on the fresh bundle and correctly fails on a perturbed expected value and on a wrong sklearn version")

    import spec_writer
    spec_writer.write(meta, examples, sh, worst, d, tr, va, pipe)


if __name__ == "__main__":
    main()
