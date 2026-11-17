"""D4 step 5: Jan-1 access-impact score and a risk x impact ranking (validation, report only).

Everything in the impact score is known on Jan 1 of t. (The existing `impact_dist_nearest_other_grocery_mi` is measured at the exit date,
so it is outcome-side and is used here ONLY to score how many 'gap' exits each ranking catches, never as an input.)

Impact inputs, per store-year (cohort year t):
  dist_other_grocery_jan1_mi  miles (EPSG:32119) to the nearest OTHER grocery-type authorization active on Jan 1 of t
                              (auth <= Jan 1 <= end, placeholder/missing start = active; same rule as 05c; the store's own Record ID excluded)
  pct_snap_hh, pct_no_vehicle tract share of households receiving SNAP / with no vehicle (ACS vintage released before Jan 1 t)
Definition, fixed before looking at any result (no tuning):
  each input -> percentile rank in the pooled 2012-2021 store-year reference (inputs only, no outcomes; distance ties share a rank)
  impact = 0.50 * rank(distance) + 0.25 * rank(pct_snap_hh) + 0.25 * rank(pct_no_vehicle)      in (0, 1]
  Weights: distance is the direct access measure; the two tract-need measures together get equal weight. Missing tract values -> rank 0.5.
  A store with no other grocery at all (distance +inf) gets distance rank 1; missing coordinates get 0.5.
  priority = predicted exit probability x impact.
Reference tables (the sorted pooled values) are saved to docs/impact_reference.json so the deployed app uses the same ranks. The score functions live in impact.py.

Compared on validation 2019-2021: top 50 statewide per cohort year by risk alone vs by priority, and top 10 in Mecklenburg. Reports overlap,
exits caught, gap exits caught (exit-time distance to nearest other grocery > 1 mile, as in 01_evaluate), and who moves in and out.
The test file is never read. Writes docs/impact_score.md, docs/impact_reference.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common_model import DOCS, PROCESSED, TOP_K_MECK, TOP_K_STATE, TRAIN_YEARS, VAL_YEARS, top_k_flags  # noqa: E402

from impact import W, Ranker, impact_score  # noqa: E402
LABEL = "exit_12m"
YEARS = list(TRAIN_YEARS) + list(VAL_YEARS)


def main():
    sy = pd.read_parquet(PROCESSED / "store_year_trainval.parquet")
    assert sy["year"].max() <= 2021
    sy = sy[sy["year"].isin(YEARS)].copy()
    # dist_other_grocery_jan1_mi is a `context` column built by src/prep/05d_context_columns.py
    n_nocoord = int(sy["latitude"].isna().sum())

    # pooled reference (inputs only); stores with no other grocery anywhere count as +inf distance
    dist_all = sy["dist_other_grocery_jan1_mi"].where(sy["latitude"].notna())
    ref = {"dist": dist_all.to_numpy(), "snap": sy["pct_snap_hh"].to_numpy(float), "veh": sy["pct_no_vehicle"].to_numpy(float)}
    rk = {k: Ranker(v) for k, v in ref.items()}
    (DOCS / "impact_reference.json").write_text(json.dumps({"weights": W, "reference_years": [YEARS[0], YEARS[-1]], "n_reference": int(len(sy)),
                                                            **{k: [round(float(x), 6) for x in rk[k].ref] for k in rk}}))
    sy["impact"] = impact_score(sy["dist_other_grocery_jan1_mi"].to_numpy(),
                                sy["pct_snap_hh"].to_numpy(float), sy["pct_no_vehicle"].to_numpy(float), rk)

    v = pd.read_parquet(PROCESSED / "val_predictions.parquet")
    v = v[v["label_name"] == LABEL][["record_id", "year", "p_hgb", "y", "impact_dist_nearest_other_grocery_mi"]]
    v = v.merge(sy[["record_id", "year", "store_name", "county", "store_type", "impact", "dist_other_grocery_jan1_mi", "pct_snap_hh", "pct_no_vehicle"]],
                on=["record_id", "year"], how="left", validate="one_to_one")
    assert v["impact"].notna().all()
    v["priority"] = v["p_hgb"] * v["impact"]
    y = v["y"].astype(bool)
    gap = v["impact_dist_nearest_other_grocery_mi"] > 1
    meck = v["county"] == "MECKLENBURG"

    def flags(col, mask=None, k=TOP_K_STATE):
        if mask is None:
            return top_k_flags(v[col], v["year"], k)
        f = pd.Series(False, index=v.index); f[mask] = top_k_flags(v.loc[mask, col], v.loc[mask, "year"], k); return f

    sets = {"Top 50/yr statewide": (TOP_K_STATE, None), "Top 10/yr Mecklenburg": (TOP_K_MECK, meck)}
    rows, moves = [], []
    for name, (k, mask) in sets.items():
        fr, fp = flags("p_hgb", mask, k), flags("priority", mask, k)
        for lab, f in (("risk alone", fr), ("risk x impact", fp)):
            rows.append({"set": name, "ranking": lab, "flagged": int(f.sum()), "exits": int((f & y).sum()), "precision": (f & y).sum() / f.sum(),
                         "gap_exits": int((f & y & gap).sum()), "mean_jan1_dist": v.loc[f, "dist_other_grocery_jan1_mi"].mean(),
                         "median_jan1_dist": v.loc[f, "dist_other_grocery_jan1_mi"].median(), "mean_snap": v.loc[f, "pct_snap_hh"].mean(),
                         "mean_novehicle": v.loc[f, "pct_no_vehicle"].mean(), "mean_risk": v.loc[f, "p_hgb"].mean()})
        moves.append((name, int((fr & fp).sum()), int((fr & ~fp).sum()), int((~fr & fp).sum()), fr, fp))
    res = pd.DataFrame(rows)

    corr = float(v.loc[y & v["impact_dist_nearest_other_grocery_mi"].notna() & v["dist_other_grocery_jan1_mi"].notna(), ["dist_other_grocery_jan1_mi", "impact_dist_nearest_other_grocery_mi"]].corr(method="spearman").iloc[0, 1])
    ex_gap = y & gap & v["dist_other_grocery_jan1_mi"].notna()
    gap_share = float((v.loc[ex_gap, "dist_other_grocery_jan1_mi"] > 1).mean())

    L = ["# Access-impact score and risk x impact ranking (validation 2019-2021, report only)", "",
         "Generated by `src/model/08_impact_score.py`. Inputs are all known on Jan 1 of t; the definition and weights were fixed before any result was looked at. The test file is not read.", "",
         "**impact = 0.50 x rank(distance to nearest other SNAP grocery on Jan 1) + 0.25 x rank(tract % SNAP households) + 0.25 x rank(tract % households with no vehicle)**, "
         "each rank a percentile against the pooled 2012-2021 store-year reference (inputs only), so impact lies in (0, 1]. **priority = predicted exit probability x impact.** "
         "Rank tables are in `docs/impact_reference.json` for the app.", "",
         f"Jan-1 distance (`dist_other_grocery_jan1_mi`, built in `05d_context_columns.py`): available for {len(sy):,} store-years 2012-2021 ({n_nocoord} without usable coordinates get distance rank 0.5). "
         f"Validity check against the exit-time gap measure (outcome-side, used only here): Spearman correlation between Jan-1 distance and distance at the exit date among validation exits = {corr:.2f}; "
         f"{gap_share:.0%} of validation 'gap exits' (> 1 mile at exit) already had no other grocery within a mile on Jan 1.", "",
         f"Gap exits in validation: {int((y & gap).sum())} of {int(y.sum())} exits ({int((y & gap).sum()) / int(y.sum()):.0%}).", "",
         "| Set | Ranking | Flagged | Exits caught | Precision | Gap exits caught | Mean Jan-1 dist (mi) | Median dist | Mean % SNAP hh | Mean % no vehicle | Mean predicted risk |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['set']} | {r['ranking']} | {r['flagged']} | {r['exits']} | {r['precision']:.1%} | {r['gap_exits']} | {r['mean_jan1_dist']:.2f} | {r['median_jan1_dist']:.2f} | "
                 f"{r['mean_snap']:.1%} | {r['mean_novehicle']:.1%} | {r['mean_risk']:.1%} |")
    L += ["", "## How the top 50 changes", ""]
    for name, both, out, inn, fr, fp in moves:
        L.append(f"- {name}: {both} of {int(fr.sum())} flagged store-years are in both lists; {out} drop out and {inn} come in.")
    fr, fp = moves[0][4], moves[0][5]
    for title, sel, key in (("Dropped from the top 50 by adding impact (highest risk first)", fr & ~fp, "p_hgb"),
                            ("Added to the top 50 by adding impact (highest priority first)", fp & ~fr, "priority")):
        sub = v[sel].sort_values(key, ascending=False)
        L += ["", f"### {title}: first 8 of {len(sub)}", "", "| Cohort | Store | County | Type | Risk | Impact | Jan-1 dist (mi) | Exited |", "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.head(8).iterrows():
            L.append(f"| {int(r['year'])} | {r['store_name']} | {str(r['county']).title()} | {r['store_type']} | {r['p_hgb']:.1%} | {r['impact']:.2f} | {r['dist_other_grocery_jan1_mi']:.2f} | {'yes' if r['y'] else 'no'} |")
    L += ["", "Caveat: with 150 flagged store-years over three years, a difference of a few exits caught is within noise; this table is descriptive, not a test.", ""]
    (DOCS / "impact_score.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
