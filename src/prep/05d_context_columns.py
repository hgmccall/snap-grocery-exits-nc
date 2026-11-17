"""05d: Jan-1 context column for the access-impact score, added to the store-year files.

Reads : data/processed/store_year_trainval.parquet, store_year_test.parquet     (from 05 / 05c)
        data/interim/periods_spells.parquet                                      (from 03)
Writes: data/processed/store_year_trainval.parquet, store_year_test.parquet       (one column added in place; rerunning is safe)
        docs/column_inventory.csv (the column gets the new role `context`), cleaning log rows (05d_context_columns)

dist_other_grocery_jan1_mi  miles (EPSG:32119 State Plane, same projection as the other distance features) from the store to the nearest OTHER
                            grocery-type authorization active on Jan 1 of the cohort year t (auth_date <= Jan 1 <= end_date; a missing/placeholder
                            start counts as active, the same rule as 05c), excluding the store's own Record ID. Missing without coordinates.

Role `context`: Jan-1 information used to score the ACCESS IMPACT of a predicted exit (docs/impact_score.md), never a model feature. It is not in
`feature_columns()`, so no model sees it. (Its exit-time counterpart, `impact_dist_nearest_other_grocery_mi`, is outcome-side and stays
`impact_`; the two correlate at 0.99 among exits, which is why the Jan-1 version can stand in for it.)

Both panels go through the same function. No label or outcome column is read. A brute-force recomputation on 200 random store-years is asserted.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import DOCS, INTERIM, PROCESSED, ensure_dirs, log_step, start_log  # noqa: E402

feat05 = importlib.import_module("05_features")
project, M_PER_MI = feat05.project, feat05.M_PER_MI

SCRIPT = "05d_context_columns"
COL = "dist_other_grocery_jan1_mi"
K_NEIGHBOURS = 6   # nearest candidates fetched; the store's own record(s) are masked, so a few extra are needed


def _active_groceries(periods: pd.DataFrame, jan1: pd.Timestamp) -> pd.DataFrame:
    per = periods[periods["is_grocery"] & periods["latitude"].notna()]
    return per[(per["auth_date"].isna() | (per["auth_date"] <= jan1)) & (per["end_date"].isna() | (per["end_date"] >= jan1))]


def jan1_distance(frame: pd.DataFrame, periods: pd.DataFrame) -> pd.Series:
    per = periods[periods["is_grocery"] & periods["latitude"].notna()].copy()
    per["px"], per["py"] = project(per["latitude"], per["longitude"])
    fx, fy = (pd.Series(np.asarray(a, dtype=float), index=frame.index) for a in project(frame["latitude"], frame["longitude"]))
    out = pd.Series(np.nan, index=frame.index)
    for y, idx in frame.groupby("year").groups.items():
        j1 = pd.Timestamp(int(y), 1, 1)
        act = per[(per["auth_date"].isna() | (per["auth_date"] <= j1)) & (per["end_date"].isna() | (per["end_date"] >= j1))]
        ok = np.isfinite(fx.loc[idx].to_numpy())
        pts = np.column_stack([fx.loc[idx].to_numpy()[ok], fy.loc[idx].to_numpy()[ok]])
        rec = frame.loc[idx, "record_id"].to_numpy()[ok]
        tree = cKDTree(np.column_stack([act["px"], act["py"]]))
        k = min(K_NEIGHBOURS, len(act))
        d, i = tree.query(pts, k=k)
        d, i = d.reshape(len(pts), k), i.reshape(len(pts), k)
        d = np.where(act["record_id"].to_numpy()[i] == rec[:, None], np.inf, d)   # the store is not its own nearest grocery
        best = d.min(axis=1) / M_PER_MI
        assert np.isfinite(best).all(), "all K nearest candidates were the store itself: raise K_NEIGHBOURS"
        out.loc[idx[ok]] = best
    return out


def brute_force_check(frame: pd.DataFrame, periods: pd.DataFrame, n: int = 200, seed: int = 5) -> str:
    s = frame[frame["latitude"].notna()].sample(n, random_state=seed)
    worst = 0.0
    for r in s.itertuples():
        act = _active_groceries(periods, pd.Timestamp(int(r.year), 1, 1))
        act = act[act["record_id"] != r.record_id]
        ax, ay = project(act["latitude"], act["longitude"])
        px, py = project(pd.Series([r.latitude]), pd.Series([r.longitude]))
        d = np.hypot(np.asarray(ax) - float(px[0]), np.asarray(ay) - float(py[0])).min() / M_PER_MI
        worst = max(worst, abs(d - getattr(r, COL)))
    assert worst < 1e-6, f"brute-force mismatch: {worst}"
    return f"{COL} matches a brute-force recomputation on {n} random store-years (max abs difference {worst:.1e} miles)"


def main() -> None:
    ensure_dirs()
    start_log(SCRIPT)
    periods = pd.read_parquet(INTERIM / "periods_spells.parquet")
    res, allrows = {}, []
    for name in ("trainval", "test"):
        f = pd.read_parquet(PROCESSED / f"store_year_{name}.parquet")
        cols = [c for c in f.columns if c != COL]
        f = f[cols].copy()
        f[COL] = jan1_distance(f, periods)
        n, nocoord = len(f), int(f[["latitude", "longitude"]].isna().any(axis=1).sum())
        log_step(SCRIPT, "jan1_distance", f"{name}: {COL} = miles to the nearest other grocery-type authorization active on Jan 1 of t (own Record ID excluded), role `context`",
                 "The access-impact score needs how far the nearest alternative is on Jan 1; the exit-time distance is outcome-side. Jan-1 information only (hard rule 2); never a model feature",
                 n, n, n - nocoord)
        log_step(SCRIPT, "jan1_distance_missing_coords", f"{name}: store-years without coordinates get a missing {COL}",
                 "No location, no distance; the impact score gives such rows the neutral rank 0.5", n, n, nocoord)
        at = cols.index("fara_lilat_flag") + 1 if "fara_lilat_flag" in cols else len(cols)
        res[name] = f[cols[:at] + [COL] + cols[at:]]
        allrows.append(res[name])
    msg = brute_force_check(res["trainval"], periods)
    log_step(SCRIPT, "timing_assertions", msg, "Hard rule 2: uses only authorizations active on Jan 1 of t", len(res["trainval"]), len(res["trainval"]), 0)
    for name in ("trainval", "test"):
        res[name].to_parquet(PROCESSED / f"store_year_{name}.parquet", index=False)
        log_step(SCRIPT, "write_store_year", f"{name}: one column added to store_year_{name}.parquet ({res[name].shape[1]} columns)", "Same code for both panels",
                 len(res[name]), len(res[name]), 0)

    # column inventory: role `context` (never `feature`)
    inv = pd.read_csv(DOCS / "column_inventory.csv")
    inv = inv[inv["column"] != COL]
    al = pd.concat(allrows, ignore_index=True)
    row = {"column": COL, "dtype": str(al[COL].dtype), "role": "context",
           **{f"missing_pct_{s}": round(100 * al.loc[al["split"] == s, COL].isna().mean(), 1) for s in ("train", "validation", "test")}}
    pos = int(np.where(inv["column"] == "fara_lilat_flag")[0][0]) + 1
    pd.concat([inv.iloc[:pos], pd.DataFrame([row]), inv.iloc[pos:]], ignore_index=True).to_csv(DOCS / "column_inventory.csv", index=False)
    print(msg)
    print(f"{COL}: missing % by split", {s: row[f"missing_pct_{s}"] for s in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
