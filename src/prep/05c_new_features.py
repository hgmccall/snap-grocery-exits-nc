"""05c: four D4 candidate features (pre-registered in docs/changes_since_d3.md), added to the store-year files.

Reads : data/processed/store_year_trainval.parquet, store_year_test.parquet     (from 05; ACS/store features)
        data/interim/periods_spells.parquet, spells.parquet                      (from 03)
        data/raw/bls_laus/laucntyYY.xlsx                                          (manual download, data/sources.yaml)
        data/raw/FoodAccessResearchAtlasData{2010,2015,2019}.xlsx                 (release dates: src/prep/reference/release_dates.csv)
Writes: data/processed/store_year_trainval.parquet, store_year_test.parquet       (the four columns added in place; rerunning is safe)
        docs/column_inventory.csv (the four columns marked `feature`), cleaning log rows (05c_new_features)

The two panels go through the SAME functions separately. No label or outcome column is read; nothing here prints exit
counts or rates. NOT loaded to Supabase (06 selects columns from supabase/schema.sql, which does not list these).

Features (all use only information available before Jan 1 of the cohort year t; hard rule 2):
  n_bigstore_openings_2mi_prior12m  Supermarket / Super Store spells that STARTED in [Jan 1 t-1, Jan 1 t) within 2 miles, excluding the
                                    store itself and ownership changes (a spell that is the successor of a spell that ended before Jan 1 t;
                                    if the predecessor ends on/after Jan 1 t the transfer is not yet knowable, so the opening counts).
  dist_nearest_bigstore_mi          miles to the nearest Supermarket / Super Store period active on Jan 1 of t, excluding the store itself.
  county_unemployment_rate          BLS LAUS county annual average (unemployed / labor force, share 0-1) for year t-2: the annual average
                                    for Y is published about April of Y+1, so Y = t-2 is the latest year certainly out on Jan 1 t.
  fara_lilat_flag                   Food Access Research Atlas LILATracts_1And10 (low income and low access, 1 mile urban / 10 miles rural)
                                    of the latest Atlas released before Jan 1 of t, on 2010 tract boundaries. Missing before the first release
                                    (cohorts 2012-2013); no missing-value indicator is created (it would mark those cohort years).
"""

import importlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import DOCS, INTERIM, PROCESSED, RAW, load_release_dates, log_step, start_log, ensure_dirs

feat05 = importlib.import_module("05_features")  # reuse projection and haversine helpers
project, haversine_mi, M_PER_MI = feat05.project, feat05.haversine_mi, feat05.M_PER_MI

SCRIPT = "05c_new_features"
BIG_TYPES = ("Supermarket", "Super Store")
NEW_FEATURES = ["n_bigstore_openings_2mi_prior12m", "dist_nearest_bigstore_mi", "county_unemployment_rate", "fara_lilat_flag"]
BLS_DIR = RAW / "bls_laus"
LAUS_LAG_YEARS = 2          # cohort t uses LAUS annual-average year t-2
OPENING_RADIUS_MI = 2.0
FARA_FILES = {2010: "FoodAccessResearchAtlasData2010.xlsx", 2015: "FoodAccessResearchAtlasData2015.xlsx", 2019: "FoodAccessResearchAtlasData2019.xlsx"}


# --------------------------------------------------------- big-store features --------------------------------------------------
def bigstore_features(frame: pd.DataFrame, periods: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """Per cohort year: nearest big store active on Jan 1 and count of non-transfer big-store openings in the prior 12 months."""
    per = periods[periods["store_type"].isin(BIG_TYPES) & periods["latitude"].notna()].copy()
    per["px"], per["py"] = project(per["latitude"], per["longitude"])
    sp = spells[spells["store_type_first"].isin(BIG_TYPES) & spells["latitude"].notna() & ~spells["start_placeholder"].astype(bool)].copy()
    sp["px"], sp["py"] = project(sp["latitude"], sp["longitude"])
    succ = spells.loc[spells["successor_spell_id"].notna(), ["successor_spell_id", "spell_end"]]
    fx, fy = project(frame["latitude"], frame["longitude"])
    frame = frame.copy()
    frame["px"], frame["py"] = fx, fy
    dist = pd.Series(np.nan, index=frame.index)
    opens = pd.Series(np.nan, index=frame.index)
    for y, idx in frame.groupby("year").groups.items():
        j1, j0 = pd.Timestamp(int(y), 1, 1), pd.Timestamp(int(y) - 1, 1, 1)
        r = frame.loc[idx]
        ok = r["px"].notna().to_numpy()
        pts = np.column_stack([r["px"].to_numpy()[ok], r["py"].to_numpy()[ok]])
        rec = r["record_id"].to_numpy()[ok]
        # nearest big store active on Jan 1 (auth <= Jan 1 <= end, same rule as the other density features)
        act = per[(per["auth_date"].isna() | (per["auth_date"] <= j1)) & (per["end_date"].isna() | (per["end_date"] >= j1))]
        if len(act):
            tree = cKDTree(np.column_stack([act["px"], act["py"]]))
            k = min(3, len(act))
            d, i = tree.query(pts, k=k)
            d, i = d.reshape(len(pts), k), i.reshape(len(pts), k)
            same = act["record_id"].to_numpy()[i] == rec[:, None]
            d = np.where(same, np.inf, d)          # the store itself is not its own nearest big store
            best = d.min(axis=1) / M_PER_MI
            dist.loc[idx[ok]] = np.where(np.isfinite(best), best, np.nan)
        # openings: spell started in [Jan 1 t-1, Jan 1 t) and is not a transfer known by Jan 1 t
        known_transfer = set(succ.loc[succ["spell_end"] < j1, "successor_spell_id"])
        o = sp[(sp["spell_start"] >= j0) & (sp["spell_start"] < j1) & ~sp["spell_id"].isin(known_transfer)]
        if len(o):
            tree = cKDTree(np.column_stack([o["px"], o["py"]]))
            near = tree.query_ball_point(pts, r=OPENING_RADIUS_MI * M_PER_MI)
            orec = o["record_id"].to_numpy()
            opens.loc[idx[ok]] = [int(sum(orec[j] != rr for j in nn)) for nn, rr in zip(near, rec)]
        else:
            opens.loc[idx[ok]] = 0
    frame["n_bigstore_openings_2mi_prior12m"] = opens.astype("Int64")
    frame["dist_nearest_bigstore_mi"] = dist
    return frame.drop(columns=["px", "py"])


# ---------------------------------------------------------- county unemployment ------------------------------------------------
def _county_key(name: str) -> str:
    return re.sub(r"[^A-Z]", "", re.sub(r",\s*NC$", "", re.sub(r"\s+(County|Municipio)\b", "", str(name))).upper())


def load_laus() -> pd.DataFrame:
    """NC county rows of every laucntyYY.xlsx: columns year, county_key, rate (unemployed / labor force, share 0-1)."""
    out = []
    for f in sorted(BLS_DIR.glob("laucnty*.xlsx")):
        d = pd.read_excel(f, engine="openpyxl", header=1, dtype={"State FIPS Code": str, "County FIPS Code": str})
        d = d[d["State FIPS Code"] == "37"]
        yr = int(d["Year"].iloc[0])
        assert d["Year"].nunique() == 1 and yr == 2000 + int(f.stem[-2:]), f"{f.name} holds year {d['Year'].unique()}"
        assert len(d) == 100, f"{f.name}: {len(d)} NC counties, expected 100"
        out.append(pd.DataFrame({"year": yr, "county_key": d["County Name/State Abbreviation"].map(_county_key),
                                 "rate": d["Unemployed"].astype(float) / d["Labor Force"].astype(float)}))
    return pd.concat(out, ignore_index=True)


def add_county_unemployment(frame: pd.DataFrame, laus: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["laus_year"] = frame["year"] - LAUS_LAG_YEARS
    key = frame["county"].map(_county_key)
    lookup = laus.set_index(["year", "county_key"])["rate"]
    mi = pd.MultiIndex.from_arrays([frame["laus_year"], key])
    frame["county_unemployment_rate"] = lookup.reindex(mi).to_numpy()
    return frame


# ------------------------------------------------------------------ Atlas ----------------------------------------------------
def load_fara() -> tuple[dict[int, pd.Series], pd.Series]:
    rel = load_release_dates()
    rel = rel[rel["dataset"] == "fara"].set_index("end_year")["release_date"]
    flags = {}
    for v, fn in FARA_FILES.items():
        a = pd.read_excel(RAW / fn, engine="openpyxl", sheet_name="Food Access Research Atlas", usecols=["CensusTract", "State", "LILATracts_1And10"], dtype={"CensusTract": str})
        # State is spelled out in some releases and abbreviated in others ('NC'), so filter on the state FIPS prefix instead
        a = a[a["CensusTract"].str.startswith("37")]
        assert 2000 < len(a) < 2400 and a["LILATracts_1And10"].isin([0, 1]).all(), f"{fn}: {len(a)} NC tracts"
        flags[v] = a.set_index("CensusTract")["LILATracts_1And10"].astype(float)
    return flags, rel


def add_fara(frame: pd.DataFrame, flags: dict[int, pd.Series], rel: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    frame["fara_vintage"] = np.nan
    frame["fara_release"] = pd.NaT
    frame["fara_lilat_flag"] = np.nan
    for y in sorted(frame["year"].unique()):
        elig = rel[rel < pd.Timestamp(int(y), 1, 1)]
        if elig.empty:
            continue
        v = int(elig.index[np.argmax(elig.to_numpy())])   # latest release before Jan 1 t
        rows = frame["year"] == y
        # Atlas 2010, 2015 and 2019 are all on 2010 census-tract boundaries (match rates are printed and asserted in main)
        frame.loc[rows, "fara_lilat_flag"] = frame.loc[rows, "tract_geoid_2010"].map(flags[v]).to_numpy()
        frame.loc[rows, "fara_vintage"] = v
        frame.loc[rows, "fara_release"] = rel[v]
    return frame


# -------------------------------------------------------------- assertions -----------------------------------------------------
def timing_assertions(allrows: pd.DataFrame, periods: pd.DataFrame, spells: pd.DataFrame, n: int = 200, seed: int = 11) -> list[str]:
    msgs = []
    j1 = pd.to_datetime(allrows["year"].astype(str) + "-01-01")
    # county unemployment: year t-2, and BLS publishes year Y around April of Y+1 (< Jan 1 t for Y <= t-2)
    assert (allrows["laus_year"] == allrows["year"] - LAUS_LAG_YEARS).all()
    pub = pd.to_datetime((allrows["laus_year"] + 1).astype(str) + "-04-30")
    assert (pub < j1).all(), "a LAUS annual average would not yet be published on Jan 1 of the cohort year"
    assert allrows["county_unemployment_rate"].notna().all(), "county not matched to LAUS"
    msgs.append(f"LAUS: all {len(allrows):,} rows use year t-2, published by Apr 30 of t-1 (latest lead {int((j1 - pub).dt.days.min())} days before Jan 1 t)")
    # Atlas: release strictly before Jan 1 t, latest eligible; missing only before the first release
    has = allrows[allrows["fara_release"].notna()]
    assert (has["fara_release"] < pd.to_datetime(has["year"].astype(str) + "-01-01")).all()
    first = allrows.loc[allrows["fara_release"].notna(), "year"].min()
    assert (allrows.loc[allrows["year"] < first, "fara_vintage"].isna()).all()
    msgs.append(f"Atlas: every release precedes Jan 1 of its cohort year; no Atlas before cohort {first}")
    # big stores: independent recomputation on a sample, from plain pandas masks and haversine distance
    s = allrows.dropna(subset=["latitude", "longitude"]).sample(n, random_state=seed)
    big = periods[periods["store_type"].isin(BIG_TYPES) & periods["latitude"].notna()]
    bigsp = spells[spells["store_type_first"].isin(BIG_TYPES) & spells["latitude"].notna() & ~spells["start_placeholder"].astype(bool)]
    worst_d, bad_n = 0.0, 0
    for r in s.itertuples():
        a, b = pd.Timestamp(r.year, 1, 1), pd.Timestamp(r.year - 1, 1, 1)
        act = big[(big["auth_date"].isna() | (big["auth_date"] <= a)) & (big["end_date"].isna() | (big["end_date"] >= a)) & (big["record_id"] != r.record_id)]
        d = haversine_mi(r.latitude, r.longitude, act["latitude"].to_numpy(), act["longitude"].to_numpy())
        assert (act["auth_date"].dropna() <= a).all()                      # nothing authorized after Jan 1 t is a neighbour
        if len(d):
            # State Plane scale error grows with distance (rural stores can be tens of miles from a big store): relative tolerance
            worst_d = max(worst_d, abs(d.min() - r.dist_nearest_bigstore_mi) - 0.003 * d.min())
        transfers = set(spells.loc[spells["successor_spell_id"].notna() & (spells["spell_end"] < a), "successor_spell_id"])
        o = bigsp[(bigsp["spell_start"] >= b) & (bigsp["spell_start"] < a) & ~bigsp["spell_id"].isin(transfers) & (bigsp["record_id"] != r.record_id)]
        assert (o["spell_start"] < a).all() and (o["spell_start"] >= b).all()
        c = int((haversine_mi(r.latitude, r.longitude, o["latitude"].to_numpy(), o["longitude"].to_numpy()) <= OPENING_RADIUS_MI).sum())
        bad_n += abs(c - r.n_bigstore_openings_2mi_prior12m) > 1
    assert worst_d < 0.01, f"nearest-big-store cross-check differs by more than 0.3% + {worst_d:.3f} mi (projected vs haversine)"
    assert bad_n == 0, f"{bad_n} openings counts differ by more than 1 from the haversine recomputation"
    msgs.append(f"Big stores: {n} random store-years recomputed with haversine and plain pandas masks "
                f"(nearest distances agree within 0.3% of the distance + 0.01 mi; opening counts agree within 1); neighbours all authorized on/before Jan 1 t, openings all start in [Jan 1 t-1, Jan 1 t)")
    return msgs


def build(frame: pd.DataFrame, periods, spells, laus, flags, rel) -> pd.DataFrame:
    drop = [c for c in NEW_FEATURES if c in frame.columns]
    out = frame.drop(columns=drop)
    out = bigstore_features(out, periods, spells)
    out = add_county_unemployment(out, laus)
    out = add_fara(out, flags, rel)
    return out


def main() -> None:
    ensure_dirs()
    start_log(SCRIPT)
    spells = pd.read_parquet(INTERIM / "spells.parquet")
    periods = pd.read_parquet(INTERIM / "periods_spells.parquet")
    laus = load_laus()
    flags, rel = load_fara()
    res, original_cols = {}, {}
    for name in ("trainval", "test"):
        f = pd.read_parquet(PROCESSED / f"store_year_{name}.parquet")
        res[name] = build(f, periods, spells, laus, flags, rel)
        original_cols[name] = [c for c in f.columns if c not in NEW_FEATURES]
        n = len(f)
        nocoord = int(f[["latitude", "longitude"]].isna().any(axis=1).sum())
        log_step(SCRIPT, "bigstore_features", f"{name}: n_bigstore_openings_2mi_prior12m (non-transfer Supermarket/Super Store spells started in the 12 months before Jan 1 t within 2 mi, "
                 "excluding the store) and dist_nearest_bigstore_mi (nearest big store active on Jan 1 t, excluding the store)",
                 "Competition from new large stores is a plausible driver of small-store exits; uses only Jan-1 information (hard rule 2)", n, n, 0)
        log_step(SCRIPT, "bigstore_missing_coords", f"{name}: store-years without coordinates get missing big-store features",
                 "No location, no distance; counted here, the model imputes like any other missing value", n, n, nocoord)
        miss = int(res[name]["county_unemployment_rate"].isna().sum())
        log_step(SCRIPT, "county_unemployment", f"{name}: county_unemployment_rate = BLS LAUS unemployed / labor force for year t-2, joined on county name (all 100 NC counties per file asserted)",
                 "Annual average for Y is published about April of Y+1, so t-2 is the latest year known on Jan 1 t; current revised series only (limitation)", n, n, n - miss)
        a = res[name]["fara_lilat_flag"]
        log_step(SCRIPT, "fara_flag", f"{name}: fara_lilat_flag = LILATracts_1And10 of the latest Atlas released before Jan 1 t (2010 tracts); missing before the first release, no missing indicator",
                 "Atlas release dates from the Read Me sheets (month only, last day used); a missing indicator would mark cohorts 2012-2013", n, n, int(a.notna().sum()))
    allrows = pd.concat([res["trainval"], res["test"]], ignore_index=True)
    # Atlas / tract join coverage (descriptive, features only)
    cov = allrows[allrows["fara_release"].notna()].groupby("fara_vintage")["fara_lilat_flag"].apply(lambda s: s.notna().mean())
    assert (cov > 0.97).all(), f"Atlas tract match rate too low (boundary mismatch?): {cov.to_dict()}"
    msgs = timing_assertions(allrows, periods, spells)
    log_step(SCRIPT, "timing_assertions", "; ".join(msgs), "Hard rule 2: every feature uses only information available before Jan 1 of t", len(allrows), len(allrows), 0)

    for name in ("trainval", "test"):
        cols = original_cols[name]
        at = cols.index("unemployment_rate") + 1
        final = cols[:at] + NEW_FEATURES + cols[at:]
        res[name][final].to_parquet(PROCESSED / f"store_year_{name}.parquet", index=False)
        log_step(SCRIPT, "write_store_year", f"{name}: four columns added to store_year_{name}.parquet ({len(final)} columns)", "Same code for both panels", len(res[name]), len(res[name]), 0)

    # column inventory: the four columns become role = feature (rows inserted after unemployment_rate)
    inv = pd.read_csv(DOCS / "column_inventory.csv")
    inv = inv[~inv["column"].isin(NEW_FEATURES)]
    tv = res["trainval"]
    rows = []
    for c in NEW_FEATURES:
        rows.append({"column": c, "dtype": str(tv[c].dtype), "role": "feature",
                     **{f"missing_pct_{s}": round(100 * allrows.loc[allrows["split"] == s, c].isna().mean(), 1) for s in ("train", "validation", "test")}})
    pos = int(np.where(inv["column"] == "unemployment_rate")[0][0]) + 1
    inv = pd.concat([inv.iloc[:pos], pd.DataFrame(rows), inv.iloc[pos:]], ignore_index=True)
    inv.to_csv(DOCS / "column_inventory.csv", index=False)

    print("Atlas tract match rate by vintage:", cov.round(4).to_dict())
    for m in msgs:
        print("ASSERTION PASSED:", m)
    print("\nNew-feature missingness (%), by split and for year >= 2012:")
    for s in ("train", "validation", "test"):
        sub = allrows[allrows["split"] == s]
        print(f"  {s:<10}", {c: round(100 * sub[c].isna().mean(), 1) for c in NEW_FEATURES},
              "| year>=2012:", {c: round(100 * sub.loc[sub["year"] >= 2012, c].isna().mean(), 1) for c in NEW_FEATURES})
    print("\nFeature summary (train+validation rows, year >= 2012; features only):")
    d = allrows[(allrows["year"] >= 2012) & (allrows["split"] != "test")]
    print(d[NEW_FEATURES].describe().round(3).T.to_string())
    print("openings by cohort year (mean per store, 2012-2021):", d.groupby("year")["n_bigstore_openings_2mi_prior12m"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
