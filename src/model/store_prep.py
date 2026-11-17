"""StoreFeaturePrep: the input preparation of the D4 trees model as a scikit-learn transformer.

Lives in an importable module (not a script) because the fitted Pipeline pickles this class by module path: the API image must have
`store_prep.py` on its import path (it needs only numpy, pandas and scikit-learn). Rebuild: src/model/09_final_refit.py.

Input  : a DataFrame with one row per store-year and the columns `year` (the cohort year t) + the 20 model inputs in `FEATURES`, with
         `median_hh_income` in DOLLARS as in the store_year table. Extra columns (ids, the `context` distance, ...) are ignored.
Output : a DataFrame of exactly the 20 `FEATURES` in that order (`get_feature_names_out`), `year` consumed and dropped, ready for the trees model.
Learned (fit) state, all from Jan-1 feature data, never outcomes:
  income_reference_[year]      sorted tract median household incomes of every grocery store-year of that cohort year. `median_hh_income` becomes the
                               store's percentile in its own cohort year's distribution (identical to pandas groupby('year').rank(pct=True) on that
                               cohort's rows). Nominal ACS dollars drift across vintages, so a dollar value means something different in 2012 and 2021.
                               `fit(X, y, reference=...)` takes the reference rows (columns year, median_hh_income; features only) so it can cover
                               cohorts the model is not trained on (2022-2024) without touching their labels.
  unemployment_median_         median `unemployment_rate` of the fitted rows. Missing values are replaced by it: a NaN would let a tree isolate the
                               2012 cohort (no ACS table B23025 that year), a calendar-year proxy that project rules forbid.
  store_type_categories_       the category set and order of `store_type` the trees were fit with; any other value is rejected.
Other missing values pass through (HistGradientBoosting handles them natively). `year` is used ONLY to pick the income reference; it is not an output column.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURES = (
    "store_type", "store_age_years", "auth_date_placeholder", "n_prior_spells_same_id", "n_prior_ownership_changes", "chain_size",
    "n_grocery_05mi", "n_grocery_1mi", "n_convenience_1mi", "convenience_to_grocery_ratio_1mi", "n_grocery_ends_1mi_prior12m",
    "acs_population", "pop_density_sqmi", "pct_poverty", "median_hh_income", "pct_no_vehicle", "pct_snap_hh", "pct_renter",
    "pct_age_65_plus", "unemployment_rate",
)
INPUT_COLUMNS = ("year",) + FEATURES
INCOME, UNEMP, CAT = "median_hh_income", "unemployment_rate", "store_type"

# Request-validation rules for the API (Pydantic), from the training-data ranges (2012-2021 store-years). StoreFeaturePrep itself rejects only an unknown
# year, an unknown/missing store_type and missing columns (ValueError); it does NOT range- or type-check numbers (a negative age or a share of 1.2 would
# be scored silently), so the API must enforce these rules and answer 422 -- see docs/model_spec_d4.md section 11. Every key must be present in a request;
# `nullable` fields may be null (missing in the source data), the others may not.
# kind: category | int | float | share (float in [0, 1]) | binary (0 or 1)
FIELD_RULES = {
    "year": dict(kind="int", min=2012, max=2024, nullable=False),
    "store_type": dict(kind="category", nullable=False),
    "store_age_years": dict(kind="float", min=0, nullable=True),
    "auth_date_placeholder": dict(kind="binary", nullable=False),
    "n_prior_spells_same_id": dict(kind="int", min=0, nullable=False),
    "n_prior_ownership_changes": dict(kind="int", min=0, nullable=False),
    "chain_size": dict(kind="int", min=1, nullable=False),
    "n_grocery_05mi": dict(kind="int", min=0, nullable=False),
    "n_grocery_1mi": dict(kind="int", min=0, nullable=False),
    "n_convenience_1mi": dict(kind="int", min=0, nullable=False),
    "convenience_to_grocery_ratio_1mi": dict(kind="float", min=0, nullable=False),
    "n_grocery_ends_1mi_prior12m": dict(kind="int", min=0, nullable=False),
    "acs_population": dict(kind="float", min=0, nullable=True),
    "pop_density_sqmi": dict(kind="float", min=0, nullable=True),
    "pct_poverty": dict(kind="share", nullable=True),
    "median_hh_income": dict(kind="float", min=0, nullable=True),
    "pct_no_vehicle": dict(kind="share", nullable=True),
    "pct_snap_hh": dict(kind="share", nullable=True),
    "pct_renter": dict(kind="share", nullable=True),
    "pct_age_65_plus": dict(kind="share", nullable=True),
    "unemployment_rate": dict(kind="share", nullable=True),
    "dist_other_grocery_jan1_mi": dict(kind="float", min=0, nullable=True),   # `context` column: impact score only
}


class StoreFeaturePrep(BaseEstimator, TransformerMixin):
    """See the module docstring."""

    def __init__(self):
        pass

    # ---------------------------------------------------------------- fit
    def fit(self, X, y=None, reference=None):
        X = self._check(X)
        ref = X[["year", INCOME]] if reference is None else pd.DataFrame(reference)[["year", INCOME]]   # only these two columns are ever read
        ref = ref.assign(year=pd.to_numeric(ref["year"]).astype(int), **{INCOME: pd.to_numeric(ref[INCOME], errors="coerce").astype(float)})
        self.income_reference_ = {int(yr): np.sort(g[INCOME].dropna().to_numpy()) for yr, g in ref.groupby("year")}
        self.unemployment_median_ = float(pd.to_numeric(X[UNEMP], errors="coerce").median())
        st = X[CAT]
        self.store_type_categories_ = [str(c) for c in (st.cat.categories if isinstance(st.dtype, pd.CategoricalDtype) else sorted(st.dropna().astype(str).unique()))]
        self.n_features_in_ = len(INPUT_COLUMNS)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, X):
        check_is_fitted(self, "income_reference_")
        X = self._check(X)
        out = X[list(FEATURES)].copy()
        years = pd.to_numeric(X["year"]).astype(int)
        unknown = sorted(set(years) - set(self.income_reference_))
        if unknown:
            raise ValueError(f"no income reference for cohort year(s) {unknown}; the reference covers {min(self.income_reference_)}-{max(self.income_reference_)}")
        inc = pd.to_numeric(out[INCOME], errors="coerce").astype(float)
        pct = pd.Series(np.nan, index=out.index)
        for yr in years.unique():
            m = (years == yr).to_numpy()
            ref = self.income_reference_[int(yr)]
            lo, hi = np.searchsorted(ref, inc[m].to_numpy(), "left"), np.searchsorted(ref, inc[m].to_numpy(), "right")
            # average-tie rank / n, the same arithmetic as pandas rank(pct=True) (rank = (lo + hi + 1) / 2)
            pct[m] = np.where(np.isnan(inc[m].to_numpy()), np.nan, (lo + hi + 1) / 2 / len(ref))
        out[INCOME] = pct
        bad = sorted(set(out[CAT].dropna().astype(str)) - set(self.store_type_categories_))
        if bad or out[CAT].isna().any():
            raise ValueError(f"{CAT} must be one of {self.store_type_categories_}; got {bad or 'missing'}")
        out[CAT] = pd.Categorical(out[CAT].astype(str), categories=self.store_type_categories_)
        for c in FEATURES:
            if c != CAT:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        out[UNEMP] = out[UNEMP].fillna(self.unemployment_median_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES, dtype=object)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StoreFeaturePrep needs a pandas DataFrame with the columns " + ", ".join(INPUT_COLUMNS))
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        return X
