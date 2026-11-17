"""Exact per-feature contributions for the frozen trees model, and plain-language top reasons.

The model is HistGradientBoostingClassifier with max_leaf_nodes=2: every boosting iteration is a stump that splits on ONE feature and
returns one of two leaf values (already multiplied by the learning rate). The raw score (log-odds) is therefore

    raw = baseline_prediction + sum over the 300 stumps of that stump's leaf value,

and grouping the stumps by the feature they split on gives an exact additive decomposition: contribution[f] = sum of the leaf values of
the stumps splitting on f. `contributions()` computes it by calling each stump's own predictor (the same call HGB's _raw_predict makes), so
it needs no re-implementation of the split logic (numeric thresholds, missing-value direction, category bitsets). It depends on sklearn
internals (`_predictors`, `_preprocess_X`, `_bin_mapper`), so the spec pins the sklearn version and 07_explanations.py verifies
baseline + sum(contributions) == the model's raw score to ~1e-12 on every row.

Reasons are ranked on contributions centred at the average training store (`reference`), so a reason means "pushes this store's risk up
relative to a typical store", not merely "has a positive leaf value" (leaf values are not centred).
"""

import numpy as np
import pandas as pd


def _hgb(model):
    """The fitted trees model: the last step of the Pipeline (pipeline[-1]); a bare HistGradientBoostingClassifier is accepted too."""
    return model[-1] if hasattr(model, "steps") else model


def _prep(model, X):
    """The transformer's output: the frame the trees see (pipeline[:-1].transform(X)); unchanged for a bare model."""
    if hasattr(model, "steps"):
        for _, step in model.steps[:-1]:   # == model[:-1].transform(X), written out so steps that are not BaseEstimators (earlier scripts) work too
            X = step.transform(X)
    return X


def _matrix(model, X):
    hgb = _hgb(model)
    return hgb._preprocess_X(_prep(model, X)[list(hgb.feature_names_in_)], reset=False)  # ordinal-encodes the category as predict does


def contributions(model, X: pd.DataFrame):
    """Return (baseline, DataFrame[row x feature] of exact log-odds contributions). X must contain the model's feature columns."""
    hgb = _hgb(model)
    A = _matrix(model, X)
    # column order of the matrix HGB sees = feature_names_in_ (verified in 07_explanations.py by the exact-sum check)
    names = list(hgb.feature_names_in_)
    known_cat_bitsets, f_idx_map = hgb._bin_mapper.make_known_categories_bitsets()
    out = np.zeros((len(X), len(names)))
    for its in hgb._predictors:
        (pred,) = its
        nodes = pred.nodes
        assert nodes.shape[0] == 3, "model is not a stump ensemble (max_leaf_nodes must be 2)"
        out[:, int(nodes["feature_idx"][0])] += pred.predict(A, known_cat_bitsets=known_cat_bitsets, f_idx_map=f_idx_map, n_threads=1)
    base = float(np.ravel(hgb._baseline_prediction)[0])
    return base, pd.DataFrame(out, index=X.index, columns=names)


def raw_score(model, X):
    hgb = _hgb(model)  # _raw_predict preprocesses its input itself
    return np.ravel(hgb._raw_predict(_prep(model, X)[list(hgb.feature_names_in_)]))


def _n(v, one, many):
    v = int(round(v))
    return one if v == 1 else many.format(n=v)


# plain-language phrase for a feature value; the reason text appends "(raises risk)"
PHRASES = {
    "store_type": lambda v: f"It is a {v}",
    "store_age_years": lambda v: "Its authorization start date is unknown" if pd.isna(v) else f"It has been authorized for {v:.1f} years",
    "auth_date_placeholder": lambda v: "Its USDA record carries a placeholder authorization date" if v else "Its USDA record has a real authorization date",
    "n_prior_spells_same_id": lambda v: "It has never lapsed and been re-authorized" if v == 0 else _n(v, "It was re-authorized once after a lapse", "It was re-authorized {n} times after lapses"),
    "n_prior_ownership_changes": lambda v: "It has never changed ownership" if v == 0 else _n(v, "It has changed ownership once before", "It has changed ownership {n} times before"),
    "chain_size": lambda v: "No other SNAP store in NC shares its name" if v <= 1 else f"{int(v)} SNAP stores in NC share its name",
    "n_grocery_05mi": lambda v: "No other SNAP grocery within half a mile" if v == 0 else _n(v, "1 other SNAP grocery within half a mile", "{n} other SNAP groceries within half a mile"),
    "n_grocery_1mi": lambda v: "No other SNAP grocery within a mile" if v == 0 else _n(v, "1 other SNAP grocery within a mile", "{n} other SNAP groceries within a mile"),
    "n_convenience_1mi": lambda v: "No SNAP convenience store within a mile" if v == 0 else _n(v, "1 SNAP convenience store within a mile", "{n} SNAP convenience stores within a mile"),
    "convenience_to_grocery_ratio_1mi": lambda v: f"Convenience stores nearby per grocery is {v:.1f} (within a mile)",
    "n_grocery_ends_1mi_prior12m": lambda v: "No nearby SNAP grocery ended last year" if v == 0 else _n(v, "1 nearby SNAP grocery authorization ended last year", "{n} nearby SNAP grocery authorizations ended last year"),
    "acs_population": lambda v: f"Its census tract has {v:,.0f} residents",
    "pop_density_sqmi": lambda v: f"Its census tract has {v:,.0f} people per square mile",
    "pct_poverty": lambda v: f"{v:.0%} of people in its tract are below the poverty line",
    "median_hh_income": lambda v: f"Its tract median household income ranks at the {v:.0%} percentile of stores that year",
    "pct_no_vehicle": lambda v: f"{v:.0%} of households in its tract have no vehicle",
    "pct_snap_hh": lambda v: f"{v:.0%} of households in its tract receive SNAP",
    "pct_renter": lambda v: f"{v:.0%} of households in its tract rent",
    "pct_age_65_plus": lambda v: f"{v:.0%} of residents in its tract are 65 or older",
    "unemployment_rate": lambda v: f"Unemployment in its tract is {v:.1%}",
}


def reference_contributions(model, X_train) -> pd.Series:
    """Mean contribution per feature over the training rows the model was fit on (the 'average store')."""
    return contributions(model, X_train)[1].mean()


def top_reasons(model, X: pd.DataFrame, reference: pd.Series, k: int = 3) -> list[list[dict]]:
    """For each row of X, up to k risk-RAISING reasons [{feature, value, effect, text}], largest first. `effect` is the log-odds contribution
    relative to the average store. `model` is the fitted Pipeline and X the RAW input rows. Fewer than k are returned when fewer features push risk above the average store."""
    _, C = contributions(model, X)
    D = C - reference.reindex(C.columns)
    Xp = _prep(model, X)
    out = []
    for i in range(len(X)):
        reasons = []
        for f, eff in D.iloc[i].sort_values(ascending=False).items():
            if eff <= 0 or len(reasons) == k:
                break
            val = Xp.iloc[i][f]
            raw = X.iloc[i][f] if f in X.columns else val   # the value as supplied: a missing one was imputed or goes down HGB's missing branch
            text = (f"Its {f.replace('_', ' ')} is not available" if pd.isna(raw) and f != "store_age_years" else PHRASES[f](val))
            reasons.append({"feature": f, "value": None if pd.isna(val) else (str(val) if f == "store_type" else float(val)),
                            "effect": float(eff), "text": text + " (raises risk)"})
        out.append(reasons)
    return out


def display_reasons(model, X: pd.DataFrame, reference: pd.Series, tiers, k: int = 3) -> list[list[dict]]:
    """Display rule: reasons are shown only for medium and high tier rows; low-tier rows get an empty list (their effects are tiny
    and read as noise). `tiers` is an array of 'low' / 'medium' / 'high' aligned with X."""
    tiers = np.asarray(tiers)
    out: list[list[dict]] = [[] for _ in range(len(X))]
    show = np.flatnonzero(tiers != "low")
    if len(show):
        for i, r in zip(show, top_reasons(model, X.iloc[show], reference, k)):
            out[i] = r
    return out
