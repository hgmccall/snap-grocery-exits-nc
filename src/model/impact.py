"""Access-impact score (definition fixed before any result was looked at; see docs/impact_score.md and 08_impact_score.py).

impact = 0.50 * rank(dist_other_grocery_jan1_mi) + 0.25 * rank(pct_snap_hh) + 0.25 * rank(pct_no_vehicle), each rank a percentile against the
pooled 2012-2021 store-year reference stored in docs/impact_reference.json (inputs only, no outcomes). priority = risk x impact.
"""

import numpy as np

W = {"dist": 0.50, "snap": 0.25, "veh": 0.25}


class Ranker:
    """Percentile rank against a fixed sorted reference (midpoint of ties), so a single store can be scored later."""

    def __init__(self, ref):
        self.ref = np.sort(np.asarray(ref, dtype=float)[np.isfinite(ref)])

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        lo, hi = np.searchsorted(self.ref, x, "left"), np.searchsorted(self.ref, x, "right")
        r = (lo + hi) / 2 / len(self.ref)
        return np.where(np.isfinite(x), r, np.nan)


def impact_score(dist, snap, veh, rk):
    """rk = dict of Rankers for 'dist', 'snap', 'veh'. No other grocery (dist NaN from none found) must be passed as np.inf -> rank 1."""
    rd = np.where(np.isposinf(dist), 1.0, rk["dist"](dist))
    rd = np.where(np.isnan(rd), 0.5, rd)
    rs = np.nan_to_num(rk["snap"](snap), nan=0.5)
    rv = np.nan_to_num(rk["veh"](veh), nan=0.5)
    return W["dist"] * rd + W["snap"] * rs + W["veh"] * rv
