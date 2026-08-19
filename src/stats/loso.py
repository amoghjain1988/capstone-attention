"""Leave-one-scene-out: one estimate per scene, and how much they disagree.

See GRACE_AGENT.md Task 7. Prem calls both functions here; he does not write
them, but he owns the statistical judgment behind what they compute. Agree
these signatures with him -- CONTRACT.md does not pin either one down beyond
"leave_one_scene_out calls fn once per held-out scene" and
"heterogeneity returns {i_squared, range}". The choices below are this
module's own, documented design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_scene_out(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Call `fn` once per scene present in `df`, on that scene's own rows.

    "Leave-one-scene-out" here means a PER-SCENE estimate: `fn` sees only one
    scene's slice at a time, not the other four. README.mmd calls this
    "per-scene estimate", not cross-validation.

    `fn` takes one scene's DataFrame slice and returns a dict of named
    estimate fields (for example an effect size, a p-value, or both). This
    function adds `scene` and `n_clusters` to every row -- `n_clusters` is
    the count of distinct `ego_id` values in that scene's slice, or the row
    count when `ego_id` is not a column of `df`.

    Print `n_clusters` beside every estimate: fact F5 says `eth` holds only
    70 eligible windows and 24 ego pedestrians, so its estimate needs that
    count in view, not trusted on its own.

    Returns one row per scene, in the order scenes first appear in `df`.
    """
    if "scene" not in df.columns:
        raise KeyError("df must carry a 'scene' column")

    rows: list[dict] = []
    for scene in df["scene"].astype(str).drop_duplicates():
        part = df.loc[df["scene"].astype(str) == scene]
        estimate = dict(fn(part))

        if "ego_id" in part.columns:
            n_clusters = int(part["ego_id"].nunique())
        else:
            n_clusters = int(len(part))

        rows.append({"scene": scene, "n_clusters": n_clusters, **estimate})

    return pd.DataFrame(rows)


def heterogeneity(estimates: pd.Series) -> dict:
    """Return {"i_squared": float, "range": float} for a set of per-scene estimates.

    `estimates` holds one point estimate per scene, for example the `scene`
    column's matching estimate column out of leave_one_scene_out(...).
    `range` is max - min, in the same units as `estimates`.

    `i_squared` here is a SIMPLIFIED, EQUALLY-WEIGHTED stand-in for
    Cochran's I-squared. The textbook statistic needs a variance or a
    standard error PER SCENE to weight each one; this function's signature
    carries only the point estimates, with no such weight attached. Treating
    every scene as equally weighted is what fills that gap. Read the result
    as "how much of the spread across scenes looks like more than chance
    under an equal-weight assumption", not as the rigorous meta-analytic
    statistic a paper would cite. Confirm with Prem before this number
    appears in a report -- he owns the hypothesis-testing judgment it feeds.

    Q = k * variance(estimates, ddof=0), the equally-weighted total.
    degrees of freedom = k - 1.
    i_squared = max(0, (Q - degrees_of_freedom) / Q) * 100, or 0.0 when
    there are fewer than two estimates or every scene agrees exactly.
    """
    values = pd.Series(estimates).dropna().to_numpy(dtype=np.float64)
    k = values.size
    if k == 0:
        return {"i_squared": float("nan"), "range": float("nan")}
    if k == 1:
        return {"i_squared": 0.0, "range": 0.0}

    value_range = float(values.max() - values.min())

    q_stat = float(k * values.var(ddof=0))
    degrees_of_freedom = k - 1
    if q_stat <= 0:
        i_squared = 0.0
    else:
        i_squared = max(0.0, (q_stat - degrees_of_freedom) / q_stat) * 100.0

    return {"i_squared": i_squared, "range": value_range}
