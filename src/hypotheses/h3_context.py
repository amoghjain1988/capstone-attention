"""H3: does context explain the faithfulness index, beyond attention alone.

See CONTRACT.md section 5.10 for the signature. CONTRACT.md pins the model
(`FI_i = b0 + b . context + scene fixed effects`, one joint test of every
coefficient) but not the exact fitting code. The choices below -- the joint
Wald test, the partial R squared as a scene-only-vs-full comparison, and the
variance inflation factor as the collinearity check -- are this module's own
design. Confirm them with the team before a p value from this module appears
in a report.

DENSITY. CONTRACT.md section 3 names two different quantities "density": the
`trajectories` table holds one row per agent per FRAME, and
`src/features/context.py` holds one row per WINDOW. This module regresses on
the WINDOW-level quantity: `context` must carry one row per window_id, not
one row per agent per frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data import schema

# The four raw covariates this module regresses on, before z-scoring.
# density and n_agents come from src/features/context.py, one row per
# window. closing_speed and inv_ttc come from src/features/kinematics.py and
# must already be reduced to one value per window before they reach this
# module -- run() does not reduce them itself, so it does not guess how.
CONTEXT_COLUMNS = ("density", "n_agents", "closing_speed", "inv_ttc")

# THE TRAP. A window with exactly this many edges into the ego gives a
# faithfulness index of exactly +1 or -1: the floor is the mean of 2 shifts
# and the ceiling is their max, so FI has only two reachable values, never a
# middle one. Those windows are about 12 percent of the sample and 78
# percent of the usable windows in the scene eth (see
# docs/CATHERINE_AGENT_2.md TASK 7). Mixed in with the continuous FI of
# windows with more edges, they inject a bimodal spike that does not reflect
# context.
TWO_EDGE_TRAP = 2

# Which of the two fits (see run()) this module treats as primary. Dropping
# the 2-edge windows removes a spike that is a property of E == 2, not of
# context, so it is the headline. Report both; state which one is which.
HEADLINE_DROPS_TWO_EDGE = True


def _zscore(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Return a copy of frame with every named column z-scored, ddof 0."""
    out = frame.copy()
    for column in columns:
        std = float(out[column].std(ddof=0))
        if std == 0.0 or np.isnan(std):
            raise ValueError(f"{column!r} has zero variance; it cannot be z-scored")
        out[column] = (out[column] - out[column].mean()) / std
    return out


def check_collinearity(covariates: pd.DataFrame) -> pd.DataFrame:
    """Return the variance inflation factor of every column of covariates.

    Scoped to the continuous covariates only, not the scene dummies: a
    scene fixed effect is expected to correlate with a covariate that
    differs by scene, and VIF on the dummies would flag that expected
    pattern, not a real redundancy between the covariates themselves.
    """
    matrix = sm.add_constant(covariates, has_constant="add")
    values = matrix.to_numpy(dtype=np.float64)
    rows = [
        {"term": name, "vif": float(variance_inflation_factor(values, position))}
        for position, name in enumerate(matrix.columns)
        if name != "const"
    ]
    return pd.DataFrame(rows)


def _fit_one(table: pd.DataFrame, clusters: np.ndarray) -> dict:
    """Fit fi ~ context + scene fixed effects, with cluster-robust errors.

    Returns n, n_clusters, the coefficient table, the joint Wald test of
    every non-intercept term (context and scene both), the partial R
    squared of the context block over a scene-only model, and the
    collinearity check.
    """
    scored = _zscore(table, CONTEXT_COLUMNS)

    formula = "fi ~ " + " + ".join(CONTEXT_COLUMNS) + " + C(scene)"
    full = smf.ols(formula, data=scored).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(clusters)}
    )

    exog_names = full.model.exog_names
    intercept_index = exog_names.index("Intercept")
    restriction = np.delete(np.eye(len(exog_names)), intercept_index, axis=0)
    joint = full.wald_test(restriction, use_f=True, scalar=True)

    reduced = smf.ols("fi ~ C(scene)", data=scored).fit()
    sse_full = float((full.resid**2).sum())
    sse_reduced = float((reduced.resid**2).sum())
    partial_r2 = (
        (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0.0 else float("nan")
    )

    coefficients = pd.DataFrame(
        {
            "term": full.params.index,
            "b": full.params.to_numpy(dtype=np.float64),
            "se": full.bse.to_numpy(dtype=np.float64),
            "p": full.pvalues.to_numpy(dtype=np.float64),
        }
    )

    return {
        "n": int(full.nobs),
        "n_clusters": int(pd.Series(np.asarray(clusters)).nunique()),
        "coefficients": coefficients,
        "joint_statistic": float(joint.statistic),
        "joint_p": float(joint.pvalue),
        "partial_r2": partial_r2,
        "vif": check_collinearity(scored[list(CONTEXT_COLUMNS)]),
    }


def run(fi: pd.DataFrame, context: pd.DataFrame, cfg) -> dict:
    """Test H0-3: every context coefficient AND every scene term equals 0.

    `fi` matches schema.FAITHFULNESS: window_id, ego_id, n_edges, floor,
    ceiling, fi. `context` carries one row per window_id with the four raw
    covariates named in CONTEXT_COLUMNS, already reduced to one row per
    window -- build it by joining src/features/context.py's window-level
    output with a per-window reduction of src/features/kinematics.py, not
    passed here.

    Rows where `fi` is null (ceiling == floor, an undefined index) are
    dropped before anything is fit. The four covariates are z-scored on the
    regression sample actually used. Scene enters as a fixed effect: 5
    scenes is too few for a random intercept. Standard errors cluster on
    `cfg.CLUSTER_UNIT` (component clusters), computed by
    `src.stats.clustered.component_clusters`, imported here and not
    reimplemented -- see the module docstring of src/stats/loso.py for the
    same convention.

    Fits the model twice: once on every window with a defined fi, once with
    the TWO_EDGE_TRAP windows removed. Returns
    {"with_two_edge": {...}, "without_two_edge": {...}, "headline": str,
    "n_dropped_two_edge": int}. Each inner dict holds n, n_clusters,
    coefficients (DataFrame of term/b/se/p), joint_statistic, joint_p,
    partial_r2, and vif (DataFrame).
    """
    cluster_unit = getattr(cfg, "CLUSTER_UNIT", None)
    if cluster_unit != "component":
        raise ValueError(
            "h3_context.run only implements the frozen cluster unit "
            f"'component' (see config.CLUSTER_UNIT); cfg.CLUSTER_UNIT is "
            f"{cluster_unit!r}."
        )

    missing = set(CONTEXT_COLUMNS) - set(context.columns)
    if missing:
        raise KeyError(
            f"context is missing {sorted(missing)}. run() expects density, "
            "n_agents, closing_speed and inv_ttc already joined onto "
            "window_id, one row per window."
        )
    required_fi = {"window_id", "ego_id", "n_edges", "fi"}
    missing_fi = required_fi - set(fi.columns)
    if missing_fi:
        raise KeyError(f"fi is missing {sorted(missing_fi)}")

    merged = fi.merge(
        context[["window_id", *CONTEXT_COLUMNS]],
        on="window_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["fi"].notna()].copy()
    merged["scene"] = (
        merged["window_id"].astype(str).map(lambda wid: schema.split_window_id(wid)[0])
    )

    from src.stats.clustered import component_clusters  # Prem's file. Do not reimplement.

    def fit(table: pd.DataFrame) -> dict:
        clusters = component_clusters(table)
        return _fit_one(table, clusters)

    with_two_edge = fit(merged)

    two_edge_mask = merged["n_edges"] == TWO_EDGE_TRAP
    without_two_edge = fit(merged.loc[~two_edge_mask].copy())

    return {
        "with_two_edge": with_two_edge,
        "without_two_edge": without_two_edge,
        "headline": "without_two_edge" if HEADLINE_DROPS_TWO_EDGE else "with_two_edge",
        "n_dropped_two_edge": int(two_edge_mask.sum()),
    }
