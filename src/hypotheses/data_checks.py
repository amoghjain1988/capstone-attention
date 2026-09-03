"""Check that every window's ablation curve is complete before any test runs.

See CONTRACT.md section 5.10 for the signature. CONTRACT.md pins the
signature and the general behaviour ("every window carries every arm, no
missing pair, raise if a primary window is incomplete") but not the exact
arm set or the exact shape of the attrition table. Both are this module's
own design; confirm them with the team before the attrition table is final.
"""

from __future__ import annotations

import pandas as pd

import config
from src.data import schema

# The arms a window needs before h1_morf_vs_lerf.py and h2_beyond_proximity.py
# can compare it. "single" is the per-edge brute-force sweep of
# src/faithfulness/index.py: it holds one row per EDGE, not one row per arm,
# and no primary test subtracts a shift from it directly, so it sits outside
# this check.
REQUIRED_ARMS = ("morf", "lerf", "weight_matched", "nearest", "random")


def check(df: pd.DataFrame) -> pd.DataFrame:
    """Check df, a perturbation_curves-shaped table, for completeness.

    A window is complete when it carries every arm of REQUIRED_ARMS at
    config.PRIMARY_N_REMOVED and config.PRIMARY_MASK_POLICY, each with a
    non-null shift. A zero difference between two arms is real data, not a
    gap: this function never drops a row that holds a zero shift. The Pratt
    rule (src/stats/effects.rank_biserial) handles a zero difference
    downstream; this function does not.

    Returns the attrition table on success: one row per scene, with the
    window count and the complete count. Every eligible window is computed
    by the team's own ablation stage, so a missing arm or a null shift is a
    pipeline bug, not a natural data gap. Raise ValueError instead, with the
    first incomplete window in the message. Do not return a table that
    hides the bug.
    """
    if "window_id" not in df.columns:
        raise KeyError("df must carry a window_id column")
    if "arm" not in df.columns:
        raise KeyError("df must carry an arm column")

    primary = df.loc[
        (df["n_removed"] == config.PRIMARY_N_REMOVED)
        & (df["mask_policy"] == config.PRIMARY_MASK_POLICY)
    ]

    per_window_rows: list[dict] = []
    incomplete: list[str] = []

    for window_id, block in primary.groupby("window_id", observed=True):
        scene, _ = schema.split_window_id(str(window_id))
        present = set(block["arm"].astype(str))
        missing = set(REQUIRED_ARMS) - present
        null_shift = block.loc[block["arm"].isin(REQUIRED_ARMS), "shift"].isna().any()

        complete = not missing and not null_shift
        if not complete:
            reason = f"missing {sorted(missing)}" if missing else "holds a null shift"
            incomplete.append(f"{window_id} {reason}")

        per_window_rows.append(
            {"scene": scene, "window_id": str(window_id), "complete": complete}
        )

    per_window = pd.DataFrame(per_window_rows, columns=["scene", "window_id", "complete"])

    if incomplete:
        raise ValueError(
            f"{len(incomplete)} primary window(s) are incomplete, "
            f"for example: {incomplete[0]}"
        )

    if per_window.empty:
        return pd.DataFrame(columns=["scene", "n_windows", "n_complete", "n_incomplete"])

    summary = (
        per_window.groupby("scene", observed=True)["complete"]
        .agg(n_windows="count", n_complete="sum")
        .reset_index()
    )
    summary["n_complete"] = summary["n_complete"].astype(int)
    summary["n_incomplete"] = summary["n_windows"] - summary["n_complete"]
    return summary
