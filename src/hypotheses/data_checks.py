"""Check that every window's ablation curve is complete before any test runs.

See CONTRACT.md section 5.10 for the signature. CONTRACT.md pins the
signature and the general behaviour ("every window carries every arm, no
missing pair, raise if a primary window is incomplete") but not the exact
arm set or the exact shape of the attrition table. Both are this module's
own design; confirm them with the team before the attrition table is final.

WHY WEIGHT_MATCHED FOLLOWS A SECOND RULE. The four ordered arms remove one
edge per step, so each one holds a row at every step from 1 to E. The
weight_matched arm matches MASS, not count: `src/ablation/curves.py::curve`
writes the SIZE of the mass matched set into `n_removed`. A window with two
edges needs both of its edges to reach the mass that MoRF removes at step 1,
so its only weight_matched row sits at n_removed 2 and never at step 1. A
rule that asked for weight_matched at config.PRIMARY_N_REMOVED therefore
failed every such window on real curves. This module asks for the four
ordered arms at the primary step, and it asks for weight_matched as at least
one row under the primary mask policy. `h1_morf_vs_lerf.paired_shift` picks
the weight_matched partner by removed_mass and never by n_removed, so that is
the rule the tests downstream actually need.
"""

from __future__ import annotations

import pandas as pd

import config
from src.data import schema

# The arms a window needs AT config.PRIMARY_N_REMOVED before
# h1_morf_vs_lerf.py and h2_beyond_proximity.py can compare it. Every one of
# the four removes one edge per step, so each one holds a row at step 1.
# "single" is the per-edge brute-force sweep of src/faithfulness/index.py: it
# holds one row per EDGE, not one row per arm, and no primary test subtracts a
# shift from it directly, so it sits outside this check.
REQUIRED_ARMS = ("morf", "lerf", "nearest", "random")

# The mass matched control. See the module docstring for why this arm carries
# its own rule: it needs one row under the primary mask policy, at any
# n_removed.
MASS_MATCHED_ARM = "weight_matched"

# The columns that check() reads.
REQUIRED_COLUMNS = ("window_id", "arm", "n_removed", "mask_policy", "shift")


def check(df: pd.DataFrame) -> pd.DataFrame:
    """Check df, a perturbation_curves-shaped table, for completeness.

    A window is complete when both rules below hold.

    1. It carries every arm of REQUIRED_ARMS at config.PRIMARY_N_REMOVED and
       config.PRIMARY_MASK_POLICY, each with a non-null shift.
    2. It carries at least one MASS_MATCHED_ARM row under
       config.PRIMARY_MASK_POLICY, at any n_removed, with a non-null shift.

    The second rule differs from the first on purpose. The mass matched arm
    writes the SIZE of its set into n_removed, so a two-edge window holds its
    only such row at n_removed 2. See the module docstring.

    A zero difference between two arms is real data, not a gap: this function
    never drops a row that holds a zero shift. The Pratt rule
    (src/stats/effects.rank_biserial) handles a zero difference downstream;
    this function does not.

    The function reads every window that carries a row under the primary mask
    policy, so a window that lost its whole primary step still appears and
    still fails.

    Returns the attrition table on success: one row per scene, with the
    window count, the complete count and the incomplete count. Every eligible
    window is computed by the team's own ablation stage, so a missing arm or a
    null shift is a pipeline bug, not a natural data gap. Raise ValueError
    instead, with the first incomplete window in the message. Do not return a
    table that hides the bug.
    """
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise KeyError(f"df must carry a {column!r} column")

    policy = df.loc[df["mask_policy"] == config.PRIMARY_MASK_POLICY]
    primary = policy.loc[policy["n_removed"] == config.PRIMARY_N_REMOVED]
    step_of = {
        str(key): part for key, part in primary.groupby("window_id", observed=True)
    }
    empty_step = primary.iloc[0:0]

    per_window_rows: list[dict] = []
    incomplete: list[str] = []

    for window_id, block in policy.groupby("window_id", observed=True):
        scene, _ = schema.split_window_id(str(window_id))
        step = step_of.get(str(window_id), empty_step)

        present = set(step["arm"].astype(str))
        missing = set(REQUIRED_ARMS) - present
        ordered = step.loc[step["arm"].isin(REQUIRED_ARMS), "shift"]
        matched = block.loc[block["arm"].astype(str) == MASS_MATCHED_ARM, "shift"]
        null_shift = bool(ordered.isna().any() or matched.isna().any())

        reason = ""
        if missing:
            reason = f"missing {sorted(missing)} at n_removed {config.PRIMARY_N_REMOVED}"
        elif len(matched) == 0:
            reason = f"holds no {MASS_MATCHED_ARM} row under the primary mask policy"
        elif null_shift:
            reason = "holds a null shift"

        complete = not reason
        if not complete:
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
