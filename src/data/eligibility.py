"""Pick one ego per window and mark the window eligible.

See CONTRACT.md section 5.2 for the signatures.

One ego is one row and one cluster key. The ego must be present for all 20
frames and must have at least one neighbour. Every agent of a window is already
present for all 20 frames, because windows.make_windows() builds the window
that way. The only remaining condition is therefore the neighbour count.

The pick is random but deterministic. The seed of a window comes from the
global seed and from the window key, so the pick does not depend on the row
order and does not change between runs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import config
from src.data import schema

NO_EGO = -1
MIN_AGENTS = 2  # one ego and one neighbour


def _generator(window_key: str, seed: int) -> np.random.Generator:
    """Return a generator that depends only on the seed and the window key."""
    digest = hashlib.sha256(f"{int(seed)}:{window_key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def pick_ego(window_row: pd.Series, seed: int = config.SEED) -> int:
    """Return the ego agent_id of one window.

    Return NO_EGO when the window holds fewer than two agents. Such a window
    has no interaction to explain, so it carries no ego.
    """
    order = [int(a) for a in window_row["agent_order"]]
    if len(order) < MIN_AGENTS:
        return NO_EGO
    generator = _generator(str(window_row["window_id"]), seed)
    return int(order[int(generator.integers(len(order)))])


def mark_eligible(windows: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Add ego_id and eligible. Never delete a row.

    A row with eligible False stays in the table. Every later stage skips it.
    """
    out = windows.copy()
    out["ego_id"] = [
        pick_ego(row, seed) for _, row in out[["window_id", "agent_order"]].iterrows()
    ]
    out["eligible"] = out["ego_id"] != NO_EGO
    out["ego_id"] = out["ego_id"].astype("int32")
    return out


def eligibility_log(windows: pd.DataFrame) -> pd.DataFrame:
    """Return one drop-log row per scene for the windows that are not eligible."""
    rows = []
    bad = windows.loc[~windows["eligible"]]
    if bad.empty:
        return schema.empty_drops()
    for scene, part in bad.groupby("scene", observed=True):
        rows.append(
            schema.drop_row(
                stage="eligibility.mark_eligible",
                scene=str(scene),
                key=f"{len(part)} windows",
                reason="the window holds fewer than two agents, so it has no ego",
                n_rows=len(part),
            )
        )
    return pd.DataFrame(rows)


def ego_edge_count(windows: pd.DataFrame) -> pd.Series:
    """Return E, the number of edges that point into the ego, per window.

    E is n_agents minus one. The brute force sweep of src/faithfulness/index.py
    costs E forward passes per window, so E drives the whole compute budget.
    """
    return (windows["n_agents"].astype("int64") - 1).where(windows["eligible"])
