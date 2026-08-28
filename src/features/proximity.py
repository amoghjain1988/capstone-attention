"""The distance ranking that attention has to beat.

See CONTRACT.md section 5.3 for the signatures.

This is the rival explanation for hypothesis 2. If attention only reproduces
this ranking, attention explains nothing that a tape measure does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema


def pair_distance(window_row: pd.Series, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Return one row per edge that points into the ego.

    Columns: window_id, src, dst, dist_at_last_frame, rank_dist.

    dst is always the ego. src is every other agent of the window. The
    distance is read from the last observed frame, that is t0 + N_HIST - 1.
    Positions come from the window payload, not from a join against
    trajectories. Rank 1 is the closest agent. A tie breaks by the lower src
    id, so the ranking never depends on row order.
    """
    order = [int(a) for a in window_row["agent_order"]]
    ego_index = schema.ego_index(window_row)
    positions = schema.unpack_hist(window_row, config.N_HIST)[:, -1, :]

    src_index = np.array([i for i in range(len(order)) if i != ego_index], dtype=np.int64)
    src_ids = np.array([order[i] for i in src_index], dtype=np.int32)

    delta = positions[src_index] - positions[ego_index]
    distance = np.sqrt((delta * delta).sum(axis=1))

    frame = pd.DataFrame(
        {
            "window_id": str(window_row["window_id"]),
            "src": src_ids,
            "dst": np.int32(int(window_row["ego_id"])),
            "dist_at_last_frame": distance,
        }
    )
    frame = frame.sort_values(
        ["dist_at_last_frame", "src"], ascending=[True, True], kind="mergesort"
    )
    frame["rank_dist"] = np.arange(1, len(frame) + 1, dtype=np.int16)
    return frame.reset_index(drop=True)[
        ["window_id", "src", "dst", "dist_at_last_frame", "rank_dist"]
    ]


def pair_distance_all(windows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Run pair_distance over every eligible window and concatenate the result."""
    columns = ["window_id", "src", "dst", "dist_at_last_frame", "rank_dist"]
    eligible = windows.loc[windows["eligible"]]
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    frames = [pair_distance(row, trajectories) for _, row in eligible.iterrows()]
    return pd.concat(frames, ignore_index=True)[columns]
