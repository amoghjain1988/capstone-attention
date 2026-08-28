"""Closing speed and inverse time to collision.

See CONTRACT.md section 5.3 for the signatures.

Every number here comes from the OBSERVED frames only. The future is never
read. Reading fut here would leak the answer into the predictor of the
answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema


def pair_kinematics(window_row: pd.Series, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Return one row per edge that points into the ego.

    Columns: window_id, src, dst, closing_speed, inv_ttc.

    closing_speed is positive when the pair approaches and is computed from
    the relative position r = p_src - p_dst and the relative velocity
    v = v_src - v_dst at the last observed frame. Velocity is the first
    difference of the last two observed frames, divided by DT. inv_ttc is 0
    when the pair never closes, and never negative and never NaN. A pair that
    sits on top of each other, ||r|| == 0, gets 0.0 for both, rather than a
    division by zero.
    """
    order = [int(a) for a in window_row["agent_order"]]
    ego_index = schema.ego_index(window_row)
    hist = schema.unpack_hist(window_row, config.N_HIST)

    position = hist[:, -1, :]
    velocity = (hist[:, -1, :] - hist[:, -2, :]) / config.DT

    src_index = np.array([i for i in range(len(order)) if i != ego_index], dtype=np.int64)
    src_ids = np.array([order[i] for i in src_index], dtype=np.int32)

    r = position[src_index] - position[ego_index]
    v = velocity[src_index] - velocity[ego_index]
    r_norm = np.sqrt((r * r).sum(axis=1))

    closing_speed = np.zeros(len(src_index), dtype=np.float64)
    inv_ttc = np.zeros(len(src_index), dtype=np.float64)
    safe = r_norm > 0.0
    closing_speed[safe] = -(r[safe] * v[safe]).sum(axis=1) / r_norm[safe]
    inv_ttc[safe] = np.clip(closing_speed[safe], 0.0, None) / r_norm[safe]

    frame = pd.DataFrame(
        {
            "window_id": str(window_row["window_id"]),
            "src": src_ids,
            "dst": np.int32(int(window_row["ego_id"])),
            "closing_speed": closing_speed,
            "inv_ttc": inv_ttc,
        }
    )
    return frame[["window_id", "src", "dst", "closing_speed", "inv_ttc"]]


def pair_kinematics_all(windows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Run pair_kinematics over every eligible window and concatenate the result."""
    columns = ["window_id", "src", "dst", "closing_speed", "inv_ttc"]
    eligible = windows.loc[windows["eligible"]]
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    frames = [pair_kinematics(row, trajectories) for _, row in eligible.iterrows()]
    return pd.concat(frames, ignore_index=True)[columns]
