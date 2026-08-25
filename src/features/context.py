"""Crowd density around the ego.

See CONTRACT.md section 5.3 for the signatures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema


def window_context(window_row: pd.Series, trajectories: pd.DataFrame) -> dict:
    """Return the crowd context of one window.

    Keys: window_id, density, n_agents. density counts the OTHER agents of
    THIS window within config.DENSITY_RADIUS_M of the ego, at the last
    observed frame. An agent outside the window never counts, even when it
    stands close in the full scene.
    """
    order = [int(a) for a in window_row["agent_order"]]
    ego_index = schema.ego_index(window_row)
    position = schema.unpack_hist(window_row, config.N_HIST)[:, -1, :]

    src_index = np.array([i for i in range(len(order)) if i != ego_index], dtype=np.int64)
    delta = position[src_index] - position[ego_index]
    distance = np.sqrt((delta * delta).sum(axis=1))
    density = float((distance <= config.DENSITY_RADIUS_M).sum())

    return {
        "window_id": str(window_row["window_id"]),
        "density": density,
        "n_agents": int(window_row["n_agents"]),
    }


def window_context_all(windows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Run window_context over every eligible window and concatenate the result."""
    columns = ["window_id", "density", "n_agents"]
    eligible = windows.loc[windows["eligible"]]
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    rows = [window_context(row, trajectories) for _, row in eligible.iterrows()]
    return pd.DataFrame(rows)[columns]
