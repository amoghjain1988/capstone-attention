"""Cut the trajectories into 20-frame windows.

See CONTRACT.md section 5.2 for the signatures.

A window covers 20 consecutive frames. The first 8 frames are the history and
the last 12 frames are the future. An agent joins a window only when the agent
is present in all 20 frames. The agent axis order is the sorted agent id, so
the order is the same on every run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema


def _tracks_of_scene(part: pd.DataFrame) -> dict[int, tuple[int, np.ndarray]]:
    """Return agent_id -> (first frame, positions).

    The positions array has shape (n_frames, 2). The function assumes that the
    track is contiguous. Call clean.gappy_tracks() first.
    """
    tracks: dict[int, tuple[int, np.ndarray]] = {}
    ordered = part.sort_values(["agent_id", "frame"], kind="mergesort")
    positions = ordered[["x", "y"]].to_numpy(dtype=np.float64)
    agents = ordered["agent_id"].to_numpy(dtype=np.int64)
    frames = ordered["frame"].to_numpy(dtype=np.int64)

    edges = np.flatnonzero(np.diff(agents)) + 1
    for block in np.split(np.arange(len(agents)), edges):
        if block.size == 0:
            continue
        tracks[int(agents[block[0]])] = (int(frames[block[0]]), positions[block])
    return tracks


def make_windows(
    df: pd.DataFrame,
    n_hist: int = config.N_HIST,
    n_fut: int = config.N_FUT,
    stride: int = config.WINDOW_STRIDE,
) -> pd.DataFrame:
    """Build one row per window.

    Columns: window_id, scene, t0, agent_order, n_agents, hist, fut.
    The columns ego_id, eligible and overlap_fraction come later.
    A window with no agent is not built at all.
    """
    length = int(n_hist) + int(n_fut)
    rows: list[dict] = []

    for scene, part in df.groupby("scene", observed=True):
        tracks = _tracks_of_scene(part)
        if not tracks:
            continue

        agent_ids = np.array(sorted(tracks), dtype=np.int64)
        starts = np.array([tracks[int(a)][0] for a in agent_ids], dtype=np.int64)
        ends = np.array(
            [tracks[int(a)][0] + len(tracks[int(a)][1]) - 1 for a in agent_ids],
            dtype=np.int64,
        )

        first = int(starts.min())
        last = int(ends.max()) - length + 1
        if last < first:
            continue

        for t0 in range(first, last + 1, int(stride)):
            keep = (starts <= t0) & (ends >= t0 + length - 1)
            if not keep.any():
                continue

            members = agent_ids[keep]
            n_agents = int(members.size)
            block = np.empty((n_agents, length, 2), dtype=np.float64)
            for slot, agent in enumerate(members):
                start, path = tracks[int(agent)]
                offset = t0 - start
                block[slot] = path[offset : offset + length]

            rows.append(
                {
                    "window_id": schema.window_id(str(scene), t0),
                    "scene": str(scene),
                    "t0": int(t0),
                    "agent_order": [int(a) for a in members],
                    "n_agents": n_agents,
                    "hist": block[:, :n_hist, :].reshape(-1).tolist(),
                    "fut": block[:, n_hist:, :].reshape(-1).tolist(),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["window_id", "scene", "t0", "agent_order", "n_agents", "hist", "fut"]
        )
    return pd.DataFrame(rows)


def overlap_fraction(
    windows: pd.DataFrame, length: int = config.WINDOW_LEN
) -> pd.Series:
    """Return the share of the 20 frames also covered by another window.

    The value is 0.0 when the window stands alone. The value is 1.0 when every
    one of its frames also belongs to at least one other window of the same
    scene. A high value is normal at stride 1, and it is the reason that
    src/eda/overlap.py must report a design effect.
    """
    result = pd.Series(0.0, index=windows.index, dtype="float64")

    for _, part in windows.groupby("scene", observed=True):
        starts = part["t0"].to_numpy(dtype=np.int64)
        base = int(starts.min())
        span = int(starts.max()) + length - base
        cover = np.zeros(span + 1, dtype=np.int32)

        # Count how many windows cover every frame. Use a difference array.
        np.add.at(cover, starts - base, 1)
        np.add.at(cover, starts - base + length, -1)
        cover = np.cumsum(cover)[:span]

        shared = np.empty(len(part), dtype=np.float64)
        for position, start in enumerate(starts - base):
            block = cover[start : start + length]
            shared[position] = float((block >= 2).sum()) / float(length)
        result.loc[part.index] = shared

    return result
