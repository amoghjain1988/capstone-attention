"""Drop the short tracks and add the per-frame kinematics.

See CONTRACT.md section 5.2 for the signatures.

Every drop leaves a row in the drop log. A missing value is NaN, never 0 and
never -1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema

KEY = ["scene", "agent_id"]


# ---------------------------------------------------------------------------
# Contiguity
# ---------------------------------------------------------------------------


def gappy_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row for every track that skips a frame.

    A track is contiguous when its frame count equals its frame span. The
    window builder assumes contiguity, so run this check before the builder.
    Columns: scene, agent_id, n_frames, span, n_gaps.
    """
    grouped = df.groupby(KEY, observed=True)["frame"]
    report = pd.DataFrame(
        {
            "n_frames": grouped.size(),
            "span": grouped.max() - grouped.min() + 1,
        }
    ).reset_index()
    report["n_gaps"] = report["span"] - report["n_frames"]
    return report.loc[report["n_gaps"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Short tracks
# ---------------------------------------------------------------------------


def drop_short_tracks(
    df: pd.DataFrame, min_frames: int = config.MIN_TRACK_FRAMES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the table into the tracks that are long enough and the rest.

    Returns (kept, dropped). The frame `dropped` carries a `reason` column.

    A track shorter than min_frames can never be an ego, because an ego must be
    present for all 20 frames of a window. Such a track is still a real
    pedestrian, and the model still sees it. Read config.DROP_SHORT_TRACKS
    before you call this function on the main path.
    """
    length = df.groupby(KEY, observed=True)["frame"].transform("size")
    is_short = length < int(min_frames)

    kept = df.loc[~is_short].reset_index(drop=True)
    dropped = df.loc[is_short].copy()
    dropped["reason"] = f"track shorter than {int(min_frames)} frames"
    return kept, dropped.reset_index(drop=True)


def short_track_log(dropped: pd.DataFrame, min_frames: int) -> pd.DataFrame:
    """Turn the dropped rows into drop-log rows, one per track."""
    rows = []
    if dropped.empty:
        return schema.empty_drops()
    for (scene, agent_id), part in dropped.groupby(KEY, observed=True):
        rows.append(
            schema.drop_row(
                stage="clean.drop_short_tracks",
                scene=str(scene),
                key=f"agent_id={int(agent_id)}",
                reason=f"track has {len(part)} frames, below {int(min_frames)}",
                n_rows=len(part),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def _velocity(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    """Add vx, vy, speed and heading. The first frame of a track gets NaN."""
    out = df.sort_values(KEY + ["frame"], kind="mergesort").copy()
    grouped = out.groupby(KEY, observed=True)

    step = grouped["frame"].diff()
    out["vx"] = grouped["x"].diff() / (step * dt)
    out["vy"] = grouped["y"].diff() / (step * dt)

    out["speed"] = np.hypot(out["vx"], out["vy"])
    heading = np.arctan2(out["vy"], out["vx"])
    out["heading"] = heading.where(out["speed"] > 0.0, other=np.nan)
    return out


def _neighbourhood(df: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Add nearest_dist and density.

    nearest_dist is the distance to the closest OTHER agent in the same scene
    and the same frame. It is NaN when the agent is alone in that frame.
    density is the count of OTHER agents inside the radius. It is 0 when the
    agent is alone, because a count of zero is a real count, not a gap.
    """
    out = df.copy()
    nearest = np.full(len(out), np.nan, dtype=np.float64)
    density = np.zeros(len(out), dtype=np.float64)

    positions = out[["x", "y"]].to_numpy(dtype=np.float64)
    codes = out.groupby(["scene", "frame"], observed=True, sort=False).ngroup()
    order = np.argsort(codes.to_numpy(), kind="stable")
    sorted_codes = codes.to_numpy()[order]
    bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
    blocks = np.split(order, bounds)

    for block in blocks:
        if block.size < 2:
            continue
        points = positions[block]
        delta = points[:, None, :] - points[None, :, :]
        distance = np.sqrt((delta * delta).sum(axis=-1))
        np.fill_diagonal(distance, np.inf)
        nearest[block] = distance.min(axis=1)
        density[block] = (distance <= radius).sum(axis=1)

    out["nearest_dist"] = nearest
    out["density"] = density
    return out


def add_kinematics(
    df: pd.DataFrame,
    dt: float = config.DT,
    radius: float = config.DENSITY_RADIUS_M,
) -> pd.DataFrame:
    """Add vx, vy, speed, heading, nearest_dist and density.

    The velocity is the first difference divided by dt. The first frame of a
    track has no earlier frame, so its velocity is NaN.
    """
    out = _velocity(df, dt)
    out = _neighbourhood(out, radius)
    out = out.sort_values(schema.TRAJECTORIES_KEY, kind="mergesort")
    return out.reset_index(drop=True)
