"""Gaps and bad frames.

See CONTRACT.md section 5.4 for the signatures.

A missing value is NaN. This module counts every NaN and names its cause. A
count with a known cause is not a data fault. A count without a known cause is.

Three causes are expected.
1. The velocity of the first frame of a track. The track has no earlier frame.
2. The heading of a frame where the speed is 0. The direction does not exist.
3. The nearest distance of a frame where the agent is alone in the scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.eda import plotting

EXPECTED_CAUSE = {
    "vx": "the first frame of a track has no earlier frame",
    "vy": "the first frame of a track has no earlier frame",
    "speed": "the first frame of a track has no earlier frame",
    "heading": "the first frame of a track, or the speed is exactly 0",
    "nearest_dist": "the agent is alone in that frame",
}


def missing_report(
    trajectories: pd.DataFrame, figure_dir: Path | None = None
) -> pd.DataFrame:
    """Return one row per column of the trajectories table.

    Columns: column, n_missing, share_missing, expected, cause, unexplained.
    """
    n_tracks = trajectories.groupby(["scene", "agent_id"], observed=True).ngroups
    zero_speed = int((trajectories["speed"] == 0).sum())
    alone = _alone_count(trajectories)

    expected_count = {
        "vx": n_tracks,
        "vy": n_tracks,
        "speed": n_tracks,
        "heading": n_tracks + zero_speed,
        "nearest_dist": alone,
    }

    rows = []
    for column in trajectories.columns:
        n_missing = int(trajectories[column].isna().sum())
        expected = int(expected_count.get(column, 0))
        rows.append(
            {
                "column": column,
                "n_missing": n_missing,
                "share_missing": n_missing / len(trajectories),
                "expected": expected,
                "cause": EXPECTED_CAUSE.get(column, "none. the column must be complete"),
                "unexplained": n_missing - expected,
            }
        )

    table = pd.DataFrame(rows)
    _figure(trajectories, table, figure_dir)
    return table


def _alone_count(trajectories: pd.DataFrame) -> int:
    """Return the number of rows where the agent is alone in its frame."""
    size = trajectories.groupby(["scene", "frame"], observed=True)["agent_id"].transform(
        "size"
    )
    return int((size == 1).sum())


def frame_occupancy(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Return the frames that hold no agent at all, per scene.

    An empty frame is not a gap in a track. It means that nobody walked in the
    view at that moment. The window builder simply finds no window there.
    """
    rows = []
    for scene, part in trajectories.groupby("scene", observed=True):
        present = part["frame"].nunique()
        first, last = int(part["frame"].min()), int(part["frame"].max())
        span = last - first + 1
        # The scene univ joins two recordings with an offset, so remove the gap.
        n_sources = len(config.SCENE_SOURCES.get(str(scene), ("",)))
        if n_sources > 1:
            span = 0
            for index in range(n_sources):
                low = index * config.SOURCE_FILE_OFFSET
                high = low + config.SOURCE_FILE_OFFSET
                block = part.loc[(part["frame"] >= low) & (part["frame"] < high), "frame"]
                if len(block):
                    span += int(block.max()) - int(block.min()) + 1
        rows.append(
            {
                "scene": str(scene),
                "first_frame": first,
                "last_frame": last,
                "frames_in_span": span,
                "frames_with_an_agent": present,
                "empty_frames": span - present,
                "share_empty": (span - present) / span if span else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _figure(trajectories: pd.DataFrame, table: pd.DataFrame, figure_dir) -> Path:
    """Draw the missing count per column and the agents per frame."""
    import matplotlib.pyplot as plt

    plotting.style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    axis = axes[0]
    body = table.loc[table["n_missing"] > 0]
    if len(body):
        y = np.arange(len(body))
        axis.barh(
            y,
            body["n_missing"],
            color=plotting.SERIES_1,
            height=0.55,
            zorder=3,
            label="missing rows",
        )
        axis.scatter(
            body["expected"],
            y,
            color=plotting.SERIES_2,
            s=42,
            zorder=5,
            edgecolor=plotting.SURFACE,
            linewidth=1.5,
            label="rows a known cause explains",
        )
        for position, (_, row) in zip(y, body.iterrows()):
            axis.annotate(
                f"{int(row['n_missing']):,}  unexplained {int(row['unexplained'])}",
                xy=(row["n_missing"], position),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=plotting.INK_SECONDARY,
            )
        axis.set_yticks(y, list(body["column"]))
        axis.set_xlim(0, float(body["n_missing"].max()) * 1.55)
        axis.invert_yaxis()
        axis.legend(loc="lower right")
    axis.set_title("Every missing value has a known cause")
    axis.set_xlabel("rows")
    plotting.tidy(axis, grid_axis="x")

    axis = axes[1]
    for scene in config.SCENES:
        part = trajectories.loc[trajectories["scene"] == scene]
        if part.empty:
            continue
        per_frame = part.groupby("frame", observed=True).size().to_numpy()
        values = np.sort(per_frame)
        share = np.arange(1, values.size + 1) / values.size
        axis.step(
            values, share, where="post", color=plotting.SCENE_COLOUR[scene], label=scene
        )
    axis.set_title("Agents present per occupied frame")
    axis.set_xlabel("agents in the frame")
    axis.set_ylabel("share of occupied frames")
    axis.legend(loc="lower right", ncols=2)
    plotting.tidy(axis, grid_axis="both")

    figure.tight_layout()
    return plotting.save(figure, "eda_missing", figure_dir)
