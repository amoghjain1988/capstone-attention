"""Counts per scene, per agent, per window.

See CONTRACT.md section 5.4 for the signatures.

The job of this module is to confirm or to correct the three numbers in the
proposal. The proposal says about 66,676 rows, about 1,950 pedestrians, about
2,416 windows, and that univ holds more than 60 percent of the rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.eda import plotting


def summary(
    trajectories: pd.DataFrame,
    windows: pd.DataFrame,
    figure_dir: Path | None = None,
) -> pd.DataFrame:
    """Return one row per scene, plus a total row.

    Columns: rows, share_of_rows, pedestrians, frames, first_frame, last_frame,
    windows, eligible, egos, windows_per_ego, mean_agents, median_agents,
    max_agents, edges_into_ego.
    """
    per_scene = []
    eligible = windows.loc[windows["eligible"]] if "eligible" in windows else windows

    for scene in config.SCENES:
        track = trajectories.loc[trajectories["scene"] == scene]
        block = windows.loc[windows["scene"] == scene]
        good = eligible.loc[eligible["scene"] == scene]
        n_egos = int(good["ego_id"].nunique()) if len(good) else 0
        per_scene.append(
            {
                "scene": scene,
                "rows": len(track),
                "pedestrians": int(track["agent_id"].nunique()),
                "frames": int(track["frame"].nunique()),
                "first_frame": int(track["frame"].min()) if len(track) else 0,
                "last_frame": int(track["frame"].max()) if len(track) else 0,
                "windows": len(block),
                "eligible": len(good),
                "egos": n_egos,
                "windows_per_ego": len(good) / n_egos if n_egos else np.nan,
                "mean_agents": float(good["n_agents"].mean()) if len(good) else np.nan,
                "median_agents": float(good["n_agents"].median()) if len(good) else np.nan,
                "max_agents": int(good["n_agents"].max()) if len(good) else 0,
                "edges_into_ego": int((good["n_agents"] - 1).sum()) if len(good) else 0,
            }
        )

    table = pd.DataFrame(per_scene)
    table["share_of_rows"] = table["rows"] / table["rows"].sum()

    total = {
        "scene": "TOTAL",
        "rows": int(table["rows"].sum()),
        "share_of_rows": 1.0,
        "pedestrians": int(table["pedestrians"].sum()),
        "frames": int(table["frames"].sum()),
        "first_frame": np.nan,
        "last_frame": np.nan,
        "windows": int(table["windows"].sum()),
        "eligible": int(table["eligible"].sum()),
        "egos": int(table["egos"].sum()),
        "windows_per_ego": table["eligible"].sum() / max(int(table["egos"].sum()), 1),
        "mean_agents": float(eligible["n_agents"].mean()) if len(eligible) else np.nan,
        "median_agents": float(eligible["n_agents"].median()) if len(eligible) else np.nan,
        "max_agents": int(eligible["n_agents"].max()) if len(eligible) else 0,
        "edges_into_ego": int(table["edges_into_ego"].sum()),
    }
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)

    order = [
        "scene",
        "rows",
        "share_of_rows",
        "pedestrians",
        "frames",
        "first_frame",
        "last_frame",
        "windows",
        "eligible",
        "egos",
        "windows_per_ego",
        "mean_agents",
        "median_agents",
        "max_agents",
        "edges_into_ego",
    ]
    table = table[order]
    _figure(table, figure_dir)
    return table


def _figure(table: pd.DataFrame, figure_dir: Path | None) -> Path:
    """Draw the four counts that decide the sample size."""
    plotting.style()
    body = table.loc[table["scene"] != "TOTAL"]
    scenes = list(body["scene"])
    colours = plotting.scene_colours(scenes)

    figure, axes = plt_subplots()
    panels = [
        ("rows", "Rows in trajectories", "{:,.0f}"),
        ("eligible", "Eligible windows", "{:,.0f}"),
        ("egos", "Ego pedestrians (the cluster unit)", "{:,.0f}"),
        ("edges_into_ego", "Edges into the ego (forward passes)", "{:,.0f}"),
    ]
    for axis, (column, title, fmt) in zip(axes.ravel(), panels):
        values = body[column].to_numpy(dtype=float)
        bars = axis.bar(scenes, values, color=colours, width=0.66, zorder=3)
        plotting.label_bars(axis, bars, values, fmt=fmt)
        axis.set_title(title)
        axis.set_ylim(0, values.max() * 1.18 if values.max() else 1.0)
        axis.tick_params(axis="x", length=0)
        plotting.tidy(axis)

    figure.suptitle(
        "Scene sizes. univ dominates the rows and the compute; eth is nearly empty.",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return plotting.save(figure, "eda_summary", figure_dir)


def plt_subplots():
    """Return a two by two grid. The helper keeps the import local."""
    import matplotlib.pyplot as plt

    return plt.subplots(2, 2, figsize=(10.0, 6.4))
