"""Neighbours per window.

See CONTRACT.md section 5.4 for the signatures.

N is the agent count of a window. E is N minus one, the number of edges that
point into the ego. E sets the cost of the brute force sweep in
src/faithfulness/index.py, because that sweep costs E forward passes per
window. This module therefore reports the compute budget of the whole study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.eda import plotting

QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


def ego_neighbourhood(
    windows: pd.DataFrame, trajectories: pd.DataFrame
) -> pd.DataFrame:
    """Return the local state of every ego at the last observed frame.

    Columns: window_id, scene, ego_id, n_agents, ego_density, ego_nearest_dist,
    ego_speed. The last observed frame is t0 plus N_HIST minus 1.
    """
    good = windows.loc[windows["eligible"], ["window_id", "scene", "t0", "ego_id", "n_agents"]]
    probe = good.copy()
    probe["frame"] = probe["t0"] + config.N_HIST - 1

    columns = ["scene", "frame", "agent_id", "density", "nearest_dist", "speed"]
    state = trajectories[columns].rename(columns={"agent_id": "ego_id"})
    merged = probe.merge(state, on=["scene", "frame", "ego_id"], how="left")

    return merged.rename(
        columns={
            "density": "ego_density",
            "nearest_dist": "ego_nearest_dist",
            "speed": "ego_speed",
        }
    ).drop(columns=["t0", "frame"])


def density_report(
    windows: pd.DataFrame,
    trajectories: pd.DataFrame,
    figure_dir: Path | None = None,
) -> pd.DataFrame:
    """Return one row per scene, plus a total row.

    Columns: eligible, mean_N, median_N, q90_N, max_N, sum_E, mean_E,
    forward_passes, ego_density, ego_nearest_dist.
    """
    local = ego_neighbourhood(windows, trajectories)
    rows = []

    for scene, part in local.groupby("scene", observed=True):
        n = part["n_agents"].to_numpy(dtype=float)
        rows.append(
            {
                "scene": str(scene),
                "eligible": len(part),
                "mean_N": float(n.mean()),
                "median_N": float(np.median(n)),
                "q90_N": float(np.quantile(n, 0.9)),
                "max_N": int(n.max()),
                "sum_E": int((n - 1).sum()),
                "mean_E": float((n - 1).mean()),
                "ego_density": float(part["ego_density"].mean()),
                "ego_nearest_dist": float(part["ego_nearest_dist"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    n_all = local["n_agents"].to_numpy(dtype=float)
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "scene": "TOTAL",
                        "eligible": len(local),
                        "mean_N": float(n_all.mean()),
                        "median_N": float(np.median(n_all)),
                        "q90_N": float(np.quantile(n_all, 0.9)),
                        "max_N": int(n_all.max()),
                        "sum_E": int((n_all - 1).sum()),
                        "mean_E": float((n_all - 1).mean()),
                        "ego_density": float(local["ego_density"].mean()),
                        "ego_nearest_dist": float(local["ego_nearest_dist"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # One unmasked pass per window, plus one pass per single edge.
    table["forward_passes"] = table["eligible"] + table["sum_E"]
    table["share_of_passes"] = table["forward_passes"] / float(
        table.loc[table["scene"] == "TOTAL", "forward_passes"].iloc[0]
    )
    _figure(local, table, figure_dir)
    return table


def _figure(local: pd.DataFrame, table: pd.DataFrame, figure_dir: Path | None) -> Path:
    """Draw the agent count per scene and the compute share."""
    import matplotlib.pyplot as plt

    plotting.style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    # Left: the empirical distribution of N, one line per scene.
    axis = axes[0]
    for scene in config.SCENES:
        values = local.loc[local["scene"] == scene, "n_agents"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        values = np.sort(values)
        share = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            share,
            where="post",
            color=plotting.SCENE_COLOUR[scene],
            label=scene,
        )
    axis.set_xlabel("agents in the window, N")
    axis.set_ylabel("share of eligible windows at or below N")
    axis.set_title("univ windows hold 25 agents; the others hold 3 to 6")
    axis.legend(loc="lower right", ncols=2)
    plotting.tidy(axis, grid_axis="both")

    # Right: the share of the forward passes.
    axis = axes[1]
    body = table.loc[table["scene"] != "TOTAL"]
    values = body["forward_passes"].to_numpy(dtype=float)
    bars = axis.bar(
        list(body["scene"]),
        values,
        color=plotting.scene_colours(body["scene"]),
        width=0.66,
        zorder=3,
    )
    plotting.label_bars(axis, bars, values, fmt="{:,.0f}")
    axis.set_title("Forward passes needed, one per window and one per single edge")
    axis.set_ylabel("forward passes")
    axis.set_ylim(0, values.max() * 1.18)
    axis.tick_params(axis="x", length=0)
    plotting.tidy(axis)

    figure.tight_layout()
    return plotting.save(figure, "eda_density", figure_dir)
