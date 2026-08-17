"""Step, speed and spacing.

See CONTRACT.md section 5.4 for the signatures.

The shape of a distribution decides the test. src/hypotheses/choose_test.py
takes branch 3a when the paired difference is symmetric, and branch 3b when it
is skewed or heavy tailed. The shift of the ablation is a distance, so it
cannot be negative and it has a long right tail. This module measures the
shape of every quantity we can see before the ablation runs, and it reports
which branch each shape implies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.eda import plotting

SHAPIRO_MAX_N = 5000

NOTES = {
    "step_m": (
        "step is speed times DT, so its shape equals the shape of speed. "
        "The panel shows the identity, not a second result."
    ),
    "turn_rad_per_frame": (
        "how much a pedestrian really manoeuvres. It bounds how large an "
        "honest ablation shift can be."
    ),
    "nearest_dist_m": "the spacing that src/features/proximity.py ranks on.",
    "density_agents": "the H3 context predictor.",
}


def shape_of(values: np.ndarray, name: str, seed: int = 0) -> dict:
    """Return the shape statistics of one column.

    The Shapiro-Wilk test uses at most SHAPIRO_MAX_N values, because the test
    rejects almost any large sample. Read the skew and the excess kurtosis as
    the real evidence. Read the p value only as a flag.
    """
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 8:
        return {"name": name, "n": int(clean.size)}

    generator = np.random.default_rng(seed)
    sample = clean
    if clean.size > SHAPIRO_MAX_N:
        sample = generator.choice(clean, size=SHAPIRO_MAX_N, replace=False)
    shapiro_p = float(stats.shapiro(sample).pvalue)

    quantiles = np.quantile(clean, [0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = quantiles[3] - quantiles[1]
    return {
        "name": name,
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(quantiles[2]),
        "sd": float(clean.std(ddof=1)),
        "iqr": float(iqr),
        "p01": float(quantiles[0]),
        "p99": float(quantiles[4]),
        "skew": float(stats.skew(clean)),
        "excess_kurtosis": float(stats.kurtosis(clean)),
        "shapiro_p": shapiro_p,
        "tail_ratio": float((quantiles[4] - quantiles[2]) / iqr) if iqr > 0 else np.nan,
        "share_zero": float((clean == 0).mean()),
    }


def branch_of(row: dict, skew_tol: float = 0.5, kurtosis_tol: float = 1.0) -> str:
    """Return the branch of choose_test.py that this shape implies."""
    skew = row.get("skew")
    kurtosis = row.get("excess_kurtosis")
    if skew is None or kurtosis is None:
        return "unknown"
    if abs(skew) <= skew_tol and abs(kurtosis) <= kurtosis_tol:
        return "3a symmetric, Wilcoxon signed-rank on the pseudomedian"
    return "3b skewed or heavy tail, sign test or bootstrap of the median"


def distributions(
    trajectories: pd.DataFrame, figure_dir: Path | None = None
) -> pd.DataFrame:
    """Return one row per measured quantity.

    The quantities are the step length, the speed, the spacing to the nearest
    neighbour, and the local density.
    """
    ordered = trajectories.sort_values(["scene", "agent_id", "frame"], kind="mergesort")
    delta = ordered.groupby(["scene", "agent_id"], observed=True)["heading"].diff()
    turn = np.abs(np.arctan2(np.sin(delta), np.cos(delta))).to_numpy()

    columns = {
        "step_m": ordered["speed"].to_numpy() * config.DT,
        "speed_mps": ordered["speed"].to_numpy(),
        "turn_rad_per_frame": turn,
        "nearest_dist_m": ordered["nearest_dist"].to_numpy(),
        "density_agents": ordered["density"].to_numpy(),
    }

    rows = [shape_of(values, name) for name, values in columns.items()]
    table = pd.DataFrame(rows)
    table["implied_branch"] = [branch_of(row) for row in rows]
    table["note"] = [NOTES.get(name, "") for name in table["name"]]
    _figure(columns, table, figure_dir)
    return table


def _figure(columns: dict[str, np.ndarray], table: pd.DataFrame, figure_dir) -> Path:
    """Draw a histogram and a normal quantile plot for every quantity."""
    import matplotlib.pyplot as plt

    plotting.style()
    names = list(columns)
    figure, axes = plt.subplots(2, len(names), figsize=(2.7 * len(names), 5.6))

    for index, name in enumerate(names):
        values = columns[name]
        values = values[np.isfinite(values)]

        axis = axes[0, index]
        axis.hist(values, bins=60, color=plotting.SERIES_1, zorder=3)
        axis.set_title(name)
        axis.set_ylabel("rows" if index == 0 else "")
        plotting.tidy(axis)

        row = table.loc[table["name"] == name].iloc[0]
        axis.annotate(
            f"skew {row['skew']:.2f}\nkurtosis {row['excess_kurtosis']:.2f}",
            xy=(0.97, 0.92),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )

        axis = axes[1, index]
        sample = values
        if sample.size > 5000:
            sample = np.random.default_rng(0).choice(sample, 5000, replace=False)
        stats.probplot(sample, dist="norm", plot=axis)
        axis.get_lines()[0].set_markersize(2.0)
        axis.get_lines()[0].set_color(plotting.SERIES_1)
        axis.get_lines()[1].set_color(plotting.INK_MUTED)
        axis.get_lines()[1].set_linewidth(1.2)
        axis.set_title("")
        axis.set_xlabel("normal quantile")
        axis.set_ylabel("observed" if index == 0 else "")
        plotting.tidy(axis, grid_axis="both")

    figure.suptitle(
        "Every quantity is right skewed. Expect branch 3b, not a t-test.",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return plotting.save(figure, "eda_distributions", figure_dir)
