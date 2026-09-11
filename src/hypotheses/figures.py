"""The figures of the hypothesis stage.

See docs/FINISH_PLAN.md section 3.8 for the first five. `summary_figure` is
the sixth: the poster needs one panel that holds the three primary results.
One function draws one figure. Every function takes a table or a result dict,
writes one png to `config.FIGURE_DIR` through `src.eda.plotting.save`, and
returns the path.

The house style lives in `src/eda/plotting.py`. That module states that it is
internal to `src/eda`, but docs/FINISH_PLAN.md section 3.8 tells this module
to use the same style, and FINISH_PLAN wins over the older note. One style and
one colour table keep every figure of the report readable as one set.

COLOUR. The five categorical slots of `plotting.SCENE_COLOUR` are a validated
palette. This module reuses those five hex values for the five drawn arms, in
a fixed order. An arm keeps its colour in every panel of every figure, so the
reader learns the colour once. Three of the five slots fall below a contrast
ratio of 3 to 1 on the light surface, so every panel with two or more series
also carries a legend, and no figure states identity by colour alone.

NUMBERS. The bands of `curves_figure` are a picture, not a result. They use
`BAND_N_BOOT` resamples, which is far below `config.N_BOOT`. Every interval
that a table or a report states comes from `src/stats/clustered.py` at
`config.N_BOOT`. Do not read a band of a figure as a reported interval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.eda import plotting
from src.hypotheses.h3_context import CONTEXT_COLUMNS
from src.stats.clustered import cluster_bootstrap_ci, cluster_labels

# src/eda/plotting.py selects the Agg backend when it is imported, so this
# import never opens a window.
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "curves_figure",
    "paired_figure",
    "fi_figure",
    "h3_figure",
    "loso_figure",
    "summary_figure",
]

# ---------------------------------------------------------------------------
# Constants of the drawn form
# ---------------------------------------------------------------------------

# The arms that curves_figure draws against n_removed. The arm `single` holds
# one row per edge, so it has no curve, and `weight_matched` matches mass, not
# count, so it belongs on the second panel.
CURVE_ARMS = ("morf", "lerf", "nearest", "random")

# The arms of the second panel. Both go on the removed mass axis.
MASS_ARMS = ("morf", "weight_matched")

# The arm that each hypothesis puts against morf. The figures name the arm,
# because "h1" alone does not tell the reader what the panel compares.
RIVAL = {"h1": "lerf", "h2": "nearest"}

# The five arms take the five validated categorical slots, in a fixed order.
# plotting.py stays the one place that knows a hex value.
_SLOTS = tuple(plotting.SCENE_COLOUR[scene] for scene in config.SCENES)
ARM_COLOUR: dict[str, str] = dict(
    zip(("morf", "lerf", "nearest", "random", "weight_matched"), _SLOTS)
)

# The band of a figure. These numbers change the drawn band only. They never
# change a number that a table or a report states. See the module docstring.
BAND_N_BOOT = 500
BAND_LEVEL = 0.95
BAND_ALPHA = 0.16

# The count of mass bins on the second panel of curves_figure.
MASS_BINS = 5

# The top of the second panel of curves_figure, as a quantile of the drawn
# shift. A few windows carry a shift of more than 1 metre. Without this clip
# those windows set the axis, and the two mean curves fall into the lowest
# fifth of the panel, where the reader cannot separate them. The title states
# how many dots the clip holds outside the panel.
MASS_CLIP_Q = 0.99

# The two lanes of one scene slot in fi_figure, in x units of that slot. A
# window at the frozen floor of config.MIN_EGO_EDGES reaches only +1 or -1.
# Those windows pile on top of the other windows at +1, so they get their own
# lane. The reader then sees them, and the reader also sees the rest.
LANE_MAIN = -0.16
LANE_TWO_EDGE = 0.26
# The half width of the jitter of each lane, in x units of one scene slot.
JITTER = 0.14
JITTER_TWO_EDGE = 0.07

# The interval of h3_figure is b plus or minus this multiple of the standard
# error. 1.96 is the normal quantile of a 95 percent interval.
Z_95 = 1.96


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scene_of(window_ids) -> pd.Series:
    """Return the scene of every window key."""
    return pd.Series(window_ids).astype(str).map(
        lambda wid: schema.split_window_id(wid)[0]
    )


def _present_scenes(scenes) -> list[str]:
    """Return the scenes that appear, in the frozen order of config.SCENES."""
    seen = {str(scene) for scene in pd.Series(scenes).dropna()}
    known = [scene for scene in config.SCENES if scene in seen]
    return known + sorted(seen - set(config.SCENES))


def _primary_rows(curves: pd.DataFrame) -> pd.DataFrame:
    """Return one row per window, arm and step, under the primary policy.

    The `perturbation_curves` table holds one row per draw and per mask
    policy. This function keeps `config.PRIMARY_MASK_POLICY` and takes the
    mean over the draws, so the unit of every panel is one window at one
    step. AgentFormer is deterministic, so the draws agree and the mean
    changes nothing. Adds a `scene` column from `schema.split_window_id`.
    """
    frame = pd.DataFrame(curves).copy()
    if "mask_policy" in frame.columns:
        keep = frame["mask_policy"].astype(str) == str(config.PRIMARY_MASK_POLICY)
        if bool(keep.any()):
            frame = frame.loc[keep]

    frame["arm"] = frame["arm"].astype(str)
    measures = [name for name in ("shift", "removed_mass") if name in frame.columns]
    grouped = frame.groupby(
        ["window_id", "ego_id", "arm", "n_removed"], observed=True, as_index=False
    )[measures].mean()
    grouped["scene"] = _scene_of(grouped["window_id"]).to_numpy()
    return grouped


def _band(values: np.ndarray, clusters: np.ndarray, stat=np.mean) -> tuple:
    """Return the point estimate and the pairs cluster bootstrap band.

    The bootstrap resamples clusters, not rows. `cluster_labels` builds the
    clusters from `window_id` and `ego_id` at `config.CLUSTER_UNIT`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(values))
    low, high = cluster_bootstrap_ci(
        values,
        clusters,
        stat=stat,
        n_boot=BAND_N_BOOT,
        level=BAND_LEVEL,
        seed=config.SEED,
    )
    return point, low, high


_CURVE_COLUMNS = ["x", "mean", "low", "high", "n_windows"]


def _step_curve(rows: pd.DataFrame, arm: str, n_max: int) -> pd.DataFrame:
    """Return the mean shift and the band of one arm at every step.

    The mean at a step runs over the windows that hold that step. A window
    that stops earlier drops out of the later steps.
    """
    part = rows.loc[(rows["arm"] == arm) & (rows["n_removed"] <= int(n_max))]
    out = []
    for step, block in part.groupby("n_removed", observed=True):
        point, low, high = _band(
            block["shift"].to_numpy(dtype=np.float64), cluster_labels(block)
        )
        out.append(
            {
                "x": float(step),
                "mean": point,
                "low": low,
                "high": high,
                "n_windows": int(len(block)),
            }
        )
    if not out:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    return pd.DataFrame(out).sort_values("x").reset_index(drop=True)


def _mass_curve(rows: pd.DataFrame, arm: str, n_bins: int = MASS_BINS) -> pd.DataFrame:
    """Return the mean shift and the band of one arm over mass bins.

    The removed mass is continuous, so the rows enter quantile bins first.
    The x of a bin is the mean removed mass inside that bin.
    """
    part = rows.loc[rows["arm"] == arm]
    if "removed_mass" not in part.columns:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    part = part.dropna(subset=["removed_mass", "shift"])
    if part.empty:
        return pd.DataFrame(columns=_CURVE_COLUMNS)

    mass = part["removed_mass"].to_numpy(dtype=np.float64)
    wanted = int(max(1, min(n_bins, np.unique(mass).size)))
    edges = np.unique(np.quantile(mass, np.linspace(0.0, 1.0, wanted + 1)))
    if edges.size < 3:
        index = np.zeros(mass.size, dtype=np.int64)
    else:
        index = np.clip(np.digitize(mass, edges[1:-1]), 0, edges.size - 2)

    out = []
    for bin_id in np.unique(index):
        block = part.loc[index == bin_id]
        point, low, high = _band(
            block["shift"].to_numpy(dtype=np.float64), cluster_labels(block)
        )
        out.append(
            {
                "x": float(block["removed_mass"].mean()),
                "mean": point,
                "low": low,
                "high": high,
                "n_windows": int(len(block)),
            }
        )
    if not out:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    return pd.DataFrame(out).sort_values("x").reset_index(drop=True)


def _draw_curve(axis, curve: pd.DataFrame, colour: str, label: str, band: bool) -> None:
    """Draw one curve, and its band when `band` is True."""
    if curve.empty:
        return
    if band:
        axis.fill_between(
            curve["x"],
            curve["low"],
            curve["high"],
            color=colour,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
    axis.plot(
        curve["x"],
        curve["mean"],
        color=colour,
        marker="o",
        markersize=5,
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=0.9,
        zorder=3,
        label=label,
    )


def _reference_line(axis, value: float, text: str, offset: int = -3) -> None:
    """Draw one dashed vertical reference line and label it directly."""
    if not np.isfinite(value):
        return
    axis.axvline(
        value, color=plotting.BASELINE, linestyle="--", linewidth=1.1, zorder=4
    )
    axis.annotate(
        text,
        xy=(value, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, offset),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=plotting.INK_MUTED,
    )


# ---------------------------------------------------------------------------
# 1. The perturbation curves
# ---------------------------------------------------------------------------


def curves_figure(
    curves: pd.DataFrame, n_max: int = 6, figure_dir: Path | None = None
) -> Path:
    """Draw the perturbation curves of every arm. Return the png path.

    The first panel holds one line per arm of CURVE_ARMS. The point at a
    step is the mean shift over the windows that hold that step, and the
    band is the pairs cluster bootstrap at BAND_LEVEL. The second panel
    holds `weight_matched` and `morf` against the removed mass, because
    `weight_matched` matches mass and not count. The third row holds one
    small panel per scene. The shift is in metres, so no axis needs a log
    scale.
    """
    rows = _primary_rows(curves)
    scenes = _present_scenes(rows["scene"])

    # The study holds five scenes, so the third row holds five panels. A sixth
    # scene gets a sixth panel. No scene ever drops out of the figure.
    n_columns = max(5, len(scenes))

    plotting.style()
    figure = plt.figure(figsize=(2.2 * n_columns, 10.0))
    grid = figure.add_gridspec(3, n_columns, height_ratios=(1.25, 1.0, 0.95))
    axis_step = figure.add_subplot(grid[0, :])
    axis_mass = figure.add_subplot(grid[1, :])
    small = [figure.add_subplot(grid[2, column]) for column in range(n_columns)]

    # Panel 1. Every arm against the count of removed edges.
    steps: list[float] = []
    for arm in CURVE_ARMS:
        curve = _step_curve(rows, arm, n_max)
        steps.extend(curve["x"].tolist())
        _draw_curve(axis_step, curve, ARM_COLOUR[arm], arm, band=True)
    axis_step.set_xlabel("edges removed, n_removed")
    axis_step.set_ylabel("mean shift, metres")
    axis_step.set_title(
        "Every arm, pooled over the windows that hold the step. "
        f"The band is a {int(BAND_LEVEL * 100)} percent cluster bootstrap."
    )
    if steps:
        axis_step.set_xticks(sorted(set(steps)))
    if axis_step.get_legend_handles_labels()[0]:
        axis_step.legend(loc="upper left", ncols=4)
    plotting.tidy(axis_step)

    # Panel 2. The mass matched arm needs the mass axis, so morf joins it there.
    # The dots are faint, because the two mean curves carry the message and a
    # dense field of solid dots hides them.
    dots: list[float] = []
    band_top = 0.0
    for arm in MASS_ARMS:
        part = rows.loc[rows["arm"] == arm]
        if "removed_mass" in part.columns and not part.empty:
            dots.extend(part["shift"].dropna().tolist())
            axis_mass.plot(
                part["removed_mass"],
                part["shift"],
                linestyle="none",
                marker="o",
                markersize=2.4,
                color=ARM_COLOUR[arm],
                alpha=0.16,
                markeredgewidth=0,
                zorder=2,
            )
        curve = _mass_curve(rows, arm)
        if not curve.empty:
            band_top = max(band_top, float(np.nanmax(curve["high"].to_numpy())))
        _draw_curve(axis_mass, curve, ARM_COLOUR[arm], arm, band=True)

    # The clip keeps the axis on the mean curves. See MASS_CLIP_Q.
    n_hidden = 0
    if dots:
        values = np.asarray(dots, dtype=np.float64)
        values = values[np.isfinite(values)]
    else:
        values = np.array([], dtype=np.float64)
    if values.size:
        top = float(np.quantile(values, MASS_CLIP_Q))
        top = max(top, band_top) * 1.08
        low = min(float(values.min()), 0.0)
        if top > low:
            axis_mass.set_ylim(low - 0.03 * (top - low), top)
            n_hidden = int((values > top).sum())

    axis_mass.set_xlabel("attention mass removed, a share of the ego attention")
    # One dot is one window at one step, so this axis is not a mean.
    axis_mass.set_ylabel("shift, metres")
    hidden_text = (
        f"\nThe axis stops at the {int(MASS_CLIP_Q * 100)}th percentile of the "
        f"shift, so {n_hidden} dots of {int(values.size)} sit above the top."
        if n_hidden
        else ""
    )
    axis_mass.set_title(
        "morf against weight_matched on the mass axis. One dot is one window "
        "at one step. The line is the mean of a mass bin." + hidden_text
    )
    if axis_mass.get_legend_handles_labels()[0]:
        axis_mass.legend(loc="upper left", ncols=2)
    plotting.tidy(axis_mass)

    # Panel row 3. One panel per scene, the same arms and the same colours.
    lows: list[float] = []
    highs: list[float] = []
    for position, axis in enumerate(small):
        if position >= len(scenes):
            axis.set_visible(False)
            continue
        scene = scenes[position]
        block = rows.loc[rows["scene"] == scene]
        for arm in CURVE_ARMS:
            curve = _step_curve(block, arm, n_max)
            if curve.empty:
                continue
            lows.append(float(curve["mean"].min()))
            highs.append(float(curve["mean"].max()))
            axis.plot(
                curve["x"],
                curve["mean"],
                color=ARM_COLOUR[arm],
                linewidth=1.5,
                marker="o",
                markersize=3.5,
                markeredgecolor=plotting.SURFACE,
                markeredgewidth=0.7,
                zorder=3,
            )
        # The count states the sample of the panel, so a flat panel of a small
        # scene does not read as a flat panel of a large one.
        n_windows = int(block["window_id"].nunique())
        axis.set_title(f"{scene}, {n_windows} windows")
        axis.set_xlabel("edges removed")
        if steps:
            axis.set_xticks(sorted(set(steps)))
        if position == 0:
            # Every scene panel shares one y axis, so the reader compares the
            # panels. The label says so, because a shared axis flattens the
            # panel of a scene with a small shift.
            axis.set_ylabel("mean shift, metres\n(one shared y axis)")
        plotting.tidy(axis)
    if lows and highs:
        pad = 0.08 * max(float(np.ptp(np.array(highs + lows))), 1e-6)
        for position, axis in enumerate(small):
            if position < len(scenes):
                axis.set_ylim(min(lows) - pad, max(highs) + pad)

    figure.suptitle(
        "The removal of the strongest attention edge first moves the forecast most",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return plotting.save(figure, "hyp_curves", figure_dir)


# ---------------------------------------------------------------------------
# 2. The paired differences
# ---------------------------------------------------------------------------


def _paired_values(table: pd.DataFrame) -> np.ndarray:
    """Return the finite d values of one D table."""
    values = np.asarray(pd.DataFrame(table).get("d", []), dtype=np.float64)
    return values[np.isfinite(values)]


def _paired_bins(values: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    """Return bin edges over `span` with one edge at exactly 0.

    A bin that straddles 0 mixes the windows that move with the windows that
    do not, and the reader then sees mass on the wrong side of 0. The edges
    below start at 0 and step outwards, so 0 is always an edge.
    """
    left, right = float(span[0]), float(span[1])
    count = int(np.clip(round(2.0 * np.sqrt(max(values.size, 1))), 8, 40))
    width = (right - left) / count
    if not np.isfinite(width) or width <= 0.0:
        return np.linspace(left, right, 9)
    below = int(np.ceil((0.0 - left) / width))
    above = int(np.ceil((right - 0.0) / width))
    return np.arange(-below, above + 1, dtype=np.float64) * width


def _paired_panel(
    axis,
    table: pd.DataFrame,
    name: str,
    partner: str,
    colour: str,
    span: tuple[float, float],
) -> None:
    """Draw the histogram of one D table with its median and its zero line."""
    values = _paired_values(table)
    n_rows = int(values.size)
    n_zero = int(np.count_nonzero(values == 0.0))
    # share_positive counts the POSITIVE differences over the NON-ZERO ones.
    # That is the effect of the sign test, and it is the number that the
    # verdicts table reports. A share over every window would put a second,
    # smaller number for the same name into the report.
    n_used = n_rows - n_zero
    n_positive = int(np.count_nonzero(values > 0.0))
    share = (n_positive / n_used) if n_used else float("nan")
    median = float(np.median(values)) if n_rows else float("nan")

    tallest = 1.0
    if n_rows:
        counts, _, _ = axis.hist(
            values,
            bins=_paired_bins(values, span),
            color=colour,
            # rwidth leaves a thin gap of the surface between two bars. The gap
            # separates the bars, so no bar needs a border.
            rwidth=0.94,
            zorder=3,
        )
        tallest = max(float(np.max(counts)), 1.0)

    # The head room holds the two reference labels. Without it the tallest bar
    # reaches the top of the panel and the labels land on top of that bar.
    axis.set_xlim(span[0], span[1])
    axis.set_ylim(0.0, tallest * 1.30)

    # The label of the zero line goes left of the line and the label of the
    # median goes right of it, so the two labels never overlap.
    axis.axvline(
        0.0, color=plotting.BASELINE, linestyle="--", linewidth=1.1, zorder=4
    )
    axis.annotate(
        "zero",
        xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-4, -3),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=plotting.INK_MUTED,
    )
    if np.isfinite(median):
        axis.axvline(median, color=plotting.INK_PRIMARY, linewidth=1.4, zorder=5)
        axis.annotate(
            "median",
            xy=(median, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )

    # share positive is the headline effect of this hypothesis, so the panel
    # states it beside the count of ties.
    axis.set_title(
        f"{name.upper()}: morf against {partner}\n"
        f"share positive = {share:.3f} of the {n_used} windows that are not ties\n"
        f"median = {median:.4f} m, n = {n_rows} windows, ties at 0 = {n_zero}"
    )
    axis.set_xlabel(f"d = shift(morf) - shift({partner}), metres")
    axis.set_ylabel("windows")
    plotting.tidy(axis)


def paired_figure(
    d_h1: pd.DataFrame, d_h2: pd.DataFrame, figure_dir: Path | None = None
) -> Path:
    """Draw the paired difference of H1 and H2. Return the png path.

    `d_h1` and `d_h2` are the D tables of docs/FINISH_PLAN.md section 3.5.
    Each one holds window_id, ego_id, shift_a, shift_b and d. A d of exactly
    0 is a tie, and the Pratt rule keeps it, so the title counts the ties.

    The two panels share one x range, because the reader compares the two
    distributions. They keep their own y range, because the count of windows
    in a bar is not a comparison.
    """
    both = np.concatenate(
        [_paired_values(d_h1), _paired_values(d_h2), np.zeros(1, dtype=np.float64)]
    )
    low = float(np.min(both))
    high = float(np.max(both))
    pad = 0.04 * max(high - low, 1e-6)
    span = (low - pad, high + pad)

    plotting.style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    _paired_panel(axes[0], d_h1, "h1", "lerf", plotting.SERIES_1, span)
    _paired_panel(axes[1], d_h2, "h2", "nearest", plotting.SERIES_2, span)
    figure.suptitle(
        "The mass of both differences sits on the positive side of zero",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return plotting.save(figure, "hyp_paired", figure_dir)


# ---------------------------------------------------------------------------
# 3. The faithfulness index
# ---------------------------------------------------------------------------


def fi_figure(faithfulness: pd.DataFrame, figure_dir: Path | None = None) -> Path:
    """Draw the faithfulness index per scene. Return the png path.

    One strip holds one scene. One dot is one window. A window at the frozen
    floor of `config.MIN_EGO_EDGES` reaches only +1 or -1, because the floor
    is the mean of two shifts and the ceiling is their maximum. Those windows
    take a triangle AND their own lane on the right of the scene slot. In one
    lane with the rest they hide inside the pile of windows at +1, and the
    reader cannot count them. The tick label of the scene states how many of
    them the scene holds. The title states the count of null values and the
    count of those windows over every scene.
    """
    table = pd.DataFrame(faithfulness).copy()
    table["scene"] = _scene_of(table["window_id"]).to_numpy()
    n_nan = int(table["fi"].isna().sum())
    two_edge = (
        (table["n_edges"] == config.MIN_EGO_EDGES).to_numpy()
        if "n_edges" in table.columns
        else np.zeros(len(table), dtype=bool)
    )
    table["two_edge"] = two_edge
    n_two_edge = int(two_edge.sum())

    good = table.loc[table["fi"].notna()]
    scenes = _present_scenes(good["scene"] if len(good) else table["scene"])
    pooled = float(good["fi"].median()) if len(good) else float("nan")

    plotting.style()
    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    generator = np.random.default_rng(config.SEED)

    two_edge_counts: dict[str, int] = {}
    for position, scene in enumerate(scenes):
        part = good.loc[good["scene"] == scene]
        values = part["fi"].to_numpy(dtype=np.float64)
        flag = part["two_edge"].to_numpy(dtype=bool)
        two_edge_counts[scene] = int(flag.sum())
        if values.size == 0:
            continue
        colour = plotting.SCENE_COLOUR.get(scene, plotting.INK_MUTED)
        # One lane per group. The lane, not the marker alone, separates the
        # two-edge windows from the rest.
        for mask, marker, lane, half in (
            (~flag, "o", LANE_MAIN, JITTER),
            (flag, "^", LANE_TWO_EDGE, JITTER_TWO_EDGE),
        ):
            if not mask.any():
                continue
            jitter = generator.uniform(-half, half, size=int(mask.sum()))
            axis.plot(
                position + lane + jitter,
                values[mask],
                linestyle="none",
                marker=marker,
                markersize=4.6,
                color=colour,
                alpha=0.75,
                markeredgecolor=plotting.SURFACE,
                markeredgewidth=0.7,
                zorder=3,
            )
        axis.plot(
            [position + LANE_MAIN - 0.20, position + LANE_TWO_EDGE + 0.14],
            [float(np.median(values))] * 2,
            color=plotting.INK_PRIMARY,
            linewidth=1.6,
            zorder=4,
        )

    for level, text in ((0.0, "chance"), (1.0, "the strongest edge")):
        axis.axhline(
            level, color=plotting.BASELINE, linestyle="--", linewidth=1.0, zorder=2
        )
        axis.annotate(
            text,
            xy=(1.0, level),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=plotting.INK_MUTED,
        )
    if np.isfinite(pooled):
        axis.axhline(pooled, color=plotting.INK_SECONDARY, linewidth=1.4, zorder=4)
        # The label goes BELOW the line. The pooled median sits near 1, and a
        # label above the line lands on the "the strongest edge" line and on
        # the dots of the first scene.
        axis.annotate(
            f"pooled median {pooled:.3f}",
            xy=(0.0, pooled),
            xycoords=("axes fraction", "data"),
            xytext=(3, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )

    # The two markers carry the edge count. Grey keys, because the colour of a
    # dot already carries the scene.
    axis.plot(
        [], [], linestyle="none", marker="o", markersize=4.6,
        color=plotting.INK_MUTED,
        label=f"left lane: more than {config.MIN_EGO_EDGES} edges",
    )
    axis.plot(
        [], [], linestyle="none", marker="^", markersize=4.6,
        color=plotting.INK_MUTED,
        label=f"right lane: {config.MIN_EGO_EDGES} edges, fi is +1 or -1",
    )
    # The black bar of a scene needs a key. Without one the reader does not
    # know what it states.
    axis.plot(
        [], [], color=plotting.INK_PRIMARY, linewidth=1.6, label="median of the scene"
    )

    # The head room at the top holds the legend and the foot room holds
    # nothing, so no key and no label sits on top of a dot.
    values = good["fi"].to_numpy(dtype=np.float64) if len(good) else np.array([0.0, 1.0])
    low = min(float(np.nanmin(values)), 0.0)
    high = max(float(np.nanmax(values)), 1.0)
    span = max(high - low, 1e-6)
    axis.set_ylim(low - 0.08 * span, high + 0.30 * span)
    axis.legend(loc="upper left", ncols=3)

    axis.set_xticks(range(len(scenes)))
    # The tick label carries the count of the right lane, so the reader reads
    # the count without a second figure.
    axis.set_xticklabels(
        [
            f"{scene}\n{config.MIN_EGO_EDGES} edges: {two_edge_counts.get(scene, 0)}"
            for scene in scenes
        ]
    )
    # The left margin holds the pooled median label and the right margin holds
    # the two reference labels. No dot reaches either margin.
    axis.set_xlim(-1.05, len(scenes) - 1 + LANE_TWO_EDGE + 1.05)
    axis.tick_params(axis="x", length=0)
    axis.set_ylabel("faithfulness index, no unit")
    axis.set_title(
        "The faithfulness index per scene. 1 is the strongest edge, 0 is chance.\n"
        f"{len(good)} windows with a defined index, null fi = {n_nan}, "
        f"windows with {config.MIN_EGO_EDGES} edges = {n_two_edge}, "
        f"pooled median = {pooled:.3f}"
    )
    plotting.tidy(axis)
    figure.tight_layout()
    return plotting.save(figure, "hyp_faithfulness", figure_dir)


# ---------------------------------------------------------------------------
# 4. The H3 coefficients
# ---------------------------------------------------------------------------


def _clean_term(term: str) -> str:
    """Return a plain label for one regression term."""
    text = str(term)
    if text.startswith("C(scene)") and "[" in text and "]" in text:
        inner = text[text.index("[") + 1 : text.rindex("]")]
        return "scene " + inner.replace("T.", "")
    return text


def _p_text(value) -> str:
    """Return one p value as text. A very small p reads as a bound.

    A p of 0.00004 prints as 0.0000 at four places, and a reader takes that
    for exactly zero. This function prints a bound instead.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    if not np.isfinite(number):
        return "not available"
    if number < 0.0001:
        return "< 0.0001"
    return f"{number:.4f}"


def _reference_scene(terms) -> str:
    """Return the scene that the scene terms compare against.

    A `C(scene)` block drops one level, and every other level states a
    difference against that dropped level. The dropped level is the scene of
    `config.SCENES` that carries no term.
    """
    named = {
        _clean_term(term)[len("scene ") :]
        for term in terms
        if str(term).startswith("C(scene)")
    }
    for scene in config.SCENES:
        if scene not in named:
            return scene
    return "the dropped scene"


def _coefficient_panel(axis, block: pd.DataFrame, colour: str, title: str, x_label: str):
    """Draw one block of coefficients, one row per term, top to bottom."""
    positions = np.arange(len(block))[::-1]
    axis.errorbar(
        block["b"].to_numpy(dtype=np.float64),
        positions,
        xerr=Z_95 * block["se"].to_numpy(dtype=np.float64),
        fmt="o",
        markersize=6,
        color=colour,
        ecolor=colour,
        elinewidth=1.6,
        capsize=0,
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=0.9,
        zorder=3,
    )
    _reference_line(axis, 0.0, "no effect", offset=-3)
    axis.set_yticks(positions)
    axis.set_yticklabels([_clean_term(term) for term in block["term"]])
    axis.set_ylim(-0.7, float(len(block)) - 0.3)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel(x_label)
    axis.set_title(title)
    plotting.tidy(axis, grid_axis="x")


def h3_figure(
    h3_result: dict,
    figure_dir: Path | None = None,
    p_bootstrap: float | None = None,
) -> Path:
    """Draw the coefficients of the headline H3 fit. Return the png path.

    `h3_result` is the dict of `src/hypotheses/h3_context.py`. The headline
    key names the fit to draw. The bar of a term is b plus or minus Z_95
    standard errors, from the cluster-robust errors of that fit. The
    intercept stays out, because its scale hides every slope.

    TWO PANELS, NOT ONE. A scene term is about 1 fi unit and a context term
    is about 0.1 fi unit. On one x axis the scene terms set the range and the
    four context terms collapse onto the zero line, where the reader cannot
    read them. The two blocks also carry two different units: a context term
    is fi per one standard deviation, and a scene term is a difference in fi
    against the dropped scene. Two panels give each block its own range, its
    own unit and its own title.

    `p_bootstrap` is the joint p of the wild cluster bootstrap. The study
    holds too few clusters for the asymptotic Wald test, so the report
    headlines the bootstrap p. Pass it, and the title states it and names it.
    Leave it out, and the title states the asymptotic Wald p alone.
    """
    headline = str(h3_result["headline"])
    fit = h3_result[headline]
    table = pd.DataFrame(fit["coefficients"]).copy()
    table["term"] = table["term"].astype(str)
    table = table.loc[~table["term"].isin(("Intercept", "const"))]

    context = [name for name in CONTEXT_COLUMNS if name in set(table["term"])]
    others = [name for name in table["term"] if name not in set(context)]
    table = table.set_index("term").loc[context + others].reset_index()
    context_block = table.loc[table["term"].isin(context)].reset_index(drop=True)
    scene_block = table.loc[~table["term"].isin(context)].reset_index(drop=True)

    blocks = [
        (
            context_block,
            plotting.SERIES_1,
            "The context block, the question of H3",
            "b, fi per one standard deviation of the term",
        ),
        (
            scene_block,
            plotting.INK_MUTED,
            "The scene block, the fixed effects",
            f"b, fi against the scene {_reference_scene(table['term'])}",
        ),
    ]
    # A fit of one scene holds no scene term. An empty panel states nothing,
    # so the figure then holds one panel.
    blocks = [item for item in blocks if len(item[0])]
    if not blocks:
        blocks = [(table, plotting.SERIES_1, "No term outside the intercept", "b")]

    plotting.style()
    tallest = max(len(item[0]) for item in blocks)
    height = max(3.4, 0.46 * tallest + 2.6)
    figure, axes = plt.subplots(
        1, len(blocks), figsize=(5.4 * len(blocks), height), squeeze=False
    )
    for axis, (block, colour, title, x_label) in zip(axes[0], blocks):
        _coefficient_panel(axis, block, colour, title, x_label)

    wald = _p_text(fit["joint_p"])
    clusters = int(fit["n_clusters"])
    if p_bootstrap is None:
        line = f"asymptotic Wald joint p of the context terms = {wald}"
        second = ""
    else:
        line = (
            "wild cluster bootstrap joint p of the context terms = "
            f"{_p_text(p_bootstrap)}"
        )
        second = (
            f". The asymptotic Wald joint p is {wald}, but {clusters} clusters "
            "are too few for that test."
        )
    figure.suptitle(
        f"H3 headline fit ({headline}): {line}\n"
        f"partial R squared = {float(fit['partial_r2']):.3f}, "
        f"n = {int(fit['n'])} windows, {clusters} clusters" + second,
        x=0.02,
        ha="left",
        fontsize=11,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.text(
        0.02,
        0.015,
        f"The bar is b plus or minus {Z_95} cluster-robust standard errors, "
        "a 95 percent interval. The two panels hold two units, so they hold "
        "two x ranges.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=plotting.INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    return plotting.save(figure, "hyp_h3_coefficients", figure_dir)


# ---------------------------------------------------------------------------
# 5. The leave one scene out forest
# ---------------------------------------------------------------------------


def _forest_panel(axis, rows: pd.DataFrame, pooled: float, name: str) -> None:
    """Draw one forest panel: one row per held-out scene, with the interval.

    The bar runs from `ci_low` to `ci_high` and the marker sits at
    `location_m`. The bar is a plain segment, not an error bar around the
    marker. A percentile bootstrap interval need not hold the point that it
    describes, so an error bar would ask matplotlib for a negative half
    width and would raise. A segment states the two bounds as they stand.

    A cap marks each bound. An interval of zero width then still shows one
    tick. Without a cap that interval draws nothing, and the reader takes the
    row for a row with no interval at all.

    The count of windows and the count of clusters go INTO the tick label.
    They took a column of empty panel on the right before, and that column
    held about half the panel.
    """
    scenes = _present_scenes(rows["variant"])
    rows = rows.set_index(rows["variant"].astype(str))
    rival = RIVAL.get(name, "the rival arm")

    lows: list[float] = []
    highs: list[float] = []
    missing: set[str] = set()
    for position, scene in enumerate(scenes):
        row = rows.loc[scene]
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        point = float(row["location_m"])
        height = len(scenes) - 1 - position
        colour = plotting.SCENE_COLOUR.get(scene, plotting.INK_MUTED)
        lows.extend([value for value in (low, point) if np.isfinite(value)])
        highs.extend([value for value in (high, point) if np.isfinite(value)])
        if np.isfinite(low) and np.isfinite(high):
            axis.plot(
                [low, high],
                [height, height],
                color=colour,
                linewidth=1.6,
                solid_capstyle="butt",
                zorder=3,
            )
            for bound in (low, high):
                axis.plot(
                    [bound, bound],
                    [height - 0.13, height + 0.13],
                    color=colour,
                    linewidth=1.6,
                    zorder=3,
                )
        else:
            missing.add(scene)
        axis.plot(
            [point],
            [height],
            linestyle="none",
            marker="o",
            markersize=6,
            color=colour,
            markeredgecolor=plotting.SURFACE,
            markeredgewidth=0.9,
            zorder=4,
        )

    span = (max(highs) - min(lows)) if lows and highs else 1.0
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    left = (min(lows) if lows else 0.0) - 0.10 * span
    right = (max(highs) if highs else 1.0) + 0.22 * span
    if np.isfinite(pooled):
        left = min(left, pooled - 0.10 * span)
    axis.set_xlim(left, right)

    # Draw the zero line only when zero falls inside the panel. A label at an
    # edge that carries no line misleads the reader.
    if left <= 0.0 <= right:
        _reference_line(axis, 0.0, "zero", offset=-3)
    if np.isfinite(pooled):
        axis.axvline(pooled, color=plotting.INK_SECONDARY, linewidth=1.4, zorder=4)
        axis.annotate(
            f"pooled {pooled:.4f} m",
            xy=(pooled, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -17),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )

    # A row with no interval says so IN THE TICK LABEL. Beside the marker the
    # text lands on the pooled line, and a bare marker on its own looks like a
    # point estimate of perfect precision, which is the opposite of the truth.
    axis.set_yticks(range(len(scenes)))
    axis.set_yticklabels(
        [
            f"{scene}\nn {int(rows.loc[scene, 'n_windows'])}, "
            f"{int(rows.loc[scene, 'n_clusters'])} clusters"
            + ("\nno interval" if scene in missing else "")
            for scene in reversed(scenes)
        ]
    )
    axis.set_ylim(-0.7, len(scenes) - 0.3)
    axis.tick_params(axis="y", length=0)
    axis.set_ylabel("the scene that the fit leaves out")
    axis.set_xlabel(f"location of d = shift(morf) - shift({rival}), metres")
    axis.set_title(f"{name.upper()}: morf against {rival}")
    plotting.tidy(axis, grid_axis="x")


def loso_figure(validate_table: pd.DataFrame, figure_dir: Path | None = None) -> Path:
    """Draw the leave one scene out forest of H1 and H2. Return the png path.

    `validate_table` is the table of docs/FINISH_PLAN.md section 3.7. The
    function reads the rows with check `loso`. It draws the pooled value as
    a line when the table holds a row with check `pooled` for that
    hypothesis. Without such a row it draws no line. The table is
    descriptive, so no p value of this figure enters a family.

    A row names the scene that the fit LEAVES OUT, not the scene that the fit
    keeps. The y axis label says so, because the two readings point opposite
    ways.
    """
    table = pd.DataFrame(validate_table).copy()
    table["check"] = table["check"].astype(str)
    table["hypothesis"] = table["hypothesis"].astype(str)
    loso = table.loc[table["check"] == "loso"]
    pooled_rows = table.loc[table["check"] == "pooled"]

    # The height follows the row count, so one held-out scene does not float
    # in the middle of an empty panel.
    tallest = max(
        [int((loso["hypothesis"] == name).sum()) for name in ("h1", "h2")] + [1]
    )
    plotting.style()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, max(3.8, 0.74 * tallest + 2.6)))
    for axis, name in zip(axes, ("h1", "h2")):
        rows = loso.loc[loso["hypothesis"] == name]
        if rows.empty:
            axis.set_title(f"{name.upper()}, no leave one scene out row")
            axis.set_axis_off()
            continue
        match = pooled_rows.loc[pooled_rows["hypothesis"] == name]
        pooled = float(match["location_m"].iloc[0]) if len(match) else float("nan")
        _forest_panel(axis, rows, pooled, name)

    figure.suptitle(
        "Every scene points the same way, or the claim is one scene alone",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.text(
        0.02,
        0.015,
        "The two panels hold two x ranges. Read each panel against its own "
        "zero line and its own pooled line, not against the other panel.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=plotting.INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    return plotting.save(figure, "hyp_loso", figure_dir)


# ---------------------------------------------------------------------------
# 6. The summary of the three primary rows
# ---------------------------------------------------------------------------

# What each primary row asks. The poster needs the question, not the key.
QUESTION = {
    "h1": "morf against lerf",
    "h2": "morf against nearest",
    "h3": "context on the faithfulness index",
}

# A plain name for each effect key of the verdicts table.
EFFECT_NAME = {
    "share_positive": "share with a positive d, ties left out",
    "partial_r2": "partial R squared of the context block",
}

# The share of the panel that the text column takes, at the right.
TEXT_COLUMN = 0.30


def summary_figure(
    verdicts: pd.DataFrame,
    hypotheses_extra: pd.DataFrame,
    figure_dir: Path | None = None,
) -> Path:
    """Draw the three primary effects in one panel. Return the png path.

    `verdicts` is the contract table of `src/stats/report.py`. The figure
    reads the rows with role `primary`. `hypotheses_extra` supplies the count
    of windows of each row.

    TWO MEASURES, ONE AXIS. H1 and H2 report a share of windows, which runs
    from 0 to 1. H3 reports a partial R squared, which also runs from 0 to 1
    and means something else. The panel therefore names the measure of every
    row beside that row, and it draws the half of the windows line across the
    two share rows alone. A reader who takes the H3 marker for a share reads
    the wrong quantity, so the figure never leaves the measure implicit.

    H1 and H2 carry the pairs cluster bootstrap interval of the verdicts
    table. A partial R squared carries no interval, and the row says so.
    """
    table = pd.DataFrame(verdicts).copy()
    table["hypothesis"] = table["hypothesis"].astype(str)
    if "role" in table.columns:
        table = table.loc[table["role"].astype(str) == "primary"]
    order = [name for name in ("h1", "h2", "h3") if name in set(table["hypothesis"])]
    table = table.set_index("hypothesis").loc[order]

    windows = pd.DataFrame(hypotheses_extra).copy()
    if "hypothesis" in windows.columns and "n_windows" in windows.columns:
        counts = (
            windows.set_index(windows["hypothesis"].astype(str))["n_windows"].to_dict()
        )
    else:
        counts = {}

    plotting.style()
    figure = plt.figure(figsize=(10.0, max(3.2, 0.95 * len(order) + 1.7)))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0 - TEXT_COLUMN, TEXT_COLUMN))
    axis = figure.add_subplot(grid[0, 0])
    # The text column is its own axes with no frame. Inside the plot axes the
    # bottom spine would run on under the text and read as more x range.
    column = figure.add_subplot(grid[0, 1], sharey=axis)
    column.set_axis_off()

    edge = 1.0
    share_heights: list[float] = []
    for position, name in enumerate(order):
        row = table.loc[name]
        height = float(len(order) - 1 - position)
        colour = _SLOTS[position % len(_SLOTS)]
        point = float(row["effect"])
        low = float(row.get("ci_low", float("nan")))
        high = float(row.get("ci_high", float("nan")))
        has_interval = np.isfinite(low) and np.isfinite(high)
        edge = max(edge, *[value for value in (point, high) if np.isfinite(value)])
        if str(row.get("effect_name", "")) == "share_positive":
            share_heights.append(height)

        if has_interval:
            axis.plot(
                [low, high],
                [height, height],
                color=colour,
                linewidth=2.0,
                solid_capstyle="butt",
                zorder=3,
            )
            for bound in (low, high):
                axis.plot(
                    [bound, bound],
                    [height - 0.11, height + 0.11],
                    color=colour,
                    linewidth=2.0,
                    zorder=3,
                )
        axis.plot(
            [point],
            [height],
            linestyle="none",
            marker="o" if has_interval else "D",
            markersize=9 if has_interval else 8,
            color=colour,
            markeredgecolor=plotting.SURFACE,
            markeredgewidth=1.2,
            zorder=4,
        )

    axis.set_xlim(-0.02, max(edge, 1.0) + 0.03)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # The text column states the measure, the number and the adjusted p. It
    # starts at one x for every row, so the three rows read as a column.
    for position, name in enumerate(order):
        row = table.loc[name]
        height = float(len(order) - 1 - position)
        point = float(row["effect"])
        low = float(row.get("ci_low", float("nan")))
        high = float(row.get("ci_high", float("nan")))
        key = str(row.get("effect_name", ""))
        measure = EFFECT_NAME.get(key, key.replace("_", " "))
        if np.isfinite(low) and np.isfinite(high):
            interval = f"{point:.3f}  [{low:.3f}, {high:.3f}]"
        else:
            interval = f"{point:.3f}, no interval"
        column.annotate(
            f"{measure}\n"
            f"{interval}\n"
            f"Holm p = {_p_text(row.get('p_adj'))}, {row.get('verdict', '')}",
            xy=(0.06, height),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="center",
            fontsize=8.5,
            color=plotting.INK_SECONDARY,
        )

    # The half line belongs to the share rows alone. It is not a reference
    # for a partial R squared, so it never crosses that row.
    if share_heights:
        axis.plot(
            [0.5, 0.5],
            [min(share_heights) - 0.42, max(share_heights) + 0.42],
            color=plotting.BASELINE,
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        axis.annotate(
            "half the windows",
            xy=(0.5, max(share_heights) + 0.42),
            xytext=(3, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color=plotting.INK_MUTED,
        )

    labels = []
    for name in order:
        count = counts.get(name)
        known = count is not None and np.isfinite(count)
        tail = f"\n{int(count)} windows" if known else ""
        labels.append(f"{name.upper()}  {QUESTION.get(name, name)}{tail}")
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(list(reversed(labels)))
    axis.set_ylim(-0.62, len(order) - 0.38)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel(
        "effect, 0 to 1. The two measures differ, so read the name of the row."
    )
    plotting.tidy(axis, grid_axis="x")

    figure.suptitle(
        "The attention edges the model weighs most are the edges that move the "
        "forecast",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.text(
        0.02,
        0.015,
        "The bar of H1 and H2 is a 95 percent pairs cluster bootstrap interval "
        "over the components. The partial R squared of H3 has no such "
        "interval. Holm adjusts the three p values as one family.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=plotting.INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    return plotting.save(figure, "hyp_summary", figure_dir)
