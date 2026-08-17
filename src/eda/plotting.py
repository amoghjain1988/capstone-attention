"""The one place that knows a colour and a figure style.

This module is internal to src/eda. No module outside src/eda imports it.

The five scene colours are the first five slots of a validated categorical
palette. The order is fixed. A scene keeps its colour in every figure, so the
reader learns the colour once. Three of the five slots fall below a contrast
ratio of 3 to 1 on the light surface, so every figure also carries a direct
label or a legend, and every function returns the table behind the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

# Categorical slots, in fixed order. Never cycle. Never reorder.
SCENE_COLOUR: dict[str, str] = {
    "eth": "#2a78d6",  # blue
    "hotel": "#eb6834",  # orange
    "univ": "#1baf7a",  # aqua
    "zara1": "#eda100",  # yellow
    "zara2": "#e87ba4",  # magenta
}
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT_STACK = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def style() -> None:
    """Apply the figure style. Call once per figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
        }
    )


def tidy(axis, grid_axis: str = "y") -> None:
    """Remove the top and the right spine. Add a recessive grid."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis in ("x", "y", "both"):
        axis.grid(True, axis=grid_axis, zorder=0)
        axis.set_axisbelow(True)


def save(figure, name: str, figure_dir: Path | None = None) -> Path:
    """Write one png and close the figure. Return the path."""
    target_dir = Path(figure_dir) if figure_dir else config.FIGURE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.png"
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    return path


def scene_colours(scenes) -> list[str]:
    """Return the colour of every scene, in the order given."""
    return [SCENE_COLOUR.get(str(scene), INK_MUTED) for scene in scenes]


def label_bars(axis, bars, values, fmt: str = "{:,.0f}", pad: float = 0.01) -> None:
    """Write the value above every bar. This is the relief for low contrast."""
    top = max(values) if len(values) else 0.0
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + pad * top,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )
