"""The transform decision.

See CONTRACT.md section 5.4 for the signatures.

The decision is already made and this module only shows the evidence.

H1 and H2 use rank tests. A rank test is invariant under any strictly rising
transform, so a transform changes no p value and no rank-biserial effect.
Therefore H1 and H2 use the RAW shift, in metres. The effect then stays in the
unit that a reader understands.

H3 fits a regression, so H3 z-scores the predictors. A z-score puts every
coefficient on one scale and makes the collinearity check readable.

A log scale belongs on a plot axis only. Never fit a model to a logged shift
and then report the answer in metres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eda import plotting

LOG_OFFSET = 1e-6


def _shift_for_log(raw: np.ndarray) -> np.ndarray:
    """Return values that are safe to log.

    A strictly positive column needs no shift. A column that touches 0 gets a
    shift of 1 percent of the interquartile range. A fixed tiny shift such as
    1e-6 would push the smallest value to about minus 14 and would invent a
    long left tail that the data does not have.
    """
    low = float(raw.min())
    if low > 0.0:
        return raw
    spread = float(np.subtract(*np.quantile(raw, [0.75, 0.25])))
    offset = max(0.01 * spread, LOG_OFFSET)
    return raw - low + offset


def compare_scales(
    values: np.ndarray, name: str = "value", figure_dir: Path | None = None
) -> pd.DataFrame:
    """Return the raw scale, the z scale and the log scale side by side.

    Columns: scale, n, mean, median, sd, skew, excess_kurtosis, min, max.
    The rank of every row is the same on all three scales, so the last column
    spearman_vs_raw is always 1.0. That column is the proof, not a result.
    """
    raw = np.asarray(values, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("compare_scales needs at least one finite value")

    z = (raw - raw.mean()) / raw.std(ddof=1)
    log = np.log(_shift_for_log(raw))

    scales = {"raw": raw, "z": z, "log": log}
    rows = []
    for scale, series in scales.items():
        rows.append(
            {
                "scale": scale,
                "n": int(series.size),
                "mean": float(series.mean()),
                "median": float(np.median(series)),
                "sd": float(series.std(ddof=1)),
                "skew": float(stats.skew(series)),
                "excess_kurtosis": float(stats.kurtosis(series)),
                "min": float(series.min()),
                "max": float(series.max()),
                "spearman_vs_raw": float(stats.spearmanr(raw, series).statistic),
            }
        )

    table = pd.DataFrame(rows)
    table.insert(0, "quantity", name)
    _figure(scales, name, figure_dir)
    return table


def _figure(scales: dict[str, np.ndarray], name: str, figure_dir) -> Path:
    """Draw the same values on the three scales."""
    import matplotlib.pyplot as plt

    plotting.style()
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    titles = {
        "raw": "raw, the unit a reader knows",
        "z": "z, for the H3 regression only",
        "log": "log, for a plot axis only",
    }
    for axis, (scale, series) in zip(axes, scales.items()):
        axis.hist(series, bins=50, color=plotting.SERIES_1, zorder=3)
        axis.set_title(titles[scale])
        axis.set_xlabel(name if scale == "raw" else scale)
        plotting.tidy(axis)
    axes[0].set_ylabel("rows")

    figure.suptitle(
        "A rank test is invariant under all three. H1 and H2 keep the raw metres.",
        x=0.02,
        ha="left",
        fontsize=11,
        weight="semibold",
        color=plotting.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return plotting.save(figure, f"eda_scaling_{name}", figure_dir)
