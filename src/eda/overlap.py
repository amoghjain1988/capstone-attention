"""The design effect and the effective sample size.

See CONTRACT.md section 5.4 for the signatures.

The proposal counts every window as one independent observation. It is not one.
Two windows that start one frame apart share 19 of their 20 frames. Many
windows also share the same ego pedestrian. Both facts inflate the sample and
both shrink the true information.

The design effect measures the loss.

    deff        = 1 + (mean_cluster_size - 1) * icc
    n_effective = n_rows / deff

src/hypotheses/design.py cannot compute power without these two numbers.

The intraclass correlation needs an outcome. The real outcome is the shift, and
the shift does not exist until the ablation runs. This module therefore reports
the intraclass correlation of every proxy that exists before the ablation, and
it takes the largest one. The largest is the safe one, because a larger
intraclass correlation gives a larger design effect and a smaller effective
sample. Repeat this calculation on the real shift after the pilot ablation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.eda import plotting

PROXIES = ("n_agents", "ego_density", "ego_nearest_dist", "ego_speed")
STRIDES = (1, 2, 4, 5, 10, 20)


def icc_one_way(values: np.ndarray, clusters: np.ndarray) -> dict:
    """Return the one-way random effects intraclass correlation.

    The design is unbalanced, so the cluster size uses the adjusted mean m0.

        m0  = (n - sum(n_i squared) / n) / (k - 1)
        icc = (msb - msw) / (msb + (m0 - 1) * msw)

    A negative estimate is reported as 0.0. A negative estimate means that the
    between-cluster variance is smaller than the noise, so the clusters carry
    no shared signal.
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]

    n = values.size
    if n < 3:
        return {"icc": np.nan, "k": 0, "n": int(n), "m0": np.nan}

    labels, codes = np.unique(clusters, return_inverse=True)
    k = labels.size
    if k < 2:
        return {"icc": np.nan, "k": int(k), "n": int(n), "m0": np.nan}
    if k == n:
        # Every cluster holds one row. No row shares a cluster with any other
        # row, so the design carries no clustering and the design effect is 1.
        return {
            "icc": 0.0,
            "icc_raw": 0.0,
            "k": int(k),
            "n": int(n),
            "m0": 1.0,
            "mean_cluster_size": 1.0,
        }

    if float(values.std()) == 0.0:
        # The column never varies, so there is no variance to split between
        # the clusters and inside them. The correlation does not exist.
        return {
            "icc": np.nan,
            "icc_raw": np.nan,
            "k": int(k),
            "n": int(n),
            "m0": np.nan,
            "mean_cluster_size": float(n / k),
            "reason": "the column is constant",
        }

    sizes = np.bincount(codes)
    sums = np.bincount(codes, weights=values)
    means = sums / sizes
    grand = values.mean()

    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((values - means[codes]) ** 2).sum())

    df_between = k - 1
    df_within = n - k
    if df_within <= 0:
        return {"icc": np.nan, "k": int(k), "n": int(n), "m0": np.nan}

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    m0 = (n - (sizes.astype(np.float64) ** 2).sum() / n) / df_between

    denominator = ms_between + (m0 - 1.0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else np.nan
    return {
        "icc": float(max(icc, 0.0)) if np.isfinite(icc) else np.nan,
        "icc_raw": float(icc) if np.isfinite(icc) else np.nan,
        "k": int(k),
        "n": int(n),
        "m0": float(m0),
        "mean_cluster_size": float(n / k),
    }


def _cluster_key(frame: pd.DataFrame, unit: str) -> pd.Series:
    """Return the cluster label of every window."""
    if unit == "pedestrian":
        return frame["scene"].astype(str) + "_" + frame["ego_id"].astype(str)
    if unit == "time_block":
        block = frame["t0"] // config.WINDOW_LEN
        return frame["scene"].astype(str) + "_t" + block.astype(str)
    if unit == "scene":
        return frame["scene"].astype(str)
    raise ValueError(f"{unit!r} is not a known cluster unit")


def design_effect(
    windows: pd.DataFrame,
    features: pd.DataFrame | None = None,
    unit: str = "pedestrian",
) -> dict:
    """Return {icc, mean_cluster_size, deff, n_effective}.

    Pass `features` to add the ego proxies. Build it with
    src/eda/density.py ego_neighbourhood(). Without it the function uses
    n_agents alone.
    """
    good = windows.loc[windows["eligible"]].copy()
    if features is not None:
        extra = [c for c in PROXIES if c in features.columns and c not in good.columns]
        good = good.merge(
            features[["window_id", *extra]], on="window_id", how="left"
        )

    clusters = _cluster_key(good, unit).to_numpy()
    available = [name for name in PROXIES if name in good.columns]

    per_proxy = {}
    for name in available:
        per_proxy[name] = icc_one_way(good[name].to_numpy(dtype=np.float64), clusters)

    usable = {k: v for k, v in per_proxy.items() if np.isfinite(v.get("icc", np.nan))}
    if not usable:
        # No proxy varies, so no correlation can be estimated. Report a design
        # effect of 1 and say plainly that the number is a fallback, not a
        # measurement. Never read this branch as evidence of independence.
        size = len(good) / max(len(set(clusters)), 1)
        return {
            "unit": unit,
            "proxy": None,
            "icc": float("nan"),
            "mean_cluster_size": float(size),
            "n_clusters": int(len(set(clusters))),
            "deff": 1.0,
            "n_rows": int(len(good)),
            "n_effective": float(len(good)),
            "per_proxy": {},
            "warning": "every proxy is constant, so the design effect is a fallback of 1",
        }

    worst = max(usable, key=lambda name: usable[name]["icc"])
    icc = usable[worst]["icc"]
    mean_cluster_size = usable[worst]["mean_cluster_size"]
    deff = 1.0 + (mean_cluster_size - 1.0) * icc
    n_rows = len(good)

    return {
        "unit": unit,
        "proxy": worst,
        "icc": float(icc),
        "mean_cluster_size": float(mean_cluster_size),
        "n_clusters": int(usable[worst]["k"]),
        "deff": float(deff),
        "n_rows": int(n_rows),
        "n_effective": float(n_rows / deff),
        "per_proxy": {k: round(float(v["icc"]), 4) for k, v in usable.items()},
    }


def overlap_report(
    windows: pd.DataFrame,
    features: pd.DataFrame | None = None,
    figure_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the design effect at every stride, for both cluster units.

    The stride sweep subsamples the stride 1 table. Keeping every s-th window
    start gives exactly the table that a build at stride s would give.
    """
    rows = []
    good = windows.loc[windows["eligible"]].copy()

    for stride in STRIDES:
        keep = pd.Series(False, index=good.index)
        for scene, part in good.groupby("scene", observed=True):
            base = int(part["t0"].min())
            keep.loc[part.index] = ((part["t0"] - base) % stride) == 0
        block = good.loc[keep]
        if len(block) < 10:
            continue

        for unit in ("pedestrian", "time_block"):
            result = design_effect(block, features=features, unit=unit)
            size = result["mean_cluster_size"]
            per_proxy = result["per_proxy"]

            # The worst proxy gives the safe answer. The best proxy gives the
            # hopeful answer. Report both, so nobody reads one number alone.
            best_icc = min(per_proxy.values()) if per_proxy else float("nan")
            best_deff = 1.0 + (size - 1.0) * best_icc
            row = {
                "stride": stride,
                "unit": unit,
                "windows": result["n_rows"],
                "n_clusters": result["n_clusters"],
                "mean_cluster_size": size,
                "worst_proxy": result["proxy"],
                "icc_worst": result["icc"],
                "deff_worst": result["deff"],
                "n_effective_worst": result["n_effective"],
                "icc_best": best_icc,
                "deff_best": best_deff,
                "n_effective_best": result["n_rows"] / best_deff,
                "mean_overlap_fraction": float(block["overlap_fraction"].mean()),
                "forward_passes": int(block["n_agents"].sum()),
            }
            for name, value in per_proxy.items():
                row[f"icc_{name}"] = value
            rows.append(row)

    table = pd.DataFrame(rows)
    _figure(table, figure_dir)
    return table


def _figure(table: pd.DataFrame, figure_dir) -> Path:
    """Draw the effective sample size and the cost against the stride."""
    import matplotlib.pyplot as plt

    plotting.style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colour = {"pedestrian": plotting.SERIES_1, "time_block": plotting.SERIES_2}

    axis = axes[0]
    for unit, part in table.groupby("unit", observed=True):
        part = part.sort_values("stride")
        axis.plot(
            part["stride"],
            part["windows"],
            color=plotting.INK_MUTED,
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        axis.fill_between(
            part["stride"],
            part["n_effective_worst"],
            part["n_effective_best"],
            color=colour[str(unit)],
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            part["stride"],
            part["n_effective_worst"],
            color=colour[str(unit)],
            marker="o",
            markersize=5,
            zorder=3,
            label=f"effective n, cluster = {unit}",
        )
    axis.annotate(
        "raw window count",
        xy=(float(table["stride"].min()), float(table["windows"].max())),
        xytext=(8, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=plotting.INK_MUTED,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(list(table["stride"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("window stride, in frames")
    axis.set_ylabel("windows")
    axis.set_title("A wider stride costs raw rows but keeps the information")
    axis.legend(loc="lower left")
    plotting.tidy(axis, grid_axis="both")

    axis = axes[1]
    for unit, part in table.groupby("unit", observed=True):
        part = part.sort_values("stride")
        axis.plot(
            part["forward_passes"],
            part["n_effective_worst"],
            color=colour[str(unit)],
            marker="o",
            markersize=5,
            zorder=3,
            label=f"cluster = {unit}",
        )
        for _, row in part.iterrows():
            axis.annotate(
                f"s={int(row['stride'])}",
                xy=(row["forward_passes"], row["n_effective_worst"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=plotting.INK_MUTED,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("forward passes needed")
    axis.set_ylabel("effective n")
    axis.set_title("What one forward pass buys")
    axis.legend(loc="lower right")
    plotting.tidy(axis, grid_axis="both")

    figure.tight_layout()
    return plotting.save(figure, "eda_overlap", figure_dir)
