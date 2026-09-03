"""The writer of the attention_edges table.

See CONTRACT.md section 3 for the table and docs/FINISH_PLAN.md section 3.9
for the signatures.

scripts/extract_attention.py caches one row per edge into the ego, for every
module and every time collapse. The cache holds window_id, src, dst, module,
time_agg, attn, row_entropy and rank_attn. It does not hold the pair features.
This module joins the cache to src/features/proximity.py and
src/features/kinematics.py and returns the full table.

The join is the place where a silent hole becomes a wrong answer. A distance
that is absent turns into a null rank_dist, and the nearest arm of
src/ablation/arms.py then refuses the whole window. This module therefore
raises as soon as a join leaves a null, rather than pass the hole downstream.
"""

from __future__ import annotations

import pandas as pd

import config
from src.data import schema
from src.features import kinematics, proximity

# The columns that scripts/extract_attention.py writes into the cache.
CACHE_COLUMNS = (
    "window_id",
    "src",
    "dst",
    "module",
    "time_agg",
    "attn",
    "row_entropy",
    "rank_attn",
)

# The columns that the two feature modules add.
FEATURE_COLUMNS = ("dist_at_last_frame", "rank_dist", "closing_speed", "inv_ttc")


def _empty_table() -> pd.DataFrame:
    """Return an empty attention_edges frame with the schema dtypes."""
    return pd.DataFrame(
        {
            name: pd.Series(dtype="object" if dtype == "object" else dtype)
            for name, dtype in schema.ATTENTION_EDGES.items()
        }
    )


def _as_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose three join columns carry one common dtype.

    A merge between an int32 column and an int64 column matches on value, but
    the dtypes of the two frames differ between the cache and the feature
    tables. Cast both sides first, so the merge never falls back to object.
    """
    out = frame.copy()
    out["window_id"] = out["window_id"].astype(str)
    out["src"] = out["src"].astype("int64")
    out["dst"] = out["dst"].astype("int64")
    return out


def build_table(
    cache_edges: pd.DataFrame,
    windows: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Return the attention_edges table.

    `cache_edges` is the concatenation of the per-scene cache files that
    scripts/extract_attention.py writes. `windows` is the windows table.
    `trajectories` is the trajectories table.

    The function keeps only the eligible windows of `windows`, joins the cache
    rows to proximity.pair_distance_all and kinematics.pair_kinematics_all on
    (window_id, src, dst), and returns the columns of schema.ATTENTION_EDGES.
    It raises when a join leaves a null.

    The cache on disk covers every window at stride 1. The frozen windows
    table is a subset of those windows with the same window ids and the same
    ego, so the filter above is also the step that cuts the cache down to the
    frozen design.
    """
    missing = [name for name in CACHE_COLUMNS if name not in cache_edges.columns]
    if missing:
        raise KeyError(
            f"the cache lacks these columns: {missing}. "
            "scripts/extract_attention.py writes them."
        )

    eligible = windows.loc[windows["eligible"]].copy()
    if eligible.empty:
        return _empty_table()

    keep = set(eligible["window_id"].astype(str))
    cache = cache_edges.loc[cache_edges["window_id"].astype(str).isin(keep)]
    cache = _as_join_keys(cache.loc[:, list(CACHE_COLUMNS)])
    if cache.empty:
        return _empty_table()

    distance = _as_join_keys(proximity.pair_distance_all(eligible, trajectories))
    kinematic = _as_join_keys(kinematics.pair_kinematics_all(eligible, trajectories))

    merged = cache.merge(
        distance[["window_id", "src", "dst", "dist_at_last_frame", "rank_dist"]],
        on=["window_id", "src", "dst"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        kinematic[["window_id", "src", "dst", "closing_speed", "inv_ttc"]],
        on=["window_id", "src", "dst"],
        how="left",
        validate="many_to_one",
    )

    null_count = merged.isna().sum()
    holes = {name: int(count) for name, count in null_count.items() if count}
    if holes:
        example = merged.loc[merged[list(holes)].isna().any(axis=1)].head(1)
        raise ValueError(
            f"the join left a null in {holes}. The cache holds an edge that the "
            f"feature tables do not. First bad row: "
            f"{example.to_dict('records')}"
        )

    out = merged.loc[:, list(schema.ATTENTION_EDGES)]
    out = schema.cast(out, "attention_edges")
    schema.validate(out, "attention_edges")
    return out.reset_index(drop=True)


def primary(
    edges: pd.DataFrame,
    module: str = config.PRIMARY_MODULE,
    time_agg: str = config.PRIMARY_TIME_AGG,
) -> pd.DataFrame:
    """Return the slice of one module and one time collapse.

    Every arm of src/ablation/ reads this slice. The sensitivity checks of
    src/hypotheses/validate.py ask for another module or another time
    collapse, so the argument stays open.
    """
    if module not in config.MODULES:
        raise ValueError(f"module must be one of {config.MODULES}, it is {module!r}")
    if time_agg not in config.TIME_AGGS:
        raise ValueError(
            f"time_agg must be one of {config.TIME_AGGS}, it is {time_agg!r}"
        )

    block = edges.loc[
        (edges["module"].astype(str) == str(module))
        & (edges["time_agg"].astype(str) == str(time_agg))
    ]
    return block.reset_index(drop=True)


def edges_of_window(edges_primary: pd.DataFrame, window_id: str) -> pd.DataFrame:
    """Return the rows of one window, strongest attention first.

    src/ablation/arms.py::order_edges reads window_id for the random arm and
    rank_dist for the nearest arm, so both columns stay in the result.
    """
    block = edges_primary.loc[
        edges_primary["window_id"].astype(str) == str(window_id)
    ]
    return block.sort_values("rank_attn", kind="mergesort").reset_index(drop=True)


def edges_by_window(edges_primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return window_id -> that window's edge rows.

    src/ablation/driver.py walks about 480 windows and asks for the edges of
    each one. One group pass costs far less than 480 filter passes over the
    whole table, so the driver builds this map once.
    """
    out: dict[str, pd.DataFrame] = {}
    if len(edges_primary) == 0:
        return out
    keys = edges_primary["window_id"].astype(str)
    for window_id, block in edges_primary.groupby(keys, sort=False):
        out[str(window_id)] = block.sort_values(
            "rank_attn", kind="mergesort"
        ).reset_index(drop=True)
    return out
