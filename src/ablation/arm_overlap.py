"""Where two arms actually disagree.

See CONTRACT.md section 5.7 for the signatures.

H2 is only meaningful on the part of the data where MoRF (attention) and
Nearest (proximity) pick a different edge. If they agree almost everywhere,
hypothesis 2 is underpowered before the first forward pass runs, and Prem
must know that before he chooses the test.
"""

from __future__ import annotations

import pandas as pd

import config
from src.data import schema


def jaccard(order_a: list[int], order_b: list[int], k: int) -> float:
    """Return the Jaccard overlap of the top k of two removal orders.

    |A intersect B| / |A union B|, computed over the first k entries of each
    list. Returns 1.0 when both top-k sets are empty. k larger than either
    list truncates to the list length.
    """
    top_a = set(order_a[:k])
    top_b = set(order_b[:k])
    if not top_a and not top_b:
        return 1.0

    union = top_a | top_b
    intersection = top_a & top_b
    return float(len(intersection) / len(union))


def _order(block: pd.DataFrame, rank_column: str) -> list[int]:
    """Return the src ids of one window's edges, sorted by the given rank.

    Raise KeyError when rank_column is absent. Raise ValueError when it
    holds a null. sort_values places a null last and is a stable sort, so a
    partial rank_column would silently fall back to the original row order
    for the null rows, and the window would then score a Jaccard overlap of
    1.0 or 0.0 on that row order alone, not on the real ranking.
    src/ablation/arms.py applies the same guard to rank_dist inside
    order_edges; this function copies its shape.
    """
    if rank_column not in block.columns:
        raise KeyError(
            f"the arm_overlap comparison needs a {rank_column!r} column. "
            "See CONTRACT.md section 3, attention_edges."
        )
    if block[rank_column].isna().any():
        raise ValueError(
            f"the arm_overlap comparison reads a null {rank_column!r}. "
            "This window gives no ranking."
        )
    ordered = block.sort_values(rank_column, kind="mergesort")
    return [int(src) for src in ordered["src"]]


def attention_vs_distance(
    edges: pd.DataFrame,
    k: int = config.PRIMARY_N_REMOVED,
    module: str = config.PRIMARY_MODULE,
    time_agg: str = config.PRIMARY_TIME_AGG,
) -> pd.DataFrame:
    """Return the per-window Jaccard overlap of the attention ranking against
    the distance ranking, at top k, on one module and time collapse.

    Columns: window_id, scene, jaccard_k.
    """
    primary = edges.loc[(edges["module"] == module) & (edges["time_agg"] == time_agg)]

    rows: list[dict] = []
    for window_id, block in primary.groupby("window_id", observed=True):
        scene, _ = schema.split_window_id(str(window_id))
        order_attn = _order(block, "rank_attn")
        order_dist = _order(block, "rank_dist")
        rows.append(
            {
                "window_id": str(window_id),
                "scene": scene,
                "jaccard_k": jaccard(order_attn, order_dist, k),
            }
        )
    return pd.DataFrame(rows, columns=["window_id", "scene", "jaccard_k"])


def summary_by_scene(overlap: pd.DataFrame) -> pd.DataFrame:
    """Return the mean, median and window count of jaccard_k, one row per scene.

    Written to outputs/tables/arm_overlap.csv. A high mean means attention
    and the nearest neighbour usually agree, and hypothesis 2 is
    underpowered by construction.
    """
    grouped = overlap.groupby("scene", observed=True)["jaccard_k"]
    table = grouped.agg(mean="mean", median="median", n_windows="count").reset_index()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLE_DIR / "arm_overlap.csv", index=False)
    return table
