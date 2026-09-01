"""The six removal strategies, and the mass-matched control.

See CONTRACT.md section 5.7 for the signatures.

`edges` is the per-window edge table with `src`, `attn`, and `rank_dist`
columns. `order_edges` returns `src` ids in removal order. Ties always break
by the LOWER `src` id, so every call with the same input gives the same
order.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

ARMS = ("morf", "lerf", "weight_matched", "nearest", "random", "single")


def order_edges(edges: pd.DataFrame, arm: str, seed: int) -> list[int]:
    """Return src ids, in the order this arm removes them.

    `morf` removes the strongest attention first. `lerf` removes the weakest
    attention first. `nearest` removes the closest neighbour first, by
    `rank_dist`. `random` shuffles with `np.random.default_rng(seed)`, so the
    same seed always gives the same order.

    `single` raises. It is not an order: the caller sweeps each edge alone.
    `weight_matched` raises here too. It has no fixed order. Call
    `weight_matched_set` with a target mass instead.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, it is {arm!r}")
    if arm == "single":
        raise ValueError("single is not an order. The caller sweeps each edge alone.")
    if arm == "weight_matched":
        raise ValueError(
            "weight_matched has no fixed order. Call weight_matched_set with a "
            "target_mass instead."
        )

    # Only the nearest arm reads rank_dist. Ask for it inside that branch, so
    # the other four arms run on the attention table that extract_attention.py
    # writes today. See CONTRACT.md section 3, attention_edges.
    if arm == "nearest":
        if "rank_dist" not in edges.columns:
            raise KeyError(
                "the nearest arm needs a rank_dist column. "
                "src/features/proximity.py writes it. See CONTRACT.md section 3."
            )
        near = edges[["src", "rank_dist"]].copy()
        near["src"] = near["src"].astype(int)
        if near["rank_dist"].isna().any():
            # A stable sort puts NaN last and leaves the rest in row order, so a
            # partial rank_dist silently turns the distance order into the row
            # order. Refuse instead, and let the caller log the window.
            raise ValueError(
                "the nearest arm reads a null rank_dist. This window gives no "
                "distance order."
            )
        ordered = near.sort_values(["rank_dist", "src"], ascending=[True, True], kind="stable")
        return [int(src) for src in ordered["src"]]

    table = edges[["src", "attn"]].copy()
    table["src"] = table["src"].astype(int)

    if arm == "morf":
        ordered = table.sort_values(["attn", "src"], ascending=[False, True], kind="stable")
        return [int(src) for src in ordered["src"]]
    if arm == "lerf":
        ordered = table.sort_values(["attn", "src"], ascending=[True, True], kind="stable")
        return [int(src) for src in ordered["src"]]

    # arm == "random"
    rng = np.random.default_rng(seed)
    base = table.sort_values("src", kind="stable")["src"].to_numpy()
    shuffled = rng.permutation(base)
    return [int(src) for src in shuffled]


def weight_matched_set(edges: pd.DataFrame, target_mass: float) -> list[int]:
    """Return the smallest set of the LOWEST-attention src ids whose summed
    `attn` reaches `target_mass`.

    This matches MASS, not count. Many light edges can carry the same total
    attention as a few heavy ones, and that is what kills the confound where
    MoRF simply removes more attention than LeRF.

    When the table's total attention is below `target_mass`, the target
    cannot be reached. Return every src id and warn with the reason, so the
    caller can log the failure. See docs/EDA_FINDINGS.md fact F10: this fires
    often, for roughly a quarter of windows.
    """
    if target_mass < 0:
        raise ValueError(f"target_mass must not be negative, it is {target_mass}")

    table = edges[["src", "attn"]].copy()
    table["src"] = table["src"].astype(int)
    table = table.sort_values(["attn", "src"], ascending=[True, True], kind="stable")

    if target_mass == 0:
        return []

    total_mass = float(table["attn"].sum())
    if total_mass < target_mass:
        warnings.warn(
            f"weight_matched_set: target_mass={target_mass} exceeds the total "
            f"available attention mass={total_mass}. Returning every edge.",
            stacklevel=2,
        )
        return [int(src) for src in table["src"]]

    chosen: list[int] = []
    cumulative = 0.0
    for src, attn in zip(table["src"], table["attn"]):
        chosen.append(int(src))
        cumulative += float(attn)
        if cumulative >= target_mass:
            break
    return chosen
