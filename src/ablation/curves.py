"""One row per number of edges removed, for one window and one arm.

See CONTRACT.md section 3 (perturbation_curves) and section 5.7 for the
signature.

`edges` carries agent ids in `src`, not window-axis positions. This module
looks the ids up in `window_row["agent_order"]` before it calls
`src.ablation.mask.build_mask`, which wants window-axis indices only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.ablation.arms import ARMS, order_edges, weight_matched_set
from src.ablation.mask import build_mask, masked_predict
from src.ablation.shift import ego_path, shift
from src.data import schema


def curve(
    window_row: pd.Series,
    model,
    edges: pd.DataFrame,
    arm: str,
    seed: int,
    mask_policy: str = config.PRIMARY_MASK_POLICY,
) -> pd.DataFrame:
    """Return one row per n_removed, for one window and one arm.

    `window_row` is one row of the `windows` table. `edges` holds only the
    edges that point INTO this window's ego, with columns `src`, `attn`, and
    `rank_dist`. `src` is an agent id, matching CONTRACT.md's
    `attention_edges` table.

    One fixed `draws` array is built here from `seed` and reused for every
    mask in this call, including the unmasked baseline, so every shift in
    the curve is comparable. Columns match CONTRACT.md section 3 exactly:
    window_id, ego_id, arm, n_removed, removed_mass, edge_src, draw_id,
    mask_policy, run_id, shift.

    NOTE on two judgement calls this function makes, because CONTRACT.md and
    docs/GRACE_AGENT.md do not pin either one down further. Confirm both with
    the team before treating the results as final.

    `single` sweeps every edge alone. `n_removed` here counts the edge's
    position in the sweep (1, 2, 3, ...), not a literal removed-edge count,
    because `single` always removes exactly one edge and `n_removed` must
    stay unique inside the perturbation_curves primary key
    (window_id, arm, n_removed, draw_id, mask_policy). `edge_src` is the
    column that names which edge a `single` row actually removed.

    `weight_matched` cannot follow a fixed step count, because it matches
    MASS, not `n`. This function targets the SAME cumulative attention that
    `morf` has removed after each of its steps, then reports the ACTUAL edge
    count and mass that `weight_matched_set` needed to reach that target. A
    step whose chosen set repeats the step before it is skipped, so
    `n_removed` stays unique.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, it is {arm!r}")

    hist = schema.unpack_hist(window_row, config.N_HIST)
    ego_pos = schema.ego_index(window_row)
    n_agents = int(hist.shape[0])
    agent_order = [int(a) for a in window_row["agent_order"]]
    index_of = {agent_id: position for position, agent_id in enumerate(agent_order)}

    window_id = window_row["window_id"]
    ego_id = int(window_row["ego_id"])
    run_id = config.config_hash()

    draws = np.arange(config.N_SAMPLES, dtype=np.int64) + int(seed)
    base_pred = model.predict(hist, edge_mask=None, draws=draws)
    base_ego = ego_path(base_pred, ego_pos)

    table = edges[["src", "attn"]].copy()
    table["src"] = table["src"].astype(int)
    attn_by_src = dict(zip(table["src"], table["attn"].astype(float)))

    rows: list[dict] = []

    def add_row(
        n_removed: int, removed_mass: float, edge_src: int, removed_ids: list[int]
    ) -> None:
        positions = [index_of[agent_id] for agent_id in removed_ids]
        mask = build_mask(n_agents, positions, ego_pos)
        masked_pred = masked_predict(model, hist, mask, draws, mask_policy)
        masked_ego = ego_path(masked_pred, ego_pos)
        rows.append(
            {
                "window_id": window_id,
                "ego_id": ego_id,
                "arm": arm,
                "n_removed": int(n_removed),
                "removed_mass": float(removed_mass),
                "edge_src": int(edge_src),
                "draw_id": int(seed),
                "mask_policy": mask_policy,
                "run_id": run_id,
                "shift": shift(masked_ego, base_ego),
            }
        )

    if arm == "single":
        ordered_ids = sorted(attn_by_src)  # deterministic: lowest src id first
        for position, src in enumerate(ordered_ids, start=1):
            add_row(position, attn_by_src[src], src, [src])
        return pd.DataFrame(rows)

    if arm == "weight_matched":
        morf_order = order_edges(edges, "morf", seed)
        cumulative = 0.0
        previous: list[int] | None = None
        for src in morf_order:
            cumulative += attn_by_src[src]
            chosen = weight_matched_set(edges, cumulative)
            if chosen == previous:
                continue
            previous = chosen
            removed_mass = sum(attn_by_src[s] for s in chosen)
            add_row(len(chosen), removed_mass, -1, chosen)
        return pd.DataFrame(rows)

    order = order_edges(edges, arm, seed)
    cumulative = 0.0
    for step, src in enumerate(order, start=1):
        cumulative += attn_by_src[src]
        add_row(step, cumulative, -1, order[:step])
    return pd.DataFrame(rows)
