"""Order the edges that point into the ego.

See CONTRACT.md section 5.6 for the signatures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_edges(
    attn: np.ndarray, ego_index: int, agent_order: list[int]
) -> pd.DataFrame:
    """Return only the edges whose dst is the ego.

    Columns: src, dst, attn, rank_attn. Rank 1 is the strongest.
    Ties break by the lower src id, so the ranking is deterministic.

    Entry attn[i, j] is how much agent i attends to agent j, so the row of the
    ego holds every edge that points into the ego.
    """
    matrix = np.asarray(attn, dtype=np.float64)
    n_agents = matrix.shape[0]
    if matrix.shape != (n_agents, n_agents):
        raise ValueError(f"attn must be square, it is {matrix.shape}")
    if len(agent_order) != n_agents:
        raise ValueError(
            f"agent_order has {len(agent_order)} ids for {n_agents} agents"
        )
    if not 0 <= int(ego_index) < n_agents:
        raise ValueError(f"ego_index {ego_index} is outside 0 to {n_agents - 1}")

    ego_index = int(ego_index)
    ids = [int(a) for a in agent_order]
    sources = [i for i in range(n_agents) if i != ego_index]

    frame = pd.DataFrame(
        {
            "src": [ids[i] for i in sources],
            "dst": ids[ego_index],
            "src_index": sources,
            "attn": [float(matrix[ego_index, i]) for i in sources],
        }
    )
    # Sort by falling attention, then by rising src id. The second key makes a
    # tie deterministic.
    frame = frame.sort_values(["attn", "src"], ascending=[False, True], kind="mergesort")
    frame["rank_attn"] = np.arange(1, len(frame) + 1, dtype=np.int16)
    return frame.reset_index(drop=True)
