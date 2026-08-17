"""Collapse a raw attention map onto the agent by agent graph.

See CONTRACT.md section 5.6 for the signatures.

The input is agent-major on both axes, that is index = agent_index * T +
time_index. src/models/agentformer.py already transposes the native time-major
map, so every caller here sees the order that CONTRACT.md section 4.1 promises.

The order of the collapse is TIME first, then heads, then layers. AgentFormer
averages the heads inside agent_aware_attention, and src/models/agentformer.py
averages the layers, so only the time step is left for this module.
"""

from __future__ import annotations

import numpy as np

TIME_AGGS = ("mean", "max", "last")


def collapse(raw: np.ndarray, n_agents: int, time_agg: str = "mean") -> np.ndarray:
    """Turn a token map into an (N, N) agent map.

    The two axes may carry different time lengths. The encoder map is
    (N*8, N*8). The cross map is (N*12, N*8). Entry (i, j) of the result is how
    much agent i attends to agent j.

    The source axis is summed, never averaged, because the row of a softmax
    sums to 1 and the mass that one agent sends to another is the sum over its
    tokens. The target axis is aggregated by time_agg.
    """
    if time_agg not in TIME_AGGS:
        raise ValueError(f"time_agg must be one of {TIME_AGGS}, it is {time_agg!r}")

    array = np.asarray(raw, dtype=np.float64)
    rows, cols = array.shape
    if rows % n_agents or cols % n_agents:
        raise ValueError(f"a {array.shape} map does not divide by {n_agents} agents")

    t_dst, t_src = rows // n_agents, cols // n_agents
    blocks = array.reshape(n_agents, t_dst, n_agents, t_src)

    # Sum the mass that the source agent receives across its own tokens.
    per_dst_step = blocks.sum(axis=3)  # (N, t_dst, N)

    if time_agg == "mean":
        out = per_dst_step.mean(axis=1)
    elif time_agg == "max":
        out = per_dst_step.max(axis=1)
    else:
        out = per_dst_step[:, -1, :]
    return out


def row_entropy(raw: np.ndarray) -> np.ndarray:
    """Return the Shannon entropy of every source row, natural log.

    The input is the token map before any collapse. A row that spreads its
    attention evenly gives a large entropy. A row that points at one token
    gives an entropy near 0.
    """
    array = np.asarray(raw, dtype=np.float64)
    safe = np.where(array > 0.0, array, 1.0)
    return -(array * np.log(safe)).sum(axis=1)


def agent_row_entropy(raw: np.ndarray, n_agents: int) -> np.ndarray:
    """Return one entropy per agent, averaged over that agent's own tokens."""
    per_token = row_entropy(raw)
    return per_token.reshape(n_agents, -1).mean(axis=1)
